"""Regression-tree initialization for the hnswlib tuning pipeline.

Given a complete parameter exploration space (the YAML ``params:`` block) and
historical trials (``<output.dir>/trials/<name>.jsonl``), this module prunes the
space in three steps:

1. **Tree growth** — greedy CART over the trials' parameter points.  Each split
   chooses ONE parameter and a cut point (midpoint of two adjacent observed
   values, floored for integer parameters; left side assignment is
   ``value <= cut``, matching ``utils/subgroup_insights.lsqm_cut_points``) that
   minimizes the within-side QPS scatter of the **recall-qualifying** points
   (``recall >= threshold``).  Non-qualifying points are still routed through
   the tree (they occupy the space and matter for round-1 pruning) but do not
   enter the split criterion.  A node stops splitting once its data count is
   below ``min_leaf_samples`` (or no strictly-improving split exists).

2. **Pruning round 1** — a leaf whose data's 95% confidence UPPER bound of
   recall does not reach the recall threshold is pruned.

3. **Pruning round 2** — a leaf whose qualifying points' 95% confidence upper
   bound of QPS does not reach the median QPS of ALL qualifying data is pruned
   (a leaf with zero qualifying points is pruned as well).

The surviving leaves are the output: per-region integer bounds in the
prompt-writable ``to_spec()`` form (``{"kind": "range", "min": .., "max": ..,
"integer": true}``), a Markdown prompt block, a bounding box usable as the
pipeline's ``frozen_search_space``, and seed candidates drawn inside the
regions.

The module is self-contained (no ``agents/`` imports) and mirrors the loader in
``utils/hnswlib_history_model._load_success_qps_samples`` plus the cut-point
convention in ``utils/subgroup_insights.lsqm_cut_points``.

Note on ``ef`` mode: in ``ef_mode: direct`` runs ``ef`` is a real build input;
in scan-mode runs ``metrics.recall``/``metrics.qps`` already reflect the
*selected* ef, so a split on ``ef`` partitions on the requested ef.  The loader
uses ``metrics.recall``/``metrics.qps`` either way.

Port note (LVTuner-VLDB): copied from RFANNSTuner/utils/regression_tree_init.py
(2026-09).  The only functional change is in ``load_trials_points``: rows
WITHOUT a ``status`` key are accepted when they carry params + recall/qps,
because this repo persists minimal trials rows (see
``functions/hnswlib_tune._project_trial_for_file``).  The reference to
``utils/hnswlib_history_model._load_success_qps_samples`` above documents
provenance only — that module exists only in RFANNSTuner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy import stats

PARAM_ORDER: List[str] = ["M", "ef_construction", "ef"]

DEFAULT_MIN_LEAF_SAMPLES = 5
DEFAULT_CI_CONFIDENCE = 0.95
DEFAULT_SEED_COUNT = 8


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Point:
    """One usable trial: parameters, recall and QPS."""

    params: Dict[str, float]
    recall: float
    qps: float
    key: Tuple[Any, ...] = ()
    line_no: int = 0


@dataclass
class Node:
    """Regression-tree node.  ``param is None`` marks a leaf."""

    node_id: int
    idx: List[int]  # indices into the points list
    depth: int
    param: Optional[str] = None
    cut: Optional[float] = None  # left assignment: params[param] <= cut
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    sse: float = 0.0  # SSE of qualifying QPS in this node (0.0 if none)
    n_qualifying: int = 0
    box: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # observed min/max

    def is_leaf(self) -> bool:
        return self.param is None


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_params_cfg(params_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a YAML ``params:`` block into ``to_spec()``-shaped dicts."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in (params_cfg or {}).items():
        if not isinstance(spec, dict):
            continue
        if "values" in spec:
            out[name] = {"kind": "discrete", "values": list(spec["values"])}
        elif "min" in spec and "max" in spec:
            out[name] = {
                "kind": "range",
                "min": spec["min"],
                "max": spec["max"],
                "integer": bool(spec.get("integer", False)),
            }
    return out


def _within_space(params: Dict[str, float], space: Dict[str, Dict[str, Any]]) -> bool:
    for name, spec in space.items():
        if name not in params:
            continue
        value = params[name]
        if spec.get("kind") == "discrete":
            if value not in spec.get("values", []):
                return False
        else:
            lo = float(spec["min"])
            hi = float(spec["max"])
            if not (lo <= value <= hi):
                return False
    return True


def load_trials_points(
    trials_path: str | Path,
    *,
    param_order: Sequence[str] = PARAM_ORDER,
    space: Optional[Dict[str, Dict[str, Any]]] = None,
    dedupe_by_param_key: str = "keep_all",
    allow_empty: bool = False,
) -> Tuple[List[Point], Dict[str, int]]:
    """Load usable points from a trials JSONL file.

    Mirrors ``utils/hnswlib_history_model._load_success_qps_samples`` but keeps
    recall as well.  Rows are kept when ``status == "success"`` and both
    ``metrics.recall`` and ``metrics.qps`` are present; rows whose parameters
    fall outside *space* (when given) are dropped and counted.

    Returns ``(points, counters)`` where counters hold per-filter drop counts
    (``n_lines``, ``n_success``, ``n_usable``, ``n_dropped_status``,
    ``n_dropped_fields``, ``n_dropped_out_of_space``, ``n_duplicate_param_keys``).
    """
    counters = {
        "n_lines": 0,
        "n_success": 0,
        "n_usable": 0,
        "n_dropped_status": 0,
        "n_dropped_fields": 0,
        "n_dropped_out_of_space": 0,
        "n_duplicate_param_keys": 0,
    }
    if dedupe_by_param_key not in ("keep_all", "first", "mean"):
        raise ValueError(
            f"dedupe_by_param_key must be one of keep_all|first|mean, got {dedupe_by_param_key!r}"
        )
    path = Path(trials_path).expanduser().resolve()
    if not path.exists():
        if allow_empty:
            return [], counters
        raise FileNotFoundError(f"Trials file not found: {path}")

    points: List[Point] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            counters["n_lines"] += 1
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            # LVTuner-VLDB persists MINIMAL rows ({"params", "metrics"}) via
            # functions/hnswlib_tune._project_trial_for_file — no "status"
            # key.  A row is therefore usable when it carries params + metrics
            # with recall and qps; an explicit non-"success" status still
            # drops it.  (RFANNSTuner rows always carry status; unchanged.)
            if not isinstance(row, dict):
                counters["n_dropped_status"] += 1
                continue
            if "status" in row and row.get("status") != "success":
                counters["n_dropped_status"] += 1
                continue
            counters["n_success"] += 1
            params = row.get("params")
            metrics = row.get("metrics")
            if not isinstance(params, dict) or not isinstance(metrics, dict):
                counters["n_dropped_fields"] += 1
                continue
            if "recall" not in metrics or "qps" not in metrics:
                counters["n_dropped_fields"] += 1
                continue
            try:
                x = {name: float(params[name]) for name in param_order}
                recall = float(metrics["recall"])
                qps = float(metrics["qps"])
            except (KeyError, TypeError, ValueError):
                counters["n_dropped_fields"] += 1
                continue
            if space is not None and not _within_space(x, space):
                counters["n_dropped_out_of_space"] += 1
                continue
            points.append(
                Point(
                    params=x,
                    recall=recall,
                    qps=qps,
                    key=tuple(x[name] for name in param_order),
                    line_no=line_no,
                )
            )

    if dedupe_by_param_key in ("first", "mean"):
        merged: Dict[Tuple[Any, ...], List[Point]] = {}
        for p in points:
            merged.setdefault(p.key, []).append(p)
        deduped: List[Point] = []
        for key, group in merged.items():
            if len(group) > 1:
                counters["n_duplicate_param_keys"] += len(group) - 1
            if dedupe_by_param_key == "first":
                deduped.append(group[0])
            else:  # mean
                deduped.append(
                    Point(
                        params=dict(group[0].params),
                        recall=mean(pp.recall for pp in group),
                        qps=mean(pp.qps for pp in group),
                        key=key,
                        line_no=group[0].line_no,
                    )
                )
        points = deduped

    counters["n_usable"] = len(points)
    if not points and not allow_empty:
        raise ValueError(f"No usable successful trials found in trials file: {path}")
    return points, counters


# ─────────────────────────────────────────────────────────────────────────────
# Confidence intervals
# ─────────────────────────────────────────────────────────────────────────────


def ci_upper(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CI_CONFIDENCE,
    n1_policy: str = "value",
    statistic: str = "mean",
    bootstrap_resamples: int = 2000,
    rng_seed: int = 42,
) -> Optional[float]:
    """95% (or *confidence*) confidence UPPER bound.

    ``statistic == "mean"``: Student-t upper bound of the mean
    (``scipy.stats.t.interval``); a zero-variance sample collapses the interval
    to the mean itself, which is the correct "nothing to learn" behaviour.

    ``statistic == "bootstrap_median"``: percentile bootstrap of the median
    (robust to right-skewed QPS; uses numpy).

    ``n1_policy`` handles single-value samples: ``"value"`` returns the value
    itself, ``"conservative"`` returns ``+inf`` (never prune on one sample).
    """
    if n1_policy not in ("value", "conservative"):
        raise ValueError(f"n1_policy must be value|conservative, got {n1_policy!r}")
    if statistic not in ("mean", "bootstrap_median"):
        raise ValueError(f"statistic must be mean|bootstrap_median, got {statistic!r}")
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        if n1_policy == "value":
            return vals[0]
        return float("inf")
    if statistic == "mean":
        mu = mean(vals)
        sem = stats.sem(vals)
        if not sem or sem <= 0.0:
            return mu  # zero variance: interval collapses to the mean (scipy 1.18 gives nan)
        _, hi = stats.t.interval(confidence, n - 1, loc=mu, scale=sem)
        return float(hi)
    # bootstrap_median
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(vals)
    boots = np.median(rng.choice(arr, size=(bootstrap_resamples, n), replace=True), axis=1)
    return float(np.percentile(boots, 100 * confidence))


def ci_bounds(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CI_CONFIDENCE,
    n1_policy: str = "value",
) -> Optional[Tuple[float, float]]:
    """Two-sided confidence interval of the mean; see :func:`ci_upper`."""
    if n1_policy not in ("value", "conservative"):
        raise ValueError(f"n1_policy must be value|conservative, got {n1_policy!r}")
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        v = vals[0]
        if n1_policy == "value":
            return (v, v)
        return (float("-inf"), float("inf"))
    mu = mean(vals)
    sem = stats.sem(vals)
    if not sem or sem <= 0.0:
        return (mu, mu)  # zero variance: interval collapses to the mean (scipy 1.18 gives nan)
    lo, hi = stats.t.interval(confidence, n - 1, loc=mu, scale=sem)
    return (float(lo), float(hi))


# ─────────────────────────────────────────────────────────────────────────────
# Tree growth
# ─────────────────────────────────────────────────────────────────────────────


def _prefix_sums(values: Sequence[float]) -> List[float]:
    result = [0.0]
    running = 0.0
    for v in values:
        running += v
        result.append(running)
    return result


def _interval_sse_from_prefix(
    pc: Sequence[float], ps: Sequence[float], ps2: Sequence[float], left: int, right: int
) -> float:
    count = pc[right + 1] - pc[left]
    if count <= 0.0:
        return 0.0
    sum_y = ps[right + 1] - ps[left]
    sum_y2 = ps2[right + 1] - ps2[left]
    return max(0.0, sum_y2 - (sum_y * sum_y) / count)


def _all_integral(values: Sequence[float]) -> bool:
    return all(float(v).is_integer() for v in values)


def _node_box(
    node: Node, points: Sequence[Point], param_order: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    box: Dict[str, Tuple[float, float]] = {}
    for name in param_order:
        vals = [points[i].params[name] for i in node.idx]
        box[name] = (min(vals), max(vals))
    return box


def _try_split_node(
    node: Node,
    points: Sequence[Point],
    param_order: Sequence[str],
    min_leaf_samples: int,
    recall_threshold: float,
) -> Optional[Tuple[str, float, float, List[int], List[int]]]:
    """Best (param, cut, cost, left_idx, right_idx) split, or None.

    The cost is the sum of the within-side SSE of the QPS of
    recall-qualifying points; a side without qualifying points contributes 0.
    """
    idx = node.idx
    n = len(idx)
    best: Optional[Tuple[str, float, float, List[int], List[int]]] = None
    for p_pos, name in enumerate(param_order):
        order = sorted(idx, key=lambda i: points[i].params[name])
        vals = [points[i].params[name] for i in order]
        # prefix stats over qualifying points only
        pc: List[float] = [0.0]
        ps: List[float] = [0.0]
        ps2: List[float] = [0.0]
        c = s = s2 = 0.0
        for i in order:
            if points[i].recall >= recall_threshold:
                c += 1.0
                s += points[i].qps
                s2 += points[i].qps * points[i].qps
            pc.append(c)
            ps.append(s)
            ps2.append(s2)
        is_integer = _all_integral(vals)
        for k in range(1, n):
            if vals[k] == vals[k - 1]:
                continue  # cut needs two distinct adjacent values
            if k < min_leaf_samples or (n - k) < min_leaf_samples:
                continue
            mid = (vals[k - 1] + vals[k]) / 2.0
            cut = float(math.floor(mid)) if is_integer else mid
            cost = _interval_sse_from_prefix(pc, ps, ps2, 0, k - 1) + _interval_sse_from_prefix(
                pc, ps, ps2, k, n - 1
            )
            if best is None or (cost, p_pos, cut) < (best[2], param_order.index(best[0]), best[1]):
                best = (name, cut, cost, order[:k], order[k:])
    return best


def build_regression_tree(
    points: Sequence[Point],
    *,
    param_order: Sequence[str] = PARAM_ORDER,
    min_leaf_samples: int = DEFAULT_MIN_LEAF_SAMPLES,
    max_depth: Optional[int] = None,
    min_sse_decrease: float = 1e-12,
    recall_threshold: float = 0.95,
) -> Node:
    """Grow a deterministic CART tree with the qualifying-QPS SSE criterion.

    Stop conditions (any one): node size ``n < 2 * min_leaf_samples``; no
    qualifying points (nothing to split on); no strictly-improving candidate
    cut; ``depth >= max_depth`` (when set).
    """
    min_leaf_samples = max(1, int(min_leaf_samples))
    if not points:
        raise ValueError("cannot grow a regression tree on zero points")
    n = len(points)
    root = Node(node_id=0, idx=list(range(n)), depth=0)
    stack: List[Node] = [root]
    next_id = 1
    while stack:
        node = stack.pop()
        node.box = _node_box(node, points, param_order)
        qual = [i for i in node.idx if points[i].recall >= recall_threshold]
        node.n_qualifying = len(qual)
        if qual:
            qs = [points[i].qps for i in qual]
            node.sse = max(0.0, sum(v * v for v in qs) - sum(qs) ** 2 / len(qs))
        else:
            node.sse = 0.0
        if max_depth is not None and node.depth >= int(max_depth):
            continue
        if len(node.idx) < 2 * min_leaf_samples or not qual:
            continue
        best = _try_split_node(node, points, param_order, min_leaf_samples, recall_threshold)
        if best is None:
            continue
        name, cut, cost, left_idx, right_idx = best
        if cost >= node.sse - min_sse_decrease * (1.0 + abs(node.sse)):
            continue  # no strict improvement
        node.param = name
        node.cut = cut
        node.left = Node(node_id=next_id, idx=left_idx, depth=node.depth + 1)
        next_id += 1
        node.right = Node(node_id=next_id, idx=right_idx, depth=node.depth + 1)
        next_id += 1
        stack.append(node.left)
        stack.append(node.right)
    return root


def collect_leaves(root: Node) -> List[Node]:
    """All leaves in DFS order."""
    leaves: List[Node] = []
    stack: List[Node] = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            leaves.append(node)
        else:
            stack.append(node.right)
            stack.append(node.left)
    return leaves


def render_tree_ascii(
    root: Node, points: Sequence[Point], *, param_order: Sequence[str] = PARAM_ORDER
) -> str:
    """Compact ASCII rendering of the grown tree (diagnostics)."""
    lines: List[str] = []

    def visit(node: Node, prefix: str, is_last: bool) -> None:
        if node.is_leaf():
            label = (
                f"leaf #{node.node_id} (n={len(node.idx)}, qual={node.n_qualifying}, "
                f"SSE={node.sse:.3g})"
            )
        else:
            cut = node.cut
            cut_label = f"{int(cut)}" if cut is not None and float(cut).is_integer() else f"{cut}"
            label = (
                f"node #{node.node_id} (n={len(node.idx)}, qual={node.n_qualifying}, "
                f"SSE={node.sse:.3g}) split {node.param} <= {cut_label}"
            )
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + label)
        if not node.is_leaf():
            child_prefix = prefix + ("    " if is_last else "│   ")
            visit(node.left, child_prefix, False)
            visit(node.right, child_prefix, True)

    lines.append(
        f"root #{root.node_id} (n={len(root.idx)}, qual={root.n_qualifying}, "
        f"SSE={root.sse:.3g})"
        + (
            f" split {root.param} <= "
            f"{int(root.cut) if root.cut is not None and float(root.cut).is_integer() else root.cut}"
            if not root.is_leaf()
            else ""
        )
    )
    if not root.is_leaf():
        visit(root.left, "", False)
        visit(root.right, "", True)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Regions
# ─────────────────────────────────────────────────────────────────────────────


def _region_bounds(
    idx: Sequence[int], points: Sequence[Point], param_order: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    bounds: Dict[str, Dict[str, Any]] = {}
    for name in param_order:
        vals = sorted(points[i].params[name] for i in idx)
        lo, hi = vals[0], vals[-1]
        if _all_integral(vals):
            bounds[name] = {
                "kind": "range",
                "min": int(lo),
                "max": int(hi),
                "integer": True,
            }
        else:
            bounds[name] = {"kind": "range", "min": lo, "max": hi, "integer": False}
    return bounds


def _count_feasible_lattice(
    bounds: Dict[str, Dict[str, Any]], param_order: Sequence[str]
) -> Tuple[Optional[int], Optional[int]]:
    """(n_feasible, n_total) lattice points with ef <= ef_construction.

    Returns ``(None, None)`` when any parameter domain is non-integer.
    """
    ranges: Dict[str, Tuple[int, int]] = {}
    for name in param_order:
        spec = bounds[name]
        if spec.get("kind") != "range" or not spec.get("integer", False):
            return None, None
        ranges[name] = (int(spec["min"]), int(spec["max"]))
    total = 1
    for lo, hi in ranges.values():
        total *= hi - lo + 1
    if "ef" not in ranges or "ef_construction" not in ranges:
        return total, total
    ef_lo, ef_hi = ranges["ef"]
    efc_lo, efc_hi = ranges["ef_construction"]
    per_ef = 0
    for ef in range(ef_lo, ef_hi + 1):
        efc_start = max(efc_lo, ef)
        if efc_start <= efc_hi:
            per_ef += efc_hi - efc_start + 1
    total_without = total // ((ef_hi - ef_lo + 1) * (efc_hi - efc_lo + 1))
    return per_ef * total_without, total


def _anchor_for(
    idx: Sequence[int], points: Sequence[Point], recall_threshold: float
) -> Tuple[Dict[str, Any], float, float]:
    qualifying = [i for i in idx if points[i].recall >= recall_threshold]
    pool = qualifying or list(idx)
    best_i = max(pool, key=lambda i: (points[i].qps, points[i].recall))
    p = points[best_i]
    anchor = {name: float(p.params[name]) for name in p.params}
    for name, value in anchor.items():
        if float(value).is_integer():
            anchor[name] = int(value)
    return anchor, p.recall, p.qps


def leaf_regions(
    root: Node,
    points: Sequence[Point],
    *,
    param_order: Sequence[str] = PARAM_ORDER,
    recall_threshold: float = 0.95,
    ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    n1_policy: str = "value",
) -> List[Dict[str, Any]]:
    """Per-leaf region summaries (bounds + recall/QPS stats + anchor)."""
    regions: List[Dict[str, Any]] = []
    for leaf_id, leaf in enumerate(collect_leaves(root)):
        idx = leaf.idx
        recalls = [points[i].recall for i in idx]
        qpss = [points[i].qps for i in idx]
        qual = [i for i in idx if points[i].recall >= recall_threshold]
        bounds = _region_bounds(idx, points, param_order)
        n_feasible, n_total = _count_feasible_lattice(bounds, param_order)
        anchor_params, anchor_recall, anchor_qps = _anchor_for(idx, points, recall_threshold)
        recall_ci = ci_bounds(recalls, confidence=ci_confidence, n1_policy=n1_policy)
        qual_ci = ci_bounds(
            [points[i].qps for i in qual], confidence=ci_confidence, n1_policy=n1_policy
        )
        regions.append(
            {
                "leaf_id": leaf_id,
                "node_id": leaf.node_id,
                "depth": leaf.depth,
                "idx": idx,
                "bounds": bounds,
                "n_trials": len(idx),
                "n_qualifying": len(qual),
                "recall": {
                    "mean": mean(recalls),
                    "median": median(recalls),
                    "ci_lower": recall_ci[0] if recall_ci else None,
                    "ci_upper": recall_ci[1] if recall_ci else None,
                    "n": len(recalls),
                },
                "qps": {
                    "mean": mean(qpss),
                    "median": median(qpss),
                    "n": len(qpss),
                },
                "qps_qualifying": {
                    "mean": mean([points[i].qps for i in qual]) if qual else None,
                    "median": median([points[i].qps for i in qual]) if qual else None,
                    "ci_lower": qual_ci[0] if qual_ci else None,
                    "ci_upper": qual_ci[1] if qual_ci else None,
                    "n": len(qual),
                },
                "anchor_params": anchor_params,
                "anchor_recall": anchor_recall,
                "anchor_qps": anchor_qps,
                "n_infeasible_in_box": (n_total - n_feasible) if n_feasible is not None else None,
            }
        )
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────────────────────────────────────


def prune_round1(
    regions: List[Dict[str, Any]],
    *,
    recall_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Round 1: prune leaves whose recall 95% CI upper bound < threshold.

    Regions must already carry ``recall.ci_upper`` (from :func:`leaf_regions`
    or an equivalent recomputation at the desired confidence).
    """
    surviving: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []
    for region in regions:
        ub = region["recall"]["ci_upper"]
        if ub is None or ub < recall_threshold:
            pruned.append(
                {
                    "leaf_id": region["leaf_id"],
                    "reason": "recall_ci_upper_below_threshold",
                    "recall_ci_upper": ub,
                    "recall_threshold": recall_threshold,
                    "n_samples": region["n_trials"],
                }
            )
        else:
            surviving.append(region)
    return surviving, pruned


def global_median_qualifying_qps(
    points: Sequence[Point], *, recall_threshold: float
) -> Optional[float]:
    """Median QPS over ALL qualifying points (recall >= threshold)."""
    qs = [p.qps for p in points if p.recall >= recall_threshold]
    return float(median(qs)) if qs else None


def prune_round2(
    regions: List[Dict[str, Any]],
    *,
    recall_threshold: float,
    global_median_qps: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Round 2: prune leaves whose qualifying-QPS 95% CI upper bound does not
    reach the global median QPS of qualifying data (leaves with zero
    qualifying points are pruned as well).

    Regions must already carry ``qps_qualifying.ci_upper``.
    """
    if global_median_qps is None:
        return list(regions), []  # no qualifying data anywhere -> no-op
    surviving: List[Dict[str, Any]] = []
    pruned: List[Dict[str, Any]] = []
    for region in regions:
        if region["n_qualifying"] == 0:
            pruned.append(
                {
                    "leaf_id": region["leaf_id"],
                    "reason": "no_qualifying_points",
                    "qps_ci_upper": None,
                    "global_median_qps": global_median_qps,
                    "n_qualifying": 0,
                }
            )
            continue
        ub = region["qps_qualifying"]["ci_upper"]
        if ub is None or ub < global_median_qps:
            pruned.append(
                {
                    "leaf_id": region["leaf_id"],
                    "reason": "qps_ci_upper_below_global_median",
                    "qps_ci_upper": ub,
                    "global_median_qps": global_median_qps,
                    "n_qualifying": region["n_qualifying"],
                }
            )
        else:
            surviving.append(region)
    return surviving, pruned


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def region_bounding_box(
    regions: List[Dict[str, Any]],
    *,
    param_order: Sequence[str] = PARAM_ORDER,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Minimal per-parameter box covering all surviving regions (or None)."""
    if not regions:
        return None
    box: Dict[str, Dict[str, Any]] = {}
    for name in param_order:
        lo = min(r["bounds"][name]["min"] for r in regions)
        hi = max(r["bounds"][name]["max"] for r in regions)
        integer = all(r["bounds"][name].get("integer", False) for r in regions)
        box[name] = {"kind": "range", "min": lo, "max": hi, "integer": bool(integer)}
    return box


def _seed_key(params: Dict[str, Any], param_order: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(params.get(name) for name in param_order)


def draw_region_seeds(
    regions: List[Dict[str, Any]],
    *,
    seed_count: int = DEFAULT_SEED_COUNT,
    seeds_per_region: int = 1,
    seed: int = 42,
    enforce_ef_le_efc: bool = True,
    param_order: Sequence[str] = PARAM_ORDER,
) -> List[Dict[str, Any]]:
    """Seed candidates: each region's best observed anchor first, then uniform
    samples inside the region boxes (round-robin) until *seed_count*."""
    seed_count = max(0, int(seed_count))
    seeds_per_region = max(0, int(seeds_per_region))
    if seed_count == 0 or not regions:
        return []
    rng = random.Random(seed)
    seeds: List[Dict[str, Any]] = []
    seen: set = set()

    def add(params: Dict[str, Any], note: str) -> bool:
        key = _seed_key(params, param_order)
        if key in seen:
            return False
        seen.add(key)
        seeds.append({"params": params, "source": "regression_tree_region", "note": note})
        return True

    # Anchors (up to seeds_per_region per region — the observed best point).
    for region in regions:
        region_id = region.get("region_id", region.get("leaf_id", "?"))
        for _ in range(seeds_per_region):
            if not add(
                dict(region["anchor_params"]),
                f"region {region_id} anchor (best observed)",
            ):
                break
        if len(seeds) >= seed_count:
            return seeds[:seed_count]

    # Space-filling samples, round-robin over regions.
    max_attempts = 64 * len(regions)
    attempts = 0
    while len(seeds) < seed_count and attempts < max_attempts:
        made_progress = False
        for region in regions:
            if len(seeds) >= seed_count:
                break
            region_id = region.get("region_id", region.get("leaf_id", "?"))
            bounds = region["bounds"]
            for _ in range(64):
                attempts += 1
                cand: Dict[str, Any] = {}
                for name in param_order:
                    spec = bounds[name]
                    if spec.get("integer", False):
                        cand[name] = rng.randint(int(spec["min"]), int(spec["max"]))
                    else:
                        cand[name] = rng.uniform(float(spec["min"]), float(spec["max"]))
                if (
                    enforce_ef_le_efc
                    and "ef" in cand
                    and "ef_construction" in cand
                    and cand["ef"] > cand["ef_construction"]
                ):
                    continue
                if add(cand, f"region {region_id} space-filling sample"):
                    made_progress = True
                    break
        if not made_progress:
            break
    return seeds[:seed_count]


def render_regions_spec(
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Surviving regions as ``to_spec()``-shaped prompt-writable entries."""
    return [
        {
            "region_id": region.get("region_id", region.get("leaf_id")),
            "leaf_id": region["leaf_id"],
            "bounds": {name: dict(spec) for name, spec in region["bounds"].items()},
        }
        for region in regions
    ]


def _fmt_range(spec: Dict[str, Any]) -> str:
    lo, hi = spec["min"], spec["max"]
    if lo == hi:
        return f"{lo}"
    return f"{lo}–{hi}"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 100.0:
        return f"{v:,.1f}"
    return f"{v:.4g}"


def render_regions_prompt(
    regions: List[Dict[str, Any]],
    *,
    recall_threshold: float,
    param_order: Sequence[str] = PARAM_ORDER,
    global_median_qps: Optional[float] = None,
    max_regions: int = 12,
    freeze_mode: str = "bbox",
    bounding_box: Optional[Dict[str, Dict[str, Any]]] = None,
    trials_path: str = "",
    n_trials_total: int = 0,
    n_leaves_total: int = 0,
    n_pruned_round1: int = 0,
    n_pruned_round2: int = 0,
    min_leaf_samples: int = DEFAULT_MIN_LEAF_SAMPLES,
) -> str:
    """Markdown block describing the surviving regions (for the LLM prompt)."""
    shown = regions[: max(0, int(max_regions))]
    lines: List[str] = []
    lines.append("## Regression-Tree Pruned Regions (history-informed focus)")
    lines.append("")
    lines.append(
        f"A regression tree (CART-style, split target = QPS of recall-qualifying "
        f"trials, min leaf = {min_leaf_samples}) partitioned the full build "
        f"space using {n_trials_total} successful trials"
        + (f" from {trials_path}." if trials_path else ".")
    )
    lines.append(
        f"Pruning round 1 removed {n_pruned_round1}/{n_leaves_total} leaves whose "
        f"95% CI upper bound of recall < τ={recall_threshold:g}."
    )
    lines.append(
        f"Pruning round 2 removed {n_pruned_round2} leaves whose 95% CI upper bound "
        f"of QPS (over recall≥τ points) < the global median QPS of qualifying data"
        + (f" ({global_median_qps:.4g})." if global_median_qps is not None else ".")
    )
    lines.append("")
    if not shown:
        lines.append(
            "**All regions were pruned** — the historical data contains no region "
            "whose recall CI upper bound reaches τ. Treat the full parameter space "
            "with extra caution; prefer high-M / high-ef_construction corners."
        )
        lines.append("")
        return "\n".join(lines)
    header = (
        f"| Region | {' | '.join(param_order)} | n | recall (mean / CI-ub) | "
        f"qual. QPS (CI-ub) | anchor ({', '.join(param_order)}) |"
    )
    sep = "|---" * (4 + len(param_order)) + "|"
    lines.append(header)
    lines.append(sep)
    for region in shown:
        bounds_cells = " | ".join(_fmt_range(region["bounds"][name]) for name in param_order)
        recall_cell = f"{_fmt_num(region['recall']['mean'])} / {_fmt_num(region['recall']['ci_upper'])}"
        qps_ub = region["qps_qualifying"]["ci_upper"]
        qps_cell = _fmt_num(qps_ub)
        anchor = ", ".join(str(region["anchor_params"].get(name, "?")) for name in param_order)
        lines.append(
            f"| R{region.get('region_id', region.get('leaf_id'))} | {bounds_cells} "
            f"| {region['n_trials']} | {recall_cell} | {qps_cell} | {anchor} |"
        )
    lines.append("")
    if freeze_mode == "bbox" and bounding_box is not None:
        box_desc = ", ".join(
            f"{name}∈[{_fmt_range(bounding_box[name])}]" for name in param_order
        )
        lines.append(f"Search space is frozen to the bounding box {box_desc}.")
        lines.append(
            "NOTE: the box is the union of the surviving regions — a gap between "
            "two boxes was pruned and carries no evidence, so prefer points inside "
            "a surviving region. Proposals outside the box are CLAMPED into it, "
            "not rejected."
        )
    else:
        lines.append(
            "Search space is UNFROZEN — treat the regions above as a prior, "
            "not a hard constraint."
        )
    lines.append("")
    lines.append(
        "**Guidance:** propose inside a surviving region; start from each "
        "region's anchor and vary one parameter at a time within the box. "
        "Constraint: ef ≤ ef_construction."
    )
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_regression_tree_init(
    params_cfg: Dict[str, Any],
    trials_path: str | Path,
    *,
    recall_threshold: float,
    min_leaf_samples: int = DEFAULT_MIN_LEAF_SAMPLES,
    recall_ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    qps_ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    max_depth: Optional[int] = None,
    n1_ci_policy: str = "value",
    qps_ci_statistic: str = "mean",
    bootstrap_resamples: int = 2000,
    dedupe_by_param_key: str = "keep_all",
    seed_count: int = DEFAULT_SEED_COUNT,
    seeds_per_region: int = 1,
    seed: int = 42,
    freeze_mode: str = "bbox",
    prompt_max_regions: int = 12,
    global_median_scope: str = "all",
    param_order: Sequence[str] = PARAM_ORDER,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Full regression-tree initialization; see module docstring.

    ``global_median_scope`` selects the round-2 comparison pool: ``"all"`` is
    the median QPS over every loaded qualifying point (the literal spec
    wording "所有达标数据"), ``"surviving"`` restricts it to the qualifying
    points of round-1 survivors.
    """
    if freeze_mode not in ("bbox", "off"):
        raise ValueError(f"freeze_mode must be bbox|off, got {freeze_mode!r}")
    if global_median_scope not in ("all", "surviving"):
        raise ValueError(f"global_median_scope must be all|surviving, got {global_median_scope!r}")
    empty_result: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "ok": False,
        "reason": "",
        "task_name": Path(trials_path).stem if isinstance(trials_path, (str, Path)) else "",
        "trials_path": str(Path(trials_path).expanduser().resolve()),
        "config": {
            "recall_threshold": recall_threshold,
            "min_leaf_samples": int(min_leaf_samples),
            "recall_ci_confidence": recall_ci_confidence,
            "qps_ci_confidence": qps_ci_confidence,
            "max_depth": max_depth,
            "n1_ci_policy": n1_ci_policy,
            "qps_ci_statistic": qps_ci_statistic,
            "dedupe_by_param_key": dedupe_by_param_key,
            "seed_count": int(seed_count),
            "seeds_per_region": int(seeds_per_region),
            "freeze_mode": freeze_mode,
            "global_median_scope": global_median_scope,
        },
        "dataset": {},
        "tree": {},
        "pruned_round1": [],
        "pruned_round2": [],
        "surviving_regions": [],
        "bounding_box": None,
        "seed_candidates": [],
        "prompt_text": "",
    }
    try:
        space = _normalize_params_cfg(params_cfg) or None
        points, counters = load_trials_points(
            trials_path,
            param_order=param_order,
            space=space,
            dedupe_by_param_key=dedupe_by_param_key,
            allow_empty=allow_empty,
        )
    except (FileNotFoundError, ValueError) as exc:
        result = dict(empty_result)
        result["reason"] = str(exc)
        return result

    n_qualifying = sum(1 for p in points if p.recall >= recall_threshold)

    if not points:
        result = dict(empty_result)
        result["dataset"] = dict(counters)
        result["reason"] = "no_usable_trials"
        return result

    tree = build_regression_tree(
        points,
        param_order=param_order,
        min_leaf_samples=min_leaf_samples,
        max_depth=max_depth,
        recall_threshold=recall_threshold,
    )
    regions = leaf_regions(
        tree,
        points,
        param_order=param_order,
        recall_threshold=recall_threshold,
        ci_confidence=recall_ci_confidence,
        n1_policy=n1_ci_policy,
    )
    # Round 2 uses the qps CI confidence/statistic (regions carry round-1
    # recall stats already); recompute the qualifying-QPS CI upper bounds with
    # the round-2 settings.
    for region in regions:
        qual_qs = [
            points[i].qps for i in region["idx"] if points[i].recall >= recall_threshold
        ]
        region["qps_qualifying"]["ci_upper"] = ci_upper(
            qual_qs,
            confidence=qps_ci_confidence,
            n1_policy=n1_ci_policy,
            statistic=qps_ci_statistic,
            bootstrap_resamples=bootstrap_resamples,
            rng_seed=seed,
        )

    surv1, pruned1 = prune_round1(regions, recall_threshold=recall_threshold)
    if global_median_scope == "surviving":
        scope_qps = [
            points[i].qps
            for region in surv1
            for i in region["idx"]
            if points[i].recall >= recall_threshold
        ]
        global_median_qps = float(median(scope_qps)) if scope_qps else None
    else:
        global_median_qps = global_median_qualifying_qps(points, recall_threshold=recall_threshold)
    if global_median_qps is None:
        surv2, pruned2 = surv1, []
    else:
        surv2, pruned2 = prune_round2(
            surv1,
            recall_threshold=recall_threshold,
            global_median_qps=global_median_qps,
        )
    # Renumber surviving regions for display (leaf_id keeps the tree identity).
    for display_id, region in enumerate(surv2, start=1):
        region["region_id"] = display_id

    bounding_box = region_bounding_box(surv2, param_order=param_order)
    seed_candidates = draw_region_seeds(
        surv2,
        seed_count=seed_count,
        seeds_per_region=seeds_per_region,
        seed=seed,
        enforce_ef_le_efc=True,
        param_order=param_order,
    )
    prompt_text = render_regions_prompt(
        surv2,
        recall_threshold=recall_threshold,
        param_order=param_order,
        global_median_qps=global_median_qps,
        max_regions=prompt_max_regions,
        freeze_mode=freeze_mode,
        bounding_box=bounding_box,
        trials_path=str(Path(trials_path).expanduser().resolve()),
        n_trials_total=len(points),
        n_leaves_total=len(regions),
        n_pruned_round1=len(pruned1),
        n_pruned_round2=len(pruned2),
        min_leaf_samples=min_leaf_samples,
    )

    tree_leaves: List[Dict[str, Any]] = []
    for region in regions:
        tree_leaves.append(
            {
                "leaf_id": region["leaf_id"],
                "node_id": region["node_id"],
                "depth": region["depth"],
                "n_samples": region["n_trials"],
                "n_qualifying": region["n_qualifying"],
                "bounds": {name: dict(spec) for name, spec in region["bounds"].items()},
                "recall": dict(region["recall"]),
                "qps_qualifying": dict(region["qps_qualifying"]),
                "best_observed_params": dict(region["anchor_params"]),
                "best_observed_qps": region["anchor_qps"],
            }
        )
    splits: List[Dict[str, Any]] = []
    max_depth_reached = 0
    stack: List[Node] = [tree]
    while stack:
        node = stack.pop()
        max_depth_reached = max(max_depth_reached, node.depth)
        if not node.is_leaf():
            splits.append(
                {
                    "node_id": node.node_id,
                    "param": node.param,
                    "cut": node.cut,
                    "n_left": len(node.left.idx) if node.left else 0,
                    "n_right": len(node.right.idx) if node.right else 0,
                    "n_qual_left": node.left.n_qualifying if node.left else 0,
                    "n_qual_right": node.right.n_qualifying if node.right else 0,
                    "sse_parent": node.sse,
                    "sse_children": (node.left.sse if node.left else 0.0)
                    + (node.right.sse if node.right else 0.0),
                }
            )
            stack.append(node.left)
            stack.append(node.right)

    result = {
        "generated_at": _utc_now_iso(),
        "ok": True,
        "reason": "ok" if surv2 else "all_pruned",
        "task_name": Path(trials_path).stem if isinstance(trials_path, (str, Path)) else "",
        "trials_path": str(Path(trials_path).expanduser().resolve()),
        "config": dict(empty_result["config"]),
        "dataset": {
            **counters,
            "n_qualifying": n_qualifying,
            "global_median_qps_qualifying": global_median_qps,
            "space": space if space is not None else {},
        },
        "tree": {
            "n_nodes": 1 + len(splits) * 2,
            "n_leaves": len(regions),
            "max_depth_reached": max_depth_reached,
            "splits": splits,
            "leaves": tree_leaves,
        },
        "tree_ascii": render_tree_ascii(tree, points, param_order=param_order),
        "pruned_round1": pruned1,
        "pruned_round2": pruned2,
        "surviving_regions": [
            {
                "region_id": region["region_id"],
                "leaf_id": region["leaf_id"],
                "depth": region["depth"],
                "bounds": {name: dict(spec) for name, spec in region["bounds"].items()},
                "n_trials": region["n_trials"],
                "n_qualifying": region["n_qualifying"],
                "recall_ci_upper": region["recall"]["ci_upper"],
                "recall_mean": region["recall"]["mean"],
                "qps_ci_upper": region["qps_qualifying"]["ci_upper"],
                "qps_qualifying_median": region["qps_qualifying"]["median"],
                "anchor_params": dict(region["anchor_params"]),
                "anchor_recall": region["anchor_recall"],
                "anchor_qps": region["anchor_qps"],
                "n_infeasible_in_box": region["n_infeasible_in_box"],
            }
            for region in surv2
        ],
        "bounding_box": bounding_box,
        "seed_candidates": seed_candidates,
        "prompt_text": prompt_text,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regression-tree initialization: prune the hnswlib parameter "
        "space from historical trials and emit prompt-writable regions."
    )
    parser.add_argument("--config", help="Pipeline YAML config (reads params + recall threshold)")
    parser.add_argument("--trials", required=True, help="Historical trials JSONL path")
    parser.add_argument("--min-leaf", type=int, default=DEFAULT_MIN_LEAF_SAMPLES)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--recall-threshold", type=float, default=None, help="Overrides --config")
    parser.add_argument("--recall-ci", type=float, default=DEFAULT_CI_CONFIDENCE)
    parser.add_argument("--qps-ci", type=float, default=DEFAULT_CI_CONFIDENCE)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--freeze-mode", default="bbox", choices=("bbox", "off"))
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--print-tree", action="store_true")
    args = parser.parse_args(argv)

    params_cfg: Dict[str, Any] = {}
    recall_threshold = args.recall_threshold
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        params_cfg = cfg.get("params") or {}
        if recall_threshold is None:
            search_cfg = cfg.get("search") or {}
            if "recall_threshold" not in search_cfg:
                parser.error("--config has no search.recall_threshold; pass --recall-threshold")
            recall_threshold = float(search_cfg["recall_threshold"])
    if recall_threshold is None:
        parser.error("--recall-threshold is required when --config is not given")

    report = run_regression_tree_init(
        params_cfg,
        args.trials,
        recall_threshold=recall_threshold,
        min_leaf_samples=args.min_leaf,
        recall_ci_confidence=args.recall_ci,
        qps_ci_confidence=args.qps_ci,
        max_depth=args.max_depth,
        seed_count=args.seed_count,
        freeze_mode=args.freeze_mode,
        allow_empty=True,
    )

    trials_stem = Path(args.trials).stem
    out_path = Path(args.out) if args.out else Path(
        f"results/hnswlib/regression_tree_init/{trials_stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path = out_path.parent / f"{out_path.stem}.prompt.txt"
    prompt_path.write_text(report["prompt_text"], encoding="utf-8")

    print(f"ok={report['ok']} reason={report['reason']} task={report['task_name']}")
    dataset = report.get("dataset") or {}
    print(
        f"dataset: usable={dataset.get('n_usable')} qualifying={dataset.get('n_qualifying')} "
        f"global_median_qps_qualifying={dataset.get('global_median_qps_qualifying')}"
    )
    tree = report.get("tree") or {}
    print(
        f"tree: n_leaves={tree.get('n_leaves')} max_depth={tree.get('max_depth_reached')} "
        f"pruned r1={len(report['pruned_round1'])} r2={len(report['pruned_round2'])} "
        f"surviving={len(report['surviving_regions'])}"
    )
    print(f"wrote {out_path}")
    print(f"wrote {prompt_path}")
    if args.print_tree and tree.get("n_leaves"):
        print(report["tree_ascii"])
    if args.print_prompt:
        print()
        print(report["prompt_text"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

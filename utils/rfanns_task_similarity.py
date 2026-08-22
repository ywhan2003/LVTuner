"""ANNS task similarity via sampling-invariant frontier alignment.

The legacy approach (``compute_task_similarity`` /
``compute_similarity_with_model_predictions``, kept below as deprecated) ranked
historical tasks by partial-order consistency of *absolute* QPS, evaluated by
running each historical QPS model on the *current* task's sampled points. That
is coupled to where the current run happened to sample, extrapolates each
historical model outside its own sampled region, and ignores recall entirely.

This module replaces it with a sampling-invariant formulation. For a fixed
``(dataset, algorithm, build b)`` the reachable ``(recall, QPS)`` frontier is a
deterministic function independent of which points were sampled -- sampling only
affects estimation precision. We therefore:

1. Define a virtual probe grid in normalized build space (never evaluated for
   real); it is the common coordinate frame both tasks are aligned onto.
2. Fit, per task, two GP proxies on that task's *own* real trials:
   a QPS proxy ``b -> G_tau(b)`` and a recall proxy ``b -> rec(b, ef_max)``.
   GP posteriors give calibrated mean + variance (mirrors the SCBO surrogates).
3. On the grid, form per-config-pair Recall-gain (RG) and QPS-cost (QC), z-score
   them within each task (kills cross-dataset magnitude differences), and weight
   each pair by inverse posterior variance so grid regions where either task
   lacks data are down-weighted.

A frontier-curve distance (Method C) provides a cold-start / cross-variant
fallback that needs no shared parameterization.
"""

import argparse
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from agents.rfanns_agent import PARAM_ORDER

EPS = 1e-9
DEFAULT_SEARCH_PARAM = "ef"


def _to_finite_float(value: Any, name: str, index: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}[{index}] is not a valid number: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name}[{index}] must be finite: {value!r}")
    return parsed


@dataclass
class SimilarityConfig:
    """Tunable knobs for frontier-alignment similarity. All have safe defaults."""

    metric: str = "response_gp"          # response_gp | w2 | frontier
    grid_per_dim: int = 4                  # probe points per build dimension
    max_pairs: int = 400                   # cap on config pairs (cost guard)
    min_trials_for_gp: int = 5             # below this -> frontier fallback
    alpha: float = 0.6                     # weight on ResponseSim
    beta: float = 0.4                      # weight on DatasetSim
    search_param: str = DEFAULT_SEARCH_PARAM
    build_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    @classmethod
    def from_mapping(cls, raw: Optional[Dict[str, Any]]) -> "SimilarityConfig":
        if not isinstance(raw, dict):
            return cls()
        cfg = cls()
        if "metric" in raw and raw["metric"]:
            cfg.metric = str(raw["metric"]).strip().lower()
        for int_key in ("grid_per_dim", "max_pairs", "min_trials_for_gp"):
            if raw.get(int_key) is not None:
                setattr(cfg, int_key, max(1, int(raw[int_key])))
        for float_key in ("alpha", "beta"):
            if raw.get(float_key) is not None:
                setattr(cfg, float_key, float(raw[float_key]))
        if raw.get("search_param"):
            cfg.search_param = str(raw["search_param"]).strip()
        return cfg


@dataclass
class BuildObservation:
    """Per-build aggregate distilled from a task's trials (sampling-invariant)."""

    build_vec: Tuple[float, ...]           # raw build-param values, in order
    recall_at_efmax: float                 # max recall reached over this build
    g_tau: Optional[float]                  # best feasible QPS (recall>=tau), or None


@dataclass
class TaskProfile:
    """Everything needed to compare a task, derived from its real trials only."""

    name: str
    build_param_order: List[str]
    observations: List[BuildObservation] = field(default_factory=list)
    frontier: List[Tuple[float, float]] = field(default_factory=list)  # (recall, qps)
    dataset_features: Dict[str, float] = field(default_factory=dict)

    @property
    def feasible_count(self) -> int:
        return sum(1 for o in self.observations if o.g_tau is not None)


# --------------------------------------------------------------------------
# Trial parsing -> per-build aggregates (sampling-invariant distillation)
# --------------------------------------------------------------------------

def _iter_success_trials(trials: Sequence[Dict[str, Any]]):
    for row in trials:
        if not isinstance(row, dict):
            continue
        if row.get("status") not in (None, "success"):
            # tolerate rows already pre-filtered (no status field)
            if row.get("status") != "success":
                continue
        params = row.get("params")
        metrics = row.get("metrics")
        if not isinstance(params, dict):
            continue
        if isinstance(metrics, dict):
            recall = metrics.get("recall")
            qps = metrics.get("qps")
        else:
            recall = row.get("recall")
            qps = row.get("qps")
        if recall is None or qps is None:
            continue
        try:
            recall_f = float(recall)
            qps_f = float(qps)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(recall_f) and math.isfinite(qps_f)):
            continue
        yield params, recall_f, qps_f


def _infer_build_order(
    param_keys: Sequence[str],
    search_param: str,
    preferred_order: Sequence[str],
) -> List[str]:
    keys = set(param_keys)
    keys.discard(search_param)
    ordered = [name for name in preferred_order if name in keys]
    extras = sorted(k for k in keys if k not in ordered)
    return ordered + extras


def build_task_profile(
    trials: Sequence[Dict[str, Any]],
    *,
    name: str,
    recall_threshold: float,
    cfg: SimilarityConfig,
    dataset_features: Optional[Dict[str, float]] = None,
    preferred_order: Sequence[str] = PARAM_ORDER,
) -> TaskProfile:
    """Distill a task's raw trials into a sampling-invariant profile.

    Groups trials by build (all params except the search param), then per build
    keeps the max recall reached (recall_at_efmax) and the best QPS among
    recall>=tau rows (g_tau). Also extracts the (recall, qps) Pareto frontier
    used by the Method-C fallback.
    """
    search_param = cfg.search_param
    # First pass: collect param key universe to fix a stable build order.
    rows = list(_iter_success_trials(trials))
    key_universe: set[str] = set()
    for params, _, _ in rows:
        key_universe.update(params.keys())
    build_order = _infer_build_order(key_universe, search_param, preferred_order)

    # Group by build vector.
    grouped: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    raw_points: List[Tuple[float, float]] = []
    for params, recall_f, qps_f in rows:
        try:
            build_vec = tuple(float(params[name]) for name in build_order)
        except (KeyError, TypeError, ValueError):
            continue
        raw_points.append((recall_f, qps_f))
        slot = grouped.setdefault(
            build_vec,
            {"recall_max": -math.inf, "g_tau": None},
        )
        if recall_f > slot["recall_max"]:
            slot["recall_max"] = recall_f
        if recall_f >= recall_threshold:
            if slot["g_tau"] is None or qps_f > slot["g_tau"]:
                slot["g_tau"] = qps_f

    observations = [
        BuildObservation(
            build_vec=bvec,
            recall_at_efmax=float(slot["recall_max"]),
            g_tau=(float(slot["g_tau"]) if slot["g_tau"] is not None else None),
        )
        for bvec, slot in grouped.items()
    ]

    return TaskProfile(
        name=name,
        build_param_order=build_order,
        observations=observations,
        frontier=_pareto_frontier(raw_points),
        dataset_features=dict(dataset_features or {}),
    )


def _pareto_frontier(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(recall, qps) upper-right Pareto frontier, sorted by recall ascending.

    A point is on the frontier if no other point has both higher recall and
    higher qps. Sampling-invariant: depends only on the reachable set, ordering
    is canonical.
    """
    if not points:
        return []
    # Sort by recall asc, then qps desc; sweep keeping running max qps from right.
    uniq = sorted(set(points), key=lambda rp: (rp[0], rp[1]))
    frontier: List[Tuple[float, float]] = []
    best_qps = -math.inf
    for recall, qps in reversed(uniq):  # high recall -> low recall
        if qps > best_qps:
            frontier.append((recall, qps))
            best_qps = qps
    frontier.reverse()
    return frontier


# --------------------------------------------------------------------------
# GP proxies on a task's own real trials (Method A core)
# --------------------------------------------------------------------------

class _ConstantProxy:
    """Degenerate proxy used when a GP cannot be fit (e.g. single distinct build).

    Returns a constant mean with large variance so the variance weighting in
    ResponseSim naturally down-weights every probe point for this task.
    """

    def __init__(self, mean: float):
        self._mean = float(mean)

    def predict(self, _grid):  # returns (mean, var) per row
        return [(self._mean, 1e6) for _ in _grid]


class _GPProxy:
    """Thin wrapper over a fitted botorch SingleTaskGP returning (mean, var)."""

    def __init__(self, model, torch_mod):
        self._model = model
        self._torch = torch_mod

    def predict(self, grid: Sequence[Sequence[float]]):
        torch = self._torch
        x = torch.tensor([list(row) for row in grid], dtype=torch.double)
        with torch.no_grad():
            posterior = self._model.posterior(x)
            mean = posterior.mean.squeeze(-1).tolist()
            var = posterior.variance.squeeze(-1).tolist()
        if not isinstance(mean, list):
            mean = [mean]
            var = [var]
        return list(zip((float(m) for m in mean), (float(v) for v in var)))


def _normalize_builds(
    observations: Sequence[BuildObservation],
    bounds: Sequence[Tuple[float, float]],
) -> List[List[float]]:
    out: List[List[float]] = []
    for obs in observations:
        row: List[float] = []
        for value, (lo, hi) in zip(obs.build_vec, bounds):
            span = hi - lo
            row.append(0.0 if span <= 0 else max(0.0, min(1.0, (value - lo) / span)))
        out.append(row)
    return out


def _fit_single_gp(train_x: List[List[float]], train_y: List[float]):
    """Fit a SingleTaskGP (mirrors utils/rfanns_scbo.py::_fit_surrogates).

    Returns a proxy exposing predict(grid)->[(mean,var)]. Falls back to a
    constant high-variance proxy on insufficient/degenerate data.
    """
    finite = [
        (x, y)
        for x, y in zip(train_x, train_y)
        if y is not None and math.isfinite(float(y))
    ]
    if len(finite) < 2:
        mean = float(finite[0][1]) if finite else 0.0
        return _ConstantProxy(mean)
    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        tx = torch.tensor([list(x) for x, _ in finite], dtype=torch.double)
        ty = torch.tensor([[float(y)] for _, y in finite], dtype=torch.double)
        model = SingleTaskGP(train_X=tx, train_Y=ty, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return _GPProxy(model, torch)
    except Exception:
        mean = sum(y for _, y in finite) / len(finite)
        return _ConstantProxy(mean)


def fit_task_frontier_gp(
    profile: TaskProfile,
    bounds: Sequence[Tuple[float, float]],
):
    """Fit (qps_proxy, recall_proxy) on a task's own builds in normalized space.

    qps proxy targets G_tau (best feasible QPS per build); recall proxy targets
    recall_at_efmax (build-level feasibility). Both return calibrated (mean,var).
    """
    norm_x = _normalize_builds(profile.observations, bounds)
    # QPS proxy: only builds with a feasible QPS contribute a target.
    qps_x = [x for x, o in zip(norm_x, profile.observations) if o.g_tau is not None]
    qps_y = [o.g_tau for o in profile.observations if o.g_tau is not None]
    recall_x = norm_x
    recall_y = [o.recall_at_efmax for o in profile.observations]
    qps_proxy = _fit_single_gp(qps_x, qps_y)  # type: ignore[arg-type]
    recall_proxy = _fit_single_gp(recall_x, recall_y)
    return qps_proxy, recall_proxy


def resolve_build_bounds(
    profiles: Sequence[TaskProfile],
    cfg: SimilarityConfig,
    build_order: Sequence[str],
) -> List[Tuple[float, float]]:
    """Shared coordinate frame: explicit cfg.build_bounds, else union of observed.

    Using the union of both tasks' observed build values guarantees the grid is
    expressed in a common frame even when no domain config is supplied.
    """
    bounds: List[Tuple[float, float]] = []
    cfg_bounds = cfg.build_bounds or {}
    for dim_idx, name in enumerate(build_order):
        if name in cfg_bounds:
            lo, hi = cfg_bounds[name]
            bounds.append((float(lo), float(hi)))
            continue
        vals: List[float] = []
        for prof in profiles:
            if name in prof.build_param_order:
                j = prof.build_param_order.index(name)
                vals.extend(o.build_vec[j] for o in prof.observations)
        if vals:
            lo, hi = min(vals), max(vals)
            bounds.append((lo, hi if hi > lo else lo + 1.0))
        else:
            bounds.append((0.0, 1.0))
    return bounds


def build_probe_grid(
    build_order: Sequence[str],
    cfg: SimilarityConfig,
) -> List[List[float]]:
    """Virtual probe grid in normalized build space [0,1]^d. Never evaluated.

    Uniform per-dim grid, capped so the pair count stays bounded.
    """
    d = len(build_order)
    if d == 0:
        return []
    n = max(2, int(cfg.grid_per_dim))
    axis = [i / (n - 1) for i in range(n)]
    grid: List[List[float]] = [[]]
    for _ in range(d):
        grid = [point + [a] for point in grid for a in axis]
        if len(grid) > 4096:  # hard guard before pairing
            break
    return grid


# --------------------------------------------------------------------------
# ResponseSim: per-pair RG/QC, intra-task z-score, variance-weighted match
# --------------------------------------------------------------------------

def _make_pairs(n_points: int, max_pairs: int) -> List[Tuple[int, int]]:
    pairs = [(a, b) for a in range(n_points) for b in range(a + 1, n_points)]
    if len(pairs) <= max_pairs:
        return pairs
    stride = max(1, len(pairs) // max_pairs)
    return pairs[::stride][:max_pairs]


def _zscore(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    std = math.sqrt(var)
    return [(v - mu) / (std + EPS) for v in values]


def _task_pair_responses(grid, pairs, qps_proxy, recall_proxy):
    """Per-pair standardized (RG, QC) and per-pair combined posterior variance."""
    qps_pred = qps_proxy.predict(grid)
    rec_pred = recall_proxy.predict(grid)
    rg_raw: List[float] = []
    qc_raw: List[float] = []
    pair_var: List[float] = []
    for a, b in pairs:
        rec_a = min(1.0 - EPS, max(0.0, rec_pred[a][0]))
        rec_b = min(1.0 - EPS, max(0.0, rec_pred[b][0]))
        rg_raw.append(math.log((1.0 - rec_a + EPS) / (1.0 - rec_b + EPS)))
        qa = max(EPS, qps_pred[a][0])
        qb = max(EPS, qps_pred[b][0])
        qc_raw.append(math.log(qa / qb))
        pair_var.append(
            qps_pred[a][1] + qps_pred[b][1] + rec_pred[a][1] + rec_pred[b][1]
        )
    return _zscore(rg_raw), _zscore(qc_raw), pair_var


def response_sim_via_gp(
    profile_new: TaskProfile,
    profile_hist: TaskProfile,
    bounds: Sequence[Tuple[float, float]],
    build_order: Sequence[str],
    cfg: SimilarityConfig,
) -> float:
    """Variance-weighted response-pattern similarity on the shared probe grid."""
    grid = build_probe_grid(build_order, cfg)
    if len(grid) < 2:
        return 0.0
    pairs = _make_pairs(len(grid), cfg.max_pairs)
    if not pairs:
        return 0.0

    new_qps, new_rec = fit_task_frontier_gp(profile_new, bounds)
    hist_qps, hist_rec = fit_task_frontier_gp(profile_hist, bounds)

    rg_n, qc_n, var_n = _task_pair_responses(grid, pairs, new_qps, new_rec)
    rg_h, qc_h, var_h = _task_pair_responses(grid, pairs, hist_qps, hist_rec)

    num = 0.0
    den = 0.0
    for i in range(len(pairs)):
        d_p = (rg_n[i] - rg_h[i]) ** 2 + (qc_n[i] - qc_h[i]) ** 2
        # inverse-variance weight: probe regions where either task lacks data
        # (high posterior variance) contribute little to the similarity.
        w_p = 1.0 / (var_n[i] + var_h[i] + EPS)
        num += w_p * math.exp(-d_p)
        den += w_p
    return float(num / den) if den > 0 else 0.0


# --------------------------------------------------------------------------
# Method C: frontier-curve distance (cold-start / cross-variant fallback)
# --------------------------------------------------------------------------

def _interp_log_qps(frontier: Sequence[Tuple[float, float]], recall: float) -> Optional[float]:
    if not frontier or recall < frontier[0][0] or recall > frontier[-1][0]:
        return None
    for i in range(1, len(frontier)):
        r0, q0 = frontier[i - 1]
        r1, q1 = frontier[i]
        if r0 <= recall <= r1:
            lq0 = math.log(max(EPS, q0))
            lq1 = math.log(max(EPS, q1))
            if r1 - r0 <= 0:
                return lq1
            t = (recall - r0) / (r1 - r0)
            return lq0 + t * (lq1 - lq0)
    return None


def frontier_curve_distance(profile_a: TaskProfile, profile_b: TaskProfile) -> float:
    """(recall, log-qps) Pareto-frontier shape similarity -> [0,1].

    Compares the reachable frontier directly; needs no shared parameterization,
    so it works cross-variant and as a cold-start fallback. Similarity is
    exp(-mean |log-qps gap|) over the overlapping recall range.
    """
    fa, fb = profile_a.frontier, profile_b.frontier
    if not fa or not fb:
        return 0.0
    lo = max(fa[0][0], fb[0][0])
    hi = min(fa[-1][0], fb[-1][0])
    if hi <= lo:
        return 0.0
    samples = 20
    gaps: List[float] = []
    for s in range(samples + 1):
        r = lo + (hi - lo) * s / samples
        la = _interp_log_qps(fa, r)
        lb = _interp_log_qps(fb, r)
        if la is not None and lb is not None:
            gaps.append(abs(la - lb))
    if not gaps:
        return 0.0
    return float(math.exp(-sum(gaps) / len(gaps)))


# --------------------------------------------------------------------------
# DatasetSim + top-level TaskSim
# --------------------------------------------------------------------------

_DATASET_WEIGHTS = {"d": 0.4, "N": 0.3, "k": 0.3}
_DATASET_SCALES = {"d": 256.0, "N": 1.0e6, "k": 10.0}


def dataset_similarity(feats_a: Dict[str, float], feats_b: Dict[str, float]) -> Optional[float]:
    """Gaussian-kernel similarity over cheap dataset features {d, N, k}.

    Returns None when neither task carries usable features, so the caller can
    fall back to response-only similarity.
    """
    total = 0.0
    used = 0.0
    for key, weight in _DATASET_WEIGHTS.items():
        va, vb = feats_a.get(key), feats_b.get(key)
        if va is None or vb is None:
            continue
        scale = _DATASET_SCALES.get(key, 1.0)
        total += weight * ((float(va) - float(vb)) / scale) ** 2
        used += weight
    if used <= 0:
        return None
    return float(math.exp(-total / used))


def task_similarity(
    new_trials: Sequence[Dict[str, Any]],
    hist_trials: Sequence[Dict[str, Any]],
    *,
    recall_threshold: float,
    cfg: Optional[SimilarityConfig] = None,
    new_name: str = "new",
    hist_name: str = "hist",
    new_dataset_features: Optional[Dict[str, float]] = None,
    hist_dataset_features: Optional[Dict[str, float]] = None,
    preferred_order: Sequence[str] = PARAM_ORDER,
) -> Dict[str, Any]:
    """Sampling-invariant similarity between current and a historical task.

    Both sides are distilled from their *own* real trials; alignment happens on
    a virtual probe grid via GP proxies. Falls back to frontier-curve distance
    when either side has too few builds for a meaningful GP, or when
    cfg.metric == 'frontier'.
    """
    cfg = cfg or SimilarityConfig()
    prof_new = build_task_profile(
        new_trials, name=new_name, recall_threshold=recall_threshold,
        cfg=cfg, dataset_features=new_dataset_features, preferred_order=preferred_order,
    )
    prof_hist = build_task_profile(
        hist_trials, name=hist_name, recall_threshold=recall_threshold,
        cfg=cfg, dataset_features=hist_dataset_features, preferred_order=preferred_order,
    )

    build_order = prof_new.build_param_order or prof_hist.build_param_order
    enough = (
        len(prof_new.observations) >= cfg.min_trials_for_gp
        and len(prof_hist.observations) >= cfg.min_trials_for_gp
    )

    if cfg.metric == "frontier" or not enough or not build_order:
        response_sim = frontier_curve_distance(prof_new, prof_hist)
        method_used = "frontier"
    else:
        bounds = resolve_build_bounds([prof_new, prof_hist], cfg, build_order)
        response_sim = response_sim_via_gp(prof_new, prof_hist, bounds, build_order, cfg)
        method_used = "response_gp"

    ds_sim = dataset_similarity(prof_new.dataset_features, prof_hist.dataset_features)
    if ds_sim is None:
        similarity = response_sim
    else:
        similarity = cfg.alpha * response_sim + cfg.beta * ds_sim

    return {
        "similarity": float(max(0.0, min(1.0, similarity))),
        "response_sim": float(response_sim),
        "dataset_sim": (float(ds_sim) if ds_sim is not None else None),
        "method": method_used,
        "build_order": list(build_order),
        "new_builds": len(prof_new.observations),
        "hist_builds": len(prof_hist.observations),
        "new_feasible_builds": prof_new.feasible_count,
        "hist_feasible_builds": prof_hist.feasible_count,
    }


# ==========================================================================
# DEPRECATED: absolute-QPS partial-order similarity (sampling-coupled).
# Retained for backward compatibility / reference. Not used by the transfer
# pipeline -- see module docstring and task_similarity() above.
# ==========================================================================

def compute_task_similarity(
    actual_qps: Sequence[float],
    predicted_qps: Sequence[float],
) -> Dict[str, Any]:
    """DEPRECATED. Pairwise partial-order consistency over absolute QPS.

    Coupled to the current run's sampled points and ignores recall; kept only
    for backward compatibility. Prefer ``task_similarity``.
    """
    actual = [_to_finite_float(value, "actual_qps", idx) for idx, value in enumerate(actual_qps)]
    predicted = [_to_finite_float(value, "predicted_qps", idx) for idx, value in enumerate(predicted_qps)]

    if len(actual) != len(predicted):
        raise ValueError(
            f"actual_qps and predicted_qps must have the same length, got {len(actual)} and {len(predicted)}."
        )
    if len(actual) < 2:
        raise ValueError(f"At least 2 samples are required, got {len(actual)}.")

    sample_count = len(actual)
    total_pairs = sample_count * (sample_count - 1) // 2
    consistent_pairs = 0
    for j in range(sample_count):
        for k in range(j + 1, sample_count):
            if (actual[j] <= actual[k]) == (predicted[j] <= predicted[k]):
                consistent_pairs += 1

    return {
        "similarity": float(consistent_pairs / total_pairs),
        "consistent_pairs": int(consistent_pairs),
        "inconsistent_pairs": int(total_pairs - consistent_pairs),
        "total_pairs": int(total_pairs),
        "sample_count": int(sample_count),
    }


def compute_similarity_with_model_predictions(
    actual_qps: Sequence[float],
    feature_rows: Sequence[Sequence[float]],
    model: Any,
) -> Dict[str, Any]:
    """DEPRECATED. Runs a historical QPS model on the current task's points.

    Extrapolates each historical model outside its own sampled region. Kept for
    backward compatibility only. Prefer ``task_similarity``.
    """
    if not hasattr(model, "predict"):
        raise ValueError("Model object must provide a 'predict' method.")
    predicted = model.predict(feature_rows)
    if len(predicted) != len(feature_rows):
        raise ValueError(
            f"Model prediction length mismatch: expected {len(feature_rows)}, got {len(predicted)}."
        )
    predicted_qps = [_to_finite_float(value, "predicted_qps", idx) for idx, value in enumerate(predicted)]
    return compute_task_similarity(actual_qps=actual_qps, predicted_qps=predicted_qps)


def _load_trials_samples(
    trials_path: Union[str, Path],
    recall_threshold: float,
    param_order: Sequence[str] = PARAM_ORDER,
) -> Tuple[List[List[float]], List[float]]:
    """DEPRECATED helper for compute_task_similarity_from_model."""
    path = Path(trials_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trials file not found: {path}")

    threshold = float(recall_threshold)
    features: List[List[float]] = []
    actual_qps: List[float] = []
    for line_idx, row in enumerate(_load_trials_jsonl(path), start=1):
        if row.get("status") != "success":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or "recall" not in metrics or "qps" not in metrics:
            continue
        try:
            recall = _to_finite_float(metrics["recall"], "metrics.recall", line_idx)
            qps = _to_finite_float(metrics["qps"], "metrics.qps", line_idx)
        except ValueError:
            continue
        if recall < threshold:
            continue
        params = row.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"Line {line_idx}: params must be a dict for qualified sample.")
        feature_row: List[float] = []
        for name in param_order:
            if name not in params:
                raise ValueError(f"Line {line_idx}: missing parameter '{name}' in params.")
            feature_row.append(_to_finite_float(params[name], f"params.{name}", line_idx))
        features.append(feature_row)
        actual_qps.append(qps)

    if len(features) < 2:
        raise ValueError(f"At least 2 usable samples are required after filtering, got {len(features)}.")
    return features, actual_qps


def compute_task_similarity_from_model(
    trials_path: Union[str, Path],
    model_path: Union[str, Path],
    recall_threshold: float,
    param_order: Sequence[str] = PARAM_ORDER,
) -> Dict[str, Any]:
    """DEPRECATED. Sampling-coupled similarity from a pickled QPS model.

    Prefer ``task_similarity``. Retained for backward compatibility.
    """
    features, actual_qps = _load_trials_samples(
        trials_path=trials_path, recall_threshold=recall_threshold, param_order=param_order,
    )
    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")
    with resolved_model_path.open("rb") as f:
        model = pickle.load(f)
    return compute_similarity_with_model_predictions(
        actual_qps=actual_qps, feature_rows=features, model=model,
    )


def _load_trials_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sampling-invariant ANNS task similarity via frontier alignment. "
            "Compares two tasks' real trials (no shared sampled points required)."
        )
    )
    parser.add_argument("--new_trials", type=str, required=True, help="Current task trials JSONL.")
    parser.add_argument("--hist_trials", type=str, required=True, help="Historical task trials JSONL.")
    parser.add_argument("--recall_threshold", type=float, required=True, help="Feasibility recall threshold tau.")
    parser.add_argument("--metric", type=str, default="response_gp", choices=["response_gp", "frontier"])
    parser.add_argument("--grid_per_dim", type=int, default=4)
    parser.add_argument("--min_trials_for_gp", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = SimilarityConfig(
        metric=args.metric,
        grid_per_dim=args.grid_per_dim,
        min_trials_for_gp=args.min_trials_for_gp,
    )
    new_path = Path(args.new_trials).expanduser().resolve()
    hist_path = Path(args.hist_trials).expanduser().resolve()
    if not new_path.exists():
        raise FileNotFoundError(f"Trials file not found: {new_path}")
    if not hist_path.exists():
        raise FileNotFoundError(f"Trials file not found: {hist_path}")

    result = task_similarity(
        new_trials=_load_trials_jsonl(new_path),
        hist_trials=_load_trials_jsonl(hist_path),
        recall_threshold=args.recall_threshold,
        cfg=cfg,
        new_name=new_path.stem,
        hist_name=hist_path.stem,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



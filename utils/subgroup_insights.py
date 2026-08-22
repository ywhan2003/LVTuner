"""Shared subgroup-mining helpers for ANN tuning insight builders."""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_WEIGHT_FEASIBLE_COUNT = 0.4
DEFAULT_WEIGHT_RECALL_MARGIN = 0.2
DEFAULT_WEIGHT_FEASIBLE_QPS = 0.4

# Subgroup mining now emits a *coarse range* (over feasible trials) plus a
# tighter *elite range* (over the top-QPS feasible trials), instead of point
# medians. These knobs control elite selection and the multiplicative quality
# score Q(g)=S(g)*q_hat(g)*[lambda*F(g)+(1-lambda)*R(g)]*Validity(g).
DEFAULT_ELITE_RATIO = 0.10          # top fraction of feasible trials = elite
DEFAULT_ELITE_FLOOR = 3             # min elite count (keeps elite range a box)
DEFAULT_MIN_FEASIBLE = 6            # min feasible trials for a usable subgroup
DEFAULT_GAMMA = 8.0                 # R(g) recall-margin penalty steepness
DEFAULT_LAMBDA_FEASIBLE = 0.5       # blend weight lambda between F(g) and R(g)
DEFAULT_MIN_RELIABLE_COVERED = 8    # m_min for coverage reliability S(g)

# Logical parameter names vary across builders (hnswlib uses "ef_construction",
# rfanns uses "efConstruction"). Roles are resolved by membership below so the
# ANNS constraint (ef <= ef_construction) and direction labels never hardcode a
# single spelling.
EF_ROLE_KEYS = ("ef", "efSearch")
EFC_ROLE_KEYS = ("ef_construction", "efConstruction", "efC")
M_ROLE_KEYS = ("M", "m")


def resolve_param_role_map(
    param_order: Sequence[str],
    param_role_map: Dict[str, str] | None = None,
) -> Dict[str, str | None]:
    """Map ANNS roles -> actual parameter names present in ``param_order``.

    Returns a dict with keys ``ef``, ``ef_construction`` and ``M``. An explicit
    ``param_role_map`` wins; otherwise roles are auto-detected by membership in
    the ``*_ROLE_KEYS`` tuples (case-insensitive).
    """
    roles: Dict[str, str | None] = {"ef": None, "ef_construction": None, "M": None}
    available = list(param_order)
    lowered = {name.lower(): name for name in available}

    if param_role_map:
        for role in roles:
            chosen = param_role_map.get(role)
            if chosen and chosen in available:
                roles[role] = chosen

    def _auto(role: str, keys: Sequence[str]) -> None:
        if roles[role] is not None:
            return
        for key in keys:
            if key in available:
                roles[role] = key
                return
            if key.lower() in lowered:
                roles[role] = lowered[key.lower()]
                return

    # ef_construction before ef so "efConstruction" is not grabbed by the "ef" rule
    _auto("ef_construction", EFC_ROLE_KEYS)
    _auto("M", M_ROLE_KEYS)
    for key in EF_ROLE_KEYS:
        if roles["ef"] is not None:
            break
        match = key if key in available else lowered.get(key.lower())
        if match and match != roles["ef_construction"]:
            roles["ef"] = match
    return roles


def _range_param_map(
    observations: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    param_order: Sequence[str],
) -> Dict[str, List[float | int]]:
    """[min, max] of each parameter over the given observation indices."""
    ranges: Dict[str, List[float | int]] = {}
    for name in param_order:
        values = [float(observations[idx]["params"][name]) for idx in indices]
        if not values:
            ranges[name] = [0, 0]
            continue
        ranges[name] = [_maybe_int(min(values)), _maybe_int(max(values))]
    return ranges


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid numeric value: {value}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"invalid numeric value: {value}")
    return parsed


def _maybe_int(value: float) -> float | int:
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return int(rounded)
    return value


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = _maybe_int(float(value))
    if isinstance(rounded, int):
        return str(rounded)
    return f"{float(rounded):.6g}"


def _median_value(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _median_param_map(
    observations: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    param_order: Sequence[str],
) -> Dict[str, float | int]:
    medians: Dict[str, float | int] = {}
    for name in param_order:
        values = [float(observations[idx]["params"][name]) for idx in indices]
        if not values:
            medians[name] = 0
            continue
        medians[name] = _maybe_int(_median_value(values))
    return medians


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "insight"


def _threshold_slug(value: float) -> str:
    text = f"{float(value):.6g}".replace(".", "p").replace("-", "m")
    return f"recall_{text}"


def _normalize_recall_thresholds(
    recall_threshold: float | None = None,
    recall_thresholds: Sequence[float] | None = None,
) -> List[float]:
    raw_values: List[Any] = []
    if recall_thresholds:
        raw_values.extend(recall_thresholds)
    elif recall_threshold is not None:
        raw_values.append(recall_threshold)
    else:
        raw_values.append(0.90)

    thresholds: List[float] = []
    seen: set[float] = set()
    for raw_value in raw_values:
        parsed = _safe_float(raw_value)
        if not (0.0 < parsed <= 1.0):
            raise ValueError("recall thresholds must be in (0, 1].")
        key = round(parsed, 12)
        if key in seen:
            continue
        seen.add(key)
        thresholds.append(float(parsed))

    if not thresholds:
        raise ValueError("At least one recall threshold is required.")
    return thresholds


def _normalize_task_name(task_name: str) -> str:
    stem = Path(str(task_name).strip()).stem
    if not stem:
        raise ValueError("task_name must not be empty")
    return stem


def _prefix_sum(values: Sequence[float]) -> List[float]:
    running = 0.0
    result = [0.0]
    for value in values:
        running += value
        result.append(running)
    return result


def lsqm_cut_points(
    values: Sequence[float | int],
    targets: Sequence[float | int],
    max_bins: int = 4,
) -> List[float]:
    """LSQM discretization: DP cut points minimizing within-bin target SSE.

    NOTE: cuts are *marginal* (a single parameter binned over all observations),
    so they can be confounded by other parameters. Multi-dimensional subgroup
    discovery partially compensates. Callers should pre-filter structurally
    invalid configs (e.g. ef>efC) so cuts are not skewed by impossible points.
    # TODO(confounding): conditional/residualized LSQM binning.
    """
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    if len(values) <= 1 or max_bins <= 1:
        return []

    sorted_pairs = sorted((float(v), float(t)) for v, t in zip(values, targets))
    unique_x: List[float] = []
    counts: List[float] = []
    sums_y: List[float] = []
    sums_y2: List[float] = []
    for x_value, y_value in sorted_pairs:
        if unique_x and abs(x_value - unique_x[-1]) <= 1e-12:
            counts[-1] += 1.0
            sums_y[-1] += y_value
            sums_y2[-1] += y_value * y_value
        else:
            unique_x.append(x_value)
            counts.append(1.0)
            sums_y.append(y_value)
            sums_y2.append(y_value * y_value)

    point_count = len(unique_x)
    if point_count <= 1:
        return []

    bin_count = min(max_bins, point_count)
    prefix_counts = _prefix_sum(counts)
    prefix_sums = _prefix_sum(sums_y)
    prefix_sums2 = _prefix_sum(sums_y2)

    def interval_sse(left_idx: int, right_idx: int) -> float:
        count = prefix_counts[right_idx + 1] - prefix_counts[left_idx]
        if count <= 0.0:
            return 0.0
        sum_y = prefix_sums[right_idx + 1] - prefix_sums[left_idx]
        sum_y2 = prefix_sums2[right_idx + 1] - prefix_sums2[left_idx]
        return max(0.0, sum_y2 - (sum_y * sum_y) / count)

    inf = float("inf")
    dp = [[inf for _ in range(point_count)] for _ in range(bin_count + 1)]
    split = [[-1 for _ in range(point_count)] for _ in range(bin_count + 1)]

    for j in range(point_count):
        dp[1][j] = interval_sse(0, j)

    for k in range(2, bin_count + 1):
        for right_idx in range(k - 1, point_count):
            best_cost = inf
            best_split = -1
            for left_split in range(k - 2, right_idx):
                candidate_cost = dp[k - 1][left_split] + interval_sse(left_split + 1, right_idx)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_split = left_split
            dp[k][right_idx] = best_cost
            split[k][right_idx] = best_split

    right = point_count - 1
    boundaries: List[int] = []
    for k in range(bin_count, 1, -1):
        left_split = split[k][right]
        if left_split < 0:
            break
        boundaries.append(left_split)
        right = left_split
    boundaries.reverse()

    cut_points: List[float] = []
    for left_idx in boundaries:
        if left_idx + 1 >= len(unique_x):
            continue
        # Cut = midpoint of two adjacent grid values. For integer parameter
        # grids there are no observed values strictly between them, so flooring
        # a fractional cut to an integer preserves bin assignment (value <= cut)
        # while keeping cut points / intervals integer-valued for display.
        cut = (unique_x[left_idx] + unique_x[left_idx + 1]) / 2.0
        if abs(cut - round(cut)) <= 1e-9:
            cut = float(round(cut))
        else:
            cut = float(math.floor(cut))
        cut_points.append(cut)
    return cut_points


def merge_cut_points(*groups: Sequence[float], tolerance: float = 1e-9) -> List[float]:
    raw_points: List[float] = []
    for group in groups:
        for value in group:
            raw_points.append(float(value))
    if not raw_points:
        return []

    raw_points.sort()
    merged = [raw_points[0]]
    for value in raw_points[1:]:
        if abs(value - merged[-1]) > tolerance:
            merged.append(value)
    return merged


def _build_intervals(
    min_value: float,
    max_value: float,
    cut_points: Sequence[float],
) -> List[Dict[str, Any]]:
    if not cut_points:
        return [
            {
                "index": 0,
                "lower": min_value,
                "upper": max_value,
                "lower_inclusive": True,
                "upper_inclusive": True,
            }
        ]

    boundaries = [min_value, *cut_points, max_value]
    intervals: List[Dict[str, Any]] = []
    for idx in range(len(boundaries) - 1):
        intervals.append(
            {
                "index": idx,
                "lower": boundaries[idx],
                "upper": boundaries[idx + 1],
                "lower_inclusive": idx == 0,
                "upper_inclusive": True,
            }
        )
    return intervals


def _assign_bin(value: float, cut_points: Sequence[float]) -> int:
    for idx, cut_point in enumerate(cut_points):
        if value <= cut_point:
            return idx
    return len(cut_points)


def build_parameter_discretization(
    observations: Sequence[Dict[str, Any]],
    param_order: Sequence[str],
    max_bins: int = 4,
) -> Dict[str, Dict[str, Any]]:
    if not observations:
        raise ValueError("observations must not be empty")

    discretization: Dict[str, Dict[str, Any]] = {}
    recalls = [float(row["recall"]) for row in observations]
    qps_values = [float(row["qps"]) for row in observations]

    for param in param_order:
        values = [float(row["params"][param]) for row in observations]
        cuts_recall = lsqm_cut_points(values, recalls, max_bins=max_bins)
        cuts_qps = lsqm_cut_points(values, qps_values, max_bins=max_bins)
        merged_cuts = merge_cut_points(cuts_recall, cuts_qps)

        min_value = min(values)
        max_value = max(values)
        intervals = _build_intervals(min_value, max_value, merged_cuts)
        bins = [_assign_bin(value, merged_cuts) for value in values]
        discretization[param] = {
            "min": min_value,
            "max": max_value,
            "cut_points": merged_cuts,
            "intervals": intervals,
            "bins": bins,
        }
    return discretization


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    sanitized = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(sanitized.values())
    if total <= 0.0:
        return {
            "feasible_count": DEFAULT_WEIGHT_FEASIBLE_COUNT,
            "recall_margin": DEFAULT_WEIGHT_RECALL_MARGIN,
            "feasible_qps": DEFAULT_WEIGHT_FEASIBLE_QPS,
        }
    return {key: value / total for key, value in sanitized.items()}


def discover_subgroups(
    observations: Sequence[Dict[str, Any]],
    discretization: Dict[str, Dict[str, Any]],
    recall_threshold: float,
    param_order: Sequence[str],
    max_subgroup_dims: int = 3,
    min_covered: int = 5,
    min_feasible: int = DEFAULT_MIN_FEASIBLE,
    elite_ratio: float = DEFAULT_ELITE_RATIO,
    *,
    elite_floor: int = DEFAULT_ELITE_FLOOR,
    param_role_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    if not observations:
        return []

    roles = resolve_param_role_map(param_order, param_role_map)
    ef_name = roles["ef"]
    efc_name = roles["ef_construction"]

    index_count = len(observations)
    all_indices = set(range(index_count))
    param_bins_to_indices: Dict[str, List[set[int]]] = {}
    for param in param_order:
        bins = discretization[param]["bins"]
        interval_count = len(discretization[param]["intervals"])
        index_sets = [set() for _ in range(interval_count)]
        for obs_idx, bin_idx in enumerate(bins):
            if 0 <= bin_idx < interval_count:
                index_sets[bin_idx].add(obs_idx)
        param_bins_to_indices[param] = index_sets

    subgroup_rows: List[Dict[str, Any]] = []
    max_dims = min(max_subgroup_dims, len(param_order))
    for dim_count in range(1, max_dims + 1):
        for params in combinations(param_order, dim_count):
            bin_ranges = [range(len(discretization[param]["intervals"])) for param in params]
            for bin_selection in product(*bin_ranges):
                covered = set(all_indices)
                for param, bin_idx in zip(params, bin_selection):
                    covered &= param_bins_to_indices[param][bin_idx]
                    if not covered:
                        break

                covered_count = len(covered)
                if covered_count < min_covered:
                    continue

                feasible = {idx for idx in covered if float(observations[idx]["recall"]) >= recall_threshold}
                feasible_count = len(feasible)
                if feasible_count < min_feasible:
                    continue

                covered_list = sorted(covered)
                feasible_list = sorted(feasible)
                recall_margins = [float(observations[idx]["recall"]) - recall_threshold for idx in covered_list]
                covered_qps = [float(observations[idx]["qps"]) for idx in covered_list]
                feasible_qps = [float(observations[idx]["qps"]) for idx in feasible_list]

                # Validity(g): ANNS hard constraint ef <= ef_construction.
                invalid_count = 0
                if ef_name and efc_name:
                    for idx in covered_list:
                        params_i = observations[idx]["params"]
                        if float(params_i[ef_name]) > float(params_i[efc_name]):
                            invalid_count += 1
                is_valid = invalid_count == 0

                # Elite = top-`elite_ratio` feasible trials by QPS, with a floor
                # so the elite range stays a box (does not collapse to a point).
                elite_count = max(int(elite_floor), int(math.ceil(feasible_count * elite_ratio)))
                elite_count = min(elite_count, feasible_count)
                elite_sorted = sorted(
                    feasible_list,
                    key=lambda idx: float(observations[idx]["qps"]),
                    reverse=True,
                )
                elite_list = elite_sorted[:elite_count]
                elite_qps_values = [float(observations[idx]["qps"]) for idx in elite_list]
                feasible_param_medians = _median_param_map(observations, feasible_list, param_order)
                elite_param_medians = _median_param_map(observations, elite_list, param_order)

                # Pinned dimensions (those defining this subgroup) collapse to a
                # single grid value in the data, so report their LSQM bin
                # interval as the range. Free dimensions keep the data-driven
                # range, where the elite set genuinely narrows vs the coarse set.
                # Parameters are integer-valued; cut points are midpoints between
                # grid values, so integerize the interval (ceil lower / floor
                # upper) to keep bounds on the real parameter grid.
                pinned_intervals: Dict[str, List[float | int]] = {}
                for param, bin_idx in zip(params, bin_selection):
                    interval = discretization[param]["intervals"][bin_idx]
                    lo = math.ceil(float(interval["lower"]) - 1e-9)
                    hi = math.floor(float(interval["upper"]) + 1e-9)
                    if lo > hi:
                        lo = hi = int(round(float(interval["lower"])))
                    pinned_intervals[param] = [int(lo), int(hi)]

                # Coarse range over feasible trials; elite range over elite trials.
                coarse_range = _range_param_map(observations, feasible_list, param_order)
                elite_range = _range_param_map(observations, elite_list, param_order)
                for param, interval_range in pinned_intervals.items():
                    coarse_range[param] = list(interval_range)
                    elite_range[param] = list(interval_range)

                direction_delta: Dict[str, float] = {}
                for param_name in param_order:
                    feasible_median = feasible_param_medians.get(param_name)
                    elite_median = elite_param_medians.get(param_name)
                    if feasible_median is None or elite_median is None:
                        direction_delta[param_name] = 0.0
                        continue
                    direction_delta[param_name] = float(elite_median) - float(feasible_median)

                region_dimensions = []
                for param, bin_idx in zip(params, bin_selection):
                    interval = discretization[param]["intervals"][bin_idx]
                    region_dimensions.append(
                        {
                            "parameter": param,
                            "interval_index": int(bin_idx),
                            "interval": interval,
                        }
                    )

                subgroup_rows.append(
                    {
                        "subgroup_key": tuple(zip(params, bin_selection)),
                        "region_dimensions": region_dimensions,
                        "covered_indices": covered_list,
                        "covered_index_set": covered,
                        "feasible_indices": feasible_list,
                        "elite_indices": elite_list,
                        "covered_count": covered_count,
                        "feasible_count": feasible_count,
                        "feasible_ratio": feasible_count / max(1, covered_count),
                        "median_recall_margin": _median_value(recall_margins),
                        "median_covered_qps": _median_value(covered_qps),
                        "median_feasible_qps": _median_value(feasible_qps),
                        "median_elite_qps": _median_value(elite_qps_values),
                        "qps_p75_feasible": _percentile(feasible_qps, 75.0),
                        "best_elite_qps": max(elite_qps_values) if elite_qps_values else 0.0,
                        "elite_ratio": elite_ratio,
                        "elite_count": elite_count,
                        "invalid_count": invalid_count,
                        "is_valid": is_valid,
                        "coarse_range": coarse_range,
                        "elite_range": elite_range,
                        "feasible_param_medians": feasible_param_medians,
                        "elite_param_medians": elite_param_medians,
                        "direction_delta": direction_delta,
                    }
                )

    return subgroup_rows


def score_subgroups(
    subgroups: Sequence[Dict[str, Any]],
    weights: Dict[str, float] | None = None,
    *,
    gamma: float = DEFAULT_GAMMA,
    lambda_feasible: float = DEFAULT_LAMBDA_FEASIBLE,
    min_reliable_covered: int = DEFAULT_MIN_RELIABLE_COVERED,
    enforce_validity: bool = True,
) -> List[Dict[str, Any]]:
    """Score subgroups with the multiplicative quality from methods.md:

        Q(g) = S(g) * q_hat(g) * [lambda*F(g) + (1-lambda)*R(g)] * Validity(g)

    where S(g)=min(1,|O(g)|/m_min) is coverage reliability, q_hat(g) is the
    normalized median covered QPS, F(g) is the feasible ratio, R(g)=
    exp(-gamma*max(0,-margin)) penalizes negative recall margin, and Validity(g)
    is 0 when the subgroup contains any ef>efC trial (when enforce_validity).

    ``weights`` is accepted for backward compatibility but no longer drives the
    score; it is echoed under ``quality_components.legacy_weights``.
    """
    if not subgroups:
        return []

    lam = min(1.0, max(0.0, float(lambda_feasible)))
    eps = 1e-12

    def _eligible(row: Dict[str, Any]) -> bool:
        return not (enforce_validity and not bool(row.get("is_valid", True)))

    covered_qps_values = [
        float(row.get("median_covered_qps", row.get("median_feasible_qps", 0.0)))
        for row in subgroups
        if _eligible(row)
    ]
    if covered_qps_values:
        q_min = min(covered_qps_values)
        q_max = max(covered_qps_values)
    else:
        q_min = q_max = 0.0

    scored: List[Dict[str, Any]] = []
    for row in subgroups:
        validity = 0.0 if (enforce_validity and not bool(row.get("is_valid", True))) else 1.0
        covered_count = float(row["covered_count"])
        median_qps = float(row.get("median_covered_qps", row.get("median_feasible_qps", 0.0)))
        margin = float(row["median_recall_margin"])
        feasible_ratio = float(row["feasible_ratio"])

        s_reliability = min(1.0, covered_count / max(1.0, float(min_reliable_covered)))
        if q_max > q_min:
            q_hat = (median_qps - q_min) / (q_max - q_min + eps)
        else:
            q_hat = 1.0 if median_qps > 0.0 else 0.0
        q_hat = min(1.0, max(0.0, q_hat))
        r_penalty = math.exp(-float(gamma) * max(0.0, -margin))
        blended = lam * feasible_ratio + (1.0 - lam) * r_penalty

        quality = s_reliability * q_hat * blended * validity

        scored_row = dict(row)
        scored_row["quality"] = float(quality)
        scored_row["quality_components"] = {
            "S_reliability": float(s_reliability),
            "q_hat": float(q_hat),
            "R_recall_penalty": float(r_penalty),
            "F_feasible_ratio": float(feasible_ratio),
            "blended": float(blended),
            "validity": float(validity),
            "gamma": float(gamma),
            "lambda_feasible": float(lam),
            "q_min": float(q_min),
            "q_max": float(q_max),
            "legacy_weights": dict(weights) if weights else {},
        }
        scored.append(scored_row)

    scored.sort(
        key=lambda item: (
            float(item["quality"]),
            float(item["median_feasible_qps"]),
            float(item["median_recall_margin"]),
            float(item["feasible_count"]),
        ),
        reverse=True,
    )
    return scored


def jaccard_overlap(indices_a: set[int], indices_b: set[int]) -> float:
    if not indices_a and not indices_b:
        return 0.0
    union = indices_a | indices_b
    if not union:
        return 0.0
    return len(indices_a & indices_b) / len(union)


def select_non_redundant_subgroups(
    scored_subgroups: Sequence[Dict[str, Any]],
    overlap_threshold: float = 0.6,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in scored_subgroups:
        current_set = row.get("covered_index_set") or set()
        redundant = False
        for picked in selected:
            picked_set = picked.get("covered_index_set") or set()
            if jaccard_overlap(current_set, picked_set) > overlap_threshold:
                redundant = True
                break
        if redundant:
            continue
        selected.append(row)
        if top_k > 0 and len(selected) >= top_k:
            break
    return selected


def _format_interval(interval: Dict[str, Any]) -> str:
    left = "[" if bool(interval.get("lower_inclusive", True)) else "("
    right = "]" if bool(interval.get("upper_inclusive", True)) else ")"
    lower = _fmt_number(float(interval["lower"]))
    upper = _fmt_number(float(interval["upper"]))
    return f"{left}{lower}, {upper}{right}"


def _region_text(region_dimensions: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in region_dimensions:
        param = item["parameter"]
        interval_text = _format_interval(item["interval"])
        parts.append(f"{param} in {interval_text}")
    return ", ".join(parts)


def _build_template_advice(card: Dict[str, Any]) -> str:
    region_text = card["region"]["description"]
    feasibility = card["feasibility"]
    perf = card["perf"]
    direction = card["direction"]["delta"]

    ranked_direction = sorted(
        direction.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    non_zero_direction = [item for item in ranked_direction if abs(float(item[1])) > 1e-12][:3]
    if non_zero_direction:
        direction_text = "; ".join(
            f"{name} {'increase' if delta > 0 else 'decrease'} ({_fmt_number(abs(float(delta)))})"
            for name, delta in non_zero_direction
        )
    else:
        direction_text = "no strong elite-direction shift"

    return (
        f"Focus on region: {region_text}. "
        f"This subgroup covers {feasibility['covered_count']} historical trials, with "
        f"{feasibility['feasible_count']} feasible trials (median recall margin "
        f"{_fmt_number(float(feasibility['median_recall_margin']))}). "
        f"Within feasible trials, elite top-{int(round(float(perf['elite_ratio']) * 100))}% "
        f"median QPS is {_fmt_number(float(perf['median_elite_qps']))}. "
        f"Recommended local direction: {direction_text}."
    )


def _delta_symbol(norm: float, strong: float = 0.30, weak: float = 0.10) -> str:
    """Map a normalized direction delta to an arrow symbol."""
    if norm > strong:
        return "↑↑"
    if norm > weak:
        return "↑"
    if norm < -strong:
        return "↓↓"
    if norm < -weak:
        return "↓"
    return "→"


def _qps_rank(value: float, all_values: Sequence[float]) -> float:
    """Fraction of values <= ``value`` (0..1), used for region typing."""
    if not all_values:
        return 0.0
    return sum(1 for v in all_values if v <= value) / float(len(all_values))


def _infer_region_type(
    *,
    feasible_ratio: float,
    covered_count: int,
    recall_margin: float,
    qps_rank: float,
    efc_over_ef: float,
    best_recall: float = 0.0,
    threshold: float = 0.0,
) -> str:
    """Region type for the single objective "maximize QPS s.t. recall>=tau".

    Since infeasible (recall<tau) and invalid (ef>efC) trials are filtered out
    before mining, the only signals that matter are reliability and QPS rank:

      unstable  -> too few samples / low feasible rate; avoid.
      top_qps   -> QPS among the highest (rank >= 0.75); explore here first.
      viable    -> feasible but not top-tier.

    A region is NOT "unstable" if its best recall exceeds the threshold —
    even a low feasible rate can be a BO exploration artifact rather than a
    genuine impossibility.
    """
    best_feasible = best_recall >= threshold if threshold > 0 else False
    if (feasible_ratio < 0.6 or covered_count < 10) and not best_feasible:
        return "unstable"
    if qps_rank >= 0.75:
        return "top_qps"
    return "viable"


def _infer_strategy(symbols: Dict[str, str], recall_margin: float) -> str:
    d_ef = symbols.get("ef", "→")
    d_efc = symbols.get("ef_construction", "→")
    d_m = symbols.get("M", "→")
    down = {"↓", "↓↓"}
    up = {"↑", "↑↑"}
    if d_ef in down and d_efc in up:
        return "graph_quality_compensation"
    if d_ef in down and d_m in down:
        return "aggressive_qps"
    if recall_margin < 0.0 and d_ef in up:
        return "search_expansion"
    if 0.0 < recall_margin < 0.02 and d_m in down:
        return "density_reduction"
    if recall_margin > 0.05 and d_ef in down:
        return "aggressive_qps"
    return "balanced_exploration"


def _build_hint(
    region_type: str,
    compact: Dict[str, List[float | int]],
    subgroup: Dict[str, Any],
) -> str:
    qps_best = float(subgroup.get("best_elite_qps", 0.0)) or float(subgroup.get("median_elite_qps", 0.0))
    qps_text = f"{qps_best / 1000.0:.1f}K" if qps_best >= 1000.0 else f"{qps_best:.0f}"
    ef_rng = compact.get("ef")
    efc_rng = compact.get("efC")
    if region_type == "unstable":
        return "low feasible rate / few samples; avoid"
    if region_type == "top_qps" and efc_rng and ef_rng:
        return f"top-QPS region: efC={_fmt_number(efc_rng[0])}-{_fmt_number(efc_rng[1])}, ef={_fmt_number(ef_rng[0])}-{_fmt_number(ef_rng[1])}, QPS~{qps_text}"
    if ef_rng:
        return f"viable: ef={_fmt_number(ef_rng[0])}-{_fmt_number(ef_rng[1])}, QPS~{qps_text}"
    return f"elite QPS~{qps_text} within tightened range"


def maybe_rewrite_advice_with_llm(
    cards: Sequence[Dict[str, Any]],
    enabled: bool = False,
    model_name: str = "",
    base_url: str = "",
    api_key: str = "",
    max_cards: int = 0,
) -> List[Dict[str, Any]]:
    rewritten_cards = [dict(card) for card in cards]
    if not enabled or not rewritten_cards:
        return rewritten_cards

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("openai package is required when --llm-advice is enabled") from exc

    resolved_base_url = base_url.strip() or os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("LLM_BASE_URL", "").strip()
    resolved_api_key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("LLM_API_KEY", "").strip()
    resolved_model_name = model_name.strip() or os.environ.get("LLM_MODEL_NAME", "").strip()
    if not resolved_base_url or not resolved_api_key or not resolved_model_name:
        raise RuntimeError(
            "LLM advice rewrite requires base URL, API key and model. "
            "Set --llm-base-url/--llm-api-key/--llm-model or "
            "LLM_BASE_URL/LLM_API_KEY/LLM_MODEL_NAME in .env."
        )

    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
    limit = len(rewritten_cards) if max_cards <= 0 else min(len(rewritten_cards), max_cards)

    for idx in range(limit):
        card = dict(rewritten_cards[idx])
        prompt = (
            "Rewrite the following RFANNS tuning advice into concise natural English. "
            "Keep all numeric details and direction semantics. "
            "Use 2-3 short sentences.\n\n"
            f"Advice:\n{card.get('advice_template', '')}\n"
        )
        response = client.chat.completions.create(
            model=resolved_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=180,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if rewritten:
            card["advice"] = rewritten
        rewritten_cards[idx] = card
    return rewritten_cards


def build_insight_cards(
    selected_subgroups: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    recall_threshold: float,
    task_name: str = "",
    *,
    param_role_map: Dict[str, str] | None = None,
    param_order: Sequence[str] | None = None,
    direction_strong_ratio: float = 0.30,
    direction_weak_ratio: float = 0.10,
) -> List[Dict[str, Any]]:
    task_slug = _slugify(task_name or "task")
    if param_order is None:
        param_order = list(selected_subgroups[0]["direction_delta"].keys()) if selected_subgroups else []
    roles = resolve_param_role_map(param_order, param_role_map)
    ef_name = roles["ef"]
    efc_name = roles["ef_construction"]
    m_name = roles["M"]
    role_for_param = {v: k for k, v in roles.items() if v}

    all_feasible_qps_median = [float(sg.get("median_feasible_qps", 0.0)) for sg in selected_subgroups]

    cards: List[Dict[str, Any]] = []
    for idx, subgroup in enumerate(selected_subgroups, start=1):
        region_dimensions = subgroup["region_dimensions"]
        region_description = _region_text(region_dimensions)
        direction_delta = subgroup["direction_delta"]
        direction_tendency = {
            name: ("up" if float(delta) > 0 else "down" if float(delta) < 0 else "flat")
            for name, delta in direction_delta.items()
        }

        coarse_range = dict(subgroup.get("coarse_range") or {})
        elite_range = dict(subgroup.get("elite_range") or {})

        # Direction symbols from delta normalized by the coarse-range span.
        symbols: Dict[str, str] = {}
        for name, delta in direction_delta.items():
            span = 0.0
            rng = coarse_range.get(name)
            if rng:
                span = float(rng[1]) - float(rng[0])
            norm = float(delta) / span if span > 1e-9 else 0.0
            symbol = _delta_symbol(norm, direction_strong_ratio, direction_weak_ratio)
            role = role_for_param.get(name, name)
            symbols[role] = symbol

        feasible_indices = subgroup["feasible_indices"]
        best_idx = -1
        if feasible_indices:
            best_idx = max(
                feasible_indices,
                key=lambda item_idx: (
                    float(observations[item_idx]["qps"]),
                    float(observations[item_idx]["recall"]),
                ),
            )
        best_params = {}
        best_qps = 0.0
        best_recall = 0.0
        if best_idx >= 0:
            best_params = dict(observations[best_idx].get("params") or {})
            best_qps = float(observations[best_idx]["qps"])
            best_recall = float(observations[best_idx]["recall"])

        recall_margin = float(subgroup["median_recall_margin"])
        feasible_ratio = float(subgroup["feasible_ratio"])
        covered_count = int(subgroup["covered_count"])
        qps_rank = _qps_rank(float(subgroup.get("median_feasible_qps", 0.0)), all_feasible_qps_median)

        # efC/ef ratio from the elite range (upper efC over lower ef).
        efc_over_ef = 0.0
        if ef_name and efc_name and elite_range.get(ef_name) and elite_range.get(efc_name):
            ef_lo = float(elite_range[ef_name][0])
            efc_hi = float(elite_range[efc_name][1])
            if ef_lo > 1e-9:
                efc_over_ef = efc_hi / ef_lo

        region_type = _infer_region_type(
            feasible_ratio=feasible_ratio,
            covered_count=covered_count,
            recall_margin=recall_margin,
            qps_rank=qps_rank,
            efc_over_ef=efc_over_ef,
            best_recall=best_recall,
            threshold=recall_threshold,
        )
        strategy = _infer_strategy(symbols, recall_margin)

        # Compact region (elite range) keyed by ANNS role names for YAML output.
        compact: Dict[str, List[float | int]] = {}
        if m_name and elite_range.get(m_name):
            compact["M"] = elite_range[m_name]
        if efc_name and elite_range.get(efc_name):
            compact["efC"] = elite_range[efc_name]
        if ef_name and elite_range.get(ef_name):
            compact["ef"] = elite_range[ef_name]

        hint = _build_hint(region_type, compact, subgroup)

        card = {
            "card_id": f"rfanns_insight_{task_slug}_{idx:03d}",
            "task_name": task_name,
            "type": region_type,
            "strategy": strategy,
            "region": {
                "description": region_description,
                "dimensions": region_dimensions,
                "coarse": coarse_range,
                "elite": elite_range,
                "compact": compact,
            },
            "feasibility": {
                "recall_threshold": recall_threshold,
                "covered_count": covered_count,
                "feasible_count": int(subgroup["feasible_count"]),
                "feasible_ratio": feasible_ratio,
                "median_recall_margin": recall_margin,
            },
            "perf": {
                "elite_ratio": float(subgroup["elite_ratio"]),
                "elite_count": int(subgroup["elite_count"]),
                "median_feasible_qps": float(subgroup["median_feasible_qps"]),
                "median_elite_qps": float(subgroup["median_elite_qps"]),
                "qps_p75_feasible": float(subgroup.get("qps_p75_feasible", 0.0)),
                "best_qps": best_qps,
                "best_recall": best_recall,
                "best_params": best_params,
            },
            "direction": {
                "feasible_median": dict(subgroup.get("feasible_param_medians") or {}),
                "elite_median": dict(subgroup.get("elite_param_medians") or {}),
                "delta": {name: float(value) for name, value in direction_delta.items()},
                "tendency": direction_tendency,
                "symbols": symbols,
            },
            "quality": {
                "score": float(subgroup["quality"]),
                "components": subgroup["quality_components"],
            },
            "hint": hint,
        }
        card["advice_template"] = _build_template_advice(card)
        card["advice"] = card["advice_template"]
        cards.append(card)
    return cards


__all__ = [
    "DEFAULT_WEIGHT_FEASIBLE_COUNT",
    "DEFAULT_WEIGHT_FEASIBLE_QPS",
    "DEFAULT_WEIGHT_RECALL_MARGIN",
    "_fmt_number",
    "_maybe_int",
    "_normalize_recall_thresholds",
    "_normalize_task_name",
    "_safe_float",
    "_slugify",
    "_threshold_slug",
    "_utc_now_iso",
    "build_insight_cards",
    "build_parameter_discretization",
    "discover_subgroups",
    "jaccard_overlap",
    "lsqm_cut_points",
    "maybe_rewrite_advice_with_llm",
    "resolve_param_role_map",
    "merge_cut_points",
    "score_subgroups",
    "select_non_redundant_subgroups",
]

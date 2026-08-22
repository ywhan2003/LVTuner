import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PARAM_ORDER = ["M", "ef_construction", "ef"]
BUILD_PARAM_ORDER = ["M", "ef_construction"]


def enable_ef_direct_mode() -> None:
    """Add *ef* to the effective build-parameter order so that dedup, BO
    modelling, and aggregation treat ef as a first-class tuned dimension
    (instead of a free-ef scan).  Idempotent — safe to call multiple times."""
    if "ef" not in BUILD_PARAM_ORDER:
        BUILD_PARAM_ORDER.append("ef")


def disable_ef_direct_mode() -> None:
    """Revert to the default scan-based ef handling (ef excluded from build keys)."""
    if "ef" in BUILD_PARAM_ORDER:
        BUILD_PARAM_ORDER.remove("ef")

OPTIONAL_METRIC_KEYS = [
    "latency_ms_p95",
    "latency_mean_us",
    "latency_999_us",
    "avg_dist_cmps",
    "build_time_s",
    "search_time_s",
    "gt_compute_time_s",
    "index_size_mb",
    "index_load_time_s",
    "original_build_time_s",
    "dist_comps_per_query",
    "visited_nodes_per_query",
    "out_degree_mean",
    "in_degree_mean",
    "in_degree_std",
    "in_degree_max",
]
DIAGNOSTIC_METRIC_KEYS = [
    "candidate_distance_stats",
    "index_reused",
    "index_path",
    "index_path_prefix",
    "result_prefix",
    "gt_file",
    "base_bin",
    "query_bin",
    "best_search_list",
    "per_search_list",
    "selected_ef",
    "selected_recall",
    "selected_qps",
]


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def params_to_key(params: Dict[str, Any], order: Sequence[str] = PARAM_ORDER) -> Tuple[Any, ...]:
    return tuple(params[name] for name in order)


def key_to_params(key: Tuple[Any, ...], order: Sequence[str] = PARAM_ORDER) -> Dict[str, Any]:
    return {order[i]: key[i] for i in range(len(order))}


def build_params_to_key(params: Dict[str, Any], order: Sequence[str] = BUILD_PARAM_ORDER) -> Tuple[Any, ...]:
    return tuple(params[name] for name in order)


def build_key_to_params(key: Tuple[Any, ...], order: Sequence[str] = BUILD_PARAM_ORDER) -> Dict[str, Any]:
    return {order[i]: key[i] for i in range(len(order))}


def parse_metrics_file(metrics_path: str | Path) -> Dict[str, Any]:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "recall" not in raw or "qps" not in raw:
        raise ValueError("Metrics JSON must include 'recall' and 'qps'.")

    metrics: Dict[str, Any] = {
        "recall": float(raw["recall"]),
        "qps": float(raw["qps"]),
    }
    for key in OPTIONAL_METRIC_KEYS:
        if key in raw and raw[key] is not None:
            metrics[key] = float(raw[key])
    for key in DIAGNOSTIC_METRIC_KEYS:
        if key in raw and raw[key] is not None:
            metrics[key] = raw[key]
    if "selection" in raw and isinstance(raw["selection"], dict):
        metrics["selection"] = raw["selection"]
    frontier = raw.get("frontier")
    if isinstance(frontier, list):
        _frontier_diag_keys = [
            "search_time_s", "dist_comps_per_query", "visited_nodes_per_query",
        ]
        metrics["frontier"] = [
            {
                "ef": int(point["ef"]),
                "recall": float(point["recall"]),
                "qps": float(point["qps"]),
                **{
                    k: (float(v) if not isinstance(v, dict) else v)
                    for k in _frontier_diag_keys
                    if point.get(k) is not None
                    for v in [point[k]]
                },
            }
            for point in frontier
            if isinstance(point, dict) and {"ef", "recall", "qps"} <= set(point)
        ]
    if "frontier_summary" in raw and isinstance(raw["frontier_summary"], dict):
        metrics["frontier_summary"] = raw["frontier_summary"]
    return metrics


def load_trials(trials_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(trials_path)
    if not path.exists():
        return []

    trials: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            # Minimal trials rows carry only params + metrics; re-derive the
            # bookkeeping fields the pipeline needs with safe defaults.
            if "stage" not in row:
                row["stage"] = "unified"
            if "repeat_idx" not in row:
                row["repeat_idx"] = 0
            if "status" not in row:
                metrics = row.get("metrics")
                row["status"] = (
                    "success"
                    if isinstance(metrics, dict) and metrics.get("recall") is not None
                    else "failed"
                )
            if "proposal_source" not in row:
                row["proposal_source"] = "unknown"
            trials.append(row)
    return trials


def append_trial(trials_path: str | Path, trial: Dict[str, Any]) -> None:
    path = Path(trials_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trial, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def aggregate_success_trials(
    trials: Iterable[Dict[str, Any]],
    order: Sequence[str] = PARAM_ORDER,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for trial in trials:
        if trial.get("status") != "success":
            continue
        params = trial.get("params", {})
        metrics = trial.get("metrics", {})
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        if "recall" not in metrics or "qps" not in metrics:
            continue

        try:
            key = params_to_key(params, order)
        except Exception:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "params": {name: params[name] for name in order},
                "stage_set": set(),
                "repeat_set": set(),
                "metrics_raw": {"recall": [], "qps": []},
            },
        )
        bucket["stage_set"].add(trial.get("stage", "unknown"))
        bucket["repeat_set"].add(int(trial.get("repeat_idx", -1)))
        bucket["metrics_raw"]["recall"].append(float(metrics["recall"]))
        bucket["metrics_raw"]["qps"].append(float(metrics["qps"]))
        for optional_key in OPTIONAL_METRIC_KEYS:
            if optional_key in metrics and metrics[optional_key] is not None:
                bucket["metrics_raw"].setdefault(optional_key, []).append(float(metrics[optional_key]))

    aggregated: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        metrics_median: Dict[str, float] = {
            "recall": float(median(bucket["metrics_raw"]["recall"])),
            "qps": float(median(bucket["metrics_raw"]["qps"])),
        }
        for optional_key in OPTIONAL_METRIC_KEYS:
            values = bucket["metrics_raw"].get(optional_key, [])
            if values:
                metrics_median[optional_key] = float(median(values))

        stages = sorted(bucket["stage_set"])
        aggregated.append(
            {
                "params": bucket["params"],
                "stage": stages[0] if stages else "unknown",
                "stages": stages,
                "success_runs": len(bucket["metrics_raw"]["recall"]),
                "repeat_indices": sorted([idx for idx in bucket["repeat_set"] if idx >= 0]),
                **metrics_median,
                "metrics": metrics_median,
            }
        )

    aggregated.sort(key=lambda item: (item["recall"], item["qps"]), reverse=True)
    return aggregated


def compute_pareto_front(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_dominated: List[Dict[str, Any]] = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            other_better_or_equal = other["recall"] >= point["recall"] and other["qps"] >= point["qps"]
            other_strictly_better = other["recall"] > point["recall"] or other["qps"] > point["qps"]
            if other_better_or_equal and other_strictly_better:
                dominated = True
                break
        if not dominated:
            non_dominated.append(point)
    return sorted(non_dominated, key=lambda item: (item["recall"], item["qps"]), reverse=True)


def compute_summary(pareto: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not pareto:
        return {"best_recall": None, "best_qps": None, "best_balanced": None}

    best_recall = max(pareto, key=lambda item: (item["recall"], item["qps"]))
    best_qps = max(pareto, key=lambda item: (item["qps"], item["recall"]))
    max_qps = max(item["qps"] for item in pareto) or 1.0
    best_balanced = max(pareto, key=lambda item: item["recall"] * (item["qps"] / max_qps))
    return {
        "best_recall": best_recall,
        "best_qps": best_qps,
        "best_balanced": best_balanced,
    }


def compute_threshold_summary(
    pareto: Sequence[Dict[str, Any]],
    recall_threshold: float,
    recall_slack: float,
) -> Dict[str, Any]:
    feasible = [item for item in pareto if float(item["recall"]) >= recall_threshold]
    within_slack = [item for item in pareto if float(item["recall"]) >= recall_threshold - recall_slack]
    return {
        "threshold_policy": {
            "mode": "hnsw_threshold_feasible_then_qps",
            "recall_threshold": float(recall_threshold),
            "recall_slack": float(recall_slack),
        },
        "best_feasible_qps": max(feasible, key=lambda item: (item["qps"], -item["recall"])) if feasible else None,
        "best_within_slack_qps": (
            max(within_slack, key=lambda item: (item["qps"], item["recall"])) if within_slack else None
        ),
    }


def compute_threshold_stage_stats(
    aggregates: Sequence[Dict[str, Any]],
    recall_threshold: float,
    recall_slack: float,
) -> Dict[str, Any]:
    feasible = [item for item in aggregates if float(item["recall"]) >= recall_threshold]
    near_feasible = [
        item
        for item in aggregates
        if recall_threshold - recall_slack <= float(item["recall"]) < recall_threshold
    ]
    return {
        "feasible_count": len(feasible),
        "near_feasible_count": len(near_feasible),
        "best_feasible_qps": (
            max(feasible, key=lambda item: (item["qps"], -item["recall"])) if feasible else None
        ),
    }


def write_json(path: str | Path, data: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


__all__ = [
    "BUILD_PARAM_ORDER",
    "PARAM_ORDER",
    "append_jsonl",
    "append_trial",
    "aggregate_success_trials",
    "build_key_to_params",
    "build_params_to_key",
    "compute_pareto_front",
    "compute_summary",
    "compute_threshold_stage_stats",
    "compute_threshold_summary",
    "disable_ef_direct_mode",
    "enable_ef_direct_mode",
    "key_to_params",
    "load_trials",
    "params_to_key",
    "parse_metrics_file",
    "utc_now_iso",
    "write_json",
]

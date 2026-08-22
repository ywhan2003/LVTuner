import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from agents.rfanns_agent import PARAM_ORDER, compute_pareto_front, params_to_key


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
    "inclusiveness_pct",
]
DIAGNOSTIC_METRIC_KEYS = [
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
    "selected_al",
    "feasible",
    "selection_mode",
    "build_params",
]


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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
    return metrics


def load_trials(trials_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(trials_path)
    if not path.exists():
        return []

    trials: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
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


def trial_success_task_keys(
    trials: Iterable[Dict[str, Any]],
    order: Sequence[str] = PARAM_ORDER,
) -> set[Tuple[Tuple[Any, ...], int]]:
    task_keys: set[Tuple[Tuple[Any, ...], int]] = set()
    for trial in trials:
        if trial.get("status") != "success":
            continue
        params = trial.get("params", {})
        repeat_idx = int(trial.get("repeat_idx", -1))
        if repeat_idx < 0:
            continue
        try:
            key = params_to_key(params, order)
        except Exception:
            continue
        task_keys.add((key, repeat_idx))
    return task_keys


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
        if not metrics or "recall" not in metrics or "qps" not in metrics:
            continue

        key = params_to_key(params, order)
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


def compute_summary(pareto: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not pareto:
        return {
            "best_recall": None,
            "best_qps": None,
            "best_balanced": None,
        }

    best_recall = max(pareto, key=lambda item: (item["recall"], item["qps"]))
    best_qps = max(pareto, key=lambda item: (item["qps"], item["recall"]))
    max_qps = max(item["qps"] for item in pareto) or 1.0
    best_balanced = max(
        pareto,
        key=lambda item: item["recall"] * (item["qps"] / max_qps),
    )

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
    threshold_policy = {
        "mode": "threshold_feasible_then_qps",
        "recall_threshold": float(recall_threshold),
        "recall_slack": float(recall_slack),
    }

    feasible = [item for item in pareto if float(item["recall"]) >= recall_threshold]
    within_slack = [item for item in pareto if float(item["recall"]) >= recall_threshold - recall_slack]

    best_feasible_qps = None
    if feasible:
        best_feasible_qps = max(feasible, key=lambda item: (item["qps"], -item["recall"]))

    best_within_slack_qps = None
    if within_slack:
        best_within_slack_qps = max(within_slack, key=lambda item: (item["qps"], item["recall"]))

    return {
        "threshold_policy": threshold_policy,
        "best_feasible_qps": best_feasible_qps,
        "best_within_slack_qps": best_within_slack_qps,
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
    best_feasible_qps = None
    if feasible:
        best_feasible_qps = max(feasible, key=lambda item: (item["qps"], -item["recall"]))
    return {
        "feasible_count": len(feasible),
        "near_feasible_count": len(near_feasible),
        "best_feasible_qps": best_feasible_qps,
    }


def write_json(path: str | Path, data: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


__all__ = [
    "append_jsonl",
    "append_trial",
    "aggregate_success_trials",
    "compute_pareto_front",
    "compute_summary",
    "compute_threshold_stage_stats",
    "compute_threshold_summary",
    "load_trials",
    "parse_metrics_file",
    "trial_success_task_keys",
    "utc_now_iso",
    "write_json",
]

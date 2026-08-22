import argparse
import copy
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency handling
    yaml = None

from agents.rfanns_agent import PARAM_ORDER, RFANNSTuningAgent, params_to_key
from configs import CommonConfig
from utils.rfanns_metrics import (
    append_jsonl,
    append_trial,
    aggregate_success_trials,
    compute_pareto_front,
    compute_summary,
    compute_threshold_stage_stats,
    compute_threshold_summary,
    load_trials,
    utc_now_iso,
    write_json,
)
from utils.rfanns_runner import RunnerConfig, build_benchmark_command, run_trial
from utils.rfanns_task_similarity import SimilarityConfig, task_similarity


HNSWLIB_PARAM_ORDER = ["M", "ef_construction", "ef"]
HNSWLIB_TRANSFER_DOMAIN = "hnswlib_tuning"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """DEPRECATED: kept only for backward compatibility in tests. Use yaml config directly."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for RFANNS tuning config parsing. "
            "Install dependency: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "search" not in cfg or "recall_threshold" not in (cfg.get("search") or {}):
        raise ValueError(f"{config_path} must explicitly set search.recall_threshold")
    return cfg


def _resolve_max_workers(value: Any) -> int:
    if value == "auto":
        return min(8, os.cpu_count() or 1)
    workers = int(value)
    if workers <= 0:
        raise ValueError("execution.max_workers must be > 0")
    return workers


def _resolve_max_rounds(value: Any, target_runs: int, batch_size: int, repeat: int) -> int:
    if value == "auto":
        per_round_runs = max(1, batch_size * max(1, repeat))
        return max(1, int(math.ceil(target_runs / per_round_runs)) + 2)
    rounds = int(value)
    if rounds <= 0:
        raise ValueError("max_rounds must be > 0 or 'auto'.")
    return rounds


def _resolve_optimizer(value: Any) -> str:
    optimizer = str(value or "agentic").strip().lower()
    if optimizer != "agentic":
        raise ValueError(
            f"search.optimizer must be 'agentic' (got '{optimizer}'; "
            "scbo_hybrid has been removed)."
        )
    return optimizer


def _resolve_param_order(cfg: Dict[str, Any]) -> List[str]:
    tuning_cfg = cfg.get("tuning") or {}
    raw_order = tuning_cfg.get("param_order", PARAM_ORDER)
    if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
        raise ValueError("tuning.param_order must be a non-empty list of parameter names.")
    order = [str(name).strip() for name in raw_order]
    if not order or any(not name for name in order):
        raise ValueError("tuning.param_order must be a non-empty list of parameter names.")
    if len(set(order)) != len(order):
        raise ValueError("tuning.param_order must not contain duplicate parameter names.")
    params_cfg = cfg.get("params") or {}
    missing = [name for name in order if name not in params_cfg]
    if missing:
        raise ValueError(f"Missing parameter space for tuning.param_order entries: {missing}")
    return order


def _count_success_runs(trials: Sequence[Dict[str, Any]]) -> int:
    success_count = 0
    for trial in trials:
        if trial.get("stage") != "unified":
            continue
        if trial.get("status") == "success":
            success_count += 1
    return success_count


def _attempted_task_keys(
    trials: Sequence[Dict[str, Any]],
    order: Sequence[str],
) -> set[Tuple[Tuple[Any, ...], int]]:
    keys: set[Tuple[Tuple[Any, ...], int]] = set()
    for trial in trials:
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        repeat_idx = int(trial.get("repeat_idx", -1))
        if repeat_idx < 0:
            continue
        try:
            keys.add((params_to_key(params, order), repeat_idx))
        except Exception:
            continue
    return keys


def _exhausted_param_keys(
    trials: Sequence[Dict[str, Any]],
    order: Sequence[str],
    repeat: int,
) -> set[Tuple[Any, ...]]:
    repeat_map: Dict[Tuple[Any, ...], set[int]] = {}
    for trial in trials:
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        repeat_idx = int(trial.get("repeat_idx", -1))
        if repeat_idx < 0:
            continue
        try:
            key = params_to_key(params, order)
        except Exception:
            continue
        repeat_map.setdefault(key, set()).add(repeat_idx)
    return {key for key, seen_repeats in repeat_map.items() if len(seen_repeats) >= repeat}


def _build_tasks(
    stage: str,
    candidates: Sequence[Dict[str, Any]],
    needed_runs: int,
    repeat: int,
    occupied_task_keys: set[Tuple[Tuple[Any, ...], int]],
    order: Sequence[str],
    round_idx: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if needed_runs <= 0:
        return tasks

    for candidate in candidates:
        params = candidate["params"]
        p_key = params_to_key(params, order)
        for repeat_idx in range(repeat):
            task_key = (p_key, repeat_idx)
            if task_key in occupied_task_keys:
                continue
            tasks.append(
                {
                    "stage": stage,
                    "params": params,
                    "repeat_idx": repeat_idx,
                    "proposal_source": candidate.get("source", "unknown"),
                    "proposal_round": round_idx,
                    "proposal_note": candidate.get("note", ""),
                }
            )
            occupied_task_keys.add(task_key)
            if len(tasks) >= needed_runs:
                return tasks
    return tasks


def _execute_tasks(
    tasks: Sequence[Dict[str, Any]],
    runner_cfg: RunnerConfig,
    trials_path: Path,
    max_workers: int,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []

    new_trials: List[Dict[str, Any]] = []
    worker_count = min(max_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                run_trial,
                task["stage"],
                task["params"],
                task["repeat_idx"],
                runner_cfg,
                {
                    "proposal_source": task.get("proposal_source", "unknown"),
                    "proposal_round": task.get("proposal_round", -1),
                    "proposal_note": task.get("proposal_note", ""),
                },
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            trial = future.result()
            task = future_map[future]
            used_names = list(task.get("used_similar_task_names", []))
            trial["used_similar_task_names"] = used_names
            trial["used_similar_task_count"] = int(task.get("used_similar_task_count", len(used_names)))
            append_trial(trials_path, trial)
            new_trials.append(trial)
    return new_trials


def _unified_report(
    all_trials: Sequence[Dict[str, Any]],
    target_runs: int,
    recall_threshold: float,
    recall_slack: float,
    order: Sequence[str],
) -> Dict[str, Any]:
    stage_trials = [t for t in all_trials if t.get("stage") == "unified"]
    success_trials = [t for t in stage_trials if t.get("status") == "success"]
    aggregates = aggregate_success_trials(stage_trials, order=order)
    pareto = compute_pareto_front(aggregates)
    report = {
        "target_runs": target_runs,
        "total_runs": len(stage_trials),
        "success_runs": len(success_trials),
        "failed_runs": len(stage_trials) - len(success_trials),
        "unique_success_params": len(aggregates),
        "pareto_count": len(pareto),
    }
    report.update(compute_threshold_stage_stats(aggregates, recall_threshold, recall_slack))
    return report


def _extract_used_similar_task_names(transfer_context: Dict[str, Any]) -> List[str]:
    if not isinstance(transfer_context, dict):
        return []
    tasks = transfer_context.get("tasks")
    if not isinstance(tasks, list):
        return []

    names: List[str] = []
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("task_name") or item.get("model_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _to_feature_row(
    params: Dict[str, Any],
    order: Sequence[str],
) -> List[float]:
    return [float(params[name]) for name in order]


def _extract_round_success_observations(
    round_trials: Sequence[Dict[str, Any]],
    order: Sequence[str],
) -> Tuple[List[List[float]], List[float]]:
    features: List[List[float]] = []
    actual_qps: List[float] = []
    for trial in round_trials:
        if trial.get("status") != "success":
            continue
        params = trial.get("params")
        metrics = trial.get("metrics")
        if not isinstance(params, dict) or not isinstance(metrics, dict) or "qps" not in metrics:
            continue
        try:
            features.append(_to_feature_row(params, order))
            actual_qps.append(float(metrics["qps"]))
        except (KeyError, TypeError, ValueError):
            continue
    return features, actual_qps


def _resolve_linked_trials_path(linked_trials_path: str, models_dir: Path) -> Path:
    linked = Path(linked_trials_path).expanduser()
    if linked.is_absolute():
        return linked.resolve()
    return (models_dir / linked).resolve()


def _load_recall_qualified_trials(
    linked_trials_path: Path,
    recall_threshold: float,
) -> List[Dict[str, Any]]:
    trials = load_trials(linked_trials_path)
    selected: List[Dict[str, Any]] = []
    for trial in trials:
        if trial.get("status") != "success":
            continue
        metrics = trial.get("metrics")
        params = trial.get("params")
        if not isinstance(metrics, dict) or not isinstance(params, dict):
            continue
        if "recall" not in metrics or "qps" not in metrics:
            continue
        try:
            recall = float(metrics["recall"])
            qps = float(metrics["qps"])
        except (TypeError, ValueError):
            continue
        if recall < recall_threshold:
            continue
        selected.append(
            {
                "params": params,
                "recall": recall,
                "qps": qps,
                "proposal_source": trial.get("proposal_source", "unknown"),
                "proposal_round": int(trial.get("proposal_round", -1)),
                "proposal_note": trial.get("proposal_note", ""),
            }
        )

    selected.sort(key=lambda item: (item["qps"], item["recall"]), reverse=True)
    return selected


def _sparse_transfer_trials(
    sorted_trials: Sequence[Dict[str, Any]],
    trials_per_task: int,
    stride: int = 10,
) -> List[Dict[str, Any]]:
    if trials_per_task <= 0 or not sorted_trials:
        return []
    stride = max(1, int(stride))

    # Keep a wider candidate window, then pick one item every `stride` points.
    window_size = min(len(sorted_trials), max(trials_per_task * stride, trials_per_task))
    window = list(sorted_trials[:window_size])

    sampled = list(window[::stride])[:trials_per_task]
    if len(sampled) >= trials_per_task:
        return sampled

    sampled_ids = {id(item) for item in sampled}
    for item in window:
        if id(item) in sampled_ids:
            continue
        sampled.append(item)
        if len(sampled) >= trials_per_task:
            break
    return sampled


def _build_transfer_context(
    round_trials: Sequence[Dict[str, Any]],
    models_dir: str | Path,
    recall_threshold: float,
    top_k_tasks: int,
    trials_per_task: int,
    param_order: Sequence[str],
    previous_context: Dict[str, Any] | None,
    similarity_cfg: SimilarityConfig | None = None,
) -> Dict[str, Any]:
    previous = previous_context or {"tasks": []}
    features, actual_qps = _extract_round_success_observations(round_trials, param_order)
    if len(actual_qps) < 2:
        return previous

    resolved_models_dir = Path(models_dir).expanduser().resolve()
    if not resolved_models_dir.exists() or not resolved_models_dir.is_dir():
        return previous

    sim_cfg = similarity_cfg or SimilarityConfig()
    scored_tasks: List[Dict[str, Any]] = []
    meta_paths = sorted(resolved_models_dir.glob("*.meta.json"))
    for meta_path in meta_paths:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if list(param_order) == HNSWLIB_PARAM_ORDER:
            if meta.get("domain") != HNSWLIB_TRANSFER_DOMAIN:
                continue
            if meta.get("param_order") != HNSWLIB_PARAM_ORDER:
                continue

        fallback_model_name = meta_path.name[:-10] if meta_path.name.endswith(".meta.json") else meta_path.stem
        model_name = str(meta.get("model_name") or fallback_model_name).strip()
        if not model_name:
            continue
        task_name = str(meta.get("task_name") or model_name).strip()

        linked_raw = meta.get("linked_trials_path")
        if not isinstance(linked_raw, str) or not linked_raw.strip():
            continue
        linked_path = _resolve_linked_trials_path(linked_raw, resolved_models_dir)
        if not linked_path.exists():
            continue

        # Sampling-invariant similarity: distill both the current task and this
        # historical task from their *own* real trials and align on a virtual
        # probe grid via GP frontier proxies. No shared sampled points, no
        # historical QPS model extrapolation, recall response included.
        try:
            hist_trials = load_trials(linked_path)
            similarity = float(
                task_similarity(
                    new_trials=round_trials,
                    hist_trials=hist_trials,
                    recall_threshold=recall_threshold,
                    cfg=sim_cfg,
                    new_name="current",
                    hist_name=task_name,
                    preferred_order=param_order,
                )["similarity"]
            )
        except Exception:
            continue

        selected_trials = _load_recall_qualified_trials(linked_path, recall_threshold)
        if not selected_trials:
            continue

        scored_tasks.append(
            {
                "model_name": model_name,
                "task_name": task_name,
                "similarity": similarity,
                "linked_trials_path": str(linked_path),
                "selected_trials": _sparse_transfer_trials(
                    sorted_trials=selected_trials,
                    trials_per_task=trials_per_task,
                    stride=10,
                ),
            }
        )

    scored_tasks.sort(
        key=lambda item: (
            item["similarity"],
            item["selected_trials"][0]["qps"] if item.get("selected_trials") else 0.0,
        ),
        reverse=True,
    )

    top_tasks = scored_tasks[:top_k_tasks]
    if not top_tasks:
        return previous

    return {
        "generated_at": utc_now_iso(),
        "source_success_count": len(actual_qps),
        "tasks": top_tasks,
    }


def _normalize_task_stem(task_name: Any) -> str:
    text = str(task_name or "").strip()
    if not text:
        return ""
    return Path(text).stem


def _resolve_similar_task_top_k(value: Any, fallback_top_k: int) -> int:
    if value is None:
        return max(1, int(fallback_top_k))
    text = str(value).strip().lower()
    if text in {"", "auto", "default"}:
        return max(1, int(fallback_top_k))
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("agentic.knowledge.similar_task_top_k must be > 0, or 'auto'.")
    return resolved


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_char_length(payload: Any) -> int:
    if isinstance(payload, str):
        return len(payload)
    return len(json.dumps(payload, ensure_ascii=False))


def _load_similar_task_insights(
    transfer_context: Dict[str, Any] | None,
    insights_dir: Path,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = transfer_context.get("tasks") if isinstance(transfer_context, dict) else []
    if not isinstance(tasks, list):
        tasks = []

    normalized_tasks: List[Dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_name = _normalize_task_stem(item.get("task_name") or item.get("model_name"))
        if not task_name:
            continue
        similarity = float(item.get("similarity", 0.0))
        normalized_tasks.append(
            {
                "task_name": task_name,
                "similarity": similarity,
            }
        )

    normalized_tasks.sort(key=lambda row: row["similarity"], reverse=True)
    selected = normalized_tasks[: max(0, int(top_k))]

    loaded: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for row in selected:
        task_name = row["task_name"]
        insight_path = insights_dir / f"{task_name}.json"
        if not insight_path.exists():
            missing.append(
                {
                    "task_name": task_name,
                    "reason": "missing_file",
                    "path": str(insight_path),
                }
            )
            continue
        payload = _load_json_object(insight_path)
        if not payload:
            missing.append(
                {
                    "task_name": task_name,
                    "reason": "invalid_or_empty_json",
                    "path": str(insight_path),
                }
            )
            continue
        loaded.append(
            {
                "task_name": task_name,
                "similarity": float(row["similarity"]),
                "insight_path": str(insight_path),
                "insight": payload,
            }
        )
    return loaded, missing


def build_knowledge_full_context(
    *,
    base_knowledge_full: Dict[str, Any] | str,
    transfer_context: Dict[str, Any] | None = None,
    insights_dir: Path | None = None,
    similar_task_top_k: int = 0,
    max_context_chars: int,
    overflow_policy: str,
    mode: str = "knowledge_base_driven",
) -> Tuple[Any, Dict[str, Any]]:
    """Build knowledge context for LLM consumption.

    Supports two modes:
    - ``knowledge_base_driven``: ``base_knowledge_full`` is a pre-formatted markdown
      string from :func:`utils.knowledge_loader.build_knowledge_context`.  Returns
      ``(str, report_dict)``.
    - ``full_context_no_rag`` (legacy): ``base_knowledge_full`` is a dict loaded from
      a compiled JSON knowledge file.  Similar-task subgroup insight cards are loaded
      from ``insights_dir`` and merged.  Returns ``(dict, report_dict)``.
    """
    if mode == "knowledge_base_driven":
        if not isinstance(base_knowledge_full, str):
            raise TypeError(
                "base_knowledge_full must be a str in knowledge_base_driven mode"
            )
        initial_chars = len(base_knowledge_full)
        report: Dict[str, Any] = {
            "mode": "knowledge_base_driven",
            "overflow_policy": overflow_policy,
            "max_context_chars": int(max_context_chars),
            "initial_chars": int(initial_chars),
            "final_chars": int(initial_chars),
            "truncated": False,
            "within_limit": initial_chars <= max_context_chars,
            "actions": [],
        }
        if initial_chars <= max_context_chars or overflow_policy != "truncate":
            return base_knowledge_full, report

        # Truncate at a markdown section boundary to keep the output readable.
        truncated = base_knowledge_full[:max_context_chars]
        last_section = truncated.rfind("\n## ")
        if last_section > max(0, max_context_chars // 2):
            truncated = truncated[:last_section] + "\n\n[... truncated ...]"
        else:
            truncated = truncated.rstrip() + "\n\n[... truncated ...]"
        final_chars = len(truncated)
        report["final_chars"] = int(final_chars)
        report["truncated"] = True
        report["within_limit"] = final_chars <= max_context_chars
        report["actions"].append(
            {
                "step": "truncate_text",
                "chars_before": int(initial_chars),
                "chars_after": int(final_chars),
            }
        )
        return truncated, report

    # --- legacy full_context_no_rag mode ---
    context = {
        "mode": "full_context_no_rag",
        "base_knowledge_full": copy.deepcopy(base_knowledge_full) if isinstance(base_knowledge_full, dict) else {},
        "similar_task_insights_full": [],
    }

    loaded_insights: List[Dict[str, Any]] = []
    missing_insights: List[Dict[str, Any]] = []
    if insights_dir is not None and transfer_context is not None and similar_task_top_k > 0:
        loaded_insights, missing_insights = _load_similar_task_insights(
            transfer_context=transfer_context,
            insights_dir=insights_dir,
            top_k=similar_task_top_k,
        )
    context["similar_task_insights_full"] = loaded_insights

    initial_chars = _json_char_length(context)
    report = {
        "mode": "full_context_no_rag",
        "overflow_policy": overflow_policy,
        "max_context_chars": int(max_context_chars),
        "initial_chars": int(initial_chars),
        "final_chars": int(initial_chars),
        "truncated": False,
        "within_limit": initial_chars <= max_context_chars,
        "actions": [],
        "missing_insight_tasks": missing_insights,
        "removed_similar_task_insights": [],
        "removed_trial_ref_fields_count": 0,
        "trimmed_base_documents": [],
    }
    if initial_chars <= max_context_chars or overflow_policy != "truncate":
        return context, report

    removed_tasks: List[str] = []
    while len(context["similar_task_insights_full"]) > 1 and _json_char_length(context) > max_context_chars:
        chars_before = _json_char_length(context)
        removed = context["similar_task_insights_full"].pop()
        chars_after = _json_char_length(context)
        removed_tasks.append(str(removed.get("task_name") or ""))
        report["actions"].append(
            {
                "step": "drop_similar_task_insight",
                "task_name": removed.get("task_name", ""),
                "similarity": float(removed.get("similarity", 0.0)),
                "chars_before": int(chars_before),
                "chars_after": int(chars_after),
            }
        )

    removed_trial_ref_fields_count = 0
    if _json_char_length(context) > max_context_chars:
        chars_before = _json_char_length(context)
        for task_entry in context["similar_task_insights_full"]:
            insight = task_entry.get("insight")
            if not isinstance(insight, dict):
                continue
            cards = insight.get("insight_cards")
            if not isinstance(cards, list):
                continue
            for card in cards:
                if not isinstance(card, dict):
                    continue
                source = card.get("source")
                if not isinstance(source, dict):
                    continue
                for key in list(source.keys()):
                    if str(key).endswith("trial_refs"):
                        source.pop(key, None)
                        removed_trial_ref_fields_count += 1
        chars_after = _json_char_length(context)
        if removed_trial_ref_fields_count > 0:
            report["actions"].append(
                {
                    "step": "drop_source_trial_refs",
                    "removed_fields": int(removed_trial_ref_fields_count),
                    "chars_before": int(chars_before),
                    "chars_after": int(chars_after),
                }
            )

    trimmed_docs: List[str] = []
    if _json_char_length(context) > max_context_chars:
        chars_before = _json_char_length(context)
        base = context.get("base_knowledge_full")
        docs = base.get("documents") if isinstance(base, dict) else None
        if isinstance(docs, list):
            for idx in reversed(range(len(docs))):
                if _json_char_length(context) <= max_context_chars:
                    break
                doc = docs[idx]
                if not isinstance(doc, dict):
                    continue
                if "content_md" not in doc:
                    continue
                doc_id = str(doc.get("doc_id") or doc.get("title") or f"documents[{idx}]")
                doc.pop("content_md", None)
                trimmed_docs.append(doc_id)
        chars_after = _json_char_length(context)
        if trimmed_docs:
            report["actions"].append(
                {
                    "step": "trim_base_documents_content_md",
                    "trimmed_docs": trimmed_docs,
                    "chars_before": int(chars_before),
                    "chars_after": int(chars_after),
                }
            )

    final_chars = _json_char_length(context)
    report["final_chars"] = int(final_chars)
    report["truncated"] = bool(final_chars < initial_chars)
    report["within_limit"] = final_chars <= max_context_chars
    report["removed_similar_task_insights"] = removed_tasks
    report["removed_trial_ref_fields_count"] = int(removed_trial_ref_fields_count)
    report["trimmed_base_documents"] = trimmed_docs
    return context, report


def _resolve_model_cfg(agentic_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_config_key = agentic_cfg.get("model_config_key", "DEEPSEEK_CONFIG")
    source = CommonConfig[model_config_key] or {}
    fallback = CommonConfig["RFANNS_AGENT_CONFIG"] or {}

    merged = dict(fallback)
    merged.update(source)

    # Keep explicit overrides available from unify_tune.yaml if provided.
    merged.update(agentic_cfg.get("model_override", {}))

    # Ensure required fields exist for OpenAI-compatible call.
    if "model_name" not in merged:
        model_name = merged.get("model") or fallback.get("model_name")
        if not model_name:
            raise ValueError(
                "No model configured — set LLM_MODEL_NAME (or RFANNS_MODEL_NAME) in .env "
                "or provide agentic.model_override in the config."
            )
        merged["model_name"] = model_name
    if "max_tokens" not in merged:
        merged["max_tokens"] = 2048
    if "temperature" not in merged:
        merged["temperature"] = 0.2
    return merged


def _resolve_prompt_cfg() -> Dict[str, str]:
    return {
        "proposer_user": CommonConfig["RFANNS_PROPOSER_PROMPT"]["user_prompt"],
        "reflector_user": CommonConfig["RFANNS_REFLECTOR_PROMPT"]["user_prompt"],
    }


def run_pipeline(
    config_path: str,
    resume: bool = True,
    dry_run: bool = False,
    agent_override: Any | None = None,
    search_space_override: Dict[str, Any] | None = None,
    seed_candidates: List[Dict[str, Any]] | None = None,
    transfer_context_override: Dict[str, Any] | None = None,
    knowledge_context_override: Any | None = None,
) -> int:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_name = str((cfg.get("output") or {}).get("trials_name", "")).strip()
    if not trials_name:
        raise ValueError("output.trials_name must be provided and non-empty.")
    if any(ch in trials_name for ch in ["/", "\\"]):
        raise ValueError("output.trials_name must be a plain file stem, not a path.")

    trials_path = output_dir / "trials" / f"{trials_name}.jsonl"
    if not resume:
        if trials_path.exists():
            backup_path = (
                output_dir
                / "trials"
                / f"{trials_name}.backup.{utc_now_iso().replace(':', '-')}.jsonl"
            )
            shutil.move(str(trials_path), str(backup_path))

    existing_trials = load_trials(trials_path) if resume else []

    repeat = int(cfg["search"]["repeat"])
    budget = int(cfg["search"]["budget"])
    optimizer = _resolve_optimizer(cfg["search"].get("optimizer", "agentic"))
    param_order = _resolve_param_order(cfg)
    recall_threshold = float(cfg["search"]["recall_threshold"])
    recall_slack = float(cfg["search"].get("recall_slack", 0.01))
    if not (0.0 < recall_threshold <= 1.0):
        raise ValueError("search.recall_threshold must be in (0, 1].")
    if not (0.0 <= recall_slack < recall_threshold):
        raise ValueError("search.recall_slack must be >= 0 and < search.recall_threshold.")

    agentic_cfg = cfg.get("agentic") or {}
    model_cfg = _resolve_model_cfg(agentic_cfg)
    prompt_cfg = _resolve_prompt_cfg()
    knowledge_cfg = agentic_cfg.get("knowledge") or {}
    transfer_cfg = agentic_cfg.get("transfer") or {}
    transfer_enabled = bool(transfer_cfg.get("enabled", True))
    transfer_models_dir = Path(transfer_cfg.get("models_dir", "results/rfanns/models")).expanduser().resolve()
    transfer_top_k_tasks = int(transfer_cfg.get("top_k_tasks", 3))
    transfer_trials_per_task = int(transfer_cfg.get("trials_per_task", 5))
    transfer_freeze_after_initial_design = bool(transfer_cfg.get("freeze_after_initial_design", False))
    transfer_similarity_cfg = SimilarityConfig.from_mapping(transfer_cfg.get("similarity"))
    if transfer_top_k_tasks <= 0:
        raise ValueError("agentic.transfer.top_k_tasks must be > 0")
    if transfer_trials_per_task <= 0:
        raise ValueError("agentic.transfer.trials_per_task must be > 0")

    knowledge_mode = str(knowledge_cfg.get("mode", "knowledge_base_driven")).strip().lower()
    if knowledge_mode not in {"full_context_no_rag", "knowledge_base_driven"}:
        raise ValueError(
            "agentic.knowledge.mode must be 'full_context_no_rag' or 'knowledge_base_driven'."
        )
    max_context_chars = int(knowledge_cfg.get("max_context_chars", 120000))
    if max_context_chars <= 0:
        raise ValueError("agentic.knowledge.max_context_chars must be > 0")
    overflow_policy = str(knowledge_cfg.get("overflow_policy", "truncate")).strip().lower()
    if overflow_policy not in {"truncate"}:
        raise ValueError("agentic.knowledge.overflow_policy must be 'truncate'.")

    if knowledge_context_override is not None:
        base_knowledge_full = knowledge_context_override
        insights_dir = Path(".")  # not used with override
        similar_task_top_k = 0  # not used with override
        knowledge_mode = "knowledge_base_driven"
    elif knowledge_mode == "knowledge_base_driven":
        # Load knowledge directly from knowledge_base/ markdown files.
        from utils.knowledge_loader import build_knowledge_context as _kb_load_context

        knowledge_base_dir = str(knowledge_cfg.get("knowledge_base_dir", "knowledge_base"))
        kb_result = _kb_load_context(base_dir=knowledge_base_dir, max_chars=max_context_chars)
        base_knowledge_full = kb_result["base_knowledge_full"]  # str: formatted markdown
        insights_dir = Path(".")  # not used in knowledge_base_driven mode
        similar_task_top_k = 0  # not used in knowledge_base_driven mode
    else:
        # Legacy full_context_no_rag mode: load from JSON + subgroup insight cards.
        insights_dir = Path(knowledge_cfg.get("insights_dir", "data/rfanns_insights")).expanduser().resolve()
        base_knowledge_path = Path(knowledge_cfg.get("path", "data/rfanns_knowledge.json")).expanduser().resolve()
        similar_task_top_k = _resolve_similar_task_top_k(
            knowledge_cfg.get("similar_task_top_k", "auto"),
            fallback_top_k=transfer_top_k_tasks,
        )
        base_knowledge_full = _load_json_object(base_knowledge_path)

    if agent_override is not None:
        agent = agent_override
    else:
        agent = RFANNSTuningAgent(
            cfg["params"],
            seed=int(cfg["search"]["seed"]),
            param_order=param_order,
            agentic_cfg=agentic_cfg,
            model_cfg=model_cfg,
            prompt_cfg=prompt_cfg,
            llm_caller=None,
            objective_preference=agentic_cfg.get("objective_preference", "pareto"),
        )
    unified_target_runs = budget

    max_workers = _resolve_max_workers(cfg["execution"]["max_workers"])
    runner_cfg = RunnerConfig(
        python_bin=cfg["benchmark"]["python_bin"],
        script_path=str(Path(cfg["benchmark"]["script_path"]).resolve()),
        metrics_output_arg=cfg["benchmark"]["metrics_output_arg"],
        timeout_s=int(cfg["execution"]["timeout_s"]),
        retries=int(cfg["execution"]["retries"]),
        output_dir=str(output_dir),
        workdir=str(Path.cwd()),
        extra_args=list(cfg["benchmark"].get("extra_args", [])),
        param_args=dict(cfg["benchmark"].get("param_args") or {}),
    )

    script_path = Path(runner_cfg.script_path)
    if not script_path.exists() and not dry_run:
        raise FileNotFoundError(f"Benchmark entry not found: {script_path}")

    batch_size = int(agentic_cfg.get("batch_size", 6))
    if batch_size <= 0:
        raise ValueError("agentic.batch_size must be > 0")

    max_rounds = _resolve_max_rounds(agentic_cfg.get("max_rounds", "auto"), unified_target_runs, batch_size, repeat)
    scbo_cfg: SCBOConfig | None = None
    scbo_optimizer: SCBOHybridOptimizer | None = None
    if optimizer == "scbo_hybrid":
        scbo_cfg = SCBOConfig.from_config(cfg.get("scbo") or {})
        recovered_state = None
        if resume:
            recovered_state = load_latest_scbo_state(
                proposal_log_path=proposal_log_path,
                init_length=scbo_cfg.trust_region_init_length,
            )
        scbo_optimizer = SCBOHybridOptimizer(
            space=agent.space,
            cfg=scbo_cfg,
            seed=int(cfg["search"]["seed"]),
            state=recovered_state,
        )

    all_trials = list(existing_trials)
    dry_plan: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "unified_tasks": [],
    }

    planned_task_keys = set(_attempted_task_keys(all_trials, agent.param_order))
    transfer_context_for_next_round: Dict[str, Any] = {"tasks": []}
    fixed_transfer_context: Dict[str, Any] | None = None
    executor_mode = "search_hsig_direct" if Path(runner_cfg.script_path).name == "search_hsig.py" else "generic"
    stage_policy = {
        "recall_threshold": recall_threshold,
        "recall_slack": recall_slack,
    }
    sysinsight_bridge_cfg = agentic_cfg.get("sysinsight_bridge") or {}
    if sysinsight_bridge_cfg.get("enabled"):
        raise ValueError(
            "agentic.sysinsight_bridge.enabled is no longer supported — the "
            "sysinsight bridge has been removed from the shared pipeline."
        )
    sysinsight_bridge = None

    if transfer_context_override is not None:
        transfer_context_for_next_round = transfer_context_override
    elif (
        transfer_enabled
        and optimizer == "scbo_hybrid"
        and transfer_freeze_after_initial_design
        and scbo_cfg is not None
    ):
        initial_target = int(scbo_cfg.initial_design_run_count)
        stage_trials_existing = [t for t in all_trials if t.get("stage") == "unified"]
        if initial_target <= 0 or len(stage_trials_existing) >= initial_target:
            initial_trials_for_similarity = (
                stage_trials_existing[:initial_target] if initial_target > 0 else stage_trials_existing
            )
            fixed_transfer_context = _build_transfer_context(
                round_trials=initial_trials_for_similarity,
                models_dir=transfer_models_dir,
                recall_threshold=recall_threshold,
                top_k_tasks=transfer_top_k_tasks,
                trials_per_task=transfer_trials_per_task,
                param_order=agent.param_order,
                previous_context={"tasks": []},
                similarity_cfg=transfer_similarity_cfg,
            )
            fixed_transfer_context["frozen"] = True
            fixed_transfer_context["freeze_reason"] = "after_initial_design"
            fixed_transfer_context["initial_design_run_count"] = initial_target
            transfer_context_for_next_round = fixed_transfer_context

    for round_idx in range(1, max_rounds + 1):
        success_total = sum(1 for t in all_trials if t.get("status") == "success")
        remaining_runs = max(0, unified_target_runs - success_total)
        if remaining_runs <= 0:
            break

        stage_trials = [t for t in all_trials if t.get("stage") == "unified"]
        initial_design_active = False
        initial_design_remaining = 0
        if optimizer == "scbo_hybrid" and scbo_cfg is not None and scbo_cfg.initial_design_run_count > 0:
            initial_design_attempts = len(stage_trials)
            initial_design_remaining = max(0, int(scbo_cfg.initial_design_run_count) - initial_design_attempts)
            initial_design_active = initial_design_remaining > 0

        if initial_design_active:
            target_unique = max(1, min(initial_design_remaining, int(math.ceil(remaining_runs / max(repeat, 1)))))
        else:
            target_unique = max(1, min(batch_size, int(math.ceil(remaining_runs / max(repeat, 1)))))

        occupied_task_keys = set(planned_task_keys)
        if not dry_run:
            occupied_task_keys = set(_attempted_task_keys(all_trials, agent.param_order))

        exhausted_param_keys = _exhausted_param_keys(all_trials, agent.param_order, repeat)
        if transfer_enabled and transfer_freeze_after_initial_design and fixed_transfer_context is not None:
            transfer_context_used = fixed_transfer_context
        elif transfer_enabled:
            transfer_context_used = transfer_context_for_next_round
        else:
            transfer_context_used = {"tasks": []}
        used_similar_task_names = _extract_used_similar_task_names(transfer_context_used)
        used_similar_task_count = len(used_similar_task_names)
        knowledge_full_context, knowledge_truncation_report = build_knowledge_full_context(
            base_knowledge_full=base_knowledge_full,
            transfer_context=transfer_context_used,
            insights_dir=insights_dir,
            similar_task_top_k=similar_task_top_k,
            max_context_chars=max_context_chars,
            overflow_policy=overflow_policy,
            mode=knowledge_mode,
        )
        rule_context_for_round: Dict[str, Any] | None = None
        sysinsight_bridge_error = ""
        if sysinsight_bridge is not None:
            try:
                rule_context_for_round = sysinsight_bridge.build_rule_context(
                    stage_trials=stage_trials,
                    round_idx=round_idx,
                )
            except Exception as exc:
                sysinsight_bridge_error = f"{exc.__class__.__name__}: {exc}"
                rule_context_for_round = None

        if optimizer == "scbo_hybrid":
            if scbo_optimizer is None:
                raise RuntimeError("SCBO optimizer is not initialized.")
            if scbo_cfg is None:
                raise RuntimeError("SCBO config is not initialized.")

            observations = extract_scbo_observations(
                trials=stage_trials,
                recall_threshold=recall_threshold,
                recall_slack=recall_slack,
                order=agent.param_order,
            )

            scbo_prepare = scbo_optimizer.prepare_round(
                observations=observations,
                batch_size=target_unique,
            )
            if initial_design_active:
                llm_seed_candidates, llm_seed_log = agent.propose_round(
                    stage="unified",
                    round_idx=round_idx,
                    target_unique=target_unique,
                    exclude_param_keys=exhausted_param_keys,
                    all_trials=all_trials,
                    stage_trials=stage_trials,
                    knowledge_full_context=knowledge_full_context,
                    stage_policy=stage_policy,
                    similar_task_context=transfer_context_used,
                    proposal_mode="llm_seed",
                    allowed_values_override=None,
                    llm_seed_source_mode="mixed",
                    scbo_reflection=scbo_prepare.get("scbo_reflection_before"),
                    rule_context=None,
                )
                candidates = list(llm_seed_candidates[:target_unique])
                scbo_round_selected_rows = []
                proposal_log = {
                    **llm_seed_log,
                    "proposal_mode": "scbo_initial_design",
                    "optimizer": optimizer,
                    "target_unique": target_unique,
                    "candidate_count": len(candidates),
                    "candidate_source": candidates[0]["source"] if candidates else "none",
                    "initial_design_run_count": int(scbo_cfg.initial_design_run_count),
                    "initial_design_remaining_before_round": int(initial_design_remaining),
                    "llm_seed_candidates": [
                        {
                            "params": row["params"],
                            "source": row.get("source", "agent"),
                            "note": row.get("note", ""),
                        }
                        for row in llm_seed_candidates
                    ],
                    "scbo_candidates": list(candidates),
                    "scbo_state_before": scbo_prepare.get("scbo_state_before", scbo_optimizer.state_dict()),
                    "trust_region_bounds": scbo_prepare.get("trust_region_bounds"),
                    "feasible_count": int(scbo_prepare.get("feasible_count", 0)),
                    "scbo_model_used": bool(scbo_prepare.get("model_used", False)),
                    "scbo_model_error": scbo_prepare.get("model_error", ""),
                    "scbo_reflection_before": scbo_prepare.get("scbo_reflection_before"),
                    "scbo_llm_rescored": [],
                    "scbo_ts_candidates": [],
                    "scbo_merged_pool": [],
                    "scbo_selected_candidate_rows": scbo_round_selected_rows,
                    "scbo_violation_filter_threshold": float(scbo_cfg.violation_filter_min_joint_feasible_prob),
                    "llm_allowed_values_subset": None,
                    "llm_tr_expansion_steps": 0,
                    "llm_allowed_combo_count": 0,
                    "llm_seed_source_mode_requested": "mixed",
                    "llm_seed_source_mode_effective": llm_seed_log.get("llm_seed_source_mode", "proposer_only"),
                    "llm_seed_source_mode": llm_seed_log.get("llm_seed_source_mode", "proposer_only"),
                }
            else:
                llm_seed_target = max(target_unique, scbo_cfg.llm_candidate_count)
                llm_allowed_values_info = scbo_optimizer.build_llm_seed_allowed_values(
                    observations=observations,
                    target_seed_count=llm_seed_target,
                    force_constrain_to_tr=True,
                    disable_auto_expand=True,
                )
                llm_allowed_values_override = llm_allowed_values_info["allowed_values"]
                llm_seed_source_mode = "mixed"
                if scbo_cfg.llm_seed_refine_proposer_only:
                    llm_seed_source_mode = "proposer_only"

                llm_seed_candidates, llm_seed_log = agent.propose_round(
                    stage="unified",
                    round_idx=round_idx,
                    target_unique=llm_seed_target,
                    exclude_param_keys=exhausted_param_keys,
                    all_trials=all_trials,
                    stage_trials=stage_trials,
                    knowledge_full_context=knowledge_full_context,
                    stage_policy=stage_policy,
                    similar_task_context=transfer_context_used,
                    proposal_mode="llm_seed",
                    allowed_values_override=llm_allowed_values_override,
                    llm_seed_source_mode=llm_seed_source_mode,
                    scbo_reflection=scbo_prepare.get("scbo_reflection_before"),
                    rule_context=None,
                    force_llm=True,
                )
                scbo_result = scbo_optimizer.rescore_merge_select(
                    observations=observations,
                    llm_candidates=llm_seed_candidates,
                    excluded_param_keys=exhausted_param_keys,
                    batch_size=target_unique,
                    prepared_round=scbo_prepare,
                )
                candidates = list(scbo_result["candidates"])
                scbo_round_selected_rows = list(scbo_result.get("selected_candidate_rows", []))
                proposal_log = {
                    **llm_seed_log,
                    "proposal_mode": "scbo_hybrid",
                    "optimizer": optimizer,
                    "target_unique": target_unique,
                    "candidate_count": len(candidates),
                    "candidate_source": candidates[0]["source"] if candidates else "none",
                    "llm_seed_candidates": [
                        {
                            "params": row["params"],
                            "source": row.get("source", "agent"),
                            "note": row.get("note", ""),
                        }
                        for row in llm_seed_candidates
                    ],
                    "scbo_candidates": list(scbo_result.get("scbo_candidates", candidates)),
                    "scbo_state_before": scbo_result.get("scbo_state_before", scbo_optimizer.state_dict()),
                    "trust_region_bounds": scbo_result.get("trust_region_bounds"),
                    "feasible_count": int(scbo_result.get("feasible_count", 0)),
                    "scbo_model_used": bool(scbo_result.get("model_used", False)),
                    "scbo_model_error": scbo_result.get("model_error", ""),
                    "scbo_reflection_before": scbo_result.get("scbo_reflection_before", scbo_prepare.get("scbo_reflection_before")),
                    "scbo_llm_rescored": scbo_result.get("scbo_llm_rescored", []),
                    "scbo_ts_candidates": scbo_result.get("scbo_ts_candidates", []),
                    "scbo_merged_pool": scbo_result.get("scbo_merged_pool", []),
                    "scbo_selected_candidate_rows": scbo_round_selected_rows,
                    "scbo_violation_filter_threshold": float(scbo_cfg.violation_filter_min_joint_feasible_prob),
                    "llm_allowed_values_subset": llm_allowed_values_info.get("allowed_values"),
                    "llm_tr_expansion_steps": int(llm_allowed_values_info.get("expansion_steps", 0)),
                    "llm_allowed_combo_count": int(llm_allowed_values_info.get("combo_count", 0)),
                    "llm_seed_source_mode_requested": llm_seed_source_mode,
                    "llm_seed_source_mode_effective": llm_seed_log.get("llm_seed_source_mode", "proposer_only"),
                    "llm_seed_source_mode": llm_seed_log.get("llm_seed_source_mode", "proposer_only"),
                }
        else:
            scbo_round_selected_rows = []
            candidates, proposal_log = agent.propose_round(
                stage="unified",
                round_idx=round_idx,
                target_unique=target_unique,
                exclude_param_keys=exhausted_param_keys,
                all_trials=all_trials,
                stage_trials=stage_trials,
                knowledge_full_context=knowledge_full_context,
                stage_policy=stage_policy,
                similar_task_context=transfer_context_used,
                rule_context=rule_context_for_round,
                force_llm=bool(round_idx > 1),
            )

        tasks = _build_tasks(
            stage="unified",
            candidates=candidates,
            needed_runs=remaining_runs,
            repeat=repeat,
            occupied_task_keys=occupied_task_keys,
            order=agent.param_order,
            round_idx=round_idx,
        )
        for task in tasks:
            task["used_similar_task_names"] = list(used_similar_task_names)
            task["used_similar_task_count"] = used_similar_task_count

        proposal_log["generated_at"] = utc_now_iso()
        proposal_log["planned_run_count"] = len(tasks)
        proposal_log["executor_script"] = runner_cfg.script_path
        proposal_log["executor_mode"] = executor_mode
        proposal_log["benchmark_artifacts"] = []
        proposal_log["optimizer"] = optimizer
        proposal_log["transfer_enabled"] = transfer_enabled
        proposal_log["transfer_context_used"] = transfer_context_used
        proposal_log["knowledge_mode"] = knowledge_mode
        proposal_log["knowledge_truncation_report"] = knowledge_truncation_report
        proposal_log["sysinsight_bridge_enabled"] = bool(sysinsight_bridge is not None)
        proposal_log["sysinsight_bridge_error"] = sysinsight_bridge_error
        if rule_context_for_round is not None:
            proposal_log.setdefault("diagnosis_tags", list(rule_context_for_round.get("diagnosis_tags") or []))
            proposal_log.setdefault("retrieved_rule_ids", list(rule_context_for_round.get("retrieved_rule_ids") or []))
            proposal_log.setdefault("rerank_applied", False)
            proposal_log.setdefault("rule_score_components", [])
            proposal_log.setdefault("pre_rerank_rank", {})
            proposal_log.setdefault("post_rerank_rank", {})
            proposal_log["rule_context_reference_params"] = rule_context_for_round.get("reference_params")
            proposal_log["rule_context_phase"] = rule_context_for_round.get("phase")

        if not tasks:
            if optimizer == "scbo_hybrid" and scbo_optimizer is not None:
                proposal_log["scbo_state_after"] = scbo_optimizer.state_dict()
                proposal_log["scbo_reflection_after"] = proposal_log.get("scbo_reflection_before")
            proposal_log["round_status"] = "no_tasks"
            pass  # proposal logs no longer stored
            break

        if dry_run:
            dry_round_tasks = []
            for task in tasks:
                command = build_benchmark_command(
                    runner_cfg.python_bin,
                    runner_cfg.script_path,
                    task["params"],
                    str(output_dir / "dry_run_metrics.json"),
                    runner_cfg.metrics_output_arg,
                    runner_cfg.extra_args,
                    runner_cfg.param_args,
                )
                row = {**task, "command": command}
                dry_round_tasks.append(row)
                planned_task_keys.add((params_to_key(task["params"], agent.param_order), task["repeat_idx"]))

                all_trials.append(
                    {
                        "run_id": f"dryrun-unified-{round_idx}-{task['repeat_idx']}-{params_to_key(task['params'], agent.param_order)}",
                        "stage": "unified",
                        "params": task["params"],
                        "metrics": None,
                        "repeat_idx": task["repeat_idx"],
                        "status": "planned",
                        "error": None,
                        "started_at": utc_now_iso(),
                        "elapsed_s": 0.0,
                        "proposal_source": task.get("proposal_source", "unknown"),
                        "proposal_round": task.get("proposal_round", -1),
                        "proposal_note": task.get("proposal_note", ""),
                        "used_similar_task_names": list(task.get("used_similar_task_names", [])),
                        "used_similar_task_count": int(task.get("used_similar_task_count", 0)),
                    }
                )

            dry_plan["unified_tasks"].extend(dry_round_tasks)
            round_trials = [
                {
                    "stage": task["stage"],
                    "params": task["params"],
                    "metrics": None,
                    "repeat_idx": task["repeat_idx"],
                    "status": "planned",
                    "proposal_source": task.get("proposal_source", "unknown"),
                    "proposal_round": task.get("proposal_round", -1),
                    "proposal_note": task.get("proposal_note", ""),
                    "used_similar_task_names": list(task.get("used_similar_task_names", [])),
                    "used_similar_task_count": int(task.get("used_similar_task_count", 0)),
                }
                for task in tasks
            ]
        else:
            round_trials = _execute_tasks(tasks, runner_cfg, trials_path, max_workers=max_workers)
            all_trials.extend(round_trials)
            proposal_log["benchmark_artifacts"] = [
                trial["benchmark_artifacts"]
                for trial in round_trials
                if isinstance(trial.get("benchmark_artifacts"), dict) and trial.get("benchmark_artifacts")
            ]

        reflection = agent.reflect_round(
            stage="unified",
            round_idx=round_idx,
            round_trials=round_trials,
            stage_trials=[t for t in all_trials if t.get("stage") == "unified"],
            stage_policy=stage_policy,
            selection_context={
                "selection_mode": proposal_log.get("selection_mode", "threshold_guided"),
                "cold_start_diversity_mode": bool(proposal_log.get("cold_start_diversity_mode", False)),
                "cold_start_diversity_reason": proposal_log.get(
                    "cold_start_diversity_reason",
                    "not_provided",
                ),
            },
        )

        if transfer_enabled:
            if optimizer == "scbo_hybrid" and transfer_freeze_after_initial_design:
                initial_target = int(scbo_cfg.initial_design_run_count) if scbo_cfg is not None else 0
                stage_trials_after_round = [t for t in all_trials if t.get("stage") == "unified"]
                initial_design_complete = initial_target <= 0 or len(stage_trials_after_round) >= initial_target
                if fixed_transfer_context is None and initial_design_complete:
                    initial_trials_for_similarity = (
                        stage_trials_after_round[:initial_target]
                        if initial_target > 0
                        else stage_trials_after_round
                    )
                    fixed_transfer_context = _build_transfer_context(
                        round_trials=initial_trials_for_similarity,
                        models_dir=transfer_models_dir,
                        recall_threshold=recall_threshold,
                        top_k_tasks=transfer_top_k_tasks,
                        trials_per_task=transfer_trials_per_task,
                        param_order=agent.param_order,
                        previous_context={"tasks": []},
                        similarity_cfg=transfer_similarity_cfg,
                    )
                    fixed_transfer_context["frozen"] = True
                    fixed_transfer_context["freeze_reason"] = "after_initial_design"
                    fixed_transfer_context["initial_design_run_count"] = initial_target
                transfer_context_for_next_round = fixed_transfer_context or {"tasks": []}
            else:
                transfer_context_for_next_round = _build_transfer_context(
                    round_trials=round_trials,
                    models_dir=transfer_models_dir,
                    recall_threshold=recall_threshold,
                    top_k_tasks=transfer_top_k_tasks,
                    trials_per_task=transfer_trials_per_task,
                    param_order=agent.param_order,
                    previous_context=transfer_context_for_next_round,
                    similarity_cfg=transfer_similarity_cfg,
                )
        else:
            transfer_context_for_next_round = {"tasks": []}

        if optimizer == "scbo_hybrid" and scbo_optimizer is not None:
            if not dry_run:
                all_observations = extract_scbo_observations(
                    trials=[t for t in all_trials if t.get("stage") == "unified"],
                    recall_threshold=recall_threshold,
                    recall_slack=recall_slack,
                    order=agent.param_order,
                )
                proposal_log["scbo_state_after"] = scbo_optimizer.update_state(observations_all=all_observations)
                if scbo_cfg is not None:
                    surrogate_ready = len(all_observations) >= int(scbo_cfg.warm_start_min_train_points)
                    surrogate_error = "" if surrogate_ready else "cold_start_insufficient_observations"
                else:
                    surrogate_ready = False
                    surrogate_error = "scbo_config_missing"
                proposal_log["scbo_reflection_after"] = scbo_optimizer.build_structured_reflection(
                    all_observations,
                    candidate_rows=scbo_round_selected_rows,
                    phase="after",
                    surrogate_ready=surrogate_ready,
                    surrogate_error=surrogate_error,
                )
            else:
                proposal_log["scbo_state_after"] = scbo_optimizer.state_dict()
                proposal_log["scbo_reflection_after"] = proposal_log.get("scbo_reflection_before")

        if sysinsight_bridge is not None and not dry_run:
            try:
                sysinsight_bridge.update_after_round(
                    round_idx=round_idx,
                    round_trials=round_trials,
                    stage_trials_before_round=stage_trials,
                    stage_trials_after_round=[t for t in all_trials if t.get("stage") == "unified"],
                    rule_context=rule_context_for_round,
                    proposal_log=proposal_log,
                )
            except Exception as exc:
                error_post = f"{exc.__class__.__name__}: {exc}"
                if proposal_log.get("sysinsight_bridge_error"):
                    proposal_log["sysinsight_bridge_error"] = (
                        f"{proposal_log.get('sysinsight_bridge_error')} | post_update: {error_post}"
                    )
                else:
                    proposal_log["sysinsight_bridge_error"] = f"post_update: {error_post}"

        proposal_log["round_status"] = "executed" if not dry_run else "planned"
        proposal_log["reflection"] = reflection.get("summary", {})
        if fixed_transfer_context is not None:
            proposal_log["transfer_context_initial"] = fixed_transfer_context
        proposal_log["transfer_context_next"] = transfer_context_for_next_round
        pass  # proposal logs no longer stored

    if dry_run:
        write_json(output_dir / "dry_run_plan.json", dry_plan)
        return 0

    all_trials = load_trials(trials_path)
    aggregates = aggregate_success_trials(all_trials, order=agent.param_order)
    pareto = compute_pareto_front(aggregates)
    summary = compute_summary(pareto)
    summary.update(compute_threshold_summary(pareto, recall_threshold, recall_slack))
    final_success_runs = _count_success_runs(all_trials)
    threshold_mode = (
        "scbo_threshold_then_qps"
        if optimizer == "scbo_hybrid"
        else "single_stage_transfer_threshold_then_qps"
    )
    stage_report = {
        "generated_at": utc_now_iso(),
        "task_name": trials_name,
        "trials_path": str(trials_path),
        "optimizer": optimizer,
        "threshold_policy": {
            "mode": threshold_mode,
            "recall_threshold": recall_threshold,
            "recall_slack": recall_slack,
        },
        "param_order": list(agent.param_order),
        "budget": {
            "total_runs": budget,
            "unified_target_runs": unified_target_runs,
            "unified_remaining": max(0, unified_target_runs - final_success_runs),
        },
        "unified": _unified_report(
            all_trials,
            unified_target_runs,
            recall_threshold,
            recall_slack,
            order=agent.param_order,
        ),
    }

    write_json(output_dir / "pareto.json", pareto)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "stage_report.json", stage_report)

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RFANNS single-stage tuning pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unify_tune.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing trials/<trials_name>.jsonl (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from existing trials. Existing task-scoped trials/proposal logs will be backed up.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate candidates and commands only; do not execute benchmark.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_pipeline(
        config_path=args.config,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

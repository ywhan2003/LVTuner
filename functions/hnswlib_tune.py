import argparse
import copy
import json
import logging
import math
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from agents.hnswlib_agent import HNSWLIBTuningAgent, PARAM_ORDER, params_to_key
from configs import CommonConfig
from utils.hnswlib_metrics import (
    BUILD_PARAM_ORDER,
    PARAM_ORDER as METRICS_PARAM_ORDER,
    append_jsonl,
    append_trial,
    aggregate_success_trials,
    build_params_to_key,
    compute_pareto_front,
    compute_summary,
    compute_threshold_stage_stats,
    compute_threshold_summary,
    enable_ef_direct_mode,
    load_trials,
    params_to_key,
    utc_now_iso,
    write_json,
)
from utils.hnswlib_runner import RunnerConfig, build_benchmark_command, run_trial
from utils.current_task_memory import (
    CurrentTaskMemory,
    build_current_task_memory_prompt,
)
from utils.posterior_proposal_checker import (
    DECISION_TOO_CONSERVATIVE,
    DECISION_TOO_WEAK,
    check_raw_proposal,
    format_check_result_for_llm,
    hard_reject,
)
from conditional_policy.runtime import (
    RuntimeConfig as ConditionalPolicyRuntimeConfig,
    build_history_arrays as build_conditional_policy_history,
    build_runtime_symptom as build_conditional_policy_symptom,
    format_policy_context as format_conditional_policy_context,
    load_policy_book as load_conditional_policy_book,
    match_policies as match_conditional_policies,
)
from utils.subgroup_insights import (
    build_insight_cards,
    build_parameter_discretization,
    discover_subgroups,
    score_subgroups,
    select_non_redundant_subgroups,
)


HNSWLIB_TRANSFER_DOMAIN = "hnswlib_tuning"

# Conservative per-parameter default spans used when no config-provided params are
# available (e.g. in offline insight-card scoring helpers).
_PARAM_DEFAULT_SPANS: Dict[str, float] = {
    "M": 56.0,               # 64 - 8
    "ef_construction": 720.0, # 820 - 100
    "ef": 105.0,              # 120 - 15
}


def _build_key_from_params(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return params_to_key(params, METRICS_PARAM_ORDER)


def _build_only_params(params: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(params, dict):
        return None
    build = {name: params[name] for name in BUILD_PARAM_ORDER if name in params}
    return build or None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """DEPRECATED: kept only for backward compatibility in tests. Use yaml config directly."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _perturb_duplicate(
    agent: "HNSWLIBTuningAgent",
    params: Dict[str, Any],
    exhausted_keys: set,
    allowed_values_override: Dict[str, Any] | None = None,
    max_attempts: int = 12,
) -> Dict[str, Any] | None:
    """Generate a small perturbation of *params* to avoid a duplicate key.

    Tries small adjustments to ef first (no rebuild), then M and efC.
    Returns the first valid, non-exhausted config, or None.
    """
    import itertools as _it

    base = dict(params)
    # Perturbation deltas: try ef first (cheap, no rebuild), then other knobs
    deltas = [
        {"ef": -2}, {"ef": +2}, {"ef": -4}, {"ef": +4},
        {"ef": -6}, {"ef": +6},
        {"M": +1}, {"M": -1}, {"M": +2},
        {"ef_construction": +10}, {"ef_construction": -10},
        {"M": +1, "ef_construction": +10},
    ]

    for delta in deltas[:max_attempts]:
        candidate = dict(base)
        for k, dv in delta.items():
            candidate[k] = candidate.get(k, 0) + dv
        try:
            canonical = agent.canonicalize(candidate)
            key = params_to_key(canonical, METRICS_PARAM_ORDER)
            if key not in exhausted_keys:
                return canonical
        except Exception:
            continue

    return None


def _nonbanned_fallback_candidates(
    agent: "HNSWLIBTuningAgent",
    target_unique: int,
    exhausted_param_keys: set,
    *,
    stage_policy: Dict[str, Any] | None,
    allowed_values_override: Dict[str, Any] | None,
    excluded_construction_pairs: set,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Initial-design fallback candidates with banned construction pairs removed.

    Falls back to the unfiltered list (with a warning) only when every
    candidate would use a banned pair.
    """
    fallback = agent.initial_design_candidates(
        target_unique,
        exhausted_param_keys,
        stage_policy=stage_policy,
        allowed_values_override=allowed_values_override,
    )
    if excluded_construction_pairs:
        allowed_fb = [
            c for c in fallback
            if isinstance(c, dict) and isinstance(c.get("params"), dict)
            and (int(c["params"].get("M", -1)), int(c["params"].get("ef_construction", -1)))
            not in excluded_construction_pairs
        ]
        if allowed_fb:
            fallback = allowed_fb
        else:
            logger.warning(
                "Fallback: no non-banned candidate available, using original fallback"
            )
    return list(fallback[:target_unique])


def load_config(config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for HNSW tuning config parsing. Install dependency: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "search" not in cfg or "recall_threshold" not in (cfg.get("search") or {}):
        raise ValueError(f"{config_path} must explicitly set search.recall_threshold")
    _resolve_param_order(cfg)
    return cfg


def _resolve_param_order(cfg: Dict[str, Any]) -> List[str]:
    tuning_cfg = cfg.get("tuning") or {}
    raw_order = tuning_cfg.get("param_order", PARAM_ORDER)
    order = [str(name).strip() for name in raw_order]
    if order != PARAM_ORDER:
        raise ValueError(f"HNSW tuning.param_order must be exactly {PARAM_ORDER}.")
    params_cfg = cfg.get("params") or {}
    missing = [name for name in PARAM_ORDER if name not in params_cfg]
    if missing:
        raise ValueError(f"Missing HNSW parameter space entries: {missing}")
    return list(PARAM_ORDER)


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
        raise ValueError("agentic.max_rounds must be > 0 or 'auto'.")
    return rounds


def _range_spec_values(spec: Dict[str, Any], *, max_points: int = 32) -> List[int]:
    min_value = int(spec["min"])
    max_value = int(spec["max"])
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    values = list(range(min_value, max_value + 1))
    if len(values) <= max_points:
        return values
    stride = max(1, int(math.ceil(len(values) / max_points)))
    sampled = values[::stride]
    if sampled[-1] != max_value:
        sampled.append(max_value)
    return sorted(set(sampled))


def _configured_ef_scan_values(cfg: Dict[str, Any]) -> List[int]:
    params_cfg = cfg.get("params") or {}
    ef_spec = params_cfg.get("ef") or {}
    values: List[int] = []
    if isinstance(ef_spec, dict) and isinstance(ef_spec.get("values"), list):
        values = sorted({int(value) for value in ef_spec.get("values", [])})
    elif isinstance(ef_spec, dict) and ef_spec.get("min") is not None and ef_spec.get("max") is not None:
        values = _range_spec_values(ef_spec)
    elif isinstance(ef_spec, list):
        values = sorted({int(value) for value in ef_spec})
    if not values:
        values = [24, 32, 40, 48, 64, 80, 96, 120]
    return values


def _resolve_model_cfg(agentic_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_config_key = agentic_cfg.get("model_config_key", "DEEPSEEK_CONFIG")
    source = CommonConfig[model_config_key] or {}
    fallback = CommonConfig["HNSWLIB_AGENT_CONFIG"] or {}
    merged = dict(fallback)
    merged.update(source)
    merged.update(agentic_cfg.get("model_override", {}))
    if "model_name" not in merged:
        model_name = merged.get("model") or fallback.get("model_name")
        if not model_name:
            raise ValueError(
                "No model configured — set LLM_MODEL_NAME (or HNSWLIB_MODEL_NAME) in .env "
                "or provide agentic.model_override in the config."
            )
        merged["model_name"] = model_name
    if "max_tokens" not in merged:
        merged["max_tokens"] = 2048
    if "temperature" not in merged:
        merged["temperature"] = 0.2
    return merged


def _param_constraint_from_values(
    agent: HNSWLIBTuningAgent,
    name: str,
    values: Sequence[Any],
    *,
    pad_low: int,
    pad_high: int,
) -> Dict[str, Any] | None:
    if not values:
        return None
    domain = agent.space.domains[name]
    if domain.kind == "discrete":
        all_values = list(domain.values or [])
        index_map = {value: idx for idx, value in enumerate(all_values)}
        indices = [index_map[value] for value in values if value in index_map]
        if not indices:
            return None
        lo = max(0, min(indices) - max(0, int(pad_low)))
        hi = min(len(all_values) - 1, max(indices) + max(0, int(pad_high)))
        return {"kind": "discrete", "values": all_values[lo : hi + 1]}

    lo_value = min(float(value) for value in values) - max(0, int(pad_low))
    hi_value = max(float(value) for value in values) + max(0, int(pad_high))
    min_bound = float(domain.min_value)  # type: ignore[arg-type]
    max_bound = float(domain.max_value)  # type: ignore[arg-type]
    lo_value = max(min_bound, lo_value)
    hi_value = min(max_bound, hi_value)
    if domain.is_integer:
        return {
            "kind": "range",
            "min": int(round(lo_value)),
            "max": int(round(hi_value)),
            "integer": True,
        }
    return {
        "kind": "range",
        "min": round(lo_value, 12),
        "max": round(hi_value, 12),
        "integer": False,
    }


def _full_domain_constraint(agent: HNSWLIBTuningAgent, name: str) -> Dict[str, Any]:
    domain = agent.space.domains[name]
    if domain.kind == "discrete":
        return {"kind": "discrete", "values": list(domain.values or [])}
    return {
        "kind": "range",
        "min": domain.min_value,
        "max": domain.max_value,
        "integer": bool(domain.is_integer),
    }


def _search_space_refinement_cfg(agent: HNSWLIBTuningAgent) -> Dict[str, Any]:
    agentic_cfg = agent.agentic_cfg if isinstance(agent.agentic_cfg, dict) else {}
    refinement_cfg = agentic_cfg.get("search_space_refinement") or {}
    return refinement_cfg if isinstance(refinement_cfg, dict) else {}


def _search_space_refinement_enabled(agent: HNSWLIBTuningAgent) -> bool:
    refinement_cfg = _search_space_refinement_cfg(agent)
    return bool(refinement_cfg.get("enabled", True))


def _llm_refinement_enabled(agent: HNSWLIBTuningAgent) -> bool:
    refinement_cfg = _search_space_refinement_cfg(agent)
    return _search_space_refinement_enabled(agent) and bool(refinement_cfg.get("llm_enabled", True))


def _subgroup_confidence_bucket(
    *,
    covered_count: int,
    feasible_ratio: float,
    recall_margin: float,
) -> str:
    if feasible_ratio >= 0.75 and recall_margin >= 0.0 and covered_count >= 8:
        return "high"
    if feasible_ratio >= 0.55 and covered_count >= 4:
        return "medium"
    return "low"


def _directional_margin_pad(name: str, recall_margin: float) -> int:
    positive_margin = max(0.0, float(recall_margin))
    if positive_margin <= 0.0:
        return 0
    scale_map = {
        "M": 40.0,
        "ef_construction": 120.0,
        "ef": 250.0,
    }
    cap_map = {
        "M": 4,
        "ef_construction": 18,
        "ef": 14,
    }
    scale = scale_map.get(name, 100.0)
    cap = cap_map.get(name, 8)
    return max(0, min(cap, int(math.ceil(positive_margin * scale))))


def _focus_padding(name: str, bucket: str, direction_delta: float, recall_margin: float) -> Tuple[int, int]:
    base_map = {
        "high": {
            "M": (0, 1),
            "ef_construction": (2, 6),
            "ef": (0, 3),
        },
        "medium": {
            "M": (1, 1),
            "ef_construction": (4, 10),
            "ef": (1, 4),
        },
        "low": {
            "M": (1, 2),
            "ef_construction": (6, 14),
            "ef": (1, 6),
        },
    }
    low_pad, high_pad = base_map.get(bucket, base_map["medium"]).get(name, (1, 1))
    directional_pad = _directional_margin_pad(name, recall_margin)
    if direction_delta > 1e-9:
        high_pad += 1 + directional_pad
    elif direction_delta < -1e-9:
        low_pad += 1 + directional_pad
    return low_pad, high_pad


def _focus_params_from_card(
    card: Dict[str, Any],
    param_order: Sequence[str] = PARAM_ORDER,
) -> Dict[str, Any]:
    perf = card.get("perf") or {}
    best_params = perf.get("best_params") or {}
    if isinstance(best_params, dict) and best_params:
        return {
            name: best_params[name]
            for name in param_order
            if name in best_params
        }
    direction = card.get("direction") or {}
    elite = direction.get("elite_median") or {}
    feasible = direction.get("feasible_median") or {}
    focus: Dict[str, Any] = {}
    # Prefer the elite region (top-QPS configurations): the feasible median
    # is the center of the whole feasible subgroup and can sit far inside
    # the feasible region (very high recall, low QPS).  feasible_median is
    # only a fallback when the elite median is missing.
    for name in param_order:
        if name in elite:
            focus[name] = elite[name]
        elif name in feasible:
            focus[name] = feasible[name]
    # Clamp ef to efC (scan-mode data may have ef > efC)
    if "ef" in focus and "ef_construction" in focus:
        focus["ef"] = min(focus["ef"], focus["ef_construction"])
    return focus


def _focused_constraint_from_profile(
    agent: HNSWLIBTuningAgent,
    *,
    name: str,
    focus_values: Sequence[Any],
    support_values: Sequence[Any],
    feasible_ratio: float,
    covered_count: int,
    recall_margin: float,
    direction_delta: float,
) -> Dict[str, Any] | None:
    bucket = _subgroup_confidence_bucket(
        covered_count=covered_count,
        feasible_ratio=feasible_ratio,
        recall_margin=recall_margin,
    )
    low_pad, high_pad = _focus_padding(name, bucket, direction_delta, recall_margin)
    focused = _param_constraint_from_values(agent, name, focus_values or support_values, pad_low=low_pad, pad_high=high_pad)
    directional_pad = _directional_margin_pad(name, recall_margin)
    support = _param_constraint_from_values(
        agent,
        name,
        support_values,
        pad_low=directional_pad if direction_delta < -1e-9 else 0,
        pad_high=directional_pad if direction_delta > 1e-9 else 0,
    )
    if focused is None:
        return support
    if support is None:
        return focused
    return _intersect_constraints(agent, name, focused, support) or support or focused


def _tokenize_match_text(value: str) -> List[str]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    stop_words = {
        "hnswlib",
        "tuning",
        "dynamic",
        "fresh",
        "trial",
        "trials",
        "recall",
        "qps",
        "data",
        "dataset",
        "index",
        "cache",
        "path",
        "hdf5",
    }
    return [token for token in tokens if token not in stop_words and not token.isdigit()]


def _task_match_score(task_name: str, hint_tokens: Sequence[str]) -> float:
    task_tokens = set(_tokenize_match_text(task_name))
    if not task_tokens or not hint_tokens:
        return 0.0
    hint_set = set(hint_tokens)
    overlap = task_tokens & hint_set
    score = len(overlap) / max(1, len(task_tokens | hint_set))
    if overlap and any(token in task_name.lower() for token in hint_set):
        score += 0.15
    return float(min(1.0, score))


def _constraint_values(agent: HNSWLIBTuningAgent, name: str, constraint: Dict[str, Any] | None) -> List[Any]:
    domain = agent.space.domains[name]
    resolved = constraint or _full_domain_constraint(agent, name)
    if domain.kind == "discrete":
        allowed = resolved.get("values")
        if not isinstance(allowed, list):
            return list(domain.values or [])
        available = set(domain.values or [])
        return [value for value in allowed if value in available]

    min_bound = float(domain.min_value)  # type: ignore[arg-type]
    max_bound = float(domain.max_value)  # type: ignore[arg-type]
    low = max(min_bound, float(resolved.get("min", min_bound)))
    high = min(max_bound, float(resolved.get("max", max_bound)))
    if high < low:
        return []
    if domain.is_integer:
        low_i = int(math.ceil(low))
        high_i = int(math.floor(high))
        if high_i < low_i:
            return []
        return list(range(low_i, high_i + 1))
    return [round(low, 12), round(high, 12)]


def _constraint_width(agent: HNSWLIBTuningAgent, name: str, constraint: Dict[str, Any] | None) -> float:
    values = _constraint_values(agent, name, constraint)
    if not values:
        return 0.0
    domain = agent.space.domains[name]
    if domain.kind == "discrete":
        full_count = max(1, len(domain.values or []))
        return len(values) / full_count
    min_bound = float(domain.min_value)  # type: ignore[arg-type]
    max_bound = float(domain.max_value)  # type: ignore[arg-type]
    span = max(1e-9, max_bound - min_bound)
    return max(0.0, min(1.0, (max(float(v) for v in values) - min(float(v) for v in values)) / span))


def _override_volume(agent: HNSWLIBTuningAgent, override: Dict[str, Any] | None) -> float:
    if not override:
        return 1.0
    volume = 1.0
    for name in PARAM_ORDER:
        volume *= max(1e-9, _constraint_width(agent, name, override.get(name)))
    return float(volume)


def _domain_values_in_interval(
    agent: HNSWLIBTuningAgent,
    name: str,
    interval: Dict[str, Any],
) -> List[Any]:
    domain = agent.space.domains[name]
    if domain.kind == "discrete":
        values = list(domain.values or [])
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        lower_ok = interval.get("lower_inclusive", True)
        upper_ok = interval.get("upper_inclusive", True)
        selected: List[Any] = []
        for value in values:
            parsed = float(value)
            if parsed < lower or (parsed == lower and not lower_ok):
                continue
            if parsed > upper or (parsed == upper and not upper_ok):
                continue
            selected.append(value)
        return selected

    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if domain.is_integer:
        lower_value = int(math.ceil(lower if interval.get("lower_inclusive", True) else lower + 1e-9))
        upper_value = int(math.floor(upper if interval.get("upper_inclusive", True) else upper - 1e-9))
        if upper_value < lower_value:
            return []
        return list(range(lower_value, upper_value + 1))

    return [round(lower, 12), round(upper, 12)]


def _intersect_constraints(
    agent: HNSWLIBTuningAgent,
    name: str,
    left: Dict[str, Any] | None,
    right: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if left is None:
        return copy.deepcopy(right) if right is not None else None
    if right is None:
        return copy.deepcopy(left)
    domain = agent.space.domains[name]
    left_values = _constraint_values(agent, name, left)
    right_values = _constraint_values(agent, name, right)
    if not left_values or not right_values:
        return None
    if domain.kind == "discrete":
        allowed = [value for value in left_values if value in set(right_values)]
        if not allowed:
            return None
        return {"kind": "discrete", "values": allowed}

    low = max(min(float(v) for v in left_values), min(float(v) for v in right_values))
    high = min(max(float(v) for v in left_values), max(float(v) for v in right_values))
    if high < low:
        return None
    if domain.is_integer:
        return {"kind": "range", "min": int(math.ceil(low)), "max": int(math.floor(high)), "integer": True}
    return {"kind": "range", "min": round(low, 12), "max": round(high, 12), "integer": False}


def _intersect_overrides(
    agent: HNSWLIBTuningAgent,
    left: Dict[str, Any] | None,
    right: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if left is None:
        return copy.deepcopy(right) if right is not None else None
    if right is None:
        return copy.deepcopy(left)
    merged: Dict[str, Any] = {}
    for name in PARAM_ORDER:
        constraint = _intersect_constraints(agent, name, left.get(name), right.get(name))
        if constraint is None:
            return None
        merged[name] = constraint
    return merged


def _constraint_from_card_dimensions(
    agent: HNSWLIBTuningAgent,
    dimensions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    values_by_param: Dict[str, List[Any]] = {name: [] for name in PARAM_ORDER}
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("parameter", "")).strip()
        interval = item.get("interval")
        if name not in values_by_param or not isinstance(interval, dict):
            continue
        values_by_param[name].extend(_domain_values_in_interval(agent, name, interval))
    override: Dict[str, Any] = {}
    for name in PARAM_ORDER:
        if values_by_param[name]:
            override[name] = _param_constraint_from_values(agent, name, values_by_param[name], pad_low=0, pad_high=0)
    return {name: value for name, value in override.items() if value is not None}


def _expand_constraint_toward_direction(
    agent: HNSWLIBTuningAgent,
    name: str,
    constraint: Dict[str, Any] | None,
    direction_score: float,
) -> Dict[str, Any]:
    base = constraint or _full_domain_constraint(agent, name)
    if abs(direction_score) <= 1e-9:
        return copy.deepcopy(base)
    values = _constraint_values(agent, name, base)
    if not values:
        return copy.deepcopy(base)
    low_pad = 1 if direction_score < 0 else 0
    high_pad = 1 if direction_score > 0 else 0
    expanded = _param_constraint_from_values(agent, name, values, pad_low=low_pad, pad_high=high_pad)
    return expanded or copy.deepcopy(base)


def _compact_card(card: Dict[str, Any]) -> Dict[str, Any]:
    perf = card.get("perf") or {}
    return {
        "card_id": card.get("card_id", ""),
        "region": (card.get("region") or {}).get("description", ""),
        "dimensions": copy.deepcopy((card.get("region") or {}).get("dimensions", [])),
        "quality": float(((card.get("quality") or {}).get("score", 0.0)) or 0.0),
        "feasibility": {
            "covered_count": int(((card.get("feasibility") or {}).get("covered_count", 0)) or 0),
            "feasible_count": int(((card.get("feasibility") or {}).get("feasible_count", 0)) or 0),
            "feasible_ratio": float(((card.get("feasibility") or {}).get("feasible_ratio", 0.0)) or 0.0),
            "median_recall_margin": float(((card.get("feasibility") or {}).get("median_recall_margin", 0.0)) or 0.0),
            "recall_threshold": float(((card.get("feasibility") or {}).get("recall_threshold", 0.0)) or 0.0),
        },
        "perf": {
            "median_feasible_qps": float((perf.get("median_feasible_qps", 0.0)) or 0.0),
            "median_elite_qps": float((perf.get("median_elite_qps", 0.0)) or 0.0),
            "best_qps": float(_perf_best_qps(perf)),
            "best_recall": float((perf.get("best_recall", 0.0)) or 0.0),
            "best_params": copy.deepcopy(perf.get("best_params") or {}),
        },
        "direction": copy.deepcopy(card.get("direction") or {}),
        "reasoning": copy.deepcopy(card.get("reasoning") or {}),
        "advice": str(card.get("advice") or card.get("advice_template") or ""),
    }


def _online_subgroup_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    observations: Sequence[Dict[str, Any]],
    stage_policy: Dict[str, Any],
    max_cards: int = 3,
) -> List[Dict[str, Any]]:
    if len(observations) < 6:
        return []

    subgroup_observations = [
        {
            "trial_ref": f"online:{idx + 1}",
            "params": dict(row["params"]),
            "recall": float(row["recall"]),
            "qps": float(row["qps"]),
        }
        for idx, row in enumerate(observations)
        if isinstance(row.get("params"), dict)
    ]
    if len(subgroup_observations) < 6:
        return []

    discretization = build_parameter_discretization(
        observations=subgroup_observations,
        param_order=PARAM_ORDER,
        max_bins=4,
    )
    raw_subgroups = discover_subgroups(
        observations=subgroup_observations,
        discretization=discretization,
        recall_threshold=float(stage_policy["recall_threshold"]),
        param_order=PARAM_ORDER,
        max_subgroup_dims=2 if len(subgroup_observations) < 12 else 3,
        min_covered=max(2, min(4, len(subgroup_observations) // 2)),
        min_feasible=2,
        elite_ratio=0.25,
    )
    if not raw_subgroups:
        return []

    scored = score_subgroups(
        raw_subgroups,
        weights={
            "feasible_count": 0.35,
            "recall_margin": 0.20,
            "feasible_qps": 0.45,
        },
    )
    selected = select_non_redundant_subgroups(scored, overlap_threshold=0.65, top_k=max_cards)
    if not selected:
        return []

    cards = build_insight_cards(
        selected_subgroups=selected,
        observations=subgroup_observations,
        recall_threshold=float(stage_policy["recall_threshold"]),
        task_name="online_hnsw_round",
    )
    candidates: List[Dict[str, Any]] = []
    for idx, (subgroup, card) in enumerate(zip(selected, cards), start=1):
        feasible_indices = list(subgroup.get("feasible_indices") or subgroup.get("covered_indices") or [])
        override: Dict[str, Any] = {}
        focus_params = _focus_params_from_card(card)
        feasibility = card.get("feasibility") or {}
        for name in PARAM_ORDER:
            support_values = [subgroup_observations[item_idx]["params"][name] for item_idx in feasible_indices]
            focus_values = [focus_params[name]] if name in focus_params else []
            feasible_center = ((card.get("direction") or {}).get("feasible_median") or {}).get(name)
            if feasible_center is not None and feasible_center not in focus_values:
                focus_values.append(feasible_center)
            direction_delta = float((card.get("direction") or {}).get("delta", {}).get(name, 0.0))
            constraint = _focused_constraint_from_profile(
                agent,
                name=name,
                focus_values=focus_values,
                support_values=support_values,
                feasible_ratio=float(feasibility.get("feasible_ratio", 0.0)),
                covered_count=int(feasibility.get("covered_count", 0)),
                recall_margin=float(feasibility.get("median_recall_margin", 0.0)),
                direction_delta=direction_delta,
            )
            if constraint is not None:
                override[name] = constraint
        if len(override) != len(PARAM_ORDER):
            continue
        candidates.append(
            {
                "candidate_id": f"online_subgroup_{idx:03d}",
                "source": "online_subgroup",
                "quality": float(subgroup.get("quality", 0.0)),
                "override": override,
                "card": _compact_card(card),
                "direction_delta": copy.deepcopy((card.get("direction") or {}).get("delta", {})),
                "direction_tendency": copy.deepcopy((card.get("direction") or {}).get("tendency", {})),
                "seed_params": focus_params,
            }
        )
    return candidates


def _insight_card_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    knowledge_context: Dict[str, Any] | str | None,
    recall_threshold: float,
    max_cards: int = 3,
) -> List[Dict[str, Any]]:
    # In knowledge_base_driven mode, knowledge_context is a string — no insight cards.
    if not isinstance(knowledge_context, dict):
        return []
    tasks = (knowledge_context or {}).get("similar_task_insights_full", [])
    if not isinstance(tasks, list):
        return []
    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_score = float(task.get("transfer_score", task.get("similarity", 0.0)))
        task_threshold_score = float((((task.get("threshold_match") or {}).get("score", 0.0)) or 0.0))
        card_payload = ((task.get("insight") or {}).get("insight_cards") or [])
        if not isinstance(card_payload, list):
            continue
        selected_card_ids = {
            str(card_id)
            for card_id in (task.get("selected_card_ids") or [])
            if str(card_id).strip()
        }
        selected_card_meta = {
            str(card.get("card_id", "")): card
            for card in (task.get("selected_cards") or [])
            if isinstance(card, dict) and str(card.get("card_id", "")).strip()
        }
        cards = [
            card
            for card in card_payload
            if isinstance(card, dict) and (not selected_card_ids or str(card.get("card_id", "")) in selected_card_ids)
        ]
        if not cards:
            cards = [card for card in card_payload if isinstance(card, dict)]
        for card in cards:
            if not isinstance(card, dict):
                continue
            feasibility = card.get("feasibility") or {}
            quality = float((card.get("quality") or {}).get("score", 0.0) or 0.0)
            feasible_ratio = float(feasibility.get("feasible_ratio", 0.0) or 0.0)
            margin_score = _recall_margin_score(float(feasibility.get("median_recall_margin", 0.0) or 0.0))
            card_threshold = float(feasibility.get("recall_threshold", recall_threshold))
            threshold_gap = abs(card_threshold - recall_threshold)
            threshold_score = _clamp01(math.exp(-18.0 * threshold_gap))
            selected_meta = selected_card_meta.get(str(card.get("card_id", "")), {})
            selected_card_score = float(selected_meta.get("transfer_card_score", 0.0) or 0.0)
            score = (
                task_score * 2.4
                + selected_card_score * 1.8
                + quality
                + feasible_ratio
                + margin_score * 0.5
                + task_threshold_score * 0.4
                + threshold_score * 0.3
            )
            ranked.append((score, task, card))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected = ranked[:max_cards]
    if not selected:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, (score, task, card) in enumerate(selected, start=1):
        dimensions = ((card.get("region") or {}).get("dimensions") or [])
        support_override = _constraint_from_card_dimensions(agent, dimensions)
        focus_params = _focus_params_from_card(card)
        feasibility = card.get("feasibility") or {}
        override: Dict[str, Any] = {}
        for name in PARAM_ORDER:
            support_values = _constraint_values(agent, name, support_override.get(name))
            if not support_values:
                support_values = _constraint_values(agent, name, None)
            focus_values = [focus_params[name]] if name in focus_params else []
            feasible_center = ((card.get("direction") or {}).get("feasible_median") or {}).get(name)
            if feasible_center is not None and feasible_center not in focus_values:
                focus_values.append(feasible_center)
            direction_delta = float(((card.get("direction") or {}).get("delta", {}) or {}).get(name, 0.0))
            constraint = _focused_constraint_from_profile(
                agent,
                name=name,
                focus_values=focus_values,
                support_values=support_values,
                feasible_ratio=float(feasibility.get("feasible_ratio", 0.0)),
                covered_count=int(feasibility.get("covered_count", 0)),
                recall_margin=float(feasibility.get("median_recall_margin", 0.0)),
                direction_delta=direction_delta,
            )
            if constraint is not None:
                override[name] = constraint
        if len(override) != len(PARAM_ORDER):
            continue
        candidates.append(
            {
                "candidate_id": f"similar_task_card_{idx:03d}",
                "source": "similar_task_insights",
                "quality": float(score),
                "override": override,
                "seed_params": focus_params,
                "cards_used": [
                    {
                        "task_name": str(task.get("task_name", "")),
                        "similarity": float(task.get("similarity", 0.0)),
                        "transfer_score": float(task.get("transfer_score", task.get("similarity", 0.0))),
                        "score_breakdown": copy.deepcopy(task.get("score_breakdown") or {}),
                        "card": _compact_card(card),
                    }
                ],
                "selected_candidate_ids": [f"similar_task_card_{idx:03d}"],
            }
        )
    return candidates


def _llm_refine_override(
    agent: HNSWLIBTuningAgent,
    *,
    stage_policy: Dict[str, Any],
    root_state: Dict[str, Any],
    frontier_override: Dict[str, Any] | None,
    candidate_overrides: Sequence[Dict[str, Any]],
    deterministic_override: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    if not _llm_refinement_enabled(agent) or not candidate_overrides:
        return None, {"enabled": False, "reason": "llm_refinement_disabled_or_no_candidates"}

    prompt_payload = {
        "task": "refine_hnsw_search_space",
        "recall_threshold": float(stage_policy["recall_threshold"]),
        "recall_slack": float(stage_policy.get("recall_slack", 0.0)),
        "optimization_stage": str(root_state.get("optimization_stage", "cold_start")),
        "current_best_feasible": root_state.get("current_best_feasible"),
        "closest_to_feasible": root_state.get("closest_to_feasible"),
        "frontier_override": frontier_override,
        "candidate_overrides": candidate_overrides,
        "deterministic_override": deterministic_override,
        "output_schema": {
            "allowed_values_override": {
                "M": {"kind": "range_or_discrete", "min": "...", "max": "...", "values": ["..."]},
                "ef_construction": {"kind": "range_or_discrete", "min": "...", "max": "...", "values": ["..."]},
                "ef": {"kind": "range_or_discrete", "min": "...", "max": "...", "values": ["..."]},
            },
            "reason": "short justification",
            "selected_candidate_ids": ["..."],
        },
    }
    prompt = (
        "You are the HNSW search-space refiner.\n"
        "Choose a tighter allowed parameter space for M, ef_construction, and ef.\n"
        "Use the subgroup candidates as region-level priors and local directions.\n"
        "Stay inside the provided frontier/candidate ranges, keep the current best feasible point plausible, "
        "and optimize for Recall >= threshold with higher QPS.\n"
        "Return strict JSON only.\n\n"
        f"Context:\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )

    raw = ""
    parsed = None
    error = ""
    try:
        raw = agent._invoke_llm("hnsw_search_space_refiner", prompt)
        parsed = agent._extract_json_payload(raw)
        if not isinstance(parsed, dict):
            error = "json_parse_failed"
        else:
            parsed = parsed.get("allowed_values_override") if isinstance(parsed.get("allowed_values_override"), dict) else parsed
    except Exception as exc:
        error = f"llm_error: {exc}"

    _llm_refine_logger = logging.getLogger("hnswlib_agent.llm")
    _llm_refine_logger.info(
        "LLM refine_override result ok=%s error=%s parsed=%s",
        isinstance(parsed, dict),
        error or "",
        json.dumps(parsed, ensure_ascii=False, indent=2) if isinstance(parsed, dict) else str(parsed),
    )

    if not isinstance(parsed, dict):
        return None, {"enabled": True, "used": False, "error": error, "raw": raw}

    try:
        normalized = agent.space.normalize_constraints(parsed)
    except Exception as exc:
        return None, {"enabled": True, "used": False, "error": f"invalid_override: {exc}", "raw": raw}

    return normalized, {"enabled": True, "used": True, "error": "", "raw": raw}


def _canonical_seed_candidate(
    agent: HNSWLIBTuningAgent,
    *,
    params: Dict[str, Any],
    allowed_values_override: Dict[str, Any] | None,
    source: str,
    note: str,
) -> Dict[str, Any] | None:
    if not isinstance(params, dict):
        return None
    normalized = dict(params)
    # Use the agent's own parameter order (generic across pipelines).
    for name in agent.space.order:
        domain = agent.space.domains[name]
        if not domain.is_integer or name not in normalized:
            continue
        value = normalized[name]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            normalized[name] = int(round(float(value)))
    try:
        canonical = agent.canonicalize(normalized, domain_constraints=allowed_values_override)
    except Exception:
        return None
    return {"params": canonical, "source": source, "note": note}


def _load_subgroup_init_payload(
    json_path: str,
    domain: str = "hnswlib_tuning",
) -> Dict[str, Any]:
    """Load and normalize a historical-task subgroup-mining insight JSON."""
    path = Path(str(json_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"agentic.subgroup_init.json_path not found: {path}")
    payload = _load_json_object(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"agentic.subgroup_init JSON is empty or invalid: {path}")
    normalized = _normalize_insight_payload(payload, fallback_task_name=path.stem, source_path=str(path))
    cards = normalized.get("insight_cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError(f"agentic.subgroup_init JSON has no insight_cards: {path}")
    if str(normalized.get("domain", "")) != domain:
        raise ValueError(
            f"agentic.subgroup_init JSON domain mismatch: expected {domain!r}, "
            f"got {normalized.get('domain')!r}."
        )
    return normalized


def _select_subgroup_init_card(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the single best insight card: highest quality.score, then best_qps."""
    cards = [c for c in (payload.get("insight_cards") or []) if isinstance(c, dict)]
    if not cards:
        raise ValueError("agentic.subgroup_init JSON has no insight_cards.")

    def _card_key(card: Dict[str, Any]) -> Tuple[float, float]:
        quality = card.get("quality") if isinstance(card.get("quality"), dict) else {}
        perf = card.get("perf") if isinstance(card.get("perf"), dict) else {}
        try:
            score = float(quality.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            best_qps = float(perf.get("best_qps", 0.0) or 0.0)
        except (TypeError, ValueError):
            best_qps = 0.0
        return (score, best_qps)

    return max(cards, key=_card_key)


def _subgroup_init_range(
    card: Dict[str, Any],
    base_specs: Dict[str, Dict[str, Any]],
    param_order: Sequence[str] = PARAM_ORDER,
) -> Dict[str, Any]:
    """Mined range from a single insight card (region.elite preferred, region.coarse fallback).

    Returns allowed_values_override-style specs covering every PARAM_ORDER name:
    params covered by the card use the mined [lo, hi]; uncovered params inherit
    the yaml base spec (no narrowing). normalize_constraints later clamps the
    mined range into the yaml base domain — the yaml params are never mutated.
    """
    region = card.get("region") if isinstance(card.get("region"), dict) else {}
    elite = region.get("elite") if isinstance(region.get("elite"), dict) else {}
    coarse = region.get("coarse") if isinstance(region.get("coarse"), dict) else {}
    mined: Dict[str, Any] = {}
    for name in param_order:
        rng = elite.get(name)
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            rng = coarse.get(name)
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            continue
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            continue
        if lo > hi:
            lo, hi = hi, lo
        base = base_specs.get(name) or {}
        is_integer = bool(base.get("integer", True))
        mined[name] = {
            "min": int(round(lo)) if is_integer else lo,
            "max": int(round(hi)) if is_integer else hi,
            "integer": is_integer,
        }
    for name in param_order:
        if name not in mined and name in base_specs:
            mined[name] = dict(base_specs[name])
    return mined


def _subgroup_init_seeds(
    agent: HNSWLIBTuningAgent,
    card: Dict[str, Any],
    allowed_values_override: Dict[str, Any],
    param_order: Sequence[str] = PARAM_ORDER,
) -> List[Dict[str, Any]]:
    """Initial-design seed from a single insight card: ONLY the best point."""
    focus = _focus_params_from_card(card, param_order=param_order)
    if not focus:
        return []
    seed = _canonical_seed_candidate(
        agent,
        params=focus,
        allowed_values_override=allowed_values_override,
        source="subgroup_init",
        note="best point from mined card",
    )
    return [seed] if seed is not None else []


def _subgroup_init_cold_start_cards(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cold-start guidance cards from a single insight card (mirrors :3219-3237 extraction)."""
    region = card.get("region") if isinstance(card.get("region"), dict) else {}
    dimensions = region.get("dimensions") if isinstance(region.get("dimensions"), list) else []
    direction = card.get("direction") if isinstance(card.get("direction"), dict) else {}
    elite = direction.get("elite_median") if isinstance(direction.get("elite_median"), dict) else {}
    feasible = direction.get("feasible_median") if isinstance(direction.get("feasible_median"), dict) else {}
    perf = card.get("perf") if isinstance(card.get("perf"), dict) else {}
    return [{
        "card_id": str(card.get("card_id", "")),
        "region_desc": str(region.get("description", "")),
        "dimensions": copy.deepcopy(dimensions),
        "elite_median": copy.deepcopy(elite) if elite else {},
        "feasible_median": copy.deepcopy(feasible) if feasible else {},
        "best_params": copy.deepcopy(perf.get("best_params") or {}),
        "best_qps": float(perf.get("best_qps", 0) or 0),
    }]


def _constraint_value_at_position(
    agent: HNSWLIBTuningAgent,
    name: str,
    constraint: Dict[str, Any] | None,
    position: float,
) -> Any | None:
    values = _constraint_values(agent, name, constraint)
    if not values:
        return None
    clipped = max(0.0, min(1.0, float(position)))
    idx = int(round(clipped * max(0, len(values) - 1)))
    return values[max(0, min(len(values) - 1, idx))]


def _expanded_override_from_rejections(
    agent: HNSWLIBTuningAgent,
    *,
    allowed_values_override: Dict[str, Any] | None,
    rejected_candidate_nodes: Sequence[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not isinstance(allowed_values_override, dict):
        return None

    expanded: Dict[str, Any] = {}
    changed = False
    for name in PARAM_ORDER:
        domain = agent.space.domains[name]
        if domain.kind != "range" or not domain.is_integer:
            constraint = allowed_values_override.get(name) if isinstance(allowed_values_override.get(name), dict) else None
            expanded[name] = copy.deepcopy(constraint) if constraint is not None else _full_domain_constraint(agent, name)
            continue

        constraint = allowed_values_override.get(name) if isinstance(allowed_values_override.get(name), dict) else {}
        base_low = int(domain.min_value)  # type: ignore[arg-type]
        base_high = int(domain.max_value)  # type: ignore[arg-type]
        if isinstance(constraint.get("values"), list) and constraint.get("values"):
            current_values = [int(round(float(value))) for value in constraint["values"]]
            current_low = min(current_values)
            current_high = max(current_values)
        else:
            current_low = int(round(float(constraint.get("min", base_low))))
            current_high = int(round(float(constraint.get("max", base_high))))

        rejected_values: List[int] = []
        for row in rejected_candidate_nodes:
            if not isinstance(row, dict) or row.get("reason") != "out_of_allowed_override":
                continue
            fields = row.get("fields")
            params = row.get("params")
            if not isinstance(fields, list) or name not in fields or not isinstance(params, dict) or name not in params:
                continue
            try:
                rejected_values.append(int(round(float(params[name]))))
            except Exception:
                continue

        next_low = max(base_low, min([current_low, *rejected_values])) if rejected_values else current_low
        next_high = min(base_high, max([current_high, *rejected_values])) if rejected_values else current_high
        expanded[name] = {"kind": "range", "min": next_low, "max": next_high, "integer": True}
        if next_low != current_low or next_high != current_high:
            changed = True

    return expanded if changed else None


def _frontier_probe_seed_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    candidate: Dict[str, Any],
    allowed_values_override: Dict[str, Any] | None,
    recall_threshold: float,
) -> List[Dict[str, Any]]:
    if allowed_values_override is None or recall_threshold > 0.88:
        return []

    candidate_override = candidate.get("override")
    constraint_source = candidate_override if isinstance(candidate_override, dict) else allowed_values_override
    if not isinstance(constraint_source, dict):
        return []

    seed_params = candidate.get("seed_params") if isinstance(candidate.get("seed_params"), dict) else {}
    base_params: Dict[str, Any] = {}
    for name in PARAM_ORDER:
        constraint = constraint_source.get(name) if isinstance(constraint_source.get(name), dict) else None
        value = None
        if name in seed_params:
            values = _constraint_values(agent, name, constraint)
            if values:
                try:
                    seed_value = float(seed_params[name])
                    value = min(values, key=lambda candidate_value: abs(float(candidate_value) - seed_value))
                except Exception:
                    value = None
        if value is None:
            value = _constraint_value_at_position(agent, name, constraint, 0.5)
        if value is None:
            return []
        base_params[name] = value

    candidate_id = str(candidate.get("candidate_id", candidate.get("source", "candidate")))
    source = str(candidate.get("source", "search_space_refinement"))
    probe_specs = [
        ("low_ef_frontier", {"ef": 0.12}),
        ("boundary_ef_frontier", {"ef": 0.06}),
        ("low_efc_low_ef_frontier", {"ef_construction": 0.22, "ef": 0.12}),
    ]
    seeds: List[Dict[str, Any]] = []
    for probe_name, positions in probe_specs:
        params = dict(base_params)
        valid = True
        for name, position in positions.items():
            constraint = constraint_source.get(name) if isinstance(constraint_source.get(name), dict) else None
            value = _constraint_value_at_position(agent, name, constraint, position)
            if value is None:
                valid = False
                break
            params[name] = value
        if not valid:
            continue
        seed = _canonical_seed_candidate(
            agent,
            params=params,
            allowed_values_override=allowed_values_override,
            source=source,
            note=f"frontier probe from {candidate_id}: {probe_name}",
        )
        if seed is not None:
            seeds.append(seed)
    return seeds


def _refinement_seed_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    candidates: Sequence[Dict[str, Any]],
    allowed_values_override: Dict[str, Any] | None,
    max_count: int,
    recall_threshold: float,
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for candidate in candidates:
        seed_params = candidate.get("seed_params")
        seed = _canonical_seed_candidate(
            agent,
            params=seed_params if isinstance(seed_params, dict) else {},
            allowed_values_override=allowed_values_override,
            source=str(candidate.get("source", "search_space_refinement")),
            note=f"focus seed from {candidate.get('candidate_id', candidate.get('source', 'candidate'))}",
        )
        if seed is None:
            continue
        key = _build_key_from_params(seed["params"])
        if key in seen:
            continue
        seen.add(key)
        seeds.append(seed)
        if len(seeds) >= max_count:
            break
        for frontier_seed in _frontier_probe_seed_candidates(
            agent,
            candidate=candidate,
            allowed_values_override=allowed_values_override,
            recall_threshold=recall_threshold,
        ):
            frontier_key = _build_key_from_params(frontier_seed["params"])
            if frontier_key in seen:
                continue
            seen.add(frontier_key)
            seeds.append(frontier_seed)
            if len(seeds) >= max_count:
                break
        if len(seeds) >= max_count:
            break
    return seeds


def _verify_hypothesis(
    hypothesis: Dict[str, Any],
    trial: Dict[str, Any],
    recall_threshold: float,
) -> Dict[str, Any]:
    """Compare the real frontier from an executed trial against the
    ``falsifiable_prediction`` attached to a frontier-reasoning hypothesis.

    Returns a dict with ``status`` (``confirmed`` / ``partially_confirmed`` /
    ``falsified``) and ``details`` describing which predictions held.
    """
    prediction = hypothesis.get("falsifiable_prediction") if isinstance(hypothesis, dict) else {}
    if not isinstance(prediction, dict) or not prediction:
        return {"status": "no_prediction", "details": "hypothesis has no falsifiable_prediction"}

    metrics = trial.get("metrics") if isinstance(trial, dict) else None
    if not isinstance(metrics, dict) or "recall" not in metrics or "qps" not in metrics:
        return {"status": "no_data", "details": "trial has no valid metrics for verification"}

    real_recall = float(metrics["recall"])
    real_qps = float(metrics["qps"])
    real_selected_ef = metrics.get("selected_ef")
    real_feasible = real_recall >= recall_threshold

    checks: List[Dict[str, Any]] = []

    # --- feasibility ---
    pred_feasible = prediction.get("feasible")
    if pred_feasible is not None:
        checks.append({
            "check": "feasible",
            "predicted": bool(pred_feasible),
            "actual": real_feasible,
            "passed": bool(pred_feasible) == real_feasible,
        })

    # --- G_tau range ---
    g_tau_range = prediction.get("G_tau_range")
    if isinstance(g_tau_range, list) and len(g_tau_range) == 2:
        lo, hi = float(g_tau_range[0]), float(g_tau_range[1])
        checks.append({
            "check": "G_tau_range",
            "predicted": [lo, hi],
            "actual": real_qps,
            "passed": lo - 1e-9 <= real_qps <= hi + 1e-9,
        })

    # --- ef_star range ---
    ef_star_range = prediction.get("ef_star_range")
    if isinstance(ef_star_range, list) and len(ef_star_range) == 2 and real_selected_ef is not None:
        lo, hi = int(ef_star_range[0]), int(ef_star_range[1])
        real_ef = int(real_selected_ef)
        checks.append({
            "check": "ef_star_range",
            "predicted": [lo, hi],
            "actual": real_ef,
            "passed": lo <= real_ef <= hi,
        })

    # --- frontier_shift direction ---
    pred_shift = prediction.get("frontier_shift", "")
    checks.append({
        "check": "frontier_shift",
        "predicted": str(pred_shift),
        "actual": {"recall": real_recall, "qps": real_qps, "feasible": real_feasible},
        "passed": None,  # directional check requires comparison with previous best
        "note": "directional check deferred to round-level comparison",
    })

    passed = [c for c in checks if c.get("passed") is True]
    failed = [c for c in checks if c.get("passed") is False]
    deferred = [c for c in checks if c.get("passed") is None]

    if not failed and not passed and deferred:
        status = "deferred"
    elif not failed:
        status = "confirmed" if passed else "no_checks"
    elif passed and failed:
        status = "partially_confirmed"
    else:
        status = "falsified"

    return {
        "status": status,
        "details": {
            "checks": checks,
            "passed_count": len(passed),
            "failed_count": len(failed),
            "deferred_count": len(deferred),
        },
    }


def _build_allowed_values_override(
    agent: HNSWLIBTuningAgent,
    *,
    stage_trials: Sequence[Dict[str, Any]],
    observations: Sequence[Dict[str, Any]],
    stage_policy: Dict[str, Any],
    knowledge_context: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    root_state = agent.build_root_state(
        round_idx=0,
        stage_trials=stage_trials,
        stage_policy=stage_policy,
    )
    stage = str(root_state.get("optimization_stage", "cold_start"))
    recall_threshold = float(stage_policy["recall_threshold"])
    recall_slack = float(stage_policy.get("recall_slack", 0.0))
    near_width = max(recall_slack, 0.003)

    feasible = [obs for obs in observations if float(obs.get("constraint", -1.0)) >= 0.0]
    near_feasible = [obs for obs in observations if float(obs.get("constraint", -1.0)) >= -near_width]
    frontier_override: Dict[str, Any] | None = None
    frontier_report: Dict[str, Any] = {"enabled": False, "reason": "insufficient_observations"}
    if len(observations) >= 4:
        selected = feasible if feasible else near_feasible
        if selected:
            if feasible:
                selected = sorted(
                    feasible,
                    key=lambda row: (float(row.get("qps", 0.0)), -abs(float(row.get("constraint", 0.0)))),
                    reverse=True,
                )[: min(len(feasible), 12)]
                pad_map = {
                    "M": [2, 2],
                    "ef_construction": [4, 6],
                    "ef": [2, 6],
                }
                if recall_threshold > 0.88:
                    pad_map = {
                        "M": [3, 3],
                        "ef_construction": [8, 10],
                        "ef": [3, 8],
                    }
                if stage == "recall_too_high_qps_low":
                    pad_map["ef"] = [2, 4]
                elif stage == "feasible_near_boundary":
                    pad_map["ef"] = [1, 5]
            else:
                selected = sorted(
                    near_feasible,
                    key=lambda row: (float(row.get("constraint", -1.0)), float(row.get("qps", 0.0))),
                    reverse=True,
                )[: min(len(near_feasible), 10)]
                pad_map = {
                    "M": [2, 4],
                    "ef_construction": [4, 10],
                    "ef": [1, 8],
                }

            elite_count = max(2, min(len(selected), max(2, len(selected) // 3)))
            elite = list(selected[:elite_count])
            selected_params = [row["params"] for row in selected if isinstance(row.get("params"), dict)]
            elite_params = [row["params"] for row in elite if isinstance(row.get("params"), dict)]

            frontier_override = {}
            local_direction: Dict[str, str] = {}
            for name in PARAM_ORDER:
                values = [params[name] for params in selected_params]
                if not values:
                    continue
                feasible_median = float(median(values))
                elite_median = float(median([params[name] for params in elite_params])) if elite_params else feasible_median
                if elite_median > feasible_median + 1e-9:
                    pad_map[name][1] += 1
                    local_direction[name] = "up"
                elif elite_median < feasible_median - 1e-9:
                    pad_map[name][0] += 1
                    local_direction[name] = "down"
                else:
                    local_direction[name] = "flat"
                constraint = _param_constraint_from_values(
                    agent,
                    name,
                    values,
                    pad_low=pad_map[name][0],
                    pad_high=pad_map[name][1],
                )
                if constraint is not None:
                    frontier_override[name] = constraint

            if len(frontier_override) == len(PARAM_ORDER):
                frontier_report = {
                    "enabled": True,
                    "reason": "frontier_band",
                    "optimization_stage": stage,
                    "selected_count": len(selected),
                    "feasible_count": len(feasible),
                    "near_feasible_count": len(near_feasible),
                    "local_direction": local_direction,
                }
            else:
                frontier_override = None
                frontier_report = {"enabled": False, "reason": "partial_frontier_override"}

    subgroup_candidates = _online_subgroup_candidates(
        agent,
        observations=observations,
        stage_policy=stage_policy,
    )
    insight_candidates = _insight_card_candidates(
        agent,
        knowledge_context=knowledge_context,
        recall_threshold=recall_threshold,
    )
    all_candidates = sorted(
        [*subgroup_candidates, *insight_candidates],
        key=lambda row: (float(row.get("quality", 0.0)), -_override_volume(agent, row.get("override"))),
        reverse=True,
    )

    deterministic_override: Dict[str, Any] | None = None
    deterministic_reason = ""
    selected_candidate_ids: List[str] = []
    if all_candidates:
        selected_candidates = all_candidates[: min(3, len(all_candidates))]
        selected_candidate_ids = [str(item.get("candidate_id", "")) for item in selected_candidates]
        deterministic_override = copy.deepcopy(selected_candidates[0].get("override"))
        deterministic_reason = "focused_subgroup_primary"
        for candidate in selected_candidates[1:]:
            candidate_override = candidate.get("override")
            intersected = _intersect_overrides(agent, deterministic_override, candidate_override)
            if intersected is not None and _override_volume(agent, intersected) > 0.0:
                deterministic_override = intersected
                deterministic_reason = "focused_subgroup_intersection"

    if deterministic_override is None and frontier_override is not None:
        deterministic_override = copy.deepcopy(frontier_override)
        deterministic_reason = "frontier_only"
    elif deterministic_override is not None and frontier_override is not None:
        intersected = _intersect_overrides(agent, deterministic_override, frontier_override)
        if intersected is not None and _override_volume(agent, intersected) > 0.0:
            deterministic_override = intersected
            deterministic_reason = "subgroup_frontier_intersection"
        elif _override_volume(agent, frontier_override) < _override_volume(agent, deterministic_override):
            deterministic_override = copy.deepcopy(frontier_override)
            deterministic_reason = "frontier_narrower_than_subgroup"

    if deterministic_override is None:
        return None, {
            "enabled": False,
            "reason": "no_frontier_or_subgroup_signal",
            "optimization_stage": stage,
            "feasible_count": len(feasible),
            "near_feasible_count": len(near_feasible),
            "frontier_report": frontier_report,
            "subgroup_candidates": subgroup_candidates,
            "insight_candidates": insight_candidates,
        }

    llm_override, llm_report = _llm_refine_override(
        agent,
        stage_policy=stage_policy,
        root_state=root_state,
        frontier_override=frontier_override,
        candidate_overrides=all_candidates[:3],
        deterministic_override=deterministic_override,
    )
    final_override = llm_override or deterministic_override
    final_reason = "llm_refined" if llm_override is not None else deterministic_reason
    seed_candidates = _refinement_seed_candidates(
        agent,
        candidates=all_candidates,
        allowed_values_override=final_override,
        max_count=4,
        recall_threshold=recall_threshold,
    )
    if llm_override is not None and not seed_candidates and deterministic_override is not None:
        final_override = deterministic_override
        final_reason = deterministic_reason
        llm_report = {
            **llm_report,
            "used": False,
            "fallback_reason": "llm_override_eliminated_all_focus_seeds",
        }
        seed_candidates = _refinement_seed_candidates(
            agent,
            candidates=all_candidates,
            allowed_values_override=final_override,
            max_count=4,
            recall_threshold=recall_threshold,
        )

    return final_override, {
        "enabled": True,
        "reason": final_reason,
        "optimization_stage": stage,
        "feasible_count": len(feasible),
        "near_feasible_count": len(near_feasible),
        "selected_candidate_ids": selected_candidate_ids,
        "frontier_report": frontier_report,
        "subgroup_candidates": subgroup_candidates,
        "insight_candidates": insight_candidates,
        "seed_candidates": seed_candidates,
        "llm_refinement": llm_report,
    }


def _count_success_runs(trials: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for trial in trials if trial.get("stage") == "unified" and trial.get("status") == "success")


def _count_stage_b_runs(trials: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1
        for trial in trials
        if trial.get("stage") == "unified"
        and isinstance(trial.get("params"), dict)
        and int(trial.get("repeat_idx", -1)) >= 0
    )


# Initialization seeds (subgroup mining / transfer / external seeds / LSH /
# regression-tree regions) execute before LLM tuning rounds; the search
# budget does NOT include them.
_SEED_PROPOSAL_SOURCES = frozenset(
    {"subgroup_init", "transfer_seed", "external_seed_file", "lsh_space_fill",
     "regression_tree_region"}
)


def _trial_params_key(params: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _count_seed_runs(
    trials: Sequence[Dict[str, Any]],
    seed_param_keys: Optional[Set[Tuple[Tuple[str, str], ...]]] = None,
) -> int:
    """Count initialization-seed executions, excluded from the tuning budget.

    Live trials carry ``proposal_source``; resume-reloaded minimal rows do
    not, so seeds are also matched by their parameter key.
    """
    count = 0
    for trial in trials:
        if trial.get("proposal_source") in _SEED_PROPOSAL_SOURCES:
            count += 1
            continue
        if seed_param_keys:
            params = trial.get("params")
            if isinstance(params, dict) and _trial_params_key(params) in seed_param_keys:
                count += 1
    return count


def _attempted_task_keys(trials: Sequence[Dict[str, Any]]) -> set[Tuple[Tuple[Any, ...], int]]:
    keys: set[Tuple[Tuple[Any, ...], int]] = set()
    for trial in trials:
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        repeat_idx = int(trial.get("repeat_idx", -1))
        if repeat_idx < 0:
            continue
        try:
            keys.add((params_to_key(params, METRICS_PARAM_ORDER), repeat_idx))
        except Exception:
            continue
    return keys


def _exhausted_param_keys(trials: Sequence[Dict[str, Any]], repeat: int) -> set[Tuple[Any, ...]]:
    repeat_map: Dict[Tuple[Any, ...], set[int]] = {}
    for trial in trials:
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        repeat_idx = int(trial.get("repeat_idx", -1))
        if repeat_idx < 0:
            continue
        try:
            key = params_to_key(params, METRICS_PARAM_ORDER)
        except Exception:
            continue
        repeat_map.setdefault(key, set()).add(repeat_idx)
    return {key for key, seen in repeat_map.items() if len(seen) >= repeat}


def _build_tasks(
    stage: str,
    candidates: Sequence[Dict[str, Any]],
    needed_runs: int,
    repeat: int,
    occupied_task_keys: set[Tuple[Tuple[Any, ...], int]],
    round_idx: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if needed_runs <= 0:
        return tasks
    for candidate in candidates:
        params = candidate["params"]
        p_key = params_to_key(params, METRICS_PARAM_ORDER)
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
                    "proposal_node_id": candidate.get("node_id", ""),
                }
            )
            occupied_task_keys.add(task_key)
            if len(tasks) >= needed_runs:
                return tasks
    return tasks


def _project_trial_for_file(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Trials-file projection: ONLY params and metrics (recall + QPS).

    Bookkeeping fields needed by the pipeline (stage/status/repeat_idx/
    proposal_source) are re-derived with safe defaults when the file is
    loaded (see ``utils/hnswlib_metrics.load_trials``); everything else
    stays in memory only.
    """
    metrics = trial.get("metrics")
    if isinstance(metrics, dict):
        kept_metrics = {k: metrics[k] for k in ("recall", "qps") if k in metrics}
    else:
        kept_metrics = metrics
    return {"params": trial.get("params"), "metrics": kept_metrics}


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
                    "task_name": task.get("task_name", ""),
                },
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            trial = future.result()
            task = future_map[future]
            trial["proposal_node_id"] = task.get("proposal_node_id", "")
            used_names = list(task.get("used_similar_task_names", []))
            trial["used_similar_task_names"] = used_names
            trial["used_similar_task_count"] = int(task.get("used_similar_task_count", len(used_names)))
            append_trial(trials_path, _project_trial_for_file(trial))
            new_trials.append(trial)
    return new_trials


def _unified_report(
    all_trials: Sequence[Dict[str, Any]],
    target_runs: int,
    recall_threshold: float,
    recall_slack: float,
) -> Dict[str, Any]:
    stage_trials = [trial for trial in all_trials if trial.get("stage") == "unified"]
    success_trials = [trial for trial in stage_trials if trial.get("status") == "success"]
    aggregates = aggregate_success_trials(stage_trials)
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


def _to_feature_row(params: Dict[str, Any]) -> List[float]:
    row = [float(params[name]) for name in BUILD_PARAM_ORDER]
    if "selected_ef" in params:
        row.append(float(params["selected_ef"]))
    elif "ef" in params:
        row.append(float(params["ef"]))
    else:
        row.append(0.0)
    return row


def _extract_round_success_observations(round_trials: Sequence[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
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
            feature_params = {
                "M": params["M"],
                "ef_construction": params["ef_construction"],
                "selected_ef": metrics.get("selected_ef", params.get("ef")),
            }
            features.append(_to_feature_row(feature_params))
            actual_qps.append(float(metrics["qps"]))
        except (KeyError, TypeError, ValueError):
            continue
    return features, actual_qps


def _resolve_linked_trials_path(linked_trials_path: str, models_dir: Path) -> Path:
    linked = Path(linked_trials_path).expanduser()
    if linked.is_absolute():
        return linked.resolve()
    return (models_dir / linked).resolve()


def _load_recall_qualified_trials(linked_trials_path: Path, recall_threshold: float) -> List[Dict[str, Any]]:
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
            ordered_params = {name: params[name] for name in PARAM_ORDER}
            ordered_params["ef"] = metrics.get("selected_ef", params.get("ef"))
        except (KeyError, TypeError, ValueError):
            continue
        if recall < recall_threshold:
            continue
        selected.append(
            {
                "params": ordered_params,
                "recall": recall,
                "qps": qps,
                "proposal_source": trial.get("proposal_source", "unknown"),
                "proposal_round": int(trial.get("proposal_round", -1)),
                "proposal_note": trial.get("proposal_note", ""),
            }
        )
    selected.sort(key=lambda item: (item["qps"], item["recall"]), reverse=True)
    return selected


def _sparse_transfer_trials(sorted_trials: Sequence[Dict[str, Any]], trials_per_task: int, stride: int = 10) -> List[Dict[str, Any]]:
    if trials_per_task <= 0 or not sorted_trials:
        return []
    window = list(sorted_trials[: min(len(sorted_trials), max(trials_per_task * max(1, stride), trials_per_task))])
    sampled = list(window[:: max(1, stride)])[:trials_per_task]
    if len(sampled) >= trials_per_task:
        return sampled
    seen_ids = {id(item) for item in sampled}
    for item in window:
        if id(item) in seen_ids:
            continue
        sampled.append(item)
        if len(sampled) >= trials_per_task:
            break
    return sampled


def _model_similarity(actual_qps: Sequence[float], feature_rows: Sequence[Sequence[float]], model: Any) -> float:
    if len(actual_qps) < 2 or len(feature_rows) != len(actual_qps):
        return 0.0
    predicted = [float(value) for value in model.predict(feature_rows)]
    if len(predicted) != len(actual_qps):
        return 0.0
    actual_mean = sum(actual_qps) / len(actual_qps)
    pred_mean = sum(predicted) / len(predicted)
    numerator = sum((a - actual_mean) * (p - pred_mean) for a, p in zip(actual_qps, predicted))
    actual_var = sum((a - actual_mean) ** 2 for a in actual_qps)
    pred_var = sum((p - pred_mean) ** 2 for p in predicted)
    if actual_var <= 1e-12 or pred_var <= 1e-12:
        mae = sum(abs(a - p) for a, p in zip(actual_qps, predicted)) / len(actual_qps)
        scale = max(1.0, sum(abs(a) for a in actual_qps) / len(actual_qps))
        return max(0.0, 1.0 - mae / scale)
    corr = numerator / math.sqrt(actual_var * pred_var)
    return max(-1.0, min(1.0, corr))


def _unique_tokens(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _param_domain_span(name: str, params_cfg: Dict[str, Any] | None = None) -> float:
    """Return the span (max - min) for *name* clamped to ≥ 1.0.

    When *params_cfg* is provided the span is read from the config-provided
    parameter space; otherwise a conservative default from ``_PARAM_DEFAULT_SPANS``
    is used.
    """
    if isinstance(params_cfg, dict):
        spec = params_cfg.get(name) or {}
        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is not None and max_value is not None:
            return max(1.0, float(max_value) - float(min_value))
    return float(_PARAM_DEFAULT_SPANS.get(name, 1.0))


def _param_profile_similarity(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> float | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    distances: List[float] = []
    for name in PARAM_ORDER:
        if name not in left or name not in right:
            return None
        span = _param_domain_span(name)
        distance = min(1.0, abs(float(left[name]) - float(right[name])) / span)
        distances.append(distance)
    if not distances:
        return None
    return _clamp01(1.0 - sum(distances) / len(distances))


def _recall_margin_score(margin: float) -> float:
    if margin >= 0.0:
        return _clamp01(0.5 + min(0.5, margin * 25.0))
    return _clamp01(0.5 - min(0.5, abs(margin) * 60.0))


def _elite_gain_score(median_feasible_qps: float, median_elite_qps: float) -> float:
    base = max(1.0, float(median_feasible_qps))
    gain_ratio = max(0.0, float(median_elite_qps) - float(median_feasible_qps)) / base
    return _clamp01(gain_ratio / 0.15)


def _threshold_match_info(
    target_threshold: float,
    available_thresholds: Sequence[Any],
) -> Dict[str, Any]:
    resolved: List[float] = []
    for raw in available_thresholds:
        try:
            resolved.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not resolved:
        return {
            "available_thresholds": [],
            "best_threshold": None,
            "threshold_gap": None,
            "score": 0.45,
            "exact_match": False,
        }
    best = min(resolved, key=lambda value: abs(value - target_threshold))
    gap = abs(best - target_threshold)
    score = math.exp(-18.0 * gap)
    return {
        "available_thresholds": sorted(set(round(value, 12) for value in resolved)),
        "best_threshold": float(best),
        "threshold_gap": float(gap),
        "score": _clamp01(score),
        "exact_match": gap <= 1e-9,
    }


def _insight_thresholds(payload: Dict[str, Any]) -> List[float]:
    thresholds: List[float] = []
    build_params = payload.get("build_params") or {}
    raw_thresholds = build_params.get("recall_thresholds")
    if isinstance(raw_thresholds, list):
        thresholds.extend(raw_thresholds)
    elif build_params.get("recall_threshold") is not None:
        thresholds.append(build_params.get("recall_threshold"))
    cards = payload.get("insight_cards")
    if isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict):
                continue
            threshold = ((card.get("feasibility") or {}).get("recall_threshold"))
            if threshold is not None:
                thresholds.append(threshold)
    seen: set[float] = set()
    resolved: List[float] = []
    for raw in thresholds:
        try:
            parsed = round(float(raw), 12)
        except (TypeError, ValueError):
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        resolved.append(float(parsed))
    return resolved


def _normalize_insight_card(
    card: Dict[str, Any],
    *,
    task_name: str,
    default_recall_threshold: float | None,
    global_best: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(card, dict):
        return {}

    dimensions, description = _recommended_interval_dimensions(card)
    feasibility = card.get("feasibility") if isinstance(card.get("feasibility"), dict) else {}
    covered_count = int((feasibility.get("covered_count", feasibility.get("covered", 0))) or 0)
    feasible_count = int((feasibility.get("feasible_count", feasibility.get("feasible", 0))) or 0)
    feasible_ratio_raw = feasibility.get("feasible_ratio")
    feasible_ratio = (
        float(feasible_ratio_raw)
        if feasible_ratio_raw is not None
        else (float(feasible_count) / float(covered_count) if covered_count > 0 else 0.0)
    )
    median_margin = float(
        (feasibility.get("median_recall_margin", feasibility.get("recall_margin_median", 0.0))) or 0.0
    )
    recall_threshold = float(
        (
            feasibility.get("recall_threshold", default_recall_threshold if default_recall_threshold is not None else 0.0)
        )
        or 0.0
    )

    perf = card.get("perf") if isinstance(card.get("perf"), dict) else {}
    qps = card.get("qps") if isinstance(card.get("qps"), dict) else {}
    top_configs = card.get("top_configs") if isinstance(card.get("top_configs"), list) else []
    best_qps = _perf_best_qps(perf)
    if best_qps <= 0.0:
        try:
            best_qps = float(qps.get("best", 0.0) or 0.0)
        except (TypeError, ValueError):
            best_qps = 0.0
    median_feasible_qps = perf.get("median_feasible_qps", qps.get("p75_feasible", 0.0))
    median_elite_qps = perf.get("median_elite_qps", best_qps or median_feasible_qps)
    best_params = perf.get("best_params") if isinstance(perf.get("best_params"), dict) else {}
    if not best_params and top_configs and isinstance(top_configs[0], dict):
        best_params = copy.deepcopy(top_configs[0].get("params") or {})
    if not best_params and isinstance(global_best, dict):
        best_params = copy.deepcopy(global_best.get("params") or {})
    best_recall = perf.get("best_recall")
    if best_recall is None:
        recall_candidates = [
            item.get("recall")
            for item in top_configs
            if isinstance(item, dict) and item.get("recall") is not None
        ]
        if recall_candidates:
            best_recall = max(recall_candidates)
        elif isinstance(global_best, dict):
            best_recall = global_best.get("recall", recall_threshold)
    if best_recall is None:
        best_recall = recall_threshold + max(0.0, median_margin)

    legacy_direction = card.get("direction") if isinstance(card.get("direction"), dict) else {}
    fallback_focus = (
        copy.deepcopy(card.get("elite_median"))
        if isinstance(card.get("elite_median"), dict)
        else copy.deepcopy(best_params)
    )
    direction = _legacy_direction_maps(legacy_direction, fallback_focus)
    reasoning = copy.deepcopy(card.get("reasoning") or {})
    if not isinstance(reasoning, dict):
        reasoning = {}
    if not reasoning.get("focus_params") and best_params:
        reasoning["focus_params"] = copy.deepcopy(best_params)
    if "broad_card" not in reasoning:
        reasoning["broad_card"] = False

    return {
        "card_id": str(card.get("card_id", "")),
        "task_name": task_name,
        "region": {
            "description": description or str(((card.get("region") or {}).get("description", "")) or ""),
            "dimensions": dimensions,
        },
        "feasibility": {
            "covered_count": covered_count,
            "feasible_count": feasible_count,
            "feasible_ratio": float(feasible_ratio),
            "median_recall_margin": median_margin,
            "recall_threshold": recall_threshold,
        },
        "perf": {
            "median_feasible_qps": float(median_feasible_qps or 0.0),
            "median_elite_qps": float(median_elite_qps or 0.0),
            "best_qps": float(best_qps or 0.0),
            "best_recall": float(best_recall or 0.0),
            "best_params": copy.deepcopy(best_params),
        },
        "quality": {"score": _coerce_quality_score(card.get("quality", 0.0))},
        "direction": direction,
        "reasoning": reasoning,
        "advice": str(card.get("advice") or card.get("advice_template") or card.get("hint") or ""),
    }


def _normalize_insight_payload(
    payload: Dict[str, Any],
    *,
    fallback_task_name: str = "",
    source_path: str = "",
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    raw_cards = payload.get("insight_cards")
    if not isinstance(raw_cards, list):
        return copy.deepcopy(payload)

    task_name = str(payload.get("task_name") or fallback_task_name or Path(source_path).stem).strip()
    global_best = payload.get("global_feasible_best") if isinstance(payload.get("global_feasible_best"), dict) else None
    thresholds = _insight_thresholds(payload)
    default_recall_threshold = thresholds[0] if thresholds else payload.get("recall_threshold")
    normalized_cards = [
        _normalize_insight_card(
            card,
            task_name=task_name,
            default_recall_threshold=float(default_recall_threshold) if default_recall_threshold is not None else None,
            global_best=global_best,
        )
        for card in raw_cards
        if isinstance(card, dict)
    ]

    normalized_payload = copy.deepcopy(payload)
    normalized_payload["task_name"] = task_name
    normalized_payload["insight_cards"] = [card for card in normalized_cards if card]
    build_params = copy.deepcopy(payload.get("build_params") or {})
    if default_recall_threshold is not None and build_params.get("recall_threshold") is None:
        build_params["recall_threshold"] = float(default_recall_threshold)
    if thresholds and build_params.get("recall_thresholds") is None:
        build_params["recall_thresholds"] = thresholds
    if build_params:
        normalized_payload["build_params"] = build_params
    return normalized_payload


def _load_insight_payload_for_task(
    *,
    insights_dir: Path | None,
    task_name: str,
) -> Tuple[Dict[str, Any], str]:
    if insights_dir is None:
        return {}, ""
    insight_path = insights_dir / f"{_normalize_task_stem(task_name)}.json"
    payload = _normalize_insight_payload(
        _load_json_object(insight_path),
        fallback_task_name=task_name,
        source_path=str(insight_path),
    )
    if not payload:
        return {}, ""
    return payload, str(insight_path)


def _value_in_interval(value: Any, interval: Dict[str, Any]) -> bool:
    try:
        parsed = float(value)
        lower = float(interval["lower"])
        upper = float(interval["upper"])
    except (KeyError, TypeError, ValueError):
        return False
    lower_inclusive = bool(interval.get("lower_inclusive", True))
    upper_inclusive = bool(interval.get("upper_inclusive", True))
    if parsed < lower or (parsed == lower and not lower_inclusive):
        return False
    if parsed > upper or (parsed == upper and not upper_inclusive):
        return False
    return True


def _params_match_card_dimensions(params: Dict[str, Any], card: Dict[str, Any]) -> bool:
    region = card.get("region")
    if isinstance(region, dict):
        dimensions = region.get("dimensions") or []
    else:
        dimensions = card.get("dimensions") or []
    if not isinstance(params, dict) or not isinstance(dimensions, list):
        return False
    for item in dimensions:
        if not isinstance(item, dict):
            return False
        name = str(item.get("parameter", "")).strip()
        interval = item.get("interval")
        if name not in PARAM_ORDER or not isinstance(interval, dict):
            return False
        if name not in params or not _value_in_interval(params[name], interval):
            return False
    return True


def _select_transfer_trials(
    *,
    qualified_trials: Sequence[Dict[str, Any]],
    selected_cards: Sequence[Dict[str, Any]],
    trials_per_task: int,
    bootstrap: bool,
) -> List[Dict[str, Any]]:
    if trials_per_task <= 0 or not qualified_trials:
        return []
    selected: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    def _append_trial(trial: Dict[str, Any]) -> None:
        params = trial.get("params")
        if not isinstance(params, dict):
            return
        key = _build_key_from_params(params)
        if key in seen:
            return
        seen.add(key)
        selected.append(copy.deepcopy(trial))

    _append_trial(qualified_trials[0])
    for card in selected_cards:
        matches = [
            trial
            for trial in qualified_trials
            if isinstance(trial.get("params"), dict) and _params_match_card_dimensions(trial["params"], card)
        ]
        if matches:
            _append_trial(matches[0])
        if len(selected) >= trials_per_task:
            return selected[:trials_per_task]

    stride = 6 if bootstrap else 10
    for trial in _sparse_transfer_trials(qualified_trials, trials_per_task=trials_per_task, stride=stride):
        _append_trial(trial)
        if len(selected) >= trials_per_task:
            break
    return selected[:trials_per_task]


def _current_anchor_params(
    round_trials: Sequence[Dict[str, Any]],
    recall_threshold: float,
) -> Tuple[Dict[str, Any] | None, str]:
    success_rows: List[Dict[str, Any]] = []
    near_width = max(0.003, min(0.02, recall_threshold * 0.01))
    for trial in round_trials:
        if trial.get("status") != "success":
            continue
        params = trial.get("params")
        metrics = trial.get("metrics")
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        if "recall" not in metrics or "qps" not in metrics:
            continue
        try:
            recall = float(metrics["recall"])
            qps = float(metrics["qps"])
        except (TypeError, ValueError):
            continue
        success_rows.append(
            {
                "params": {
                    **{name: params[name] for name in PARAM_ORDER if name in params},
                    "ef": metrics.get("selected_ef", params.get("ef")),
                },
                "recall": recall,
                "qps": qps,
                "constraint": recall - recall_threshold,
            }
        )
    if not success_rows:
        return None, ""

    feasible = [row for row in success_rows if row["constraint"] >= 0.0]
    if feasible:
        best = max(feasible, key=lambda row: (row["qps"], row["recall"]))
        return best["params"], "best_feasible"

    near_feasible = [row for row in success_rows if row["constraint"] >= -near_width]
    if near_feasible:
        best = max(near_feasible, key=lambda row: (row["constraint"], row["qps"]))
        return best["params"], "near_feasible"

    best = max(success_rows, key=lambda row: row["qps"])
    return best["params"], "best_qps"


def _build_transfer_target_profile(
    *,
    round_trials: Sequence[Dict[str, Any]],
    recall_threshold: float,
    benchmark_extra_args: Sequence[Any],
    trials_name: str,
) -> Dict[str, Any]:
    features, actual_qps = _extract_round_success_observations(round_trials)
    data_path = _extra_arg_value(benchmark_extra_args, "--data-path")
    space = _extra_arg_value(benchmark_extra_args, "--space")
    hint_tokens = _tokenize_match_text(" ".join([trials_name, Path(data_path).stem, Path(data_path).name, space]))
    anchor_params, anchor_source = _current_anchor_params(round_trials, recall_threshold)
    return {
        "feature_rows": features,
        "actual_qps": actual_qps,
        "success_count": len(actual_qps),
        "data_path": data_path,
        "data_stem": Path(data_path).stem if data_path else "",
        "space": space,
        "trials_name": trials_name,
        "hint_tokens": _unique_tokens(hint_tokens),
        "anchor_params": copy.deepcopy(anchor_params) if isinstance(anchor_params, dict) else None,
        "anchor_source": anchor_source,
    }


def _score_transfer_card(
    *,
    card: Dict[str, Any],
    recall_threshold: float,
    target_profile: Dict[str, Any],
) -> Dict[str, Any]:
    feasibility = card.get("feasibility") or {}
    perf = card.get("perf") or {}
    direction = card.get("direction") or {}
    reasoning = card.get("reasoning") or {}
    feasible_center = direction.get("feasible_median") or {}
    elite_center = direction.get("elite_median") or {}
    best_params = perf.get("best_params") or {}
    reasoning_focus = reasoning.get("focus_params") or {}
    if isinstance(reasoning_focus, dict) and reasoning_focus:
        focus_params = reasoning_focus
    elif isinstance(best_params, dict) and best_params:
        focus_params = best_params
    else:
        focus_params = elite_center if isinstance(elite_center, dict) and elite_center else feasible_center
    threshold_gap = abs(float(feasibility.get("recall_threshold", recall_threshold)) - recall_threshold)
    threshold_score = _clamp01(math.exp(-18.0 * threshold_gap))
    anchor_similarity = _param_profile_similarity(target_profile.get("anchor_params"), focus_params)
    quality_score = _clamp01(float((card.get("quality") or {}).get("score", 0.0)))
    feasible_ratio = _clamp01(float(feasibility.get("feasible_ratio", 0.0)))
    margin_score = _recall_margin_score(float(feasibility.get("median_recall_margin", 0.0)))
    elite_score = _elite_gain_score(
        float(perf.get("median_feasible_qps", 0.0)),
        float(perf.get("median_elite_qps", 0.0)),
    )
    best_qps_score = _clamp01(math.log1p(max(0.0, _perf_best_qps(perf))) / math.log1p(1000.0))
    support_score = _clamp01(min(1.0, float(feasibility.get("covered_count", 0)) / 20.0))
    specificity_score = _clamp01(float(reasoning.get("specificity_score", 0.0) or 0.0))
    actionability_score = _clamp01(float(reasoning.get("actionability_score", 0.0) or 0.0))
    boundary_score = _clamp01(float(reasoning.get("boundary_proximity_score", 0.0) or 0.0))
    broad_penalty = 0.72 if bool(reasoning.get("broad_card", False)) else 1.0

    components = {
        "quality_score": quality_score,
        "feasible_ratio_score": feasible_ratio,
        "margin_score": margin_score,
        "elite_gain_score": elite_score,
        "best_qps_score": best_qps_score,
        "threshold_score": threshold_score,
        "support_score": support_score,
        "specificity_score": specificity_score,
        "actionability_score": actionability_score,
        "boundary_score": boundary_score,
    }
    weights = {
        "quality_score": 0.14,
        "feasible_ratio_score": 0.12,
        "margin_score": 0.10,
        "elite_gain_score": 0.10,
        "best_qps_score": 0.12,
        "threshold_score": 0.10,
        "support_score": 0.06,
        "specificity_score": 0.14,
        "actionability_score": 0.18,
        "boundary_score": 0.08,
    }
    if anchor_similarity is not None:
        components["anchor_similarity"] = anchor_similarity
        weights["anchor_similarity"] = 0.14
    total_weight = sum(weights.values())
    score = sum(weights[name] * components[name] for name in weights) / max(1e-9, total_weight)
    score *= broad_penalty
    components["broad_penalty"] = broad_penalty
    return {
        "score": float(score),
        "components": components,
        "focus_params": copy.deepcopy(focus_params) if isinstance(focus_params, dict) else {},
    }


def _select_transfer_cards(
    *,
    insight_payload: Dict[str, Any],
    recall_threshold: float,
    target_profile: Dict[str, Any],
    max_cards: int = 2,
) -> List[Dict[str, Any]]:
    cards = insight_payload.get("insight_cards")
    if not isinstance(cards, list) or max_cards <= 0:
        return []
    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_score = _score_transfer_card(
            card=card,
            recall_threshold=recall_threshold,
            target_profile=target_profile,
        )
        ranked.append((float(card_score["score"]), card, card_score))
    ranked.sort(
        key=lambda item: (
            item[0],
            float(((item[1].get("quality") or {}).get("score", 0.0)) or 0.0),
            float(((item[1].get("feasibility") or {}).get("feasible_ratio", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    for score, card, card_score in ranked[:max_cards]:
        compact = _compact_card(card)
        compact["focus_params"] = card_score["focus_params"]
        compact["transfer_card_score"] = float(score)
        compact["transfer_card_components"] = card_score["components"]
        selected.append(compact)
    return selected


def _task_evidence_score(
    *,
    sample_count: int,
    feasible_pool_count: int,
    selected_card_count: int,
) -> float:
    sample_score = min(1.0, math.log1p(max(0, sample_count)) / math.log1p(200.0))
    feasible_score = min(1.0, math.log1p(max(0, feasible_pool_count)) / math.log1p(60.0))
    card_score = min(1.0, max(0, selected_card_count) / 3.0)
    return float(0.4 * sample_score + 0.4 * feasible_score + 0.2 * card_score)


def _score_transfer_task(
    *,
    target_profile: Dict[str, Any],
    task_name: str,
    task_tokens: Sequence[str],
    recall_threshold: float,
    sample_count: int,
    feasible_pool_count: int,
    model_similarity: float | None,
    insight_payload: Dict[str, Any],
    selected_cards: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    lexical_score = _task_match_score(" ".join(task_tokens) or task_name, target_profile.get("hint_tokens", []))
    threshold_match = _threshold_match_info(recall_threshold, _insight_thresholds(insight_payload))
    insight_scores = [float(card.get("transfer_card_score", 0.0)) for card in selected_cards if isinstance(card, dict)]
    insight_score = sum(insight_scores) / len(insight_scores) if insight_scores else 0.0
    evidence_score = _task_evidence_score(
        sample_count=sample_count,
        feasible_pool_count=feasible_pool_count,
        selected_card_count=len(selected_cards),
    )
    anchor_similarity = None
    if isinstance(target_profile.get("anchor_params"), dict):
        candidate_scores = [
            _param_profile_similarity(target_profile["anchor_params"], card.get("focus_params"))
            for card in selected_cards
            if isinstance(card, dict)
        ]
        valid_scores = [score for score in candidate_scores if score is not None]
        if valid_scores:
            anchor_similarity = max(valid_scores)

    component_scores = {
        "lexical_similarity": _clamp01(lexical_score),
        "threshold_compatibility": _clamp01(float(threshold_match.get("score", 0.0))),
        "subgroup_quality": _clamp01(insight_score),
        "evidence_strength": _clamp01(evidence_score),
    }
    if model_similarity is not None:
        component_scores["behavior_similarity"] = _clamp01(model_similarity)
    if anchor_similarity is not None:
        component_scores["anchor_similarity"] = _clamp01(anchor_similarity)

    if model_similarity is not None or anchor_similarity is not None:
        weights = {
            "behavior_similarity": 0.38,
            "anchor_similarity": 0.18,
            "threshold_compatibility": 0.16,
            "subgroup_quality": 0.16,
            "lexical_similarity": 0.07,
            "evidence_strength": 0.05,
        }
    else:
        weights = {
            "lexical_similarity": 0.35,
            "threshold_compatibility": 0.25,
            "subgroup_quality": 0.25,
            "evidence_strength": 0.15,
        }

    active_weights = {name: weight for name, weight in weights.items() if name in component_scores}
    total_weight = sum(active_weights.values()) or 1.0
    transfer_score = sum(active_weights[name] * component_scores[name] for name in active_weights) / total_weight
    return (
        float(transfer_score),
        {
            "weights": active_weights,
            "scores": component_scores,
            "threshold_match": threshold_match,
        },
        threshold_match,
    )


def _build_historical_transfer_context(
    *,
    round_trials: Sequence[Dict[str, Any]],
    models_dir: str | Path,
    insights_dir: str | Path | None,
    recall_threshold: float,
    top_k_tasks: int,
    trials_per_task: int,
    previous_context: Dict[str, Any] | None,
    benchmark_extra_args: Sequence[Any],
    trials_name: str,
    bootstrap: bool,
    min_match_score: float,
) -> Dict[str, Any]:
    previous = previous_context or {"tasks": []}
    resolved_models_dir = Path(models_dir).expanduser().resolve()
    if not resolved_models_dir.exists() or not resolved_models_dir.is_dir():
        return {"tasks": []} if bootstrap else previous

    resolved_insights_dir = None
    if insights_dir:
        candidate_dir = Path(insights_dir).expanduser().resolve()
        if candidate_dir.exists() and candidate_dir.is_dir():
            resolved_insights_dir = candidate_dir

    target_profile = _build_transfer_target_profile(
        round_trials=round_trials,
        recall_threshold=recall_threshold,
        benchmark_extra_args=benchmark_extra_args,
        trials_name=trials_name,
    )
    has_behavior_signal = len(target_profile["actual_qps"]) >= 2
    has_anchor_signal = isinstance(target_profile.get("anchor_params"), dict)
    has_hint_signal = bool(target_profile.get("hint_tokens"))
    if not (has_behavior_signal or has_anchor_signal or has_hint_signal):
        return {"tasks": []} if bootstrap else previous

    scored_tasks: List[Dict[str, Any]] = []
    for meta_path in sorted(resolved_models_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("domain") != HNSWLIB_TRANSFER_DOMAIN:
            continue
        if meta.get("param_order") != PARAM_ORDER:
            continue

        fallback_model_name = meta_path.name[:-10] if meta_path.name.endswith(".meta.json") else meta_path.stem
        model_name = str(meta.get("model_name") or fallback_model_name).strip()
        task_name = str(meta.get("task_name") or model_name).strip()
        if not model_name or not task_name:
            continue
        linked_raw = meta.get("linked_trials_path")
        if not isinstance(linked_raw, str) or not linked_raw.strip():
            continue
        linked_path = _resolve_linked_trials_path(linked_raw, resolved_models_dir)
        if not linked_path.exists():
            continue

        task_tokens = _unique_tokens(
            _tokenize_match_text(
                " ".join(
                    [
                        task_name,
                        model_name,
                        linked_path.stem,
                        str(meta.get("dataset_name", "")),
                        str(meta.get("space", "")),
                    ]
                )
            )
        )
        lexical_score = _task_match_score(" ".join(task_tokens) or task_name, target_profile.get("hint_tokens", []))
        if bootstrap and lexical_score < min_match_score:
            continue

        qualified_trials = _load_recall_qualified_trials(linked_path, recall_threshold)
        if not qualified_trials:
            continue

        insight_payload, insight_path = _load_insight_payload_for_task(
            insights_dir=resolved_insights_dir,
            task_name=task_name,
        )
        selected_cards = _select_transfer_cards(
            insight_payload=insight_payload,
            recall_threshold=recall_threshold,
            target_profile=target_profile,
            max_cards=2,
        )

        model_similarity = None
        if has_behavior_signal:
            model_path = resolved_models_dir / f"{model_name}.pkl"
            if model_path.exists():
                try:
                    with model_path.open("rb") as f:
                        model = pickle.load(f)
                    raw_similarity = _model_similarity(
                        target_profile["actual_qps"],
                        target_profile["feature_rows"],
                        model,
                    )
                    model_similarity = max(0.0, raw_similarity)
                except Exception:
                    model_similarity = None

        transfer_score, score_breakdown, threshold_match = _score_transfer_task(
            target_profile=target_profile,
            task_name=task_name,
            task_tokens=task_tokens,
            recall_threshold=recall_threshold,
            sample_count=int(meta.get("sample_count", 0) or 0),
            feasible_pool_count=len(qualified_trials),
            model_similarity=model_similarity,
            insight_payload=insight_payload,
            selected_cards=selected_cards,
        )
        selected_trials = _select_transfer_trials(
            qualified_trials=qualified_trials,
            selected_cards=selected_cards,
            trials_per_task=trials_per_task,
            bootstrap=bootstrap,
        )
        if not selected_trials:
            continue

        scored_tasks.append(
            {
                "model_name": model_name,
                "task_name": task_name,
                "similarity": float(transfer_score),
                "transfer_score": float(transfer_score),
                "score_breakdown": score_breakdown,
                "threshold_match": threshold_match,
                "linked_trials_path": str(linked_path),
                "insight_path": insight_path,
                "task_tokens": task_tokens,
                "qualified_trial_count": len(qualified_trials),
                "selected_trials": selected_trials,
                "selected_card_ids": [str(card.get("card_id", "")) for card in selected_cards if card.get("card_id")],
                "selected_cards": copy.deepcopy(selected_cards),
            }
        )

    scored_tasks.sort(
        key=lambda item: (
            float(item.get("transfer_score", item.get("similarity", 0.0))),
            max((float(card.get("transfer_card_score", 0.0)) for card in item.get("selected_cards", [])), default=0.0),
            item["selected_trials"][0]["qps"] if item.get("selected_trials") else 0.0,
        ),
        reverse=True,
    )
    top_tasks = scored_tasks[: max(0, int(top_k_tasks))]
    if not top_tasks:
        return {"tasks": []} if bootstrap else previous
    return {
        "generated_at": utc_now_iso(),
        "source_success_count": int(target_profile.get("success_count", 0)),
        "bootstrap": bootstrap,
        "bootstrap_hint_tokens": list(target_profile.get("hint_tokens", [])) if bootstrap else [],
        "target_profile": {
            "data_stem": str(target_profile.get("data_stem", "")),
            "space": str(target_profile.get("space", "")),
            "trials_name": str(target_profile.get("trials_name", "")),
            "hint_tokens": list(target_profile.get("hint_tokens", [])),
            "anchor_source": str(target_profile.get("anchor_source", "")),
            "anchor_params": copy.deepcopy(target_profile.get("anchor_params")),
        },
        "tasks": top_tasks,
    }


def _build_transfer_context(
    round_trials: Sequence[Dict[str, Any]],
    models_dir: str | Path,
    recall_threshold: float,
    top_k_tasks: int,
    trials_per_task: int,
    previous_context: Dict[str, Any] | None,
    benchmark_extra_args: Sequence[Any] = (),
    trials_name: str = "",
    insights_dir: str | Path | None = None,
) -> Dict[str, Any]:
    return _build_historical_transfer_context(
        round_trials=round_trials,
        models_dir=models_dir,
        insights_dir=insights_dir,
        recall_threshold=recall_threshold,
        top_k_tasks=top_k_tasks,
        trials_per_task=trials_per_task,
        previous_context=previous_context,
        benchmark_extra_args=benchmark_extra_args,
        trials_name=trials_name,
        bootstrap=False,
        min_match_score=0.0,
    )


def _extra_arg_value(extra_args: Sequence[Any], flag: str) -> str:
    items = [str(item) for item in extra_args]
    for idx, item in enumerate(items):
        if item == flag and idx + 1 < len(items):
            return items[idx + 1]
    return ""


def _build_bootstrap_transfer_context(
    *,
    models_dir: str | Path,
    recall_threshold: float,
    top_k_tasks: int,
    trials_per_task: int,
    benchmark_extra_args: Sequence[Any],
    trials_name: str,
    min_match_score: float,
    insights_dir: str | Path | None = None,
) -> Dict[str, Any]:
    return _build_historical_transfer_context(
        round_trials=[],
        models_dir=models_dir,
        insights_dir=insights_dir,
        recall_threshold=recall_threshold,
        top_k_tasks=top_k_tasks,
        trials_per_task=trials_per_task,
        previous_context={"tasks": []},
        benchmark_extra_args=benchmark_extra_args,
        trials_name=trials_name,
        bootstrap=True,
        min_match_score=min_match_score,
    )


def _extract_used_similar_task_names(transfer_context: Dict[str, Any]) -> List[str]:
    tasks = transfer_context.get("tasks") if isinstance(transfer_context, dict) else []
    names: List[str] = []
    seen: set[str] = set()
    if not isinstance(tasks, list):
        return names
    for item in tasks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("task_name") or item.get("model_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _transfer_seed_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    transfer_context: Dict[str, Any] | None,
    allowed_values_override: Dict[str, Any] | None,
    max_count: int,
) -> List[Dict[str, Any]]:
    tasks = transfer_context.get("tasks") if isinstance(transfer_context, dict) else []
    if not isinstance(tasks, list) or max_count <= 0:
        return []
    seeds: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_name = str(task.get("task_name") or task.get("model_name") or "").strip()
        selected_trials = task.get("selected_trials")
        if not isinstance(selected_trials, list):
            continue
        for trial in selected_trials:
            params = trial.get("params")
            seed = _canonical_seed_candidate(
                agent,
                params=params if isinstance(params, dict) else {},
                allowed_values_override=allowed_values_override,
                source="transfer_seed",
                note=f"transfer seed from {task_name}",
            )
            if seed is None:
                continue
            key = _build_key_from_params(seed["params"])
            if key in seen:
                continue
            seen.add(key)
            seeds.append(seed)
            if len(seeds) >= max_count:
                return seeds
    return seeds


def _normalize_task_stem(task_name: Any) -> str:
    text = str(task_name or "").strip()
    if not text:
        return ""
    if text.endswith(".jsonl"):
        return text[: -len(".jsonl")]
    if text.endswith(".json"):
        return text[: -len(".json")]
    return text


def _resolve_similar_task_top_k(value: Any, fallback_top_k: int) -> int:
    text = str(value if value is not None else "auto").strip().lower()
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


def _perf_best_qps(perf: Dict[str, Any] | None) -> float:
    payload = perf if isinstance(perf, dict) else {}
    for key in ("best_qps", "best_elite_qps", "median_elite_qps", "median_feasible_qps"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _legacy_tendency_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "flat"
    if "↓" in text or "down" in text or "decrease" in text:
        return "down"
    if "↑" in text or "up" in text or "increase" in text:
        return "up"
    return "flat"


def _legacy_direction_maps(direction: Dict[str, Any], fallback_focus: Dict[str, Any]) -> Dict[str, Any]:
    tendency = copy.deepcopy(direction.get("tendency") or {})
    delta = copy.deepcopy(direction.get("delta") or {})
    if not isinstance(tendency, dict):
        tendency = {}
    if not isinstance(delta, dict):
        delta = {}

    for name in PARAM_ORDER:
        if name not in tendency:
            tendency[name] = _legacy_tendency_value(direction.get(name))
        if name not in delta:
            tendency_value = str(tendency.get(name, "flat")).strip().lower()
            if tendency_value == "down":
                delta[name] = -1.0
            elif tendency_value == "up":
                delta[name] = 1.0
            else:
                delta[name] = 0.0

    elite = direction.get("elite_median") if isinstance(direction.get("elite_median"), dict) else {}
    feasible = direction.get("feasible_median") if isinstance(direction.get("feasible_median"), dict) else {}
    if not elite:
        elite = copy.deepcopy(fallback_focus)
    if not feasible:
        feasible = copy.deepcopy(elite)
    return {
        "tendency": tendency,
        "delta": delta,
        "elite_median": elite,
        "feasible_median": feasible,
    }


def _coerce_quality_score(value: Any) -> float:
    candidate = value.get("score", 0.0) if isinstance(value, dict) else value
    try:
        return float(candidate or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recommended_interval_dimensions(card: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    region = card.get("region") if isinstance(card.get("region"), dict) else {}
    dimensions = region.get("dimensions") if isinstance(region.get("dimensions"), list) else []
    if dimensions:
        return copy.deepcopy(dimensions), str(region.get("description", "") or "")

    interval_source = card.get("recommended_interval")
    if not isinstance(interval_source, dict):
        interval_source = card.get("coarse_interval") if isinstance(card.get("coarse_interval"), dict) else {}

    built_dimensions: List[Dict[str, Any]] = []
    description_parts: List[str] = []
    for name in PARAM_ORDER:
        bounds = interval_source.get(name)
        if not isinstance(bounds, list) or len(bounds) < 2:
            continue
        try:
            lower_value = float(bounds[0])
            upper_value = float(bounds[1])
        except (TypeError, ValueError):
            continue
        if upper_value < lower_value:
            lower_value, upper_value = upper_value, lower_value
        lower = int(lower_value) if lower_value.is_integer() else lower_value
        upper = int(upper_value) if upper_value.is_integer() else upper_value
        built_dimensions.append(
            {
                "parameter": name,
                "interval": {
                    "lower": lower,
                    "upper": upper,
                    "lower_inclusive": True,
                    "upper_inclusive": True,
                },
            }
        )
        description_parts.append(f"{name} in [{lower}, {upper}]")
    return built_dimensions, ", ".join(description_parts)


def _load_explicit_bootstrap_insights(
    knowledge_cfg: Dict[str, Any],
    *,
    recall_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_entries = knowledge_cfg.get("bootstrap_insight_paths")
    if raw_entries is None:
        raw_entries = knowledge_cfg.get("explicit_insight_paths")
    if not isinstance(raw_entries, list):
        return [], []

    loaded: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for raw in raw_entries:
        if isinstance(raw, str):
            path_text = raw
            task_name_override = ""
            selected_card_ids: List[str] = []
            transfer_score = 1.25
            threshold_match_mode = "exact"
        elif isinstance(raw, dict):
            path_text = str(raw.get("path", "")).strip()
            task_name_override = str(raw.get("task_name", "")).strip()
            selected_card_ids = [str(item) for item in (raw.get("selected_card_ids") or []) if str(item).strip()]
            transfer_score = float(raw.get("transfer_score", 1.25) or 1.25)
            threshold_match_mode = str(raw.get("threshold_match", "exact")).strip().lower()
        else:
            continue
        if not path_text:
            continue
        path = Path(path_text).expanduser().resolve()
        if not path.exists():
            missing.append({"path": str(path), "reason": "missing_file"})
            continue
        payload = _normalize_insight_payload(
            _load_json_object(path),
            fallback_task_name=task_name_override or path.stem,
            source_path=str(path),
        )
        if not payload:
            missing.append({"path": str(path), "reason": "invalid_or_empty_json"})
            continue
        task_name = str(payload.get("task_name") or task_name_override or path.stem).strip()
        if not selected_card_ids:
            selected_card_ids = [
                str(card.get("card_id", ""))
                for card in (payload.get("insight_cards") or [])
                if isinstance(card, dict) and str(card.get("card_id", "")).strip()
            ]
        loaded.append(
            {
                "task_name": task_name,
                "similarity": float(transfer_score),
                "transfer_score": float(transfer_score),
                "score_breakdown": {"source": "explicit_bootstrap_insight", "path": str(path)},
                "selected_card_ids": selected_card_ids,
                "selected_cards": [],
                "threshold_match": _threshold_match_info(recall_threshold, _insight_thresholds(payload)),
                "threshold_match_mode": threshold_match_mode,
                "model_name": task_name,
                "linked_trials_path": "",
                "insight_path": str(path),
                "insight": payload,
            }
        )
    return loaded, missing


def _load_similar_task_insights(
    transfer_context: Dict[str, Any] | None,
    insights_dir: Path,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = transfer_context.get("tasks") if isinstance(transfer_context, dict) else []
    if not isinstance(tasks, list):
        tasks = []
    normalized = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_name = _normalize_task_stem(item.get("task_name") or item.get("model_name"))
        if task_name:
            normalized.append(
                {
                    "task_name": task_name,
                    "similarity": float(item.get("similarity", 0.0)),
                    "transfer_score": float(item.get("transfer_score", item.get("similarity", 0.0))),
                    "score_breakdown": copy.deepcopy(item.get("score_breakdown") or {}),
                    "selected_card_ids": list(item.get("selected_card_ids") or []),
                    "selected_cards": copy.deepcopy(item.get("selected_cards") or []),
                    "threshold_match": copy.deepcopy(item.get("threshold_match") or {}),
                    "model_name": str(item.get("model_name", "")),
                    "linked_trials_path": str(item.get("linked_trials_path", "")),
                }
            )
    normalized.sort(key=lambda row: row["transfer_score"], reverse=True)
    loaded: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for row in normalized[: max(0, int(top_k))]:
        insight_path = insights_dir / f"{row['task_name']}.json"
        if not insight_path.exists():
            missing.append({"task_name": row["task_name"], "reason": "missing_file", "path": str(insight_path)})
            continue
        payload = _normalize_insight_payload(
            _load_json_object(insight_path),
            fallback_task_name=row["task_name"],
            source_path=str(insight_path),
        )
        if not payload:
            missing.append({"task_name": row["task_name"], "reason": "invalid_or_empty_json", "path": str(insight_path)})
            continue
        loaded.append(
            {
                "task_name": row["task_name"],
                "similarity": row["similarity"],
                "transfer_score": row["transfer_score"],
                "score_breakdown": row["score_breakdown"],
                "selected_card_ids": row["selected_card_ids"],
                "selected_cards": row["selected_cards"],
                "threshold_match": row["threshold_match"],
                "model_name": row["model_name"],
                "linked_trials_path": row["linked_trials_path"],
                "insight_path": str(insight_path),
                "insight": payload,
            }
        )
    return loaded, missing


def build_knowledge_context(
    *,
    base_knowledge_full: Dict[str, Any] | str,
    transfer_context: Dict[str, Any] | None = None,
    insights_dir: Path | None = None,
    similar_task_top_k: int = 0,
    max_context_chars: int,
    overflow_policy: str,
    seed_task_insights: Sequence[Dict[str, Any]] | None = None,
    mode: str = "knowledge_base_driven",
) -> Tuple[Any, Dict[str, Any]]:
    """Build knowledge context for HNSW LLM consumption.

    Supports two modes:
    - ``knowledge_base_driven``: ``base_knowledge_full`` is a pre-formatted markdown
      string.  Returns ``(str, report_dict)``.
    - ``full_context_no_rag`` (legacy): ``base_knowledge_full`` is a dict loaded from
      a compiled JSON knowledge file.  Similar-task subgroup insight cards and
      seed-task insights are merged in.  Returns ``(dict, report_dict)``.
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
        "mode": "hnsw_full_context_no_rag",
        "base_knowledge_full": copy.deepcopy(base_knowledge_full) if isinstance(base_knowledge_full, dict) else {},
        "similar_task_insights_full": [],
    }
    loaded: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    if insights_dir is not None and transfer_context is not None and similar_task_top_k > 0:
        loaded, missing = _load_similar_task_insights(transfer_context, insights_dir, similar_task_top_k)
    merged: List[Dict[str, Any]] = []
    seen_task_names: set[str] = set()
    for item in list(seed_task_insights or []) + loaded:
        if not isinstance(item, dict):
            continue
        task_name = str(item.get("task_name", "")).strip()
        dedupe_key = task_name or str(item.get("insight_path", "")).strip()
        if dedupe_key and dedupe_key in seen_task_names:
            continue
        if dedupe_key:
            seen_task_names.add(dedupe_key)
        merged.append(copy.deepcopy(item))
    context["similar_task_insights_full"] = merged
    initial_chars = _json_char_length(context)
    report = {
        "mode": "hnsw_full_context_no_rag",
        "overflow_policy": overflow_policy,
        "max_context_chars": int(max_context_chars),
        "initial_chars": int(initial_chars),
        "final_chars": int(initial_chars),
        "truncated": False,
        "within_limit": initial_chars <= max_context_chars,
        "missing_insight_tasks": missing,
        "removed_similar_task_insights": [],
    }
    if initial_chars <= max_context_chars or overflow_policy != "truncate":
        return context, report
    removed: List[str] = []
    while len(context["similar_task_insights_full"]) > 1 and _json_char_length(context) > max_context_chars:
        item = context["similar_task_insights_full"].pop()
        removed.append(str(item.get("task_name", "")))
    final_chars = _json_char_length(context)
    report["final_chars"] = int(final_chars)
    report["truncated"] = final_chars < initial_chars
    report["within_limit"] = final_chars <= max_context_chars
    report["removed_similar_task_insights"] = removed
    return context, report


def _history_only_override(
    agent: HNSWLIBTuningAgent,
    *,
    knowledge_context: Dict[str, Any] | None,
    stage_policy: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any], List[Dict[str, Any]]]:
    insight_candidates = _insight_card_candidates(
        agent,
        knowledge_context=knowledge_context,
        recall_threshold=float(stage_policy["recall_threshold"]),
    )
    deterministic_override: Dict[str, Any] | None = None
    deterministic_reason = ""
    selected_candidate_ids: List[str] = []
    primary_seed_params: Dict[str, Any] = {}
    if insight_candidates:
        selected_candidates = insight_candidates[: min(3, len(insight_candidates))]
        primary_candidate = max(
            selected_candidates,
            key=lambda item: _perf_best_qps(
                (((item.get("cards_used") or [{}])[0].get("card") or {}).get("perf") or {})
            ),
        )
        selected_candidates = [primary_candidate] + [item for item in selected_candidates if item is not primary_candidate]
        selected_candidate_ids = [str(item.get("candidate_id", "")) for item in selected_candidates]
        deterministic_override = copy.deepcopy(selected_candidates[0].get("override"))
        primary_seed_params = copy.deepcopy(selected_candidates[0].get("seed_params") or {})
        deterministic_reason = "historical_best_feasible_qps_card"
        for candidate in selected_candidates[1:]:
            candidate_override = candidate.get("override")
            intersected = _intersect_overrides(agent, deterministic_override, candidate_override)
            if intersected is not None and _override_volume(agent, intersected) > 0.0:
                if primary_seed_params and agent.space.out_of_constraint_fields(primary_seed_params, intersected):
                    continue
                deterministic_override = intersected
                deterministic_reason = "historical_insight_intersection"

    if deterministic_override is None:
        return None, {
            "enabled": False,
            "reason": "no_historical_insight_candidates",
            "selected_candidate_ids": [],
            "insight_candidates": [],
            "llm_refinement": {"enabled": False, "reason": "no_historical_candidates"},
        }, []

    root_state = agent.build_root_state(
        round_idx=0,
        stage_trials=[],
        stage_policy=stage_policy,
    )
    llm_override, llm_report = _llm_refine_override(
        agent,
        stage_policy=stage_policy,
        root_state=root_state,
        frontier_override=None,
        candidate_overrides=insight_candidates[:3],
        deterministic_override=deterministic_override,
    )
    if llm_override is not None and primary_seed_params and agent.space.out_of_constraint_fields(primary_seed_params, llm_override):
        llm_override = None
    final_override = llm_override or deterministic_override
    final_reason = "llm_refined" if llm_override is not None else deterministic_reason
    return final_override, {
        "enabled": True,
        "reason": final_reason,
        "selected_candidate_ids": selected_candidate_ids,
        "insight_candidates": insight_candidates,
        "llm_refinement": llm_report,
    }, insight_candidates


def _history_only_seed_candidates(
    agent: HNSWLIBTuningAgent,
    *,
    transfer_context: Dict[str, Any] | None,
    allowed_values_override: Dict[str, Any],
    insight_candidates: Sequence[Dict[str, Any]],
    refinement_seed_cfg: Dict[str, Any],
    recall_threshold: float,
) -> List[Dict[str, Any]]:
    transfer_seed_candidates = _transfer_seed_candidates(
        agent,
        transfer_context=transfer_context,
        allowed_values_override=allowed_values_override,
        max_count=int(refinement_seed_cfg.get("max_transfer_trials", 2)),
    )
    refinement_seed_candidates = _refinement_seed_candidates(
        agent,
        candidates=insight_candidates,
        allowed_values_override=allowed_values_override,
        max_count=max(0, int(refinement_seed_cfg.get("max_focus_points", 2))),
        recall_threshold=recall_threshold,
    )
    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for candidate in [*transfer_seed_candidates, *refinement_seed_candidates]:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("params"), dict):
            continue
        key = _build_key_from_params(candidate["params"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(copy.deepcopy(candidate))
    return merged


def _stage_a_plan_hnsw(
    *,
    agent: HNSWLIBTuningAgent,
    stage_policy: Dict[str, Any],
    base_knowledge_full: Dict[str, Any] | str,
    transfer_enabled: bool,
    transfer_models_dir: Path,
    transfer_top_k_tasks: int,
    transfer_trials_per_task: int,
    knowledge_cfg: Dict[str, Any],
    insights_dir: Path,
    benchmark_extra_args: Sequence[Any],
    trials_name: str,
    refinement_cfg: Dict[str, Any],
    refinement_bootstrap_cfg: Dict[str, Any],
    refinement_seed_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    base_search_space = agent.space.export_parameter_space()
    knowledge_mode = str(knowledge_cfg.get("mode", "knowledge_base_driven")).strip().lower()
    bootstrap_enabled = bool(refinement_bootstrap_cfg.get("enabled", True))
    similar_task_top_k = _resolve_similar_task_top_k(
        knowledge_cfg.get("similar_task_top_k", "auto"),
        fallback_top_k=transfer_top_k_tasks,
    )
    max_context_chars = int(knowledge_cfg.get("max_context_chars", 120000))
    overflow_policy = str(knowledge_cfg.get("overflow_policy", "truncate")).strip().lower()

    explicit_task_insights: List[Dict[str, Any]] = []
    explicit_missing: List[Dict[str, Any]] = []
    cold_start_cards: List[Dict[str, Any]] = []
    if knowledge_mode == "full_context_no_rag":
        explicit_task_insights, explicit_missing = _load_explicit_bootstrap_insights(
            knowledge_cfg,
            recall_threshold=float(stage_policy["recall_threshold"]),
        )
        # ── Filter insight cards by per-entry threshold_match mode ──
        # "exact" (default): keep only cards mined at the current τ.
        # "le": keep cards mined at τ' <= current τ — lower-τ cards are
        #   conservative priors for a higher-τ run.
        # "all": no threshold filtering.
        stage_tau = float(stage_policy["recall_threshold"])
        for item in explicit_task_insights:
            insight = item.get("insight") if isinstance(item.get("insight"), dict) else {}
            all_cards = insight.get("insight_cards") if isinstance(insight.get("insight_cards"), list) else []
            mode = str(item.get("threshold_match_mode", "exact")).strip().lower()
            if mode == "le":
                def _keep(card_tau: float) -> bool:
                    return card_tau <= stage_tau + 1e-6
            elif mode == "all":
                def _keep(card_tau: float) -> bool:
                    return True
            else:  # "exact"
                def _keep(card_tau: float) -> bool:
                    return abs(card_tau - stage_tau) < 1e-6
            if all_cards:
                filtered = [
                    c for c in all_cards
                    if isinstance(c, dict)
                    and _keep(float((c.get("feasibility") or {}).get("recall_threshold", 0.0)))
                ]
                insight["insight_cards"] = filtered
                item["selected_card_ids"] = [str(c.get("card_id", "")) for c in filtered if c.get("card_id")]
            else:
                insight["insight_cards"] = []
        # ── Extract region info for cold_start guidance ──
        cold_start_cards: List[Dict[str, Any]] = []
        for item in explicit_task_insights:
            insight = item.get("insight") if isinstance(item.get("insight"), dict) else {}
            for card in (insight.get("insight_cards") or []):
                region = card.get("region") if isinstance(card.get("region"), dict) else {}
                dimensions = region.get("dimensions") if isinstance(region.get("dimensions"), list) else []
                direction = card.get("direction") if isinstance(card.get("direction"), dict) else {}
                elite = direction.get("elite_median") if isinstance(direction.get("elite_median"), dict) else {}
                feasible = direction.get("feasible_median") if isinstance(direction.get("feasible_median"), dict) else {}
                perf = card.get("perf") if isinstance(card.get("perf"), dict) else {}
                cold_start_cards.append({
                    "card_id": str(card.get("card_id", "")),
                    "region_desc": str(region.get("description", "")),
                    "dimensions": copy.deepcopy(dimensions),
                    "elite_median": copy.deepcopy(elite) if elite else {},
                    "feasible_median": copy.deepcopy(feasible) if feasible else {},
                    "best_params": copy.deepcopy(perf.get("best_params") or {}),
                    "best_qps": float(perf.get("best_qps", 0) or 0),
                })
    transfer_context = {"tasks": []}
    if transfer_enabled and bootstrap_enabled:
        transfer_context = _build_bootstrap_transfer_context(
            models_dir=transfer_models_dir,
            recall_threshold=float(stage_policy["recall_threshold"]),
            top_k_tasks=int(refinement_bootstrap_cfg.get("top_k_tasks", min(2, transfer_top_k_tasks))),
            trials_per_task=transfer_trials_per_task,
            benchmark_extra_args=benchmark_extra_args,
            trials_name=trials_name,
            min_match_score=float(refinement_bootstrap_cfg.get("min_match_score", 0.2)),
            insights_dir=insights_dir,
        )
    knowledge_context, knowledge_report = build_knowledge_context(
        base_knowledge_full=base_knowledge_full,
        transfer_context=transfer_context,
        insights_dir=insights_dir,
        similar_task_top_k=similar_task_top_k,
        max_context_chars=max_context_chars,
        overflow_policy=overflow_policy,
        seed_task_insights=explicit_task_insights if explicit_task_insights else None,
        mode=knowledge_mode,
    )
    if explicit_missing:
        knowledge_report.setdefault("missing_insight_tasks", []).extend(explicit_missing)
    insight_candidates = _insight_card_candidates(
        agent,
        knowledge_context=knowledge_context,
        recall_threshold=float(stage_policy["recall_threshold"]),
    )
    initial_design_seed_candidates = _history_only_seed_candidates(
        agent,
        transfer_context=transfer_context,
        allowed_values_override=base_search_space,
        insight_candidates=insight_candidates,
        refinement_seed_cfg=refinement_seed_cfg,
        recall_threshold=float(stage_policy["recall_threshold"]),
    )
    # ── external seed file ──────────────────────────────────────────────
    external_seed_file = refinement_seed_cfg.get("external_seed_file", "").strip()
    if external_seed_file:
        ext_path = Path(external_seed_file)
        if not ext_path.is_absolute():
            ext_path = Path.cwd() / ext_path
        if ext_path.exists():
            try:
                ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
                ext_configs = ext_data if isinstance(ext_data, list) else ext_data.get("configurations", [])
                ext_seeds: List[Dict[str, Any]] = []
                for cfg in ext_configs:
                    seed = _canonical_seed_candidate(
                        agent,
                        params=cfg,
                        allowed_values_override=base_search_space,
                        source="external_seed_file",
                        note=f"LLM-proposed from {ext_path.name}",
                    )
                    if seed is not None:
                        ext_seeds.append(seed)
                if ext_seeds:
                    initial_design_seed_candidates = list(initial_design_seed_candidates)
                    initial_design_seed_candidates.extend(ext_seeds)
                    logging.getLogger("hnswlib.pipeline").info(
                        "Loaded %d external seed candidates from %s",
                        len(ext_seeds), str(ext_path),
                    )
            except Exception as exc:
                logging.getLogger("hnswlib.pipeline").warning(
                    "Failed to load external seed file %s: %s", str(ext_path), exc
                )
    used_similar_tasks = _extract_used_similar_task_names(transfer_context)
    for item in explicit_task_insights:
        task_name = str(item.get("task_name", "")).strip()
        if task_name and task_name not in used_similar_tasks:
            used_similar_tasks.append(task_name)
    used_insight_cards = [
        card.get("card", {})
        for candidate in insight_candidates
        for card in (candidate.get("cards_used") or [])
        if isinstance(card, dict)
    ]
    stage_a_report = {
        "space_frozen": False,
        "refinement_source": "cold_start_lsh_space_fill",
        "fallback_to_base_space": False,
        "reason": "online_initialization_no_history_freeze",
        "used_similar_tasks": used_similar_tasks,
        "used_insight_cards": used_insight_cards,
        "knowledge_truncation_report": knowledge_report,
        "history_override_report": {
            "enabled": bool(initial_design_seed_candidates),
            "reason": "history_used_as_optional_initial_design_anchors_only" if initial_design_seed_candidates else "no_history_seeds",
            "seed_candidate_count": len(initial_design_seed_candidates),
        },
        "execution_per_round": 1,
    }
    return {
        "generated_at": utc_now_iso(),
        "base_search_space": base_search_space,
        "frozen_search_space": copy.deepcopy(base_search_space),
        "transfer_context": transfer_context,
        "knowledge_context": knowledge_context,
        "stage_a_report": stage_a_report,
        "initial_design_seed_candidates": initial_design_seed_candidates,
        "cold_start_cards": cold_start_cards,
    }


def _build_best_so_far_curve(
    trials: Sequence[Dict[str, Any]],
    recall_threshold: float,
) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    best_feasible_qps: float | None = None
    for idx, trial in enumerate(trials, start=1):
        metrics = trial.get("metrics") or {}
        status = str(trial.get("status", ""))
        recall = None
        qps = None
        feasible = False
        if status == "success" and isinstance(metrics, dict):
            try:
                recall = float(metrics["recall"])
                qps = float(metrics["qps"])
                feasible = recall >= float(recall_threshold)
            except Exception:
                recall = None
                qps = None
                feasible = False
        if feasible and qps is not None:
            best_feasible_qps = qps if best_feasible_qps is None else max(best_feasible_qps, qps)
        curve.append(
            {
                "stage_b_round_idx": idx,
                "status": status,
                "recall": recall,
                "qps": qps,
                "feasible": feasible,
                "best_feasible_qps": best_feasible_qps,
            }
        )
    return curve


def _build_conditional_policy_text(
    *,
    all_trials: Sequence[Dict[str, Any]],
    last_metrics: Optional[Dict[str, Any]],
    recall_threshold: float,
    book: Dict[str, Any],
    cfg: "ConditionalPolicyRuntimeConfig",
    logger: logging.Logger,
) -> Tuple[str, Dict[str, Any]]:
    """Deterministic conditional-policy prompt block for one round (fail-open).

    Symptom <- last executed trial + this run's own successful trials.
    hnswlib has no expansion / traversal-effectiveness counters, so those
    symptom dimensions are wildcards; with no (or too little) history the
    full accepted set is injected instead.
    """
    log: Dict[str, Any] = {
        "enabled": True,
        "n_accepted_policies": len(book.get("policies") or []),
        "n_history_trials": 0,
        "matched_policy_ids": [],
        "fallback": False,
        "fallback_reason": "",
        "symptom": None,
        "symptom_detail": {},
        "chars": 0,
        "skipped_reason": "",
        "elapsed_s": 0.0,
    }
    t0 = time.perf_counter()
    try:
        if not book.get("policies"):
            log["skipped_reason"] = book.get("reason") or "no_accepted_policies"
            return "", log
        history_by_metric, recalls, n_history = build_conditional_policy_history(all_trials)
        log["n_history_trials"] = n_history
        symptom, detail = build_conditional_policy_symptom(
            last_metrics, history_by_metric, recalls, recall_threshold, cfg
        )
        log["symptom"] = symptom.to_dict() if symptom is not None else None
        log["symptom_detail"] = detail
        matches = (
            match_conditional_policies(symptom, book["policies"], cfg)
            if symptom is not None else []
        )
        log["matched_policy_ids"] = [m["policy"]["policy_id"] for m in matches]
        log["fallback"] = not matches
        log["fallback_reason"] = "" if matches else (
            "no_matching_policy" if detail.get("reason") == "ok"
            else (detail.get("reason") or "no_matching_policy")
        )
        text = format_conditional_policy_context(
            matches,
            policies=book["policies"],
            config=cfg,
            detail=detail,
            source=str((book.get("meta") or {}).get("source") or ""),
        )
        log["chars"] = len(text)
        return text, log
    except Exception as exc:  # fail-open: never break the tuning loop
        logger.warning("Conditional policy matching failed (ignored): %s", exc)
        log["skipped_reason"] = f"error: {exc}"
        return "", log
    finally:
        log["elapsed_s"] = round(time.perf_counter() - t0, 4)


def _configure_agent_logging(agentic_cfg: Dict[str, Any]) -> None:
    """Configure Python logging so LLM prompts and structured outputs are visible.

    Reads ``agentic.logging`` from the merged config.  Supported modes:

    * ``summary`` — only warnings and errors (quiet).
    * ``verbose`` — INFO level for the ``hnswlib_agent.llm`` logger so that
      full LLM prompts and parsed structured JSON outputs appear on stderr.
    * ``debug`` — DEBUG level; adds raw LLM completions to the verbose output.

    Also configures a ``hnswlib.pipeline`` logger at INFO+ so round-by-round
    progress is always visible in the nohup log.

    All log messages are written to stderr so they land in the nohup log when
    the pipeline is invoked with ``> test.log 2>&1``.
    """
    logging_cfg = agentic_cfg.get("logging") if isinstance(agentic_cfg, dict) else {}
    mode = str(logging_cfg.get("mode", "summary")).strip().lower()

    level_map: Dict[str, int] = {
        "summary": logging.WARNING,
        "verbose": logging.INFO,
        "debug": logging.DEBUG,
    }
    llm_level = level_map.get(mode, logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # --- LLM / agent logger (prompts + structured outputs) ---
    llm_handler = logging.StreamHandler()
    llm_handler.setLevel(llm_level)
    llm_handler.setFormatter(formatter)
    llm_logger = logging.getLogger("hnswlib_agent.llm")
    llm_logger.setLevel(llm_level)
    llm_logger.handlers.clear()
    llm_logger.addHandler(llm_handler)
    llm_logger.propagate = False

    # --- Pipeline progress logger (always INFO) ---
    pipeline_handler = logging.StreamHandler()
    pipeline_handler.setLevel(logging.INFO)
    pipeline_handler.setFormatter(formatter)
    pipeline_logger = logging.getLogger("hnswlib.pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.handlers.clear()
    pipeline_logger.addHandler(pipeline_handler)
    pipeline_logger.propagate = False


def run_pipeline(config_path: str, resume: bool = True, dry_run: bool = False) -> int:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_name = str((cfg.get("output") or {}).get("trials_name", "")).strip()
    if not trials_name:
        raise ValueError("output.trials_name must be provided and non-empty.")
    if any(ch in trials_name for ch in ["/", "\\"]):
        raise ValueError("output.trials_name must be a plain file stem, not a path.")

    trials_path = output_dir / "trials" / f"{trials_name}.jsonl"
    interval_tables_path = output_dir / "current_task_memory" / f"{trials_name}.interval_tables.jsonl"

    if not resume:
        if trials_path.exists():
            backup = output_dir / "trials" / f"{trials_name}.backup.{utc_now_iso().replace(':', '-')}.jsonl"
            shutil.move(str(trials_path), str(backup))
        if interval_tables_path.exists():
            backup = output_dir / "current_task_memory" / f"{trials_name}.backup.{utc_now_iso().replace(':', '-')}.interval_tables.jsonl"
            shutil.move(str(interval_tables_path), str(backup))

    existing_trials = load_trials(trials_path) if resume else []
    repeat = int(cfg["search"]["repeat"])
    budget = int(cfg["search"]["budget"])
    recall_threshold = float(cfg["search"]["recall_threshold"])
    recall_slack = float(cfg["search"].get("recall_slack", 0.001))
    if not (0.0 < recall_threshold <= 1.0):
        raise ValueError("search.recall_threshold must be in (0, 1].")
    if not (0.0 <= recall_slack < recall_threshold):
        raise ValueError("search.recall_slack must be >= 0 and < search.recall_threshold.")

    agentic_cfg = cfg.get("agentic") or {}
    _configure_agent_logging(agentic_cfg)
    _plog = logging.getLogger("hnswlib.pipeline")
    _plog.info("HNSW pipeline starting: trials=%s budget=%d threshold=%.4f slack=%.4f agentic=%s",
               trials_name, budget, recall_threshold, recall_slack,
               bool(agentic_cfg.get("enabled", True)))
    model_cfg = _resolve_model_cfg(agentic_cfg)
    knowledge_cfg = agentic_cfg.get("knowledge") or {}
    transfer_cfg = agentic_cfg.get("transfer") or {}
    transfer_enabled = bool(transfer_cfg.get("enabled", True))
    transfer_models_dir = Path(transfer_cfg.get("models_dir", "results/hnswlib/models")).expanduser().resolve()
    transfer_top_k_tasks = int(transfer_cfg.get("top_k_tasks", 3))
    transfer_trials_per_task = int(transfer_cfg.get("trials_per_task", 5))
    transfer_freeze_after_initial_design = bool(transfer_cfg.get("freeze_after_initial_design", False))
    refinement_cfg = (
        agentic_cfg.get("search_space_refinement") if isinstance(agentic_cfg.get("search_space_refinement"), dict) else {}
    )
    refinement_bootstrap_cfg = refinement_cfg.get("bootstrap") if isinstance(refinement_cfg.get("bootstrap"), dict) else {}
    refinement_seed_cfg = (
        refinement_cfg.get("seed_initial_design") if isinstance(refinement_cfg.get("seed_initial_design"), dict) else {}
    )

    # ── Initialization mode (history | regression_tree | regtree) ──
    # NOTE: agentic.subgroup_init is no longer honoured by this pipeline; the
    # nhq / unify / filter_diskann pipelines keep their own subgroup wiring.
    initial_design_cfg = agentic_cfg.get("initial_design") or {}
    init_mode = str(initial_design_cfg.get("mode", "history")).strip().lower()

    knowledge_mode = str(knowledge_cfg.get("mode", "knowledge_base_driven")).strip().lower()
    if knowledge_mode not in {"full_context_no_rag", "knowledge_base_driven"}:
        raise ValueError(
            "agentic.knowledge.mode must be 'full_context_no_rag' or 'knowledge_base_driven'."
        )
    overflow_policy = str(knowledge_cfg.get("overflow_policy", "truncate")).strip().lower()
    if overflow_policy not in {"truncate"}:
        raise ValueError("agentic.knowledge.overflow_policy must be 'truncate'.")

    # Build a simple (prompt) -> str wrapper around the agent's LLM caller
    # (used by CurrentTaskMemory.update_memory below).
    _llm_caller = lambda prompt: agent._invoke_llm("knowledge_match", prompt)

    # ── Conditional policy book (offline-validated symptom -> intervention) ──
    # Replaces the per-round static-knowledge card selection, which cost one
    # LLM call per round; matching is deterministic and LLM-free.
    conditional_policy_cfg = (
        agentic_cfg.get("conditional_policy")
        if isinstance(agentic_cfg.get("conditional_policy"), dict) else {}
    )
    conditional_policy_enabled = bool(conditional_policy_cfg.get("enabled", True))
    conditional_policy_runtime_cfg = ConditionalPolicyRuntimeConfig.from_mapping(conditional_policy_cfg)
    conditional_policy_book = (
        load_conditional_policy_book(conditional_policy_runtime_cfg.policy_file)
        if conditional_policy_enabled
        else {"policies": [], "meta": {}, "loaded": False, "reason": "disabled"}
    )
    _plog.info(
        "Conditional policy book: %d accepted policies (source=%s) from %s%s",
        len(conditional_policy_book["policies"]),
        (conditional_policy_book.get("meta") or {}).get("source"),
        (conditional_policy_book.get("meta") or {}).get("path"),
        "" if conditional_policy_book["policies"] else
        f" — DISABLED ({conditional_policy_book.get('reason', 'no_accepted_policies')})",
    )

    if knowledge_mode == "knowledge_base_driven":
        # Legacy monolithic knowledge (still loaded for stage_a fallback)
        from utils.knowledge_loader import build_knowledge_context as _kb_load_context

        knowledge_base_dir = str(knowledge_cfg.get("knowledge_base_dir", "knowledge_base"))
        max_context_chars = int(knowledge_cfg.get("max_context_chars", 120000))
        kb_result = _kb_load_context(base_dir=knowledge_base_dir, max_chars=max_context_chars)
        base_knowledge_full = kb_result["base_knowledge_full"]  # str: formatted markdown
        insights_dir = Path(".")  # not used in knowledge_base_driven mode
    else:
        # Legacy full_context_no_rag mode.
        insights_dir = Path(knowledge_cfg.get("insights_dir", "data/hnswlib_insights")).expanduser().resolve()
        base_knowledge_path = Path(knowledge_cfg.get("path", "data/hnswlib_knowledge.json")).expanduser().resolve()
        base_knowledge_full = _load_json_object(base_knowledge_path)

    agent = HNSWLIBTuningAgent(
        cfg["params"],
        seed=int(cfg["search"]["seed"]),
        agentic_cfg=agentic_cfg,
        model_cfg=model_cfg,
        llm_caller=None,
    )

    # ── Current Task Memory (the sole structured memory for this task) ─────
    current_task_cfg = agentic_cfg.get("current_task_memory") or {}
    memory_enabled = bool(current_task_cfg.get("enabled", True))
    current_memory: CurrentTaskMemory | None = None
    if memory_enabled:
        current_memory = CurrentTaskMemory(
            task_context={
                "recall_threshold": recall_threshold,
                "param_order": list(PARAM_ORDER),
                "knob_bounds": {
                    name: agent.space.domains[name].to_spec() for name in PARAM_ORDER
                },
            },
            parameter_space=agent.space,
            output_dir=output_dir,
            trials_name=trials_name,
        )
        _plog.info(
            "Current task memory enabled: delta_low=%.4f delta_high=%.4f points=%d",
            *current_memory.boundaries,
            current_memory.point_memory.count(),
        )
    else:
        _plog.info("Current task memory disabled — ablation mode")

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

    batch_size = int(agentic_cfg.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("agentic.batch_size must be > 0")
    stage_b_execution_per_round = 1
    max_rounds = _resolve_max_rounds(agentic_cfg.get("max_rounds", "auto"), budget, stage_b_execution_per_round, 1)
    stage_policy = {"recall_threshold": recall_threshold, "recall_slack": recall_slack}

    # --- ef mode: scan (default) vs direct ---
    tuning_cfg = cfg.get("tuning") or {}
    ef_mode = str(tuning_cfg.get("ef_mode", "scan")).strip().lower()
    if ef_mode not in ("scan", "direct"):
        raise ValueError("tuning.ef_mode must be 'scan' or 'direct'")

    if ef_mode == "direct":
        # Treat ef as a first-class tuned dimension — no free-ef scan.
        enable_ef_direct_mode()
        _plog.info("ef_mode=direct: ef added to BUILD_PARAM_ORDER; free-ef scan disabled")
        # Only the candidate's own ef value is tested (benchmark uses --ef directly).
        runner_cfg.extra_args = [
            *list(runner_cfg.extra_args),
            "--select-recall-threshold",
            f"{recall_threshold:g}",
            "--select-recall-slack",
            f"{recall_slack:g}",
        ]
    else:
        # Default scan mode: scan ef from min to max, pick best feasible.
        ef_spec = (cfg.get("params") or {}).get("ef") or {}
        if isinstance(ef_spec, dict) and isinstance(ef_spec.get("values"), list) and ef_spec["values"]:
            ef_values = sorted(int(v) for v in ef_spec["values"])
            ef_min, ef_max = ef_values[0], ef_values[-1]
        elif isinstance(ef_spec, dict) and ef_spec.get("min") is not None and ef_spec.get("max") is not None:
            ef_min = int(ef_spec["min"])
            ef_max = int(ef_spec["max"])
        else:
            ef_min, ef_max = 15, 120
        if ef_min > ef_max:
            ef_min, ef_max = ef_max, ef_min
        runner_cfg.extra_args = [
            *list(runner_cfg.extra_args),
            "--ef-scan-start",
            str(ef_min),
            "--ef-scan-end",
            str(ef_max),
            "--select-recall-threshold",
            f"{recall_threshold:g}",
            "--select-recall-slack",
            f"{recall_slack:g}",
        ]

    all_trials = list(existing_trials)
    stage_a_plan = _stage_a_plan_hnsw(
        agent=agent,
        stage_policy=stage_policy,
        base_knowledge_full=base_knowledge_full,
        transfer_enabled=transfer_enabled,
        transfer_models_dir=transfer_models_dir,
        transfer_top_k_tasks=transfer_top_k_tasks,
        transfer_trials_per_task=transfer_trials_per_task,
        knowledge_cfg=knowledge_cfg,
            insights_dir=insights_dir,
            benchmark_extra_args=runner_cfg.extra_args,
            trials_name=trials_name,
            refinement_cfg=refinement_cfg,
            refinement_bootstrap_cfg=refinement_bootstrap_cfg,
            refinement_seed_cfg=refinement_seed_cfg,
        )

    base_search_space = copy.deepcopy(stage_a_plan.get("base_search_space") or agent.space.export_parameter_space())
    frozen_search_space = copy.deepcopy(stage_a_plan.get("frozen_search_space") or base_search_space)
    transfer_context_fixed = copy.deepcopy(stage_a_plan.get("transfer_context") or {"tasks": []})
    knowledge_context_fixed = copy.deepcopy(stage_a_plan.get("knowledge_context") or {})
    stage_a_report = copy.deepcopy(stage_a_plan.get("stage_a_report") or {})
    initial_design_seed_candidates_fixed = copy.deepcopy(stage_a_plan.get("initial_design_seed_candidates") or [])
    cold_start_cards = copy.deepcopy(stage_a_plan.get("cold_start_cards") or [])
    # Regression-tree initialization context: rendered into every round's
    # proposal prompt (empty unless init_mode == "regression_tree").
    regression_tree_context = ""
    regression_tree_report: Dict[str, Any] = {}

    # ── Regression-tree initialization ───────────────────────────────────────
    # Prunes the full build space from historical trials (regression-tree
    # splits on qualifying-QPS SSE + two CI pruning rounds) and injects the
    # surviving regions as prompt context, seed anchors, and the frozen space.
    rt_cfg = initial_design_cfg.get("regression_tree")
    rt_cfg = rt_cfg if isinstance(rt_cfg, dict) else {}
    rt_trials = str(rt_cfg.get("trials_path", "")).strip()
    if init_mode in ("regression_tree", "regtree"):
        if not rt_trials:
            _plog.warning(
                "initial_design.mode=regression_tree requires "
                "initial_design.regression_tree.trials_path; keeping base space"
            )
        else:
            rt_path = Path(rt_trials)
            if not rt_path.is_absolute():
                rt_path = Path.cwd() / rt_path
            try:
                from utils.regression_tree_init import run_regression_tree_init

                regression_tree_report = run_regression_tree_init(
                    cfg.get("params") or {},
                    rt_path,
                    recall_threshold=recall_threshold,
                    min_leaf_samples=int(rt_cfg.get("min_leaf_samples", 5)),
                    recall_ci_confidence=float(rt_cfg.get("recall_ci_confidence", 0.95)),
                    qps_ci_confidence=float(rt_cfg.get("qps_ci_confidence", 0.95)),
                    max_depth=rt_cfg.get("max_depth"),
                    n1_ci_policy=str(rt_cfg.get("n1_ci_policy", "value")),
                    qps_ci_statistic=str(rt_cfg.get("qps_ci_statistic", "mean")),
                    bootstrap_resamples=int(rt_cfg.get("bootstrap_resamples", 2000)),
                    dedupe_by_param_key=str(rt_cfg.get("dedupe_by_param_key", "keep_all")),
                    seed_count=int(rt_cfg.get("seed_count", 8)),
                    seeds_per_region=int(rt_cfg.get("seeds_per_region", 1)),
                    seed=int(cfg["search"]["seed"]),
                    freeze_mode=str(rt_cfg.get("freeze_mode", "bbox")),
                    prompt_max_regions=int(rt_cfg.get("prompt_max_regions", 12)),
                    global_median_scope=str(rt_cfg.get("global_median_scope", "all")),
                    allow_empty=True,
                )
            except Exception as exc:  # noqa: BLE001 — init must never kill the run
                _plog.warning("Regression-tree init failed (%s); keeping base space", exc)

        if regression_tree_report.get("ok"):
            regression_tree_context = str(regression_tree_report.get("prompt_text") or "")
            rt_seeds = [
                s
                for s in (
                    _canonical_seed_candidate(
                        agent,
                        params=seed_candidate["params"],
                        allowed_values_override=base_search_space,
                        source="regression_tree_region",
                        note=str(seed_candidate.get("note", "")),
                    )
                    for seed_candidate in (regression_tree_report.get("seed_candidates") or [])
                )
                if s is not None
            ]
            if rt_seeds:
                initial_design_seed_candidates_fixed = rt_seeds
                cold_start_cards = []  # the pruned regions replace the cards
            if str(rt_cfg.get("freeze_mode", "bbox")) != "off" and regression_tree_report.get("bounding_box"):
                frozen_search_space = copy.deepcopy(regression_tree_report["bounding_box"])
            if bool(rt_cfg.get("write_artifact", True)):
                try:
                    rt_artifact_dir = Path(str(rt_cfg.get("artifact_dir", "results/hnswlib/regression_tree_init")))
                    if not rt_artifact_dir.is_absolute():
                        rt_artifact_dir = Path.cwd() / rt_artifact_dir
                    write_json(rt_artifact_dir / f"{trials_name}.json", regression_tree_report)
                    (rt_artifact_dir / f"{trials_name}.prompt.txt").write_text(
                        str(regression_tree_report.get("prompt_text") or ""), encoding="utf-8"
                    )
                except Exception as exc:  # noqa: BLE001 — artifacts are diagnostics only
                    _plog.warning("Regression-tree artifact write failed (%s)", exc)
            _plog.info(
                "Regression-tree init: %d/%d leaves survived (r1=%d, r2=%d), %d seeds, "
                "context %d chars | trials=%s",
                len(regression_tree_report.get("surviving_regions") or []),
                (regression_tree_report.get("tree") or {}).get("n_leaves", 0),
                len(regression_tree_report.get("pruned_round1") or []),
                len(regression_tree_report.get("pruned_round2") or []),
                len(initial_design_seed_candidates_fixed),
                len(regression_tree_context),
                rt_trials,
            )
        stage_a_report["regression_tree_init"] = {
            "enabled": bool(regression_tree_report.get("ok")),
            "trials_path": rt_trials,
            "reason": regression_tree_report.get("reason", "disabled"),
            "n_surviving_regions": len(regression_tree_report.get("surviving_regions") or []),
            "n_pruned_round1": len(regression_tree_report.get("pruned_round1") or []),
            "n_pruned_round2": len(regression_tree_report.get("pruned_round2") or []),
            "seed_candidate_count": len(initial_design_seed_candidates_fixed),
            "prompt_context_chars": len(regression_tree_context),
        }
    # ── End regression-tree initialization ───────────────────────────────────

    frozen_search_space_report = {
        "enabled": True,
        "reason": "stage_a_base_build_space",
        "space_frozen": bool(stage_a_report.get("space_frozen", False)),
        "fallback_to_base_space": bool(stage_a_report.get("fallback_to_base_space", False)),
        "refinement_source": str(stage_a_report.get("refinement_source", "base_space_fallback")),
        "history_override_report": copy.deepcopy(stage_a_report.get("history_override_report") or {}),
        "knowledge_truncation_report": copy.deepcopy(stage_a_report.get("knowledge_truncation_report") or {}),
        "used_similar_tasks": list(stage_a_report.get("used_similar_tasks") or []),
        "used_insight_cards": copy.deepcopy(stage_a_report.get("used_insight_cards") or []),
        "execution_per_round": stage_b_execution_per_round,
        "regression_tree_freeze": {
            "active": init_mode in ("regression_tree", "regtree")
            and bool(regression_tree_report.get("ok")),
            "freeze_mode": str(rt_cfg.get("freeze_mode", "bbox"))
            if init_mode in ("regression_tree", "regtree")
            else "off",
            "bounding_box": copy.deepcopy(regression_tree_report.get("bounding_box")),
            "n_surviving_regions": len(regression_tree_report.get("surviving_regions") or []),
        },
    }

    dry_plan: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "stage_a": {
            "base_search_space": copy.deepcopy(base_search_space),
            "frozen_search_space": copy.deepcopy(frozen_search_space),
            "transfer_context": copy.deepcopy(transfer_context_fixed),
            "knowledge_context": copy.deepcopy(knowledge_context_fixed),
            "stage_a_report": copy.deepcopy(stage_a_report),
            "initial_design_seed_candidates": copy.deepcopy(initial_design_seed_candidates_fixed),
            "regression_tree_init": copy.deepcopy(stage_a_report.get("regression_tree_init") or {}),
            "regression_tree_regions": copy.deepcopy(
                regression_tree_report.get("surviving_regions") or []
            ),
        },
        "stage_b_budget_total": budget,
        "stage_b_execution_per_round": stage_b_execution_per_round,
        "unified_tasks": [],
    }
    planned_task_keys = set(_attempted_task_keys(all_trials))
    previous_attribution: Dict[str, Any] | None = None
    # On resume, prime round_trials from the last executed trial so the
    # diagnostic loop starts with actual metrics to diagnose.
    existing_unified = [t for t in all_trials if t.get("stage") == "unified"]
    round_trials: List[Dict[str, Any]] = existing_unified[-1:] if existing_unified else []

    # ── Current Task Memory warm-up (replay existing trials on resume) ────
    if current_memory is not None and existing_unified:
        current_memory.warm_up(existing_unified)
        _plog.info(
            "Current task memory warm-up: %d points, %d transitions replayed",
            current_memory.point_memory.count(),
            current_memory.transition_memory.count(),
        )

    # Token accounting: snapshot after stage A so per-round deltas capture
    # only the current round's LLM calls (diagnose + attribution).
    round_token_snapshot = agent.token_snapshot()

    # Feedback from the previous round's posterior proposal check, shown to
    # the LLM in the next round's diagnose prompt.
    last_check_feedback = ""
    # Per-(M, efC) visit counts.  A pair is BANNED once it has been explored
    # 3 times — the model can no longer propose it (any round).
    construction_visit_counts: dict = {}
    # Per-round conditional-policy matching logs (persisted to dry_run_plan /
    # stage_report since proposal_log itself is never serialized).
    conditional_policy_round_logs: List[Dict[str, Any]] = []

    # Initialization-seed configs: used to exclude seed executions from the
    # tuning budget even when trials are reloaded in minimal format (which
    # loses proposal_source).
    seed_param_keys: Set[Tuple[Tuple[str, str], ...]] = {
        _trial_params_key(c["params"])
        for c in initial_design_seed_candidates_fixed
        if isinstance(c.get("params"), dict)
    }

    while True:
        stage_b_budget_used = _count_stage_b_runs(all_trials)
        # The search budget covers LLM tuning rounds only; initialization
        # seeds (subgroup mining / transfer / external / LSH /
        # regression-tree regions) do not consume it.
        tuning_budget_used = stage_b_budget_used - _count_seed_runs(all_trials, seed_param_keys)
        if tuning_budget_used >= budget or stage_b_budget_used >= max_rounds:
            break

        stage_b_round_idx = stage_b_budget_used + 1
        remaining_runs = max(0, budget - tuning_budget_used)
        if remaining_runs <= 0:
            break

        stage_trials = [trial for trial in all_trials if trial.get("stage") == "unified"]
        target_unique = stage_b_execution_per_round
        occupied_task_keys = set(planned_task_keys) if dry_run else set(_attempted_task_keys(all_trials))
        exhausted_param_keys = _exhausted_param_keys(all_trials, repeat)
        # Construction pairs explored >=3 times this run are banned from new
        # proposals (hard-checked on the finalized candidate; see below).
        banned_construction_pairs = {
            pair for pair, cnt in construction_visit_counts.items() if cnt >= 3
        }
        excluded_construction_pairs = set(banned_construction_pairs)
        transfer_context_used = copy.deepcopy(transfer_context_fixed)
        knowledge_context = copy.deepcopy(knowledge_context_fixed)
        used_similar_task_names = _extract_used_similar_task_names(transfer_context_used)
        allowed_values_override = copy.deepcopy(frozen_search_space)
        allowed_values_report = copy.deepcopy(frozen_search_space_report)

        # Build root state summarizing all trials so far
        root_state = agent.build_root_state(
            round_idx=stage_b_round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
            previous_attribution=previous_attribution,
        )

        # Get the last executed trial for per-execution diagnostic
        last_trial = round_trials[-1] if round_trials else None

        _plog.info("Round %d/%d | budget_used=%d/%d | stage=%s",
                   stage_b_round_idx, max_rounds, stage_b_budget_used, budget,
                   root_state.get("optimization_stage", "cold_start"))

        # ── Diagnostic LLM call: diagnose last execution → propose next ──
        # ── Per-round timing (initialised for all code paths) ──
        policy_match_time_s = 0.0
        memory_retrieval_time_s = 0.0
        llm_call_time_s = 0.0

        # Cold start: use Stage A seed candidates instead of LLM proposal
        if last_trial is None and initial_design_seed_candidates_fixed:
            seed_candidates = initial_design_seed_candidates_fixed
            candidates = [
                c for c in seed_candidates
                if params_to_key(c["params"], METRICS_PARAM_ORDER) not in exhausted_param_keys
            ]
            if candidates:
                candidate = candidates[0]
                diag_log = {
                    "ok": True, "diagnosis": {"strategy": "cold_start_seed"},
                    "candidate": candidate,
                    "classification": "cold_start",
                    "metric_actions": [], "tuning_action": {},
                    "error": "", "source": "stage_a_seed",
                    "prompt_payload": {},
                }
            else:
                candidate = None
        else:
            # ── Extract last-trial data once (shared by memory + knowledge) ──
            memory_ctx: Dict[str, Any] | None = None
            mem_full_state = None
            mem_obs: Dict[str, Any] | None = None
            last_metrics: Dict[str, Any] | None = None
            last_params: Dict[str, Any] | None = None

            if last_trial is not None and last_trial.get("status") == "success":
                last_metrics = last_trial.get("metrics") or {}
                last_params = last_trial.get("params") or {}
                mem_obs = {
                    "config": last_params,
                    "qps": float(last_metrics.get("qps", 0)),
                    "recall": float(last_metrics.get("recall", 0)),
                    "recall_threshold": recall_threshold,
                    "diagnostic_metrics": {
                        "visited_nodes_per_query": last_metrics.get("visited_nodes_per_query"),
                        "distance_computations": last_metrics.get("dist_comps_per_query"),
                        "out_degree_mean": last_metrics.get("out_degree_mean"),
                        "in_degree_mean": last_metrics.get("in_degree_mean"),
                        "in_degree_std": last_metrics.get("in_degree_std"),
                        "in_degree_max": last_metrics.get("in_degree_max"),
                        "index_size_mb": last_metrics.get("index_size_mb"),
                        "build_time_s": last_metrics.get("build_time_s"),
                        "selected_ef": last_metrics.get("selected_ef"),
                        "candidate_distance_stats": last_metrics.get("candidate_distance_stats"),
                        "max_recall": (last_metrics.get("frontier_summary") or {}).get("max_recall"),
                        "_best_feasible_qps": (
                            root_state.get("current_best_feasible", {}).get("QPS", 0)
                            if isinstance(root_state.get("current_best_feasible"), dict)
                            else 0
                        ),
                    },
                }

            # ── Retrieve current task memory context ──
            memory_retrieval_time_s = 0.0
            interval_table_context = ""
            t_mem_start = time.perf_counter()
            if current_memory is not None and mem_obs is not None:
                mem_full_state = current_memory.build_full_state(mem_obs)
                memory_ctx = current_memory.retrieve_memory_context(
                    current_full_state=mem_full_state,
                    current_config=last_params,
                )
                if agentic_cfg.get("interval_table_in_prompt", True):
                    interval_table = (memory_ctx or {}).get("runtime_structural_interval_table")
                    if isinstance(interval_table, dict):
                        interval_table_context = build_current_task_memory_prompt(interval_table)
            memory_retrieval_time_s = round(time.perf_counter() - t_mem_start, 4)

            # ── Conditional policy matching (deterministic, no LLM) ──
            # Symptom <- last executed trial + this run's own successful
            # trials.  hnswlib has no expansion / traversal-effectiveness
            # counters, so those dimensions are wildcards; with no (or too
            # little) history the full accepted set is injected.  Fail-open.
            conditional_policy_text = ""
            conditional_policy_log: Dict[str, Any] = {"enabled": False, "skipped_reason": "startup_failed"}
            policy_match_time_s = 0.0
            if conditional_policy_enabled:
                conditional_policy_text, conditional_policy_log = _build_conditional_policy_text(
                    all_trials=all_trials,
                    last_metrics=last_metrics if isinstance(last_metrics, dict) else None,
                    recall_threshold=recall_threshold,
                    book=conditional_policy_book,
                    cfg=conditional_policy_runtime_cfg,
                    logger=_plog,
                )
                policy_match_time_s = conditional_policy_log.get("elapsed_s", 0.0)
                conditional_policy_round_logs.append(
                    {**conditional_policy_log, "round_idx": stage_b_round_idx}
                )
                _plog.info(
                    "Round %d conditional policy: matched=%s fallback=%s history=%d chars=%d (%.4fs)%s",
                    stage_b_round_idx,
                    conditional_policy_log.get("matched_policy_ids") or "-",
                    conditional_policy_log.get("fallback"),
                    conditional_policy_log.get("n_history_trials", 0),
                    conditional_policy_log.get("chars", 0),
                    policy_match_time_s,
                    f" skipped={conditional_policy_log.get('skipped_reason')}"
                    if conditional_policy_log.get("skipped_reason") else "",
                )

            # ── LLM call (timed) ──
            t_llm_start = time.perf_counter()
            # ── Construction-probe: force M+efC change on recall stagnation ──
            # Triggers when (a) every N rounds (interval backstop), or (b) the
            # recall of the last K successful rounds has barely moved — the
            # current region is exhausted, explore new construction settings.
            stagnation_rounds = int(agentic_cfg.get("recall_stagnation_rounds", 3))
            stagnation_delta = float(agentic_cfg.get("recall_stagnation_delta", 0.005))
            recent_recalls = [
                float(t.get("metrics", {}).get("recall"))
                for t in all_trials[-stagnation_rounds:]
                if t.get("status") == "success"
                and isinstance(t.get("metrics"), dict)
                and t.get("metrics", {}).get("recall") is not None
            ]
            recall_stagnated = bool(
                len(recent_recalls) >= stagnation_rounds
                and max(recent_recalls) - min(recent_recalls) < stagnation_delta
            )
            force_construction = recall_stagnated
            if force_construction:
                _plog.info(
                    "Construction-probe round %d: M+efC change required (recall stagnation)",
                    stage_b_round_idx,
                )

            candidate, diag_log = agent.diagnose_last_execution(
                last_trial=last_trial,
                root_state=root_state,
                stage_policy=stage_policy,
                allowed_values_override=allowed_values_override,
                cold_start_cards=cold_start_cards,
                memory_context=memory_ctx,
                conditional_policy_context=conditional_policy_text,
                proposal_check_feedback=last_check_feedback,
                interval_table_context=interval_table_context,
                regression_tree_context=regression_tree_context,
                force_construction=force_construction,
                excluded_construction_pairs=excluded_construction_pairs,
            )
            llm_call_time_s = round(time.perf_counter() - t_llm_start, 4)

        # Fallback: if LLM fails or proposes duplicate build, use fallback candidates
        if candidate is None:
            candidates = _nonbanned_fallback_candidates(
                agent,
                target_unique,
                exhausted_param_keys,
                stage_policy=stage_policy,
                allowed_values_override=allowed_values_override,
                excluded_construction_pairs=excluded_construction_pairs,
                logger=_plog,
            )
            diag_log["fallback_used"] = True
        else:
            candidates = [candidate]
            diag_log["fallback_used"] = False
            # If the LLM's candidate is a duplicate key, perturb it slightly
            # instead of falling back to random exploration anchors.
            c_key = params_to_key(candidate["params"], METRICS_PARAM_ORDER)
            if c_key in exhausted_param_keys:
                perturbed = _perturb_duplicate(
                    agent, candidate["params"], exhausted_param_keys,
                    allowed_values_override=allowed_values_override,
                )
                if perturbed is not None:
                    _plog.info(
                        "LLM proposed duplicate %s, perturbed to %s",
                        c_key, params_to_key(perturbed, METRICS_PARAM_ORDER),
                    )
                    candidates = [{"params": perturbed, "source": "diagnostic_perturbed", "note": candidate.get("note", "")}]
                    diag_log["fallback_used"] = False
                else:
                    _plog.info("LLM proposed duplicate build key %s, falling back", c_key)
                    candidates = _nonbanned_fallback_candidates(
                        agent,
                        target_unique,
                        exhausted_param_keys,
                        stage_policy=stage_policy,
                        allowed_values_override=allowed_values_override,
                        excluded_construction_pairs=excluded_construction_pairs,
                        logger=_plog,
                    )
                    diag_log["fallback_used"] = True
                    diag_log["fallback_reason"] = "duplicate_build_key"

        # ── Posterior proposal check (interval evidence) ──
        # Checks the finalized proposal against the runtime structural
        # interval table.  A too_weak / too_conservative verdict feeds the
        # failure reasons back to the LLM, which may revise the proposal
        # (bounded number of attempts).  The checker never silently
        # replaces the proposal.
        proposal_check = None
        check_revisions: List[Dict[str, Any]] = []
        interval_table = None
        current_state = None
        if current_memory is not None and candidates:
            try:
                interval_table = current_memory.build_runtime_structural_interval_table()
                if last_trial is not None and last_trial.get("status") == "success":
                    m_last = last_trial.get("metrics") or {}
                    rec_last = m_last.get("recall")
                    current_state = {
                        "current_configuration": dict(last_trial.get("params") or {}),
                        "current_recall": (
                            float(rec_last) if rec_last is not None else None
                        ),
                        "current_qps": m_last.get("qps"),
                        "current_feasible": bool(
                            rec_last is not None and float(rec_last) >= recall_threshold
                        ),
                    }
                proposal_check = check_raw_proposal(
                    runtime_structural_interval_table=interval_table,
                    raw_proposal=dict(candidates[0]["params"]),
                    current_state=current_state,
                )
            except Exception:
                proposal_check = None

        # ── Checker-guided LLM re-proposal loop ──
        posterior_check_cfg = agentic_cfg.get("posterior_check") or {}
        max_check_revisions = int(posterior_check_cfg.get("max_revisions", 2))
        if (
            current_memory is not None
            and candidates
            and proposal_check is not None
            and max_check_revisions > 0
            and proposal_check.get("decision")
            in (DECISION_TOO_WEAK, DECISION_TOO_CONSERVATIVE)
        ):
            for _ in range(max_check_revisions):
                feedback = format_check_result_for_llm(proposal_check)
                revised = agent.repropose_after_proposal_check(
                    check_feedback=feedback,
                    allowed_values_override=allowed_values_override,
                )
                if revised is None or not isinstance(revised.get("params"), dict):
                    break
                rev_params = dict(revised["params"])
                rev_key = params_to_key(rev_params, METRICS_PARAM_ORDER)
                if rev_key in exhausted_param_keys:
                    perturbed = _perturb_duplicate(
                        agent, rev_params, exhausted_param_keys,
                        allowed_values_override=allowed_values_override,
                    )
                    if perturbed is None:
                        break
                    rev_params = perturbed
                recheck = None
                try:
                    recheck = check_raw_proposal(
                        runtime_structural_interval_table=interval_table,
                        raw_proposal=dict(rev_params),
                        current_state=current_state,
                    )
                except Exception:
                    recheck = None
                check_revisions.append(
                    {
                        "attempt": len(check_revisions) + 1,
                        "previous_decision": proposal_check.get("decision"),
                        "revised_params": copy.deepcopy(rev_params),
                        "revised_check": copy.deepcopy(recheck),
                    }
                )
                candidates = [
                    {
                        "params": rev_params,
                        "source": "diagnostic_checked_revision",
                        "note": (str(revised.get("rationale", "")) + " [POSTERIOR CHECK REVISION]"),
                    }
                ]
                diag_log["posterior_check_revised"] = True
                proposal_check = recheck
                _plog.info(
                    "Proposal check: %s → LLM re-proposed %s → new check: %s",
                    check_revisions[-1]["previous_decision"],
                    params_to_key(rev_params, METRICS_PARAM_ORDER),
                    recheck.get("decision") if recheck else "unknown",
                )
                if not recheck or recheck.get("decision") not in (
                    DECISION_TOO_WEAK, DECISION_TOO_CONSERVATIVE
                ):
                    break

        if proposal_check is not None:
            last_check_feedback = format_check_result_for_llm(proposal_check)
            suggested = proposal_check.get("suggested_proposal")
            _plog.info(
                "Proposal check: %s%s",
                proposal_check.get("decision"),
                f" → suggested={suggested}" if suggested else "",
            )

        # ── Banned construction-pair hard check (LLM feedback loop) ──
        # The BANNED list is already shown in the diagnose prompt, but the
        # LLM can still end up proposing a banned pair (directly, via
        # duplicate perturbation, or via a checker-guided revision).  When
        # the finalized proposal uses a construction pair explored >=3
        # times, do NOT silently substitute it: feed the rejection back to
        # the LLM and ask it to re-propose a different pair grounded in the
        # diagnostic evidence (knowledge + diagnosis, not a random retry).
        # Only after the bounded re-proposal attempts are exhausted do we
        # fall back to non-banned design candidates.
        banned_reproposal_cfg = agentic_cfg.get("banned_reproposal") or {}
        max_banned_reproposals = max(0, int(banned_reproposal_cfg.get("max_attempts", 2)))
        banned_reproposal_attempts = 0
        while candidates and banned_reproposal_attempts < max_banned_reproposals:
            final_params = candidates[0].get("params") or {}
            final_pair = (
                int(final_params.get("M", -1)),
                int(final_params.get("ef_construction", -1)),
            )
            if final_pair not in excluded_construction_pairs:
                break
            banned_reproposal_attempts += 1
            _plog.warning(
                "Proposal uses banned construction pair %s (explored >=3x) — "
                "asking LLM to re-propose (%d/%d)",
                final_pair, banned_reproposal_attempts, max_banned_reproposals,
            )
            rejection_feedback = (
                f"Your proposed configuration uses construction pair "
                f"(M={final_pair[0]}, ef_construction={final_pair[1]}), which "
                f"has already been explored >=3 times in this task and is "
                f"BANNED to avoid over-exploiting one local region. "
                f"Re-propose ONE configuration with a DIFFERENT construction "
                f"pair (M and/or ef_construction must change; changing ef "
                f"alone is not acceptable). Ground the re-proposal in the "
                f"diagnostic evidence: the diagnostic tree actions, the "
                f"cross-construction comparison table, and the runtime "
                f"interval table."
            )
            t_repropose_start = time.perf_counter()
            re_candidate, re_diag_log = agent.diagnose_last_execution(
                last_trial=last_trial,
                root_state=root_state,
                stage_policy=stage_policy,
                allowed_values_override=allowed_values_override,
                cold_start_cards=cold_start_cards,
                memory_context=memory_ctx,
                conditional_policy_context=conditional_policy_text,
                proposal_check_feedback=rejection_feedback,
                interval_table_context=interval_table_context,
                regression_tree_context=regression_tree_context,
                force_construction=force_construction,
                excluded_construction_pairs=excluded_construction_pairs,
            )
            llm_call_time_s += round(time.perf_counter() - t_repropose_start, 4)
            if re_candidate is None or not isinstance(re_candidate.get("params"), dict):
                break
            re_params = dict(re_candidate["params"])
            re_key = params_to_key(re_params, METRICS_PARAM_ORDER)
            if re_key in exhausted_param_keys:
                perturbed = _perturb_duplicate(
                    agent, re_params, exhausted_param_keys,
                    allowed_values_override=allowed_values_override,
                )
                if perturbed is None:
                    break
                re_params = perturbed
            candidates = [
                {
                    "params": re_params,
                    "source": "banned_reproposal",
                    "note": (str(re_candidate.get("note", "")) + " [BANNED RE-PROPOSAL]"),
                }
            ]
            diag_log.update(re_diag_log)
            diag_log["banned_reproposals"] = banned_reproposal_attempts

        # If the LLM still insists on a banned pair after all re-proposal
        # attempts, fall back to non-banned design candidates.
        if candidates:
            final_params = candidates[0].get("params") or {}
            final_pair = (
                int(final_params.get("M", -1)),
                int(final_params.get("ef_construction", -1)),
            )
            if final_pair in excluded_construction_pairs:
                _plog.warning(
                    "LLM kept proposing banned pair %s after %d re-proposal(s) — "
                    "using non-banned fallback candidates",
                    final_pair, banned_reproposal_attempts,
                )
                candidates = _nonbanned_fallback_candidates(
                    agent,
                    target_unique,
                    exhausted_param_keys,
                    stage_policy=stage_policy,
                    allowed_values_override=allowed_values_override,
                    excluded_construction_pairs=excluded_construction_pairs,
                    logger=_plog,
                )
                diag_log["fallback_used"] = True
                diag_log["fallback_reason"] = "banned_pair_exhausted"

        # Keep the posterior-check verdict consistent with the finalized
        # candidate when a banned re-proposal replaced it.
        if banned_reproposal_attempts > 0 and candidates and interval_table is not None:
            try:
                recheck_final = check_raw_proposal(
                    runtime_structural_interval_table=interval_table,
                    raw_proposal=dict(candidates[0]["params"]),
                    current_state=current_state,
                )
            except Exception:
                recheck_final = None
            if recheck_final is not None:
                proposal_check = recheck_final
                last_check_feedback = format_check_result_for_llm(proposal_check)
                _plog.info(
                    "Proposal check (after banned re-proposal): %s",
                    proposal_check.get("decision"),
                )

        # ── Hard repository check (runtime.tex dominance rules) ──
        repository_check_cfg = agentic_cfg.get("repository_check") or {}
        repo_check_max_attempts = max(1, int(repository_check_cfg.get("max_attempts", 2)))
        if candidates and interval_table is not None:
            repo_attempts = 0
            while repo_attempts < repo_check_max_attempts:
                rejected_flag, reject_feedback = hard_reject(
                    runtime_structural_interval_table=interval_table,
                    proposal=dict(candidates[0]["params"]),
                )
                if not rejected_flag:
                    break
                repo_attempts += 1
                _plog.warning(
                    "Repository check round %d (%d/%d): %s",
                    stage_b_round_idx, repo_attempts, repo_check_max_attempts,
                    reject_feedback,
                )
                t_repo_start = time.perf_counter()
                re_candidate, _repo_diag = agent.diagnose_last_execution(
                    last_trial=last_trial,
                    root_state=root_state,
                    stage_policy=stage_policy,
                    allowed_values_override=allowed_values_override,
                    cold_start_cards=cold_start_cards,
                    memory_context=memory_ctx,
                    conditional_policy_context=conditional_policy_text,
                    proposal_check_feedback=reject_feedback,
                    interval_table_context=interval_table_context,
                    regression_tree_context=regression_tree_context,
                    force_construction=force_construction,
                    excluded_construction_pairs=excluded_construction_pairs,
                )
                llm_call_time_s += round(time.perf_counter() - t_repo_start, 4)
                if re_candidate is None or not isinstance(re_candidate.get("params"), dict):
                    break
                candidates = [re_candidate]
            else:
                # Attempts exhausted and still rejected: fall back to
                # repository-safe design candidates; if none qualify, use the
                # least-violating fallback candidate to avoid deadlock.
                _plog.warning(
                    "LLM kept proposing repository-violating efS after %d "
                    "re-proposal(s) — using repository-safe fallback candidates",
                    repo_check_max_attempts,
                )
                fb = _nonbanned_fallback_candidates(
                    agent,
                    target_unique,
                    exhausted_param_keys,
                    stage_policy=stage_policy,
                    allowed_values_override=allowed_values_override,
                    excluded_construction_pairs=excluded_construction_pairs,
                    logger=_plog,
                )
                safe = [
                    c for c in fb
                    if not hard_reject(interval_table, dict(c["params"]))[0]
                ]
                if safe:
                    candidates = [safe[0]]
                elif fb:
                    candidates = [fb[0]]
                    _plog.warning(
                        "No repository-safe fallback candidate exists — using the "
                        "least-violating fallback candidate to avoid deadlock"
                    )
                diag_log["fallback_used"] = True
                diag_log["fallback_reason"] = "repository_check_exhausted"

        proposal_log = {
            "stage": "unified",
            "round_idx": stage_b_round_idx,
            "stage_b_round_idx": stage_b_round_idx,
            "stage_b_budget_total": budget,
            "stage_b_budget_used": stage_b_budget_used,
            "target_unique": target_unique,
            "candidate_count": len(candidates),
            "candidate_source": candidates[0]["source"] if candidates else "none",
            "proposal_mode": "diagnostic",
            "optimizer": "diagnostic",
            "root_state": root_state,
            "diagnosis": diag_log.get("diagnosis", {}),
            "diagnostic_log": diag_log,
            "optimization_stage": root_state.get("optimization_stage", "cold_start"),
            # ── New diagnostic fields ──
            "classification": diag_log.get("classification", root_state.get("classification", "")),
            "metric_actions": diag_log.get("metric_actions", []),
            "tuning_action": diag_log.get("tuning_action", {}),
            "allowed_values_override_used": True,
            "allowed_values_override": copy.deepcopy(allowed_values_override),
            "allowed_values_override_report": copy.deepcopy(allowed_values_report),
            "param_order": list(PARAM_ORDER),
            "stage_policy": dict(stage_policy),
            "stage_a_report": copy.deepcopy(stage_a_report),
            "base_search_space": copy.deepcopy(base_search_space),
            "frozen_search_space": copy.deepcopy(frozen_search_space),
            "stage_b_execution_per_round": stage_b_execution_per_round,
            "proposal_check_result": proposal_check,
            "posterior_check_revisions": check_revisions,
        }

        tasks = _build_tasks(
            stage="unified",
            candidates=candidates,
            needed_runs=min(stage_b_execution_per_round, remaining_runs),
            repeat=repeat,
            occupied_task_keys=occupied_task_keys,
            round_idx=stage_b_round_idx,
        )
        for task in tasks:
            task["used_similar_task_names"] = list(used_similar_task_names)
            task["used_similar_task_count"] = len(used_similar_task_names)

        proposal_log["generated_at"] = utc_now_iso()
        proposal_log["planned_run_count"] = len(tasks)
        proposal_log["executor_script"] = runner_cfg.script_path
        proposal_log["executor_mode"] = "hnswlib_direct" if Path(runner_cfg.script_path).name == "hnswlib_benchmark.py" else "hnswlib_generic"
        proposal_log["benchmark_artifacts"] = []
        proposal_log["transfer_enabled"] = bool(transfer_enabled)
        proposal_log["transfer_context_used"] = copy.deepcopy(transfer_context_used)
        proposal_log["knowledge_mode"] = "hnsw_full_context_no_rag"
        proposal_log["knowledge_truncation_report"] = copy.deepcopy(stage_a_report.get("knowledge_truncation_report") or {})

        if not tasks:
            proposal_log["round_status"] = "no_tasks"
            # Log round tokens before breaking
            round_prompt = agent.total_prompt_tokens - round_token_snapshot[0]
            round_completion = agent.total_completion_tokens - round_token_snapshot[1]
            proposal_log["round_prompt_tokens"] = round_prompt
            proposal_log["round_completion_tokens"] = round_completion
            _plog.info("Round %d LLM tokens: +%d prompt, +%d completion (cumulative %d/%d)",
                       stage_b_round_idx, round_prompt, round_completion,
                       agent.total_prompt_tokens, agent.total_completion_tokens)
            break

        round_interval_table = None

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
                planned_task_keys.add((_build_key_from_params(task["params"]), task["repeat_idx"]))
                all_trials.append(
                    {
                        "run_id": f"dryrun-hnsw-{stage_b_round_idx}-{task['repeat_idx']}-{_build_key_from_params(task['params'])}",
                        "stage": task["stage"],
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
            for _t in round_trials:
                _tp = _t.get("params")
                if isinstance(_tp, dict) and "M" in _tp and "ef_construction" in _tp:
                    _pair = (int(_tp["M"]), int(_tp["ef_construction"]))
                    construction_visit_counts[_pair] = construction_visit_counts.get(_pair, 0) + 1
                    if construction_visit_counts[_pair] >= 3:
                        _plog.info(
                            "Construction pair %s explored %d times — BANNED",
                            _pair, construction_visit_counts[_pair],
                        )
            all_trials.extend(round_trials)
            proposal_log["benchmark_artifacts"] = [
                trial["benchmark_artifacts"]
                for trial in round_trials
                if isinstance(trial.get("benchmark_artifacts"), dict) and trial.get("benchmark_artifacts")
            ]

            # ── Trial attribution (optional, for logging) ──
            if round_trials:
                try:
                    previous_attribution = agent.analyze_trial_attribution(
                        round_trials=round_trials,
                        stage_trials=stage_trials,
                        stage_policy=stage_policy,
                    )
                except Exception:
                    previous_attribution = None

            # ── Update Current Task Memory ──
            if current_memory is not None:
                    last_success = round_trials[-1] if round_trials else None
                    if last_success and last_success.get("status") == "success":
                        m_after = last_success.get("metrics") or {}
                        p_after = last_success.get("params") or {}
                        # Build before-state from previous trial
                        before_fs = current_memory._last_full_state
                        # Build after observation
                        after_obs = {
                            "config": p_after,
                            "qps": float(m_after.get("qps", 0)),
                            "recall": float(m_after.get("recall", 0)),
                            "recall_threshold": recall_threshold,
                            "diagnostic_metrics": {
                                "visited_nodes_per_query": m_after.get("visited_nodes_per_query"),
                                "distance_computations": m_after.get("dist_comps_per_query"),
                                "out_degree_mean": m_after.get("out_degree_mean"),
                                "in_degree_mean": m_after.get("in_degree_mean"),
                                "index_size_mb": m_after.get("index_size_mb"),
                                "max_recall": (m_after.get("frontier_summary") or {}).get("max_recall"),
                                "_best_feasible_qps": (
                                    root_state.get("current_best_feasible", {}).get("QPS", 0)
                                    if isinstance(root_state.get("current_best_feasible"), dict)
                                    else 0
                                ),
                            },
                        }
                        if before_fs is not None:
                            action = current_memory.build_action(
                                before_fs["structured_state"]["config"],
                                p_after,
                            )
                            current_memory.update_memory(
                                before_fs, action, after_obs, llm_caller=_llm_caller,
                            )
                        else:
                            current_memory.add_initial_observation(after_obs)

            # ── Per-round interval table snapshot ──
            # Logged after the round's point is absorbed: this is the table
            # that will guide the NEXT round's proposal.
            if current_memory is not None:
                try:
                    round_interval_table = (
                        current_memory.build_runtime_structural_interval_table()
                    )
                    append_jsonl(
                        interval_tables_path,
                        {
                            "round_idx": stage_b_round_idx,
                            "generated_at": utc_now_iso(),
                            "runtime_structural_interval_table": round_interval_table,
                        },
                    )
                except Exception:
                    round_interval_table = None

        proposal_log["transfer_context_initial"] = copy.deepcopy(transfer_context_fixed)
        proposal_log["transfer_context_next"] = copy.deepcopy(transfer_context_fixed)
        proposal_log["round_status"] = "planned" if dry_run else "executed"
        # Per-round token delta
        round_prompt = agent.total_prompt_tokens - round_token_snapshot[0]
        round_completion = agent.total_completion_tokens - round_token_snapshot[1]
        proposal_log["round_prompt_tokens"] = round_prompt
        proposal_log["round_completion_tokens"] = round_completion
        # Per-round timing
        proposal_log["knowledge_selection_time_s"] = policy_match_time_s  # legacy key kept for consumers
        proposal_log["policy_match_time_s"] = policy_match_time_s
        proposal_log["conditional_policy"] = conditional_policy_log
        proposal_log["memory_retrieval_time_s"] = memory_retrieval_time_s
        proposal_log["llm_call_time_s"] = llm_call_time_s
        proposal_log["workload_time_s"] = round(
            sum(t.get("elapsed_s", 0) for t in round_trials), 4
        )
        # Per-round runtime structural interval table snapshot
        proposal_log["runtime_structural_interval_table"] = round_interval_table
        if round_interval_table is not None:
            table_rows = round_interval_table.get("interval_table") or []
            statuses = [
                c.get("status")
                for row in table_rows
                for c in row.get("efC_cells", [])
            ]
            status_summary = ", ".join(
                f"{status}×{statuses.count(status)}"
                for status in ("bracketed", "feasible_observed", "infeasible_only", "unresolved")
                if statuses.count(status)
            )
            _plog.info(
                "Interval table: %d M-row(s), %d cell(s) | %s",
                len(table_rows), len(statuses), status_summary or "empty",
            )
        _plog.info(
            "Round %d LLM tokens: +%d prompt, +%d completion (cumulative %d/%d) | "
            "timing: policy=%.3fs memory=%.3fs llm=%.3fs workload=%.3fs",
            stage_b_round_idx, round_prompt, round_completion,
            agent.total_prompt_tokens, agent.total_completion_tokens,
            policy_match_time_s, memory_retrieval_time_s,
            llm_call_time_s,
            sum(t.get("elapsed_s", 0) for t in round_trials),
        )
        round_token_snapshot = agent.token_snapshot()

    if dry_run:
        dry_plan["conditional_policy_rounds"] = conditional_policy_round_logs
        write_json(output_dir / "dry_run_plan.json", dry_plan)
        return 0

    all_trials = load_trials(trials_path)
    aggregates = aggregate_success_trials(all_trials)
    pareto = compute_pareto_front(aggregates)
    summary = compute_summary(pareto)
    summary.update(compute_threshold_summary(pareto, recall_threshold, recall_slack))
    final_success_runs = _count_success_runs(all_trials)
    final_stage_b_runs = _count_stage_b_runs(all_trials)
    stage_trials = [trial for trial in all_trials if trial.get("stage") == "unified"]
    stage_report = {
        "generated_at": utc_now_iso(),
        "task_name": trials_name,
        "trials_path": str(trials_path),
        "optimizer": "diagnostic",
        "threshold_policy": {
            "mode": "hnsw_diagnostic_threshold_then_qps",
            "recall_threshold": recall_threshold,
            "recall_slack": recall_slack,
        },
        "param_order": list(PARAM_ORDER),
        "budget": {
            "total_runs": budget,
            "stage_a_runs": 0,
            "stage_b_total_runs": budget,
            "stage_b_used_runs": final_stage_b_runs,
            "stage_b_remaining": max(0, budget - final_stage_b_runs),
            "unified_target_runs": budget,
            "unified_remaining": max(0, budget - final_stage_b_runs),
        },
        "stage_a": {
            "base_search_space": copy.deepcopy(base_search_space),
            "frozen_search_space": copy.deepcopy(frozen_search_space),
            "transfer_context": copy.deepcopy(transfer_context_fixed),
            "knowledge_context": copy.deepcopy(knowledge_context_fixed),
            "stage_a_report": copy.deepcopy(stage_a_report),
            "initial_design_seed_candidates": copy.deepcopy(initial_design_seed_candidates_fixed),
            "regression_tree_init": copy.deepcopy(stage_a_report.get("regression_tree_init") or {}),
            "regression_tree_regions": copy.deepcopy(
                regression_tree_report.get("surviving_regions") or []
            ),
        },
        "stage_b": {
            "execution_per_round": stage_b_execution_per_round,
            "max_rounds": max_rounds,
            "used_rounds": final_stage_b_runs,
            "best_so_far_curve": _build_best_so_far_curve(stage_trials, recall_threshold),
            "conditional_policy": {
                "n_rounds": len(conditional_policy_round_logs),
                "n_matched_rounds": sum(
                    1 for r in conditional_policy_round_logs if r.get("matched_policy_ids")
                ),
                "n_fallback_rounds": sum(
                    1 for r in conditional_policy_round_logs if r.get("fallback")
                ),
                "matched_policy_ids": sorted({
                    pid
                    for r in conditional_policy_round_logs
                    for pid in (r.get("matched_policy_ids") or [])
                }),
            },
        },
        "llm_tokens": {
            "init_prompt_tokens": agent.init_prompt_tokens,
            "init_completion_tokens": agent.init_completion_tokens,
            "total_prompt_tokens": agent.total_prompt_tokens,
            "total_completion_tokens": agent.total_completion_tokens,
        },
        "unified": _unified_report(all_trials, budget, recall_threshold, recall_slack),
    }
    write_json(output_dir / "pareto.json", pareto)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "stage_report.json", stage_report)
    best_feasible = summary.get("best_feasible")
    best_feasible_str = ""
    if isinstance(best_feasible, dict):
        best_feasible_str = "params=%s recall=%.4f qps=%.1f" % (
            str(best_feasible.get("params", {})),
            float(best_feasible.get("recall", 0)),
            float(best_feasible.get("qps", 0)),
        )
    _plog.info("Pipeline finished: success_runs=%d/%d pareto=%d best_feasible=%s",
               final_success_runs, final_stage_b_runs, len(pareto), best_feasible_str)
    _plog.info("LLM tokens — init: %d prompt + %d completion | total: %d prompt + %d completion",
               agent.init_prompt_tokens, agent.init_completion_tokens,
               agent.total_prompt_tokens, agent.total_completion_tokens)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent hnswlib recall/QPS tuning pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hnswlib_tune.yaml",
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
    return run_pipeline(args.config, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_knowledge_context", "load_config", "run_pipeline", "_build_transfer_context"]

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from utils.subgroup_insights import (
    DEFAULT_WEIGHT_FEASIBLE_COUNT,
    DEFAULT_WEIGHT_FEASIBLE_QPS,
    DEFAULT_WEIGHT_RECALL_MARGIN,
    _normalize_task_name,
    _normalize_recall_thresholds,
    _slugify,
    _threshold_slug,
    _utc_now_iso,
    build_insight_cards,
    build_parameter_discretization,
    discover_subgroups,
    maybe_rewrite_advice_with_llm,
    score_subgroups,
    select_non_redundant_subgroups,
)


DOMAIN = "hnswlib_tuning"
PARAM_ORDER = ["M", "ef_construction", "ef"]
INSIGHT_SOURCE_DOC_PREFIX = "hnswlib_tuning_insights"
INSIGHT_UNIT_PREFIX = "hnswlib_tuning_insight__"


def _normalize_task_file_stem(task_name: str) -> str:
    raw = str(task_name).strip()
    if raw.endswith(".jsonl"):
        raw = raw[: -len(".jsonl")]
    if not raw:
        raise ValueError("task_name must not be empty")
    return raw


def _safe_float(value: Any) -> float:
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
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


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed["_line_idx"] = line_idx
                yield parsed


def _load_observations_from_file(trial_path: Path) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for row in _iter_jsonl(trial_path):
        if row.get("status") != "success":
            continue
        params = row.get("params")
        metrics = row.get("metrics")
        if not isinstance(params, dict) or not isinstance(metrics, dict):
            continue
        if "recall" not in metrics or "qps" not in metrics:
            continue

        try:
            recall = _safe_float(metrics["recall"])
            qps = _safe_float(metrics["qps"])
            canonical_params = {
                name: _maybe_int(_safe_float(params[name]))
                for name in PARAM_ORDER
            }
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(f"{trial_path}: missing HNSWLIB parameter '{missing}'.") from exc
        except (TypeError, ValueError):
            continue

        observations.append(
            {
                "trial_ref": f"{trial_path.name}:{row.get('_line_idx', 0)}",
                "source_file": trial_path.name,
                "params": canonical_params,
                "recall": recall,
                "qps": qps,
            }
        )
    return observations


def _discover_trial_files(trials_dir: Path, task_name: str | None = None) -> List[Path]:
    if not trials_dir.exists() or not trials_dir.is_dir():
        raise FileNotFoundError(f"Trials directory not found: {trials_dir}")

    if task_name:
        normalized = _normalize_task_file_stem(task_name)
        candidate = trials_dir / f"{normalized}.jsonl"
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Task trial file not found: {candidate}")
        return [candidate]

    files = sorted(path for path in trials_dir.glob("*.jsonl") if path.is_file())
    if not files:
        raise ValueError(f"No trial jsonl files found in {trials_dir}")
    return files


def load_task_observations(
    trials_dir: Path,
    task_name: str | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    task_observations: Dict[str, List[Dict[str, Any]]] = {}
    for trial_path in _discover_trial_files(trials_dir, task_name=task_name):
        normalized = _normalize_task_file_stem(trial_path.stem)
        task_observations[normalized] = _load_observations_from_file(trial_path)
    return task_observations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_param_ranges(raw_ranges: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for name in PARAM_ORDER:
        candidate = raw_ranges.get(name) if isinstance(raw_ranges.get(name), dict) else {}
        try:
            lower = float(candidate.get("min"))
            upper = float(candidate.get("max"))
        except (TypeError, ValueError):
            lower = 0.0
            upper = 0.0
        if upper < lower:
            lower, upper = upper, lower
        normalized[name] = {"min": lower, "max": upper}
    return normalized


def _param_ranges_from_observations(observations: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    raw: Dict[str, Dict[str, float]] = {
        name: {"min": float("inf"), "max": float("-inf")}
        for name in PARAM_ORDER
    }
    for row in observations:
        params = row.get("params")
        if not isinstance(params, dict):
            continue
        for name in PARAM_ORDER:
            if name not in params:
                continue
            try:
                parsed = float(params[name])
            except (TypeError, ValueError):
                continue
            raw[name]["min"] = min(raw[name]["min"], parsed)
            raw[name]["max"] = max(raw[name]["max"], parsed)
    for name in PARAM_ORDER:
        if not math.isfinite(raw[name]["min"]) or not math.isfinite(raw[name]["max"]):
            raw[name] = {"min": 0.0, "max": 0.0}
    return _normalize_param_ranges(raw)


def _param_ranges_from_cards(cards: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    raw: Dict[str, Dict[str, float]] = {
        name: {"min": float("inf"), "max": float("-inf")}
        for name in PARAM_ORDER
    }
    for card in cards:
        if not isinstance(card, dict):
            continue
        region = card.get("region") if isinstance(card.get("region"), dict) else {}
        for dim in region.get("dimensions") or []:
            if not isinstance(dim, dict):
                continue
            name = str(dim.get("parameter", "")).strip()
            interval = dim.get("interval")
            if name not in raw or not isinstance(interval, dict):
                continue
            for bound_name in ("lower", "upper"):
                try:
                    parsed = float(interval[bound_name])
                except (KeyError, TypeError, ValueError):
                    continue
                raw[name]["min"] = min(raw[name]["min"], parsed)
                raw[name]["max"] = max(raw[name]["max"], parsed)
        for params_blob in [
            (card.get("perf") or {}).get("best_params") or {},
            (card.get("direction") or {}).get("feasible_median") or {},
            (card.get("direction") or {}).get("elite_median") or {},
        ]:
            if not isinstance(params_blob, dict):
                continue
            for name in PARAM_ORDER:
                if name not in params_blob:
                    continue
                try:
                    parsed = float(params_blob[name])
                except (TypeError, ValueError):
                    continue
                raw[name]["min"] = min(raw[name]["min"], parsed)
                raw[name]["max"] = max(raw[name]["max"], parsed)
    for name in PARAM_ORDER:
        if not math.isfinite(raw[name]["min"]) or not math.isfinite(raw[name]["max"]):
            raw[name] = {"min": 0.0, "max": 0.0}
    return _normalize_param_ranges(raw)


def _focus_params_from_card(card: Dict[str, Any]) -> Dict[str, Any]:
    perf = card.get("perf") or {}
    direction = card.get("direction") or {}
    best_params = perf.get("best_params") or {}
    elite = direction.get("elite_median") or {}
    feasible = direction.get("feasible_median") or {}
    result: Dict[str, Any] = {}
    for name in PARAM_ORDER:
        if name in best_params:
            result[name] = best_params[name]
        elif name in elite:
            result[name] = elite[name]
        elif name in feasible:
            result[name] = feasible[name]
    return result


def _dimension_interval_map(card: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    region = card.get("region") if isinstance(card.get("region"), dict) else {}
    dimensions = region.get("dimensions") or []
    mapped: Dict[str, Dict[str, Any]] = {}
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        name = str(dim.get("parameter", "")).strip()
        interval = dim.get("interval")
        if name in PARAM_ORDER and isinstance(interval, dict):
            mapped[name] = interval
    return mapped


def _interval_span_fraction(interval: Dict[str, Any], param_range: Dict[str, float]) -> float:
    try:
        lower = float(interval.get("lower"))
        upper = float(interval.get("upper"))
    except (TypeError, ValueError):
        return 1.0
    total = max(1e-9, float(param_range["max"]) - float(param_range["min"]))
    width = max(0.0, min(float(param_range["max"]), upper) - max(float(param_range["min"]), lower))
    return _clamp01(width / total)


def _reasoning_payload(
    card: Dict[str, Any],
    *,
    param_ranges: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    feasibility = card.get("feasibility") or {}
    perf = card.get("perf") or {}
    interval_map = _dimension_interval_map(card)
    constrained_parameters = [name for name in PARAM_ORDER if name in interval_map]
    span_fraction: Dict[str, float] = {}
    specificity_terms: List[float] = []
    for name in PARAM_ORDER:
        if name not in interval_map:
            span_fraction[name] = 1.0
            continue
        fraction = _interval_span_fraction(interval_map[name], param_ranges[name])
        span_fraction[name] = fraction
        specificity_terms.append(1.0 - fraction)
    dimension_ratio = float(len(constrained_parameters)) / float(len(PARAM_ORDER))
    precision_score = sum(specificity_terms) / float(len(specificity_terms)) if specificity_terms else 0.0
    specificity_score = _clamp01(0.45 * dimension_ratio + 0.55 * precision_score)

    margin = float(feasibility.get("median_recall_margin", 0.0) or 0.0)
    boundary_proximity = _clamp01(1.0 - min(abs(margin), 0.25) / 0.25)
    support_score = _clamp01(min(1.0, float(feasibility.get("covered_count", 0)) / 20.0))
    best_qps = max(0.0, float(perf.get("best_qps", 0.0) or 0.0))
    median_feasible_qps = max(0.0, float(perf.get("median_feasible_qps", 0.0) or 0.0))
    local_gain_score = _clamp01((best_qps - median_feasible_qps) / max(best_qps, 1.0))
    actionability_score = _clamp01(
        0.5 * specificity_score
        + 0.25 * boundary_proximity
        + 0.15 * support_score
        + 0.10 * local_gain_score
    )
    broad_card = bool(
        len(constrained_parameters) <= 1
        and any(span_fraction[name] >= 0.55 for name in constrained_parameters)
    ) or specificity_score < 0.35

    return {
        "focus_params": _focus_params_from_card(card),
        "constrained_parameters": constrained_parameters,
        "interval_span_fraction": span_fraction,
        "specificity_score": float(specificity_score),
        "boundary_proximity_score": float(boundary_proximity),
        "actionability_score": float(actionability_score),
        "broad_card": broad_card,
    }


def _card_group_key(card: Dict[str, Any]) -> str:
    threshold = float(((card.get("feasibility") or {}).get("recall_threshold", 0.0)) or 0.0)
    best_params = (card.get("perf") or {}).get("best_params") or {}
    payload = {
        "threshold": round(threshold, 6),
        "best_params": {name: best_params.get(name) for name in PARAM_ORDER},
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _select_group_representative(cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        cards,
        key=lambda card: (
            float(((card.get("reasoning") or {}).get("actionability_score", 0.0)) or 0.0),
            float(((card.get("reasoning") or {}).get("specificity_score", 0.0)) or 0.0),
            len(((card.get("reasoning") or {}).get("constrained_parameters") or [])),
            float(((card.get("quality") or {}).get("score", 0.0)) or 0.0),
        ),
    )


def refine_insight_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cards = payload.get("insight_cards") or []
    if not isinstance(cards, list):
        payload["insight_cards"] = []
        payload["retrieval_units"] = []
        return payload

    param_ranges = _normalize_param_ranges(payload.get("param_ranges") or _param_ranges_from_cards(cards))
    enriched: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        enriched_card = dict(card)
        enriched_card.pop("advice_template", None)
        reasoning = _reasoning_payload(enriched_card, param_ranges=param_ranges)
        enriched_card["reasoning"] = reasoning
        grouped.setdefault(_card_group_key(enriched_card), []).append(enriched_card)

    for group_cards in grouped.values():
        representative = _select_group_representative(group_cards)
        merged_ids = [str(item.get("card_id", "")) for item in group_cards if str(item.get("card_id", "")).strip()]
        merged_regions = [str((item.get("region") or {}).get("description", "")) for item in group_cards]
        reasoning = dict(representative.get("reasoning") or {})
        reasoning["duplicate_group_size"] = len(group_cards)
        reasoning["merged_card_ids"] = merged_ids
        reasoning["merged_regions"] = merged_regions
        representative["reasoning"] = reasoning
        enriched.append(representative)

    enriched.sort(
        key=lambda card: (
            float(((card.get("feasibility") or {}).get("recall_threshold", 0.0)) or 0.0),
            -float(((card.get("reasoning") or {}).get("actionability_score", 0.0)) or 0.0),
            -float(((card.get("quality") or {}).get("score", 0.0)) or 0.0),
        ),
    )

    threshold_counts: Dict[float, int] = {}
    for card in enriched:
        threshold = float(((card.get("feasibility") or {}).get("recall_threshold", 0.0)) or 0.0)
        threshold_counts[threshold] = threshold_counts.get(threshold, 0) + 1

    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    summaries = stats.get("threshold_summaries") if isinstance(stats.get("threshold_summaries"), list) else []
    for row in summaries:
        if not isinstance(row, dict):
            continue
        threshold = float(row.get("recall_threshold", 0.0) or 0.0)
        row["card_count"] = int(threshold_counts.get(threshold, 0))
        row["retrieval_unit_count"] = int(threshold_counts.get(threshold, 0))

    payload["schema_version"] = "1.1.0"
    payload["param_ranges"] = param_ranges
    payload["insight_cards"] = enriched
    payload["retrieval_units"] = build_retrieval_units_from_cards(enriched, task_name=str(payload.get("task_name", "")))
    payload["stats"] = {
        **stats,
        "card_count": len(enriched),
        "retrieval_unit_count": len(payload["retrieval_units"]),
        "threshold_summaries": summaries,
    }
    return payload


def build_retrieval_units_from_cards(
    cards: Sequence[Dict[str, Any]],
    task_name: str = "",
) -> List[Dict[str, Any]]:
    task_slug = _slugify(task_name or "task")
    source_doc_id = f"{INSIGHT_SOURCE_DOC_PREFIX}::{task_slug}"
    units: List[Dict[str, Any]] = []
    for idx, card in enumerate(cards, start=1):
        title = f"HNSWLIB Subgroup Insight {idx} ({task_name or task_slug})"
        unit_id = f"{INSIGHT_UNIT_PREFIX}{task_slug}_{idx:03d}_{_slugify(card.get('card_id', title))}"
        perf = card["perf"]
        best_params = perf.get("best_params") or {}
        reasoning = card.get("reasoning") if isinstance(card.get("reasoning"), dict) else {}
        constrained = ",".join(reasoning.get("constrained_parameters") or []) or "none"
        focus_params = reasoning.get("focus_params") or best_params
        text = (
            f"Task: {task_name}\n"
            f"Domain: {DOMAIN}\n"
            f"Region: {card['region']['description']}\n"
            f"Focus params: {json.dumps(focus_params, ensure_ascii=False, sort_keys=True)}\n"
            f"Constrained parameters: {constrained}\n"
            f"Reasoning: actionability={_fmt_number(float(reasoning.get('actionability_score', 0.0) or 0.0))}, "
            f"specificity={_fmt_number(float(reasoning.get('specificity_score', 0.0) or 0.0))}, "
            f"boundary_proximity={_fmt_number(float(reasoning.get('boundary_proximity_score', 0.0) or 0.0))}, "
            f"duplicate_group_size={int(reasoning.get('duplicate_group_size', 1) or 1)}\n"
            f"Quality score: {_fmt_number(float(card['quality']['score']))}\n"
            f"Feasibility: recall_threshold={_fmt_number(float(card['feasibility']['recall_threshold']))}, "
            f"covered={card['feasibility']['covered_count']}, "
            f"feasible={card['feasibility']['feasible_count']}, "
            f"median_recall_margin={_fmt_number(float(card['feasibility']['median_recall_margin']))}\n"
            f"Performance: median_feasible_qps={_fmt_number(float(perf['median_feasible_qps']))}, "
            f"median_elite_qps={_fmt_number(float(perf['median_elite_qps']))}, "
            f"best_qps={_fmt_number(float(perf.get('best_qps', 0.0)))}, "
            f"best_recall={_fmt_number(float(perf.get('best_recall', 0.0)))}, "
            f"best_params={json.dumps(best_params, ensure_ascii=False, sort_keys=True)}\n"
            f"Advice: {card.get('advice', card.get('advice_template', ''))}"
        )
        units.append(
            {
                "unit_id": unit_id,
                "source_doc_id": source_doc_id,
                "section_title": title,
                "text": text,
            }
        )
    return units


def build_task_insight_payload(
    *,
    task_name: str,
    observations: Sequence[Dict[str, Any]],
    recall_threshold: float,
    recall_thresholds: Sequence[float] | None = None,
    lsqm_max_bins: int,
    max_subgroup_dims: int,
    min_covered: int,
    min_feasible: int,
    elite_ratio: float,
    overlap_threshold: float,
    top_k_cards: int,
    quality_weights: Dict[str, float],
    llm_advice: bool,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_max_cards: int,
) -> Dict[str, Any]:
    cards: List[Dict[str, Any]] = []
    retrieval_units: List[Dict[str, Any]] = []
    subgroup_count = 0
    threshold_summaries: List[Dict[str, Any]] = []
    resolved_thresholds = _normalize_recall_thresholds(
        recall_threshold=recall_threshold,
        recall_thresholds=recall_thresholds,
    )
    multi_threshold = len(resolved_thresholds) > 1
    task_slug = _slugify(task_name or "task")

    if observations:
        param_ranges = _param_ranges_from_observations(observations)
        discretization = build_parameter_discretization(
            observations=observations,
            param_order=PARAM_ORDER,
            max_bins=lsqm_max_bins,
        )
        for threshold in resolved_thresholds:
            raw_subgroups = discover_subgroups(
                observations=observations,
                discretization=discretization,
                recall_threshold=threshold,
                param_order=PARAM_ORDER,
                max_subgroup_dims=max_subgroup_dims,
                min_covered=min_covered,
                min_feasible=min_feasible,
                elite_ratio=elite_ratio,
            )
            subgroup_count += len(raw_subgroups)
            scored_subgroups = score_subgroups(raw_subgroups, weights=quality_weights)
            selected_subgroups = select_non_redundant_subgroups(
                scored_subgroups,
                overlap_threshold=overlap_threshold,
                top_k=top_k_cards,
            )
            threshold_cards = build_insight_cards(
                selected_subgroups=selected_subgroups,
                observations=observations,
                recall_threshold=threshold,
                task_name=task_name,
                param_order=PARAM_ORDER,
            )
            for idx, card in enumerate(threshold_cards, start=1):
                if multi_threshold:
                    card["card_id"] = f"hnswlib_insight_{task_slug}_{_threshold_slug(threshold)}_{idx:03d}"
                else:
                    card["card_id"] = f"hnswlib_insight_{task_slug}_{idx:03d}"
                card["domain"] = DOMAIN
            cards.extend(threshold_cards)
            threshold_summaries.append(
                {
                    "recall_threshold": float(threshold),
                    "candidate_subgroup_count": len(raw_subgroups),
                    "card_count": len(threshold_cards),
                    "retrieval_unit_count": len(threshold_cards),
                }
            )
        cards = maybe_rewrite_advice_with_llm(
            cards=cards,
            enabled=llm_advice,
            model_name=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            max_cards=llm_max_cards,
        )
        payload = {
            "schema_version": "1.0.0",
            "domain": DOMAIN,
            "task_name": task_name,
            "generated_at": _utc_now_iso(),
            "param_order": list(PARAM_ORDER),
            "param_ranges": param_ranges,
            "build_params": {
                "method": "lsqm_subgroup_insight",
                "recall_threshold": resolved_thresholds[0],
                "recall_thresholds": list(resolved_thresholds),
                "lsqm_max_bins": lsqm_max_bins,
                "max_subgroup_dims": max_subgroup_dims,
                "min_covered": min_covered,
                "min_feasible": min_feasible,
                "elite_ratio": elite_ratio,
                "overlap_threshold": overlap_threshold,
                "top_k_cards": top_k_cards,
                "quality_weights": dict(quality_weights),
                "llm_advice": bool(llm_advice),
            },
            "stats": {
                "observation_count": len(observations),
                "candidate_subgroup_count": subgroup_count,
                "card_count": len(cards),
                "retrieval_unit_count": len(cards),
                "threshold_summaries": threshold_summaries,
            },
            "insight_cards": cards,
            "retrieval_units": [],
        }
        return refine_insight_payload(payload)
    else:
        threshold_summaries = [
            {
                "recall_threshold": float(threshold),
                "candidate_subgroup_count": 0,
                "card_count": 0,
                "retrieval_unit_count": 0,
            }
            for threshold in resolved_thresholds
        ]

    return refine_insight_payload({
        "schema_version": "1.0.0",
        "domain": DOMAIN,
        "task_name": task_name,
        "generated_at": _utc_now_iso(),
        "param_order": list(PARAM_ORDER),
        "param_ranges": _normalize_param_ranges({}),
        "build_params": {
            "method": "lsqm_subgroup_insight",
            "recall_threshold": resolved_thresholds[0],
            "recall_thresholds": list(resolved_thresholds),
            "lsqm_max_bins": lsqm_max_bins,
            "max_subgroup_dims": max_subgroup_dims,
            "min_covered": min_covered,
            "min_feasible": min_feasible,
            "elite_ratio": elite_ratio,
            "overlap_threshold": overlap_threshold,
            "top_k_cards": top_k_cards,
            "quality_weights": dict(quality_weights),
            "llm_advice": bool(llm_advice),
        },
        "stats": {
            "observation_count": len(observations),
            "candidate_subgroup_count": subgroup_count,
            "card_count": len(cards),
            "retrieval_unit_count": len(retrieval_units),
            "threshold_summaries": threshold_summaries,
        },
        "insight_cards": cards,
        "retrieval_units": retrieval_units,
    })


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_task_payload(output_dir: Path, task_name: str, payload: Dict[str, Any]) -> Path:
    output_path = output_dir / f"{_normalize_task_file_stem(task_name)}.json"
    _write_json(output_path, payload)
    return output_path


def rebuild_insight_index(output_dir: Path) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("domain") != DOMAIN:
            continue
        task_name = str(payload.get("task_name") or path.stem).strip()
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        tasks.append(
            {
                "task_name": task_name,
                "file": path.name,
                "observation_count": int(stats.get("observation_count", 0)),
                "card_count": int(stats.get("card_count", 0)),
                "retrieval_unit_count": int(stats.get("retrieval_unit_count", 0)),
                "generated_at": payload.get("generated_at", ""),
            }
        )

    index_payload = {
        "schema_version": "1.0.0",
        "domain": DOMAIN,
        "generated_at": _utc_now_iso(),
        "insights_dir": str(output_dir.as_posix()),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    _write_json(output_dir / "index.json", index_payload)
    return index_payload


def build_task_insights(
    *,
    trials_dir: Path,
    output_dir: Path,
    task_name: str | None,
    recall_threshold: float,
    recall_thresholds: Sequence[float] | None = None,
    lsqm_max_bins: int,
    max_subgroup_dims: int,
    min_covered: int,
    min_feasible: int,
    elite_ratio: float,
    overlap_threshold: float,
    top_k_cards: int,
    quality_weight_feasible_count: float,
    quality_weight_recall_margin: float,
    quality_weight_feasible_qps: float,
    llm_advice: bool,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_max_cards: int,
) -> Dict[str, Any]:
    grouped = load_task_observations(trials_dir=trials_dir, task_name=task_name)
    quality_weights = {
        "feasible_count": float(quality_weight_feasible_count),
        "recall_margin": float(quality_weight_recall_margin),
        "feasible_qps": float(quality_weight_feasible_qps),
    }

    built: List[Dict[str, Any]] = []
    for name in sorted(grouped.keys()):
        observations = grouped[name]
        payload = build_task_insight_payload(
            task_name=name,
            observations=observations,
            recall_threshold=recall_threshold,
            recall_thresholds=recall_thresholds,
            lsqm_max_bins=lsqm_max_bins,
            max_subgroup_dims=max_subgroup_dims,
            min_covered=min_covered,
            min_feasible=min_feasible,
            elite_ratio=elite_ratio,
            overlap_threshold=overlap_threshold,
            top_k_cards=top_k_cards,
            quality_weights=quality_weights,
            llm_advice=llm_advice,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_max_cards=llm_max_cards,
        )
        output_path = _write_task_payload(output_dir, name, payload)
        built.append(
            {
                "task_name": name,
                "observation_count": len(observations),
                "card_count": int(payload.get("stats", {}).get("card_count", 0)),
                "output_path": str(output_path),
            }
        )

    index_payload = rebuild_insight_index(output_dir)
    return {
        "generated_at": _utc_now_iso(),
        "trials_dir": str(trials_dir),
        "output_dir": str(output_dir),
        "requested_task": _normalize_task_name(task_name) if task_name else None,
        "requested_task": _normalize_task_file_stem(task_name) if task_name else None,
        "built": built,
        "index": index_payload,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HNSWLIB tuning insights per task from trials JSONL."
    )
    parser.add_argument("--trials-dir", default="results/hnswlib/trials")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--output-dir", default="data/hnswlib_insights")
    parser.add_argument(
        "--recall-threshold",
        dest="recall_thresholds",
        action="append",
        type=float,
        default=None,
        help="Feasibility threshold. Repeat to build multiple thresholds.",
    )
    parser.add_argument("--lsqm-max-bins", type=int, default=4)
    parser.add_argument("--max-subgroup-dims", type=int, default=3)
    parser.add_argument("--min-covered", type=int, default=5)
    parser.add_argument("--min-feasible", type=int, default=2)
    parser.add_argument("--elite-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-threshold", type=float, default=0.6)
    parser.add_argument("--top-k-cards", type=int, default=12)
    parser.add_argument("--quality-weight-feasible-count", type=float, default=DEFAULT_WEIGHT_FEASIBLE_COUNT)
    parser.add_argument("--quality-weight-recall-margin", type=float, default=DEFAULT_WEIGHT_RECALL_MARGIN)
    parser.add_argument("--quality-weight-feasible-qps", type=float, default=DEFAULT_WEIGHT_FEASIBLE_QPS)
    parser.add_argument("--llm-advice", action="store_true")
    parser.add_argument("--llm-model", default="", help="Model name; defaults to LLM_MODEL_NAME from .env.")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-max-cards", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    recall_thresholds = _normalize_recall_thresholds(recall_thresholds=args.recall_thresholds)
    summary = build_task_insights(
        trials_dir=Path(args.trials_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        task_name=args.task_name.strip() or None,
        recall_threshold=recall_thresholds[0],
        recall_thresholds=recall_thresholds,
        lsqm_max_bins=int(args.lsqm_max_bins),
        max_subgroup_dims=int(args.max_subgroup_dims),
        min_covered=int(args.min_covered),
        min_feasible=int(args.min_feasible),
        elite_ratio=float(args.elite_ratio),
        overlap_threshold=float(args.overlap_threshold),
        top_k_cards=int(args.top_k_cards),
        quality_weight_feasible_count=float(args.quality_weight_feasible_count),
        quality_weight_recall_margin=float(args.quality_weight_recall_margin),
        quality_weight_feasible_qps=float(args.quality_weight_feasible_qps),
        llm_advice=bool(args.llm_advice),
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_max_cards=int(args.llm_max_cards),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

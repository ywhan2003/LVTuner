"""UNIFY (HSIG) tuning pipeline — Stage A planning + Stage B iterative tuning.

Modeled on the HNSW pipeline core: knowledge-driven cold-start planning
followed by per-round LLM diagnosis and candidate proposal.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from agents.unify_agent import BUILD_PARAM_ORDER, PARAM_ORDER, UNIFTuningAgent, params_to_key
from configs import CommonConfig
from utils.current_task_memory import (
    CurrentTaskMemory,
    _UNIFY_KNOBS,
    _UNIFY_DEFAULT_KNOB_WEIGHTS,
    _UNIFY_PATTERN_WEIGHTS,
)
from utils.knowledge_loader import build_knowledge_context as _kb_build_context
from utils.rfanns_metrics import (
    append_jsonl,
    append_trial,
    aggregate_success_trials,
    compute_pareto_front,
    compute_threshold_summary,
    load_trials,
    utc_now_iso,
    write_json,
)
from utils.rfanns_runner import RunnerConfig, build_benchmark_command, run_trial
from utils.static_knowledge import StaticKnowledgeBase, StaticKnowledgeSelector
from functions.hnswlib_tune import (
    _load_json_object,
    _load_subgroup_init_payload,
    _project_trial_for_file,
    _select_subgroup_init_card,
    _subgroup_init_range,
    _subgroup_init_seeds,
    _trial_params_key,
)
from utils.posterior_proposal_checker import hard_reject

logger = logging.getLogger("unify.pipeline")


# ── Config loading ─────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "search" not in cfg or "recall_threshold" not in (cfg.get("search") or {}):
        raise ValueError(f"{config_path} must set search.recall_threshold")
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_max_workers(value: Any) -> int:
    if value == "auto":
        return min(8, os.cpu_count() or 1)
    workers = int(value)
    if workers <= 0:
        raise ValueError("execution.max_workers must be > 0")
    return workers


def _resolve_max_rounds(value: Any, budget: int) -> int:
    if value == "auto":
        return max(1, budget + 2)
    rounds = int(value)
    if rounds <= 0:
        raise ValueError("max_rounds must be > 0 or 'auto'.")
    return rounds


def _attempted_task_keys(
    trials: Sequence[Dict[str, Any]],
    order: Sequence[str],
) -> set[Tuple[Any, ...]]:
    keys: set[Tuple[Any, ...]] = set()
    for trial in trials:
        params = trial.get("params")
        if not isinstance(params, dict):
            continue
        try:
            keys.add(params_to_key(params, order))
        except Exception:
            continue
    return keys


def _build_tasks(
    candidates: Sequence[Dict[str, Any]],
    needed_runs: int,
    occupied_task_keys: set[Tuple[Any, ...]],
    order: Sequence[str],
    round_idx: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if needed_runs <= 0:
        return tasks
    for candidate in candidates:
        params = candidate.get("params", {})
        tasks.append({
            "stage": "unified",
            "params": params,
            "repeat_idx": 0,
            "proposal_source": candidate.get("source", "unknown"),
            "proposal_round": round_idx,
            "proposal_note": candidate.get("note", ""),
        })
        if len(tasks) >= needed_runs:
            break
    return tasks


def _build_memory_observation(
    last_trial: Dict[str, Any] | None,
    stage_trials: Sequence[Dict[str, Any]],
    recall_threshold: float,
) -> Dict[str, Any] | None:
    """Build a memory-compatible observation dict from the last trial."""
    if last_trial is None:
        return None
    metrics = last_trial.get("metrics") or {}
    params = last_trial.get("params") or {}
    if not isinstance(metrics, dict) or not isinstance(params, dict):
        return None
    recall = float(metrics.get("recall", 0))
    if recall <= 0:
        return None

    # Compute best feasible QPS from all stage trials
    best_qps = 0.0
    for t in stage_trials:
        tm = t.get("metrics") or {}
        tr = float(tm.get("recall", 0))
        tq = float(tm.get("qps", 0))
        if tr >= recall_threshold and tq > best_qps:
            best_qps = tq

    diag_metrics = _extract_diag_metrics_for_static(metrics)
    if best_qps > 0:
        diag_metrics["_best_feasible_qps"] = best_qps

    return {
        "config": params,
        "qps": float(metrics.get("qps", 0)),
        "recall": recall,
        "recall_threshold": recall_threshold,
        "diagnostic_metrics": diag_metrics,
    }


def _extract_diag_metrics_for_static(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract diagnostic metrics for static knowledge selection."""
    diag: Dict[str, Any] = {}
    for key in (
        "visited_nodes_per_query",
        "dist_comps_per_query",
        "distance_computations",
        "index_size_mb",
        "out_degree_mean",
        "in_degree_mean",
        "build_time_s",
        "inclusiveness_pct",
        "max_recall",
    ):
        val = metrics.get(key)
        if val is not None:
            diag[key] = val
    # selected_ef / selected_al
    for k in ("selected_ef", "selected_al"):
        if metrics.get(k) is not None:
            diag[k] = metrics[k]
    return diag


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
            append_trial(trials_path, _project_trial_for_file(trial))
            new_trials.append(trial)
    return new_trials


# ── Stage A: Planning ─────────────────────────────────────────────────────

def _stage_a_plan_unify(
    *,
    agent: UNIFTuningAgent,
    stage_policy: Dict[str, Any],
    agentic_cfg: Dict[str, Any],
    benchmark_extra_args: Sequence[str],
    trials_name: str,
    subgroup_init: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Stage A: load knowledge, search for similar tasks, generate seed candidates."""
    base_search_space = agent.space.export_parameter_space()
    if subgroup_init is not None:
        # Whole-task execution range = the mined range narrowed into the yaml
        # base space (yaml params untouched; per-round override narrows only).
        base_search_space = agent.space.normalize_constraints(subgroup_init["mined_specs"])

    # ── 1. Load knowledge base ─────────────────────────────────────────
    knowledge_cfg = agentic_cfg.get("knowledge") or {}
    knowledge_base_dir = str(knowledge_cfg.get("knowledge_base_dir", "knowledge_base"))
    max_context_chars = int(knowledge_cfg.get("max_context_chars", 120000))
    kb_result = _kb_build_context(base_dir=knowledge_base_dir, max_chars=max_context_chars, algorithm="unify")
    knowledge_context = kb_result.get("base_knowledge_full", "")

    # ── 2. Transfer learning: scan for similar historical UNIFY tasks ──
    transfer_cfg = agentic_cfg.get("transfer") or {}
    transfer_enabled = bool(transfer_cfg.get("enabled", True))
    transfer_context: Dict[str, Any] = {"tasks": []}
    cold_start_guidance = ""

    if subgroup_init is None and transfer_enabled:
        models_dir = Path(transfer_cfg.get("models_dir", "results/unify/models")).expanduser().resolve()
        if models_dir.exists():
            # Simple similarity: load meta.json files, score by recall-threshold match.
            scored_tasks: List[Tuple[float, Path, Dict[str, Any]]] = []
            threshold = float(stage_policy["recall_threshold"])
            for meta_path in sorted(models_dir.rglob("*.meta.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                task_threshold = float(meta.get("recall_threshold", 0.0) or 0.0)
                # Score by dataset-name overlap (simplified lexical) + threshold proximity.
                dataset = str(meta.get("dataset", "") or "").lower()
                extra_dataset = ""
                for arg in benchmark_extra_args:
                    if arg.endswith(".hdf5"):
                        extra_dataset = Path(arg).stem.lower()
                lex_score = 1.0 if extra_dataset and extra_dataset in dataset else 0.5
                threshold_gap = abs(task_threshold - threshold)
                thresh_score = math.exp(-10.0 * threshold_gap)
                score = lex_score * 0.4 + thresh_score * 0.6
                if score > 0.3:
                    scored_tasks.append((score, meta_path, meta))

            scored_tasks.sort(key=lambda x: x[0], reverse=True)
            top_k = int(transfer_cfg.get("top_k_tasks", 2))
            trials_per_task = int(transfer_cfg.get("trials_per_task", 5))

            for idx, (score, meta_path, meta) in enumerate(scored_tasks[:top_k]):
                task_name = meta.get("trials_name", meta_path.parent.name)
                task_data: Dict[str, Any] = {
                    "task_name": task_name,
                    "score": score,
                    "recall_threshold": meta.get("recall_threshold"),
                }
                # Load best trials from this task.
                trials_file = meta_path.parent / f"{meta_path.stem.replace('.meta', '')}.jsonl"
                if not trials_file.exists():
                    trials_file = meta_path.with_suffix(".jsonl")
                if trials_file.exists():
                    task_trials = load_trials(trials_file)
                    success = [t for t in task_trials if t.get("status") == "success"]
                    success.sort(
                        key=lambda t: (
                            float((t.get("metrics") or {}).get("recall", 0.0)) >= threshold,
                            float((t.get("metrics") or {}).get("qps", 0.0)),
                        ),
                        reverse=True,
                    )
                    task_data["selected_trials"] = success[:trials_per_task]
                transfer_context["tasks"].append(task_data)

            if transfer_context["tasks"]:
                # Build cold-start guidance from best trials.
                lines = [
                    "## Cold Start — Transfer Learning Guidance",
                    f"Found {len(transfer_context['tasks'])} similar historical task(s):",
                    "",
                ]
                for task in transfer_context["tasks"]:
                    lines.append(f"- **{task['task_name']}** (score={task['score']:.2f})")
                    for trial in (task.get("selected_trials") or [])[:3]:
                        p = trial.get("params", {})
                        m = trial.get("metrics", {})
                        lines.append(
                            f"  - M={p.get('M')}, B={p.get('B')}, efC={p.get('efConstruction')}, "
                            f"ef={m.get('selected_ef', p.get('ef'))}, al={m.get('selected_al', p.get('al'))} "
                            f"→ recall={m.get('recall', 0):.4f} QPS={m.get('qps', 0):.1f}"
                        )
                lines.append("")
                lines.append("**Start with parameters close to these successful configurations.**")
                lines.append("")
                cold_start_guidance = "\n".join(lines)

    # ── 2.5. Insight-card-based guidance (subgroup discovery) ──────────
    insight_seed_candidates: List[Dict[str, Any]] = []
    insight_guidance = ""
    insight_path = transfer_cfg.get("insight_path", "")
    if insight_path:
        insight_path_resolved = Path(insight_path).expanduser()
        if not insight_path_resolved.is_absolute():
            insight_path_resolved = Path.cwd() / insight_path_resolved
        if insight_path_resolved.exists():
            try:
                insight_data = json.loads(insight_path_resolved.read_text(encoding="utf-8"))
            except Exception:
                insight_data = None
            if insight_data:
                all_cards = insight_data.get("insight_cards", [])
                threshold = float(stage_policy["recall_threshold"])
                # Filter cards matching the target recall threshold
                matching = [
                    c for c in all_cards
                    if abs(float((c.get("feasibility") or {}).get("recall_threshold", 0)) - threshold) < 0.001
                ]
                # Sort by quality score descending
                matching.sort(key=lambda c: (c.get("quality") or {}).get("score", 0), reverse=True)
                # Deduplicate by (M, B, efConstruction) for seed extraction
                seen_build: set = set()
                top_k_for_seeds = int(transfer_cfg.get("insight_top_k", 4))
                insight_used_cards = int(transfer_cfg.get("insight_guidance_cards", 6))

                for card in matching:
                    bp = (card.get("perf") or {}).get("best_params") or {}
                    if not bp:
                        continue
                    build_key = (bp.get("M"), bp.get("B"), bp.get("efConstruction"))
                    if build_key in seen_build:
                        continue
                    seen_build.add(build_key)
                    try:
                        canonical = agent.canonicalize({
                            "M": bp["M"], "B": bp["B"],
                            "efConstruction": bp["efConstruction"],
                            "ef": bp.get("ef", 40), "al": bp.get("al", 32),
                        })
                        insight_seed_candidates.append({
                            "params": canonical,
                            "source": "insight_card",
                            "note": f"Insight {card.get('card_id', '')}: "
                                    f"region={card.get('region', {}).get('description', '')}",
                        })
                    except Exception:
                        continue
                    if len(insight_seed_candidates) >= top_k_for_seeds:
                        break

                # ── Build insight-based cold-start guidance for LLM ──
                if matching:
                    lines_ig = [
                        "## Subgroup Discovery Insights",
                        f"(from {Path(insight_path).name} @ recall {threshold:.2f})",
                        "",
                        "The following insight cards were discovered via LSQM subgroup mining",
                        "on a related range-filtered ANN task. Use them to guide your initial",
                        "parameter selection — but adapt to the current dataset.",
                        "",
                        f"### Top Insight Cards (ranked by quality score, showing top {insight_used_cards}):",
                        "",
                    ]
                    for i, card in enumerate(matching[:insight_used_cards]):
                        q = card.get("quality") or {}
                        perf = card.get("perf") or {}
                        feas = card.get("feasibility") or {}
                        region = card.get("region") or {}
                        bp = perf.get("best_params") or {}
                        lines_ig.append(
                            f"**Card {i+1}** "
                            f"(score={q.get('score', 0):.3f}, "
                            f"type={card.get('type', '?')}):"
                        )
                        lines_ig.append(f"- Region: {region.get('description', 'N/A')}")
                        if bp:
                            lines_ig.append(
                                f"- Best in region: M={bp.get('M')}, B={bp.get('B')}, "
                                f"efConstruction={bp.get('efConstruction')}, "
                                f"ef={bp.get('ef')}, al={bp.get('al')} "
                                f"→ recall={perf.get('best_recall', 0):.4f}, "
                                f"QPS={perf.get('best_qps', 0):.1f}"
                            )
                        lines_ig.append(
                            f"- Feasible rate: {feas.get('feasible_count', 0)}/"
                            f"{feas.get('covered_count', 0)} "
                            f"({feas.get('feasible_ratio', 0)*100:.0f}%), "
                            f"median recall margin: {feas.get('median_recall_margin', 0):+.4f}"
                        )
                        hint = card.get("hint", "")
                        if hint:
                            lines_ig.append(f"- Hint: {hint}")
                        advice = card.get("advice", "")
                        if advice:
                            lines_ig.append(f"- Advice: {advice}")
                        lines_ig.append("")

                    # Key observations
                    lines_ig.extend([
                        "### Key Observations from Subgroup Mining:",
                        "1. These cards come from a DIFFERENT dataset — treat as suggestive, not prescriptive",
                        "2. Most cards are marked 'unstable' — small parameter changes can flip feasibility",
                        "3. Use the region descriptions to understand which parameter ranges are promising",
                        "4. Start with parameters in a region that balances feasibility rate and QPS",
                        "",
                        "**Your task**: Based on these insights, propose a starting",
                        "(M, B, efConstruction, ef, al) for the CURRENT dataset.",
                        "Consider both the successful regions and the instability warnings.",
                        "You are NOT required to copy any best_params exactly — use your judgment.",
                        "",
                    ])
                    insight_guidance = "\n".join(lines_ig)

    # ── Merge insight guidance into cold_start_guidance ──
    if insight_guidance:
        if cold_start_guidance:
            cold_start_guidance = insight_guidance + "\n" + cold_start_guidance
        else:
            cold_start_guidance = insight_guidance

    # ── 3. Seed candidates from similar tasks ──────────────────────────
    seed_candidates: List[Dict[str, Any]] = []

    # Priority 1: insight card seeds (as fallback if LLM fails)
    seed_candidates.extend(insight_seed_candidates)

    # Priority 2: transfer learning from historical UNIFY tasks
    for task in transfer_context.get("tasks", []):
        for trial in (task.get("selected_trials") or [])[:2]:
            params = trial.get("params", {})
            if not isinstance(params, dict):
                continue
            try:
                canonical = agent.canonicalize(params)
                seed_candidates.append({
                    "params": canonical,
                    "source": "transfer_seed",
                    "note": f"Seed from {task.get('task_name', 'unknown')}",
                })
            except Exception:
                continue

    # Fallback: baseline if nothing else is available.
    if not seed_candidates:
        from agents.unify_agent import BASELINE_PARAMS
        try:
            canonical = agent.canonicalize(dict(BASELINE_PARAMS))
            seed_candidates.append({
                "params": canonical,
                "source": "baseline",
                "note": "Default baseline (no similar tasks found)",
            })
        except Exception:
            pass

    stage_a_report = {
        "space_frozen": True,
        "refinement_source": "knowledge_base_and_transfer",
        "used_similar_tasks": [
            t.get("task_name", "") for t in transfer_context.get("tasks", [])
        ],
        "insight_path_used": insight_path if insight_path else None,
        "insight_seed_count": len(insight_seed_candidates),
        "seed_candidate_count": len(seed_candidates),
    }

    if subgroup_init is not None:
        # JSON-only initialization: seeds come exclusively from the mined card.
        seed_candidates = list(subgroup_init.get("seeds") or [])
        card = subgroup_init["card"]
        perf = card.get("perf") or {}
        region = card.get("region") or {}
        cold_start_guidance = (
            "## Subgroup Mining Initialization\n"
            f"- Mined region: {region.get('description', 'N/A')}\n"
            f"- Best point: {json.dumps(perf.get('best_params') or {})} "
            f"(recall={perf.get('best_recall', 0):.4f}, QPS={perf.get('best_qps', 0):.1f})\n"
        )

    return {
        "generated_at": utc_now_iso(),
        "base_search_space": base_search_space,
        "frozen_search_space": copy.deepcopy(base_search_space),
        "transfer_context": transfer_context,
        "knowledge_context": knowledge_context,
        "cold_start_guidance": cold_start_guidance,
        "stage_a_report": stage_a_report,
        "initial_design_seed_candidates": seed_candidates,
        "subgroup_init": (
            {
                "json_path": subgroup_init.get("json_path", ""),
                "card_id": subgroup_init.get("card_id", ""),
                "mined_specs": copy.deepcopy(subgroup_init.get("mined_specs") or {}),
                "seed_candidate_count": len(subgroup_init.get("seeds") or []),
            }
            if subgroup_init is not None
            else None
        ),
    }


# ── Stage B + main pipeline ────────────────────────────────────────────────

def run_pipeline(
    config_path: str,
    resume: bool = True,
    dry_run: bool = False,
) -> int:
    """Run the complete UNIFY tuning pipeline (Stage A → Stage B)."""
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_name = str((cfg.get("output") or {}).get("trials_name", "")).strip()
    if not trials_name:
        raise ValueError("output.trials_name must be provided and non-empty.")

    trials_path = output_dir / "trials" / f"{trials_name}.jsonl"
    proposal_logs_memory: List[Dict[str, Any]] = []

    # Backup if not resuming.
    if not resume:
        if trials_path.exists():
            backup = trials_path.with_suffix(f".backup.{utc_now_iso().replace(':', '-')}.jsonl")
            shutil.move(str(trials_path), str(backup))

    existing_trials = load_trials(trials_path) if resume else []

    # ── Resolve config ─────────────────────────────────────────────────
    budget = int(cfg["search"]["budget"])
    recall_threshold = float(cfg["search"]["recall_threshold"])
    recall_slack = float(cfg["search"].get("recall_slack", 0.005))
    max_workers = _resolve_max_workers(cfg["execution"]["max_workers"])
    agentic_cfg = cfg.get("agentic") or {}
    model_cfg = _resolve_model_cfg(agentic_cfg)
    max_rounds = _resolve_max_rounds(agentic_cfg.get("max_rounds", "auto"), budget)

    if not (0.0 < recall_threshold <= 1.0):
        raise ValueError("search.recall_threshold must be in (0, 1].")

    # ── Build agent ────────────────────────────────────────────────────
    agent = UNIFTuningAgent(
        cfg["params"],
        seed=int(cfg["search"]["seed"]),
        agentic_cfg=agentic_cfg,
        model_cfg=model_cfg,
        llm_caller=None,
    )

    # ── Subgroup-mining initialization (optional, hnsw-style) ─────────
    subgroup_init: Dict[str, Any] | None = None
    subgroup_init_cfg = (
        agentic_cfg.get("subgroup_init") if isinstance(agentic_cfg.get("subgroup_init"), dict) else {}
    )
    if subgroup_init_cfg:
        json_path_raw = subgroup_init_cfg.get("json_path")
        if not isinstance(json_path_raw, str) or not json_path_raw.strip():
            raise ValueError("agentic.subgroup_init.json_path must be a string path to an insight JSON.")
        payload = _load_subgroup_init_payload(json_path_raw.strip(), domain="unify_tuning")
        raw_path = Path(json_path_raw.strip()).expanduser().resolve()
        raw_payload = _load_json_object(raw_path)
        card = _select_subgroup_init_card(raw_payload if raw_payload.get("insight_cards") else payload)
        subgroup_init = {
            "json_path": str(raw_path),
            "payload": payload,
            "card_id": str(card.get("card_id", "")),
            "card": card,
        }
        base_specs = agent.space.export_parameter_space()
        subgroup_init["mined_specs"] = _subgroup_init_range(
            subgroup_init["card"], base_specs, param_order=agent.param_order
        )
        subgroup_init["seeds"] = _subgroup_init_seeds(
            agent,
            subgroup_init["card"],
            agent.space.normalize_constraints(subgroup_init["mined_specs"]),
            param_order=agent.param_order,
        )
        logger.info(
            "Subgroup-mining init: card=%s mined=%s seeds=%d (transfer bootstrap skipped)",
            subgroup_init["card_id"],
            json.dumps(subgroup_init["mined_specs"], ensure_ascii=False),
            len(subgroup_init["seeds"]),
        )

    # ── Initialize current-task memory ─────────────────────────────────
    memory_cfg = agentic_cfg.get("current_task_memory") or {}
    memory_enabled = bool(memory_cfg.get("enabled", True))
    current_memory: CurrentTaskMemory | None = None
    if memory_enabled:
        current_memory = CurrentTaskMemory(
            task_context={
                "recall_threshold": recall_threshold,
                "param_order": list(BUILD_PARAM_ORDER),
            },
            parameter_space=agent.space,
            output_dir=output_dir,
            trials_name=trials_name,
            knobs=_UNIFY_KNOBS,
            default_knob_weights=_UNIFY_DEFAULT_KNOB_WEIGHTS,
            pattern_weights=_UNIFY_PATTERN_WEIGHTS,
            algorithm_name="UNIFY",
        )
        # Warm up memory from existing (resumed) trials
        if existing_trials:
            unified = [t for t in existing_trials if t.get("stage") == "unified"]
            current_memory.warm_up(unified)

    # ── Initialize static knowledge ────────────────────────────────────
    knowledge_cfg = agentic_cfg.get("knowledge") or {}
    static_kb: StaticKnowledgeBase | None = None
    static_selector: StaticKnowledgeSelector | None = None
    if knowledge_cfg.get("static_enabled", True):
        try:
            static_kb = StaticKnowledgeBase.for_algorithm("UNIFY")
            # Pass LLM caller for description-based knowledge matching.
            _knowledge_llm = lambda prompt: agent._invoke_llm(prompt)
            static_selector = StaticKnowledgeSelector(static_kb, llm_caller=_knowledge_llm)
        except Exception:
            logger.warning("Failed to load UNIFY static knowledge cards.", exc_info=True)
            static_kb = None
            static_selector = None

    # ── Build runner config ────────────────────────────────────────────
    benchmark_cfg = cfg["benchmark"]
    runner_cfg = RunnerConfig(
        python_bin=benchmark_cfg["python_bin"],
        script_path=str(Path(benchmark_cfg["script_path"]).resolve()),
        metrics_output_arg=benchmark_cfg.get("metrics_output_arg", "--metrics-output"),
        timeout_s=int(cfg["execution"]["timeout_s"]),
        retries=int(cfg["execution"]["retries"]),
        output_dir=str(output_dir),
        workdir=str(Path.cwd()),
        extra_args=list(benchmark_cfg.get("extra_args", [])),
        param_args=dict(benchmark_cfg.get("param_args") or {}),
    )

    script_path = Path(runner_cfg.script_path)
    if not script_path.exists() and not dry_run:
        raise FileNotFoundError(f"Benchmark entry not found: {script_path}")

    stage_policy = {
        "recall_threshold": recall_threshold,
        "recall_slack": recall_slack,
    }

    # ── Stage A: Planning ──────────────────────────────────────────────
    stage_a_plan = _stage_a_plan_unify(
        agent=agent,
        stage_policy=stage_policy,
        agentic_cfg=agentic_cfg,
        benchmark_extra_args=runner_cfg.extra_args,
        trials_name=trials_name,
        subgroup_init=subgroup_init,
    )

    seed_candidates = list(stage_a_plan.get("initial_design_seed_candidates", []))
    cold_start_guidance = stage_a_plan.get("cold_start_guidance", "")

    # ── Stage B: Iterative tuning ─────────────────────────────────────
    all_trials = list(existing_trials)
    planned_task_keys = set(_attempted_task_keys(all_trials, BUILD_PARAM_ORDER))
    seed_idx = 0
    prev_mem_full_state: Dict[str, Any] | None = None
    # Initialize prev_mem_full_state from resumed trials if memory is active
    if current_memory is not None and all_trials:
        last_success = None
        for t in reversed(all_trials):
            if t.get("status") == "success":
                last_success = t
                break
        if last_success:
            obs = _build_memory_observation(last_success, all_trials, recall_threshold)
            if obs:
                prev_mem_full_state = current_memory.build_full_state(obs)

    for round_idx in range(1, max_rounds + 1):
        success_count = sum(1 for t in all_trials if t.get("status") == "success")
        # Initialization seeds do not consume the tuning budget.
        seed_param_keys = {
            _trial_params_key(c["params"]) for c in seed_candidates if isinstance(c.get("params"), dict)
        }
        seed_success = sum(
            1
            for t in all_trials
            if t.get("status") == "success"
            and isinstance(t.get("params"), dict)
            and _trial_params_key(t["params"]) in seed_param_keys
        )
        remaining = max(0, budget - (success_count - seed_success))
        if remaining <= 0:
            logger.info("Budget exhausted (%d success), stopping.", success_count)
            break

        # ── Per-round accumulators ──
        memory_gen_llm_time_s = 0.0
        memory_gen_prompt_tokens = 0
        memory_gen_completion_tokens = 0

        stage_trials = [t for t in all_trials if t.get("stage") == "unified"]
        occupied = set(planned_task_keys) if dry_run else set(
            _attempted_task_keys(all_trials, BUILD_PARAM_ORDER)
        )

        # Build root state.
        root_state = agent.build_root_state(
            round_idx=round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
        )

        # Cold start → use seed candidates (only if no guidance for LLM).
        last_trial = stage_trials[-1] if stage_trials else None

        # ── Per-round timing and token tracking ──
        memory_retrieval_time_s = 0.0
        knowledge_selection_time_s = 0.0
        memory_context_str = ""
        static_knowledge_str = ""
        knowledge_llm_time_s = 0.0
        knowledge_llm_prompt_tokens = 0
        knowledge_llm_completion_tokens = 0
        llm_call_time_s = 0.0
        mem_full_state = None
        round_token_snapshot = agent.token_snapshot()

        if last_trial is None and seed_idx < len(seed_candidates) and (
            not cold_start_guidance or subgroup_init is not None
        ):
            candidate = seed_candidates[seed_idx]
            seed_idx += 1
            diag_log: Dict[str, Any] = {
                "ok": True, "diagnosis": {"strategy": "cold_start_seed"},
                "candidate": candidate, "classification": "cold_start",
                "tuning_action": {}, "error": "", "source": "stage_a_seed",
                "prompt_payload": {},
            }
        else:
            # ── Retrieve memory context ──────────────────────────────────
            memory_context_str = ""
            if current_memory is not None:
                try:
                    t_mem_start = time.perf_counter()
                    # Build memory observation from last trial
                    mem_obs = _build_memory_observation(last_trial, stage_trials, recall_threshold)
                    mem_full_state = current_memory.build_full_state(mem_obs) if mem_obs else None
                    if mem_full_state:
                        mem_ctx = current_memory.retrieve_memory_context(
                            current_full_state=mem_full_state,
                            current_config=mem_obs.get("config") if mem_obs else None,
                        )
                        if mem_ctx:
                            memory_context_str = CurrentTaskMemory.format_memory_context(
                                mem_ctx, knobs=_UNIFY_KNOBS,
                            )
                    memory_retrieval_time_s = round(time.perf_counter() - t_mem_start, 4)
                except Exception:
                    logger.warning("Memory context retrieval failed.", exc_info=True)

            # ── Select static knowledge ──────────────────────────────────
            static_knowledge_str = ""
            knowledge_llm_time_s = 0.0
            knowledge_llm_prompt_tokens = 0
            knowledge_llm_completion_tokens = 0
            if static_selector is not None and last_trial is not None:
                try:
                    t_know_start = time.perf_counter()
                    know_snap_before = agent.token_snapshot()
                    metrics = last_trial.get("metrics") or {}
                    params = last_trial.get("params") or {}
                    observation = {
                        "config": params,
                        "qps": float(metrics.get("qps", 0)),
                        "recall": float(metrics.get("recall", 0)),
                        "recall_threshold": recall_threshold,
                        "diagnostic_metrics": _extract_diag_metrics_for_static(metrics),
                    }
                    task_desc = {"algorithm": "UNIFY", "recall_threshold": recall_threshold}
                    selected = static_selector.select_best_match(
                        task_desc, observation, full_state=mem_full_state,
                    )
                    static_knowledge_str = StaticKnowledgeSelector.format_knowledge_context_for_llm(selected)
                    know_snap_after = agent.token_snapshot()
                    knowledge_selection_time_s = round(time.perf_counter() - t_know_start, 4)
                    knowledge_llm_time_s = knowledge_selection_time_s  # includes LLM + rule matching
                    knowledge_llm_prompt_tokens = know_snap_after[0] - know_snap_before[0]
                    knowledge_llm_completion_tokens = know_snap_after[1] - know_snap_before[1]
                except Exception:
                    logger.warning("Static knowledge selection failed.", exc_info=True)

            # LLM diagnosis (also used for cold start when insight guidance is available).
            t_llm_start = time.perf_counter()
            candidate, diag_log = agent.diagnose_last_execution(
                last_trial=last_trial,
                root_state=root_state,
                stage_policy=stage_policy,
                cold_start_guidance=cold_start_guidance if not stage_trials else "",
                memory_context=memory_context_str,
                static_knowledge_context=static_knowledge_str,
            )
            llm_call_time_s = round(time.perf_counter() - t_llm_start, 4)

        # LLM proposal is mandatory once stage B has started; only cold-start
        # initial design may fall back to deterministic candidates.
        if candidate is None:
            if stage_trials:
                raise RuntimeError(
                    "LLM diagnosis/proposal failed during stage B "
                    f"(error={diag_log.get('error', 'unknown')}) — rule-based fallback is disabled."
                )
            exhausted = {params_to_key(
                agent.canonicalize((t.get("params") or {})), BUILD_PARAM_ORDER
            ) for t in stage_trials if isinstance(t.get("params"), dict)}
            fallback = agent.initial_design_candidates(
                target_count=1,
                exhausted_keys=exhausted,
                stage_policy=stage_policy,
            )
            candidate = fallback[0] if fallback else None
            diag_log["fallback_used"] = True

        if candidate is None:
            logger.warning("No candidate generated, stopping.")
            break

        # ── Dominance repository hard check (runtime.tex rules) ───────
        interval_table = None
        if current_memory is not None:
            try:
                interval_table = current_memory.build_runtime_structural_interval_table()
                append_jsonl(
                    output_dir / "current_task_memory" / f"{trials_name}.interval_tables.jsonl",
                    {"round_idx": round_idx, "timestamp": utc_now_iso(), **interval_table},
                )
            except Exception:
                interval_table = None
        repo_check_max_attempts = max(
            1, int((agentic_cfg.get("repository_check") or {}).get("max_attempts", 2))
        )
        if candidate is not None and interval_table is not None:
            repo_attempts = 0
            while repo_attempts < repo_check_max_attempts:
                rejected_flag, reject_feedback = hard_reject(
                    interval_table, dict(candidate["params"])
                )
                if not rejected_flag:
                    break
                repo_attempts += 1
                logger.warning("Repository check: %s", reject_feedback)
                re_candidate, _ = agent.diagnose_last_execution(
                    last_trial=last_trial,
                    root_state=root_state,
                    stage_policy=stage_policy,
                    cold_start_guidance=cold_start_guidance if not stage_trials else "",
                    memory_context=memory_context_str,
                    static_knowledge_context=static_knowledge_str,
                    rejection_feedback=reject_feedback,
                )
                if re_candidate is None:
                    break
                candidate = re_candidate
            else:
                exhausted = {params_to_key(
                    agent.canonicalize((t.get("params") or {})), BUILD_PARAM_ORDER
                ) for t in stage_trials if isinstance(t.get("params"), dict)}
                fb = agent.initial_design_candidates(
                    target_count=1, exhausted_keys=exhausted, stage_policy=stage_policy
                )
                safe = [c for c in fb if not hard_reject(interval_table, dict(c["params"]))[0]]
                if safe:
                    candidate = safe[0]
                elif fb:
                    candidate = fb[0]
                    logger.warning(
                        "No repository-safe fallback candidate — using the least-violating one"
                    )

        # Inject recall threshold for benchmark.
        run_params = dict(candidate["params"])
        run_params["_select_recall_threshold"] = recall_threshold
        run_params["_select_recall_slack"] = recall_slack

        tasks = _build_tasks(
            candidates=[{"params": run_params, "source": candidate.get("source", "unknown"),
                        "note": candidate.get("note", "")}],
            needed_runs=min(1, remaining),
            occupied_task_keys=occupied,
            order=agent.param_order,
            round_idx=round_idx,
        )

        if not tasks:
            logger.info("No tasks to execute (all exhausted), stopping.")
            break

        # Token delta for this round.
        round_prompt = agent.total_prompt_tokens - round_token_snapshot[0]
        round_completion = agent.total_completion_tokens - round_token_snapshot[1]

        # Log proposal with two-phase metrics when available.
        proposal_log = {
            "round_idx": round_idx,
            "root_state": root_state,
            "diagnosis": diag_log.get("diagnosis", {}),
            "classification": diag_log.get("classification", ""),
            "tuning_action": diag_log.get("tuning_action", {}),
            "candidate": candidate,
            "stage_policy": stage_policy,
            "generated_at": utc_now_iso(),
            "memory_enabled": current_memory is not None,
            "static_knowledge_enabled": static_selector is not None,
            "memory_context_chars": len(memory_context_str) if memory_context_str else 0,
            "static_knowledge_chars": len(static_knowledge_str) if static_knowledge_str else 0,
            # ── Knowledge / memory / combined LLM timing ──
            "knowledge_selection_time_s": knowledge_selection_time_s,
            "knowledge_llm_time_s": knowledge_llm_time_s,
            "knowledge_llm_prompt_tokens": knowledge_llm_prompt_tokens,
            "knowledge_llm_completion_tokens": knowledge_llm_completion_tokens,
            "memory_retrieval_time_s": memory_retrieval_time_s,
            "llm_call_time_s": diag_log.get("total_llm_time_s", llm_call_time_s),
            "round_prompt_tokens": diag_log.get("total_prompt_tokens_this_round", round_prompt),
            "round_completion_tokens": diag_log.get("total_completion_tokens_this_round", round_completion),
            # ── Two-phase LLM metrics (diagnosis + proposal) ──
            "diagnosis_llm_time_s": diag_log.get("diagnosis_llm_time_s", 0),
            "diagnosis_prompt_tokens": diag_log.get("diagnosis_prompt_tokens", 0),
            "diagnosis_completion_tokens": diag_log.get("diagnosis_completion_tokens", 0),
            "proposal_llm_time_s": diag_log.get("proposal_llm_time_s", 0),
            "proposal_prompt_tokens": diag_log.get("proposal_prompt_tokens", 0),
            "proposal_completion_tokens": diag_log.get("proposal_completion_tokens", 0),
            # ── Memory generation LLM metrics (filled after update_memory) ──
            "memory_gen_llm_time_s": 0.0,
            "memory_gen_prompt_tokens": 0,
            "memory_gen_completion_tokens": 0,
        }
        if dry_run:
            proposal_log["workload_time_s"] = 0.0
            proposal_logs_memory.append(proposal_log)
            logger.info("Dry-run round %d: candidate=%s", round_idx, candidate["params"])
            planned_task_keys.update(
                params_to_key(t["params"], BUILD_PARAM_ORDER) for t in tasks
            )
            continue

        # Execute.
        new_trials = _execute_tasks(tasks, runner_cfg, trials_path, max_workers)
        all_trials.extend(new_trials)

        # Compute workload time.
        workload_time_s = round(sum(t.get("elapsed_s", 0) for t in new_trials), 4)

        if new_trials:
            t = new_trials[-1]
            m = t.get("metrics") or {}
            logger.info(
                "Round %d: M=%s B=%s efC=%s ef=%s al=%s → recall=%s QPS=%s status=%s",
                round_idx,
                t["params"].get("M"), t["params"].get("B"),
                t["params"].get("efConstruction"),
                m.get("selected_ef", t["params"].get("ef")),
                m.get("selected_al", t["params"].get("al")),
                m.get("recall"), m.get("qps"), t.get("status"),
            )

            # ── Update current-task memory ────────────────────────────
            if current_memory is not None and t.get("status") == "success":
                try:
                    after_obs = _build_memory_observation(t, all_trials, recall_threshold)
                    if after_obs:
                        if prev_mem_full_state is not None and last_trial is not None:
                            # Build action from last to current params
                            lp = (last_trial.get("params") or {}) if isinstance(last_trial.get("params"), dict) else {}
                            cp = t.get("params") or {}
                            action = current_memory.build_action(
                                {k: lp.get(k) for k in BUILD_PARAM_ORDER if k in lp},
                                {k: cp.get(k) for k in BUILD_PARAM_ORDER if k in cp},
                            )
                            # ── Memory LLM caller with token/timing tracking ──
                            mem_snap_before = agent.token_snapshot()
                            t_mem_gen_start = time.perf_counter()

                            def _memory_llm_caller(prompt):
                                return agent._invoke_llm(prompt)

                            reflection = current_memory.update_memory(
                                prev_mem_full_state, action, after_obs,
                                llm_caller=_memory_llm_caller,
                            )
                            mem_snap_after = agent.token_snapshot()
                            memory_gen_llm_time_s = round(time.perf_counter() - t_mem_gen_start, 4)
                            memory_gen_prompt_tokens = mem_snap_after[0] - mem_snap_before[0]
                            memory_gen_completion_tokens = mem_snap_after[1] - mem_snap_before[1]
                            if reflection is not None:
                                logger.info(
                                    "Memory gen: %.3fs +%d/%d tok | outcome=%s",
                                    memory_gen_llm_time_s, memory_gen_prompt_tokens,
                                    memory_gen_completion_tokens,
                                    reflection.get("outcome_label", "?"),
                                )
                            else:
                                logger.info("Memory gen: skipped (reflection=None)")
                        else:
                            current_memory.add_initial_observation(after_obs)
                        prev_mem_full_state = current_memory.build_full_state(after_obs)
                except Exception:
                    logger.warning("Memory update failed.", exc_info=True)

        # ── Finalize and write proposal log (after memory update) ────────
        proposal_log["workload_time_s"] = workload_time_s
        proposal_log["memory_gen_llm_time_s"] = memory_gen_llm_time_s
        proposal_log["memory_gen_prompt_tokens"] = memory_gen_prompt_tokens
        proposal_log["memory_gen_completion_tokens"] = memory_gen_completion_tokens
        proposal_logs_memory.append(proposal_log)

        logger.info(
            "Round %d: diag=%.2fs(+%d/%d) prop=%.2fs(+%d/%d) | "
            "mem=%.3fs(+%d/%d) know=%.3fs(+%d/%d) | "
            "mem_ret=%.3fs workload=%.3fs",
            round_idx,
            diag_log.get("diagnosis_llm_time_s", 0),
            diag_log.get("diagnosis_prompt_tokens", 0),
            diag_log.get("diagnosis_completion_tokens", 0),
            diag_log.get("proposal_llm_time_s", 0),
            diag_log.get("proposal_prompt_tokens", 0),
            diag_log.get("proposal_completion_tokens", 0),
            memory_gen_llm_time_s, memory_gen_prompt_tokens, memory_gen_completion_tokens,
            knowledge_llm_time_s, knowledge_llm_prompt_tokens, knowledge_llm_completion_tokens,
            knowledge_selection_time_s, memory_retrieval_time_s,
            workload_time_s,
        )

    # ── Final report ───────────────────────────────────────────────────────
    stage_trials_final = [t for t in all_trials if t.get("stage") == "unified"]
    success = [t for t in stage_trials_final if t.get("status") == "success"]
    aggregates = aggregate_success_trials(stage_trials_final, order=agent.param_order)
    pareto = compute_pareto_front(aggregates)
    summary = compute_threshold_summary(aggregates, recall_threshold, recall_slack)

    # ── Aggregate timing and tokens from in-memory proposal logs ────────
    proposal_logs: List[Dict[str, Any]] = list(proposal_logs_memory)

    agg_timing: Dict[str, Any] = {
        "total_knowledge_selection_time_s": round(sum(
            pl.get("knowledge_selection_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_knowledge_llm_time_s": round(sum(
            pl.get("knowledge_llm_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_memory_retrieval_time_s": round(sum(
            pl.get("memory_retrieval_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_llm_call_time_s": round(sum(
            pl.get("llm_call_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_workload_time_s": round(sum(
            pl.get("workload_time_s", 0) for pl in proposal_logs
        ), 4),
        # ── Two-phase LLM timing ──
        "total_diagnosis_llm_time_s": round(sum(
            pl.get("diagnosis_llm_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_proposal_llm_time_s": round(sum(
            pl.get("proposal_llm_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_memory_gen_llm_time_s": round(sum(
            pl.get("memory_gen_llm_time_s", 0) for pl in proposal_logs
        ), 4),
        "total_rounds_logged": len(proposal_logs),
    }
    llm_tokens: Dict[str, Any] = {
        "total_prompt_tokens": agent.total_prompt_tokens,
        "total_completion_tokens": agent.total_completion_tokens,
        "init_prompt_tokens": agent.init_prompt_tokens,
        "init_completion_tokens": agent.init_completion_tokens,
        # ── Two-phase token breakdown ──
        # ── Knowledge LLM token breakdown ──
        "total_knowledge_llm_prompt_tokens": sum(
            pl.get("knowledge_llm_prompt_tokens", 0) for pl in proposal_logs
        ),
        "total_knowledge_llm_completion_tokens": sum(
            pl.get("knowledge_llm_completion_tokens", 0) for pl in proposal_logs
        ),
        "total_diagnosis_prompt_tokens": sum(
            pl.get("diagnosis_prompt_tokens", 0) for pl in proposal_logs
        ),
        "total_diagnosis_completion_tokens": sum(
            pl.get("diagnosis_completion_tokens", 0) for pl in proposal_logs
        ),
        "total_proposal_prompt_tokens": sum(
            pl.get("proposal_prompt_tokens", 0) for pl in proposal_logs
        ),
        "total_proposal_completion_tokens": sum(
            pl.get("proposal_completion_tokens", 0) for pl in proposal_logs
        ),
        # ── Memory generation token breakdown ──
        "total_memory_gen_prompt_tokens": sum(
            pl.get("memory_gen_prompt_tokens", 0) for pl in proposal_logs
        ),
        "total_memory_gen_completion_tokens": sum(
            pl.get("memory_gen_completion_tokens", 0) for pl in proposal_logs
        ),
    }

    report_path = output_dir / "summary.json"
    write_json(report_path, {
        "trials_name": trials_name,
        "total_runs": len(stage_trials_final),
        "success_runs": len(success),
        "unique_params": len(aggregates),
        "pareto_count": len(pareto),
        "timing": agg_timing,
        "llm_tokens": llm_tokens,
        **summary,
    })
    logger.info("Pipeline complete. Report: %s", report_path)
    return 0


def _resolve_model_cfg(agentic_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve LLM model configuration from CommonConfig."""
    config_key = str(agentic_cfg.get("model_config_key", "")).strip()
    if config_key:
        cfg = getattr(CommonConfig, config_key, None)
        if isinstance(cfg, dict):
            return cfg
    # Fallback: build from CommonConfig attributes.
    model_cfg: Dict[str, Any] = {}
    for attr in ("model_name", "model", "url", "authorization", "temperature", "max_tokens", "timeout_s"):
        val = getattr(CommonConfig, attr, None)
        if val is not None:
            model_cfg[attr] = val
    return model_cfg


# ── CLI entry point ────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNIFY single-stage tuning pipeline.")
    parser.add_argument("--config", type=str, default="configs/unify_tune.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from existing trials.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume; backup existing trials.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate candidates only; do not execute benchmark.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_pipeline(config_path=args.config, resume=args.resume, dry_run=args.dry_run)


__all__ = [
    "load_config",
    "run_pipeline",
    "parse_args",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())

"""NHQ (Native Hybrid Query) tuning pipeline — Stage A planning + Stage B iterative tuning.

Modeled on the Filter-DiskANN pipeline core: knowledge-driven cold-start planning
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
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from agents.nhq_agent import (
    BASELINE_PARAMS,
    BUILD_PARAM_ORDER,
    PARAM_ORDER,
    NHQTuningAgent,
    params_to_key,
)
from configs import CommonConfig
from utils.hnswlib_metrics import (
    append_jsonl,
    append_trial,
    aggregate_success_trials,
    compute_pareto_front,
    compute_summary,
    compute_threshold_summary,
    load_trials,
    parse_metrics_file,
    utc_now_iso,
    write_json,
)
from utils.hnswlib_runner import RunnerConfig, _load_artifacts_sidecar
from utils.knowledge_loader import build_knowledge_context as _kb_build_context
from utils.static_knowledge import StaticKnowledgeBase, StaticKnowledgeSelector
from functions.hnswlib_tune import (
    _load_json_object,
    _load_subgroup_init_payload,
    _project_trial_for_file,
    _select_subgroup_init_card,
    _subgroup_init_cold_start_cards,
    _subgroup_init_range,
    _subgroup_init_seeds,
    _trial_params_key,
)
from utils.current_task_memory import CurrentTaskMemory, _NHQ_KNOBS
from utils.posterior_proposal_checker import hard_reject

logger = logging.getLogger("nhq.pipeline")


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
        try:
            key = params_to_key(params, order)
        except Exception:
            key = None
        if key is not None and key in occupied_task_keys:
            logger.debug("Skipping already-tried candidate: %s", params)
            continue
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


# ── Benchmark command builder ──────────────────────────────────────────────

def _to_cli_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _build_benchmark_command(
    python_bin: str,
    script_path: str,
    params: Dict[str, Any],
    metrics_output_path: str,
    metrics_output_arg: str = "--metrics-output",
    extra_args: Sequence[str] | None = None,
    param_args: Dict[str, str] | None = None,
) -> List[str]:
    """Build the NHQ benchmark CLI command."""
    resolved_param_args = dict(param_args or {})
    command = [python_bin, os.path.abspath(script_path)]

    for name, value in params.items():
        if name.startswith("_"):
            continue  # skip internal params
        flag = resolved_param_args.get(name)
        if flag is None:
            continue
        if isinstance(value, list):
            command.extend([str(flag)] + [_to_cli_number(v) for v in value])
        else:
            command.extend([str(flag), _to_cli_number(value)])

    command.extend([metrics_output_arg, metrics_output_path])
    if extra_args:
        command.extend(list(extra_args))
    return command


def _run_trial(
    stage: str,
    params: Dict[str, Any],
    repeat_idx: int,
    config: RunnerConfig,
    proposal_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute a single NHQ benchmark trial via subprocess."""
    run_id = str(uuid.uuid4())
    started_at = utc_now_iso()
    run_dir = Path(config.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    command = _build_benchmark_command(
        python_bin=config.python_bin,
        script_path=config.script_path,
        params=params,
        metrics_output_path=str(metrics_path),
        metrics_output_arg=config.metrics_output_arg,
        extra_args=config.extra_args,
        param_args=dict(config.param_args or {}),
    )

    max_attempts = max(1, config.retries + 1)
    start_ts = time.time()
    metrics = None
    error_msg = None
    for attempt in range(1, max_attempts + 1):
        try:
            completed = subprocess.run(
                command,
                cwd=config.workdir,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
                check=False,
            )

            if completed.returncode != 0:
                stderr_tail = (completed.stderr or "").strip()[-2000:]
                error_msg = (
                    f"Command failed (code={completed.returncode}) "
                    f"attempt={attempt}/{max_attempts}"
                )
                if stderr_tail:
                    error_msg += f": {stderr_tail}"
                continue

            metrics = parse_metrics_file(metrics_path)
            error_msg = None
            break
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {config.timeout_s}s attempt={attempt}/{max_attempts}"
        except Exception as exc:
            error_msg = f"Runner exception attempt={attempt}/{max_attempts}: {exc}"

    wall_clock_elapsed_s = round(time.time() - start_ts, 6)
    elapsed_s = wall_clock_elapsed_s
    status = "success" if metrics is not None else "failed"
    benchmark_artifacts = _load_artifacts_sidecar(metrics_path)
    if (
        status == "success"
        and isinstance(metrics, dict)
        and bool(metrics.get("index_reused"))
        and metrics.get("original_build_time_s") is not None
    ):
        try:
            elapsed_s = round(wall_clock_elapsed_s + float(metrics["original_build_time_s"]), 6)
        except (TypeError, ValueError):
            elapsed_s = wall_clock_elapsed_s

    proposal_meta = proposal_meta or {}
    return {
        "run_id": run_id,
        "stage": stage,
        "params": params,
        "metrics": metrics,
        "repeat_idx": repeat_idx,
        "status": status,
        "error": error_msg,
        "started_at": started_at,
        "elapsed_s": elapsed_s,
        "wall_clock_elapsed_s": wall_clock_elapsed_s,
        "command": command,
        "metrics_path": str(metrics_path),
        "executor_script": config.script_path,
        "executor_mode": "nhq_direct",
        "benchmark_artifacts": benchmark_artifacts,
        "proposal_source": proposal_meta.get("proposal_source", "unknown"),
        "proposal_round": proposal_meta.get("proposal_round", -1),
        "proposal_note": proposal_meta.get("proposal_note", ""),
        "task_name": proposal_meta.get("task_name", ""),
    }


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
                _run_trial,
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

def _stage_a_plan_nhq(
    *,
    agent: NHQTuningAgent,
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
    kb_result = _kb_build_context(base_dir=knowledge_base_dir, max_chars=max_context_chars)
    knowledge_context = kb_result.get("base_knowledge_full", "")

    # ── 2. Transfer learning: scan for similar historical tasks ────────
    transfer_cfg = agentic_cfg.get("transfer") or {}
    transfer_enabled = bool(transfer_cfg.get("enabled", True))
    transfer_context: Dict[str, Any] = {"tasks": []}
    cold_start_guidance = ""

    if subgroup_init is None and transfer_enabled:
        models_dir = Path(transfer_cfg.get("models_dir", "results/nhq/models")).expanduser().resolve()
        if models_dir.exists():
            scored_tasks: List[Tuple[float, Path, Dict[str, Any]]] = []
            threshold = float(stage_policy["recall_threshold"])
            for meta_path in sorted(models_dir.rglob("*.meta.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                task_threshold = float(meta.get("recall_threshold", 0.0) or 0.0)
                dataset = str(meta.get("dataset", "") or "").lower()
                extra_dataset = ""
                for arg in benchmark_extra_args:
                    if arg.endswith(".hdf5") or arg.endswith(".fvecs"):
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
                            f"  - M={p.get('M')}, efC={p.get('efConstruction')}, "
                            f"ef={m.get('best_search_list', p.get('ef'))}, weight={p.get('weight')} "
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
                matching = [
                    c for c in all_cards
                    if abs(float((c.get("feasibility") or {}).get("recall_threshold", 0)) - threshold) < 0.001
                ]
                matching.sort(key=lambda c: (c.get("quality") or {}).get("score", 0), reverse=True)
                seen_build: set = set()
                top_k_for_seeds = int(transfer_cfg.get("insight_top_k", 4))
                insight_used_cards = int(transfer_cfg.get("insight_guidance_cards", 6))

                for card in matching:
                    bp = (card.get("perf") or {}).get("best_params") or {}
                    if not bp:
                        continue
                    build_key = (bp.get("M"), bp.get("efConstruction"))
                    if build_key in seen_build:
                        continue
                    seen_build.add(build_key)
                    try:
                        canonical = agent.canonicalize({
                            "M": bp.get("M"),
                            "efConstruction": bp.get("efConstruction"),
                            "ef": bp.get("ef"),
                            "weight": bp.get("weight", 0),
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

                if matching:
                    lines_ig = [
                        "## Subgroup Discovery Insights",
                        f"(from {Path(insight_path).name} @ recall {threshold:.2f})",
                        "",
                        "The following insight cards were discovered via LSQM subgroup mining",
                        "on related NHQ tasks. Use them to guide your initial",
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
                                f"- Best in region: M={bp.get('M')}, efC={bp.get('efConstruction')}, "
                                f"ef={bp.get('ef')}, weight={bp.get('weight', 0)} "
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

                    lines_ig.extend([
                        "### Key Observations from Subgroup Mining:",
                        "1. These cards are derived from BO trials on the SAME dataset family",
                        "   — prioritize them over general HNSMW knowledge.",
                        "2. The top-ranked cards show the empirically PROVEN parameter regions.",
                        "3. Cards marked 'top-QPS' or 'viable' are safe starting points.",
                        "4. Cards marked 'unstable' should be avoided.",
                        "",
                        "⚠️  **CRITICAL: Start your proposal from the BEST PARAMS of the",
                        "top-ranked card.** Adapt only if you have strong reason.",
                        "The card's region description and best params are the most",
                        "reliable guide available. Do NOT override with generic heuristics",
                        "like 'higher efC = better graph'.",
                        "",
                    ])
                    insight_guidance = "\n".join(lines_ig)

    if insight_guidance:
        if cold_start_guidance:
            cold_start_guidance = insight_guidance + "\n" + cold_start_guidance
        else:
            cold_start_guidance = insight_guidance

    # ── 3. Seed candidates ──────────────────────────────────────────────
    seed_candidates: List[Dict[str, Any]] = []

    # Priority 1: insight card seeds
    seed_candidates.extend(insight_seed_candidates)

    # Priority 2: transfer learning from similar tasks
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
    """Run the complete NHQ tuning pipeline (Stage A → Stage B)."""
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_name = str((cfg.get("output") or {}).get("trials_name", "")).strip()
    if not trials_name:
        raise ValueError("output.trials_name must be provided and non-empty.")

    trials_path = output_dir / "trials" / f"{trials_name}.jsonl"
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
    agent = NHQTuningAgent(
        cfg["params"],
        seed=int(cfg["search"]["seed"]),
        agentic_cfg=agentic_cfg,
        model_cfg=model_cfg,
        llm_caller=None,
    )

    # ── Current Task Memory (point memory + dominance repository) ─────
    current_memory: CurrentTaskMemory | None = None
    try:
        current_memory = CurrentTaskMemory(
            task_context={
                "recall_threshold": recall_threshold,
                "param_order": list(agent.param_order),
                "knob_bounds": {
                    name: agent.space.domains[name].to_spec() for name in agent.param_order
                },
            },
            parameter_space=agent.space,
            output_dir=output_dir,
            trials_name=trials_name,
            algorithm_name="NHQ",
            knobs=_NHQ_KNOBS,
        )
    except Exception:
        logger.warning("Failed to initialize current-task memory.", exc_info=True)

    # ── Subgroup-mining initialization (optional, hnsw-style) ─────────
    subgroup_init: Dict[str, Any] | None = None
    subgroup_init_cfg = (
        agentic_cfg.get("subgroup_init") if isinstance(agentic_cfg.get("subgroup_init"), dict) else {}
    )
    if subgroup_init_cfg:
        json_path_raw = subgroup_init_cfg.get("json_path")
        if not isinstance(json_path_raw, str) or not json_path_raw.strip():
            raise ValueError("agentic.subgroup_init.json_path must be a string path to an insight JSON.")
        payload = _load_subgroup_init_payload(json_path_raw.strip(), domain="nhq_tuning")
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

    # ── Static knowledge (single closest card per round) ───────────────
    static_selector = None
    try:
        static_kb = StaticKnowledgeBase.for_algorithm("NHQ")
        _knowledge_llm = lambda prompt: agent._invoke_llm(prompt)
        static_selector = StaticKnowledgeSelector(static_kb, llm_caller=_knowledge_llm)
        logger.info(
            "Static knowledge loaded: %d NHQ cards (%d infeasible, %d feasible)",
            len(static_kb.get_all_cards("NHQ")),
            len(static_kb.get_cards_by_branch("NHQ", "recall-infeasible")),
            len(static_kb.get_cards_by_branch("NHQ", "recall-feasible")),
        )
    except Exception:
        logger.warning("Failed to load NHQ static knowledge cards.", exc_info=True)

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
    stage_a_plan = _stage_a_plan_nhq(
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
    seed_idx = 0

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

        stage_trials = [t for t in all_trials if t.get("stage") == "unified"]
        occupied = set(
            _attempted_task_keys(all_trials, agent.param_order)
        )

        # Build root state.
        root_state = agent.build_root_state(
            round_idx=round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
        )

        # Cold start → use seed candidates FIRST (insight cards / transfer learning).
        # Only fall back to LLM after seeds are exhausted.
        static_knowledge_str = ""
        memory_summary = ""
        last_trial = stage_trials[-1] if stage_trials else None
        if last_trial is None and seed_idx < len(seed_candidates):
            candidate = seed_candidates[seed_idx]
            seed_idx += 1
            diag_log: Dict[str, Any] = {
                "ok": True, "diagnosis": {"strategy": "cold_start_seed"},
                "candidate": candidate, "classification": "cold_start",
                "tuning_action": {}, "error": "", "source": "stage_a_seed",
                "prompt_payload": {},
            }
        else:
            # ── Select the single closest static knowledge card ────────
            static_knowledge_str = ""
            if static_selector is not None:
                try:
                    last_metrics = (last_trial or {}).get("metrics") or {}
                    obs = {
                        "config": dict((last_trial or {}).get("params") or {}),
                        "qps": float(last_metrics.get("qps", 0) or 0),
                        "recall": float(last_metrics.get("recall", 0) or 0),
                        "recall_threshold": recall_threshold,
                        "diagnostic_metrics": {
                            "visited_nodes_per_query": last_metrics.get("visited_nodes_per_query"),
                            "distance_computations": last_metrics.get("dist_comps_per_query"),
                            "build_time_s": last_metrics.get("build_time_s"),
                            "index_size_mb": last_metrics.get("index_size_mb"),
                            "out_degree_mean": last_metrics.get("out_degree_mean"),
                            "in_degree_mean": last_metrics.get("in_degree_mean"),
                        },
                    }
                    selected = static_selector.select_best_match(
                        {"algorithm": "NHQ", "recall_threshold": recall_threshold},
                        obs,
                    )
                    static_knowledge_str = StaticKnowledgeSelector.format_knowledge_context_for_llm(selected)
                except Exception:
                    static_knowledge_str = ""
            # ── Current task memory summary ───────────────────────────
            memory_summary = ""
            if current_memory is not None:
                try:
                    last_metrics = (last_trial or {}).get("metrics") or {}
                    mem_obs = {
                        "config": dict((last_trial or {}).get("params") or {}),
                        "qps": float(last_metrics.get("qps", 0) or 0),
                        "recall": float(last_metrics.get("recall", 0) or 0),
                        "recall_threshold": recall_threshold,
                        "diagnostic_metrics": {
                            "visited_nodes_per_query": last_metrics.get("visited_nodes_per_query"),
                            "distance_computations": last_metrics.get("dist_comps_per_query"),
                            "build_time_s": last_metrics.get("build_time_s"),
                            "index_size_mb": last_metrics.get("index_size_mb"),
                        },
                    }
                    mem_full = current_memory.build_full_state(mem_obs)
                    mem_ctx = current_memory.retrieve_memory_context(
                        current_full_state=mem_full,
                        current_config=dict((last_trial or {}).get("params") or {}),
                    )
                    memory_summary = CurrentTaskMemory.format_memory_context(mem_ctx, knobs=_NHQ_KNOBS)
                except Exception:
                    memory_summary = ""
            # LLM diagnosis (used after seeds exhausted)
            candidate, diag_log = agent.diagnose_last_execution(
                last_trial=last_trial,
                root_state=root_state,
                stage_policy=stage_policy,
                cold_start_guidance=cold_start_guidance if not stage_trials else "",
                exhausted_keys=occupied,
                static_knowledge_context=static_knowledge_str,
                memory_summary=memory_summary,
            )

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
                    exhausted_keys=occupied,
                    rejection_feedback=reject_feedback,
                    static_knowledge_context=static_knowledge_str,
                    memory_summary=memory_summary,
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

        run_params = dict(candidate["params"])

        tasks = _build_tasks(
            candidates=[{"params": run_params, "source": candidate.get("source", "unknown"),
                        "note": candidate.get("note", "")}],
            needed_runs=min(1, remaining),
            occupied_task_keys=occupied,
            order=agent.param_order,
            round_idx=round_idx,
        )

        # If candidate was rejected (duplicate), retry with LLM feedback.
        dup_retry = 0
        while not tasks and dup_retry < 3:
            dup_retry += 1
            rejected_params = candidate.get("params", {})
            logger.warning(
                "Candidate rejected (duplicate round %d retry %d): %s",
                round_idx, dup_retry, rejected_params,
            )
            rejection_msg = (
                f"Your previous proposal (M={rejected_params.get('M')}, "
                f"efConstruction={rejected_params.get('efConstruction')}, "
                f"ef={rejected_params.get('ef')}, "
                f"weight={rejected_params.get('weight')}) was REJECTED because it has "
                f"already been tried. You MUST pick DIFFERENT values."
            )
            retry_temp = 0.5 + dup_retry * 0.25
            candidate, diag_log = agent.diagnose_last_execution(
                last_trial=last_trial,
                root_state=root_state,
                stage_policy=stage_policy,
                cold_start_guidance=cold_start_guidance if not stage_trials else "",
                exhausted_keys=occupied,
                rejection_feedback=rejection_msg,
                temperature_override=retry_temp,
                static_knowledge_context=static_knowledge_str,
                memory_summary=memory_summary,
            )
            if candidate is None:
                raise RuntimeError(
                    "LLM diagnosis/proposal failed after duplicate-rejection retries "
                    f"(error={diag_log.get('error', 'unknown')}) — rule-based fallback is disabled."
                )
            run_params = dict(candidate["params"])
            tasks = _build_tasks(
                candidates=[{"params": run_params, "source": candidate.get("source", "unknown"),
                            "note": candidate.get("note", "")}],
                needed_runs=min(1, remaining),
                occupied_task_keys=occupied,
                order=agent.param_order,
                round_idx=round_idx,
            )

        if not tasks:
            raise RuntimeError(
                "LLM diagnosis/proposal exhausted without producing new candidates "
                f"(error={diag_log.get('error', 'unknown')}) — rule-based fallback is disabled."
            )

        # Log proposal.
        proposal_log = {
            "round_idx": round_idx,
            "root_state": root_state,
            "diagnosis": diag_log.get("diagnosis", {}),
            "classification": diag_log.get("classification", ""),
            "tuning_action": diag_log.get("tuning_action", {}),
            "candidate": candidate,
            "stage_policy": stage_policy,
            "generated_at": utc_now_iso(),
        }
        pass  # proposal logs no longer stored

        if dry_run:
            logger.info("Dry-run round %d: candidate=%s", round_idx, candidate["params"])
            continue

        # Execute.
        new_trials = _execute_tasks(tasks, runner_cfg, trials_path, max_workers)
        all_trials.extend(new_trials)

        if current_memory is not None and new_trials and not dry_run:
            try:
                for t in new_trials:
                    if t.get("status") != "success":
                        continue
                    m = t.get("metrics") or {}
                    fs = current_memory.build_full_state({
                        "config": t.get("params") or {},
                        "qps": float(m.get("qps", 0) or 0),
                        "recall": float(m.get("recall", 0) or 0),
                        "recall_threshold": recall_threshold,
                        "diagnostic_metrics": {
                            "visited_nodes_per_query": m.get("visited_nodes_per_query"),
                            "distance_computations": m.get("dist_comps_per_query"),
                            "build_time_s": m.get("build_time_s"),
                            "index_size_mb": m.get("index_size_mb"),
                        },
                    })
                    current_memory.add_initial_observation(fs)
            except Exception:
                logger.warning("Memory update failed.", exc_info=True)

        if new_trials:
            t = new_trials[-1]
            m = t.get("metrics") or {}
            logger.info(
                "Round %d: M=%s efC=%s ef=%s weight=%s → recall=%s QPS=%s status=%s",
                round_idx,
                t["params"].get("M"), t["params"].get("efConstruction"),
                t["params"].get("ef"), t["params"].get("weight"),
                m.get("recall"), m.get("qps"), t.get("status"),
            )

    # ── Final report ───────────────────────────────────────────────────────
    stage_trials_final = [t for t in all_trials if t.get("stage") == "unified"]
    success = [t for t in stage_trials_final if t.get("status") == "success"]
    aggregates = aggregate_success_trials(stage_trials_final, order=agent.param_order)
    pareto = compute_pareto_front(aggregates)
    summary = compute_threshold_summary(pareto, recall_threshold, recall_slack)

    report_path = output_dir / "summary.json"
    write_json(report_path, {
        "trials_name": trials_name,
        "total_runs": len(stage_trials_final),
        "success_runs": len(success),
        "unique_params": len(aggregates),
        "pareto_count": len(pareto),
        "init_prompt_tokens": agent.init_prompt_tokens,
        "init_completion_tokens": agent.init_completion_tokens,
        "total_prompt_tokens": agent.total_prompt_tokens,
        "total_completion_tokens": agent.total_completion_tokens,
        **summary,
    })
    logger.info("Pipeline complete. Report: %s", report_path)
    logger.info("LLM tokens — init: %d prompt + %d completion | total: %d prompt + %d completion",
                agent.init_prompt_tokens, agent.init_completion_tokens,
                agent.total_prompt_tokens, agent.total_completion_tokens)
    return 0


def _resolve_model_cfg(agentic_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve LLM model configuration from CommonConfig."""
    config_key = str(agentic_cfg.get("model_config_key", "")).strip()
    if config_key:
        cfg = getattr(CommonConfig, config_key, None)
        if isinstance(cfg, dict):
            return cfg
    model_cfg: Dict[str, Any] = {}
    for attr in ("model_name", "model", "url", "authorization", "temperature", "max_tokens", "timeout_s"):
        val = getattr(CommonConfig, attr, None)
        if val is not None:
            model_cfg[attr] = val
    return model_cfg


# ── CLI entry point ────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NHQ tuning pipeline.")
    parser.add_argument("--config", type=str, default="configs/nhq_tune.yaml",
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

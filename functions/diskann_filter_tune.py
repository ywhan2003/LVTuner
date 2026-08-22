"""Filtered-DiskANN in-memory tuning pipeline with Stage A planning.

Extends the shared single-stage pipeline with DiskANN-specific:
- Stage A cold-start planning (selectivity analysis, knowledge building, seeding)
- DiskANNTuningAgent with parameter constraints and domain knowledge
- DiskANN-specific knowledge base integration
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from agents.diskann_agent import DISKANN_PARAM_ORDER, DiskANNTuningAgent
from functions.shared_single_stage_tune import run_pipeline as _run_shared_pipeline


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "search" not in cfg or "recall_threshold" not in (cfg.get("search") or {}):
        raise ValueError(f"{config_path} must explicitly set search.recall_threshold")
    return cfg


def _extract_selectivity(cfg: Dict[str, Any]) -> float | None:
    """Extract filtering selectivity from config extra_args."""
    extra = cfg.get("benchmark", {}).get("extra_args", [])
    # Look for filter_label to determine selectivity
    for i, arg in enumerate(extra):
        if arg == "--filter_label" and i + 1 < len(extra):
            label = str(extra[i + 1])
            # Parse selectivity from label like "sel_0p50" -> 0.50
            if "sel_" in label or "selectivity" in label:
                import re
                match = re.search(r"(\d+p?\d+)", label)
                if match:
                    val = match.group(1).replace("p", ".")
                    try:
                        return float(val)
                    except ValueError:
                        pass
    return None


def _build_diskann_knowledge(cfg: Dict[str, Any]) -> str:
    """Build DiskANN-focused knowledge context from knowledge base."""
    agentic_cfg = cfg.get("agentic") or {}
    knowledge_cfg = agentic_cfg.get("knowledge") or {}
    max_context_chars = int(knowledge_cfg.get("max_context_chars", 120000))
    knowledge_base_dir = str(knowledge_cfg.get("knowledge_base_dir", "knowledge_base"))

    from utils.knowledge_loader import build_knowledge_context as _kb_load_context

    kb_result = _kb_load_context(base_dir=knowledge_base_dir, max_chars=max_context_chars)
    return kb_result["base_knowledge_full"]


def _stage_a_plan_diskann(
    cfg: Dict[str, Any],
    agent: DiskANNTuningAgent,
    output_dir: Path,
    trials_name: str,
) -> Dict[str, Any]:
    """Stage A: cold-start planning for DiskANN tuning.

    Returns a dict with keys:
    - knowledge_context: pre-built knowledge string
    - selectivity: parsed selectivity or None
    - initial_seeds: list of seed candidates from domain knowledge
    - stage_a_plan: full plan metadata dict
    """
    # Build knowledge context
    knowledge_context = _build_diskann_knowledge(cfg)

    # Extract selectivity
    selectivity = _extract_selectivity(cfg)

    # Generate initial seeds from domain knowledge
    initial_seeds = agent.build_selectivity_seeds(selectivity=selectivity)

    plan = {
        "generated_at": _utc_now_iso(),
        "task_name": trials_name,
        "selectivity": selectivity,
        "seed_count": len(initial_seeds),
        "seeds": [
            {"params": s, "source": "diskann_domain_knowledge"} for s in initial_seeds
        ],
        "knowledge_chars": len(knowledge_context) if isinstance(knowledge_context, str) else 0,
        "param_order": list(agent.param_order),
        "build_params": agent.build_order,
        "search_params": agent.search_params,
    }

    # Persist plan
    plan_path = output_dir / "stage_a_plan.json"
    import json
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "knowledge_context": knowledge_context,
        "selectivity": selectivity,
        "initial_seeds": initial_seeds,
        "stage_a_plan": plan,
    }


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(
    config_path: str,
    resume: bool = True,
    dry_run: bool = False,
) -> int:
    """Run Filtered-DiskANN tuning pipeline with Stage A planning."""
    cfg = _load_config(config_path)

    output_dir = Path(cfg["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_name = str((cfg.get("output") or {}).get("trials_name", "")).strip()
    if not trials_name:
        raise ValueError("output.trials_name must be provided and non-empty.")

    # Resolve param order from config
    tuning_cfg = cfg.get("tuning") or {}
    raw_order = tuning_cfg.get("param_order", DISKANN_PARAM_ORDER)
    if isinstance(raw_order, Sequence) and not isinstance(raw_order, (str, bytes)):
        param_order = [str(name).strip() for name in raw_order]
    else:
        param_order = list(DISKANN_PARAM_ORDER)

    # Build agentic config
    agentic_cfg = cfg.get("agentic") or {}
    model_cfg = _resolve_model_cfg(agentic_cfg)

    # Create DiskANN agent
    agent = DiskANNTuningAgent(
        cfg["params"],
        seed=int(cfg["search"]["seed"]),
        param_order=param_order,
        agentic_cfg=agentic_cfg,
        model_cfg=model_cfg,
        prompt_cfg=_resolve_prompt_cfg(),
        llm_caller=None,
        objective_preference=agentic_cfg.get("objective_preference", "pareto"),
    )

    # Stage A: cold-start planning
    stage_a = _stage_a_plan_diskann(
        cfg=cfg,
        agent=agent,
        output_dir=output_dir,
        trials_name=trials_name,
    )

    # Build seed candidates list with source annotation
    seed_candidates = [
        {"params": s, "source": "stage_a_diskann_seed", "note": "selectivity_aware"}
        for s in stage_a["initial_seeds"]
    ]

    print(f"[Stage A] Selectivity: {stage_a['selectivity']}")
    print(f"[Stage A] Generated {len(seed_candidates)} selectivity-aware seed candidates")
    print(f"[Stage A] Knowledge context: {stage_a['stage_a_plan']['knowledge_chars']} chars")
    print(f"[Stage A] Plan saved to {output_dir / 'stage_a_plan.json'}")

    # Stage B: delegate to shared pipeline with DiskANN overrides
    return _run_shared_pipeline(
        config_path=config_path,
        resume=resume,
        dry_run=dry_run,
        agent_override=agent,
        seed_candidates=seed_candidates,
        knowledge_context_override=stage_a["knowledge_context"],
    )


def _resolve_model_cfg(agentic_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from configs import CommonConfig

    model_config_key = agentic_cfg.get("model_config_key", "DEEPSEEK_CONFIG")
    source = CommonConfig[model_config_key] or {}
    fallback = CommonConfig["RFANNS_AGENT_CONFIG"] or {}

    merged = dict(fallback)
    merged.update(source)
    merged.update(agentic_cfg.get("model_override", {}))

    if "model_name" not in merged:
        model_name = merged.get("model") or fallback.get("model_name")
        if not model_name:
            raise ValueError(
                "No model configured — set LLM_MODEL_NAME (or per-pipeline override) in .env "
                "or provide agentic.model_override in the config."
            )
        merged["model_name"] = model_name
    if "max_tokens" not in merged:
        merged["max_tokens"] = 2048
    if "temperature" not in merged:
        merged["temperature"] = 0.2
    return merged


def _resolve_prompt_cfg() -> Dict[str, str]:
    from configs import CommonConfig

    return {
        "proposer_user": CommonConfig["RFANNS_PROPOSER_PROMPT"]["user_prompt"],
        "reflector_user": CommonConfig["RFANNS_REFLECTOR_PROMPT"]["user_prompt"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Filtered-DiskANN / AF-ANNS tuning with Stage A planning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diskann_filter_tune.yaml",
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
        help="Do not resume from existing trials.",
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

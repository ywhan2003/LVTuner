"""NHQ (Native Hybrid Query) tuning agent — LLM-driven diagnosis and candidate proposal.

Modeled on the Filter-DiskANN agent's core diagnostic loop: parameter space,
recall classification, root state aggregation, diagnostic-tree-guided
LLM diagnosis, step-size compliance, and deterministic fallback.

Tunable parameters: M, efConstruction, ef, weight (4 params).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger("nhq_agent.llm")

# ── Parameter order ────────────────────────────────────────────────────────
PARAM_ORDER = ["M", "efConstruction", "ef", "weight"]
BUILD_PARAM_ORDER = ["M", "efConstruction"]
SEARCH_PARAM_ORDER = ["ef", "weight"]

# ── Reasonable step-size bounds per parameter ──────────────────────────────
# (recall_far → aggressive, recall_near → conservative)
STEP_BOUNDS: Dict[str, Dict[str, int | float]] = {
    "M":              {"far": 48, "near": 24},
    "efConstruction": {"far": 300, "near": 100},
    "ef":             {"far": 500, "near": 200},
    "weight":         {"far": 50, "near": 20},
}

# ── Baseline params for cold start ─────────────────────────────────────────
BASELINE_PARAMS: Dict[str, Any] = {
    "M": 64, "efConstruction": 200, "ef": 400, "weight": 0,
}


def _load_diagnostic_tree() -> List[Dict[str, Any]]:
    """Load the NHQ diagnostic tree from the knowledge base."""
    tree_path = (
        Path(__file__).resolve().parent.parent
        / "knowledge_base"
        / "nhq_diagnostic_tree.json"
    )
    if tree_path.exists():
        try:
            return json.loads(tree_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


# Module-level constant loaded once.
DIAGNOSTIC_TREE: List[Dict[str, Any]] = _load_diagnostic_tree()


# ── Reuse rfanns_agent parameter infrastructure ────────────────────────────
from agents.rfanns_agent import (  # noqa: E402
    FLOAT_ROUND_DECIMALS,
    ParameterDomain,
    ParameterSpace,
    params_to_key,
)


class NHQTuningAgent:
    """LLM-driven tuning agent for NHQ (Native Hybrid Query) filtered ANN search.

    Parameters
    ----------
    params_cfg : dict
        YAML ``params`` block mapping each of ``[M, efConstruction, ef, weight]``
        to a range ``{min, max}`` spec.
    seed : int
        Random seed for deterministic fallback logic.
    agentic_cfg : dict | None
        Agentic configuration (LLM reasoning, fallback, etc.).
    model_cfg : dict | None
        LLM API configuration (model_name, url, authorization, …).
    llm_caller : callable | None
        Optional custom ``(role, prompt) -> str`` callback overriding
        the default OpenAI client.
    """

    def __init__(
        self,
        params_cfg: Dict[str, Any],
        seed: int = 42,
        agentic_cfg: Dict[str, Any] | None = None,
        model_cfg: Dict[str, Any] | None = None,
        llm_caller: Any | None = None,
    ) -> None:
        self.space = ParameterSpace.from_config(params_cfg, order=PARAM_ORDER)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.agentic_cfg = agentic_cfg or {}
        self.model_cfg = model_cfg or {}
        self.llm_caller = llm_caller
        if self.agentic_cfg.get("enabled", True) is False:
            raise ValueError("agentic.enabled must be true — the nhq pipeline requires LLM proposals.")
        self.enable_agentic = True
        # Alias mapping for parameter name variants.
        self.alias_to_canonical: Dict[str, str] = {
            "MaxM0": "M",
            "maxM0": "M",
            "efc": "efConstruction",
            "ef_construction": "efConstruction",
            "efConstruction": "efConstruction",
            "ef_search": "ef",
            "efSearch": "ef",
            "weight_search": "weight",
            "weightSearch": "weight",
            "w": "weight",
        }
        # Max LLM retries if candidate fails validation.
        retry_cfg = self.agentic_cfg.get("proposer_retry") or {}
        self.proposer_retry_max = max(1, int(retry_cfg.get("max_attempts", 2)))
        # Token usage counters (cumulative across all LLM calls).
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # First-call token counters (initialization phase).
        self.init_prompt_tokens = 0
        self.init_completion_tokens = 0
        self._first_call_done = False

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def param_order(self) -> List[str]:
        return list(PARAM_ORDER)

    # ── Canonicalize ───────────────────────────────────────────────────────

    def canonicalize(
        self,
        params: Dict[str, Any],
        domain_constraints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Normalize parameters: resolve aliases, enforce domain bounds and hard constraints."""
        normalized = dict(params or {})
        for alias, canonical in self.alias_to_canonical.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized[alias]

        # Pre-clamp values to domain ranges so space.canonicalize won't reject them.
        for name in PARAM_ORDER:
            if name not in normalized:
                continue
            domain = self.space.domains.get(name)
            if domain is None:
                continue
            val = float(normalized[name])
            if domain.min_value is not None and val < domain.min_value:
                normalized[name] = domain.min_value if domain.is_integer else float(domain.min_value)
            if domain.max_value is not None and val > domain.max_value:
                normalized[name] = domain.max_value if domain.is_integer else float(domain.max_value)
            if domain.is_integer:
                normalized[name] = int(normalized[name])

        result = self.space.canonicalize(normalized, constraints=domain_constraints)
        # Enforce hard constraints
        if "M" in result:
            result["M"] = max(int(result["M"]), 2)
        if "efConstruction" in result:
            result["efConstruction"] = max(int(result["efConstruction"]), 1)
        if "ef" in result:
            result["ef"] = max(int(result["ef"]), 1)
        if "weight" in result:
            result["weight"] = max(int(result["weight"]), 0)
        return result

    # ── JSON extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        stripped = text.strip()
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, list):
                return {"candidates": loaded}
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        code_match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", stripped, flags=re.DOTALL)
        if code_match:
            try:
                loaded = json.loads(code_match.group(1))
                if isinstance(loaded, list):
                    return {"candidates": loaded}
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
        for start in [m.start() for m in re.finditer(r"\{", stripped)]:
            depth = 0
            for idx in range(start, len(stripped)):
                if stripped[idx] == "{":
                    depth += 1
                elif stripped[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            loaded = json.loads(stripped[start : idx + 1])
                            if isinstance(loaded, dict):
                                return loaded
                        except Exception:
                            break
        return None

    # ── LLM call ───────────────────────────────────────────────────────────

    def _default_llm_call(self, prompt: str, temperature_override: float | None = None) -> str:
        from openai import OpenAI

        model_name = self.model_cfg.get("model_name") or self.model_cfg.get("model")
        if not model_name:
            raise RuntimeError(
                "NHQ agent model_name is not configured — "
                "set LLM_MODEL_NAME (or per-pipeline override) in .env."
            )
        base_url = self.model_cfg.get("url")
        api_key = self.model_cfg.get("authorization")
        temperature = temperature_override if temperature_override is not None else float(self.model_cfg.get("temperature", 0.2))
        max_tokens = int(self.model_cfg.get("max_tokens", 2048))
        timeout_s = float(self.model_cfg.get("timeout_s", 120))
        if not base_url or "<base_url>" in str(base_url):
            raise RuntimeError("NHQ agent model URL is not configured.")
        if not api_key or "<token>" in str(api_key):
            raise RuntimeError("NHQ agent model authorization is not configured.")
        logger.info("LLM call start model=%s prompt_chars=%d", model_name, len(prompt))
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("LLM call done response_chars=%d", len(response.choices[0].message.content or ""))
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0
            if not self._first_call_done:
                self.init_prompt_tokens = response.usage.prompt_tokens or 0
                self.init_completion_tokens = response.usage.completion_tokens or 0
                self._first_call_done = True
        return response.choices[0].message.content or ""

    def _invoke_llm(self, prompt: str, temperature_override: float | None = None) -> str:
        logger.info("LLM call prompt_chars=%d caller=%s", len(prompt),
                     "custom" if self.llm_caller else "default_openai")
        logger.debug("LLM prompt:\n%s", prompt)
        if self.llm_caller is not None:
            try:
                raw = self.llm_caller(role="diagnose", prompt=prompt)
            except TypeError:
                raw = self.llm_caller("diagnose", prompt)
        else:
            raw = self._default_llm_call(prompt, temperature_override=temperature_override)
        raw = raw or ""
        logger.debug("LLM raw response:\n%s", raw)
        return raw

    # ── Recall classification ──────────────────────────────────────────────

    @staticmethod
    def _classify_recall_status(
        recall: float,
        threshold: float,
    ) -> Tuple[str, str]:
        """Classify recall margin into one of three statuses."""
        margin = recall - threshold
        if threshold >= 0.99:
            tight = 0.001
        elif threshold >= 0.95:
            tight = 0.005
        else:
            tight = 0.01

        if margin < -tight:
            return "recall_far_below", (
                f"Recall ({recall:.4f}) is far below target ({threshold:.4f}). "
                "Need significant improvements: increase ef (cheap, no rebuild), "
                "or increase M/efConstruction for better graph quality. "
                "If weight>0 and recall collapsed, reduce weight toward 0 first."
            )
        elif margin <= tight:
            return "recall_near_threshold", (
                f"Recall ({recall:.4f}) is near target ({threshold:.4f}). "
                "Make small, careful adjustments. Small ef changes "
                "can fine-tune the recall-QPS tradeoff."
            )
        else:
            return "recall_far_above", (
                f"Recall ({recall:.4f}) exceeds target ({threshold:.4f}). "
                "Headroom available — can trade recall for QPS by decreasing ef, "
                "or increase weight for more filtering."
            )

    # ── Step-size check ────────────────────────────────────────────────────

    def _check_step_compliance(
        self,
        candidate: Dict[str, Any],
        last_params: Dict[str, Any] | None,
        classification: str,
    ) -> Tuple[bool, str]:
        """Check that the candidate's parameter changes respect step-size bounds."""
        if last_params is None:
            return True, ""
        step_key = "far" if "far" in classification else "near"
        violations: List[str] = []
        for name in PARAM_ORDER:
            if name not in candidate or name not in last_params:
                continue
            bound = STEP_BOUNDS.get(name, {}).get(step_key)
            if bound is None:
                continue
            delta = abs(float(candidate[name]) - float(last_params[name]))
            if delta > float(bound) + 1e-9:
                violations.append(
                    f"{name}: delta={delta:.3f} exceeds bound={bound}"
                )
        if violations:
            return False, f"Step-size violation ({step_key}): " + "; ".join(violations)
        return True, ""

    # ── Build root state ───────────────────────────────────────────────────

    def build_root_state(
        self,
        round_idx: int,
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        previous_attribution: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Aggregate all completed trials into a structured root state."""
        threshold = float(stage_policy["recall_threshold"])
        success_trials = [t for t in stage_trials if t.get("status") == "success"]

        # ── Build per-trial observations ──
        trial_observations: List[Dict[str, Any]] = []
        for trial in success_trials:
            params = trial.get("params") or {}
            metrics = trial.get("metrics") or {}
            if not isinstance(params, dict) or not isinstance(metrics, dict):
                continue
            if "recall" not in metrics or "qps" not in metrics:
                continue
            recall = float(metrics["recall"])
            obs = {
                "M": params.get("M"),
                "efConstruction": params.get("efConstruction"),
                "ef": metrics.get("best_search_list", metrics.get("selected_ef", params.get("ef"))),
                "weight": params.get("weight"),
                "recall": recall,
                "qps": float(metrics["qps"]),
                "feasible": recall >= threshold,
                "recall_margin": recall - threshold,
                "build_time_s": float(metrics.get("build_time_s", 0.0)),
            }
            if "latency_mean_us" in metrics:
                obs["latency_mean_us"] = float(metrics["latency_mean_us"])
            if "search_time_s" in metrics:
                obs["search_time_s"] = float(metrics["search_time_s"])
            if "dist_comps_per_query" in metrics:
                obs["dist_comps_per_query"] = float(metrics["dist_comps_per_query"])
            trial_observations.append(obs)

        # ── Best feasible ──
        feasible = [o for o in trial_observations if o["feasible"]]
        current_best_feasible = max(feasible, key=lambda o: (o["qps"], o["recall"])) if feasible else None

        # ── Closest to feasible ──
        infeasible = [o for o in trial_observations if not o["feasible"]]
        closest_to_feasible = max(infeasible, key=lambda o: o["recall"]) if infeasible else None

        # ── Classification ──
        last_obs = trial_observations[-1] if trial_observations else None
        if last_obs:
            classification, strategy = self._classify_recall_status(last_obs["recall"], threshold)
        else:
            classification, strategy = "cold_start", "No trials completed yet."

        optimization_stage = "cold_start"
        if current_best_feasible:
            optimization_stage = "refinement"

        # ── Recent failures ──
        all_completed = [t for t in stage_trials if t.get("status") in ("success", "failed")]
        recent_failures: List[Dict[str, Any]] = []
        for t in reversed(all_completed[-5:]):
            if t.get("status") == "failed":
                p = (t.get("params") or {})
                recent_failures.append({
                    "M": p.get("M"), "efConstruction": p.get("efConstruction"),
                    "ef": p.get("ef"), "weight": p.get("weight"),
                    "error": str(t.get("error", "unknown"))[:150],
                })

        return {
            "round_idx": round_idx,
            "optimization_stage": optimization_stage,
            "classification": classification,
            "strategy": strategy,
            "trial_observations": trial_observations,
            "current_best_feasible": current_best_feasible,
            "closest_to_feasible": closest_to_feasible,
            "total_success_trials": len(success_trials),
            "recent_failures": recent_failures,
            "total_trials": len(stage_trials),
            "unique_builds": len({(o["M"], o["efConstruction"]) for o in trial_observations}),
        }

    # ── Core: diagnose last execution ──────────────────────────────────────

    def diagnose_last_execution(
        self,
        *,
        last_trial: Dict[str, Any] | None,
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        memory_summary: str = "",
        allowed_values_override: Dict[str, Any] | None = None,
        historical_context: str = "",
        static_knowledge_context: str = "",
        cold_start_guidance: str = "",
        exhausted_keys: set | None = None,
        rejection_feedback: str = "",
        temperature_override: float | None = None,
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
        """Diagnose the last trial and propose the next candidate via LLM.

        Returns ``(candidate_dict, diagnostic_log)``.
        """
        empty_log: Dict[str, Any] = {
            "ok": False, "diagnosis": {}, "candidate": None, "error": "", "source": "none",
        }
        if not self.enable_agentic:
            empty_log["error"] = "agentic_disabled"
            return None, empty_log

        threshold = float(stage_policy["recall_threshold"])
        trial_obs = root_state.get("trial_observations") or []

        # ── 1. Build last-execution observation ────────────────────────────
        last_obs: Dict[str, Any] | None = None
        last_failed: Dict[str, Any] | None = None
        if last_trial is not None:
            params = last_trial.get("params") or {}
            if isinstance(params, dict):
                params_canonical = self.canonicalize(params)
            else:
                params_canonical = {}
            if last_trial.get("status") == "success":
                metrics = last_trial.get("metrics") or {}
                if isinstance(metrics, dict):
                    recall = float(metrics.get("recall", 0.0))
                    last_obs = {
                        "M": params_canonical.get("M"),
                        "efConstruction": params_canonical.get("efConstruction"),
                        "ef": metrics.get("best_search_list", metrics.get("selected_ef", params_canonical.get("ef"))),
                        "weight": params_canonical.get("weight"),
                        "recall": recall,
                        "qps": float(metrics.get("qps", 0.0)),
                        "feasible": recall >= threshold,
                        "recall_margin": recall - threshold,
                        "build_time_s": float(metrics.get("build_time_s", 0.0)),
                    }
                    if "latency_mean_us" in metrics and metrics["latency_mean_us"] is not None:
                        last_obs["latency_mean_us"] = float(metrics["latency_mean_us"])
                    if "dist_comps_per_query" in metrics:
                        last_obs["dist_comps_per_query"] = float(metrics["dist_comps_per_query"])
            else:
                err = last_trial.get("error") or "unknown error"
                last_failed = {
                    "M": params_canonical.get("M"),
                    "efConstruction": params_canonical.get("efConstruction"),
                    "ef": params_canonical.get("ef"),
                    "weight": params_canonical.get("weight"),
                    "error": str(err)[:300],
                }

        # ── 2. Cross-trial comparison table ────────────────────────────────
        cross_table_lines: List[str] = []
        if trial_obs:
            header = "| M | efC | ef | w | recall | QPS | margin | build_s |"
            sep =    "|----|-----|----|---|--------|------|--------|----------|"
            cross_table_lines = [header, sep]
            for obs in trial_obs:
                margin_str = f"{obs['recall_margin']:+.4f}"
                cross_table_lines.append(
                    f"| {obs.get('M','?')} | {obs.get('efConstruction','?')} | "
                    f"{obs.get('ef','?')} | {obs.get('weight','?')} | "
                    f"{obs['recall']:.4f} | {obs['qps']:.1f} | {margin_str} | "
                    f"{obs.get('build_time_s',0):.0f} |"
                )

        # ── 3. Diagnostic tree summary ─────────────────────────────────────
        diag_tree_summary: List[str] = []
        for node in DIAGNOSTIC_TREE:
            metric = node.get("metric", "?")
            display = node.get("display", metric)
            for state_key, state in (node.get("states") or {}).items():
                for branch in state.get("branches") or []:
                    rec_status = branch.get("recall_status", "?")
                    cause = branch.get("cause", "")
                    actions = ", ".join(
                        f"{a['parameter']} {a['direction']}"
                        for a in branch.get("actions", [])
                    )
                    if actions:
                        diag_tree_summary.append(
                            f"- **{display}** *{state_key}* + *{rec_status}*: "
                            f"{cause} → {actions}"
                        )

        # ── 4. Parameter space summary ─────────────────────────────────────
        param_space_lines: List[str] = []
        for name in PARAM_ORDER:
            domain = self.space.domains[name]
            spec = domain.to_spec()
            param_space_lines.append(f"  {name}: {json.dumps(spec)}")

        # ── 5. Build the prompt ────────────────────────────────────────────
        if last_obs is not None:
            classification, strategy = self._classify_recall_status(last_obs["recall"], threshold)
        elif trial_obs:
            most_recent = trial_obs[-1]
            classification, strategy = self._classify_recall_status(most_recent["recall"], threshold)
        else:
            classification, strategy = "cold_start", "First round — no observations yet."

        best_feasible = root_state.get("current_best_feasible")
        closest = root_state.get("closest_to_feasible")

        prompt_parts: List[str] = [
            "# NHQ (Native Hybrid Query) Tuning Agent — Diagnose & Propose",
            "",
            "You are tuning an NHQ (Native Hybrid Query) index for filtered ANN search.",
            "NHQ fuses vector similarity with structured attribute constraints in a proximity graph.",
            "",
            f"**🎯  YOUR GOAL**: MAXIMIZE QPS while keeping recall ≥ target ({threshold:.4f}).",
            "You are NOT trying to maximize recall — recall above the target is WASTED.",
            "Every unit of recall above target should be TRADED for higher QPS.",
            "",
            "## Key NHQ Concepts",
            "",
            "- **GT is pure vector distance** (L2/angular) — NO attribute penalty in ground truth.",
            "- **weight=0** → pure vector search → best match with GT (but no filtering).",
            "- **weight>0** → additive penalty `dist + weight × #mismatches` →",
            "  pushes same-attribute candidates to top but may miss true L2-neighbors",
            "  with different attributes → lower recall against pure-L2 GT.",
            "- **weight range**: 0–100 is the useful regime; large weights (1000+)",
            "  overwhelmingly prefer same-attribute candidates → recall collapse vs GT.",
            "- **ef is HALVED internally** (ef >> 1 loop bound) — effective beam = ef/2.",
            "  Always propose even ef values for predictable behavior.",
            "- **Build params (M, efConstruction)** determine graph quality → expensive to change (rebuild).",
            "- **Search params (ef, weight)** are cheap to probe → reuse the same index.",
            "",
            f"**Recall target**: {threshold:.4f}",
            f"**Optimization stage**: {root_state.get('optimization_stage', 'unknown')}",
            f"**Classification**: {classification}",
            f"**Strategy**: {strategy}",
            "",
        ]

        if cold_start_guidance:
            prompt_parts.extend([cold_start_guidance, ""])

        if historical_context:
            prompt_parts.extend(["## Historical Context", historical_context, ""])
        if static_knowledge_context:
            prompt_parts.extend(["## Selected Static Knowledge", static_knowledge_context, ""])

        # ── Last execution ─────────────────────────────────────────────────
        if last_obs is not None:
            prompt_parts.extend([
                "## Last Execution — SUCCESS",
                "",
                f"- **Params**: M={last_obs['M']}, efConstruction={last_obs['efConstruction']}, "
                f"ef={last_obs['ef']}, weight={last_obs['weight']}",
                f"- **Recall**: {last_obs['recall']:.4f} (target: {threshold:.4f}, margin: {last_obs['recall_margin']:+.4f})",
                f"- **QPS**: {last_obs['qps']:.1f}",
                f"- **Build time**: {last_obs.get('build_time_s', 0):.1f}s",
                f"- **Feasible**: {'✅ YES' if last_obs['feasible'] else '❌ NO — below threshold'}",
                "",
            ])
            if last_obs.get("latency_mean_us") is not None:
                prompt_parts.append(f"- **Latency**: {last_obs['latency_mean_us']:.1f} µs")
                prompt_parts.append("")
            if last_obs.get("dist_comps_per_query") is not None:
                prompt_parts.append(f"- **Dist comps/query**: {last_obs['dist_comps_per_query']:.1f}")
                prompt_parts.append("")

            if last_obs["feasible"]:
                margin = last_obs["recall_margin"]
                if margin > 0.02:
                    prompt_parts.extend([
                        f"**Your task**: Recall {last_obs['recall']:.4f} >> target {threshold:.4f} "
                        f"(margin=+{margin:.4f}). **You are WASTING recall. Trade it for QPS.**",
                        "",
                        "⛔  **NEVER increase M or efConstruction when recall already exceeds target.**",
                        "    Larger M/efC = SLOWER search = LOWER QPS. This is the WRONG direction.",
                        "",
                        "✅  **Valid moves (all increase QPS):**",
                        f"  1. DECREASE ef on SAME build (try ef={int(last_obs['ef']*0.5)} — no rebuild, instant QPS gain)",
                        f"  2. DECREASE M (try M={int(last_obs['M']*0.5)} — smaller graph, faster search)",
                        f"  3. DECREASE efConstruction (try efC={int(last_obs['efConstruction']*0.5)} — faster build)",
                        f"  4. INCREASE weight for more filtering (try weight={int(last_obs['weight'])+10})",
                        "     (may reduce recall — check if still feasible)",
                        "  5. Strategy B: larger efC + much smaller ef (better graph may enable tiny ef → net QPS gain)",
                        "",
                        f"**Make BOLD changes — recall can drop by {margin:.4f} and STILL meet target!**",
                        "",
                    ])
                else:
                    prompt_parts.extend([
                        f"**Your task**: Recall {last_obs['recall']:.4f} barely above target {threshold:.4f} "
                        f"(margin=+{margin:.4f}). **Optimize QPS carefully.**",
                        "",
                        "⛔  Do NOT increase M/efC — you risk unnecessary slowness.",
                        "",
                        "✅  Valid moves:",
                        "- Fine-tune ef down on SAME build (try 10-20 units lower — no rebuild)",
                        "- Try Strategy B: slightly larger efC + slightly smaller ef",
                        "- Adjust weight carefully (±5)",
                        "- Change only ONE param at a time",
                        "",
                    ])
            else:
                margin = last_obs["recall_margin"]
                prompt_parts.extend([
                    f"**Your task**: This trial is NOT feasible "
                    f"(recall {last_obs['recall']:.4f} vs target {threshold:.4f}, gap={-margin:.4f}).",
                    "",
                ])
                # Check if weight is the cause
                if last_obs.get("weight", 0) > 0 and margin < -0.05:
                    prompt_parts.extend([
                        "🔥  **weight>0 may be causing recall collapse vs pure-L2 GT!**",
                        f"   FIRST: try weight=0 on SAME build (no rebuild) to check baseline recall.",
                        "   If weight=0 fixes recall → the weight was too aggressive for the GT.",
                        "   Then gradually increase weight from 0 while monitoring recall.",
                        "",
                    ])
                elif margin < -0.02:
                    prompt_parts.extend([
                        "🔥  **Large recall gap — need to IMPROVE recall:**",
                        f"  1. FIRST: try increasing ef on SAME build (no rebuild, try ef={int(last_obs['ef']*1.5)})",
                        f"  2. If weight>0: reduce weight toward 0 (less filtering → better GT match)",
                        f"  3. If ef maxed out: increase efConstruction (try efC={int(last_obs['efConstruction']*1.5)})",
                        f"  4. Also increase M (try M={int(last_obs['M']*1.3)}) for better graph connectivity",
                        "",
                    ])
                else:
                    prompt_parts.extend([
                        f"**Small recall gap ({-margin:.4f}) — try ef or weight first:**",
                        f"- First: increase ef on SAME build (try +50-100, no rebuild needed)",
                        "- If weight>0: try reducing weight slightly",
                        "- Only build a new index if ef at max still insufficient",
                        "",
                    ])

        elif last_failed is not None:
            prompt_parts.extend([
                "## ⛔  Last Execution — FAILED",
                "",
                f"- **Failed params**: M={last_failed['M']}, efConstruction={last_failed['efConstruction']}, "
                f"ef={last_failed['ef']}, weight={last_failed['weight']}",
                f"- **Error**: {last_failed['error']}",
                "",
                "⚠️  **These parameters caused a BENCHMARK FAILURE** (timeout, OOM, or crash).",
                "**DO NOT propose these exact params again.**",
                "Reduce M and/or efConstruction significantly for the next attempt,",
                "or use a previously successful build with different ef/weight.",
                "",
            ])
        else:
            prompt_parts.extend([
                "## Last Execution — None",
                "",
                "No trials have been executed yet. This is the first round.",
                "Start with baseline params: M=64, efConstruction=200, ef=400, weight=0",
                "(weight=0 = pure vector search, best match with pure-L2 GT).",
                "",
            ])

        # Best so far
        if best_feasible:
            prompt_parts.extend([
                "## Best Feasible So Far",
                f"M={best_feasible['M']}, efConstruction={best_feasible['efConstruction']}, "
                f"ef={best_feasible['ef']}, weight={best_feasible.get('weight','?')} "
                f"→ recall={best_feasible['recall']:.4f} QPS={best_feasible['qps']:.1f}",
                "",
            ])
        if closest and not closest.get("feasible", False):
            prompt_parts.extend([
                "## Closest to Feasible",
                f"M={closest['M']}, efConstruction={closest['efConstruction']}, "
                f"ef={closest['ef']}, weight={closest.get('weight','?')} "
                f"→ recall={closest['recall']:.4f} (gap: {threshold - closest['recall']:.4f})",
                "",
            ])

        # Cross-trial table
        if cross_table_lines:
            prompt_parts.extend(["## All Trials", "", *cross_table_lines, ""])

        # Recent failures
        recent_failures = root_state.get("recent_failures") or []
        if recent_failures:
            prompt_parts.extend([
                "## ⚠️  Recent FAILED Configurations — AVOID THESE",
                "These parameter combinations caused benchmark failures (timeout/OOM/crash).",
                "Reduce M/efC significantly from these values, or use a known-good build.",
                "",
            ])
            for rf in recent_failures:
                prompt_parts.append(
                    f"- M={rf.get('M')}, efConstruction={rf.get('efConstruction')}, "
                    f"ef={rf.get('ef')}, weight={rf.get('weight')} → {rf.get('error','unknown')}"
                )
            prompt_parts.append("")

        # Diagnostic tree
        if diag_tree_summary:
            prompt_parts.extend([
                "## Diagnostic Tree",
                "Use this tree to interpret metrics and decide parameter changes:",
                "",
                *diag_tree_summary,
                "",
            ])

        # Already-tried configurations
        if exhausted_keys:
            exhausted_lines: List[str] = []
            for key in sorted(exhausted_keys):
                exhausted_lines.append(f"- {key}")
            if exhausted_lines:
                prompt_parts.extend([
                    "## ⚠️  ALREADY-TRIED CONFIGURATIONS — DO NOT REPEAT",
                    "These exact parameter tuples have been evaluated before.",
                    "You MUST propose a DIFFERENT configuration. Repeating any of these",
                    "is a waste of trials and will be rejected.",
                    "",
                    *exhausted_lines,
                    "",
                ])

        # Parameter space with boundary warnings
        param_boundary_warnings: List[str] = []
        for name in PARAM_ORDER:
            domain = self.space.domains[name]
            if last_obs is not None and name in last_obs:
                val = last_obs[name]
                if domain.max_value is not None and float(val) >= float(domain.max_value):
                    param_boundary_warnings.append(
                        f"⚠️  **{name}={val}** is at the MAXIMUM ({domain.max_value}). You CANNOT increase it further."
                    )
                if domain.min_value is not None and float(val) <= float(domain.min_value):
                    param_boundary_warnings.append(
                        f"⚠️  **{name}={val}** is at the MINIMUM ({domain.min_value}). You CANNOT decrease it further."
                    )

        prompt_parts.extend([
            "## Parameter Space",
            "```json",
            *param_space_lines,
            "```",
            "",
        ])

        if param_boundary_warnings:
            prompt_parts.extend([
                "## ⚠️  PARAMETER BOUNDARY WARNINGS",
                "These parameters are at the edge of their allowed range:",
                "",
                *param_boundary_warnings,
                "",
                "**You MUST change OTHER parameters** — tweaking a boundary parameter",
                "in the blocked direction is impossible.",
                "",
            ])

        # Rejection feedback
        if rejection_feedback:
            prompt_parts.extend([
                "## ⛔  YOUR PREVIOUS PROPOSAL WAS REJECTED",
                rejection_feedback,
                "",
                "**You MUST propose a DIFFERENT configuration this time.**",
                "Look at the ALREADY-TRIED list above and pick parameters that are NOT on it.",
                "",
            ])

        # Step-size guidance
        prompt_parts.extend([
            "## Step-Size Guidance",
            f"Classification: **{classification}** — use {'larger' if 'far' in classification else 'conservative'} steps.",
            "",
            "## Task",
            "Diagnose the last trial's metrics using the diagnostic tree above.",
            "Propose exactly ONE **NEW** candidate configuration for the next trial.",
            "",
            "⚠️  CRITICAL: Your candidate MUST be different from all configurations",
            "listed under \"ALREADY-TRIED CONFIGURATIONS\" above. Proposing a duplicate",
            "is the #1 failure mode — double-check before returning.",
            "",
            "Your proposal must be a (M, efConstruction, ef, weight) tuple. Each trial builds an",
            "NHQ index with (M, efConstruction) then runs search with (ef, weight).",
            "",
            "Remember:",
            "- M >= 2 (graph degree)",
            "- efConstruction >= 1 (build beam width)",
            "- ef >= 1 (search beam, HALVED internally — effective beam = ef/2)",
            "- weight >= 0 (attribute penalty: 0 = pure vector, >0 = filtering)",
            "- ef is HALVED internally — always propose even values",
            "- (M, efConstruction) are BUILD params → expensive to change (rebuild required)",
            "- (ef, weight) are SEARCH params → cheap to probe (same index reused)",
            "- GT is pure vector distance — weight=0 gives best GT match",
            "- weight>0 trades recall for filtering (may reduce recall vs pure-L2 GT)",
            "",
            "Return a strict JSON object:",
            "```json",
            "{",
            '  "classification": "' + classification + '",',
            '  "diagnosis": {"metric_name": "finding", ...},',
            '  "tuning_action": {',
            '    "M": "increase|decrease|keep",',
            '    "efConstruction": "increase|decrease|keep",',
            '    "ef": "increase|decrease|keep",',
            '    "weight": "increase|decrease|keep"',
            "  },",
            '  "candidate": {"M": <int>, "efConstruction": <int>, "ef": <int>, "weight": <int>},',
            '  "rationale": "<one sentence explaining the choice>"',
            "}",
            "```",
        ])

        prompt = "\n".join(prompt_parts)

        # ── 6. Call LLM ────────────────────────────────────────────────────
        try:
            raw = self._invoke_llm(prompt, temperature_override=temperature_override)
            parsed = self._extract_json_payload(raw)
        except Exception as exc:
            return None, {**empty_log, "error": f"llm_error: {exc}"}

        if parsed is None:
            return None, {**empty_log, "error": "json_parse_failed", "raw": raw}

        candidate_raw = parsed.get("candidate")
        if not isinstance(candidate_raw, dict):
            return None, {**empty_log, "error": "missing_candidate", "raw": raw}

        # ── 7. Validate candidate ──────────────────────────────────────────
        try:
            candidate_params = self.canonicalize(candidate_raw, allowed_values_override)
        except Exception as exc:
            return None, {**empty_log, "error": f"canonicalize_failed: {exc}", "raw": raw}

        last_params = None
        if last_trial is not None and last_trial.get("status") == "success":
            last_params = last_trial.get("params") or {}
            if isinstance(last_params, dict):
                last_params = self.canonicalize(last_params, allowed_values_override)

        compliant, step_msg = self._check_step_compliance(candidate_params, last_params, classification)
        if not compliant:
            retry_prompt = (
                prompt + f"\n\n⚠️ Your previous candidate was rejected: {step_msg}\n"
                "Please propose a new candidate within the step-size bounds."
            )
            try:
                raw2 = self._invoke_llm(retry_prompt, temperature_override=temperature_override)
                parsed2 = self._extract_json_payload(raw2)
                if parsed2 and isinstance(parsed2.get("candidate"), dict):
                    candidate_params = self.canonicalize(parsed2["candidate"], allowed_values_override)
                    compliant2, msg2 = self._check_step_compliance(candidate_params, last_params, classification)
                    if not compliant2:
                        candidate_params = self._clamp_to_bounds(candidate_params, last_params, classification)
            except Exception:
                candidate_params = self._clamp_to_bounds(candidate_params, last_params, classification)

        candidate_key = params_to_key(candidate_params, self.param_order)
        candidate = {
            "params": candidate_params,
            "key": candidate_key,
            "source": "diagnostic",
            "note": str(parsed.get("rationale", "")).strip(),
        }

        diag_log = {
            "ok": True,
            "diagnosis": parsed.get("diagnosis", {}),
            "candidate": candidate,
            "classification": classification,
            "tuning_action": parsed.get("tuning_action", {}),
            "error": "",
            "source": "diagnostic",
            "prompt_payload": {},
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
        return candidate, diag_log

    @staticmethod
    def _clamp_to_bounds(
        candidate: Dict[str, Any],
        last_params: Dict[str, Any],
        classification: str,
    ) -> Dict[str, Any]:
        """Clamp candidate parameters to step-size bounds from last_params."""
        step_key = "far" if "far" in classification else "near"
        clamped = dict(candidate)
        for name in PARAM_ORDER:
            if name not in clamped or name not in last_params:
                continue
            bound = STEP_BOUNDS.get(name, {}).get(step_key)
            if bound is None:
                continue
            current = float(clamped[name])
            last = float(last_params[name])
            if abs(current - last) > float(bound):
                if current > last:
                    clamped[name] = type(clamped[name])(last + float(bound))
                else:
                    clamped[name] = type(clamped[name])(last - float(bound))
        return clamped

    # ── Fallback: initial design candidates ────────────────────────────────

    def initial_design_candidates(
        self,
        target_count: int,
        exhausted_keys: set[Tuple[Any, ...]],
        stage_policy: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate deterministic fallback candidates when LLM fails.

        Keys are formed on BUILD_PARAM_ORDER so that ef/weight variations
        within the same build are not treated as separate builds.
        """
        candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()

        # Start with baseline.
        try:
            base = self.canonicalize(dict(BASELINE_PARAMS), allowed_values_override)
            key = params_to_key(base, BUILD_PARAM_ORDER)
            if key not in exhausted_keys and key not in seen:
                seen.add(key)
                candidates.append({
                    "params": base, "key": key, "source": "fallback_baseline",
                    "note": "Baseline fallback (M=64, efC=200, ef=400, weight=0)",
                })
        except Exception:
            pass

        # Generate neighbors by varying one BUILD parameter at a time.
        # Grid: M ∈ {16,32,64,96,128} × efC ∈ {50,100,200,400,800} at ef=400, weight=0
        deltas = [
            {"M": 16}, {"M": 32}, {"M": 96}, {"M": 128},
            {"efConstruction": 50}, {"efConstruction": 100},
            {"efConstruction": 400}, {"efConstruction": 800},
        ]
        for delta in deltas:
            if len(candidates) >= target_count:
                break
            try:
                params = self.canonicalize({**BASELINE_PARAMS, **delta}, allowed_values_override)
                key = params_to_key(params, BUILD_PARAM_ORDER)
                if key not in exhausted_keys and key not in seen:
                    seen.add(key)
                    candidates.append({
                        "params": params, "key": key, "source": "fallback_neighbor",
                        "note": f"Neighbor: {delta}",
                    })
            except Exception:
                continue

        return candidates[:max(1, target_count)]


__all__ = [
    "BASELINE_PARAMS",
    "BUILD_PARAM_ORDER",
    "DIAGNOSTIC_TREE",
    "NHQTuningAgent",
    "PARAM_ORDER",
    "SEARCH_PARAM_ORDER",
    "STEP_BOUNDS",
    "params_to_key",
]

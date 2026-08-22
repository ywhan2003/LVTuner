"""UNIFY (HSIG) tuning agent — LLM-driven diagnosis and candidate proposal.

Modeled on the HNSW agent's core diagnostic loop but stripped of the deprecated
mechanism library, multi-skill tree search, and empirical signal computation.
Keeps: parameter space, recall classification, root state aggregation,
diagnostic-tree-guided LLM diagnosis, and a simple deterministic fallback.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger("unify_agent.llm")

# ── Parameter order ────────────────────────────────────────────────────────
PARAM_ORDER = ["M", "B", "efConstruction", "ef", "al"]
BUILD_PARAM_ORDER = ["M", "B", "efConstruction"]
SEARCH_PARAM_ORDER = ["ef", "al"]

# ── Reasonable step-size bounds per parameter ──────────────────────────────
# (recall_far → aggressive, recall_near → conservative)
STEP_BOUNDS: Dict[str, Dict[str, int]] = {
    "M":               {"far": 20, "near": 4},
    "B":               {"far": 2,  "near": 1},
    "efConstruction":  {"far": 250, "near": 50},
    "ef":              {"far": 150, "near": 10},
    "al":              {"far": 64,  "near": 16},
}

# ── Baseline params for cold start ─────────────────────────────────────────
BASELINE_PARAMS: Dict[str, Any] = {
    "M": 16, "B": 6, "efConstruction": 200, "ef": 40, "al": 32,
}


def _load_diagnostic_tree() -> List[Dict[str, Any]]:
    """Load the UNIFY diagnostic tree from the knowledge base."""
    tree_path = Path(__file__).resolve().parent.parent / "knowledge_base" / "unify_diagnostic_tree.json"
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


class UNIFTuningAgent:
    """LLM-driven tuning agent for UNIFY (HSIG) range-filtered ANN search.

    Parameters
    ----------
    params_cfg : dict
        YAML ``params`` block mapping each of ``[M, B, efConstruction, ef, al]``
        to a range ``{min, max}`` or discrete ``{values: [...]}`` spec.
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
            raise ValueError("agentic.enabled must be true — the unify pipeline requires LLM proposals.")
        self.enable_agentic = True
        # Token usage counters (cumulative across all LLM calls).
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # First-call token counters (initialization phase).
        self.init_prompt_tokens = 0
        self.init_completion_tokens = 0
        self._first_call_done = False
        # Alias mapping for parameter name variants.
        self.alias_to_canonical: Dict[str, str] = {
            "efC": "efConstruction",
            "efConstruction": "efConstruction",
            "ef_construction": "efConstruction",
            "efConstruction": "efConstruction",
            "num_slots": "B",
            "b": "B",
        }
        # Max LLM retries if candidate fails validation.
        retry_cfg = self.agentic_cfg.get("proposer_retry") or {}
        self.proposer_retry_max = max(1, int(retry_cfg.get("max_attempts", 2)))

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def param_order(self) -> List[str]:
        return list(PARAM_ORDER)

    def token_snapshot(self) -> tuple:
        """Return (total_prompt_tokens, total_completion_tokens) for delta computation."""
        return (self.total_prompt_tokens, self.total_completion_tokens)

    # ── Canonicalize ───────────────────────────────────────────────────────

    def canonicalize(
        self,
        params: Dict[str, Any],
        domain_constraints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized = dict(params or {})
        for alias, canonical in self.alias_to_canonical.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized[alias]
        result = self.space.canonicalize(normalized, constraints=domain_constraints)
        # Enforce ef <= efConstruction
        if "ef" in result and "efConstruction" in result:
            result["ef"] = min(result["ef"], result["efConstruction"])
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

    def _default_llm_call(self, prompt: str) -> str:
        from openai import OpenAI

        model_name = self.model_cfg.get("model_name") or self.model_cfg.get("model")
        if not model_name:
            raise RuntimeError(
                "UNIFY agent model_name is not configured — "
                "set LLM_MODEL_NAME (or per-pipeline override) in .env."
            )
        base_url = self.model_cfg.get("url")
        api_key = self.model_cfg.get("authorization")
        temperature = float(self.model_cfg.get("temperature", 0.2))
        max_tokens = int(self.model_cfg.get("max_tokens", 2048))
        timeout_s = float(self.model_cfg.get("timeout_s", 120))
        if not base_url or "<base_url>" in str(base_url):
            raise RuntimeError("UNIFY agent model URL is not configured.")
        if not api_key or "<token>" in str(api_key):
            raise RuntimeError("UNIFY agent model authorization is not configured.")
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

    def _invoke_llm(self, prompt: str) -> str:
        logger.info("LLM call prompt_chars=%d caller=%s", len(prompt),
                     "custom" if self.llm_caller else "default_openai")
        logger.debug("LLM prompt:\n%s", prompt)
        if self.llm_caller is not None:
            try:
                raw = self.llm_caller(role="diagnose", prompt=prompt)
            except TypeError:
                raw = self.llm_caller("diagnose", prompt)
        else:
            raw = self._default_llm_call(prompt)
        raw = raw or ""
        logger.debug("LLM raw response:\n%s", raw)
        return raw

    # ── Memory context formatting ──────────────────────────────────────────

    @staticmethod
    def _format_memory_context_section(memory_context: str) -> str:
        """Render a memory context string into a prompt section.

        Delegates to ``CurrentTaskMemory.format_memory_context`` when the
        context is a dict; otherwise returns the string as-is.
        """
        if not memory_context:
            return ""
        if isinstance(memory_context, str):
            return memory_context
        try:
            from utils.current_task_memory import CurrentTaskMemory
            return CurrentTaskMemory.format_memory_context(
                memory_context,
                knobs=("M", "B", "efConstruction"),
            )
        except ImportError:
            return ""

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
                "Need significant construction improvements (higher M, efC) "
                "or increased search effort (higher ef, al)."
            )
        elif margin <= tight:
            return "recall_near_threshold", (
                f"Recall ({recall:.4f}) is near target ({threshold:.4f}). "
                "Make small, careful adjustments. Small ef/al changes "
                "can fine-tune the recall-QPS tradeoff."
            )
        else:
            return "recall_far_above", (
                f"Recall ({recall:.4f}) exceeds target ({threshold:.4f}). "
                "Headroom available — can trade recall for QPS by decreasing "
                "ef and/or al."
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
            if delta > bound:
                violations.append(
                    f"{name}: delta={delta:.0f} exceeds bound={bound}"
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
                "B": params.get("B"),
                "efConstruction": params.get("efConstruction"),
                "ef": metrics.get("selected_ef", params.get("ef")),
                "al": metrics.get("selected_al", params.get("al")),
                "recall": recall,
                "qps": float(metrics["qps"]),
                "feasible": recall >= threshold,
                "recall_margin": recall - threshold,
                "build_time_s": float(metrics.get("build_time_s", 0.0)),
            }
            if "inclusiveness_pct" in metrics:
                obs["inclusiveness_pct"] = float(metrics["inclusiveness_pct"])
            trial_observations.append(obs)

        # ── Best feasible ──
        feasible = [o for o in trial_observations if o["feasible"]]
        current_best_feasible = max(feasible, key=lambda o: (o["qps"], o["recall"])) if feasible else None

        # ── Closest to feasible ──
        infeasible = [o for o in trial_observations if not o["feasible"]]
        closest_to_feasible = max(infeasible, key=lambda o: o["recall"]) if infeasible else None

        # ── Classification ──
        last_feasible = [o for o in trial_observations if o["feasible"]]
        last_obs = trial_observations[-1] if trial_observations else None
        if last_obs:
            classification, strategy = self._classify_recall_status(last_obs["recall"], threshold)
        else:
            classification, strategy = "cold_start", "No trials completed yet."

        optimization_stage = "cold_start"
        if current_best_feasible:
            optimization_stage = "refinement"

        return {
            "round_idx": round_idx,
            "optimization_stage": optimization_stage,
            "classification": classification,
            "strategy": strategy,
            "trial_observations": trial_observations,
            "current_best_feasible": current_best_feasible,
            "closest_to_feasible": closest_to_feasible,
            "total_success_trials": len(success_trials),
            "total_trials": len(stage_trials),
            "unique_builds": len({(o["M"], o["B"], o["efConstruction"]) for o in trial_observations}),
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
        cold_start_guidance: str = "",
        memory_context: str = "",
        static_knowledge_context: str = "",
        rejection_feedback: str = "",
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
        if last_trial is not None and last_trial.get("status") == "success":
            metrics = last_trial.get("metrics") or {}
            params = last_trial.get("params") or {}
            if isinstance(metrics, dict) and isinstance(params, dict):
                recall = float(metrics.get("recall", 0.0))
                last_obs = {
                    "M": params.get("M"),
                    "B": params.get("B"),
                    "efConstruction": params.get("efConstruction"),
                    "ef": metrics.get("selected_ef", params.get("ef")),
                    "al": metrics.get("selected_al", params.get("al")),
                    "recall": recall,
                    "qps": float(metrics.get("qps", 0.0)),
                    "feasible": recall >= threshold,
                    "recall_margin": recall - threshold,
                    "build_time_s": float(metrics.get("build_time_s", 0.0)),
                }
                if "inclusiveness_pct" in metrics and metrics["inclusiveness_pct"] is not None:
                    last_obs["inclusiveness_pct"] = float(metrics["inclusiveness_pct"])

        # ── 2. Cross-trial comparison table ────────────────────────────────
        cross_table_lines: List[str] = []
        if trial_obs:
            header = "| M | B | efC | ef* | al* | recall | QPS | margin | build_s | incl% |"
            sep =    "|---|----|-----|-----|------|--------|------|--------|----------|-------|"
            cross_table_lines = [header, sep]
            for obs in trial_obs:
                margin_str = f"{obs['recall_margin']:+.4f}"
                incl_str = f"{obs.get('inclusiveness_pct', '?'):.1f}" if obs.get('inclusiveness_pct') is not None else "?"
                cross_table_lines.append(
                    f"| {obs.get('M','?')} | {obs.get('B','?')} | {obs.get('efConstruction','?')} | "
                    f"{obs.get('ef','?')} | {obs.get('al','?')} | "
                    f"{obs['recall']:.4f} | {obs['qps']:.1f} | {margin_str} | "
                    f"{obs.get('build_time_s',0):.0f} | {incl_str} |"
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

        # ── 5. Build diagnosis prompt ─────────────────────────────────────
        classification, strategy = self._classify_recall_status(
            last_obs["recall"] if last_obs else 0.0, threshold
        ) if last_obs else ("cold_start", "First round — no observations yet.")

        best_feasible = root_state.get("current_best_feasible")
        closest = root_state.get("closest_to_feasible")

        # Common context shared by both phases.
        context_parts: List[str] = [
            "# UNIFY Tuning Agent",
            "",
            "You are tuning a UNIFY (HSIG) index for range-filtered ANN search.",
            f"**Recall target**: {threshold:.4f}",
            f"**Optimization stage**: {root_state.get('optimization_stage', 'unknown')}",
            f"**Classification**: {classification}",
            f"**Strategy**: {strategy}",
            "",
        ]

        if cold_start_guidance:
            context_parts.extend([cold_start_guidance, ""])

        if historical_context:
            context_parts.extend(["## Historical Context", historical_context, ""])

        if static_knowledge_context:
            context_parts.extend([static_knowledge_context, ""])
        if rejection_feedback:
            context_parts.extend(["## Proposal Check Feedback", rejection_feedback, ""])

        if memory_context:
            context_parts.extend([memory_context, ""])

        # Last execution details
        if last_obs:
            context_parts.extend([
                "## Last Execution",
                "",
                f"- **Params**: M={last_obs['M']}, B={last_obs['B']}, efConstruction={last_obs['efConstruction']}, "
                f"ef={last_obs['ef']}, al={last_obs['al']}",
                f"- **Recall**: {last_obs['recall']:.4f} (margin: {last_obs['recall_margin']:+.4f})",
                f"- **QPS**: {last_obs['qps']:.1f}",
                f"- **Build time**: {last_obs.get('build_time_s', 0):.1f}s",
                f"- **Feasible**: {last_obs['feasible']}",
                "",
            ])
            if last_obs.get("inclusiveness_pct") is not None:
                context_parts.append(f"- **Inclusiveness**: {last_obs['inclusiveness_pct']:.1f}%")
                context_parts.append("")

        # Best so far
        if best_feasible:
            context_parts.extend([
                "## Best Feasible So Far",
                f"M={best_feasible['M']}, B={best_feasible['B']}, efC={best_feasible['efConstruction']}, "
                f"ef={best_feasible['ef']}, al={best_feasible['al']} "
                f"→ recall={best_feasible['recall']:.4f} QPS={best_feasible['qps']:.1f}",
                "",
            ])
        if closest and not closest.get("feasible", False):
            context_parts.extend([
                "## Closest to Feasible",
                f"M={closest['M']}, B={closest['B']}, efC={closest['efConstruction']}, "
                f"ef={closest['ef']}, al={closest['al']} "
                f"→ recall={closest['recall']:.4f} (gap: {threshold - closest['recall']:.4f})",
                "",
            ])

        # Cross-trial table
        if cross_table_lines:
            context_parts.extend(["## All Trials", "", *cross_table_lines, ""])

        # Diagnostic tree
        if diag_tree_summary:
            context_parts.extend([
                "## Diagnostic Tree",
                "Use this tree to interpret metrics and decide parameter changes:",
                "",
                *diag_tree_summary,
                "",
            ])

        # Parameter space
        context_parts.extend([
            "## Parameter Space",
            "```json",
            *param_space_lines,
            "```",
            "",
        ])

        context_text = "\n".join(context_parts)

        # ── 6. Phase 1: Diagnosis (LLM call with timing) ──────────────────
        diag_prompt = context_text + "\n".join([
            "",
            "## Task: Diagnosis",
            "Analyze the last trial's metrics using the diagnostic tree above.",
            "Identify the root cause of any issues (recall gap, low QPS, high build time, etc.).",
            "Determine the tuning direction for each parameter.",
            "",
            "Return a strict JSON object:",
            "```json",
            "{",
            f'  "classification": "{classification}",',
            '  "diagnosis": {"metric_name": "finding", ...},',
            '  "tuning_action": {',
            '    "M": "increase|decrease|keep",',
            '    "B": "increase|decrease|keep",',
            '    "efConstruction": "increase|decrease|keep",',
            '    "ef": "increase|decrease|keep",',
            '    "al": "increase|decrease|keep"',
            "  }",
            "}",
            "```",
        ])

        snap_before_diag = self.token_snapshot()
        t_diag_start = time.perf_counter()
        try:
            raw_diag = self._invoke_llm(diag_prompt)
            parsed_diag = self._extract_json_payload(raw_diag)
        except Exception as exc:
            return None, {**empty_log, "error": f"diagnosis_llm_error: {exc}"}
        diag_llm_time_s = round(time.perf_counter() - t_diag_start, 4)
        snap_after_diag = self.token_snapshot()
        diag_prompt_tokens = snap_after_diag[0] - snap_before_diag[0]
        diag_completion_tokens = snap_after_diag[1] - snap_before_diag[1]

        if parsed_diag is None:
            return None, {**empty_log, "error": "diagnosis_json_parse_failed", "raw": raw_diag}

        diagnosis = parsed_diag.get("diagnosis", {})
        tuning_action = parsed_diag.get("tuning_action", {})

        # ── 7. Phase 2: Proposal (LLM call with timing) ───────────────────
        diag_summary = json.dumps({
            "classification": classification,
            "diagnosis": diagnosis,
            "tuning_action": tuning_action,
        }, indent=2)

        step_size_guidance = (
            f"Classification: **{classification}** — use "
            f"{'larger' if 'far' in classification else 'conservative'} steps."
        )

        proposal_prompt = context_text + "\n".join([
            "",
            "## Diagnosis from Analysis Phase",
            "```json",
            diag_summary,
            "```",
            "",
            "## Step-Size Guidance",
            step_size_guidance,
            "",
            "## Task: Propose Candidate",
            "Based on the diagnosis above, propose exactly ONE candidate configuration.",
            "Each trial builds an index with (M, B, efConstruction) then sweeps (ef, al).",
            "",
            "Remember:",
            "- ef must be <= efConstruction",
            "- Higher M and B increase memory and build time",
            "- B controls filtering granularity; lower B = denser per-slot graphs",
            "- al is the per-slot search depth; ef is the global beam width",
            "- If inclusiveness is low (<70%), reduce B or increase M",
            "",
            "Return a strict JSON object:",
            "```json",
            "{",
            '  "candidate": {"M": <int>, "B": <int>, "efConstruction": <int>, "ef": <int>, "al": <int>},',
            '  "rationale": "<one sentence explaining the choice>"',
            "}",
            "```",
        ])

        snap_before_prop = self.token_snapshot()
        t_prop_start = time.perf_counter()
        try:
            raw_prop = self._invoke_llm(proposal_prompt)
            parsed_prop = self._extract_json_payload(raw_prop)
        except Exception as exc:
            return None, {**empty_log, "error": f"proposal_llm_error: {exc}",
                          "diagnosis": diagnosis, "tuning_action": tuning_action}
        prop_llm_time_s = round(time.perf_counter() - t_prop_start, 4)
        snap_after_prop = self.token_snapshot()
        prop_prompt_tokens = snap_after_prop[0] - snap_before_prop[0]
        prop_completion_tokens = snap_after_prop[1] - snap_before_prop[1]

        if parsed_prop is None:
            # Retry proposal once.
            retry_prop_prompt = (
                proposal_prompt + "\n\n⚠️ Previous response could not be parsed. "
                "Please return ONLY the JSON object with candidate and rationale."
            )
            try:
                raw_prop = self._invoke_llm(retry_prop_prompt)
                parsed_prop = self._extract_json_payload(raw_prop)
                prop_llm_time_s += round(time.perf_counter() - t_prop_start, 4)
                snap_retry = self.token_snapshot()
                prop_prompt_tokens += snap_retry[0] - snap_after_prop[0]
                prop_completion_tokens += snap_retry[1] - snap_after_prop[1]
            except Exception:
                pass

        if parsed_prop is None:
            return None, {**empty_log, "error": "proposal_json_parse_failed",
                          "diagnosis": diagnosis, "tuning_action": tuning_action}

        candidate_raw = parsed_prop.get("candidate")
        if not isinstance(candidate_raw, dict):
            return None, {**empty_log, "error": "missing_candidate",
                          "diagnosis": diagnosis, "tuning_action": tuning_action}

        # ── 8. Validate candidate ──────────────────────────────────────────
        try:
            candidate_params = self.canonicalize(candidate_raw, allowed_values_override)
        except Exception as exc:
            return None, {**empty_log, "error": f"canonicalize_failed: {exc}",
                          "diagnosis": diagnosis, "tuning_action": tuning_action}

        last_params = None
        if last_trial is not None and last_trial.get("status") == "success":
            last_params = last_trial.get("params") or {}
            if isinstance(last_params, dict):
                last_params = self.canonicalize(last_params, allowed_values_override)

        compliant, step_msg = self._check_step_compliance(candidate_params, last_params, classification)
        if not compliant:
            # Retry proposal once with step-size feedback.
            retry_prompt = (
                proposal_prompt + f"\n\n⚠️ Your previous candidate was rejected: {step_msg}\n"
                "Please propose a new candidate within the step-size bounds."
            )
            try:
                raw2 = self._invoke_llm(retry_prompt)
                prop_llm_time_s += round(time.perf_counter() - t_prop_start, 4)
                snap_retry2 = self.token_snapshot()
                prop_prompt_tokens += snap_retry2[0] - snap_after_prop[0]
                prop_completion_tokens += snap_retry2[1] - snap_after_prop[1]
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
            "note": str(parsed_prop.get("rationale", "")).strip(),
        }

        diag_log = {
            "ok": True,
            "diagnosis": diagnosis,
            "candidate": candidate,
            "classification": classification,
            "tuning_action": tuning_action,
            "error": "",
            "source": "diagnostic_two_phase",
            "prompt_payload": {},
            # ── Two-phase timing ──
            "diagnosis_llm_time_s": diag_llm_time_s,
            "diagnosis_prompt_tokens": diag_prompt_tokens,
            "diagnosis_completion_tokens": diag_completion_tokens,
            "proposal_llm_time_s": prop_llm_time_s,
            "proposal_prompt_tokens": prop_prompt_tokens,
            "proposal_completion_tokens": prop_completion_tokens,
            "total_llm_time_s": round(diag_llm_time_s + prop_llm_time_s, 4),
            "total_prompt_tokens_this_round": diag_prompt_tokens + prop_prompt_tokens,
            "total_completion_tokens_this_round": diag_completion_tokens + prop_completion_tokens,
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
            if abs(current - last) > bound:
                if current > last:
                    clamped[name] = type(clamped[name])(last + bound)
                else:
                    clamped[name] = type(clamped[name])(last - bound)
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

        Keys are formed on BUILD_PARAM_ORDER so that ef/al variations
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
                    "note": "Baseline fallback (M=16, B=6, efC=200, ef=40, al=32)",
                })
        except Exception:
            pass

        # Generate neighbors by varying one BUILD parameter at a time.
        deltas = [
            {"M": 8}, {"M": 24}, {"M": 32}, {"M": 48},
            {"B": 4}, {"B": 8},
            {"efConstruction": 100}, {"efConstruction": 400}, {"efConstruction": 600},
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

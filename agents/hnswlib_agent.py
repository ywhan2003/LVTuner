import copy
import itertools
import json
import logging
import math
import random
import re
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.hnswlib_metrics import BUILD_PARAM_ORDER, build_params_to_key


logger = logging.getLogger("hnswlib_agent.llm")


def _format_memory_context_section(memory_context: Dict[str, Any]) -> str:
    """Render a CurrentTaskMemory context dict into a prompt section.

    Delegates to ``CurrentTaskMemory.format_memory_context`` to avoid
    duplicating the formatting logic.
    """
    try:
        from utils.current_task_memory import CurrentTaskMemory

        return CurrentTaskMemory.format_memory_context(memory_context)
    except ImportError:
        return ""


PARAM_ORDER = ["M", "ef_construction", "ef"]
SKILL_NAMES = [
    "best_neighborhood_exploration",
    "construction_parameter_perturbation",
    "search_parameter_adjustment",
]
NODE_STATES = [
    "recall_too_high",
    "feasible_near_boundary",
    "infeasible_near_threshold",
    "infeasible_far",
    "high_uncertainty",
]
REASONING_MODES = {"single_agent", "multi_agent", "frontier"}


# ── Deprecated mechanism library placeholder (kept for reference only) ────
# Diagnostic reasoning now goes directly from signal diagnosis to candidate
# proposals without an intermediate mechanism layer.
_MECHANISM_LIBRARY_DEPRECATED: List[Dict[str, Any]] = [
    # === Graph Density (M) ===
    {
        "name": "reduce_density",
        "aspect": "graph_density",
        "parameter": "M",
        "direction": "down",
        "description": (
            "Reduce M to lower graph out-degree. Fewer edges per node means "
            "fewer neighbor checks per hop during search → QPS↑. "
            "The recall(ef) curve shifts downward slightly; ef* may increase "
            "by 1~2, but the traversal savings dominate when M is over-provisioned."
        ),
        "diagnostic_conditions": [
            "Fix efC: smaller M → Q_τ not worse or better (M is over-provisioned)",
            "recall margin > 0 (safety constraint: can absorb slight recall drop)",
            "M not at lower bound",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "recall(ef) shifts down slightly, QPS(ef) shifts up",
            "ef_star_prediction": "ef* may increase by 1~2, still feasible",
            "QPS_prediction": "Q_τ > current best",
            "feasible_prediction": True,
        },
    },
    {
        "name": "increase_density",
        "aspect": "graph_density",
        "parameter": "M",
        "direction": "up",
        "description": (
            "Raise M to increase graph out-degree. More edges per node creates "
            "more paths to ground-truth neighbors, lifting the entire recall(ef) "
            "curve. Use when the graph ceiling (max_recall at ef=efC) is too low."
        ),
        "diagnostic_conditions": [
            "Fix efC: larger M → Q_τ↑ or max_recall↑ (M is under-provisioned)",
            "OR all observed constructions are infeasible (systemic under-connectivity)",
            "OR ef* = efC and still infeasible (ef at hard ceiling, must change construction)",
            "M not at upper bound",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "recall(ef) shifts up, QPS(ef) shifts down",
            "ef_star_prediction": "ef* exists (build becomes feasible) or ef* decreases",
            "QPS_prediction": "Q_τ improves if previously infeasible",
            "feasible_prediction": True,
        },
    },
    # === Edge Precision (efC) ===
    {
        "name": "increase_precision",
        "aspect": "edge_precision",
        "parameter": "ef_construction",
        "direction": "up",
        "description": (
            "Raise ef_construction to enlarge the candidate pool during construction. "
            "Better edge selection quality shifts the recall(ef) curve upward "
            "especially at low ef, allowing a smaller ef* to clear τ. "
            "Downside: larger candidate pool → edges spread more evenly across "
            "nodes → effective out-degree rises → QPS(ef) shifts down slightly. "
            "Net Q_τ improves when efC was under-provisioned."
        ),
        "diagnostic_conditions": [
            "Fix M: larger efC → Q_τ↑ (efC is under-provisioned — ef* compression gain exceeds QPS cost)",
            "efC not at upper bound",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "recall(ef) shifts up at low ef, QPS(ef) shifts down slightly",
            "ef_star_prediction": "ef* decreases, feasible",
            "QPS_prediction": "Q_τ > current best",
            "feasible_prediction": True,
        },
    },
    {
        "name": "reduce_precision",
        "aspect": "edge_precision",
        "parameter": "ef_construction",
        "direction": "down",
        "description": (
            "Reduce ef_construction to shrink the candidate pool during construction. "
            "Smaller candidate pool → edges become more locally-biased → effective "
            "out-degree drops → QPS(ef) shifts up. "
            "Use when efC is over-provisioned: increasing it further yields "
            "diminishing recall gains while QPS continues to degrade."
        ),
        "diagnostic_conditions": [
            "Fix M: larger efC → Q_τ flat or down (efC is over-provisioned / saturated)",
            "recall margin > 0 (safety constraint: can absorb slight recall drop at low ef)",
            "efC not at lower bound",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "QPS(ef) shifts up, recall(ef) shifts down slightly at low ef",
            "ef_star_prediction": "ef* may increase by 1~2, still feasible",
            "QPS_prediction": "Q_τ > current best",
            "feasible_prediction": True,
        },
    },
    # === Meta-strategies ===
    {
        "name": "boundary_refinement",
        "aspect": "meta",
        "parameter": "either",
        "direction": "small_move",
        "description": (
            "Make a small local move within the subgroup elite region when "
            "cross-construction comparisons show no clear over/under-provisioning "
            "signal — the current construction is near-optimal."
        ),
        "diagnostic_conditions": [
            "feasible is true and a stable optimum has been found",
            "cross-construction comparisons show both M and efC are near their optimal values",
            "subgroup elite region provides a local direction Δ(g) for fine-tuning",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "minimal (local exploitation)",
            "ef_star_prediction": "ef* near current best",
            "QPS_prediction": "Q_τ ≥ current best (marginal improvement)",
            "feasible_prediction": True,
        },
    },
    {
        "name": "uncertainty_exploration",
        "aspect": "meta",
        "parameter": "either",
        "direction": "probe",
        "description": (
            "Probe a region of construction space where the GP surface has high "
            "posterior uncertainty. Gathers information to reduce uncertainty "
            "and clarify dominance relationships."
        ),
        "diagnostic_conditions": [
            "surface_uncertainty shows high σ region with few observations",
            "BO information_gain is high for constructions in that region",
            "no clearly dominating feasible construction exists in that region",
        ],
        "falsifiable_prediction_schema": {
            "frontier_shift": "uncertain (exploration)",
            "ef_star_prediction": "unpredictable — information-gathering move",
            "QPS_prediction": "may or may not improve — information value is the goal",
            "feasible_prediction": None,
        },
    },
]
RANGE_STEP_BUCKETS = 16
FLOAT_ROUND_DECIMALS = 12




# ── Signal Detail Procedures ──────────────────────────────────────────────

def _node_to_detail(node: Dict[str, Any]) -> str:
    """Format a diagnostic tree node as a readable detail block for Phase 1b."""
    node_id = node.get("node_id", "?")
    metrics = node.get("metrics", "")
    description = node.get("description", "")
    solution = node.get("solution", "")
    lines = [f"## {node_id}: {metrics}", ""]
    if description:
        lines.append(f"**Description**: {description}")
    if solution:
        lines.append(f"**Solution**: {solution}")
    return "\n".join(lines)


def params_to_key(params: Dict[str, Any], order: Sequence[str] = PARAM_ORDER) -> Tuple[Any, ...]:
    return tuple(params[name] for name in order)


def key_to_params(key: Tuple[Any, ...], order: Sequence[str] = PARAM_ORDER) -> Dict[str, Any]:
    return {order[i]: key[i] for i in range(len(order))}


def _parse_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {field_name}: {value}") from exc
    else:
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")
    return parsed


def _is_integral_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        try:
            return float(value.strip()).is_integer()
        except ValueError:
            return False
    return False


def _normalize_numeric_list(values: Sequence[Any]) -> List[Any]:
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Parameter value list cannot be empty.")
    return unique_values


@dataclass
class ParameterDomain:
    kind: str
    values: List[Any] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    is_integer: bool = False

    @classmethod
    def from_spec(cls, name: str, spec: Any) -> "ParameterDomain":
        if isinstance(spec, dict):
            if "values" in spec:
                return cls(kind="discrete", values=_normalize_numeric_list(spec["values"]))
            if "min" in spec or "max" in spec:
                if "min" not in spec or "max" not in spec:
                    raise ValueError(f"Parameter '{name}' range spec must include min and max.")
                min_number = _parse_number(spec["min"], f"{name}.min")
                max_number = _parse_number(spec["max"], f"{name}.max")
                if min_number > max_number:
                    raise ValueError(f"Parameter '{name}' range spec requires min <= max.")
                is_integer = _is_integral_number(spec["min"]) and _is_integral_number(spec["max"])
                if is_integer:
                    return cls(
                        kind="range",
                        min_value=int(round(min_number)),
                        max_value=int(round(max_number)),
                        is_integer=True,
                    )
                return cls(
                    kind="range",
                    min_value=round(min_number, FLOAT_ROUND_DECIMALS),
                    max_value=round(max_number, FLOAT_ROUND_DECIMALS),
                    is_integer=False,
                )
            raise ValueError(f"Parameter '{name}' spec must be a list, values object, or min/max range.")
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            return cls(kind="discrete", values=_normalize_numeric_list(spec))
        raise ValueError(f"Parameter '{name}' spec must be a list, values object, or min/max range.")

    def to_spec(self) -> Dict[str, Any]:
        if self.kind == "discrete":
            return {"kind": "discrete", "values": list(self.values or [])}
        return {
            "kind": "range",
            "min": self.min_value,
            "max": self.max_value,
            "integer": bool(self.is_integer),
        }

    def step_size(self) -> float:
        if self.kind != "range":
            return 1.0
        span = float(self.max_value) - float(self.min_value)  # type: ignore[arg-type]
        if span <= 0.0:
            return 1.0
        if self.is_integer:
            return float(max(1, int(round(span / RANGE_STEP_BUCKETS))))
        return span / float(RANGE_STEP_BUCKETS)

    def midpoint(self) -> Any:
        if self.kind == "discrete":
            values = list(self.values or [])
            return values[len(values) // 2]
        value = (float(self.min_value) + float(self.max_value)) / 2.0  # type: ignore[arg-type]
        if self.is_integer:
            return int(round(value))
        return round(value, FLOAT_ROUND_DECIMALS)

    def edge(self, side: str) -> Any:
        if self.kind == "discrete":
            values = list(self.values or [])
            return values[0] if side == "low" else values[-1]
        return self.min_value if side == "low" else self.max_value


@dataclass
class ParameterSpace:
    domains: Dict[str, ParameterDomain]
    order: List[str]

    @classmethod
    def from_config(cls, params_cfg: Dict[str, Any], order: Sequence[str] = PARAM_ORDER) -> "ParameterSpace":
        resolved_order = list(order)
        if resolved_order != PARAM_ORDER:
            raise ValueError(f"HNSW parameter order must be exactly {PARAM_ORDER}.")
        domains: Dict[str, ParameterDomain] = {}
        for name in resolved_order:
            if name not in params_cfg:
                raise ValueError(f"Missing HNSW parameter space for '{name}'.")
            domains[name] = ParameterDomain.from_spec(name, params_cfg[name])
        return cls(domains=domains, order=resolved_order)

    @property
    def values(self) -> Dict[str, List[Any]]:
        result: Dict[str, List[Any]] = {}
        for name in self.order:
            domain = self.domains[name]
            if domain.kind == "discrete":
                result[name] = list(domain.values or [])
                continue
            min_value = domain.min_value
            max_value = domain.max_value
            if min_value is None or max_value is None:
                result[name] = []
                continue
            if domain.is_integer:
                step = max(1, int(round(domain.step_size())))
                values = list(range(int(min_value), int(max_value) + 1, step))
                if values[-1] != int(max_value):
                    values.append(int(max_value))
                result[name] = values
            else:
                step = domain.step_size()
                values = [round(float(min_value) + i * step, FLOAT_ROUND_DECIMALS) for i in range(RANGE_STEP_BUCKETS + 1)]
                values[-1] = round(float(max_value), FLOAT_ROUND_DECIMALS)
                result[name] = values
        return result

    def export_parameter_space(self, constraints: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
        domains = self._resolve_constraints(constraints)
        return {name: domains[name].to_spec() for name in self.order}

    def normalize_constraints(self, raw_constraints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return self.export_parameter_space(raw_constraints)

    def _coerce_discrete(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        values = list(domain.values or [])
        if value in values:
            return value
        parsed: Any = value
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                try:
                    parsed = float(value.strip())
                except ValueError:
                    parsed = value
        if parsed in values:
            return parsed
        if isinstance(parsed, (int, float)):
            for candidate in values:
                if isinstance(candidate, (int, float)) and abs(float(candidate) - float(parsed)) <= 1e-9:
                    return candidate
            # Snap to nearest discrete value instead of rejecting.
            numeric = [v for v in values if isinstance(v, (int, float))]
            if numeric:
                return min(numeric, key=lambda v: abs(float(v) - float(parsed)))
        raise ValueError(f"Invalid value for {key}: {value}")

    def _coerce_range(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        parsed = _parse_number(value, key)
        min_value = domain.min_value
        max_value = domain.max_value
        if min_value is None or max_value is None:
            raise ValueError(f"Range domain for {key} is missing min/max.")
        if domain.is_integer:
            rounded = round(parsed)
            if abs(parsed - rounded) > 1e-9:
                raise ValueError(f"Invalid value for {key}: {value}")
            normalized = int(rounded)
            # Clamp to bounds instead of rejecting (LLM may propose slightly out-of-range).
            return max(int(min_value), min(int(max_value), normalized))
        normalized = round(float(parsed), FLOAT_ROUND_DECIMALS)
        # Clamp to bounds.
        normalized = max(float(min_value), min(float(max_value), normalized))
        return normalized

    def _coerce_value(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        if domain.kind == "discrete":
            return self._coerce_discrete(key, value, domain)
        return self._coerce_range(key, value, domain)

    def _subset_domain(self, key: str, base: ParameterDomain, override: ParameterDomain) -> ParameterDomain:
        if base.kind == "discrete":
            base_values = list(base.values or [])
            if override.kind == "discrete":
                values = [self._coerce_discrete(key, value, base) for value in (override.values or [])]
                return ParameterDomain(kind="discrete", values=_normalize_numeric_list(values))
            lo = self._coerce_range(
                key,
                override.min_value,
                ParameterDomain(
                    kind="range",
                    min_value=min(base_values),
                    max_value=max(base_values),
                    is_integer=all(_is_integral_number(value) for value in base_values),
                ),
            )
            hi = self._coerce_range(
                key,
                override.max_value,
                ParameterDomain(
                    kind="range",
                    min_value=min(base_values),
                    max_value=max(base_values),
                    is_integer=all(_is_integral_number(value) for value in base_values),
                ),
            )
            values = [value for value in base_values if float(lo) <= float(value) <= float(hi)]
            if not values:
                raise ValueError(f"allowed_values_override for '{key}' yields an empty subset")
            return ParameterDomain(kind="discrete", values=values)

        if override.kind == "discrete":
            values = [self._coerce_range(key, value, base) for value in (override.values or [])]
            return ParameterDomain(kind="discrete", values=_normalize_numeric_list(values))

        lo_value = self._coerce_range(key, override.min_value, base)
        hi_value = self._coerce_range(key, override.max_value, base)
        if float(lo_value) > float(hi_value):
            raise ValueError(f"allowed_values_override for '{key}' has min > max")
        return ParameterDomain(
            kind="range",
            min_value=int(lo_value) if base.is_integer else round(float(lo_value), FLOAT_ROUND_DECIMALS),
            max_value=int(hi_value) if base.is_integer else round(float(hi_value), FLOAT_ROUND_DECIMALS),
            is_integer=base.is_integer,
        )

    def _resolve_constraints(self, constraints: Dict[str, Any] | None) -> Dict[str, ParameterDomain]:
        if constraints is None:
            return self.domains
        resolved: Dict[str, ParameterDomain] = {}
        for name in self.order:
            if name not in constraints:
                raise ValueError(f"allowed_values_override for '{name}' cannot be empty")
            override = ParameterDomain.from_spec(name, constraints[name])
            resolved[name] = self._subset_domain(name, self.domains[name], override)
        return resolved

    def canonicalize(self, params: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Dict[str, Any]:
        domains = self._resolve_constraints(constraints)
        return {name: self._coerce_value(name, params[name], domains[name]) for name in self.order}

    def is_valid(self, params: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> bool:
        try:
            self.canonicalize(params, constraints)
            return True
        except Exception:
            return False

    def out_of_constraint_fields(self, params: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        domains = self._resolve_constraints(constraints)
        fields: List[str] = []
        for name in self.order:
            if name not in params:
                fields.append(name)
                continue
            try:
                self._coerce_value(name, params[name], domains[name])
            except Exception:
                fields.append(name)
        return fields

    def baseline(self) -> Dict[str, Any]:
        return {name: self.domains[name].midpoint() for name in self.order}

    def edge_value(self, name: str, side: str) -> Any:
        return self.domains[name].edge(side)

    def normalized_position(self, name: str, value: Any, *, clamp: bool = False) -> float:
        domain = self.domains[name]
        if clamp:
            try:
                canonical = self._coerce_value(name, value, domain)
            except Exception:
                if domain.kind == "discrete":
                    values = list(domain.values or [])
                    if not values:
                        return 0.0
                    try:
                        parsed = float(value)
                    except Exception:
                        canonical = values[0]
                    else:
                        canonical = min(values, key=lambda candidate: abs(float(candidate) - parsed))
                else:
                    min_value = float(domain.min_value)  # type: ignore[arg-type]
                    max_value = float(domain.max_value)  # type: ignore[arg-type]
                    try:
                        parsed = float(value)
                    except Exception:
                        parsed = min_value
                    parsed = min(max_value, max(min_value, parsed))
                    canonical = int(round(parsed)) if domain.is_integer else round(parsed, FLOAT_ROUND_DECIMALS)
        else:
            canonical = self._coerce_value(name, value, domain)
        if domain.kind == "discrete":
            values = list(domain.values or [])
            idx = values.index(canonical)
            return idx / max(1, len(values) - 1)
        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        if max_value <= min_value:
            return 0.0
        return max(0.0, min(1.0, (float(canonical) - min_value) / (max_value - min_value)))

    def normalized_vector(self, params: Dict[str, Any], *, clamp: bool = False) -> Tuple[float, ...]:
        return tuple(self.normalized_position(name, params[name], clamp=clamp) for name in self.order)

    def value_from_position(self, name: str, pos: float) -> Any:
        domain = self.domains[name]
        clipped = max(0.0, min(1.0, float(pos)))
        if domain.kind == "discrete":
            values = list(domain.values or [])
            idx = int(round(clipped * max(0, len(values) - 1)))
            return values[max(0, min(len(values) - 1, idx))]
        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        value = min_value + clipped * (max_value - min_value)
        if domain.is_integer:
            return int(max(int(min_value), min(int(max_value), round(value))))
        return round(value, FLOAT_ROUND_DECIMALS)

    def value_at_delta(self, name: str, value: Any, delta: int) -> Any | None:
        domain = self.domains[name]
        if domain.kind == "discrete":
            values = list(domain.values or [])
            current = self._coerce_discrete(name, value, domain)
            idx = values.index(current) + int(delta)
            if idx < 0 or idx >= len(values):
                return None
            return values[idx]
        current = self._coerce_range(name, value, domain)
        shifted = float(current) + int(delta) * domain.step_size()
        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        shifted = min(max_value, max(min_value, shifted))
        if domain.is_integer:
            return int(max(int(min_value), min(int(max_value), round(shifted))))
        return round(shifted, FLOAT_ROUND_DECIMALS)

    def sample_random_params(
        self,
        rng: random.Random,
        constraints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        domains = self._resolve_constraints(constraints)
        sampled: Dict[str, Any] = {}
        for name in self.order:
            domain = domains[name]
            if domain.kind == "discrete":
                sampled[name] = rng.choice(list(domain.values or []))
                continue
            if domain.is_integer:
                sampled[name] = rng.randint(int(domain.min_value), int(domain.max_value))  # type: ignore[arg-type]
            else:
                value = rng.uniform(float(domain.min_value), float(domain.max_value))  # type: ignore[arg-type]
                sampled[name] = round(value, FLOAT_ROUND_DECIMALS)
        return sampled

    def estimated_cardinality(self, constraints: Dict[str, Any] | None = None) -> int:
        domains = self._resolve_constraints(constraints)
        count = 1
        for name in self.order:
            domain = domains[name]
            if domain.kind == "discrete":
                count *= max(1, len(domain.values or []))
            elif domain.is_integer:
                count *= max(1, int(domain.max_value) - int(domain.min_value) + 1)  # type: ignore[arg-type]
            else:
                count *= RANGE_STEP_BUCKETS + 1
        return count


class HNSWLIBTuningAgent:
    def __init__(
        self,
        params_cfg: Dict[str, Any],
        seed: int = 42,
        agentic_cfg: Dict[str, Any] | None = None,
        model_cfg: Dict[str, Any] | None = None,
        skill_dir: str | Path | None = None,
        llm_caller: Any | None = None,
    ) -> None:
        self.space = ParameterSpace.from_config(params_cfg, order=PARAM_ORDER)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.agentic_cfg = agentic_cfg or {}
        self.model_cfg = model_cfg or {}
        self.llm_caller = llm_caller
        if self.agentic_cfg.get("enabled", True) is False:
            raise ValueError("agentic.enabled must be true — the hnswlib pipeline requires LLM proposals.")
        self.enable_agentic = True
        # Configurable near/far boundary override (None = use default adaptive logic).
        override_val = float(agentic_cfg.get("near_boundary", 0))
        self.near_boundary_override = override_val if override_val > 0 else None
        # Token usage counters (cumulative across all LLM calls).
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # First-call token counters (initialization phase).
        self.init_prompt_tokens = 0
        self.init_completion_tokens = 0
        self._first_call_done = False
        # Last LLM call wall-clock time (for per-call timing).
        self.last_call_elapsed_s = 0.0
        # Per-skill token tracking (for granular split)
        self.last_skill_name = ""
        self.last_skill_prompt_tokens = 0
        self.last_skill_completion_tokens = 0
        # Cumulative per-phase tracking (reset each round)
        self.compliance_prompt_tokens = 0
        self.compliance_completion_tokens = 0
        self.compliance_elapsed_s = 0.0
        self.llm_reasoning_cfg = (
            self.agentic_cfg.get("llm_reasoning")
            if isinstance(self.agentic_cfg.get("llm_reasoning"), dict)
            else {}
        )
        if skill_dir is None:
            skill_dir = Path(__file__).resolve().parent.parent / "configs" / "prompts" / "hnswlib_skills"
        self.skill_dir = Path(skill_dir)
        self.alias_to_canonical = {
            "efC": "ef_construction",
            "efConstruction": "ef_construction",
            "ef_construction": "ef_construction",
        }

    def token_snapshot(self) -> tuple:
        """Return (total_prompt_tokens, total_completion_tokens) for delta computation."""
        return (self.total_prompt_tokens, self.total_completion_tokens)

    @property
    def param_order(self) -> List[str]:
        return list(PARAM_ORDER)

    @property
    def reasoning_mode(self) -> str:
        mode = str(self.llm_reasoning_cfg.get("mode", "single_agent")).strip().lower()
        if mode not in REASONING_MODES:
            return "single_agent"
        return mode

    def canonicalize(
        self,
        params: Dict[str, Any],
        domain_constraints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized = dict(params or {})
        for alias, canonical in self.alias_to_canonical.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized[alias]
        forbidden = [name for name in ["al", "B", "efConstruction"] if name in normalized and name != "efConstruction"]
        if forbidden:
            raise ValueError(f"RFANNS-only parameters are not valid for HNSW: {forbidden}")
        result = self.space.canonicalize(normalized, constraints=domain_constraints)
        # Enforce ef ≤ efC
        if "ef" in result and "ef_construction" in result:
            result["ef"] = min(result["ef"], result["ef_construction"])
        return result

    def _extract_json_payload(self, text: str) -> Dict[str, Any] | None:
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

    def _safe_format(self, template: str, payload: Dict[str, Any]) -> str:
        rendered = template
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                value_text = str(value)
            rendered = rendered.replace("{" + key + "}", value_text)
        return rendered

    def _skill_template(self, skill_name: str) -> str:
        path = self.skill_dir / f"{skill_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if "ef" in BUILD_PARAM_ORDER:
            return (
                "You are a HNSW tuning skill named {skill_name}.\n"
                "Tune M, ef_construction, and ef.  ef↑ → recall↑ QPS↓;  ef↓ → QPS↑ recall↓.\n"
                "Adjust any of ef / M / efC indicated by the evidence; ef changes are zero-rebuild, M/efC changes rebuild the index.\n"
                "Return strict JSON with skill_name, branch, and candidates.\n"
                "Context: {context_json}\n"
            )
        return (
            "You are a HNSW tuning skill named {skill_name}.\n"
            "Tune only build parameters M and ef_construction; ef is placeholder-only.\n"
            "Return strict JSON with skill_name, branch, and candidates.\n"
            "Context: {context_json}\n"
        )

    def _default_llm_call(self, skill_name: str, prompt: str) -> str:
        from openai import OpenAI

        model_name = self.model_cfg.get("model_name") or self.model_cfg.get("model")
        if not model_name:
            raise RuntimeError(
                "HNSW agent model_name is not configured — "
                "set LLM_MODEL_NAME (or per-pipeline override) in .env."
            )
        base_url = self.model_cfg.get("url")
        api_key = self.model_cfg.get("authorization")
        temperature = float(self.model_cfg.get("temperature", 0.2))
        max_tokens = int(self.model_cfg.get("max_tokens", 2048))
        timeout_s = float(self.model_cfg.get("timeout_s", 120))
        if not base_url or "<base_url>" in str(base_url):
            raise RuntimeError("HNSWLIB agent model URL is not configured.")
        if not api_key or "<token>" in str(api_key):
            raise RuntimeError("HNSWLIB agent model authorization is not configured.")
        logger.info("LLM API call start skill=%s model=%s prompt_chars=%d timeout_s=%d",
                     skill_name, model_name, len(prompt), int(timeout_s))
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        t_call_start = time.time()
        extra_kwargs = {}
        reasoning_effort = self.model_cfg.get("reasoning_effort")
        if reasoning_effort:
            extra_kwargs["reasoning_effort"] = str(reasoning_effort)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        self.last_call_elapsed_s = round(time.time() - t_call_start, 4)
        logger.info("LLM API call done skill=%s response_chars=%d elapsed=%.3fs",
                     skill_name, len(response.choices[0].message.content or ""),
                     self.last_call_elapsed_s)
        if response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0
            if not self._first_call_done:
                self.init_prompt_tokens = response.usage.prompt_tokens or 0
                self.init_completion_tokens = response.usage.completion_tokens or 0
                self._first_call_done = True
        # Per-skill tracking (for granular split)
        self.last_skill_name = skill_name
        self.last_skill_prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        self.last_skill_completion_tokens = response.usage.completion_tokens if response.usage else 0
        return response.choices[0].message.content or ""

    def _invoke_llm(self, skill_name: str, prompt: str) -> str:
        """Dispatch an LLM call (custom caller or default OpenAI) and log the prompt.

        Logs the full outgoing prompt at DEBUG and a compact summary (prompt size +
        truncated preview) at INFO so runs can be traced without enabling DEBUG. The
        returned raw completion is logged at DEBUG as well.
        """
        logger.info(
            "LLM call skill=%s caller=%s prompt_chars=%d",
            skill_name,
            "custom" if self.llm_caller is not None else "default_openai",
            len(prompt),
        )
        logger.info("LLM prompt skill=%s\n%s", skill_name, prompt)
        if self.llm_caller is not None:
            try:
                raw = self.llm_caller(skill_name=skill_name, prompt=prompt)
            except TypeError:
                raw = self.llm_caller(skill_name, prompt)
        else:
            raw = self._default_llm_call(skill_name, prompt)
        raw = raw or ""
        logger.debug("LLM raw response skill=%s\n%s", skill_name, raw)
        return raw

    def _call_skill(self, skill_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        template = self._skill_template(skill_name)
        prompt = self._safe_format(
            template,
            {
                **payload,
                "skill_name": skill_name,
                "context_json": payload,
            },
        )
        raw = ""
        parsed = None
        error = None
        try:
            raw = self._invoke_llm(skill_name, prompt)
            parsed = self._extract_json_payload(raw)
            if parsed is None:
                error = "json_parse_failed"
        except Exception as exc:
            error = f"llm_error: {exc}"
        result = {
            "skill_name": skill_name,
            "ok": error is None,
            "error": error,
            "raw": raw,
            "parsed": parsed,
        }
        logger.info(
            "LLM skill result skill=%s ok=%s error=%s parsed=%s",
            skill_name,
            result["ok"],
            error or "",
            json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "<none>",
        )
        if error is not None:
            raise RuntimeError(
                f"LLM skill '{skill_name}' failed: {error} — "
                "candidate proposals are mandatory, no rule-based fallback."
            )
        return result

    def _observation_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are the HNSW tuning agent's tree-search observation step.\n"
            "Classify the current build candidate using only HNSW build-space context and BO/SCBO statistics.\n"
            "Do not introduce proposer, selector, reflector, RFANNS parameters, or non-HNSW knobs.\n"
            "Available node_state values: recall_too_high, feasible_near_boundary, "
            "infeasible_near_threshold, infeasible_far, high_uncertainty.\n"
            "Choose next_skill only from: best_neighborhood_exploration, "
            "construction_parameter_perturbation, search_parameter_adjustment, or null.\n"
            "search_parameter_adjustment is a legacy branch name; in this pipeline it still proposes build-only moves.\n"
            "Return strict JSON only with node_id, node_state, expand, next_skill, "
            "expansion_intent, reason, and priority_score.\n\n"
            f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _call_observation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = ""
        parsed = None
        error = None
        prompt = self._observation_prompt(payload)
        try:
            raw = self._invoke_llm("hnsw_tree_observation", prompt)
            parsed = self._extract_json_payload(raw)
            if parsed is None:
                error = "json_parse_failed"
        except Exception as exc:
            error = f"llm_error: {exc}"
        result = {"ok": error is None, "error": error, "raw": raw, "parsed": parsed}
        logger.info(
            "LLM observation result ok=%s error=%s parsed=%s",
            result["ok"],
            error or "",
            json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "<none>",
        )
        if error is not None:
            raise RuntimeError(
                f"LLM observation failed: {error} — rule-based observation fallback is disabled."
            )
        return result

    def _fallback_next_skill(self, node_state: str, current_skill: str, action_stats: Dict[str, Any]) -> str | None:
        if node_state == "recall_too_high":
            return "search_parameter_adjustment"
        if node_state == "feasible_near_boundary":
            return "best_neighborhood_exploration"
        if node_state == "infeasible_near_threshold":
            return "search_parameter_adjustment"
        if node_state == "infeasible_far":
            return None
        if node_state == "high_uncertainty":
            constraint_mean = float(action_stats.get("recall_constraint_mean", 0.0))
            if constraint_mean < 0.0:
                return "search_parameter_adjustment"
            return current_skill if current_skill in SKILL_NAMES else "best_neighborhood_exploration"
        return None

    def _fallback_observation(
        self,
        *,
        node: Dict[str, Any],
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        max_depth: int,
    ) -> Dict[str, Any]:
        action_stats = node.get("action_stats") or node.get("scbo_action") or {}
        constraint_mean = float(action_stats.get("recall_constraint_mean", 0.0))
        constraint_std = float(action_stats.get("recall_constraint_std", 0.0))
        qps_mean = float(action_stats.get("qps_mean", 0.0))
        qps_std = float(action_stats.get("qps_std", 0.0))
        feasible_prob = float(action_stats.get("feasible_probability", node.get("joint_feasible_prob", 0.0)))
        filter_reason = str(action_stats.get("filter_reason", node.get("filter_reason", "")) or "")
        recall_slack = float(stage_policy.get("recall_slack", root_state.get("recall_slack", 0.0)))
        near_width = max(recall_slack, 0.005)
        high_margin = max(3.0 * max(recall_slack, 0.001), 0.02)
        pf_threshold = float(self.agentic_cfg.get("tree_search", {}).get("high_uncertainty_min_feasible_prob", 0.35))
        qps_uncertain = qps_std > max(abs(qps_mean) * 0.25, 1e-6)
        recall_uncertain = constraint_std >= max(0.05, near_width * 4.0)

        if feasible_prob >= pf_threshold and (recall_uncertain or qps_uncertain):
            node_state = "high_uncertainty"
        elif constraint_mean >= high_margin:
            node_state = "recall_too_high"
        elif constraint_mean >= -near_width:
            node_state = "feasible_near_boundary" if constraint_mean >= 0.0 else "infeasible_near_threshold"
        else:
            node_state = "infeasible_far"

        depth = int(node.get("depth", 1))
        next_skill = self._fallback_next_skill(node_state, str(node.get("source_skill", "")), action_stats)
        expand = bool(next_skill and depth < int(max_depth))
        if filter_reason == "qps_upper_bound_below_current_best":
            expand = False
            next_skill = None
        if node_state == "high_uncertainty" and feasible_prob < pf_threshold:
            expand = False
            next_skill = None
        intent_by_state = {
            "recall_too_high": "Reduce search width or construction cost while preserving enough recall.",
            "feasible_near_boundary": "Run small local exploitation around the boundary-feasible point.",
            "infeasible_near_threshold": "Apply a small recall compensation move to try to become feasible.",
            "infeasible_far": "Stop expansion because the predicted recall gap is too large.",
            "high_uncertainty": "Probe a promising uncertain region without taking RFANNS parameters.",
        }
        return {
            "node_id": str(node.get("node_id", "")),
            "node_state": node_state,
            "expand": expand,
            "next_skill": next_skill,
            "expansion_intent": intent_by_state[node_state],
            "reason": (
                f"fallback observation: constraint_mean={constraint_mean:.6f}, "
                f"constraint_std={constraint_std:.6f}, feasible_prob={feasible_prob:.6f}, "
                f"filter_reason={filter_reason or 'none'}"
            ),
            "priority_score": float(node.get("score", 0.0)) + 0.1 * feasible_prob,
            "source": "fallback",
        }

    def _sanitize_observation(
        self,
        parsed: Dict[str, Any] | None,
        *,
        fallback: Dict[str, Any],
        node: Dict[str, Any],
        max_depth: int,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return fallback
        node_state = str(parsed.get("node_state", "")).strip()
        if node_state not in NODE_STATES:
            return fallback
        raw_next = parsed.get("next_skill")
        next_skill = None if raw_next in {None, "", "null"} else str(raw_next).strip()
        if next_skill is not None and next_skill not in SKILL_NAMES:
            return fallback
        expand = bool(parsed.get("expand", False))
        if int(node.get("depth", 1)) >= int(max_depth):
            expand = False
            next_skill = None
        if expand and next_skill is None:
            return fallback
        try:
            priority_score = float(parsed.get("priority_score", fallback.get("priority_score", 0.0)))
        except Exception:
            priority_score = float(fallback.get("priority_score", 0.0))
        return {
            "node_id": str(parsed.get("node_id") or node.get("node_id", "")),
            "node_state": node_state,
            "expand": expand,
            "next_skill": next_skill,
            "expansion_intent": str(parsed.get("expansion_intent") or fallback.get("expansion_intent", "")),
            "reason": str(parsed.get("reason") or fallback.get("reason", "")),
            "priority_score": priority_score,
            "source": "llm",
        }

    def _observe_node(
        self,
        *,
        node: Dict[str, Any],
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        max_depth: int,
        scbo_reflection: Dict[str, Any] | None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fallback = self._fallback_observation(
            node=node,
            root_state=root_state,
            stage_policy=stage_policy,
            max_depth=max_depth,
        )
        llm_result = {"ok": False, "error": "agentic_disabled", "raw": "", "parsed": None}
        if self.enable_agentic:
            payload = {
                "root_state": root_state,
                "node": {
                    "node_id": node.get("node_id", ""),
                    "parent_id": node.get("parent_id"),
                    "depth": node.get("depth"),
                    "source_skill": node.get("source_skill"),
                    "branch": node.get("branch"),
                    "params": node.get("params"),
                    "thinking": node.get("thinking", ""),
                    "expected_behavior": node.get("expected_behavior", {}),
                },
                "action_stats": node.get("action_stats") or node.get("scbo_action", {}),
                "accepted": bool(node.get("accepted", True)),
                "filter_reason": str(node.get("filter_reason", "") or ""),
                "stage_policy": stage_policy,
                "available_skills": list(SKILL_NAMES),
                "state_definitions": {
                    "recall_too_high": "Recall has large positive margin; configuration is conservative and QPS can likely improve.",
                    "feasible_near_boundary": "Recall is feasible but close to the guardrail; use small exploitation moves.",
                    "infeasible_near_threshold": "Recall is slightly below the guardrail; small compensation may make it feasible.",
                    "infeasible_far": "Recall is far below the guardrail; stop expanding this node.",
                    "high_uncertainty": "BO confidence is low while feasible probability is not low; potential gain remains.",
                },
                "scbo_reflection": scbo_reflection or {},
                "output_schema": {
                    "node_id": node.get("node_id", ""),
                    "node_state": "one of NODE_STATES",
                    "expand": True,
                    "next_skill": "one of SKILL_NAMES or null",
                    "expansion_intent": "how the next Skill should react to this node state",
                    "reason": "short evidence-based observation",
                    "priority_score": 0.0,
                },
            }
            llm_result = self._call_observation(payload)
        observation = self._sanitize_observation(
            llm_result.get("parsed") if isinstance(llm_result, dict) else None,
            fallback=fallback,
            node=node,
            max_depth=max_depth,
        )
        if not bool(llm_result.get("ok", False)):
            observation["source"] = "fallback"
        return observation, {
            "node_id": node.get("node_id", ""),
            "llm_ok": bool(llm_result.get("ok", False)),
            "llm_error": llm_result.get("error", ""),
            "observation": observation,
        }

    def _aggregate_success_points(self, trials: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for trial in trials:
            if trial.get("status") != "success":
                continue
            params = trial.get("params")
            metrics = trial.get("metrics") or {}
            if not isinstance(params, dict) or "recall" not in metrics or "qps" not in metrics:
                continue
            try:
                canonical = self.canonicalize(params)
                build_params = {name: canonical[name] for name in BUILD_PARAM_ORDER}
            except Exception:
                continue
            key = build_params_to_key(build_params, BUILD_PARAM_ORDER)
            bucket = grouped.setdefault(
                key,
                {
                    "params": build_params,
                    "recall": [],
                    "qps": [],
                    "selected_ef": [],
                    "frontiers": [],
                    "build_time_s": [],
                    "index_size_mb": [],
                    "dist_comps": [],
                    "visited_nodes": [],
                    "candidate_distance_means": [],
                    "candidate_distance_stds": [],
                    "out_degree_means": [],
                    "in_degree_means": [],
                    "in_degree_stds": [],
                    "in_degree_maxs": [],
                },
            )
            bucket["recall"].append(float(metrics["recall"]))
            bucket["qps"].append(float(metrics["qps"]))
            if isinstance(metrics.get("build_time_s"), (int, float)):
                bucket["build_time_s"].append(float(metrics["build_time_s"]))
            if isinstance(metrics.get("index_size_mb"), (int, float)):
                bucket["index_size_mb"].append(float(metrics["index_size_mb"]))
            if isinstance(metrics.get("dist_comps_per_query"), (int, float)):
                bucket["dist_comps"].append(float(metrics["dist_comps_per_query"]))
            if isinstance(metrics.get("visited_nodes_per_query"), (int, float)):
                bucket["visited_nodes"].append(float(metrics["visited_nodes_per_query"]))
            if metrics.get("selected_ef") is not None:
                try:
                    bucket["selected_ef"].append(int(metrics["selected_ef"]))
                except Exception:
                    pass
            if isinstance(metrics.get("frontier"), list):
                bucket["frontiers"].append(metrics["frontier"])
            # ── New topology & candidate-distance metrics ──
            cds = metrics.get("candidate_distance_stats")
            if isinstance(cds, dict):
                if cds.get("mean") is not None:
                    bucket["candidate_distance_means"].append(float(cds["mean"]))
                if cds.get("std") is not None:
                    bucket["candidate_distance_stds"].append(float(cds["std"]))
            if isinstance(metrics.get("out_degree_mean"), (int, float)):
                bucket["out_degree_means"].append(float(metrics["out_degree_mean"]))
            if isinstance(metrics.get("in_degree_mean"), (int, float)):
                bucket["in_degree_means"].append(float(metrics["in_degree_mean"]))
            if isinstance(metrics.get("in_degree_std"), (int, float)):
                bucket["in_degree_stds"].append(float(metrics["in_degree_std"]))
            if isinstance(metrics.get("in_degree_max"), (int, float)):
                bucket["in_degree_maxs"].append(float(metrics["in_degree_max"]))
        points: List[Dict[str, Any]] = []
        for bucket in grouped.values():
            dc = float(median(bucket["dist_comps"])) if bucket["dist_comps"] else 0.0
            vn = float(median(bucket["visited_nodes"])) if bucket["visited_nodes"] else 0.0
            ef = int(median(bucket["selected_ef"])) if bucket["selected_ef"] else 1
            efC = int(bucket["params"].get("ef_construction", 1))
            points.append(
                {
                    "params": bucket["params"],
                    "recall": float(median(bucket["recall"])),
                    "build_time_s": float(median(bucket["build_time_s"])) if bucket["build_time_s"] else 0.0,
                    "index_size_mb": float(median(bucket["index_size_mb"])) if bucket["index_size_mb"] else 0.0,
                    "dist_comps_per_query": dc,
                    "visited_nodes_per_query": vn,
                    "qps": float(median(bucket["qps"])),
                    "selected_ef": ef if bucket["selected_ef"] else None,
                    "frontier": bucket["frontiers"][-1] if bucket["frontiers"] else [],
                    # ── New diagnostics ──
                    "dc_vn_ratio": dc / max(1.0, vn) if dc and vn else None,
                    "efC_ef_ratio": efC / max(1, ef) if efC and ef else None,
                    "candidate_distance_stats": (
                        {
                            "mean": float(median(bucket["candidate_distance_means"])),
                            "std": float(median(bucket["candidate_distance_stds"])),
                        }
                        if bucket["candidate_distance_means"]
                        else {}
                    ),
                    "out_degree_mean": float(median(bucket["out_degree_means"])) if bucket["out_degree_means"] else None,
                    "in_degree_mean": float(median(bucket["in_degree_means"])) if bucket["in_degree_means"] else None,
                    "in_degree_std": float(median(bucket["in_degree_stds"])) if bucket["in_degree_stds"] else None,
                    "in_degree_max": int(median(bucket["in_degree_maxs"])) if bucket["in_degree_maxs"] else None,
                }
            )
        points.sort(key=lambda item: (item["recall"], item["qps"]), reverse=True)
        return points

    def _trial_observations(
        self,
        *,
        points: Sequence[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Build a flat list of per-trial observations with all available diagnostic metrics.

        Each observation includes the full parameter set (M, efC, ef), recall / QPS,
        feasibility, and hardware-level diagnostics (build time, index size, etc.)
        so that the full signal catalog (S1–S10) can be matched.
        """
        observations: List[Dict[str, Any]] = []
        for point in points:
            params = point.get("params") if isinstance(point.get("params"), dict) else {}
            frontier = point.get("frontier") if isinstance(point.get("frontier"), list) else []
            _max_recall = None
            if frontier:
                _best = max(frontier, key=lambda row: (float(row.get("recall", 0.0)), -int(row.get("ef", 0))))
                _max_recall = float(_best.get("recall", 0.0))
            observations.append(
                {
                    "M": params.get("M"),
                    "ef_construction": params.get("ef_construction"),
                    "ef": point.get("selected_ef"),
                    "recall": float(point.get("recall", 0.0)),
                    "qps": float(point.get("qps", 0.0)),
                    "feasible": float(point.get("recall", 0.0)) >= threshold,
                    "recall_margin": float(point.get("recall", 0.0)) - threshold,
                    "max_recall": _max_recall,
                    "build_time_s": float(point.get("build_time_s", 0.0)),
                    "index_size_mb": float(point.get("index_size_mb", 0.0)),
                    "dist_comps_per_query": float(point.get("dist_comps_per_query", 0.0)),
                    "visited_nodes_per_query": float(point.get("visited_nodes_per_query", 0.0)),
                    # ── New diagnostics ──
                    "dc_vn_ratio": point.get("dc_vn_ratio"),
                    "efC_ef_ratio": point.get("efC_ef_ratio"),
                    "candidate_distance_stats": point.get("candidate_distance_stats", {}),
                    "out_degree_mean": point.get("out_degree_mean"),
                    "in_degree_mean": point.get("in_degree_mean"),
                    "in_degree_std": point.get("in_degree_std"),
                    "in_degree_max": point.get("in_degree_max"),
                }
            )
        # Sort by QPS descending so the LLM sees the best-performing configs first.
        observations.sort(key=lambda o: (o["feasible"], o["qps"]), reverse=True)
        return observations

    def _surface_uncertainty_payload(
        self,
        *,
        points: Sequence[Dict[str, Any]],
        scbo_reflection: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        reflection = scbo_reflection if isinstance(scbo_reflection, dict) else {}
        surrogate = reflection.get("surrogate") if isinstance(reflection.get("surrogate"), dict) else {}
        candidate_rows = reflection.get("candidate_posteriors") if isinstance(reflection.get("candidate_posteriors"), list) else []
        uncertain_candidates: List[Dict[str, Any]] = []
        for row in candidate_rows:
            if not isinstance(row, dict):
                continue
            critic = row.get("bo_critic") if isinstance(row.get("bo_critic"), dict) else {}
            uncertain_candidates.append(
                {
                    "node_id": row.get("node_id", ""),
                    "params": row.get("params"),
                    "sigma_G_tau": float(critic.get("sigma_G_tau", 0.0)),
                    "P_feas": float(critic.get("P_feas", 0.0)),
                    "information_gain": float(critic.get("information_gain", 0.0)),
                    "judgement": str(critic.get("judgement", "")),
                }
            )
        uncertain_candidates.sort(
            key=lambda item: (item["information_gain"], item["sigma_G_tau"], item["P_feas"]),
            reverse=True,
        )
        return {
            "surrogate_ready": bool(surrogate.get("ready", len(points) >= 3)),
            "train_points": int(surrogate.get("train_points", len(points))),
            "trust_region_length": float(reflection.get("length", 0.0)),
            "recent_patterns": reflection.get("recent_patterns", {}),
            "high_uncertainty_candidates": uncertain_candidates[:3],
            "needs_exploration": len(points) < 3 or any(item["information_gain"] >= 0.65 for item in uncertain_candidates[:3]),
        }

    def _build_only_params(self, params: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not isinstance(params, dict):
            return None
        build = {name: params[name] for name in BUILD_PARAM_ORDER if name in params}
        return build or None

    @staticmethod
    def _strip_qps_from_insights(insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove absolute QPS values from historical insight cards.

        Absolute QPS values are dataset- and hardware-specific and cannot
        be compared across tasks.  Only parameter-level information
        (regions, directions, top configs) is preserved.
        """
        stripped: List[Dict[str, Any]] = []
        for task in insights:
            if not isinstance(task, dict):
                continue
            task_copy = copy.deepcopy(task)
            # Remove task-level QPS references
            task_copy.pop("global_feasible_best", None)
            task_copy.pop("global_best_params", None)
            build_params = task_copy.get("build_params")
            if isinstance(build_params, dict):
                build_params.pop("best_qps", None)
                build_params.pop("target_qps", None)
            cards = task_copy.get("insight_cards") if isinstance(task_copy.get("insight_cards"), list) else []
            for card in cards:
                if not isinstance(card, dict):
                    continue
                # Remove absolute QPS from card-level fields
                perf = card.get("perf") if isinstance(card.get("perf"), dict) else {}
                for key in ("best_qps", "median_feasible_qps", "median_elite_qps"):
                    perf.pop(key, None)
                if perf:
                    card["perf"] = perf
                # Remove QPS from quality section
                quality = card.get("quality") if isinstance(card.get("quality"), dict) else {}
                quality.pop("score", None)
                # Remove QPS from direction section
                card.pop("qps", None)
                # Strip QPS from top_configs — only keep params
                top_cfgs = card.get("top_configs")
                if isinstance(top_cfgs, list):
                    card["top_configs"] = [
                        {k: v for k, v in cfg.items() if k != "qps"}
                        for cfg in top_cfgs
                        if isinstance(cfg, dict)
                    ]
                # Strip QPS numbers from text fields (hint, advice)
                for text_key in ("hint", "advice"):
                    text_val = str(card.get(text_key, "") or "")
                    if text_val:
                        text_val = re.sub(r'QPS\s*[~≈]?\s*[\d.]+K?', 'QPS', text_val)
                        card[text_key] = text_val
                # Recursively strip high numeric values (>10000 = likely QPS)
                HNSWLIBTuningAgent._strip_high_numbers(card)
            stripped.append(task_copy)
        return stripped

    @staticmethod
    def _strip_high_numbers(obj):
        """Remove numeric values > 10000 (likely QPS) from nested dicts/lists."""
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                val = obj[key]
                if isinstance(val, (int, float)) and val > 10000 and key not in ("recall_threshold",):
                    obj[key] = None
                elif isinstance(val, (dict, list)):
                    HNSWLIBTuningAgent._strip_high_numbers(val)
        elif isinstance(obj, list):
            for item in obj:
                HNSWLIBTuningAgent._strip_high_numbers(item)

    # ── 3-Category Classification ─────────────────────────────────────────

    # Module-level constants for classification boundaries.
    RECALL_FAR_BELOW_MARGIN: float = -0.02
    RECALL_FAR_ABOVE_MARGIN: float = 0.03
    _CLASSIFY_EPSILON: float = 1e-9

    def _near_boundary(self, threshold: float) -> float:
        """NEAR step-size boundary: adaptive to recall threshold.

        For high τ (≥0.99) use 0.001, otherwise 0.01.
        This prevents EVERYTHING being NEAR when max possible margin is tiny.

        When ``self.near_boundary_override`` is set (config key
        ``tuning.near_boundary``), that value is used instead.
        """
        if self.near_boundary_override is not None:
            return self.near_boundary_override
        return 0.001 if threshold >= 0.99 else 0.01

    def _classify_recall_status(
        self,
        last_obs: Dict[str, Any] | None,
        threshold: float,
    ) -> Tuple[str, str]:
        """Classify the current tuning state into 3 categories based on last recall.

        Returns ``(classification, strategy_description)``.

        Classification values:
          - ``"cold_start"``: no execution data available yet
          - ``"recall_far_below"``: recall < threshold + RECALL_FAR_BELOW_MARGIN
          - ``"recall_near_threshold"``: RECALL_FAR_BELOW_MARGIN <= recall < threshold + RECALL_FAR_ABOVE_MARGIN
          - ``"recall_far_above"``: recall >= threshold + RECALL_FAR_ABOVE_MARGIN
        """
        if last_obs is None or last_obs.get("recall") is None:
            return ("cold_start",
                    "No execution data yet. Choose a moderate, conservative configuration to establish a baseline.")

        recall = float(last_obs["recall"])
        margin = recall - threshold

        # Use a tiny epsilon to avoid floating-point boundary issues.
        eps = self._CLASSIFY_EPSILON

        # Adaptive thresholds: tighter for high τ
        adaptive_margin = min(self.RECALL_FAR_ABOVE_MARGIN, (1.0 - threshold) * 0.5)
        far_below = -adaptive_margin
        far_above = adaptive_margin

        if margin < far_below - eps:
            strategy = (
                f"RECALL CRITICAL (margin={margin:+.4f}). "
                f"|margin| > {abs(far_below):.3f} — FAR below threshold, aggressive repair needed. "
                "M +5~20, efC +50~250, ef +20~150. "
                "In direct mode max_recall equals recall at the selected ef "
                "(it is NOT a structural ceiling): try a modest ef increase first; "
                "increase M only if the same (M, efC) repeatedly fails to reach τ "
                "across ef values."
            )
            return ("recall_far_below", strategy)
        elif margin >= far_above - eps:
            strategy = (
                f"QPS OPTIMIZATION (margin={margin:+.4f}). "
                f"|margin| > {far_above:.3f} — FAR from threshold, aggressive steps allowed. "
                
                "Reduce parameters for QPS (ef / M / efC — ef changes are zero-rebuild, M/efC changes rebuild). "
                "Then reduce M/efC if more QPS is needed."
            )
            return ("recall_far_above", strategy)
        else:
            strategy = (
                f"BALANCED (margin={margin:+.4f}). "
                f"|margin| ≤ {far_above:.3f} — NEAR threshold, CONSERVATIVE steps ONLY. "
                
                "Single-axis moves preferred; M and ef may be adjusted together "
                "when evidence supports both. "
                "If previous M↓+ef↓ LOST QPS → try opposite direction."
            )
            return ("recall_near_threshold", strategy)

    # ── Metric Resolution Helper ──────────────────────────────────────────

    @staticmethod
    def _resolve_metric(obs: Dict[str, Any], metric_spec: str) -> Any:
        """Resolve a metric field specification into an actual value from *obs*.

        Supports three patterns:

        1. **Direct field**: ``"max_recall"`` → ``obs["max_recall"]``
        2. **Nested dot-notation**: ``"candidate_distance_stats.mean"`` →
           ``obs["candidate_distance_stats"]["mean"]``
        3. **Composite ratio**: ``"A / B"`` → numeric division if both fields are numeric;
           returns ``None`` if denominator is 0 or fields are missing.
        """
        spec = metric_spec.strip()
        if not spec:
            return None

        # Composite metric: "A / B"
        if " / " in spec:
            parts = [p.strip() for p in spec.split(" / ")]
            if len(parts) == 2:
                num = HNSWLIBTuningAgent._resolve_metric(obs, parts[0])
                den = HNSWLIBTuningAgent._resolve_metric(obs, parts[1])
                try:
                    n = float(num) if num is not None else 0.0
                    d = float(den) if den is not None else 1.0
                    if abs(d) < 1e-12:
                        return None
                    return round(n / d, 4)
                except (TypeError, ValueError):
                    return None
            return None

        # Dot-notation for nested dicts
        if "." in spec:
            keys = spec.split(".")
            val: Any = obs
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                else:
                    return None
            return val

        # Direct field lookup
        return obs.get(spec)

    # ── Root State ────────────────────────────────────────────────────────

    def build_root_state(
        self,
        *,
        round_idx: int,
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        scbo_reflection: Dict[str, Any] | None = None,
        previous_attribution: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy.get("recall_slack", 0.0))
        points = self._aggregate_success_points(stage_trials)
        feasible = [point for point in points if point["recall"] >= threshold]
        best_feasible = max(feasible, key=lambda item: (item["qps"], -item["recall"])) if feasible else None
        closest = max(points, key=lambda item: (item["recall"], item["qps"])) if points else None
        fastest = max(points, key=lambda item: (item["qps"], item["recall"])) if points else None

        trial_observations = self._trial_observations(
            points=points,
            threshold=threshold,
        )

        # ── 3-category classification based on the chronologically last trial ──
        last_success = None
        for t in reversed(list(stage_trials)):
            if t.get("status") == "success" and isinstance(t.get("metrics"), dict):
                m = t["metrics"]
                last_success = {"recall": float(m.get("recall", 0)), "qps": float(m.get("qps", 0))}
                break
        if last_success is None and trial_observations:
            last_success = trial_observations[-1]  # fallback
        classification, stage_strategy = self._classify_recall_status(last_success, threshold)

        return {
            "round": int(round_idx),
            "observed_success_count": len(points),
            "recall_threshold": threshold,
            "recall_slack": slack,
            "optimization_stage": classification,
            "classification": classification,
            "stage_strategy": stage_strategy,
            "current_best_feasible": (
                {
                    "build": best_feasible["params"],
                    "ef_star": best_feasible.get("selected_ef"),
                    "QPS": float(best_feasible["qps"]),
                    "recall_at_ef_star": float(best_feasible["recall"]),
                }
                if best_feasible is not None
                else None
            ),
            "closest_to_feasible": (
                {
                    "params": closest["params"],
                    "recall": float(closest["recall"]),
                    "qps": float(closest["qps"]),
                    "ef_star": closest.get("selected_ef"),
                }
                if closest is not None
                else None
            ),
            "fastest_observed": (
                {
                    "params": fastest["params"],
                    "recall": float(fastest["recall"]),
                    "qps": float(fastest["qps"]),
                    "ef_star": fastest.get("selected_ef"),
                }
                if fastest is not None
                else None
            ),
            "trial_observations": trial_observations,
            "previous_attribution": (
                previous_attribution if isinstance(previous_attribution, dict) else None
            ),
        }

    def activate_skills(self, root_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        stage = str(root_state.get("optimization_stage", "recall_far_below"))
        if stage == "recall_far_below":
            names = ["construction_parameter_perturbation", "best_neighborhood_exploration"]
        elif stage == "recall_near_threshold":
            names = ["best_neighborhood_exploration", "construction_parameter_perturbation"]
        elif stage == "recall_far_above":
            names = ["best_neighborhood_exploration", "construction_parameter_perturbation"]
        else:  # cold_start or unknown
            names = ["best_neighborhood_exploration", "construction_parameter_perturbation"]
        reasons = {
            "best_neighborhood_exploration": "Explore small build-space moves around the current best feasible build.",
            "construction_parameter_perturbation": "Adjust graph quality/cost parameters M and ef_construction.",
        }
        return [{"skill": name, "branch": name, "activated": True, "reason": reasons[name]} for name in names]

    def _reference_params(self, root_state: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["current_best_feasible", "closest_to_feasible", "fastest_observed"]:
            point = root_state.get(key)
            build = point.get("build") if isinstance(point, dict) and isinstance(point.get("build"), dict) else None
            if build is None and isinstance(point, dict) and isinstance(point.get("params"), dict):
                build = point.get("params")
            if isinstance(build, dict):
                try:
                    base = self.space.baseline()
                    merged = dict(base)
                    merged.update(build)
                    return self.canonicalize(merged)
                except Exception:
                    continue
        return self.space.baseline()

    def _build_key(self, params: Dict[str, Any]) -> Tuple[Any, ...]:
        return build_params_to_key(params, BUILD_PARAM_ORDER)

    def _apply_deltas(self, params: Dict[str, Any], deltas: Dict[str, int]) -> Dict[str, Any] | None:
        next_params = dict(params)
        for name, delta in deltas.items():
            next_value = self.space.value_at_delta(name, next_params[name], delta)
            if next_value is None:
                return None
            next_params[name] = next_value
        return next_params

    def _space_apply_deltas(
        self,
        space: ParameterSpace,
        params: Dict[str, Any],
        deltas: Dict[str, int],
    ) -> Dict[str, Any] | None:
        next_params = dict(params)
        for name, delta in deltas.items():
            next_value = space.value_at_delta(name, next_params[name], delta)
            if next_value is None:
                return None
            next_params[name] = next_value
        return next_params

    def _initial_design_seed_delta_sets(self, recall_threshold: float) -> List[Dict[str, int]]:
        base: List[Dict[str, int]]
        if recall_threshold <= 0.88:
            base = [
                {"M": 1},
                {"M": -1},
                {"ef_construction": -1},
                {"ef_construction": 1},
                {"ef_construction": -2},
                {"M": 1, "ef_construction": -1},
                {"M": -1, "ef_construction": 1},
            ]
        elif recall_threshold <= 0.92:
            base = [
                {"ef_construction": 1},
                {"M": 1},
                {"ef_construction": -1},
                {"M": -1},
                {"M": 1, "ef_construction": 1},
                {"M": -1, "ef_construction": -1},
            ]
        else:
            base = [
                {"ef_construction": 1},
                {"M": 1},
                {"ef_construction": 2},
                {"M": 1, "ef_construction": 1},
                {"M": -1, "ef_construction": -1},
            ]
        # When ef is a directly-tuned parameter, add ef deltas.
        if "ef" in BUILD_PARAM_ORDER:
            base.extend([{"ef": 2}, {"ef": -2}, {"ef": 4}, {"ef": -4}])
        return base

    def _fallback_delta_sets(
        self,
        skill_name: str,
        root_state: Dict[str, Any],
        node_state: str | None = None,
    ) -> List[Dict[str, int]]:
        stage = str(root_state.get("optimization_stage", "cold_start"))
        ef_direct = "ef" in BUILD_PARAM_ORDER
        if node_state == "recall_too_high":
            # Margin-proportional: small/medium/large ef reductions
            deltas: List[Dict[str, int]] = [{"M": -1}, {"ef_construction": -1}, {"M": -1, "ef_construction": -1}]
            if ef_direct:
                deltas.extend([{"ef": -2}, {"ef": -4}, {"ef": -8}, {"M": -1, "ef": -2}, {"M": -1, "ef": -4}])
            return deltas
        if node_state == "feasible_near_boundary":
            deltas = [{"M": -1}, {"ef_construction": -1}, {"M": 1}]
            if ef_direct:
                deltas.extend([{"ef": -1}, {"ef": 1}, {"ef": -2}])
            return deltas
        if node_state == "infeasible_near_threshold":
            # Small steps for near-threshold, bigger for larger gaps
            deltas = [{"M": 1}, {"ef_construction": 1}, {"M": 1, "ef_construction": 1}]
            if ef_direct:
                deltas.extend([{"ef": 2}, {"ef": 4}, {"ef": 8}, {"M": 1, "ef": 2}, {"M": 1, "ef": 4}])
            return deltas
        if node_state == "high_uncertainty":
            deltas = [{"M": 1}, {"M": -1}, {"ef_construction": 1}, {"ef_construction": -1}]
            if ef_direct:
                deltas.extend([{"ef": 2}, {"ef": -2}])
            return deltas
            return [{"M": 1}, {"M": -1}, {"ef_construction": 1}, {"ef_construction": -1}]
        if skill_name == "best_neighborhood_exploration":
            return [
                {"M": -1},
                {"ef_construction": -1},
                {"M": -1, "ef_construction": -1},
                {"M": 1},
                {"ef_construction": 1},
            ]
        if skill_name == "construction_parameter_perturbation":
            return [
                {"M": -1},
                {"ef_construction": -1},
                {"M": -1, "ef_construction": -1},
                {"M": 1},
                {"ef_construction": 1},
                {"M": 1, "ef_construction": 1},
            ]
        if stage in {"cold_start", "no_feasible_point", "infeasible_near_threshold"}:
            return [{"M": 1}, {"ef_construction": 1}, {"M": 1, "ef_construction": 1}]
        return [{"M": -1}, {"ef_construction": -1}, {"M": 1}, {"ef_construction": 1}]

    @staticmethod
    def _check_step_compliance(
        last_params: Dict[str, int],
        proposed_params: Dict[str, int],
        margin: float,
        near_boundary: float = 0.01,
        force_construction: bool = False,
        excluded_construction_pairs: Optional[set] = None,
    ) -> Tuple[bool, List[str]]:
        """Check if proposed step sizes comply with tier requirements.

        Returns (compliant, violations).

        - NEAR (|margin| ≤ near_boundary): steps must be small (M ≤2, ef ≤5, efC ≤30).
          M and ef may be adjusted together within these caps.
        - FAR (|margin| > near_boundary): at least one param must reach minimum step in
          the CORRECT direction. margin>0 → reduce params for QPS; margin<0 → increase for repair.
        """
        dm = proposed_params["M"] - last_params["M"]          # signed delta
        de = proposed_params["ef"] - last_params["ef"]         # signed delta
        dec = proposed_params["ef_construction"] - last_params["ef_construction"]  # signed delta
        adm = abs(dm); ade = abs(de); adec = abs(dec)
        violations: List[str] = []

        if force_construction and (dm == 0 or dec == 0):
            violations.append(
                "construction-probe round: must change BOTH M and efC from the last execution"
            )
        if excluded_construction_pairs is not None:
            pair = (proposed_params["M"], proposed_params["ef_construction"])
            if pair in excluded_construction_pairs:
                violations.append(
                    f"(M, efC) pair {pair} is BANNED (explored ≥3 times) — "
                    "propose a different construction pair"
                )

        # Direction sanity + ef step cap (M/efC are unrestricted).
        if abs(margin) <= near_boundary:
            if ade > 10:
                violations.append(f"ef step {ade} exceeds NEAR max 10")
        elif ade > 150:
            violations.append(f"ef step {ade} exceeds FAR max 150")

        if adm == 0 and ade == 0 and adec == 0:
            violations.append("proposal must change at least one parameter")
        elif margin > 0:
            if dm > 0 and de > 0 and dec > 0:
                violations.append(
                    "margin>0 (QPS optimization): ALL params increased, at least one should decrease. "
                    f"ΔM={dm:+d}, ΔefC={dec:+d}, Δef={de:+d}."
                )
        else:
            if dm < 0 and de < 0 and dec < 0:
                violations.append(
                    "margin<0 (recall repair): ALL params decreased, at least one should increase. "
                    f"ΔM={dm:+d}, ΔefC={dec:+d}, Δef={de:+d}."
                )

        return len(violations) == 0, violations

    def _fallback_skill_candidates(
        self,
        skill_name: str,
        root_state: Dict[str, Any],
        target_count: int,
        reference_params: Dict[str, Any] | None = None,
        node_state: str | None = None,
        expansion_intent: str | None = None,
    ) -> List[Dict[str, Any]]:
        reference = reference_params or self._reference_params(root_state)
        candidates: List[Dict[str, Any]] = []
        for deltas in self._fallback_delta_sets(skill_name, root_state, node_state=node_state):
            params = self._apply_deltas(reference, deltas)
            if params is None or params == reference:
                continue
            candidates.append(
                {
                    "params": params,
                    "thinking": (
                        f"{skill_name} fallback applies deltas {deltas} from reference"
                        f" for node_state={node_state or root_state.get('optimization_stage')}."
                    ),
                    "expected_behavior": {
                        "recall": "move according to HNSW recall/QPS tradeoff",
                        "qps": "seek better throughput under recall guardrail",
                        "expansion_intent": expansion_intent or "",
                    },
                }
            )
            if len(candidates) >= target_count:
                return candidates

        seen = {self._build_key(item["params"]) for item in candidates}
        attempts = max(100, target_count * 40)
        while len(candidates) < target_count and attempts > 0:
            attempts -= 1
            params = self.space.sample_random_params(self.rng)
            key = self._build_key(params)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "params": params,
                    "thinking": f"{skill_name} random fill inside HNSW parameter space.",
                    "expected_behavior": {"recall": "unknown", "qps": "unknown"},
                }
            )
        return candidates

    def initial_design_candidates(
        self,
        target_unique: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        *,
        stage_policy: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
        seed_candidates: Sequence[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        target_unique = max(0, int(target_unique))
        if target_unique <= 0:
            return []
        search_space = self.space
        if allowed_values_override is not None:
            search_space = ParameterSpace.from_config(allowed_values_override, order=PARAM_ORDER)

        anchors: List[Dict[str, Any]] = []
        baseline = search_space.baseline()
        recall_threshold = 1.0
        if isinstance(stage_policy, dict):
            try:
                recall_threshold = float(stage_policy.get("recall_threshold", 1.0))
            except Exception:
                recall_threshold = 1.0
        ef_placeholder = baseline["ef"]
        # When ef is directly tuned, cycle through low/mid/high values
        # so the initial design covers the ef dimension.
        _ef_values: List[Any] = [ef_placeholder]
        if "ef" in BUILD_PARAM_ORDER:
            ef_domain = self.space.domains.get("ef")
            if ef_domain is not None:
                _ef_low = ef_domain.edge("low")
                _ef_high = ef_domain.edge("high")
                _ef_values = [_ef_low, ef_placeholder, _ef_high, ef_placeholder]
        _ef_cycle = itertools.cycle(_ef_values)

        if recall_threshold <= 0.88:
            corners = [
                {
                    "M": search_space.edge_value("M", "low"),
                    "ef_construction": search_space.edge_value("ef_construction", "low"),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.edge_value("M", "low"),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.18),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.value_from_position("M", 0.10),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.24),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.value_from_position("M", 0.14),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.30),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.value_from_position("M", 0.18),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.35),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.value_from_position("M", 0.24),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.45),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.value_from_position("M", 0.30),
                    "ef_construction": search_space.value_from_position("ef_construction", 0.20),
                    "ef": next(_ef_cycle),
                },
            ]
            corners.extend(
                [
                    {
                        "M": search_space.edge_value("M", "low"),
                        "ef_construction": search_space.value_from_position("ef_construction", 0.2),
                        "ef": next(_ef_cycle),
                    },
                    {
                        "M": search_space.value_from_position("M", 0.1),
                        "ef_construction": search_space.value_from_position("ef_construction", 0.25),
                        "ef": next(_ef_cycle),
                    },
                    {
                        "M": search_space.value_from_position("M", 0.15),
                        "ef_construction": search_space.value_from_position("ef_construction", 0.35),
                        "ef": next(_ef_cycle),
                    },
                ]
            )
        else:
            corners = [
                baseline,
                {name: search_space.edge_value(name, "low") for name in PARAM_ORDER},
                {name: search_space.edge_value(name, "high") for name in PARAM_ORDER},
                {
                    "M": search_space.edge_value("M", "low"),
                    "ef_construction": search_space.edge_value("ef_construction", "high"),
                    "ef": next(_ef_cycle),
                },
                {
                    "M": search_space.edge_value("M", "high"),
                    "ef_construction": search_space.edge_value("ef_construction", "low"),
                    "ef": next(_ef_cycle),
                },
            ]
            corners.extend(
                [
                    {
                        "M": search_space.value_from_position("M", 0.25),
                        "ef_construction": search_space.value_from_position("ef_construction", 0.65),
                        "ef": next(_ef_cycle),
                    },
                    {
                        "M": search_space.value_from_position("M", 0.5),
                        "ef_construction": search_space.value_from_position("ef_construction", 0.75),
                        "ef": next(_ef_cycle),
                    },
                ]
            )
        seen = set(exclude_param_keys)
        seed_rows = list(seed_candidates or [])
        accepted_seed_rows: List[Dict[str, Any]] = []
        neighborhood_seed_rows: List[Dict[str, Any]] = []
        for seed in seed_rows:
            params = seed.get("params") if isinstance(seed, dict) else None
            if not isinstance(params, dict):
                continue
            try:
                canonical = self.canonicalize(params, domain_constraints=allowed_values_override)
            except Exception:
                continue
            key = self._build_key(canonical)
            neighborhood_seed = {
                "params": canonical,
                "source": str(seed.get("source", "initial_design_seed")),
                "note": str(seed.get("note", "HNSW transfer/local focus seed")),
            }
            neighborhood_seed_rows.append(neighborhood_seed)
            if key in seen:
                continue
            seen.add(key)
            accepted_seed = dict(neighborhood_seed)
            anchors.append(accepted_seed)
            accepted_seed_rows.append(accepted_seed)
            if len(anchors) >= target_unique:
                return anchors
        for seed in neighborhood_seed_rows:
            for deltas in self._initial_design_seed_delta_sets(recall_threshold):
                params = self._space_apply_deltas(search_space, seed["params"], deltas)
                if params is None:
                    continue
                canonical = self.canonicalize(params, domain_constraints=allowed_values_override)
                key = self._build_key(canonical)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(
                    {
                        "params": canonical,
                        "source": "initial_design_seed_neighborhood",
                        "note": f"local neighborhood around {seed['source']}",
                    }
                )
                if len(anchors) >= target_unique:
                    return anchors
        for params in corners:
            canonical = self.canonicalize(params, domain_constraints=allowed_values_override)
            key = self._build_key(canonical)
            if key in seen:
                continue
            seen.add(key)
            anchors.append({"params": canonical, "source": "initial_design", "note": "HNSW coverage anchor"})
            if len(anchors) >= target_unique:
                return anchors
        attempts = max(200, target_unique * 80)
        while len(anchors) < target_unique and attempts > 0:
            attempts -= 1
            params = search_space.sample_random_params(self.rng)
            key = self._build_key(params)
            if key in seen:
                continue
            seen.add(key)
            anchors.append({"params": params, "source": "initial_design", "note": "HNSW initial random fill"})
        return anchors

    def _root_candidates_per_skill(self, target_nodes: int, skill_count: int, root_state: Dict[str, Any]) -> int:
        if skill_count <= 0:
            return 1
        stage = str(root_state.get("optimization_stage", "cold_start"))
        base = max(1, int(math.ceil(max(1, target_nodes) / max(1, skill_count * 2))))
        if (
            stage in {"feasible_near_boundary", "recall_too_high_qps_low", "feasible_balanced"}
            and target_nodes >= skill_count * 2 + 1
        ):
            base = max(base, 2)
        return max(1, min(3, base))

    def _raw_candidates_from_skill_payload(self, payload: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        raw = payload.get("candidates", [])
        return raw if isinstance(raw, list) else []

    def _sanitize_skill_candidates(
        self,
        *,
        skill_name: str,
        raw_candidates: Sequence[Dict[str, Any]],
        seen_keys: set[Tuple[Any, ...]],
        rejected: List[Dict[str, Any]],
        allowed_values_override: Dict[str, Any] | None,
        source: str,
    ) -> List[Dict[str, Any]]:
        accepted: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_candidates, start=1):
            if not isinstance(raw, dict):
                rejected.append({"skill": skill_name, "reason": "candidate_not_object", "candidate": raw})
                continue
            if isinstance(raw.get("params"), dict):
                params = dict(raw.get("params") or {})
                if isinstance(raw.get("build"), dict):
                    params.update(raw.get("build") or {})
            elif isinstance(raw.get("build"), dict):
                params = dict(raw.get("build") or {})
                if "ef" not in params:
                    params["ef"] = raw.get("ef", raw.get("ef_placeholder", self.space.baseline()["ef"]))
            else:
                params = raw
            try:
                canonical = self.canonicalize(params)
                if allowed_values_override:
                    fields = self.space.out_of_constraint_fields(canonical, allowed_values_override)
                    if fields:
                        rejected.append(
                            {
                                "skill": skill_name,
                                "reason": "out_of_allowed_override",
                                "params": canonical,
                                "fields": fields,
                            }
                        )
                        continue
            except Exception as exc:
                rejected.append({"skill": skill_name, "reason": "invalid", "params": params, "error": str(exc)})
                continue
            key = self._build_key(canonical)
            if key in seen_keys:
                rejected.append({"skill": skill_name, "reason": "duplicate", "params": canonical})
                continue
            seen_keys.add(key)
            node_id = str(raw.get("node_id") or f"{skill_name[:3].upper()}-{idx:03d}")
            accepted.append(
                {
                    "node_id": node_id,
                    "branch": str(raw.get("branch") or skill_name),
                    "source_skill": skill_name,
                    "params": canonical,
                    "thinking": str(raw.get("thinking") or raw.get("note") or "").strip(),
                    "expected_behavior": raw.get("expected_behavior") if isinstance(raw.get("expected_behavior"), dict) else {},
                    "source": source,
                }
            )
        return accepted

    def _action_stats_summary(self, scored_node: Dict[str, Any]) -> Dict[str, Any]:
        action = scored_node.get("scbo_action")
        if isinstance(action, dict):
            return dict(action)
        bo_critic = scored_node.get("bo_critic") if isinstance(scored_node.get("bo_critic"), dict) else {}
        objective = scored_node.get("objective", {}).get("posterior", {})
        constraint = {}
        constraints = scored_node.get("constraints") or []
        if constraints and isinstance(constraints[0], dict):
            constraint = constraints[0].get("posterior", {})
        return {
            "qps_mean": float(objective.get("mean", 0.0)),
            "qps_std": float(objective.get("std", 0.0)),
            "recall_constraint_mean": float(constraint.get("mean", 0.0)),
            "recall_constraint_std": float(constraint.get("std", 0.0)),
            "feasible_probability": float(scored_node.get("joint_feasible_prob", 0.0)),
            "qps_upper_bound": float(scored_node.get("qps_upper_bound", 0.0)),
            "score": float(scored_node.get("score", 0.0)),
            "accepted": bool(scored_node.get("accepted", True)),
            "filter_reason": str(scored_node.get("filter_reason", "") or ""),
            "mu_G_tau": float(bo_critic.get("mu_G_tau", objective.get("mean", 0.0))),
            "sigma_G_tau": float(bo_critic.get("sigma_G_tau", objective.get("std", 0.0))),
            "P_feas": float(bo_critic.get("P_feas", scored_node.get("joint_feasible_prob", 0.0))),
            "P_improve_over_best": float(bo_critic.get("P_improve_over_best", 0.0)),
            "information_gain": float(bo_critic.get("information_gain", 0.0)),
            "judgement": str(bo_critic.get("judgement", "")),
            "judgement_reason": str(bo_critic.get("judgement_reason", "")),
            "dominance_status": str(bo_critic.get("dominance_status", "")),
            "trust_region_membership": str(bo_critic.get("trust_region_membership", "unknown")),
        }

    def _candidate_summary(self, node: Dict[str, Any]) -> Dict[str, Any]:
        action_stats = self._action_stats_summary(node)
        return {
            "node_id": node.get("node_id", ""),
            "parent_id": node.get("parent_id"),
            "depth": int(node.get("depth", 1)),
            "build": self._build_only_params(node.get("params")),
            "params": node.get("params"),
            "source": node.get("source", "skill"),
            "source_skill": node.get("source_skill", ""),
            "branch": node.get("branch", ""),
            "thinking": node.get("thinking", ""),
            "expected_behavior": node.get("expected_behavior", {}),
            "action_stats": action_stats,
            "score": float(node.get("score", action_stats.get("score", 0.0))),
            "joint_feasible_prob": float(node.get("joint_feasible_prob", action_stats.get("feasible_probability", 0.0))),
            "accepted": bool(node.get("accepted", action_stats.get("accepted", True))),
            "filter_reason": str(node.get("filter_reason", action_stats.get("filter_reason", "")) or ""),
            "objective": node.get("objective", {}),
            "constraints": node.get("constraints", []),
            "bo_critic": node.get("bo_critic", {}),
            "observation": node.get("observation", {}),
            "children": list(node.get("children", [])),
        }

    def _tree_search_cfg(self, target_nodes: int, tree_search_cfg: Dict[str, Any] | None) -> Dict[str, int]:
        raw = dict(self.agentic_cfg.get("tree_search") or {})
        raw.update(tree_search_cfg or {})
        max_depth = int(raw.get("max_depth", 2))
        max_total_nodes = int(raw.get("max_total_nodes", target_nodes))
        max_children_per_node = int(raw.get("max_children_per_node", 1))
        return {
            "max_depth": max(1, max_depth),
            "max_total_nodes": max(1, max_total_nodes),
            "max_children_per_node": max(1, max_children_per_node),
        }

    def _prepare_skill_payload(
        self,
        *,
        skill_name: str,
        round_idx: int,
        target_count: int,
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        branch_intent: str,
        parent_node: Dict[str, Any] | None,
        node_state: str | None,
        expansion_intent: str | None,
        scbo_reflection: Dict[str, Any] | None,
        knowledge_context: Dict[str, Any] | str | None,
        similar_task_context: Dict[str, Any] | None,
        allowed_values_override: Dict[str, Any] | None,
        search_space_refinement: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        resolved_space = (
            self.space.normalize_constraints(allowed_values_override)
            if allowed_values_override is not None
            else self.space.export_parameter_space()
        )
        return {
            "round_idx": round_idx,
            "target_count": target_count,
            "recall_threshold": stage_policy["recall_threshold"],
            "recall_slack": stage_policy.get("recall_slack", 0.0),
            "allowed_parameter_space": resolved_space,
            "allowed_build_parameter_space": {
                name: spec for name, spec in resolved_space.items() if name in BUILD_PARAM_ORDER
            },
            "build_param_order": list(BUILD_PARAM_ORDER),
            "ef_role": {
                "tuning_role": "directly_tuned" if "ef" in BUILD_PARAM_ORDER else "frontier_scan_only",
                "selection_rule": (
                    "ef is a tuned parameter — propose a specific integer value within the allowed range"
                    if "ef" in BUILD_PARAM_ORDER
                    else "ef_star is chosen after execution from the scanned frontier under the recall policy"
                ),
                "candidate_schema_note": (
                    "ef must be an explicit integer value in every candidate"
                    if "ef" in BUILD_PARAM_ORDER
                    else "candidate params may still carry a placeholder ef for compatibility"
                ),
            },
            "root_state": root_state,
            "branch": skill_name,
            "branch_intent": branch_intent,
            "tree_context": {
                "phase": "root_path_thinking" if parent_node is None else "node_expansion_thinking",
                "parent_node": self._candidate_summary(parent_node) if parent_node is not None else None,
                "parent_node_state": node_state,
                "expansion_intent": expansion_intent,
                "thinking_action_observation_protocol": [
                    "Thinking: this Skill generates exactly HNSW configs for the current path/state.",
                    "Action: BO/SCBO scores the generated config immediately after generation.",
                    "Observation: the HNSW agent classifies the scored node and selects any next Skill.",
                ],
            },
            "allowed_adjustments": {
                "best_neighborhood_exploration": ["small local build moves around a feasible reference build"],
                "construction_parameter_perturbation": ["M", "ef_construction", "modest coupled build moves"],
                "search_parameter_adjustment": ["legacy branch name; propose only small build moves in M/ef_construction"],
            }.get(skill_name, []),
            "scbo_reflection": scbo_reflection or {},
            "knowledge_context": knowledge_context or {},
            "similar_task_context": similar_task_context or {},
            "search_space_refinement": search_space_refinement or {},
            "output_schema": {
                "skill_name": skill_name,
                "branch": skill_name,
                "candidates": [
                    {
                        "node_id": "...",
                        "build": {name: "..." for name in BUILD_PARAM_ORDER},
                        "params": {name: "..." for name in PARAM_ORDER},
                        "thinking": "state-aware branch-specific reasoning",
                        "expected_behavior": {"recall": "...", "qps": "..."},
                    }
                ],
            },
        }

    def _generate_tree_candidates(
        self,
        *,
        skill_name: str,
        round_idx: int,
        target_count: int,
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        branch_intent: str,
        parent_node: Dict[str, Any] | None,
        node_state: str | None,
        expansion_intent: str | None,
        seen_keys: set[Tuple[Any, ...]],
        rejected: List[Dict[str, Any]],
        allowed_values_override: Dict[str, Any] | None,
        scbo_reflection: Dict[str, Any] | None,
        knowledge_context: Dict[str, Any] | str | None,
        similar_task_context: Dict[str, Any] | None,
        search_space_refinement: Dict[str, Any] | None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        payload = self._prepare_skill_payload(
            skill_name=skill_name,
            round_idx=round_idx,
            target_count=target_count,
            root_state=root_state,
            stage_policy=stage_policy,
            branch_intent=branch_intent,
            parent_node=parent_node,
            node_state=node_state,
            expansion_intent=expansion_intent,
            scbo_reflection=scbo_reflection,
            knowledge_context=knowledge_context,
            similar_task_context=similar_task_context,
            allowed_values_override=allowed_values_override,
            search_space_refinement=search_space_refinement,
        )
        skill_result = None
        accepted: List[Dict[str, Any]] = []
        if self.enable_agentic:
            skill_result = self._call_skill(skill_name, payload)
            accepted = self._sanitize_skill_candidates(
                skill_name=skill_name,
                raw_candidates=self._raw_candidates_from_skill_payload(skill_result.get("parsed")),
                seen_keys=seen_keys,
                rejected=rejected,
                allowed_values_override=allowed_values_override,
                source="skill_llm",
            )

        if len(accepted) < target_count:
            raise RuntimeError(
                f"LLM skill '{skill_name}' produced {len(accepted)}/{target_count} valid candidates "
                "after sanitization — rule-based fallback is disabled."
            )

        skill_log = {
            "skill_name": skill_name,
            "branch": skill_name,
            "parent_node_id": parent_node.get("node_id") if isinstance(parent_node, dict) else None,
            "parent_node_state": node_state,
            "expansion_intent": expansion_intent,
            "llm_ok": bool(skill_result and skill_result.get("ok")),
            "llm_error": skill_result.get("error") if skill_result else "agentic_disabled",
            "accepted_count": len(accepted),
            "candidates": accepted,
        }
        return accepted, skill_log

    def run_tree_search(
        self,
        *,
        round_idx: int,
        target_nodes: int,
        execution_count: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        scbo_optimizer: Any,
        observations: Sequence[Dict[str, Any]],
        prepared_round: Dict[str, Any] | None = None,
        scbo_reflection: Dict[str, Any] | None = None,
        knowledge_context: Dict[str, Any] | None = None,
        similar_task_context: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
        search_space_refinement: Dict[str, Any] | None = None,
        tree_search_cfg: Dict[str, Any] | None = None,
        previous_attribution: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        root_state = self.build_root_state(
            round_idx=round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
            scbo_reflection=scbo_reflection,
            previous_attribution=previous_attribution,
        )
        activated = self.activate_skills(root_state)
        cfg = self._tree_search_cfg(target_nodes, tree_search_cfg)
        max_depth = cfg["max_depth"]
        max_total_nodes = cfg["max_total_nodes"]
        max_children_per_node = cfg["max_children_per_node"]
        seen_keys = set(exclude_param_keys)
        rejected: List[Dict[str, Any]] = []
        skill_outputs: List[Dict[str, Any]] = []
        node_observations: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        terminal_node_ids: set[str] = set()
        frontier: List[Dict[str, Any]] = []
        node_seq = 0
        root_candidates_per_skill = self._root_candidates_per_skill(target_nodes, len(activated), root_state)

        def materialize_node(raw_node: Dict[str, Any], parent: Dict[str, Any] | None, depth: int) -> Dict[str, Any]:
            nonlocal node_seq
            node_seq += 1
            original_node_id = str(raw_node.get("node_id") or "")
            node_id = f"HTS-{node_seq:03d}"
            candidate = {
                **raw_node,
                "node_id": node_id,
                "raw_node_id": original_node_id,
                "parent_id": parent.get("node_id") if isinstance(parent, dict) else None,
                "depth": int(depth),
                "children": [],
            }
            scored = scbo_optimizer.score_single_node(
                candidate_node=candidate,
                observations=observations,
                prepared_round=prepared_round,
            )
            scored["action_stats"] = self._action_stats_summary(scored)
            observation, observation_log = self._observe_node(
                node=scored,
                root_state=root_state,
                stage_policy=stage_policy,
                max_depth=max_depth,
                scbo_reflection=scbo_reflection,
            )
            scored["observation"] = observation
            scored["children"] = []
            nodes.append(scored)
            node_observations.append(observation_log)
            if parent is not None:
                parent.setdefault("children", []).append(node_id)
                edges.append(
                    {
                        "parent_id": parent.get("node_id"),
                        "child_id": node_id,
                        "skill": scored.get("source_skill", ""),
                        "node_state": observation.get("node_state", ""),
                    }
                )
            return scored

        for entry in activated:
            if len(nodes) >= max_total_nodes:
                break
            skill_name = entry["skill"]
            raw_candidates, skill_log = self._generate_tree_candidates(
                skill_name=skill_name,
                round_idx=round_idx,
                target_count=min(root_candidates_per_skill, max_total_nodes - len(nodes)),
                root_state=root_state,
                stage_policy=stage_policy,
                branch_intent=entry["reason"],
                parent_node=None,
                node_state=None,
                expansion_intent=entry["reason"],
                seen_keys=seen_keys,
                rejected=rejected,
                allowed_values_override=allowed_values_override,
                scbo_reflection=scbo_reflection,
                knowledge_context=knowledge_context,
                similar_task_context=similar_task_context,
                search_space_refinement=search_space_refinement,
            )
            skill_outputs.append(skill_log)
            for raw_node in raw_candidates[: min(root_candidates_per_skill, max_total_nodes - len(nodes))]:
                node = materialize_node(raw_node, parent=None, depth=1)
                observation = node.get("observation", {})
                if bool(observation.get("expand")) and int(node.get("depth", 1)) < max_depth:
                    frontier.append(node)
                else:
                    terminal_node_ids.add(str(node.get("node_id", "")))

        while frontier and len(nodes) < max_total_nodes:
            frontier.sort(
                key=lambda item: (
                    float(item.get("observation", {}).get("priority_score", 0.0)),
                    float(item.get("score", 0.0)),
                    float(item.get("joint_feasible_prob", 0.0)),
                ),
                reverse=True,
            )
            parent = frontier.pop(0)
            parent_obs = parent.get("observation", {})
            parent_depth = int(parent.get("depth", 1))
            if parent_depth >= max_depth:
                terminal_node_ids.add(str(parent.get("node_id", "")))
                continue
            next_skill = parent_obs.get("next_skill")
            if next_skill not in SKILL_NAMES:
                terminal_node_ids.add(str(parent.get("node_id", "")))
                continue

            child_count = min(max_children_per_node, max_total_nodes - len(nodes))
            raw_candidates, skill_log = self._generate_tree_candidates(
                skill_name=str(next_skill),
                round_idx=round_idx,
                target_count=child_count,
                root_state=root_state,
                stage_policy=stage_policy,
                branch_intent=str(parent_obs.get("expansion_intent", "")),
                parent_node=parent,
                node_state=str(parent_obs.get("node_state", "")),
                expansion_intent=str(parent_obs.get("expansion_intent", "")),
                seen_keys=seen_keys,
                rejected=rejected,
                allowed_values_override=allowed_values_override,
                scbo_reflection=scbo_reflection,
                knowledge_context=knowledge_context,
                similar_task_context=similar_task_context,
                search_space_refinement=search_space_refinement,
            )
            skill_outputs.append(skill_log)
            if not raw_candidates:
                parent["terminal_reason"] = "expansion_produced_no_valid_child"
                terminal_node_ids.add(str(parent.get("node_id", "")))
                continue

            for raw_node in raw_candidates[:child_count]:
                child = materialize_node(raw_node, parent=parent, depth=parent_depth + 1)
                child_obs = child.get("observation", {})
                if bool(child_obs.get("expand")) and int(child.get("depth", 1)) < max_depth:
                    frontier.append(child)
                else:
                    terminal_node_ids.add(str(child.get("node_id", "")))

        for node in frontier:
            node["terminal_reason"] = "tree_search_budget_exhausted"
            terminal_node_ids.add(str(node.get("node_id", "")))

        terminal_nodes = [node for node in nodes if str(node.get("node_id", "")) in terminal_node_ids]
        selection_pool = terminal_nodes or nodes
        candidates, selection_log = self.select_executions(
            selection_pool,
            count=execution_count,
            root_state=root_state,
            scbo_reflection=scbo_reflection,
        )
        scored_summaries = [self._candidate_summary(node) for node in nodes]
        filtered_summaries = [summary for summary in scored_summaries if not bool(summary.get("accepted", True))]
        terminal_summaries = [self._candidate_summary(node) for node in terminal_nodes]
        tree_log = {
            "root_state": root_state,
            "activated_skills": activated,
            "skill_outputs": skill_outputs,
            "rejected_candidate_nodes": rejected,
            "tree_search": {
                "max_depth": max_depth,
                "max_total_nodes": max_total_nodes,
                "max_children_per_node": max_children_per_node,
                "nodes": scored_summaries,
                "edges": edges,
            },
            "node_observations": node_observations,
            "terminal_nodes": terminal_summaries,
            "scored_candidate_nodes": scored_summaries,
            "filtered_candidate_nodes": filtered_summaries,
            **selection_log,
        }
        return candidates, tree_log

    def run_diagnostic_round(
        self,
        *,
        round_idx: int,
        target_count: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        observations: Sequence[Dict[str, Any]],
        scbo_reflection: Dict[str, Any] | None = None,
        prepared_round: Dict[str, Any] | None = None,
        knowledge_context: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
        search_space_refinement: Dict[str, Any] | None = None,
        previous_attribution: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run one round of frontier-surface reasoning (methods.md Steps 1–3, LLM.md §6).

        Two-phase LLM interaction with BO critic in between:

        1. Phase 1 — LLM diagnoses the frontier surface, selects a mechanism,
           and proposes a target construction with a falsifiable prediction.
        2. BO critic — evaluates the target construction via GP posterior,
           returning μ_G, σ_G, P_feas, P_improve, IG, and judgement.
        3. Phase 2 — LLM reconciles its hypothesis with the BO verdict and
           decides to commit / revise / skip.

        Returns a list with at most one candidate and a structured log dict.
        """
        # --- extract GP models from prepared_round ---
        obj_model = None
        con_model = None
        model_used = False
        if isinstance(prepared_round, dict):
            ctx = prepared_round.get("context")
            if isinstance(ctx, dict):
                obj_model = ctx.get("obj_model")
                con_model = ctx.get("con_model")
                model_used = bool(ctx.get("model_used", False)) and obj_model is not None and con_model is not None

        target_count = max(0, int(target_count))
        root_state = self.build_root_state(
            round_idx=round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
            scbo_reflection=scbo_reflection,
            previous_attribution=previous_attribution,
        )

        # Call Phase 1+2 with retry on duplicates
        max_retries = 2
        diagnostic_result: Dict[str, Any] = {}
        for retry_idx in range(max_retries + 1):
            diagnostic_result = self._run_diagnostic_reasoning(
                root_state=root_state,
                scbo_reflection=scbo_reflection,
                stage_policy=stage_policy,
                knowledge_context=knowledge_context,
                allowed_values_override=allowed_values_override,
                observations=observations,
                obj_model=obj_model,
                con_model=con_model,
                model_used=model_used,
                prepared_round=prepared_round,
            )

            if diagnostic_result.get("error") != "duplicate_construction":
                break  # success or non-duplicate error

            logger.info(
                "Frontier reasoning proposed duplicate — retry %d/%d.",
                retry_idx + 1, max_retries,
            )
            # Refresh root_state so LLM sees latest observed constructions
            root_state = self.build_root_state(
                round_idx=round_idx,
                stage_trials=stage_trials,
                stage_policy=stage_policy,
                scbo_reflection=scbo_reflection,
                previous_attribution=previous_attribution,
            )

        # All retries exhausted on duplicates — fall back to initial design
        if diagnostic_result.get("error") == "duplicate_construction":
            logger.info("All frontier retries exhausted (duplicate). Falling back to initial design.")
            diagnostic_result = {
                "ok": False,
                "error": "all_retries_duplicate",
                "canonical_build": None,
                "final_decision": {"decision": "skip"},
            }

        candidates: List[Dict[str, Any]] = []
        canonical_build = diagnostic_result.get("canonical_build")
        if isinstance(canonical_build, dict) and canonical_build:
            build_params = {name: canonical_build[name] for name in PARAM_ORDER}
            key = build_params_to_key(build_params, BUILD_PARAM_ORDER)
            if key not in exclude_param_keys:
                candidates.append(
                    {
                        "params": canonical_build,
                        "source": "frontier_reasoning",
                        "note": str(diagnostic_result.get("selected_mechanism", {}).get("name", "")),
                        "node_id": "FR-001",
                    }
                )

        # Fallback: if frontier reasoning produced no valid build, use
        # initial-design candidates to keep the round productive.
        if not candidates:
            fallback_design = self.initial_design_candidates(
                max(1, target_count),
                exclude_param_keys,
                stage_policy=stage_policy,
                allowed_values_override=allowed_values_override,
            )
            candidates = fallback_design

        frontier_log = {
            "root_state": root_state,
            "activated_skills": [],
            "skill_outputs": [],
            "rejected_candidate_nodes": [],
            "tree_search": {"nodes": [], "edges": []},
            "node_observations": [],
            "terminal_nodes": [],
            "scored_candidate_nodes": [],
            "filtered_candidate_nodes": [],
            "selected_execution": candidates[0] if candidates else None,
            "selected_executions": candidates[:1],
            "reasoning_mode": "frontier",
            "llm_reasoning": {
                "mode": "frontier",
                "executed_mode": "frontier",
                "llm_ok": bool(diagnostic_result.get("ok")),
                "llm_error": diagnostic_result.get("error", ""),
                "diagnosis": diagnostic_result.get("diagnosis", {}),
                "candidates": diagnostic_result.get("candidates", []),
                "final_decision": diagnostic_result.get("final_decision", {}),
                "bo_critic": diagnostic_result.get("bo_critic", {}),
                "model_used": bool(model_used),
                "phase1_ok": bool(diagnostic_result.get("phase1_ok")),
                "phase2_ok": bool(diagnostic_result.get("phase2_ok")),
            },
            "selection_fallback_reason": (
                ""
                if candidates
                else f"frontier_reasoning_failed: {diagnostic_result.get('error', 'unknown')}"
            ),
        }
        return candidates, frontier_log

    def build_candidate_nodes(
        self,
        *,
        round_idx: int,
        target_nodes: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        scbo_reflection: Dict[str, Any] | None = None,
        knowledge_context: Dict[str, Any] | None = None,
        similar_task_context: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
        search_space_refinement: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        root_state = self.build_root_state(
            round_idx=round_idx,
            stage_trials=stage_trials,
            stage_policy=stage_policy,
            scbo_reflection=scbo_reflection,
        )
        activated = self.activate_skills(root_state)
        target_nodes = max(0, int(target_nodes))
        per_skill = max(2, int(math.ceil(max(1, target_nodes) / max(1, len(activated)))) + 1)
        seen_keys = set(exclude_param_keys)
        rejected: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []
        skill_outputs: List[Dict[str, Any]] = []

        for entry in activated:
            skill_name = entry["skill"]
            payload = {
                "round_idx": round_idx,
                "target_count": per_skill,
                "recall_threshold": stage_policy["recall_threshold"],
                "recall_slack": stage_policy.get("recall_slack", 0.0),
                "allowed_parameter_space": (
                    self.space.normalize_constraints(allowed_values_override)
                    if allowed_values_override is not None
                    else self.space.export_parameter_space()
                ),
                "root_state": root_state,
                "branch": skill_name,
                "branch_intent": entry["reason"],
                "scbo_reflection": scbo_reflection or {},
                "knowledge_context": knowledge_context or {},
                "similar_task_context": similar_task_context or {},
                "search_space_refinement": search_space_refinement or {},
                "output_schema": {
                    "skill_name": skill_name,
                    "branch": skill_name,
                    "candidates": [
                        {
                            "node_id": "...",
                            "params": {name: "..." for name in PARAM_ORDER},
                            "thinking": "short branch-specific reasoning",
                            "expected_behavior": {"recall": "...", "qps": "..."},
                        }
                    ],
                },
            }
            skill_result = None
            accepted: List[Dict[str, Any]] = []
            if self.enable_agentic:
                skill_result = self._call_skill(skill_name, payload)
                accepted = self._sanitize_skill_candidates(
                    skill_name=skill_name,
                    raw_candidates=self._raw_candidates_from_skill_payload(skill_result.get("parsed")),
                    seen_keys=seen_keys,
                    rejected=rejected,
                    allowed_values_override=allowed_values_override,
                    source="skill_llm",
                )

            if len(accepted) < per_skill:
                raise RuntimeError(
                    f"LLM skill '{skill_name}' produced {len(accepted)}/{per_skill} valid candidates "
                    "after sanitization — rule-based fallback is disabled."
                )

            nodes.extend(accepted)
            skill_outputs.append(
                {
                    "skill_name": skill_name,
                    "branch": skill_name,
                    "llm_ok": bool(skill_result and skill_result.get("ok")),
                    "llm_error": skill_result.get("error") if skill_result else "agentic_disabled",
                    "accepted_count": len(accepted),
                    "candidates": accepted,
                }
            )

        if len(nodes) < target_nodes:
            fill = self.initial_design_candidates(target_nodes - len(nodes), seen_keys)
            for item in fill:
                node = {
                    "node_id": f"GLOBAL-{len(nodes) + 1:03d}",
                    "branch": "global_fill",
                    "source_skill": "global_fill",
                    "params": item["params"],
                    "thinking": item.get("note", "global fill"),
                    "expected_behavior": {"recall": "unknown", "qps": "unknown"},
                    "source": item.get("source", "global_fill"),
                }
                nodes.append(node)

        log = {
            "root_state": root_state,
            "activated_skills": activated,
            "skill_outputs": skill_outputs,
            "rejected_candidate_nodes": rejected,
        }
        return nodes[: max(target_nodes, len(nodes))], log

    def _selection_rank_key(self, node: Dict[str, Any], stage: str) -> Tuple[float, float, float, float, float]:
        action_stats = self._action_stats_summary(node)
        qps_mean = float(action_stats.get("mu_G_tau", action_stats.get("qps_mean", 0.0)))
        qps_std = float(action_stats.get("sigma_G_tau", action_stats.get("qps_std", 0.0)))
        feasible_prob = float(action_stats.get("P_feas", action_stats.get("feasible_probability", 0.0)))
        score = float(node.get("score", action_stats.get("score", 0.0)))
        observation_priority = float(node.get("observation", {}).get("priority_score", 0.0))
        if stage in {"cold_start", "no_feasible_point", "infeasible_near_threshold"}:
            return (feasible_prob, score, observation_priority, qps_mean, qps_std)
        if stage == "recall_too_high_qps_low":
            return (qps_mean, score, observation_priority, feasible_prob, qps_std)
        return (score, feasible_prob, observation_priority, qps_mean, qps_std)

    def _max_hypotheses(self) -> int:
        return max(1, int(self.llm_reasoning_cfg.get("max_hypotheses", 4)))

    def _include_bo_baseline_hypothesis(self) -> bool:
        return bool(self.llm_reasoning_cfg.get("include_ts_baseline_candidate", True))

    def _mechanism_from_node(self, node: Dict[str, Any]) -> str:
        node_state = str(node.get("observation", {}).get("node_state", "")).strip()
        if node_state == "high_uncertainty":
            return "high_uncertainty_exploration"
        source_skill = str(node.get("source_skill", "")).strip()
        if source_skill == "construction_parameter_perturbation":
            return "graph_quality_compensation"
        if source_skill == "search_parameter_adjustment":
            return "search_width_adjustment"
        if source_skill == "best_neighborhood_exploration":
            return "boundary_local_exploitation"
        return "bo_baseline"

    def _candidate_reasoning_record(self, node: Dict[str, Any]) -> Dict[str, Any]:
        action_stats = self._action_stats_summary(node)
        critic = node.get("bo_critic") if isinstance(node.get("bo_critic"), dict) else {}
        return {
            "node_id": node.get("node_id", ""),
            "build": self._build_only_params(node.get("params")),
            "params": node.get("params"),
            "source_skill": node.get("source_skill", ""),
            "node_state": node.get("observation", {}).get("node_state", ""),
            "thinking": node.get("thinking", ""),
            "accepted": bool(node.get("accepted", action_stats.get("accepted", True))),
            "filter_reason": str(node.get("filter_reason", action_stats.get("filter_reason", "")) or ""),
            "score": float(node.get("score", action_stats.get("score", 0.0))),
            "bo_critic": critic,
            "expected_behavior": node.get("expected_behavior", {}),
        }

    def _resolve_reasoning_node(
        self,
        raw: Dict[str, Any] | None,
        *,
        candidates_by_id: Dict[str, Dict[str, Any]],
        candidates_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        for field in ["candidate_node_id", "target_node_id", "selected_node_id", "node_id"]:
            node_id = str(raw.get(field, "")).strip()
            if node_id and node_id in candidates_by_id:
                return candidates_by_id[node_id]
        params = raw.get("target_build") if isinstance(raw.get("target_build"), dict) else raw.get("selected_build")
        if isinstance(params, dict):
            try:
                canonical = self.canonicalize(params)
            except Exception:
                return None
            return candidates_by_key.get(self._build_key(canonical))
        return None

    def _build_bo_baseline_hypothesis(self, ranked_nodes: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not ranked_nodes or not self._include_bo_baseline_hypothesis():
            return None
        node = ranked_nodes[0]
        return {
            "hypothesis_id": "H0",
            "candidate_node_id": node.get("node_id", ""),
            "mechanism": "bo_baseline",
            "target_build": self._build_only_params(node.get("params")),
            "rationale": "BO critic ranks this candidate highest among the current pool.",
            "predicted_frontier_effect": "High posterior utility within the current candidate pool.",
            "falsifiable_prediction": "The executed build should outperform the current incumbent if the posterior is well calibrated.",
            "bo_critic": node.get("bo_critic", {}),
            "source": "bo_baseline",
        }

    def _sanitize_reasoning_hypotheses(
        self,
        raw_hypotheses: Any,
        *,
        ranked_nodes: Sequence[Dict[str, Any]],
        candidates_by_id: Dict[str, Dict[str, Any]],
        candidates_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        baseline = self._build_bo_baseline_hypothesis(ranked_nodes)
        if baseline is not None:
            seen_node_ids.add(str(baseline["candidate_node_id"]))
            sanitized.append(baseline)

        if isinstance(raw_hypotheses, list):
            for idx, raw in enumerate(raw_hypotheses, start=1):
                node = self._resolve_reasoning_node(
                    raw if isinstance(raw, dict) else None,
                    candidates_by_id=candidates_by_id,
                    candidates_by_key=candidates_by_key,
                )
                if node is None:
                    continue
                node_id = str(node.get("node_id", ""))
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                sanitized.append(
                    {
                        "hypothesis_id": str((raw or {}).get("hypothesis_id") or f"H{len(sanitized)}"),
                        "candidate_node_id": node_id,
                        "mechanism": str((raw or {}).get("mechanism") or self._mechanism_from_node(node)),
                        "target_build": self._build_only_params(node.get("params")),
                        "rationale": str((raw or {}).get("rationale") or node.get("thinking", "")),
                        "predicted_frontier_effect": str(
                            (raw or {}).get("predicted_frontier_effect")
                            or node.get("expected_behavior", {}).get("qps", "")
                            or "Seek a better recall-QPS tradeoff near the current frontier."
                        ),
                        "falsifiable_prediction": str(
                            (raw or {}).get("falsifiable_prediction")
                            or "The real frontier should move in the predicted direction if this mechanism is correct."
                        ),
                        "bo_critic": node.get("bo_critic", {}),
                        "source": "llm",
                    }
                )
                if len(sanitized) >= self._max_hypotheses():
                    break

        if not sanitized:
            for idx, node in enumerate(ranked_nodes[: self._max_hypotheses()], start=1):
                node_id = str(node.get("node_id", ""))
                if node_id in seen_node_ids:
                    continue
                sanitized.append(
                    {
                        "hypothesis_id": f"H{idx}",
                        "candidate_node_id": node_id,
                        "mechanism": self._mechanism_from_node(node),
                        "target_build": self._build_only_params(node.get("params")),
                        "rationale": str(node.get("thinking", "") or "Fallback from BO-ranked candidate."),
                        "predicted_frontier_effect": "Fallback exploitation of a BO-ranked candidate.",
                        "falsifiable_prediction": "If selected, the candidate should validate its posterior ranking.",
                        "bo_critic": node.get("bo_critic", {}),
                        "source": "fallback",
                    }
                )
        return sanitized[: self._max_hypotheses()]

    def _sanitize_reasoning_decision(
        self,
        raw_decision: Any,
        *,
        hypotheses: Sequence[Dict[str, Any]],
        ranked_nodes: Sequence[Dict[str, Any]],
        candidates_by_id: Dict[str, Dict[str, Any]],
        candidates_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypothesis_by_id = {str(item.get("hypothesis_id", "")): item for item in hypotheses}
        selected_hypothesis = None
        selected_node = None
        if isinstance(raw_decision, dict):
            hypothesis_id = str(raw_decision.get("selected_hypothesis_id", "")).strip()
            if hypothesis_id and hypothesis_id in hypothesis_by_id:
                selected_hypothesis = hypothesis_by_id[hypothesis_id]
                selected_node = candidates_by_id.get(str(selected_hypothesis.get("candidate_node_id", "")))
            if selected_node is None:
                selected_node = self._resolve_reasoning_node(
                    raw_decision,
                    candidates_by_id=candidates_by_id,
                    candidates_by_key=candidates_by_key,
                )
                if selected_node is not None:
                    for item in hypotheses:
                        if str(item.get("candidate_node_id", "")) == str(selected_node.get("node_id", "")):
                            selected_hypothesis = item
                            break

        if selected_node is None:
            selected_node = ranked_nodes[0] if ranked_nodes else None
            if selected_node is not None:
                for item in hypotheses:
                    if str(item.get("candidate_node_id", "")) == str(selected_node.get("node_id", "")):
                        selected_hypothesis = item
                        break

        if selected_node is None:
            return {
                "selected_hypothesis_id": "",
                "selected_node_id": "",
                "selected_build": None,
                "decision_type": "no_candidate",
                "why_selected": "No candidate node was available.",
                "why_rejected_others": [],
            }

        why_rejected = raw_decision.get("why_rejected_others", []) if isinstance(raw_decision, dict) else []
        if isinstance(why_rejected, str):
            why_rejected = [why_rejected]
        if not isinstance(why_rejected, list):
            why_rejected = []
        return {
            "selected_hypothesis_id": "" if selected_hypothesis is None else str(selected_hypothesis.get("hypothesis_id", "")),
            "selected_node_id": str(selected_node.get("node_id", "")),
            "selected_build": self._build_only_params(selected_node.get("params")),
            "decision_type": str(
                (raw_decision or {}).get("decision_type")
                if isinstance(raw_decision, dict)
                else "ranked_fallback"
            )
            or "ranked_fallback",
            "why_selected": str(
                (raw_decision or {}).get("why_selected")
                if isinstance(raw_decision, dict)
                else "Fallback to the highest-ranked candidate under BO statistics."
            )
            or "Fallback to the highest-ranked candidate under BO statistics.",
            "why_rejected_others": why_rejected,
        }

    def _call_reasoning_json(self, skill_name: str, prompt: str) -> Dict[str, Any]:
        raw = ""
        parsed = None
        error = None
        try:
            raw = self._invoke_llm(skill_name, prompt)
            parsed = self._extract_json_payload(raw)
            if parsed is None:
                error = "json_parse_failed"
        except Exception as exc:
            error = f"llm_error: {exc}"
        result = {"ok": error is None, "error": error, "raw": raw, "parsed": parsed}
        logger.info(
            "LLM reasoning result skill=%s ok=%s error=%s parsed=%s",
            skill_name,
            result["ok"],
            error or "",
            json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "<none>",
        )
        return result

    def _fallback_reasoning(
        self,
        *,
        ranked_nodes: Sequence[Dict[str, Any]],
        root_state: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        candidates_by_id = {str(node.get("node_id", "")): node for node in ranked_nodes}
        candidates_by_key = {self._build_key(node["params"]): node for node in ranked_nodes if isinstance(node.get("params"), dict)}
        hypotheses = self._sanitize_reasoning_hypotheses(
            None,
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        final_decision = self._sanitize_reasoning_decision(
            {},
            hypotheses=hypotheses,
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        critic_reviews = [
            {
                "hypothesis_id": item.get("hypothesis_id", ""),
                "candidate_node_id": item.get("candidate_node_id", ""),
                "bo_critic": item.get("bo_critic", {}),
            }
            for item in hypotheses
        ]
        return {
            "mode": self.reasoning_mode,
            "executed_mode": "fallback",
            "llm_ok": False,
            "fallback_used": True,
            "fallback_reason": reason,
            "frontier_diagnosis": {
                "optimization_stage": root_state.get("optimization_stage", "cold_start"),
                "stage_reason": root_state.get("stage_reason", ""),
                "trial_observations": root_state.get("trial_observations", []),
            },
            "surface_uncertainty": root_state.get("surface_uncertainty", {}),
            "hypotheses": hypotheses,
            "critic_reviews": critic_reviews,
            "bo_review": critic_reviews,
            "final_decision": final_decision,
        }

    def _run_single_agent_reasoning(
        self,
        *,
        ranked_nodes: Sequence[Dict[str, Any]],
        root_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates_by_id = {str(node.get("node_id", "")): node for node in ranked_nodes}
        candidates_by_key = {self._build_key(node["params"]): node for node in ranked_nodes if isinstance(node.get("params"), dict)}
        payload = {
            "root_state": root_state,
            "candidate_pool": [self._candidate_reasoning_record(node) for node in ranked_nodes[: max(6, self._max_hypotheses())]],
            "bo_baseline_candidate": self._candidate_reasoning_record(ranked_nodes[0]) if ranked_nodes else None,
            "output_schema": {
                "frontier_diagnosis": {"optimization_stage": "...", "summary": "..."},
                "surface_uncertainty": {"summary": "..."},
                "hypotheses": [
                    {
                        "hypothesis_id": "H1",
                        "candidate_node_id": "HTS-001",
                        "mechanism": "graph_quality_compensation",
                        "rationale": "...",
                        "predicted_frontier_effect": "...",
                        "falsifiable_prediction": "...",
                    }
                ],
                "bo_review": [{"hypothesis_id": "H1", "summary": "..."}],
                "final_decision": {
                    "selected_hypothesis_id": "H1",
                    "selected_node_id": "HTS-001",
                    "decision_type": "exploit",
                    "why_selected": "...",
                    "why_rejected_others": ["..."],
                },
            },
        }
        prompt = (
            "You are the HNSW LLM reasoner in single-agent mode.\n"
            "Use only the supplied candidate pool and BO critic statistics.\n"
            "Return strict JSON matching the provided schema. Do not invent new candidates.\n\n"
            f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        result = self._call_reasoning_json("hnsw_reasoning_single_agent", prompt)
        if not bool(result.get("ok", False)) or not isinstance(result.get("parsed"), dict):
            return self._fallback_reasoning(
                ranked_nodes=ranked_nodes,
                root_state=root_state,
                reason=str(result.get("error", "single_agent_invalid_output")),
            )
        parsed = result["parsed"]
        hypotheses = self._sanitize_reasoning_hypotheses(
            parsed.get("hypotheses"),
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        final_decision = self._sanitize_reasoning_decision(
            parsed.get("final_decision"),
            hypotheses=hypotheses,
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        critic_reviews = [
            {
                "hypothesis_id": item.get("hypothesis_id", ""),
                "candidate_node_id": item.get("candidate_node_id", ""),
                "bo_critic": item.get("bo_critic", {}),
            }
            for item in hypotheses
        ]
        return {
            "mode": self.reasoning_mode,
            "executed_mode": "single_agent",
            "llm_ok": True,
            "fallback_used": False,
            "frontier_diagnosis": parsed.get("frontier_diagnosis", {}),
            "surface_uncertainty": parsed.get("surface_uncertainty", root_state.get("surface_uncertainty", {})),
            "hypotheses": hypotheses,
            "critic_reviews": critic_reviews,
            "bo_review": parsed.get("bo_review", critic_reviews),
            "final_decision": final_decision,
        }

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))

    def _evaluate_build_critic(
        self,
        *,
        build: Dict[str, Any],
        observations: Sequence[Dict[str, Any]],
        obj_model: Any,
        con_model: Any,
        current_best_qps: float | None,
        prepared_round: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate a construction via GP posterior (BO critic — LLM.md module 3).

        Reuses the same logic as ``HNSWLIBSCBOOptimizer._build_candidate_critic``
        but self-contained so the agent can call it without a full SCBO instance.
        """
        import torch

        # --- vectorize build params (use self.space for bounds) ---
        build_names = list(BUILD_PARAM_ORDER)
        normalized: Dict[str, float] = {}
        for name in build_names:
            if name == "ef" and "ef" not in BUILD_PARAM_ORDER:
                normalized[name] = 0.5  # placeholder: ef scanned, not modelled
            else:
                normalized[name] = self.space.normalized_position(name, build.get(name, 0), clamp=True)
        vec = [normalized.get(name, 0.5) for name in build_names]

        # --- GP posterior ---
        x = torch.tensor([vec], dtype=torch.double)
        with torch.no_grad():
            # Handle both derivative-augmented GP (_DerivGP) and standard SingleTaskGP
            if hasattr(obj_model, '_is_derivative_gp') and obj_model._is_derivative_gp:
                obj_mean_f, obj_var_f = obj_model.predict_function_values(x)
                qps_mean = float(obj_mean_f[0])
                qps_std = float(obj_var_f.clamp_min(1e-12).sqrt()[0])
            else:
                obj_post = obj_model.posterior(x)
                qps_mean = float(obj_post.mean.squeeze(-1)[0])
                qps_std = float(obj_post.variance.clamp_min(1e-9).sqrt().squeeze(-1)[0])

            if hasattr(con_model, '_is_derivative_gp') and con_model._is_derivative_gp:
                con_mean_f, con_var_f = con_model.predict_function_values(x)
                con_mean = float(con_mean_f[0])
                con_std = float(con_var_f.clamp_min(1e-12).sqrt()[0])
            else:
                con_post = con_model.posterior(x)
                con_mean = float(con_post.mean.squeeze(-1)[0])
                con_std = float(con_post.variance.clamp_min(1e-9).sqrt().squeeze(-1)[0])

        qps_std = max(1e-6, qps_std)
        con_std = max(1e-9, con_std)

        # --- feasibility probability ---
        feasible_prob = 0.5 * (1.0 + math.erf(con_mean / (con_std * math.sqrt(2.0))))

        # --- improvement probability ---
        improve_prob = 0.5
        if current_best_qps is not None:
            denom = max(qps_std, 1e-9)
            improve_prob = 0.5 * (1.0 + math.erf((qps_mean - current_best_qps) / (denom * math.sqrt(2.0))))
        elif feasible_prob < 0.5:
            improve_prob = 0.25

        # --- information gain proxy (normalized uncertainty: higher = more to learn) ---
        # Note: this is a proxy, not true expected entropy reduction.
        # True IG requires Monte-Carlo over frontier outcomes; the SCBO optimizer
        # has a more complete implementation via _estimate_information_gain.
        ig_proxy = float(min(1.0, (qps_std / max(1e-6, abs(qps_mean)) + con_std / max(1e-9, abs(con_mean) + 0.1)) / 2.0))

        # --- GP reliability: how much should we trust the GP prediction? ---
        # Compute distance to nearest observed construction (in normalized space).
        nearest_distance = 1.0  # default: far
        nearest_params = None
        for obs in observations:
            obs_params = obs.get("params", {})
            if not isinstance(obs_params, dict):
                continue
            obs_vec = []
            for name in build_names:
                if name == "ef" and "ef" not in BUILD_PARAM_ORDER:
                    obs_vec.append(0.5)  # placeholder: ef is scanned, not tuned
                else:
                    obs_vec.append(self.space.normalized_position(name, obs_params.get(name, 0), clamp=True))
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, obs_vec)))
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_params = {name: obs_params.get(name) for name in build_names if name != "ef"}

        # Map distance to a reliability label.
        if nearest_distance < 0.05:
            gp_reliability = "high — target is nearly identical to an already-observed construction"
        elif nearest_distance < 0.15:
            gp_reliability = "medium — target is close to training data, GP is reasonably reliable"
        elif nearest_distance < 0.35:
            gp_reliability = "low — target is far from training data, GP is extrapolating"
        else:
            gp_reliability = "very low — target is very far from training data, GP prediction is unreliable"

        nearest_str = (
            f"M={nearest_params.get('M', '?')}, efC={nearest_params.get('ef_construction', '?')}"
            if nearest_params else "none"
        )

        # --- judgement ---
        judgement = "uncertain"
        judgement_reason = "Posterior is mixed; candidate may still be worth verifying."
        if feasible_prob < 0.35:
            judgement = "risky"
            judgement_reason = f"Recall feasibility probability ({feasible_prob:.3f}) is below guardrail."
        elif improve_prob >= 0.55 and feasible_prob >= 0.55:
            judgement = "promising"
            judgement_reason = "Posterior mean and feasibility both support improvement."
        elif current_best_qps is not None and qps_mean < current_best_qps and improve_prob < 0.3:
            judgement = "risky"
            judgement_reason = f"QPS posterior mean ({qps_mean:.0f}) below current best ({current_best_qps:.0f}) with low improvement probability ({improve_prob:.3f})."

        # --- trust region membership ---
        tr_membership = "unknown"
        if isinstance(prepared_round, dict):
            tr_bounds = prepared_round.get("trust_region_bounds")
            if isinstance(tr_bounds, dict):
                lower = tr_bounds.get("lower", [])
                upper = tr_bounds.get("upper", [])
                if lower and upper:
                    inside = all(lo <= v <= hi for lo, v, hi in zip(lower, vec, upper))
                    tr_membership = "inside" if inside else "outside"

        return {
            "mu_G_tau": qps_mean,
            "sigma_G_tau": qps_std,
            "P_feas": feasible_prob,
            "P_improve_over_best": improve_prob,
            "ig_proxy": ig_proxy,
            "judgement": judgement,
            "judgement_reason": judgement_reason,
            "dominance_status": "unknown",
            "trust_region_membership": tr_membership,
            "gp_reliability": gp_reliability,
            "nearest_observed": nearest_str,
            "nearest_distance": round(nearest_distance, 4),
            "prediction_source": "gp",
        }

    def _run_diagnostic_reasoning(
        self,
        *,
        root_state: Dict[str, Any],
        scbo_reflection: Dict[str, Any] | None = None,
        stage_policy: Dict[str, Any] | None = None,
        knowledge_context: Dict[str, Any] | None = None,
        allowed_values_override: Dict[str, Any] | None = None,
        observations: Sequence[Dict[str, Any]] | None = None,
        obj_model: Any = None,
        con_model: Any = None,
        model_used: bool = False,
        prepared_round: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Diagnostic reasoning: signal selection → trial diagnosis → candidate proposal.

        Phase 1a — LLM matches diagnostic signals to trial observations.
        Phase 1b — LLM diagnoses using selected signal procedures and proposes
                   candidate (M, efC, ef) configurations directly.
        Phase 2  — BO critic scores candidates; LLM selects best.

        Returns a dict with keys ``diagnosis``, ``candidates``, ``bo_critic``,
        ``selected_candidate``, and raw LLM metadata.
        """
        empty_return = {
            "ok": False,
            "error": "",
            "phase1a_ok": False,
            "phase1b_ok": False,
            "phase2_ok": False,
            "diagnosis": {},
            "candidates": [],
            "selected_candidate": None,
            "bo_critic": {},
        }
        if not self.enable_agentic:
            empty_return["error"] = "agentic_disabled"
            return empty_return

        threshold = float((stage_policy or {}).get("recall_threshold", 1.0))
        slack = float((stage_policy or {}).get("recall_slack", 0.0))
        reflection = scbo_reflection if isinstance(scbo_reflection, dict) else {}
        knowledge = knowledge_context
        obs_list = list(observations) if observations else []

        # --- compact BO context (trust_region, best_feasible in root_state) ---
        bo_context: Dict[str, Any] = {
            "trust_region": reflection.get("trust_region"),
            "length": float(reflection.get("length", 0.0)),
        }

        # ============================================================
        # Phase 1a: Diagnostic Tree navigation
        # ============================================================
        signal_selection_payload = {
            "task": "signal_selection",
            "recall_threshold": threshold,
            "diagnostic_tree": DIAGNOSTIC_TREE,
            "trial_observations": root_state.get("trial_observations", []),
            "previous_attribution": root_state.get("previous_attribution"),
        }
        signal_selection_prompt = (
            "You are an HNSW diagnostic agent. Your task: navigate the diagnostic tree.\n\n"
            "PREVIOUS ROUND ATTRIBUTION: If previous_attribution is provided in the\n"
            "context, use it to focus your signal selection — the attribution already\n"
            "identified the likely bottleneck, so prioritize signals related to that\n"
            "root cause.\n\n"
            "STEP 1 — Identify the current problem:\n"
            "  Check trial_observations. Is recall ≥ τ for any trial?\n"
            "  If NO trial is feasible → you have a RECALL problem. Focus on R-nodes.\n"
            "  If some trials are feasible → you have a QPS problem. Focus on Q-nodes.\n\n"
            "STEP 2 — For each node under the relevant problem, check whether its\n"
            "  described metric pattern is visible in the trial data.\n"
            "  The tree is organized as: problem → metric → description → solution.\n\n"
            "STEP 3 — Select ALL nodes whose metric pattern matches the data.\n"
            '  Return JSON: {"problem": "recall"|"qps", '
            '"selected_nodes": ["R1", "R3"], '
            '"reasoning": "why each selected node matches"}\n\n'
            "IMPORTANT: If NO trial is feasible, do NOT select any Q-node (Q1-Q9).\n"
            "  Reducing parameters when recall is below τ will make things worse.\n\n"
            f"Context:\n{json.dumps(signal_selection_payload, ensure_ascii=False, indent=2)}"
        )
        signal_result = self._call_reasoning_json("hnsw_signal_selection", signal_selection_prompt)
        selected_ids: List[str] = []
        if signal_result.get("ok") and isinstance(signal_result.get("parsed"), dict):
            selected_ids = signal_result["parsed"].get("selected_nodes") or signal_result["parsed"].get("selected_signals") or []
            if isinstance(selected_ids, list):
                selected_ids = [str(s) for s in selected_ids if any(n["node_id"] == str(s) for n in DIAGNOSTIC_TREE)]
            else:
                selected_ids = []
        if not selected_ids:
            # Fallback: always include S7 (recall ceiling) and S5 (efC saturation)
            selected_ids = ["R1", "Q1"]

        # Assemble selected signal details for Phase 1b
        selected_details = "\n\n".join(
            _node_to_detail(n) for n in DIAGNOSTIC_TREE
            if n["node_id"] in selected_ids
        )
        logger.info(
            "Signal selection: selected=%s reasoning=%s",
            selected_ids,
            (signal_result.get("parsed") or {}).get("reasoning", "") if signal_result.get("ok") else "fallback",
        )

        # ============================================================
        # Phase 1b: LLM diagnosis + hypothesis (NO final_decision)
        # ============================================================
        phase1_payload = {
            "task": "diagnostic_reasoning",
            "phase": "diagnose_and_propose",
            "recall_threshold": threshold,
            "recall_slack": slack,
            "parameter_space": self.space.export_parameter_space(
                allowed_values_override
            ) if allowed_values_override else self.space.export_parameter_space(),
            "trial_observations": root_state.get("trial_observations", []),
            "best_so_far": root_state.get("current_best_feasible"),
            "selected_diagnostic_signals": selected_ids,
            "diagnostic_procedures": selected_details,
            "knowledge_context": (
                {
                    "similar_task_insights": (
                        self._strip_qps_from_insights(
                            knowledge.get("similar_task_insights_full", [])[:3]
                        )
                        if int(root_state.get("observed_success_count", 0)) < 8
                        else []
                    ),
                }
                if isinstance(knowledge, dict)
                else {}
            ),
            "output_schema": {
                "diagnosis": {
                    "key_findings": ["bullet-point summary of what the diagnostic procedures revealed"],
                    "signals_matched": ["which signals were confirmed and how"],
                },
                "candidates": [
                    {
                        "M": 0,
                        "ef_construction": 0,
                        "ef": 0,
                        "rationale": "why this config — reference specific signals and trial data",
                    }
                ],
            },
        }

        phase1_prompt = (
            "You are the HNSW diagnostic reasoner (Phase 1b: Diagnose & Propose).\n\n"
            "STEP 0 — Read the diagnostic procedures:\n"
            "  The `diagnostic_procedures` field contains the full diagnostic workflow for\n"
            "  the signals that were pre-selected as most relevant. Read them carefully.\n\n"
            "STEP 1 — Diagnose:\n"
            "  Apply the diagnostic procedures to the `trial_observations`.\n"
            "  Compare trials that differ in only one parameter to isolate effects.\n"
            "  For each signal, state whether it is confirmed and what action it implies.\n\n"
            "  Cross-construction comparison guide:\n"
            "    Fix efC, compare M: higher M → higher QPS at same recall? → M under/over provisioned\n"
            "    Fix M, compare efC: higher efC → higher QPS at same recall? → efC under/over provisioned\n"
            "    Compare trials at same (M,efC) with different ef → observe recall-QPS trade-off\n\n"
        )
        if "ef" in BUILD_PARAM_ORDER:
            phase1_prompt += (
            "\n"
            "  Search Width (ef) — direct tuning (no free-ef scan):\n"
            "    ef↑ → recall↑, QPS↓  (monotonic, safe, low-cost — no index rebuild)\n"
            "    ef↓ → QPS↑, recall↓  (monotonic, safe, low-cost)\n"
            "    ⚠ PREFER adjusting ef over M/efC when possible: ef changes don't require\n"
            "      rebuilding the index and are instantly testable.\n"
            "    ⚠ If recall is slightly below τ, adjust the parameter(s) indicated by evidence — ef increase is zero-rebuild, M/efC increases rebuild the index.\n"
            "    ⚠ If recall has margin > 0.01, decrease ef by 2-10 to boost QPS.\n"
            "    Treat ef, M, and efC as equal tuning knobs; vary them across trials based on evidence.\n"
        )
        phase1_prompt += (
            "\n"
            "STEP 2 — Propose candidates:\n"
            "  Propose 1-3 new (M, ef_construction, ef) configurations.\n"
            "  Each candidate MUST have a rationale referencing specific signals and trial data.\n"
            "  You may adjust multiple parameters; choose the lever(s) indicated by evidence.\n"
            "  Vary ef across candidates — don't keep it constant.\n\n"
            "Rules:\n"
            "- Do NOT propose configs that already exist in trial_observations.\n"
            "- Use only M, ef_construction, ef. Stay in parameter space.\n"
            "- Return strict JSON matching the output schema.\n\n"
            f"Context:\n{json.dumps(phase1_payload, ensure_ascii=False, indent=2)}"
        )

        phase1_result = self._call_reasoning_json("hnsw_diagnostic_phase1", phase1_prompt)
        phase1_ok = bool(phase1_result.get("ok", False)) and isinstance(phase1_result.get("parsed"), dict)
        if not phase1_ok:
            empty_return["error"] = phase1_result.get("error", "phase1_invalid_output")
            empty_return["raw"] = phase1_result.get("raw", "")
            return empty_return

        phase1_parsed = phase1_result["parsed"]
        diagnosis = phase1_parsed.get("diagnosis") or {}
        raw_candidates = phase1_parsed.get("candidates") or []
        if not isinstance(raw_candidates, list):
            raw_candidates = []

        # --- process candidates from Phase 1b ---
        if not raw_candidates:
            return {
                **empty_return,
                "error": "no_candidates",
                "phase1a_ok": True,
                "phase1b_ok": True,
            }
        # Take the first candidate as the primary target
        first_candidate = raw_candidates[0] if isinstance(raw_candidates[0], dict) else {}
        target_build = {
            "M": first_candidate.get("M", 0),
            "ef_construction": first_candidate.get("ef_construction", 0),
            "ef": first_candidate.get("ef", 0),
        }
        # If ef is 0 or missing, fill it with a valid default.
        if isinstance(target_build, dict) and int(target_build.get("ef", 0)) <= 0:
            ef_vals = []
            for obs in obs_list:
                obs_params = obs.get("params") if isinstance(obs.get("params"), dict) else {}
                sef = obs.get("selected_ef")
                if sef is not None:
                    try:
                        ef_vals.append(int(sef))
                    except Exception:
                        pass
                else:
                    oef = obs_params.get("ef")
                    if oef is not None:
                        try:
                            ef_vals.append(int(oef))
                        except Exception:
                            pass
            default_ef = int(statistics.median(ef_vals)) if ef_vals else 16
            target_build["ef"] = max(1, default_ef)
        canonical_build: Dict[str, Any] | None = None
        try:
            canonical_build = self.canonicalize(target_build, domain_constraints=allowed_values_override)
        except Exception:
            canonical_build = None

        # --- check for duplicate: has this construction already been executed? ---
        if canonical_build is not None:
            target_key = build_params_to_key(
                {name: canonical_build[name] for name in BUILD_PARAM_ORDER},
                BUILD_PARAM_ORDER,
            )
            observed_keys = {
                build_params_to_key(
                    {name: (r.get("build") or {}).get(name, 0) for name in BUILD_PARAM_ORDER},
                    BUILD_PARAM_ORDER,
                )
                for r in (root_state.get("trial_observations") or [])
                if isinstance(r.get("build"), dict)
            }
            if target_key in observed_keys:
                return {
                    **empty_return,
                    "error": "duplicate_construction",
                    "phase1_ok": True,
                    "phase2_ok": False,
                    "canonical_build": None,
                    "bo_critic": {
                        "judgement": "duplicate",
                        "judgement_reason": "Target construction has already been executed. Propose a different one.",
                    },
                }

        # ============================================================
        # BO critic: evaluate the target construction
        # ============================================================
        bo_critic: Dict[str, Any] = {}
        if canonical_build is not None and model_used:
            bo_critic = self._evaluate_build_critic(
                build=canonical_build,
                observations=obs_list,
                obj_model=obj_model,
                con_model=con_model,
                current_best_qps=float((root_state.get("current_best_feasible") or {}).get("QPS", 0)),
                prepared_round=prepared_round,
            )
        elif canonical_build is not None:
            # GP not ready — heuristic fallback
            bo_critic = {
                "mu_G_tau": float(first_candidate.get("ef", 0)),
                "sigma_G_tau": 1.0,
                "P_feas": 0.5,
                "P_improve_over_best": 0.5,
                "ig_proxy": 0.5,
                "judgement": "uncertain",
                "judgement_reason": "GP surrogate not ready — heuristic-only evaluation.",
                "dominance_status": "unknown",
                "trust_region_membership": "unknown",
                "gp_reliability": "unknown — GP not fitted yet",
                "nearest_observed": "none",
                "nearest_distance": 1.0,
                "prediction_source": "heuristic",
            }
        else:
            bo_critic = {
                "mu_G_tau": 0,
                "sigma_G_tau": 1.0,
                "P_feas": 0,
                "P_improve_over_best": 0,
                "ig_proxy": 0,
                "judgement": "risky",
                "judgement_reason": "Target build could not be canonicalized.",
                "dominance_status": "unknown",
                "trust_region_membership": "unknown",
                "gp_reliability": "unknown — invalid target",
                "nearest_observed": "none",
                "nearest_distance": 1.0,
                "prediction_source": "invalid",
            }

        # --- Post-hoc consistency check ---
        # (mechanism library removed — diagnosis goes directly to candidates)
        bo_critic["diagnosis_note"] = "candidate proposed directly from diagnostic reasoning"

        # ============================================================
        # Phase 2: LLM reconciles with BO verdict → final_decision
        # ============================================================
        phase2_payload = {
            "task": "frontier_reconciliation",
            "phase": "reconcile_with_bo",
            "recall_threshold": threshold,
            "your_diagnosis": diagnosis,
            "your_candidates": raw_candidates,
            "bo_critic": bo_critic,
            "output_schema": {
                "final_decision": {
                    "decision": "commit / revise / skip",
                    "why": "reconciliation reasoning against BO posterior",
                    "selected_build": {"M": 0, "ef_construction": 0, "ef": 0},
                },
            },
        }

        phase2_prompt = (
            "You are the HNSW reconciliation step (Phase 2).\n\n"
            "You previously proposed a hypothesis with a target construction.\n"
            "The BO surrogate has now evaluated that construction.\n"
            "Reconcile your prediction with the BO posterior and make a final decision.\n\n"
            "Reconciliation rules:\n"
            "  - P_improve ≥ 0.5 AND P_feas ≥ 0.5 → BO is confident this improves → commit\n"
            "  - P_improve < 0.5 but mechanism is well-supported by cross-construction data\n"
            "    AND target is in an unexplored region (IG ≥ 0.01 or far from training data)\n"
            "    → commit (validate hypothesis + gather high-value data)\n"
            "  - Target is a duplicate of an already-executed construction → revise (do NOT commit)\n"
            "  - Mechanism lacks cross-construction support → revise or skip\n"
            "  - HARD GUARDRAIL: P_feas < 0.3 → force skip (too likely infeasible)\n\n"
            "BO verdict:\n"
            f"  μ_G^τ    = {bo_critic.get('mu_G_tau', 0):.1f}\n"
            f"  σ_G^τ    = {bo_critic.get('sigma_G_tau', 0):.1f}\n"
            f"  P_feas   = {bo_critic.get('P_feas', 0):.3f}\n"
            f"  P_improve = {bo_critic.get('P_improve_over_best', 0):.3f}\n"
            f"  IG       = {bo_critic.get('ig_proxy', 0):.3f} (uncertainty proxy, higher = more to learn)\n"
            f"  judgement = {bo_critic.get('judgement', 'unknown')}\n"
            f"  reason    = {bo_critic.get('judgement_reason', '')}\n"
            f"  gp_reliability = {bo_critic.get('gp_reliability', 'unknown')}\n"
            f"  nearest_observed = {bo_critic.get('nearest_observed', 'none')}\n"
            f"  trust_region = {bo_critic.get('trust_region_membership', 'unknown')}\n\n"
            "How to use gp_reliability:\n"
            "  high   → GP is confident about this region (near training data). Trust the numbers.\n"
            "  medium → GP is reasonably reliable. Use numbers as one signal among others.\n"
            "  low/very low → GP is extrapolating. Numbers may be unreliable.\n"
            "                 Your cross-construction evidence should carry MORE weight.\n\n"
            "Self-check before deciding:\n"
            "  Re-read your own key_readings above. Verify they actually support\n"
            "  your selected mechanism. Common mistakes:\n"
            "    - Claiming 'M is over-provisioned' but margins are all negative → recall is\n"
            "      already below τ, reducing M would make it worse. Switch to increase_density.\n"
            "    - Claiming 'efC is saturated' but efC↑ actually improved Q_τ → efC is NOT\n"
            "      saturated. Do not reduce it.\n"
            "    - A number like 'ef* by 1-2' is NOT a negative margin. Read carefully.\n"
            "  If your key_readings contradict your mechanism, you MUST revise or skip.\n\n"
            "Return strict JSON with final_decision only.\n\n"
            f"Context:\n{json.dumps(phase2_payload, ensure_ascii=False, indent=2)}"
        )

        phase2_result = self._call_reasoning_json("hnsw_frontier_reasoning_phase2", phase2_prompt)
        phase2_ok = bool(phase2_result.get("ok", False)) and isinstance(phase2_result.get("parsed"), dict)
        final_decision: Dict[str, Any] = {}
        if phase2_ok:
            phase2_parsed = phase2_result["parsed"]
            final_decision = phase2_parsed.get("final_decision") or {}
            # Normalize: LLM may return bare string "commit"/"skip" instead of dict
            if isinstance(final_decision, str):
                final_decision = {"decision": final_decision, "selected_build": target_build}
        else:
            # Phase 2 failed — fall back to heuristic: commit if BO is promising
            if bo_critic.get("judgement") in ("promising", "uncertain"):
                final_decision = {
                    "decision": "commit",
                    "why": f"Phase 2 LLM call failed ({phase2_result.get('error', 'unknown')}). "
                           f"BO judgement={bo_critic.get('judgement')}. Auto-committing.",
                    "selected_build": target_build,
                }
            else:
                final_decision = {
                    "decision": "skip",
                    "why": f"Phase 2 LLM call failed ({phase2_result.get('error', 'unknown')}). "
                           f"BO judgement={bo_critic.get('judgement')}. Auto-skipping.",
                    "selected_build": target_build,
                }

        # Hard guardrail: force skip if feasibility is too low
        if bo_critic.get("P_feas", 0.5) < 0.3 and final_decision.get("decision") == "commit":
            final_decision = {
                "decision": "skip",
                "why": f"BO P_feas={bo_critic.get('P_feas', 0):.3f} < 0.3 guardrail. Target construction too likely infeasible.",
                "selected_build": target_build,
            }

        # (mechanism override removed — diagnosis goes directly to candidates)

        # Handle revise: use the revised selected_build, validate it
        if final_decision.get("decision") == "revise":
            revised_build = final_decision.get("selected_build") or {}
            if revised_build:
                try:
                    revised_canonical = self.canonicalize(revised_build, domain_constraints=allowed_values_override)
                    if revised_canonical:
                        canonical_build = revised_canonical
                except Exception:
                    pass  # keep original canonical_build if revised is invalid

        # Handle skip: no build to execute
        if final_decision.get("decision") == "skip":
            canonical_build = None

        return {
            "ok": True,
            "phase1_ok": True,
            "phase2_ok": phase2_ok,
            "error": "",
            "phase1_raw": phase1_result.get("raw", ""),
            "phase2_raw": phase2_result.get("raw", ""),
            "frontier_diagnosis": diagnosis,
            "diagnosis": diagnosis,
            "candidates": raw_candidates,
            "final_decision": final_decision,
            "bo_critic": bo_critic,
            "canonical_build": canonical_build,
            "_mechanism_override": None,
        }

    @staticmethod
    def _validate_mechanism_consistency(
        *,
        mechanism_name: str,
        key_readings: list,
        target_build: dict | None,
        current_best_feasible: dict | None,
        best_qps: float,
    ) -> dict | None:
        """Mechanism-data consistency check.

        Regex-based checks were removed — they produced false positives
        (e.g. matching '1-2' as a negative margin).  Instead, the Phase 2
        prompt asks the LLM to self-check its key_readings against its
        selected mechanism before making a final decision.
        """
        return None

    def _run_multi_agent_reasoning(
        self,
        *,
        ranked_nodes: Sequence[Dict[str, Any]],
        root_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates_by_id = {str(node.get("node_id", "")): node for node in ranked_nodes}
        candidates_by_key = {self._build_key(node["params"]): node for node in ranked_nodes if isinstance(node.get("params"), dict)}
        base_payload = {
            "root_state": root_state,
            "candidate_pool": [self._candidate_reasoning_record(node) for node in ranked_nodes[: max(6, self._max_hypotheses())]],
            "bo_baseline_candidate": self._candidate_reasoning_record(ranked_nodes[0]) if ranked_nodes else None,
        }
        diag_prompt = (
            "You are the HNSW frontier diagnoser.\n"
            "Summarize the current frontier state and uncertainty using only the provided context.\n"
            "Return strict JSON with frontier_diagnosis and surface_uncertainty.\n\n"
            f"Context:\n{json.dumps(base_payload, ensure_ascii=False, indent=2)}"
        )
        diag_result = self._call_reasoning_json("hnsw_reasoning_diagnoser", diag_prompt)
        if not bool(diag_result.get("ok", False)) or not isinstance(diag_result.get("parsed"), dict):
            return self._fallback_reasoning(
                ranked_nodes=ranked_nodes,
                root_state=root_state,
                reason=str(diag_result.get("error", "diagnoser_invalid_output")),
            )

        hypothesis_payload = {
            **base_payload,
            "frontier_diagnosis": diag_result["parsed"].get("frontier_diagnosis", {}),
            "surface_uncertainty": diag_result["parsed"].get("surface_uncertainty", {}),
        }
        hypothesis_prompt = (
            "You are the HNSW hypothesis generator.\n"
            "Choose from the provided candidate pool only.\n"
            "Return strict JSON with a hypotheses array.\n\n"
            f"Context:\n{json.dumps(hypothesis_payload, ensure_ascii=False, indent=2)}"
        )
        hypothesis_result = self._call_reasoning_json("hnsw_reasoning_hypothesizer", hypothesis_prompt)
        if not bool(hypothesis_result.get("ok", False)) or not isinstance(hypothesis_result.get("parsed"), dict):
            return self._fallback_reasoning(
                ranked_nodes=ranked_nodes,
                root_state=root_state,
                reason=str(hypothesis_result.get("error", "hypothesizer_invalid_output")),
            )

        hypotheses = self._sanitize_reasoning_hypotheses(
            hypothesis_result["parsed"].get("hypotheses"),
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        critic_reviews = [
            {
                "hypothesis_id": item.get("hypothesis_id", ""),
                "candidate_node_id": item.get("candidate_node_id", ""),
                "bo_critic": item.get("bo_critic", {}),
            }
            for item in hypotheses
        ]
        arbiter_payload = {
            **hypothesis_payload,
            "hypotheses": hypotheses,
            "critic_reviews": critic_reviews,
        }
        arbiter_prompt = (
            "You are the HNSW arbiter.\n"
            "Read the candidate hypotheses and BO critic reviews, then select exactly one candidate.\n"
            "Return strict JSON with final_decision.\n\n"
            f"Context:\n{json.dumps(arbiter_payload, ensure_ascii=False, indent=2)}"
        )
        arbiter_result = self._call_reasoning_json("hnsw_reasoning_arbiter", arbiter_prompt)
        if not bool(arbiter_result.get("ok", False)) or not isinstance(arbiter_result.get("parsed"), dict):
            return self._fallback_reasoning(
                ranked_nodes=ranked_nodes,
                root_state=root_state,
                reason=str(arbiter_result.get("error", "arbiter_invalid_output")),
            )

        final_decision = self._sanitize_reasoning_decision(
            arbiter_result["parsed"].get("final_decision"),
            hypotheses=hypotheses,
            ranked_nodes=ranked_nodes,
            candidates_by_id=candidates_by_id,
            candidates_by_key=candidates_by_key,
        )
        return {
            "mode": self.reasoning_mode,
            "executed_mode": "multi_agent",
            "llm_ok": True,
            "fallback_used": False,
            "frontier_diagnosis": diag_result["parsed"].get("frontier_diagnosis", {}),
            "surface_uncertainty": diag_result["parsed"].get("surface_uncertainty", root_state.get("surface_uncertainty", {})),
            "hypotheses": hypotheses,
            "critic_reviews": critic_reviews,
            "bo_review": critic_reviews,
            "final_decision": final_decision,
        }

    def _run_reasoning(
        self,
        *,
        ranked_nodes: Sequence[Dict[str, Any]],
        root_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not ranked_nodes:
            return self._fallback_reasoning(ranked_nodes=ranked_nodes, root_state=root_state, reason="empty_candidate_pool")
        if not self.enable_agentic:
            return self._fallback_reasoning(ranked_nodes=ranked_nodes, root_state=root_state, reason="agentic_disabled")
        if self.reasoning_mode == "multi_agent":
            return self._run_multi_agent_reasoning(ranked_nodes=ranked_nodes, root_state=root_state)
        return self._run_single_agent_reasoning(ranked_nodes=ranked_nodes, root_state=root_state)

    # ── Trial Attribution Analysis ───────────────────────────────────────

    def _build_attribution_prompt(
        self,
        *,
        round_observations: List[Dict[str, Any]],
        stage_observations: List[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        knowledge_context: Dict[str, Any] | None = None,
    ) -> str:
        """Build the attribution analysis prompt with pure observational data.

        Provides raw metric values and domain knowledge (diagnostic signal
        catalog), but NO hard thresholds — the LLM reasons about bottlenecks
        on its own.
        """
        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy.get("recall_slack", 0.0))

        # ── build per-trial observation blocks ──
        trial_blocks: List[str] = []
        for idx, obs in enumerate(round_observations):
            params = {
                "M": obs.get("M"),
                "ef_construction": obs.get("ef_construction"),
                "ef": obs.get("ef"),
            }
            recall = float(obs.get("recall", 0.0))
            qps = float(obs.get("qps", 0.0))
            max_recall = obs.get("max_recall")
            dc = float(obs.get("dist_comps_per_query", 0.0))
            vn = float(obs.get("visited_nodes_per_query", 0.0))
            dc_vn_ratio = dc / max(vn, 1e-9) if vn > 0 else None

            block = (
                f"### Trial {idx + 1}\n"
                f"- Parameters: M={params['M']}, ef_construction={params['ef_construction']}, "
                f"selected_ef={params['ef']}\n"
                f"- Results: recall={recall:.6f}, QPS={qps:.1f}\n"
                f"- Threshold: τ={threshold}, slack={slack}, "
                f"recall_margin={recall - threshold:+.6f}\n"
                f"- Graph quality indicators:\n"
                f"  - max_recall (highest recall across full ef scan): {max_recall}\n"
                f"  - dist_comps_per_query: {dc:.1f}\n"
                f"  - visited_nodes_per_query: {vn:.1f}\n"
                f"  - dc/vn ratio: {dc_vn_ratio:.1f}" if dc_vn_ratio is not None else f"  - dc/vn ratio: N/A"
                f"\n"
                f"- Cost indicators:\n"
                f"  - build_time_s: {float(obs.get('build_time_s', 0.0)):.1f}\n"
                f"  - index_size_mb: {float(obs.get('index_size_mb', 0.0)):.1f}\n"
            )
            trial_blocks.append(block)

        # ── cross-construction comparison context ──
        cross_lines: List[str] = []
        if len(stage_observations) >= 2:
            cross_lines.append("### Cross-Construction Comparison")
            cross_lines.append(
                "Comparing trials that differ in only one parameter helps isolate "
                "each parameter's effect."
            )
            # Group by build key for structured comparison
            by_m: Dict[int, List[Dict[str, Any]]] = {}
            by_efc: Dict[int, List[Dict[str, Any]]] = {}
            for obs in stage_observations:
                m_val = obs.get("M")
                efc_val = obs.get("ef_construction")
                if isinstance(m_val, (int, float)):
                    by_m.setdefault(int(m_val), []).append(obs)
                if isinstance(efc_val, (int, float)):
                    by_efc.setdefault(int(efc_val), []).append(obs)

            # Same M, different efC
            for m_val, group in sorted(by_m.items()):
                if len(group) >= 2:
                    entries = sorted(group, key=lambda o: int(o.get("ef_construction", 0)))
                    cross_lines.append(
                        f"\nSame M={m_val}, varying efC:"
                    )
                    for o in entries:
                        cross_lines.append(
                            f"  efC={o.get('ef_construction')}, ef*={o.get('ef')}, "
                            f"recall={float(o.get('recall', 0)):.5f}, QPS={float(o.get('qps', 0)):.0f}"
                        )

            # Same efC, different M
            for efc_val, group in sorted(by_efc.items()):
                if len(group) >= 2:
                    entries = sorted(group, key=lambda o: int(o.get("M", 0)))
                    cross_lines.append(
                        f"\nSame efC={efc_val}, varying M:"
                    )
                    for o in entries:
                        cross_lines.append(
                            f"  M={o.get('M')}, ef*={o.get('ef')}, "
                            f"recall={float(o.get('recall', 0)):.5f}, QPS={float(o.get('qps', 0)):.0f}"
                        )

        cross_context = "\n".join(cross_lines) if len(cross_lines) > 1 else ""

        # ── knowledge context ──
        knowledge_text = ""
        if isinstance(knowledge_context, dict):
            knowledge_base = knowledge_context.get("base_knowledge_full", "")
            if isinstance(knowledge_base, str) and knowledge_base.strip():
                # Truncate to avoid blowing up the prompt
                knowledge_text = knowledge_base[:6000]
        elif isinstance(knowledge_context, str):
            knowledge_text = knowledge_context[:6000]

        # ── assemble prompt ──
        prompt = (
            "You are the HNSW tuning attribution analyst.\n\n"
            "Your task: analyze the JUST-EXECUTED trial(s) and attribute the recall\n"
            "result to its ROOT CAUSE. Explain WHY the recall is at its current level\n"
            "and WHERE the bottleneck (or excess) lies.\n\n"
            "## Trial Data (just executed)\n\n"
            + "\n".join(trial_blocks) +
            "\n\n"
        )
        if cross_context:
            prompt += cross_context + "\n\n"

        prompt += (
            "## Attribution Analysis Framework\n\n"
            "Reason through the following steps:\n\n"
            "**Step 1 — Check the structural ceiling (max_recall):**\n"
            "- Look at max_recall: the highest recall achievable by pushing ef to its\n"
            "  maximum in the frontier scan for this (M, ef_construction).\n"
            "- If max_recall is BELOW the threshold τ: the graph STRUCTURE is the\n"
            "  bottleneck — M is too small, connectivity is insufficient. No amount\n"
            "  of ef or efC increase can break through this ceiling. Must increase M.\n"
            "- If max_recall is well ABOVE τ: the graph itself is capable, the\n"
            "  bottleneck (if any) lies elsewhere.\n\n"
            "**Step 2 — Evaluate ef adequacy (if max_recall >= τ):**\n"
            "- Compare the selected ef* to the ef range in the frontier.\n"
            "- If recall at ef* is below τ but max_recall is above τ:\n"
            "  → ef is insufficient. Simply increasing ef can fix the gap.\n"
            "- If recall at ef* is far above τ:\n"
            "  → ef may be excessive. Reducing ef would boost QPS with little\n"
            "    recall loss.\n"
            "- Consider: is recall still rising significantly at ef* or has it\n"
            "  saturated? If saturated, further ef increases bring little benefit.\n\n"
            "**Step 3 — Assess graph density (dist_comps / visited_nodes):**\n"
            "- dc/vn ratio reflects the graph's effective out-degree (~ M/2).\n"
            "- A very LOW ratio means the graph is sparse — each hop examines few\n"
            "  neighbors, search paths are long (high visited_nodes), and the\n"
            "  search may miss optimal paths.\n"
            "- A very HIGH ratio means the graph is dense — each hop is expensive\n"
            "  (many distance computations per node).\n"
            "- visited_nodes alone tells you how many hops search takes — high\n"
            "  visited_nodes with low dc/vn ratio → sparse graph, long paths.\n"
            "- Use the cross-construction data to see how M changes affect both\n"
            "  dc/vn ratio and recall.\n\n"
            "**Step 4 — Cross-construction reasoning:**\n"
            "- Same M, different efC: does higher efC improve recall meaningfully?\n"
            "  If yes → efC investment still pays off. If no → efC is saturated.\n"
            "- Same efC, different M: does higher M improve recall/QPS?\n"
            "  If higher M gives same recall at lower ef* → M investment pays off.\n"
            "- efC/ef ratio: a large ratio alone is NOT a problem (it's the\n"
            "  \"invest in build, save on search\" strategy). Only flag efC as\n"
            "  over-provisioned if cross-efC comparison shows no recall gain.\n\n"
            "**Step 5 — Classify and recommend:**\n"
            "- Synthesize your findings into a root_cause classification.\n"
            "- Propose a concrete recommended action with rationale.\n\n"
        )
        if knowledge_text:
            prompt += (
                "## Domain Knowledge (for reference)\n\n"
                + knowledge_text +
                "\n\n"
            )

        prompt += (
            "## Output Schema\n\n"
            "Return strict JSON matching:\n"
            "```json\n"
            "{\n"
            '  "attributions": [\n'
            '    {\n'
            '      "trial_params": {"M": 0, "ef_construction": 0, "ef": 0},\n'
            '      "recall_status": "below_threshold | above_threshold | near_threshold",\n'
            '      "recall_gap": 0.0,\n'
            '      "root_cause": "graph_quality | ef_insufficient | ef_excessive | graph_too_dense | balanced",\n'
            '      "bottleneck_analysis": "detailed explanation referencing specific metric values",\n'
            '      "confidence": "high | medium | low",\n'
            '      "confidence_reasoning": "why this confidence level",\n'
            '      "diagnostic_signals_triggered": ["S1", "S7"],\n'
            '      "recommended_direction": {\n'
            '        "action": "increase_M | decrease_M | increase_ef | decrease_ef | increase_efC | decrease_efC | increase_M_decrease_ef | fine_tune_ef | ...",\n'
            '        "rationale": "why this action follows from the attribution"\n'
            '      }\n'
            '    }\n'
            '  ],\n'
            '  "summary": {\n'
            '    "primary_root_cause": "...",\n'
            '    "primary_recall_status": "...",\n'
            '    "key_insight": "one-sentence summary of the attribution conclusion",\n'
            '    "trial_count": 1\n'
            '  }\n'
            '}\n'
            '```\n'
        )

        return prompt

    # ── Systematic Diagnostic Section Builder ─────────────────────────────

    def _build_systematic_diag_section(
        self,
        diag_tree: List[Dict[str, Any]],
        last_obs: Dict[str, Any] | None,
        threshold: float,
        root_state: Dict[str, Any],
    ) -> str:
        """Generate the Phase 2 diagnostic tree navigation.

        For each metric in the diagnostic tree, resolve the current value,
        classify its state (too_high / too_low / unknown), and display the
        matching recall_margin branch with the prescribed tuning action.
        """
        if last_obs is None:
            return ("## Phase 2: Diagnostic Tree Navigation\n\n"
                    "_No execution data available — cold start._\n")

        classification = str(root_state.get("classification", "recall_near_threshold"))

        lines = [
            "## Phase 2: Diagnostic Tree Navigation",
            "",
            "For each metric: current value → state → recall_margin → action.",
            "The diagnostic tree provides **prescribed actions** — trust them",
            "unless empirical evidence clearly contradicts.",
            "",
        ]

        # Below-threshold guard: reduction branches are unsafe while infeasible.
        if last_obs is not None:
            try:
                cur_margin = float(last_obs.get("recall", threshold)) - threshold
            except (TypeError, ValueError):
                cur_margin = 0.0
            if cur_margin < 0:
                lines.append(
                    f"**WARNING: current recall is BELOW the threshold "
                    f"(margin={cur_margin:+.4f}). Reduction actions "
                    "(ef↓ / M↓ / efC↓) are unsafe — prefer repair (increase) actions.**"
                )
                lines.append("")

        # Map classification to recall_status key for branch matching
        recall_map = {
            "recall_far_below": "recall_far_below",
            "recall_near_threshold": "recall_near_threshold",
            "recall_far_above": "recall_far_above",
            "cold_start": "recall_far_below",
        }
        target_recall_status = recall_map.get(classification, "recall_near_threshold")

        for node in diag_tree:
            metric_field = str(node.get("metric", ""))
            display = str(node.get("display", metric_field))
            states = node.get("states") if isinstance(node.get("states"), dict) else {}

            # Resolve current value
            value = self._resolve_metric(last_obs, metric_field)

            # Format value
            if value is None:
                value_str = "unavailable"
            elif isinstance(value, float):
                value_str = f"{value:.4f}"
            elif isinstance(value, dict):
                value_str = json.dumps(value, ensure_ascii=False)[:120]
            else:
                value_str = str(value)[:60]

            # Show both branches for every metric — LLM decides which state applies
            lines.append(f"### {display} = {value_str}")
            lines.append("")

            high_branches = states.get("too_high", {}).get("branches", [])
            low_branches = states.get("too_low", {}).get("branches", [])

            if not high_branches and not low_branches:
                lines.append("_No diagnostic branches available._")
                lines.append("")
                continue

            for state_label, branches in [("too_high", high_branches), ("too_low", low_branches)]:
                if not branches:
                    continue
                state_desc = states.get(state_label, {}).get("description", "")
                # Find matching recall_status branch
                match = None
                for b in branches:
                    if b.get("recall_status") == target_recall_status:
                        match = b
                        break
                if match is None:
                    match = branches[0]  # fallback to first branch

                cause = str(match.get("cause", ""))
                action = match.get("action") if isinstance(match.get("action"), dict) else {}
                alt = match.get("alternative") if isinstance(match.get("alternative"), dict) else {}

                lines.append(f"├── **If {state_label}**: {state_desc}")
                lines.append(f"│   └── Cause: {cause}")
                actions = match.get("actions") if isinstance(match.get("actions"), list) else []
                if not actions:
                    # backward compat: single action
                    act = match.get("action") if isinstance(match.get("action"), dict) else {}
                    if act and act.get("parameter") != "none":
                        actions = [act]
                for act in actions:
                    lines.append(f"│       └── **{act.get('parameter', '?')} {act.get('direction', '?')}**")
                lines.append("")

        lines.append("---")
        lines.append(f"**Classification**: `{classification}`")
        lines.append("**Instructions**: The prescribed actions above are based on the diagnostic tree.")
        lines.append("Collect all actions. If they conflict, resolve in Phase 3 Step 2 using empirical data.")
        lines.append("")

        return "\n".join(lines)

    # ── Diagnostic Last-Execution Reasoning ────────────────────────────────

    def repropose_after_proposal_check(
        self,
        *,
        check_feedback: str,
        allowed_values_override: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Ask the LLM to revise a proposal flagged by the posterior checker.

        Parameters
        ----------
        check_feedback : str
            The ``format_check_result_for_llm`` text explaining why the
            previous proposal was flagged.
        allowed_values_override : dict or None
            Optional frozen search-space constraints; a revision outside
            them is rejected (returns ``None``).

        Returns
        -------
        dict or None
            ``{"params": {"M", "ef_construction", "ef"}, "rationale": str}``
            or ``None`` when the LLM call / parse / validation fails.
        """
        param_space_lines: List[str] = []
        for name in PARAM_ORDER:
            domain = self.space.domains[name]
            if allowed_values_override and name in allowed_values_override:
                override = allowed_values_override[name]
                if isinstance(override, dict):
                    param_space_lines.append(f"  {name}: {json.dumps(override)}  # constrained")
                    continue
            param_space_lines.append(f"  {name}: {json.dumps(domain.to_spec())}")

        prompt = (
            "Your previously proposed HNSW configuration was checked against "
            "the interval evidence of configurations already evaluated in this "
            "task. The posterior checker flagged it:\n\n"
            f"{check_feedback}\n\n"
            "Re-propose ONE corrected configuration that addresses this "
            "feedback. Requirements:\n"
            "- Stay within the parameter space below.\n"
            "- ef must NOT exceed ef_construction.\n"
            "- Do NOT repeat a configuration that was already executed.\n"
            "- You may adopt the suggested revision, but it is based on "
            "interval evidence only and is not guaranteed optimal.\n\n"
            "## Parameter Space\n"
            + "\n".join(param_space_lines) + "\n\n"
            'Return strict JSON: {"M": <int>, "ef_construction": <int>, '
            '"ef": <int>, "rationale": "<explain>"}'
        )
        result = self._call_reasoning_json("hnsw_posterior_recheck", prompt)
        parsed = result.get("parsed") if result.get("ok") else None
        if not isinstance(parsed, dict):
            return None
        try:
            params = {
                "M": int(parsed["M"]),
                "ef_construction": int(parsed["ef_construction"]),
                "ef": int(parsed["ef"]),
            }
            # Range overrides clamp (same semantics as the diagnose hard
            # bound clamp); out-of-set discrete values raise.
            canonical = self.canonicalize(params, domain_constraints=allowed_values_override)
        except (KeyError, TypeError, ValueError):
            return None
        if allowed_values_override:
            fields = self.space.out_of_constraint_fields(canonical, allowed_values_override)
            if fields:
                return None
        return {
            "params": canonical,
            "rationale": str(parsed.get("rationale", "")),
        }

    def diagnose_last_execution(
        self,
        *,
        last_trial: Dict[str, Any] | None,
        root_state: Dict[str, Any],
        stage_policy: Dict[str, Any],
        allowed_values_override: Dict[str, Any] | None = None,
        historical_context: str = "",
        cold_start_cards: List[Dict[str, Any]] | None = None,
        memory_context: Dict[str, Any] | None = None,
        static_knowledge_context: str = "",
        proposal_check_feedback: str = "",
        interval_table_context: str = "",
        force_construction: bool = False,
        excluded_construction_pairs: Optional[set] = None,
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
        """Diagnose the last executed trial and propose the next configuration.

        This is the core diagnostic loop: the LLM receives the most recently
        executed configuration's full metrics, the current optimization state,
        the success memory, and the diagnostic tree, then outputs a diagnosis
        and exactly one candidate configuration for the next round.

        Returns ``(candidate_dict, diagnostic_log)`` where *candidate_dict* is
        ``{"params": {...}, "source": "diagnostic", "note": "..."}`` or
        ``None`` if the LLM call failed.
        """
        empty_log: Dict[str, Any] = {
            "ok": False,
            "diagnosis": {},
            "candidate": None,
            "error": "",
            "source": "none",
        }

        # Reset per-phase compliance counters
        self.compliance_prompt_tokens = 0
        self.compliance_completion_tokens = 0
        self.compliance_elapsed_s = 0.0

        if not self.enable_agentic:
            empty_log["error"] = "agentic_disabled"
            return None, empty_log

        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy.get("recall_slack", 0.0))

        # ── build last-execution observation ──
        last_obs: Dict[str, Any] | None = None
        if last_trial is not None and last_trial.get("status") == "success":
            metrics = last_trial.get("metrics") or {}
            params = last_trial.get("params") or {}
            if isinstance(metrics, dict) and isinstance(params, dict):
                _max_recall = None
                frontier_summary = metrics.get("frontier_summary") or {}
                if isinstance(frontier_summary, dict) and frontier_summary.get("max_recall") is not None:
                    _max_recall = float(frontier_summary["max_recall"])
                last_obs = {
                    "M": params.get("M"),
                    "ef_construction": params.get("ef_construction"),
                    "ef": metrics.get("selected_ef", params.get("ef")),
                    "recall": float(metrics.get("recall", 0.0)),
                    "qps": float(metrics.get("qps", 0.0)),
                    "feasible": float(metrics.get("recall", 0.0)) >= threshold,
                    "recall_margin": float(metrics.get("recall", 0.0)) - threshold,
                    "max_recall": _max_recall,
                    "build_time_s": float(metrics.get("build_time_s", 0.0)),
                    "index_size_mb": float(metrics.get("index_size_mb", 0.0)),
                    "dist_comps_per_query": float(metrics.get("dist_comps_per_query", 0.0)),
                    "visited_nodes_per_query": float(metrics.get("visited_nodes_per_query", 0.0)),
                    "out_degree_mean": float(metrics.get("out_degree_mean", 0.0)) if metrics.get("out_degree_mean") is not None else None,
                    "in_degree_mean": float(metrics.get("in_degree_mean", 0.0)) if metrics.get("in_degree_mean") is not None else None,
                    "in_degree_std": float(metrics.get("in_degree_std", 0.0)) if metrics.get("in_degree_std") is not None else None,
                    "in_degree_max": int(metrics.get("in_degree_max", 0)) if metrics.get("in_degree_max") is not None else None,
                    "candidate_distance_stats": metrics.get("candidate_distance_stats", {}),
                    "selected_ef": metrics.get("selected_ef"),
                }

        best_feasible = root_state.get("current_best_feasible")
        closest = root_state.get("closest_to_feasible")

        # ── build cross-construction comparison table ──
        trial_obs = root_state.get("trial_observations") if isinstance(root_state.get("trial_observations"), list) else []
        cross_table_lines: List[str] = []
        if trial_obs:
            cross_table_lines.append("| M | efC | ef* | recall | QPS | margin | max_recall | visited | dist_comp | out_deg |")
            cross_table_lines.append("|---|-----|-----|--------|-----|--------|------------|---------|-----------|---------|")
            for obs in trial_obs:
                if not isinstance(obs, dict):
                    continue
                margin_val = obs.get("recall_margin")
                margin_str = f"{margin_val:+.4f}" if isinstance(margin_val, (int, float)) else "?"
                cross_table_lines.append(
                    f"| {obs.get('M','?')} | {obs.get('ef_construction','?')} | {obs.get('selected_ef', obs.get('ef','?'))} | "
                    f"{obs.get('recall',0):.4f} | {obs.get('qps',0):.1f} | {margin_str} | "
                    f"{obs.get('max_recall',0):.4f} | {obs.get('visited_nodes_per_query',0):.1f} | "
                    f"{obs.get('dist_comps_per_query',0):.1f} | {obs.get('out_degree_mean','?')} |"
                )

        # ── build diagnostic tree summary for prompt ──
        diag_tree_summary: List[str] = []
        for node in DIAGNOSTIC_TREE:
            diag_tree_summary.append(
                f"- **{node.get('node_id', '?')}** [{node.get('problem', '?')}] "
                f"metrics: {node.get('metrics', '?')} "
                f"→ {node.get('solution', '')}"
            )

        # ── build parameter space summary ──
        param_space_lines: List[str] = []
        for name in PARAM_ORDER:
            domain = self.space.domains[name]
            if allowed_values_override and name in allowed_values_override:
                override = allowed_values_override[name]
                if isinstance(override, dict):
                    param_space_lines.append(f"  {name}: {json.dumps(override)}  # constrained")
                    continue
            param_space_lines.append(f"  {name}: {json.dumps(domain.to_spec())}")

        # ── cold start guidance ──
        cold_start_guidance = ""
        if not trial_obs:
            if cold_start_cards:
                lines = [
                    "## Cold Start — Insight-Based Initialization",
                    "",
                    "No prior executions exist. Use the following insight cards from",
                    f"similar tasks (filtered to τ={threshold:.4f}) to choose your first",
                    "configuration. Each card provides a recommended parameter region.",
                    "",
                    "| Card | Region | Elite Median | Feasible Median |",
                    "|------|--------|-------------|-----------------|",
                ]
                for card in cold_start_cards[:5]:
                    cid = str(card.get("card_id", ""))[-20:]
                    region = str(card.get("region_desc", ""))[:60]
                    elite = dict(card.get("elite_median") or {})
                    feasible = dict(card.get("feasible_median") or {})
                    # Clamp ef to efC (scan-mode data may have ef > efC)
                    for d in (elite, feasible):
                        if "ef" in d and "ef_construction" in d:
                            d["ef"] = min(d["ef"], d["ef_construction"])
                    elite_str = ", ".join(f"{k}={v}" for k, v in elite.items()) if elite else "—"
                    feasible_str = ", ".join(f"{k}={v}" for k, v in feasible.items()) if feasible else "—"
                    lines.append(f"| {cid} | {region} | {elite_str} | {feasible_str} |")
                lines.append("")
                lines.append("**Use the elite median values as your starting point.**")
                lines.append("If multiple cards agree on the same values, that is a strong signal.")
                lines.append("**Constraint: ef must be ≤ ef_construction.**")
                lines.append("")
                cold_start_guidance = "\n".join(lines) + "\n\n"
            else:
                cold_start_guidance = (
                    "## Cold Start Guidance\n"
                    "This is the FIRST round — no prior executions exist.\n"
                    f"- For recall target τ={threshold:.4f}, start with M ≥ 12.\n"
                    "- Choose ef_construction ≥ 2× M for good edge quality.\n"
                    "- Choose a moderate ef to establish a recall baseline.\n\n"
                )

        # ── build systematic diagnostic section ──
        systematic_diag = self._build_systematic_diag_section(
            diag_tree=DIAGNOSTIC_TREE,
            last_obs=last_obs,
            threshold=threshold,
            root_state=root_state,
        )

        # ── build parameter space summary ──
        param_space_lines: List[str] = []
        for name in PARAM_ORDER:
            domain = self.space.domains[name]
            if allowed_values_override and name in allowed_values_override:
                override = allowed_values_override[name]
                if isinstance(override, dict):
                    param_space_lines.append(f"  {name}: {json.dumps(override)}  # constrained")
                    continue
            param_space_lines.append(f"  {name}: {json.dumps(domain.to_spec())}")

        # ── cold start guidance ──
        cold_start_guidance = ""
        if not trial_obs:
            if cold_start_cards:
                lines = [
                    "## Cold Start — Insight-Based Initialization",
                    "",
                    "No prior executions exist. Use the following insight cards from",
                    f"similar tasks (filtered to τ={threshold:.4f}) to choose your first",
                    "configuration. Each card provides a recommended parameter region.",
                    "",
                    "| Card | Region | Elite Median | Feasible Median |",
                    "|------|--------|-------------|-----------------|",
                ]
                for card in cold_start_cards[:5]:
                    cid = str(card.get("card_id", ""))[-20:]
                    region = str(card.get("region_desc", ""))[:60]
                    elite = dict(card.get("elite_median") or {})
                    feasible = dict(card.get("feasible_median") or {})
                    # Clamp ef to efC (scan-mode data may have ef > efC)
                    for d in (elite, feasible):
                        if "ef" in d and "ef_construction" in d:
                            d["ef"] = min(d["ef"], d["ef_construction"])
                    elite_str = ", ".join(f"{k}={v}" for k, v in elite.items()) if elite else "—"
                    feasible_str = ", ".join(f"{k}={v}" for k, v in feasible.items()) if feasible else "—"
                    lines.append(f"| {cid} | {region} | {elite_str} | {feasible_str} |")
                lines.append("")
                lines.append("**Use the elite median values as your starting point.**")
                lines.append("If multiple cards agree on the same values, that is a strong signal.")
                lines.append("**Constraint: ef must be ≤ ef_construction.**")
                lines.append("")
                cold_start_guidance = "\n".join(lines) + "\n\n"
            else:
                cold_start_guidance = (
                    "## Cold Start Guidance\n"
                    "This is the FIRST round — no prior executions exist.\n"
                    f"- For recall target τ={threshold:.4f}, start with M ≥ 12.\n"
                    "- Choose ef_construction ≥ 2× M for good edge quality.\n"
                    "- Choose a moderate ef to establish a recall baseline.\n\n"
                )

        # ── build prompt ──
        classification = root_state.get("classification", "cold_start")
        stage_strategy = root_state.get("stage_strategy", "")

        # ── recall-margin guidance: how much room is there to move ──
        margin_guidance = ""
        if last_obs is not None:
            try:
                rec = float(last_obs.get("recall", 0) or 0)
                m = rec - threshold
                if m > 0.001:
                    margin_guidance = (
                        f"Recall margin: {m:+.4f} — LARGE headroom. "
                        "Analyze the current state, select the matching knowledge "
                        "(matched cards, diagnostic tree, interval table, cross table), "
                        "and let the evidence decide BOTH direction and magnitude of "
                        "your adjustment. Estimate the recall-vs-ef slope from the "
                        "observed (efS, recall) pairs in the CURRENT-TASK interval table "
                        "(other tasks' tables must NOT be used) and take "
                        "the LARGEST ef reduction that the slope evidence says keeps "
                        "recall >= tau (no extra buffer). Do not take needlessly "
                        "small steps when the observed slope is shallow."
                    )
                elif m >= -0.001:
                    margin_guidance = (
                        f"Recall margin: {m:+.4f} — close to the threshold. "
                        "Derive small, evidence-based adjustments from the knowledge: "
                        "keep efS > L, and use the U-probe rule of the interval table."
                    )
                else:
                    margin_guidance = (
                        f"Recall margin: {m:+.4f} — below the threshold. "
                        "Repair direction REQUIRED — do NOT decrease ef / M / efC "
                        "while below the threshold."
                    )
            except (TypeError, ValueError):
                margin_guidance = ""

        # ── compute hard step-size bounds from last execution ──
        step_bounds_text = ""
        if last_obs and isinstance(last_obs, dict):
            step_bounds_text = ""

        prompt_payload = {
            "task": "diagnose_last_execution",
            "recall_threshold": threshold,
            "recall_slack": slack,
            "classification": classification,
            "optimization_stage": classification,
            "current_best_feasible": best_feasible,
            "closest_to_feasible": closest,
            "last_execution": last_obs,
            "all_executed_params": [
                {"M": trial.get("M"), "ef_construction": trial.get("ef_construction"), "ef": trial.get("ef")}
                for trial in root_state.get("trial_observations", [])
                if isinstance(trial, dict) and trial.get("M") is not None
            ] if isinstance(root_state.get("trial_observations"), list) else [],
            "historical_context": historical_context if historical_context else "(none)",
            "proposal_check_feedback": proposal_check_feedback if proposal_check_feedback else "(none)",
            "interval_table_context": (
                interval_table_context if interval_table_context else "(none)"
            ),
        }


        prompt = (
            "You are an HNSW tuning diagnostician. Follow the structured flow below.\n\n"
            f"{cold_start_guidance}"
            + (
                # ── Selected Static Knowledge (mechanism-level priors) ──
                f"{static_knowledge_context}\n"
                if static_knowledge_context else ""
            )
            + "## Context\n"
            f"RECALL THRESHOLD τ = {threshold:.4f}  (slack = {slack:.4f})\n"
            f"Current Classification: **{classification}**\n"
            f"Strategy Direction: {stage_strategy}\n"
            + (margin_guidance + "\n" if margin_guidance else "")
            + f"Current Best Feasible: {json.dumps(best_feasible, ensure_ascii=False) if best_feasible else 'none'}\n\n"
            + (
                # ── Current Task Memory (structured dual-view evidence) ──
                _format_memory_context_section(memory_context) + "\n"
                if memory_context else ""
            )
            + (
                # ── Runtime Structural Interval Table (current task) ──
                f"## Runtime Structural Interval Table (current task)\n"
                f"{interval_table_context}\n\n"
                if interval_table_context else ""
            )
            + (
                # ── Posterior Proposal Check (previous round) ──
                f"## Posterior Proposal Check (previous round)\n"
                f"{proposal_check_feedback}\n\n"
                if proposal_check_feedback else ""
            )
            + "## Last Execution — Detailed Metrics\n"
            f"{json.dumps(last_obs, ensure_ascii=False, indent=2) if last_obs else '(no trial executed yet)'}\n\n"
            + ("## All Executed Configurations (cross-construction comparison)\n"
               + "\n".join(cross_table_lines) + "\n\n"
               "Compare rows: same efC, different M → isolate M effect. Same M, different efC → isolate efC effect.\n"
               "CRITICAL: if M↓ forces ef*↑ and QPS↓, this validates Signal S8 — do NOT decrease M further.\n\n"
               if cross_table_lines else "")
            + systematic_diag + "\n"
            "## Parameter Space\n"
            + "\n".join(param_space_lines) + "\n\n"
            "## Already Executed (M, ef_construction, ef) — do NOT repeat\n"
            f"{json.dumps(prompt_payload['all_executed_params'], ensure_ascii=False)}\n\n"
            "## Phase 1: Classification (done)\n\n"
            f"The current classification is **{classification}**. Use this to guide which\n"
            "diagnostic signals you prioritize:\n"
            f"- **recall_far** (|margin| > {self._near_boundary(threshold):.3f}): AGGRESSIVE.\n"
            "  margin > 0 → reduce params for QPS (any of ef / M / efC indicated by evidence).\n"
            "  margin < 0 → repair: try ef↑ first; use M↑ only if the same graph repeatedly fails across ef values.\n"
            f"- **recall_near** (|margin| ≤ {self._near_boundary(threshold):.3f}): CONSERVATIVE.\n"
            "  Prefer single-axis moves; M and ef may be adjusted together when evidence supports both.\n"
            "- **cold_start**: Use the insight cards above as your starting point.\n"
            "  Start at the elite median values, not at conservative lower bounds.\n\n"
            "## Phase 2: How to Use the Diagnostic Tree\n\n"
            "The diagnostic tree above prescribes actions based on each metric's state\n"
            "and your current recall_margin status. For each metric:\n\n"
            "1. If the **state is already determined** (too_high / too_low), the\n"
            "   prescribed action is shown. Trust it unless empirical data contradicts.\n"
            "2. If the **state is unknown**, evaluate the value manually using the\n"
            "   displayed knowledge (both too_high and too_low branches are shown).\n"
            "3. Collect all prescribed actions. Note which parameter each action targets\n"
            "   and which direction it recommends.\n"
            "4. **You MUST evaluate ALL 6 metrics.** Report your state/action decision\n"
            "   for every metric in metric_actions, even if the action is no_change.\n"
            "   The metrics are: visited_nodes_per_query, dist_comps_per_query,\n"
            "   index_size_mb, out_degree_mean, in_degree_distribution,\n"
            "   candidate_distance_stats.\n"
            "5. **Consult the History lines** under each branch. If the same state\n"
            "   previously led to QPS improvement, that direction is validated.\n"
            "   If it previously led to QPS regression, consider the opposite.\n\n"
            "Conflicting actions (e.g., one says increase M, another says decrease M)\n"
            "must be resolved in Phase 3 Step 2 using the empirical evidence table.\n\n"
            "## Phase 3: Synthesis and Proposal\n\n"
            "### Step 1 — Aggregate findings from Phase 2\n"
            "Collect all nodes where action_needed=true. These are your primary signals.\n"
            "- If multiple signals point to the same parameter, note the consensus.\n"
            "- If signals conflict (e.g., one says increase M, another says decrease M),\n"
            "  resolve by checking the cross-construction comparison table above:\n"
            "  what did the DATA actually show when that parameter was changed?\n\n"
            "### Step 2 — Anchor against empirical evidence\n"
            "The cross-construction comparison table shows what ACTUALLY happened in\n"
            "previous trials. Before finalizing your proposal:\n"
            "- Which (M, efC) combination achieved the highest QPS while feasible?\n"
            "- Does the direction suggested by the diagnostic signals align with\n"
            "  what the empirical data shows?\n"
            "- **If the signals and the data disagree, trust the DATA.**\n"
            "- **HISTORY CHECK**: Look at the History lines under each metric branch.\n"
            "  If the SAME actions repeatedly led to QPS LOSS (negative qps_delta)\n"
            "  under the current classification, you MUST try the OPPOSITE direction:\n"
            "  - If M↑ kept losing QPS → try M↓\n"
            "  - If efC↑ kept losing QPS → try efC↓\n"
            "  - If ef↓ kept losing QPS → try ef↑\n"
            "- If the best feasible configuration was found at a lower M and your\n"
            "  proposal moves to a higher M, the expected QPS gain must outweigh\n"
            "  the risk of regression away from the known best region.\n\n"
            "### Step 3 — Propose next configuration\n"
            "Based on your Phase 2 diagnosis AND the empirical evidence:\n"
            "1. Select the 2-4 signals most relevant given the current classification.\n"
            "2. Propose exactly 1 next (M, ef_construction, ef).\n"
            "3. In the rationale, cite which diagnostic node(s) and which knowledge\n"
            "   from the Notes column led to this proposal.\n\n"
            + step_bounds_text
            + (
                "**CONSTRUCTION-PROBE ROUND**: your candidate MUST change BOTH M "
                "and efC from the last execution, and must NOT use a banned pair. "
                "Choose the new (M, efC) based on the available knowledge (diagnostic "
                "tree directions, interval table rows/columns, matched cards) — not "
                "arbitrarily.\n\n"
                if force_construction else ""
            )
            + (
                "**BANNED (M, efC) pairs** (explored ≥3 times — do NOT propose): "
                + json.dumps(
                    sorted(
                        [[int(m), int(efc)] for (m, efc) in excluded_construction_pairs]
                    ),
                    ensure_ascii=False,
                )
                + "\n"
                if excluded_construction_pairs else ""
            )
            + "Important:\n"
            "- Stay within parameter space. Do NOT repeat already-executed configs.\n"
            "  If margin > 0: reduce parameters for QPS (any of ef / M / efC; ef changes are zero-rebuild, M/efC changes rebuild).\n"
            "  If margin < 0: try a modest ef increase first; increase M only if the same (M, efC) repeatedly fails to reach τ (max_recall in direct mode is not a structural ceiling).\n"
            "- **EF STEP LIMIT**: |Δef| ≤ 10 when near the boundary (|margin| ≤ near), ≤ 150 when far. M and efC have no step limit.\n"
            "- **PARAMETER CHANGE RULE**: If tuning_action says increase/decrease,\n"
            "  the candidate MUST show a different numeric value from the last\n"
            "  execution. If last M=18 and you say M↑, candidate M must be >18.\n"
            "  If last efC=220 and you say efC↑, candidate efC must be >220.\n"
            "  All three parameters must show actual numeric changes when their\n"
            "  direction is not no_change.\n"
            "- **ef ≤ efC**: ef must NOT exceed ef_construction.\n"
            "- Return strict JSON only:\n"
            "{\n"
            '  "classification": "' + classification + r'",\n'
            '  "diagnosis": {\n'
            '    "recall_status": "below_threshold | near_threshold | above_threshold",\n'
            '    "metric_actions": [\n'
            '      {"metric": "visited_nodes_per_query", "state": "too_low", "actions": [{"parameter": "ef", "direction": "decrease"}]},\n'
            '      {"metric": "dist_comps_per_query",    "state": "too_high","actions": [{"parameter": "M","direction": "decrease"}]},\n'
            '      ... (one entry per metric; state MUST be \"too_high\", \"too_low\", or \"normal\"; copy actions from the tree above)\n'
            "    ],\n"
            '    "key_findings": ["..."],\n'
            '    "strategy": "one sentence"\n'
            "  },\n"
            '  "tuning_action": {\n'
            '    "M": "increase | decrease | no_change",\n'
            '    "ef_construction": "increase | decrease | no_change",\n'
            '    "ef": "increase | decrease | no_change"\n'
            "  },\n"
            '  "candidate": {\n'
            '    "M": 16,\n'
            '    "ef_construction": 200,\n'
            '    "ef": 40,\n'
            '    "rationale": "reference signals and data"\n'
            "  }\n"
            "}\n"
        )

        result = self._call_reasoning_json("hnsw_diagnose_last", prompt)

        if not result.get("ok") or not isinstance(result.get("parsed"), dict):
            return None, {
                "ok": False,
                "diagnosis": {},
                "candidate": None,
                "error": result.get("error", "llm_call_failed"),
                "source": "llm_error",
                "prompt_payload": prompt_payload,
            }

        parsed = result["parsed"]
        diagnosis = parsed.get("diagnosis") or {}
        candidate_raw = parsed.get("candidate") or {}

        # ── extract new fields for memory recording ──
        llm_classification = str(parsed.get("classification", classification))
        tuning_action = parsed.get("tuning_action") or {}
        if not isinstance(tuning_action, dict):
            tuning_action = {}
        metric_actions = diagnosis.get("metric_actions") or []
        if not isinstance(metric_actions, list):
            metric_actions = []

        # ── validate candidate ──
        try:
            candidate_params = {
                "M": int(candidate_raw["M"]),
                "ef_construction": int(candidate_raw["ef_construction"]),
                "ef": int(candidate_raw["ef"]),
            }
            canonical = self.canonicalize(candidate_params)

            # ── MINIMUM step-size compliance check (enforce lower bounds) ──
            if last_obs and isinstance(last_obs, dict):
                last_M = last_obs.get("M")
                last_efC = last_obs.get("ef_construction")
                last_ef = last_obs.get("ef") or last_obs.get("selected_ef")
                if last_M and last_efC and last_ef:
                    margin_c = float(last_obs.get("recall", threshold)) - threshold
                    near_b = self._near_boundary(threshold)
                    compliant, violations = self._check_step_compliance(
                        {"M": last_M, "ef_construction": last_efC, "ef": last_ef},
                        canonical,
                        margin_c,
                        near_boundary=near_b,
                        force_construction=force_construction,
                        excluded_construction_pairs=excluded_construction_pairs,
                    )
                    if not compliant:
                        tier_c = "NEAR" if abs(margin_c) <= near_b else "FAR"
                        direction_hint = (
                            "margin>0 → REDUCE params for QPS (any of ef / M / efC).\n"
                            if margin_c > 0 else
                            "margin<0 → INCREASE params for recall repair (any of ef / M / efC indicated by evidence).\n"
                        )
                        checker_prompt = (
                            "STEP-SIZE COMPLIANCE CHECK FAILED\n"
                            f"Your proposal: M={canonical['M']}, efC={canonical['ef_construction']}, ef={canonical['ef']}\n"
                            f"Last params: M={last_M}, efC={last_efC}, ef={last_ef}\n"
                            f"Tier: {tier_c} (margin={margin_c:+.4f})\n"
                            + direction_hint
                            + "Violations:\n"
                            + "".join(f"  - {v}\n" for v in violations)
                            + "Please re-propose a corrected candidate satisfying these requirements.\n"
                            + 'Return JSON: {"M": <int>, "ef_construction": <int>, "ef": <int>, "rationale": "<explain>"}'
                        )
                        for _ in range(2):
                            cr = self._call_reasoning_json("hnsw_step_compliance", checker_prompt)
                            self.compliance_prompt_tokens += self.last_skill_prompt_tokens
                            self.compliance_completion_tokens += self.last_skill_completion_tokens
                            self.compliance_elapsed_s += self.last_call_elapsed_s
                            if cr.get("ok") and isinstance(cr.get("parsed"), dict):
                                try:
                                    rp = {
                                        "M": int(cr["parsed"]["M"]),
                                        "ef_construction": int(cr["parsed"]["ef_construction"]),
                                        "ef": int(cr["parsed"]["ef"]),
                                    }
                                    rc = self.canonicalize(rp)
                                    c2, _ = self._check_step_compliance(
                                        {"M": last_M, "ef_construction": last_efC, "ef": last_ef}, rc, margin_c,
                                        near_boundary=near_b,
                                        force_construction=force_construction,
                                        excluded_construction_pairs=excluded_construction_pairs)
                                    if c2:
                                        candidate_raw["rationale"] = str(cr["parsed"].get("rationale", "")) + " [COMPLIANCE CHECK PASSED]"
                                        canonical = rc
                                        break
                                    checker_prompt = (
                                        f"STILL NOT COMPLIANT. Retry M={rc['M']}, efC={rc['ef_construction']}, ef={rc['ef']}. "
                                        + ("margin>0: REDUCE params for QPS (any of ef / M / efC). "
                                           if margin_c > 0 else
                                           "margin<0: INCREASE params for recall (any of ef / M / efC indicated by evidence). ")
                                        + "Make LARGER changes in the correct direction.\n"
                                        + 'Return JSON: {"M": <int>, "ef_construction": <int>, "ef": <int>, "rationale": "<explain>"}'
                                    )
                                except Exception:
                                    break
                            else:
                                break
                        else:
                            candidate_raw["rationale"] = candidate_raw.get("rationale", "") + " [COMPLIANCE WARNING: " + "; ".join(violations) + "]"

            # ── step-size enforcement: clamp to allowed bounds ──
            if last_obs and isinstance(last_obs, dict):
                last_M = last_obs.get("M")
                last_efC = last_obs.get("ef_construction")
                last_ef = last_obs.get("ef") or last_obs.get("selected_ef")
                if last_M and last_efC and last_ef:
                    # Full parameter-space bounds (no step-size restriction)
                    dM = self.space.domains["M"]
                    dE = self.space.domains["ef_construction"]
                    dF = self.space.domains["ef"]
                    M_min_s, M_max_s = int(dM.min_value), int(dM.max_value)
                    efC_min_s, efC_max_s = int(dE.min_value), int(dE.max_value)
                    ef_min_s, ef_max_s = int(dF.min_value), int(dF.max_value)

                    clamped = dict(canonical)
                    warnings = []
                    for param, val, lo, hi in [
                        ("M", canonical["M"], M_min_s, M_max_s),
                        ("ef_construction", canonical["ef_construction"], efC_min_s, efC_max_s),
                        ("ef", canonical["ef"], ef_min_s, ef_max_s),
                    ]:
                        if val < lo:
                            warnings.append(f"{param}={val} below bound {lo}, clamped to {lo}")
                            clamped[param] = lo
                        elif val > hi:
                            warnings.append(f"{param}={val} above bound {hi}, clamped to {hi}")
                            clamped[param] = hi

                    if warnings:
                        # ── LLM re-proposal: ask LLM to fix its own violation ──
                        correction_prompt = (
                            "YOUR PROPOSAL WAS REJECTED due to step-size violations:\n"
                            + "\n".join(f"  - {w}" for w in warnings) + "\n"
                            f"Current classification: {classification}, |margin|={abs_margin:.4f}\n"
                            f"Allowed ranges: M∈[{M_min_s},{M_max_s}], ef_construction∈[{efC_min_s},{efC_max_s}], ef∈[{ef_min_s},{ef_max_s}]\n"
                            "Please re-propose a corrected candidate within these bounds.\n"
                            "Return ONLY the corrected JSON:\n"
                            '{"M": <int>, "ef_construction": <int>, "ef": <int>, "rationale": "<corrected rationale>"}'
                        )
                        retry_result = self._call_reasoning_json("hnsw_step_check", correction_prompt)
                        if retry_result.get("ok") and isinstance(retry_result.get("parsed"), dict):
                            retry = retry_result["parsed"]
                            try:
                                retry_params = {
                                    "M": int(retry["M"]),
                                    "ef_construction": int(retry["ef_construction"]),
                                    "ef": int(retry["ef"]),
                                }
                                retry_canonical = self.canonicalize(retry_params)
                                # Check if retry is now within bounds
                                retry_ok = all(
                                    lo <= retry_canonical[p] <= hi
                                    for p, lo, hi in [
                                        ("M", M_min_s, M_max_s),
                                        ("ef_construction", efC_min_s, efC_max_s),
                                        ("ef", ef_min_s, ef_max_s),
                                    ]
                                )
                                if retry_ok:
                                    candidate_raw["rationale"] = str(retry.get("rationale", "")) + " [RE-PROPOSED after step check]"
                                    canonical = retry_canonical
                                else:
                                    # Still violated — clamp and note
                                    clamped2 = dict(retry_canonical)
                                    for param, lo, hi in [
                                        ("M", M_min_s, M_max_s),
                                        ("ef_construction", efC_min_s, efC_max_s),
                                        ("ef", ef_min_s, ef_max_s),
                                    ]:
                                        clamped2[param] = max(lo, min(hi, clamped2[param]))
                                    clamped2 = self.canonicalize(clamped2)
                                    candidate_raw["rationale"] = str(retry.get("rationale", "")) + " [CLAMPED after re-proposal failure]"
                                    canonical = clamped2
                            except Exception:
                                # Retry parsing failed — fall through to clamp
                                clamped_final = dict(canonical)
                                for param, lo, hi in [
                                    ("M", M_min_s, M_max_s),
                                    ("ef_construction", efC_min_s, efC_max_s),
                                    ("ef", ef_min_s, ef_max_s),
                                ]:
                                    clamped_final[param] = max(lo, min(hi, clamped_final[param]))
                                clamped_final = self.canonicalize(clamped_final)
                                candidate_raw["rationale"] = candidate_raw.get("rationale", "") + " [STEP-CLAMPED: " + "; ".join(warnings) + "]"
                                canonical = clamped_final
                        else:
                            # LLM retry failed — clamp
                            clamped_final = dict(canonical)
                            for param, lo, hi in [
                                ("M", M_min_s, M_max_s),
                                ("ef_construction", efC_min_s, efC_max_s),
                                ("ef", ef_min_s, ef_max_s),
                            ]:
                                clamped_final[param] = max(lo, min(hi, clamped_final[param]))
                            clamped_final = self.canonicalize(clamped_final)
                            candidate_raw["rationale"] = candidate_raw.get("rationale", "") + " [STEP-CLAMPED: " + "; ".join(warnings) + "]"
                            canonical = clamped_final

            if allowed_values_override:
                fields = self.space.out_of_constraint_fields(canonical, allowed_values_override)
                if fields:
                    return None, {
                        "ok": False,
                        "diagnosis": diagnosis,
                        "candidate": None,
                        "error": f"candidate out of allowed override: {fields}",
                        "source": "validation_error",
                        "prompt_payload": prompt_payload,
                    }
        except Exception as exc:
            return None, {
                "ok": False,
                "diagnosis": diagnosis,
                "candidate": None,
                "error": f"invalid candidate params: {exc}",
                "source": "validation_error",
                "prompt_payload": prompt_payload,
            }

        candidate = {
            "params": canonical,
            "source": "diagnostic",
            "note": str(candidate_raw.get("rationale", "")),
        }

        diag_log = {
            "ok": True,
            "diagnosis": diagnosis,
            "candidate": candidate,
            "classification": llm_classification,
            "tuning_action": tuning_action,
            "metric_actions": metric_actions,
            "error": "",
            "source": "llm",
            "prompt_payload": prompt_payload,
        }
        return candidate, diag_log

    def analyze_trial_attribution(
        self,
        *,
        round_trials: Sequence[Dict[str, Any]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        knowledge_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Post-execution attribution analysis for just-executed trials.

        Aggregates the round's trial results into structured observations,
        builds an attribution prompt for the LLM, and returns a structured
        attribution dict explaining WHY each trial achieved its recall level.

        Returns a dict with keys ``ok``, ``attributions``, ``summary``,
        ``source``, and optionally ``error``.
        """
        empty_return: Dict[str, Any] = {
            "ok": False,
            "error": "",
            "attributions": [],
            "summary": {},
            "source": "none",
        }

        if not self.enable_agentic:
            empty_return["error"] = "agentic_disabled"
            return empty_return

        # ── filter to successful trials ──
        success_trials = [t for t in round_trials if t.get("status") == "success"]
        if not success_trials:
            empty_return["error"] = "no_successful_trials"
            return empty_return

        # ── aggregate into observation points ──
        threshold = float(stage_policy["recall_threshold"])
        round_points = self._aggregate_success_points(success_trials)
        if not round_points:
            empty_return["error"] = "no_aggregated_points"
            return empty_return

        round_observations = self._trial_observations(
            points=round_points,
            threshold=threshold,
        )
        if not round_observations:
            empty_return["error"] = "no_observations"
            return empty_return

        # ── build stage-level observations for cross-construction context ──
        stage_success = [t for t in stage_trials if t.get("status") == "success"]
        stage_points = self._aggregate_success_points(stage_success)
        stage_observations = self._trial_observations(
            points=stage_points,
            threshold=threshold,
        )

        # ── build prompt and call LLM ──
        prompt = self._build_attribution_prompt(
            round_observations=round_observations,
            stage_observations=stage_observations,
            stage_policy=stage_policy,
            knowledge_context=knowledge_context,
        )

        result = self._call_reasoning_json("hnsw_trial_attribution", prompt)

        if not result.get("ok") or not isinstance(result.get("parsed"), dict):
            return {
                "ok": False,
                "error": result.get("error", "llm_call_failed"),
                "attributions": [],
                "summary": {},
                "source": "llm_error",
            }

        parsed = result["parsed"]
        attributions = parsed.get("attributions") or []
        if not isinstance(attributions, list):
            attributions = []

        summary = parsed.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}

        return {
            "ok": True,
            "error": "",
            "attributions": attributions,
            "summary": summary,
            "source": "llm",
        }

    # ── Execution Selection ──────────────────────────────────────────────

    def select_executions(
        self,
        scored_nodes: Sequence[Dict[str, Any]],
        *,
        count: int,
        root_state: Dict[str, Any],
        scbo_reflection: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        count = max(0, int(count))
        accepted = [node for node in scored_nodes if bool(node.get("accepted", True))]
        fallback_reason = ""
        if not accepted:
            accepted = list(scored_nodes)
            fallback_reason = "no_unfiltered_nodes"

        stage = str(root_state.get("optimization_stage", "cold_start"))
        ranked = sorted(accepted, key=lambda node: self._selection_rank_key(node, stage), reverse=True)
        reasoning_root = dict(root_state)
        if scbo_reflection and "surface_uncertainty" not in reasoning_root:
            reasoning_root["surface_uncertainty"] = scbo_reflection
        reasoning = self._run_reasoning(ranked_nodes=ranked, root_state=reasoning_root)
        selected_node_id = str(reasoning.get("final_decision", {}).get("selected_node_id", ""))
        chosen = next((node for node in ranked if str(node.get("node_id", "")) == selected_node_id), None)
        if chosen is None and ranked:
            chosen = ranked[0]
        selected: List[Dict[str, Any]] = []
        if chosen is not None and count > 0:
            selected.append(chosen)
            selected_ids = {str(chosen.get("node_id", ""))}
            for node in ranked:
                if len(selected) >= count:
                    break
                node_id = str(node.get("node_id", ""))
                if node_id in selected_ids:
                    continue
                selected.append(node)
                selected_ids.add(node_id)

        selected_candidates = [
            {
                "build": self._build_only_params(node.get("params")),
                "params": node["params"],
                "source": "skill_agent",
                "note": f"{node.get('source_skill', 'skill')} selected node {node.get('node_id', '')}",
                "node_id": node.get("node_id", ""),
            }
            for node in selected
        ]
        executions = [
            {
                "node_id": node.get("node_id", ""),
                "build": self._build_only_params(node.get("params")),
                "config": node.get("params"),
                "ef_placeholder": node.get("params", {}).get("ef") if isinstance(node.get("params"), dict) else None,
                "source_branch": node.get("branch", ""),
                "source_skill": node.get("source_skill", ""),
                "selection_reason": [
                    f"optimization_stage={stage}",
                    f"node_state={node.get('observation', {}).get('node_state', '')}",
                    f"score={float(node.get('score', 0.0)):.6f}",
                    f"joint_feasible_prob={float(node.get('joint_feasible_prob', 0.0)):.6f}",
                ],
                "expected_behavior": node.get("expected_behavior", {}),
            }
            for node in selected
        ]
        return selected_candidates, {
            "selection_mode": "hnsw_skill_agent",
            "selection_fallback_reason": fallback_reason,
            "reasoning_mode": self.reasoning_mode,
            "llm_reasoning": reasoning,
            "selected_executions": executions,
            "selected_execution": executions[0] if executions else None,
        }


__all__ = [
    "HNSWLIBTuningAgent",
    "NODE_STATES",
    "PARAM_ORDER",
    "ParameterSpace",
    "key_to_params",
    "params_to_key",
]# ── Diagnostic Tree ──────────────────────────────────────────────────────
# Tree-structured diagnostic nodes.  Each node maps an observable *metric*
# to a parameter-change *solution*.  The LLM navigates from the current
# problem (recall / qps) down to specific metric → action nodes.
# Format: {node_id, problem, metric, description, solution}

def _load_diagnostic_tree(json_path: str | Path = "knowledge_base/diagnostic_tree.json") -> List[Dict[str, Any]]:
    """Load the diagnostic decision tree from a JSON file.

    Each node has: ``metric``, ``display``, and ``states`` (too_high / too_low),
    with each state containing ``branches`` keyed by ``recall_status``.
    """
    path = Path(json_path)
    if not path.exists():
        logger.warning("Diagnostic tree JSON not found: %s", json_path)
        return []
    try:
        nodes = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(nodes, list):
            return []
        return nodes
    except Exception as exc:
        logger.warning("Failed to load diagnostic tree from %s: %s", json_path, exc)
        return []


DIAGNOSTIC_TREE: List[Dict[str, Any]] = _load_diagnostic_tree()





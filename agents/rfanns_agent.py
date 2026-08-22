import itertools
import json
import math
import random
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PARAM_ORDER = ["al", "B", "ef", "efConstruction", "M"]
RANGE_STEP_BUCKETS = 16
FLOAT_ROUND_DECIMALS = 12


def _normalize_numeric_list(values: Sequence[Any]) -> List[Any]:
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Parameter value list cannot be empty.")
    return unique_values


def _is_int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {field_name}: {value}") from exc
    else:
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")

    if not math.isfinite(parsed):
        raise ValueError(f"Invalid numeric value for {field_name}: {value}")
    return parsed


def _is_integral_number(value: Any) -> bool:
    if _is_int_like(value):
        return True
    if isinstance(value, float):
        return float(value).is_integer()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            parsed = float(text)
        except ValueError:
            return False
        return parsed.is_integer()
    return False


def params_to_key(params: Dict[str, Any], order: Sequence[str] = PARAM_ORDER) -> Tuple[Any, ...]:
    return tuple(params[name] for name in order)


def key_to_params(key: Tuple[Any, ...], order: Sequence[str] = PARAM_ORDER) -> Dict[str, Any]:
    return {order[i]: key[i] for i in range(len(order))}


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
            has_min = "min" in spec
            has_max = "max" in spec
            if has_min or has_max:
                if not (has_min and has_max):
                    raise ValueError(f"Parameter '{name}' range spec must include both min and max.")
                min_number = _parse_number(spec["min"], f"{name}.min")
                max_number = _parse_number(spec["max"], f"{name}.max")
                if min_number > max_number:
                    raise ValueError(f"Parameter '{name}' range spec requires min <= max.")

                is_integer = _is_integral_number(spec["min"]) and _is_integral_number(spec["max"])
                if is_integer:
                    min_value = int(round(min_number))
                    max_value = int(round(max_number))
                else:
                    min_value = round(float(min_number), FLOAT_ROUND_DECIMALS)
                    max_value = round(float(max_number), FLOAT_ROUND_DECIMALS)
                return cls(
                    kind="range",
                    values=None,
                    min_value=min_value,
                    max_value=max_value,
                    is_integer=is_integer,
                )

            if "values" in spec:
                values = _normalize_numeric_list(spec["values"])
                return cls(kind="discrete", values=values, is_integer=False)

            raise ValueError(
                f"Parameter '{name}' spec must be a discrete list, "
                "a range object {min,max}, or a values object {values:[...]}."
            )

        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            values = _normalize_numeric_list(spec)
            return cls(kind="discrete", values=values, is_integer=False)

        raise ValueError(
            f"Parameter '{name}' spec must be a discrete list, "
            "a range object {min,max}, or a values object {values:[...]}."
        )

    def to_spec(self) -> Dict[str, Any]:
        if self.kind == "discrete":
            return {"kind": "discrete", "values": list(self.values or [])}
        return {
            "kind": "range",
            "min": self.min_value,
            "max": self.max_value,
            "integer": bool(self.is_integer),
        }

    def midpoint(self) -> Any:
        if self.kind == "discrete":
            values = self.values or []
            return values[len(values) // 2]
        min_value = self.min_value
        max_value = self.max_value
        if min_value is None or max_value is None:
            raise ValueError("Range domain is missing min/max.")
        midpoint = (float(min_value) + float(max_value)) / 2.0
        if self.is_integer:
            return int(round(midpoint))
        return round(midpoint, FLOAT_ROUND_DECIMALS)

    def edge(self, side: str) -> Any:
        if self.kind == "discrete":
            values = self.values or []
            if side == "low":
                return values[0]
            if side == "high":
                return values[-1]
            raise ValueError(f"Unknown side: {side}")
        if side == "low":
            return self.min_value
        if side == "high":
            return self.max_value
        raise ValueError(f"Unknown side: {side}")

    def step_size(self) -> float:
        if self.kind != "range":
            return 1.0
        min_value = self.min_value
        max_value = self.max_value
        if min_value is None or max_value is None:
            raise ValueError("Range domain is missing min/max.")
        span = float(max_value) - float(min_value)
        if span <= 0.0:
            return 1.0 if self.is_integer else 0.0
        if self.is_integer:
            return float(max(1, int(round(span / RANGE_STEP_BUCKETS))))
        return span / float(RANGE_STEP_BUCKETS)


@dataclass
class ParameterSpace:
    domains: Dict[str, ParameterDomain]
    order: List[str]

    @classmethod
    def from_config(cls, params_cfg: Dict[str, Any], order: Sequence[str] = PARAM_ORDER) -> "ParameterSpace":
        domains: Dict[str, ParameterDomain] = {}
        for name in order:
            if name not in params_cfg:
                raise ValueError(f"Missing parameter space for '{name}'.")
            domains[name] = ParameterDomain.from_spec(name=name, spec=params_cfg[name])
        return cls(domains=domains, order=list(order))

    @property
    def values(self) -> Dict[str, List[Any]]:
        compatible: Dict[str, List[Any]] = {}
        for name in self.order:
            domain = self.domains[name]
            if domain.kind == "discrete":
                compatible[name] = list(domain.values or [])
                continue
            min_value = domain.min_value
            max_value = domain.max_value
            if min_value is None or max_value is None:
                compatible[name] = []
                continue
            step = domain.step_size()
            if domain.is_integer:
                start = int(min_value)
                end = int(max_value)
                int_step = int(round(step)) if step > 0 else 1
                int_step = max(1, int_step)
                values = list(range(start, end + 1, int_step))
                if values[-1] != end:
                    values.append(end)
                compatible[name] = values
                continue
            if float(max_value) == float(min_value):
                compatible[name] = [round(float(min_value), FLOAT_ROUND_DECIMALS)]
                continue
            float_step = step if step > 0 else (float(max_value) - float(min_value))
            values = []
            for i in range(RANGE_STEP_BUCKETS + 1):
                values.append(round(float(min_value) + i * float_step, FLOAT_ROUND_DECIMALS))
            values[-1] = round(float(max_value), FLOAT_ROUND_DECIMALS)
            compatible[name] = values
        return compatible

    def has_range_domain(self) -> bool:
        return any(self.domains[name].kind == "range" for name in self.order)

    def export_parameter_space(self, constraints: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
        resolved = self._resolve_constraints(constraints)
        return {name: resolved[name].to_spec() for name in self.order}

    def normalize_constraints(self, raw_constraints: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return self.export_parameter_space(constraints=raw_constraints)

    def baseline(self) -> Dict[str, Any]:
        baseline: Dict[str, Any] = {}
        for name in self.order:
            baseline[name] = self.domains[name].midpoint()
        return baseline

    def _coerce_discrete(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        allowed = domain.values or []
        if value in allowed:
            return value

        parsed = value
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed = float(text)
                except ValueError:
                    parsed = text

        if parsed in allowed:
            return parsed

        if isinstance(parsed, (int, float)):
            for candidate in allowed:
                if isinstance(candidate, (int, float)) and abs(float(candidate) - float(parsed)) <= 1e-9:
                    return candidate
        raise ValueError(f"Invalid value for {key}: {value}")

    def _coerce_range(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        min_value = domain.min_value
        max_value = domain.max_value
        if min_value is None or max_value is None:
            raise ValueError(f"Range domain for {key} is missing min/max.")

        parsed = _parse_number(value, key)
        if domain.is_integer:
            rounded = round(parsed)
            if abs(parsed - rounded) > 1e-9:
                raise ValueError(f"Invalid value for {key}: {value}")
            normalized = int(rounded)
            if normalized < int(min_value) or normalized > int(max_value):
                raise ValueError(f"Invalid value for {key}: {value}")
            return normalized

        normalized = round(float(parsed), FLOAT_ROUND_DECIMALS)
        if normalized < float(min_value) - 1e-9 or normalized > float(max_value) + 1e-9:
            raise ValueError(f"Invalid value for {key}: {value}")
        clipped = min(float(max_value), max(float(min_value), normalized))
        return round(clipped, FLOAT_ROUND_DECIMALS)

    def _coerce_value(self, key: str, value: Any, domain: ParameterDomain) -> Any:
        if domain.kind == "discrete":
            return self._coerce_discrete(key, value, domain)
        return self._coerce_range(key, value, domain)

    def _subset_domain(self, key: str, base: ParameterDomain, override: ParameterDomain) -> ParameterDomain:
        if base.kind == "discrete":
            base_values = list(base.values or [])
            if override.kind == "discrete":
                values = [self._coerce_discrete(key, v, base) for v in (override.values or [])]
                dedup = _normalize_numeric_list(values)
                if any(v not in set(base_values) for v in dedup):
                    raise ValueError(f"allowed_values_override for '{key}' contains invalid values")
                return ParameterDomain(kind="discrete", values=dedup)

            min_override = self._coerce_range(key, override.min_value, ParameterDomain(  # type: ignore[arg-type]
                kind="range",
                min_value=min(base_values),
                max_value=max(base_values),
                is_integer=all(_is_integral_number(v) for v in base_values),
            ))
            max_override = self._coerce_range(key, override.max_value, ParameterDomain(  # type: ignore[arg-type]
                kind="range",
                min_value=min(base_values),
                max_value=max(base_values),
                is_integer=all(_is_integral_number(v) for v in base_values),
            ))
            if float(min_override) > float(max_override):
                raise ValueError(f"allowed_values_override for '{key}' has min > max")
            filtered = [v for v in base_values if float(min_override) <= float(v) <= float(max_override)]
            if not filtered:
                raise ValueError(f"allowed_values_override for '{key}' yields an empty discrete subset")
            return ParameterDomain(kind="discrete", values=filtered)

        if override.kind == "discrete":
            values = [self._coerce_range(key, v, base) for v in (override.values or [])]
            dedup = _normalize_numeric_list(values)
            return ParameterDomain(kind="discrete", values=dedup)

        if override.min_value is None or override.max_value is None:
            raise ValueError(f"allowed_values_override for '{key}' range is missing min/max")
        min_value = self._coerce_range(key, override.min_value, base)
        max_value = self._coerce_range(key, override.max_value, base)
        if float(min_value) > float(max_value):
            raise ValueError(f"allowed_values_override for '{key}' has min > max")

        if base.is_integer:
            return ParameterDomain(
                kind="range",
                min_value=int(min_value),
                max_value=int(max_value),
                is_integer=True,
            )
        return ParameterDomain(
            kind="range",
            min_value=round(float(min_value), FLOAT_ROUND_DECIMALS),
            max_value=round(float(max_value), FLOAT_ROUND_DECIMALS),
            is_integer=False,
        )

    def _resolve_constraints(self, constraints: Dict[str, Any] | None) -> Dict[str, ParameterDomain]:
        if constraints is None:
            return self.domains
        if not isinstance(constraints, dict):
            raise ValueError("allowed_values_override must be a dict when provided")

        resolved: Dict[str, ParameterDomain] = {}
        for name in self.order:
            if name not in constraints:
                raise ValueError(f"allowed_values_override for '{name}' cannot be empty")
            raw_spec = constraints[name]
            override_domain = ParameterDomain.from_spec(name=name, spec=raw_spec)
            resolved[name] = self._subset_domain(key=name, base=self.domains[name], override=override_domain)
        return resolved

    def canonicalize(self, params: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> Dict[str, Any]:
        domains = self._resolve_constraints(constraints)
        canonical: Dict[str, Any] = {}
        for name in self.order:
            if name not in params:
                raise ValueError(f"Missing parameter '{name}'")
            canonical[name] = self._coerce_value(name, params[name], domains[name])
        return canonical

    def out_of_constraint_fields(
        self,
        params: Dict[str, Any],
        constraints: Dict[str, Any] | None = None,
    ) -> List[str]:
        domains = self._resolve_constraints(constraints)
        fields: List[str] = []
        for name in self.order:
            if name not in params:
                fields.append(name)
                continue
            try:
                _ = self._coerce_value(name, params[name], domains[name])
            except Exception:
                fields.append(name)
        return fields

    def is_valid(self, params: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> bool:
        try:
            self.canonicalize(params, constraints=constraints)
            return True
        except Exception:
            return False

    def all_combinations(self) -> Iterable[Dict[str, Any]]:
        if self.has_range_domain():
            raise ValueError("all_combinations is only available when all parameters are discrete.")
        product_lists = [list(self.domains[name].values or []) for name in self.order]
        for combo in itertools.product(*product_lists):
            yield {self.order[i]: combo[i] for i in range(len(self.order))}

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
                values = list(domain.values or [])
                if not values:
                    raise ValueError(f"Parameter domain '{name}' has no available values.")
                sampled[name] = rng.choice(values)
                continue

            min_value = domain.min_value
            max_value = domain.max_value
            if min_value is None or max_value is None:
                raise ValueError(f"Range domain for '{name}' is missing min/max.")
            if domain.is_integer:
                sampled[name] = rng.randint(int(min_value), int(max_value))
            else:
                value = rng.uniform(float(min_value), float(max_value))
                sampled[name] = round(value, FLOAT_ROUND_DECIMALS)
        return sampled

    def estimated_cardinality(self, constraints: Dict[str, Any] | None = None) -> int:
        domains = self._resolve_constraints(constraints)
        count = 1
        for name in self.order:
            domain = domains[name]
            if domain.kind == "discrete":
                count *= max(1, len(domain.values or []))
                continue
            min_value = domain.min_value
            max_value = domain.max_value
            if min_value is None or max_value is None:
                continue
            if domain.is_integer:
                count *= max(1, int(max_value) - int(min_value) + 1)
            else:
                count *= RANGE_STEP_BUCKETS + 1
        return count

    def index_of(self, name: str, value: Any) -> int:
        domain = self.domains[name]
        if domain.kind == "discrete":
            return list(domain.values or []).index(value)
        pos = self.normalized_position(name, value)
        return int(round(pos * RANGE_STEP_BUCKETS))

    def midpoint_index(self, name: str) -> int:
        domain = self.domains[name]
        if domain.kind == "discrete":
            values = list(domain.values or [])
            return len(values) // 2
        return RANGE_STEP_BUCKETS // 2

    def edge_value(self, name: str, side: str) -> Any:
        return self.domains[name].edge(side)

    def normalized_position(self, name: str, value: Any) -> float:
        domain = self.domains[name]
        canonical = self._coerce_value(name, value, domain)
        if domain.kind == "discrete":
            values = list(domain.values or [])
            idx = values.index(canonical)
            denom = max(1, len(values) - 1)
            return idx / denom
        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        if max_value <= min_value:
            return 0.0
        return max(0.0, min(1.0, (float(canonical) - min_value) / (max_value - min_value)))

    def normalized_vector(self, params: Dict[str, Any]) -> Tuple[float, ...]:
        return tuple(self.normalized_position(name, params[name]) for name in self.order)

    def large_step_span(self, name: str) -> int:
        domain = self.domains[name]
        if domain.kind == "discrete":
            values = list(domain.values or [])
            return int(math.ceil(max(0, len(values) - 1) / 2.0))
        return max(1, RANGE_STEP_BUCKETS // 2)

    def value_from_position(self, name: str, pos: float) -> Any:
        domain = self.domains[name]
        normalized = max(0.0, min(1.0, float(pos)))
        if domain.kind == "discrete":
            values = list(domain.values or [])
            idx = int(round(normalized * max(0, len(values) - 1)))
            idx = max(0, min(len(values) - 1, idx))
            return values[idx]
        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        value = min_value + normalized * (max_value - min_value)
        if domain.is_integer:
            return int(max(int(min_value), min(int(max_value), round(value))))
        return round(value, FLOAT_ROUND_DECIMALS)

    def value_at_delta(self, name: str, value: Any, delta: int) -> Any | None:
        domain = self.domains[name]
        if domain.kind == "discrete":
            values = list(domain.values or [])
            idx = self.index_of(name, value) + delta
            if idx < 0 or idx >= len(values):
                return None
            return values[idx]

        min_value = float(domain.min_value)  # type: ignore[arg-type]
        max_value = float(domain.max_value)  # type: ignore[arg-type]
        current = self._coerce_value(name, value, domain)
        step = domain.step_size()
        shifted = float(current) + float(delta) * float(step)
        shifted = min(max_value, max(min_value, shifted))
        if domain.is_integer:
            return int(max(int(min_value), min(int(max_value), round(shifted))))
        return round(shifted, FLOAT_ROUND_DECIMALS)

    def neighbor_candidates(self, params: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int]]:
        neighbors: List[Tuple[Dict[str, Any], int]] = []
        for deltas in itertools.product([-1, 0, 1], repeat=len(self.order)):
            if all(delta == 0 for delta in deltas):
                continue

            next_params: Dict[str, Any] = {}
            valid = True
            l1_step = 0
            for i, name in enumerate(self.order):
                next_value = self.value_at_delta(name, params[name], deltas[i])
                if next_value is None:
                    valid = False
                    break
                next_params[name] = next_value
                l1_step += abs(deltas[i])
            if valid and next_params != params:
                neighbors.append((next_params, l1_step))
        return neighbors


def compute_pareto_front(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    non_dominated: List[Dict[str, Any]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            q_better_or_equal = q["recall"] >= p["recall"] and q["qps"] >= p["qps"]
            q_strictly_better = q["recall"] > p["recall"] or q["qps"] > p["qps"]
            if q_better_or_equal and q_strictly_better:
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)
    return sorted(non_dominated, key=lambda x: (x["recall"], x["qps"]), reverse=True)


class RFANNSTuningAgent:
    def __init__(
        self,
        params_cfg: Dict[str, Any],
        seed: int = 42,
        param_order: Sequence[str] | None = None,
        agentic_cfg: Dict[str, Any] | None = None,
        model_cfg: Dict[str, Any] | None = None,
        prompt_cfg: Dict[str, str] | None = None,
        llm_caller: Any | None = None,
        objective_preference: str = "pareto",
    ):
        self.space = ParameterSpace.from_config(params_cfg, order=param_order or PARAM_ORDER)
        self.seed = seed
        self._rng = random.Random(seed)

        self.agentic_cfg = agentic_cfg or {}
        self.model_cfg = model_cfg or {}
        self.prompt_cfg = prompt_cfg or {}
        self.llm_caller = llm_caller
        self.objective_preference = (objective_preference or "pareto").lower()
        if self.agentic_cfg.get("enabled", True) is False:
            raise ValueError("agentic.enabled must be true — the rfanns/unify pipeline requires LLM proposals.")
        self.enable_agentic = True
        proposer_retry_cfg = self.agentic_cfg.get("proposer_retry", {})
        if not isinstance(proposer_retry_cfg, dict):
            proposer_retry_cfg = {}
        self.proposer_retry_max_attempts = max(1, int(proposer_retry_cfg.get("max_attempts", 3)))

        self.alias_to_canonical = self._build_alias_to_canonical()

        self.reflection_memory: Dict[str, Any] = self._empty_reflection()

    def _build_alias_to_canonical(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        if "B" in self.space.domains:
            aliases.update({"num_slots": "B", "b": "B"})
        if "efConstruction" in self.space.domains:
            aliases["ef_construction"] = "efConstruction"
        if "ef_construction" in self.space.domains:
            aliases["efConstruction"] = "ef_construction"
        return aliases

    @property
    def param_order(self) -> List[str]:
        return self.space.order

    def _empty_reflection(self) -> Dict[str, Any]:
        return {
            "high_potential": [],
            "failure_modes": [],
            "next_focus": "",
            "threshold_reached": False,
            "fastest_feasible": None,
        }

    def _resolve_unified_policy(self, stage_policy: Dict[str, Any] | None) -> Dict[str, Any]:
        resolved = {
            "stage": "unified",
            "recall_threshold": None,
            "recall_slack": 0.01,
            "objective_text": "",
        }
        if stage_policy:
            resolved.update(stage_policy)

        if resolved.get("recall_threshold") is not None:
            resolved["recall_threshold"] = float(resolved["recall_threshold"])
        if resolved.get("recall_slack") is not None:
            resolved["recall_slack"] = float(resolved["recall_slack"])

        if resolved.get("recall_threshold") is None:
            raise ValueError("unified strategy requires recall_threshold in stage_policy.")
        resolved["objective_text"] = (
            f"use recall >= {resolved['recall_threshold']:.4f} as a guardrail, then optimize qps; "
            f"points as low as {resolved['recall_threshold'] - resolved['recall_slack']:.4f} "
            "can be kept when qps improves materially"
        )
        return resolved

    def canonicalize(
        self,
        params: Dict[str, Any],
        domain_constraints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized = dict(params or {})

        for alias, canonical in self.alias_to_canonical.items():
            if alias in normalized and canonical not in normalized:
                normalized[canonical] = normalized[alias]

        return self.space.canonicalize(normalized, constraints=domain_constraints)

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
            snippet = code_match.group(1)
            try:
                loaded = json.loads(snippet)
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
                        snippet = stripped[start : idx + 1]
                        try:
                            loaded = json.loads(snippet)
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

    def _candidate_schema(self) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "params": {name: "..." for name in self.space.order},
                    "note": "short rationale",
                }
            ]
        }

    def _default_template(self, role: str) -> str:
        if role == "proposer":
            return (
                "You are an ANN tuning proposer.\n"
                "Round: {round_idx}.\n"
                "Objective: {objective_text}.\n"
                "Recall threshold: {recall_threshold}. Recall slack: {recall_slack}.\n"
                "Selection mode: {selection_mode}.\n"
                "Cold-start diversity mode: {cold_start_diversity_mode}.\n"
                "Cold-start diversity reason: {cold_start_diversity_reason}.\n"
                "Allowed parameter space constraints: {allowed_parameter_space}.\n"
                "History summary: {history_summary}.\n"
                "Reflection memory: {reflection}.\n"
                "SCBO structured reflection: {scbo_reflection}.\n"
                "Knowledge base (curated tuning reference): {knowledge_full_context}.\n"
                "Similar task transfer context: {similar_task_context}.\n"
                "If selection_mode is cold_start_diversity, prioritize spread and coverage.\n"
                "If selection_mode is threshold_guided, prioritize recall guardrail and qps tradeoff.\n"
                "Propose exactly {target_unique} candidates.\n"
                "Return strict JSON matching this schema: {candidate_schema}."
            )
        if role == "reflector":
            return (
                "You are an ANN tuning reflector.\n"
                "Objective: {objective_text}.\n"
                "Recall threshold: {recall_threshold}. Recall slack: {recall_slack}.\n"
                "Selection mode: {selection_mode}.\n"
                "Cold-start diversity mode: {cold_start_diversity_mode}.\n"
                "Cold-start diversity reason: {cold_start_diversity_reason}.\n"
                "Selection context: {selection_context}.\n"
                "Round results: {round_results}.\n"
                "Return strict JSON: {'high_potential':['...'],'failure_modes':['...'],'next_focus':'...'}"
            )
        raise ValueError(f"Unsupported RFANNS role '{role}'")

    def _default_llm_call(self, role: str, prompt: str) -> str:
        from openai import OpenAI

        model_name = self.model_cfg.get("model_name") or self.model_cfg.get("model")
        if not model_name:
            raise RuntimeError(
                "RFANNS agent model_name is not configured — "
                "set LLM_MODEL_NAME (or per-pipeline override) in .env."
            )
        base_url = self.model_cfg.get("url")
        api_key = self.model_cfg.get("authorization")
        temperature = float(self.model_cfg.get("temperature", 0.2))
        max_tokens = int(self.model_cfg.get("max_tokens", 2048))

        if not base_url or "<base_url>" in str(base_url):
            raise RuntimeError("RFANNS agent model URL is not configured.")
        if not api_key or "<token>" in str(api_key):
            raise RuntimeError("RFANNS agent model authorization is not configured.")

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_role(self, role: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        template = self.prompt_cfg.get(f"{role}_user", self._default_template(role))
        prompt = self._safe_format(template, payload)

        raw = ""
        parsed = None
        error = None
        try:
            if self.llm_caller is not None:
                raw = self.llm_caller(role=role, prompt=prompt)
            else:
                raw = self._default_llm_call(role=role, prompt=prompt)
            parsed = self._extract_json_payload(raw)
            if parsed is None:
                error = "json_parse_failed"
        except Exception as exc:
            error = f"llm_error: {exc}"

        return {
            "role": role,
            "ok": error is None,
            "error": error,
            "raw": raw,
            "parsed": parsed,
        }

    def _make_history_summary(self, trials: Sequence[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for trial in trials[-limit:]:
            item = {
                "params": trial.get("params"),
                "status": trial.get("status"),
                "proposal_source": trial.get("proposal_source", "unknown"),
                "proposal_round": trial.get("proposal_round", -1),
            }
            metrics = trial.get("metrics") or {}
            if "recall" in metrics and "qps" in metrics:
                item["recall"] = metrics["recall"]
                item["qps"] = metrics["qps"]
            if trial.get("error"):
                item["error"] = trial.get("error")
            summary.append(item)
        return summary

    def _extract_candidates(self, parsed: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not parsed:
            return []
        raw_candidates = parsed.get("candidates", parsed if isinstance(parsed, list) else [])
        if not isinstance(raw_candidates, list):
            return []

        results: List[Dict[str, Any]] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            params = raw.get("params") if isinstance(raw.get("params"), dict) else raw
            note = str(raw.get("note", "")).strip()
            results.append({"params": params, "note": note})
        return results

    def _sanitize_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        exclude_param_keys: set[Tuple[Any, ...]],
        rejected: List[Dict[str, Any]],
        source: str,
        allowed_values_override: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        seen = set(exclude_param_keys)
        for item in candidates:
            try:
                params = self.canonicalize(item.get("params", {}))
                if allowed_values_override:
                    out_of_override = self.space.out_of_constraint_fields(params, allowed_values_override)
                    if out_of_override:
                        rejected.append(
                            {
                                "reason": "out_of_allowed_override",
                                "params": params,
                                "fields": out_of_override,
                            }
                        )
                        continue
                key = params_to_key(params, self.space.order)
                if key in seen:
                    rejected.append({"reason": "duplicate", "params": params})
                    continue
                seen.add(key)
                sanitized.append(
                    {
                        "params": params,
                        "note": (item.get("note") or "").strip(),
                        "source": source,
                    }
                )
            except Exception as exc:
                reason = "invalid"
                raw_params = item.get("params")
                if "al" in self.space.domains and isinstance(raw_params, dict) and "al" in raw_params:
                    raw_al = raw_params.get("al")
                    try:
                        al_value = float(raw_al)
                        if 0.0 < al_value <= 1.0:
                            reason = "invalid_al_semantics"
                    except (TypeError, ValueError):
                        pass
                rejected.append(
                    {
                        "reason": reason,
                        "params": raw_params,
                        "error": str(exc),
                    }
                )
        return sanitized

    def _heuristic_value(self, name: str, value: Any) -> float:
        return self.space.normalized_position(name, value)

    def _heuristic_prediction(self, params: Dict[str, Any]) -> Dict[str, float]:
        if all(name in self.space.domains for name in PARAM_ORDER):
            recall_proxy = (
                self._heuristic_value("al", params["al"]) * 0.28
                + self._heuristic_value("ef", params["ef"]) * 0.28
                + self._heuristic_value("efConstruction", params["efConstruction"]) * 0.2
                + self._heuristic_value("M", params["M"]) * 0.2
                + (1.0 - self._heuristic_value("B", params["B"])) * 0.04
            )
            qps_proxy = (
                (1.0 - self._heuristic_value("al", params["al"])) * 0.3
                + (1.0 - self._heuristic_value("ef", params["ef"])) * 0.3
                + (1.0 - self._heuristic_value("M", params["M"])) * 0.2
                + (1.0 - self._heuristic_value("efConstruction", params["efConstruction"])) * 0.1
                + (1.0 - self._heuristic_value("B", params["B"])) * 0.1
            )
            return {
                "recall": max(0.0, min(1.0, recall_proxy)),
                "qps": max(1e-6, qps_proxy),
            }

        positions = [self._heuristic_value(name, params[name]) for name in self.space.order]
        if not positions:
            return {"recall": 0.0, "qps": 1e-6}
        recall_proxy = sum(positions) / len(positions)
        qps_proxy = sum(1.0 - pos for pos in positions) / len(positions)
        return {
            "recall": max(0.0, min(1.0, recall_proxy)),
            "qps": max(1e-6, qps_proxy),
        }

    def _candidate_score(self, params: Dict[str, Any]) -> float:
        prediction = self._heuristic_prediction(params)
        if self.objective_preference == "recall":
            return prediction["recall"]
        if self.objective_preference == "qps":
            return prediction["qps"]
        return prediction["recall"] + prediction["qps"]

    def random_candidates(
        self,
        count: int,
        exclude_keys: set[Tuple[Any, ...]] | None = None,
        constraints: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        exclude_keys = exclude_keys or set()
        sampled: List[Dict[str, Any]] = []
        max_attempts = max(200, count * 80)
        attempts = 0
        while len(sampled) < count and attempts < max_attempts:
            attempts += 1
            params = self.space.sample_random_params(self._rng, constraints=constraints)
            key = params_to_key(params, self.space.order)
            if key in exclude_keys:
                continue
            exclude_keys.add(key)
            sampled.append(params)
        return sampled

    def _params_from_trials(self, trials: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        for trial in trials:
            params = trial.get("params")
            if not isinstance(params, dict):
                continue
            try:
                params_list.append(self.canonicalize(params))
            except Exception:
                continue
        return params_list

    def _distance(self, params_a: Dict[str, Any], params_b: Dict[str, Any]) -> float:
        vec_a = self.space.normalized_vector(params_a)
        vec_b = self.space.normalized_vector(params_b)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    def _min_distance(self, params: Dict[str, Any], references: Sequence[Dict[str, Any]]) -> float:
        if not references:
            return 1.0
        return min(self._distance(params, ref) for ref in references)

    def _avg_distance(self, params: Dict[str, Any], references: Sequence[Dict[str, Any]]) -> float:
        if not references:
            return 1.0
        return sum(self._distance(params, ref) for ref in references) / len(references)

    def _coverage_labels(self, params: Dict[str, Any]) -> set[Tuple[str, str]]:
        labels: set[Tuple[str, str]] = set()
        for name in self.space.order:
            domain = self.space.domains[name]
            if domain.kind == "discrete" and len(domain.values or []) <= 1:
                continue
            pos = self.space.normalized_position(name, params[name])
            if pos <= 0.25:
                labels.add((name, "low"))
            if pos >= 0.75:
                labels.add((name, "high"))
        return labels

    def _diversity_anchor_candidates(self) -> List[Dict[str, Any]]:
        baseline = self.space.baseline()
        candidates: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()

        def add_candidate(params: Dict[str, Any], note: str) -> None:
            canonical = self.canonicalize(params)
            key = params_to_key(canonical, self.space.order)
            if key in seen:
                return
            seen.add(key)
            candidates.append({"params": canonical, "note": note, "source": "coverage_anchor"})

        add_candidate(baseline, "diversity midpoint anchor")
        add_candidate({name: self.space.edge_value(name, "low") for name in self.space.order}, "diversity all-low anchor")
        add_candidate({name: self.space.edge_value(name, "high") for name in self.space.order}, "diversity all-high anchor")
        add_candidate(
            {
                name: self.space.edge_value(name, "low" if idx % 2 == 0 else "high")
                for idx, name in enumerate(self.space.order)
            },
            "diversity mixed corner A",
        )
        add_candidate(
            {
                name: self.space.edge_value(name, "high" if idx % 2 == 0 else "low")
                for idx, name in enumerate(self.space.order)
            },
            "diversity mixed corner B",
        )
        for name in self.space.order:
            low_params = dict(baseline)
            low_params[name] = self.space.edge_value(name, "low")
            add_candidate(low_params, f"diversity low-anchor for {name}")

            high_params = dict(baseline)
            high_params[name] = self.space.edge_value(name, "high")
            add_candidate(high_params, f"diversity high-anchor for {name}")

        return candidates

    def _merge_candidate_pools(
        self,
        *pools: Sequence[Dict[str, Any]],
        exclude_param_keys: set[Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set(exclude_param_keys)
        for pool in pools:
            for item in pool:
                params = item.get("params")
                if not isinstance(params, dict):
                    continue
                try:
                    canonical = self.canonicalize(params)
                except Exception:
                    continue
                key = params_to_key(canonical, self.space.order)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "params": canonical,
                        "note": str(item.get("note", "")).strip(),
                        "source": item.get("source", "unknown"),
                    }
                )
        return merged

    def _build_diversity_structured_pool(
        self,
        target_unique: int,
        exclude_param_keys: set[Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        anchors = [item for item in self._diversity_anchor_candidates() if params_to_key(item["params"], self.space.order) not in exclude_param_keys]
        pool = list(anchors)
        target_pool_size = max(target_unique * 4, len(anchors))
        seen = exclude_param_keys.union({params_to_key(item["params"], self.space.order) for item in pool})
        if len(pool) < target_pool_size:
            fill = self.random_candidates(target_pool_size - len(pool), exclude_keys=seen)
            for params in fill:
                pool.append({"params": params, "note": "diversity fill", "source": "coverage_fill"})
        return pool

    def _select_diversity_candidates(
        self,
        candidate_pool: Sequence[Dict[str, Any]],
        target_unique: int,
        stage_trials: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if target_unique <= 0 or not candidate_pool:
            return []

        attempted_params = self._params_from_trials(stage_trials)
        covered = set()
        for params in attempted_params:
            covered.update(self._coverage_labels(params))

        remaining = list(candidate_pool)
        selected: List[Dict[str, Any]] = []
        while remaining and len(selected) < target_unique:
            references = attempted_params + [item["params"] for item in selected]
            best_idx = -1
            best_score: Tuple[Any, ...] | None = None
            for idx, item in enumerate(remaining):
                new_coverage = self._coverage_labels(item["params"]) - covered
                min_distance = self._min_distance(item["params"], references)
                avg_distance = self._avg_distance(item["params"], references)
                anchor_bonus = 1 if item.get("source") == "coverage_anchor" else 0
                score = (
                    len(new_coverage),
                    min_distance,
                    avg_distance,
                    anchor_bonus,
                )
                if best_score is None or score > best_score:
                    best_idx = idx
                    best_score = score
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            covered.update(self._coverage_labels(chosen["params"]))

        return selected

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
            except Exception:
                continue
            key = params_to_key(canonical, self.space.order)
            bucket = grouped.setdefault(key, {"params": canonical, "recall": [], "qps": []})
            bucket["recall"].append(float(metrics["recall"]))
            bucket["qps"].append(float(metrics["qps"]))

        points: List[Dict[str, Any]] = []
        for bucket in grouped.values():
            points.append(
                {
                    "params": bucket["params"],
                    "recall": float(median(bucket["recall"])),
                    "qps": float(median(bucket["qps"])),
                }
            )
        points.sort(key=lambda item: (item["recall"], item["qps"]), reverse=True)
        return points

    def _predict_metrics(
        self,
        params: Dict[str, Any],
        success_points: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        if len(success_points) < 3:
            return self._heuristic_prediction(params)

        target_vec = self.space.normalized_vector(params)
        distances: List[Tuple[float, Dict[str, Any]]] = []
        for point in success_points:
            point_vec = self.space.normalized_vector(point["params"])
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(target_vec, point_vec)))
            if distance <= 1e-12:
                return {"recall": float(point["recall"]), "qps": float(point["qps"])}
            distances.append((distance, point))

        nearest = sorted(distances, key=lambda item: item[0])[:5]
        total_weight = 0.0
        recall = 0.0
        qps = 0.0
        for distance, point in nearest:
            weight = 1.0 / max(distance, 1e-6)
            total_weight += weight
            recall += weight * float(point["recall"])
            qps += weight * float(point["qps"])

        if total_weight <= 0:
            return self._heuristic_prediction(params)
        return {"recall": recall / total_weight, "qps": qps / total_weight}

    def _classify_recall_bucket(self, recall: float, threshold: float, slack: float) -> str:
        if recall >= threshold:
            return "feasible"
        if recall >= threshold - slack:
            return "near_feasible"
        return "below_slack"

    def _select_threshold_centers(
        self,
        success_points: Sequence[Dict[str, Any]],
        threshold: float,
        slack: float,
    ) -> List[Dict[str, Any]]:
        feasible = [point for point in success_points if point["recall"] >= threshold]
        near = [point for point in success_points if threshold - slack <= point["recall"] < threshold]
        if feasible:
            centers = sorted(feasible, key=lambda item: (-item["qps"], item["recall"]))[:3]
            extra = sorted(near, key=lambda item: (-item["qps"], abs(item["recall"] - threshold)))[:2]
            merged: List[Dict[str, Any]] = []
            seen = set()
            for item in centers + extra:
                key = params_to_key(item["params"], self.space.order)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
            return merged
        return sorted(success_points, key=lambda item: (-item["recall"], -item["qps"]))[:3]

    def _apply_deltas(self, params: Dict[str, Any], deltas: Dict[str, int]) -> Dict[str, Any] | None:
        next_params = dict(params)
        for name, delta in deltas.items():
            next_value = self.space.value_at_delta(name, next_params[name], delta)
            if next_value is None:
                return None
            next_params[name] = next_value
        return next_params

    def _threshold_one_step_pool(
        self,
        center_params: Dict[str, Any],
        center_bucket: str,
    ) -> List[Dict[str, Any]]:
        priority = list(self.space.order)
        if all(name in self.space.domains for name in PARAM_ORDER):
            priority = [name for name in ["al", "ef", "B", "M", "efConstruction"] if name in self.space.domains]

        if center_bucket == "below_slack":
            deltas_by_name = {name: [1, -1] for name in priority}
        elif center_bucket == "near_feasible":
            deltas_by_name = {name: [1, -1] for name in priority}
        else:
            deltas_by_name = {name: [-1, 1] for name in priority}

        if all(name in self.space.domains for name in PARAM_ORDER):
            deltas_by_name.update(
                {
                    "feasible": {
                        "al": [-1, 1],
                        "ef": [-1, 1],
                        "B": [-1, 1],
                        "M": [-1, 1],
                        "efConstruction": [-1, 1],
                    },
                    "near_feasible": {
                        "al": [-1, 1],
                        "ef": [-1, 1],
                        "B": [-1, 1],
                        "M": [1, -1],
                        "efConstruction": [1, -1],
                    },
                    "below_slack": {
                        "al": [1, -1],
                        "ef": [1, -1],
                        "B": [-1, 1],
                        "M": [1, -1],
                        "efConstruction": [1, -1],
                    },
                }[center_bucket]
            )

        pool: List[Dict[str, Any]] = []
        for name in priority:
            for delta in deltas_by_name.get(name, [-1, 1]):
                candidate = self._apply_deltas(center_params, {name: delta})
                if candidate is None:
                    continue
                direction = "up" if delta > 0 else "down"
                pool.append(
                    {
                        "params": candidate,
                        "note": f"threshold 1-step {name} {direction} from {center_bucket} center",
                        "source": "threshold_local",
                    }
                )
        return pool

    def _threshold_pair_pool(
        self,
        center_params: Dict[str, Any],
        center_bucket: str,
    ) -> List[Dict[str, Any]]:
        if all(name in self.space.domains for name in PARAM_ORDER) and center_bucket == "below_slack":
            delta_sets = [
                {"al": 1, "B": -1},
                {"ef": 1, "B": -1},
                {"M": 1, "al": -1},
                {"efConstruction": 1, "ef": -1},
            ]
        elif all(name in self.space.domains for name in PARAM_ORDER):
            delta_sets = [
                {"al": -1, "M": 1},
                {"al": -1, "efConstruction": 1},
                {"ef": -1, "M": 1},
                {"ef": -1, "efConstruction": 1},
                {"al": -1, "B": -1},
                {"ef": -1, "B": -1},
            ]
        else:
            names = list(self.space.order)
            if len(names) < 2:
                return []
            pair_names = list(itertools.combinations(names, 2))[: max(1, len(names))]
            if center_bucket == "below_slack":
                delta_sets = [{left: 1, right: 1} for left, right in pair_names]
            else:
                delta_sets = [{left: -1, right: 1} for left, right in pair_names]

        pool: List[Dict[str, Any]] = []
        for deltas in delta_sets:
            candidate = self._apply_deltas(center_params, deltas)
            if candidate is None:
                continue
            delta_note = ", ".join(f"{name}:{step:+d}" for name, step in deltas.items())
            pool.append(
                {
                    "params": candidate,
                    "note": f"threshold 2-step compensation from {center_bucket} center ({delta_note})",
                    "source": "threshold_local",
                }
            )
        return pool

    def _build_threshold_guided_pool(
        self,
        target_unique: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        all_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy["recall_slack"])
        success_points = self._aggregate_success_points(all_trials)
        centers = self._select_threshold_centers(success_points, threshold, slack)

        pool: List[Dict[str, Any]] = []
        for center in centers:
            center_bucket = self._classify_recall_bucket(center["recall"], threshold, slack)
            pool.append(
                {
                    "params": center["params"],
                    "note": f"threshold center reuse candidate ({center_bucket})",
                    "source": "threshold_center",
                }
            )
            pool.extend(self._threshold_one_step_pool(center["params"], center_bucket))
            pool.extend(self._threshold_pair_pool(center["params"], center_bucket))

        target_pool_size = max(target_unique * 5, len(pool))
        seen = exclude_param_keys.union({params_to_key(item["params"], self.space.order) for item in pool})
        if len(pool) < target_pool_size:
            fill = self.random_candidates(target_pool_size - len(pool), exclude_keys=seen)
            for params in fill:
                pool.append({"params": params, "note": "threshold fallback fill", "source": "threshold_fill"})
        return pool

    def _threshold_sort_key(self, item: Dict[str, Any], threshold: float, slack: float) -> Tuple[Any, ...]:
        predicted = item["predicted_metrics"]
        bucket_order = {"feasible": 0, "near_feasible": 1, "below_slack": 2}
        bucket = item["recall_bucket"]
        if bucket == "feasible":
            return (
                bucket_order[bucket],
                -predicted["qps"],
                abs(predicted["recall"] - threshold),
                item["source"],
            )
        if bucket == "near_feasible":
            return (
                bucket_order[bucket],
                -predicted["qps"],
                abs(predicted["recall"] - threshold),
                item["source"],
            )
        return (
            bucket_order[bucket],
            abs(predicted["recall"] - threshold),
            -predicted["qps"],
            item["source"],
        )

    def _select_threshold_guided_candidates(
        self,
        candidate_pool: Sequence[Dict[str, Any]],
        target_unique: int,
        all_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if target_unique <= 0 or not candidate_pool:
            return []

        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy["recall_slack"])
        success_points = self._aggregate_success_points(all_trials)
        actual_feasible_exists = any(point["recall"] >= threshold for point in success_points)

        enriched: List[Dict[str, Any]] = []
        for item in candidate_pool:
            predicted = self._predict_metrics(item["params"], success_points)
            bucket = self._classify_recall_bucket(predicted["recall"], threshold, slack)
            if actual_feasible_exists and bucket == "below_slack":
                continue
            enriched.append(
                {
                    **item,
                    "predicted_metrics": predicted,
                    "recall_bucket": bucket,
                }
            )

        if not enriched:
            for item in candidate_pool:
                predicted = self._predict_metrics(item["params"], success_points)
                enriched.append(
                    {
                        **item,
                        "predicted_metrics": predicted,
                        "recall_bucket": self._classify_recall_bucket(predicted["recall"], threshold, slack),
                    }
                )

        ranked = sorted(enriched, key=lambda item: self._threshold_sort_key(item, threshold, slack))
        return ranked[:target_unique]

    def _rule_action_items(
        self,
        reference_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
    ) -> set[str]:
        items: set[str] = set()
        for name in self.space.order:
            if name not in reference_params or name not in candidate_params:
                continue
            reference = self.space.normalized_position(name, reference_params[name])
            current = self.space.normalized_position(name, candidate_params[name])
            delta = current - reference
            if abs(delta) <= 1e-9:
                continue
            direction = "up" if delta > 0 else "down"
            magnitude = abs(delta)
            if magnitude <= 0.2:
                bucket = "small"
            elif magnitude <= 0.5:
                bucket = "medium"
            else:
                bucket = "large"
            items.add(f"act:{name}:{direction}:{bucket}")
        return items

    def _base_threshold_scores(
        self,
        candidates: Sequence[Dict[str, Any]],
        all_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
    ) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
        if not candidates:
            return {}
        threshold = float(stage_policy["recall_threshold"])
        slack = float(stage_policy["recall_slack"])
        success_points = self._aggregate_success_points(all_trials)

        enriched: List[Dict[str, Any]] = []
        for item in candidates:
            predicted = self._predict_metrics(item["params"], success_points)
            enriched.append(
                {
                    "params": item["params"],
                    "source": item.get("source", "agent"),
                    "predicted_metrics": predicted,
                    "recall_bucket": self._classify_recall_bucket(predicted["recall"], threshold, slack),
                }
            )

        ranked = sorted(enriched, key=lambda item: self._threshold_sort_key(item, threshold, slack))
        rank_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        denominator = max(1, len(ranked) - 1)
        for idx, item in enumerate(ranked):
            key = params_to_key(item["params"], self.space.order)
            base_score = 1.0 - (idx / denominator) if len(ranked) > 1 else 1.0
            rank_map[key] = {
                "base_score": float(base_score),
                "predicted_metrics": item["predicted_metrics"],
                "recall_bucket": item["recall_bucket"],
                "pre_rank": idx + 1,
            }
        return rank_map

    def _apply_rule_rerank(
        self,
        candidates: Sequence[Dict[str, Any]],
        target_unique: int,
        all_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
        rule_context: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        default_meta = {
            "rerank_applied": False,
            "diagnosis_tags": [],
            "retrieved_rule_ids": [],
            "matched_rule_count": 0,
            "rule_score_components": [],
            "pre_rerank_rank": {},
            "post_rerank_rank": {},
        }
        if not candidates or target_unique <= 0:
            return list(candidates), default_meta
        if not isinstance(rule_context, dict) or not bool(rule_context.get("enabled", False)):
            return list(candidates), default_meta

        retrieved_rules = [row for row in (rule_context.get("retrieved_rules") or []) if isinstance(row, dict)]
        if not retrieved_rules:
            meta = dict(default_meta)
            meta["diagnosis_tags"] = list(rule_context.get("diagnosis_tags") or [])
            meta["retrieved_rule_ids"] = list(rule_context.get("retrieved_rule_ids") or [])
            return list(candidates), meta

        reference_params = rule_context.get("reference_params")
        if not isinstance(reference_params, dict):
            meta = dict(default_meta)
            meta["diagnosis_tags"] = list(rule_context.get("diagnosis_tags") or [])
            meta["retrieved_rule_ids"] = list(rule_context.get("retrieved_rule_ids") or [])
            return list(candidates), meta

        try:
            canonical_reference = self.canonicalize(reference_params)
        except Exception:
            meta = dict(default_meta)
            meta["diagnosis_tags"] = list(rule_context.get("diagnosis_tags") or [])
            meta["retrieved_rule_ids"] = list(rule_context.get("retrieved_rule_ids") or [])
            return list(candidates), meta

        weights_cfg = rule_context.get("rerank_weights") or {}
        weight_ei = max(0.0, float(weights_cfg.get("ei", 0.35)))
        weight_risk = max(0.0, float(weights_cfg.get("risk", 0.45)))
        weight_diag = max(0.0, float(weights_cfg.get("diag", 0.20)))
        risk_ceiling = max(0.0, float(rule_context.get("risk_ceiling", 0.25)))
        diagnosis_tags = list(rule_context.get("diagnosis_tags") or [])
        diagnosis_set = set(str(tag) for tag in diagnosis_tags)

        base_scores = self._base_threshold_scores(candidates, all_trials=all_trials, stage_policy=stage_policy)
        if not base_scores:
            return list(candidates), default_meta

        scored_rows: List[Dict[str, Any]] = []
        for candidate in candidates:
            key = params_to_key(candidate["params"], self.space.order)
            base_row = base_scores.get(key, {})
            base_score = float(base_row.get("base_score", 0.0))
            action_items = self._rule_action_items(canonical_reference, candidate["params"])

            matched_rule_ids: List[str] = []
            expected_improvements: List[float] = []
            recall_risks: List[float] = []
            diagnosis_matches: List[float] = []
            for rule in retrieved_rules:
                rule_id = str(rule.get("id") or "")
                rule_actions = set(str(item) for item in (rule.get("action") or []))
                if not rule_actions:
                    continue
                if not action_items.intersection(rule_actions):
                    continue

                matched_rule_ids.append(rule_id)
                expected_improvements.append(float(rule.get("expected_improvement", 0.0)))
                recall_risks.append(float(rule.get("recall_risk", 0.0)))

                diag_requirements = {
                    item.split("diag:", 1)[1]
                    for item in (rule.get("antecedent") or [])
                    if isinstance(item, str) and item.startswith("diag:")
                }
                if not diag_requirements:
                    diagnosis_matches.append(0.0)
                else:
                    overlap = len(diag_requirements.intersection(diagnosis_set))
                    diagnosis_matches.append(overlap / max(1, len(diag_requirements)))

            expected_improvement = max(expected_improvements) if expected_improvements else 0.0
            recall_risk = max(recall_risks) if recall_risks else 0.0
            diagnosis_match = max(diagnosis_matches) if diagnosis_matches else 0.0

            final_score = (
                base_score
                + weight_ei * expected_improvement
                - weight_risk * recall_risk
                + weight_diag * diagnosis_match
            )
            if recall_risk > risk_ceiling:
                final_score -= weight_risk * (recall_risk - risk_ceiling)

            scored_rows.append(
                {
                    "candidate": candidate,
                    "key": key,
                    "final_score": float(final_score),
                    "base_score": base_score,
                    "expected_improvement": float(expected_improvement),
                    "recall_risk": float(recall_risk),
                    "diagnosis_match": float(diagnosis_match),
                    "matched_rule_ids": matched_rule_ids,
                    "pre_rank": int(base_row.get("pre_rank", 0)),
                }
            )

        ranked_rows = sorted(
            scored_rows,
            key=lambda item: (
                item["final_score"],
                item["base_score"],
            ),
            reverse=True,
        )
        reranked = [row["candidate"] for row in ranked_rows[:target_unique]]

        pre_rank = {str(row["key"]): int(row["pre_rank"]) for row in scored_rows}
        post_rank = {str(row["key"]): idx + 1 for idx, row in enumerate(ranked_rows)}
        matched_rule_ids = sorted({rule_id for row in scored_rows for rule_id in row["matched_rule_ids"]})
        rule_score_components = [
            {
                "params_key": str(row["key"]),
                "base_threshold_score": round(float(row["base_score"]), 6),
                "expected_improvement": round(float(row["expected_improvement"]), 6),
                "recall_risk": round(float(row["recall_risk"]), 6),
                "diagnosis_match": round(float(row["diagnosis_match"]), 6),
                "final_score": round(float(row["final_score"]), 6),
                "matched_rule_ids": row["matched_rule_ids"],
            }
            for row in ranked_rows
        ]
        meta = {
            "rerank_applied": True,
            "diagnosis_tags": diagnosis_tags,
            "retrieved_rule_ids": [str(rule.get("id")) for rule in retrieved_rules],
            "matched_rule_count": len(matched_rule_ids),
            "rule_score_components": rule_score_components,
            "pre_rerank_rank": pre_rank,
            "post_rerank_rank": post_rank,
        }
        return reranked, meta

    def _select_llm_seed_candidates(
        self,
        candidate_pool: Sequence[Dict[str, Any]],
        target_unique: int,
        stage_trials: Sequence[Dict[str, Any]] | None = None,
        diversity_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        if target_unique <= 0 or not candidate_pool:
            return []
        if not diversity_mode:
            ranked = sorted(
                candidate_pool,
                key=lambda item: (
                    0 if item.get("source") == "agent" else 1,
                    -self._candidate_score(item["params"]),
                ),
            )
            return ranked[:target_unique]

        selected: List[Dict[str, Any]] = []
        attempted_params = self._params_from_trials(stage_trials or [])
        remaining = list(candidate_pool)
        while remaining and len(selected) < target_unique:
            references = attempted_params + [item["params"] for item in selected]
            best_idx = 0
            best_score: Tuple[Any, ...] | None = None
            for idx, item in enumerate(remaining):
                source_bonus = 1 if item.get("source") == "agent" else 0
                min_distance = self._min_distance(item["params"], references)
                avg_distance = self._avg_distance(item["params"], references)
                heuristic_bonus = self._candidate_score(item["params"])
                score = (
                    source_bonus,
                    min_distance,
                    avg_distance,
                    heuristic_bonus,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def propose_round(
        self,
        stage: str,
        round_idx: int,
        target_unique: int,
        exclude_param_keys: set[Tuple[Any, ...]],
        all_trials: Sequence[Dict[str, Any]],
        stage_trials: Sequence[Dict[str, Any]],
        knowledge_full_context: Dict[str, Any] | str | None = None,
        stage_policy: Dict[str, Any] | None = None,
        similar_task_context: Dict[str, Any] | None = None,
        proposal_mode: str = "default",
        allowed_values_override: Dict[str, Any] | None = None,
        llm_seed_source_mode: str = "mixed",
        scbo_reflection: Dict[str, Any] | None = None,
        rule_context: Dict[str, Any] | None = None,
        force_llm: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if stage != "unified":
            raise ValueError(f"Only unified strategy is supported, got stage='{stage}'.")

        target_unique = max(0, int(target_unique))
        proposal_mode = str(proposal_mode or "default").strip().lower()
        if proposal_mode not in {"default", "llm_seed"}:
            raise ValueError("proposal_mode must be one of: default, llm_seed")
        llm_seed_source_mode_requested = str(llm_seed_source_mode or "mixed").strip().lower()
        if llm_seed_source_mode_requested not in {"mixed", "proposer_only"}:
            raise ValueError("llm_seed_source_mode must be one of: mixed, proposer_only")
        llm_seed_source_mode_effective = "proposer_only" if proposal_mode == "llm_seed" else "not_applicable"

        parameter_space = self.space.export_parameter_space()
        proposer_allowed_parameter_space = parameter_space
        if allowed_values_override is not None:
            proposer_allowed_parameter_space = self.space.normalize_constraints(allowed_values_override)

        resolved_stage_policy = self._resolve_unified_policy(stage_policy)
        transfer_context = similar_task_context or {}
        transfer_tasks = transfer_context.get("tasks", []) if isinstance(transfer_context, dict) else []
        if int(round_idx) != 1:
            cold_start_diversity_reason = "not_first_round"
        elif bool(all_trials):
            cold_start_diversity_reason = "history_trials_available"
        elif bool(transfer_tasks):
            cold_start_diversity_reason = "transfer_context_available"
        else:
            cold_start_diversity_reason = "first_round_without_history_or_transfer"
        cold_start_diversity_mode = cold_start_diversity_reason == "first_round_without_history_or_transfer"
        selection_mode = "cold_start_diversity" if cold_start_diversity_mode else "threshold_guided"

        history_summary = self._make_history_summary(stage_trials, limit=12)
        knowledge_payload = knowledge_full_context if knowledge_full_context is not None else {}
        if isinstance(knowledge_payload, str):
            knowledge_context_chars = len(knowledge_payload)
        else:
            knowledge_context_chars = len(json.dumps(knowledge_payload, ensure_ascii=False))

        role_results: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        seen_candidate_keys = set(exclude_param_keys)
        attempts_used = 0
        max_attempts = self.proposer_retry_max_attempts if target_unique > 0 else 0

        if self.enable_agentic and target_unique > 0:
            for attempt_idx in range(1, max_attempts + 1):
                remaining_needed = target_unique - len(candidates)
                if remaining_needed <= 0:
                    break

                proposer_payload = {
                    "stage": stage,
                    "round_idx": round_idx,
                    "target_unique": remaining_needed,
                    "param_order": list(self.space.order),
                    "candidate_schema": self._candidate_schema(),
                    "allowed_values": proposer_allowed_parameter_space,
                    "allowed_parameter_space": proposer_allowed_parameter_space,
                    "history_summary": history_summary,
                    "reflection": self.reflection_memory,
                    "scbo_reflection": scbo_reflection or {},
                    "knowledge_full_context": knowledge_payload,
                    "similar_task_context": transfer_context,
                    "objective": self.objective_preference,
                    "objective_text": resolved_stage_policy["objective_text"],
                    "recall_threshold": resolved_stage_policy.get("recall_threshold"),
                    "recall_slack": resolved_stage_policy.get("recall_slack"),
                    "selection_mode": selection_mode,
                    "cold_start_diversity_mode": cold_start_diversity_mode,
                    "cold_start_diversity_reason": cold_start_diversity_reason,
                }
                proposer = self._call_role("proposer", proposer_payload)
                attempts_used = attempt_idx
                proposed = self._extract_candidates(proposer.get("parsed"))
                sanitized = self._sanitize_candidates(
                    proposed,
                    seen_candidate_keys,
                    rejected,
                    source="agent",
                    allowed_values_override=(
                        proposer_allowed_parameter_space if allowed_values_override is not None else None
                    ),
                )
                for item in sanitized:
                    seen_candidate_keys.add(params_to_key(item["params"], self.space.order))
                candidates.extend(sanitized)
                role_results.append(
                    {
                        "role": "proposer",
                        "attempt": attempt_idx,
                        "requested_count": remaining_needed,
                        "proposed_count": len(proposed),
                        "accepted_count": len(sanitized),
                        "ok": proposer["ok"],
                        "error": proposer["error"],
                    }
                )

                if len(candidates) >= target_unique:
                    break

        candidates = candidates[:target_unique]
        fallback_reason = ""
        if target_unique > 0:
            if not self.enable_agentic:
                fallback_reason = "agentic_disabled"
            elif not candidates:
                fallback_reason = "proposer_failed_or_empty"
            elif len(candidates) < target_unique:
                fallback_reason = "proposer_underfilled_after_retries"

        if target_unique > 0 and len(candidates) < target_unique:
            if force_llm:
                raise RuntimeError(
                    f"LLM proposer underfilled round {round_idx}: {len(candidates)}/{target_unique} "
                    f"candidates after {attempts_used} attempts ({fallback_reason}) — "
                    "rule-based fallback is disabled for LLM-mandatory rounds."
                )
            if llm_seed_source_mode_requested == "proposer_only":
                raise RuntimeError(
                    f"LLM proposer underfilled round {round_idx} in proposer_only seed mode: "
                    f"{len(candidates)}/{target_unique} after {attempts_used} attempts — fallback disabled."
                )
        if target_unique > 0 and len(candidates) < target_unique:
            structured_exclude = set(exclude_param_keys)
            structured_exclude.update(params_to_key(item["params"], self.space.order) for item in candidates)
            if selection_mode == "cold_start_diversity":
                structured_pool = self._build_diversity_structured_pool(
                    target_unique=target_unique,
                    exclude_param_keys=structured_exclude,
                )
                selected_fallback = self._select_diversity_candidates(
                    structured_pool,
                    target_unique=target_unique - len(candidates),
                    stage_trials=stage_trials,
                )
            else:
                structured_pool = self._build_threshold_guided_pool(
                    target_unique=target_unique,
                    exclude_param_keys=structured_exclude,
                    all_trials=all_trials,
                    stage_policy=resolved_stage_policy,
                )
                if proposal_mode == "llm_seed":
                    selected_fallback = self._select_llm_seed_candidates(
                        structured_pool,
                        target_unique=target_unique - len(candidates),
                        stage_trials=stage_trials,
                        diversity_mode=cold_start_diversity_mode,
                    )
                else:
                    selected_fallback = self._select_threshold_guided_candidates(
                        structured_pool,
                        target_unique=target_unique - len(candidates),
                        all_trials=all_trials,
                        stage_policy=resolved_stage_policy,
                    )

            for item in selected_fallback:
                item = dict(item)
                item["source"] = "fallback"
                item["note"] = item.get("note") or "structured fallback"
                key = params_to_key(item["params"], self.space.order)
                if key in structured_exclude:
                    continue
                structured_exclude.add(key)
                candidates.append(item)
                if len(candidates) >= target_unique:
                    break

        rerank_meta = {
            "rerank_applied": False,
            "diagnosis_tags": [],
            "retrieved_rule_ids": [],
            "matched_rule_count": 0,
            "rule_score_components": [],
            "pre_rerank_rank": {},
            "post_rerank_rank": {},
        }
        if target_unique > 0 and candidates:
            candidates, rerank_meta = self._apply_rule_rerank(
                candidates=candidates,
                target_unique=target_unique,
                all_trials=all_trials,
                stage_policy=resolved_stage_policy,
                rule_context=rule_context,
            )

        proposal_log = {
            "stage": stage,
            "round_idx": round_idx,
            "target_unique": target_unique,
            "candidate_count": len(candidates),
            "candidate_source": candidates[0]["source"] if candidates else "none",
            "proposal_mode": proposal_mode,
            "force_llm": bool(force_llm),
            "llm_seed_source_mode": llm_seed_source_mode_effective,
            "llm_seed_source_mode_requested": llm_seed_source_mode_requested,
            "llm_seed_diversity_mode": bool(proposal_mode == "llm_seed" and cold_start_diversity_mode),
            "proposer_retry_max_attempts": max_attempts,
            "proposer_attempts": attempts_used,
            "objective": self.objective_preference,
            "param_order": list(self.space.order),
            "stage_policy": {
                "objective_text": resolved_stage_policy["objective_text"],
                "recall_threshold": resolved_stage_policy.get("recall_threshold"),
                "recall_slack": resolved_stage_policy.get("recall_slack"),
            },
            "parameter_space": parameter_space,
            "allowed_values_override_used": allowed_values_override is not None,
            "allowed_values_override": (
                proposer_allowed_parameter_space if allowed_values_override is not None else None
            ),
            "role_results": role_results,
            "rejected": rejected,
            "knowledge_full_context_chars": int(knowledge_context_chars),
            "similar_tasks_used": len(transfer_tasks),
            "transfer_context_size": len(transfer_context) if isinstance(transfer_context, dict) else 0,
            "selection_mode": selection_mode,
            "cold_start_diversity_mode": cold_start_diversity_mode,
            "cold_start_diversity_reason": cold_start_diversity_reason,
            "fallback_reason": fallback_reason,
            "diagnosis_tags": rerank_meta.get("diagnosis_tags", []),
            "retrieved_rule_ids": rerank_meta.get("retrieved_rule_ids", []),
            "matched_rule_count": int(rerank_meta.get("matched_rule_count", 0)),
            "rerank_applied": bool(rerank_meta.get("rerank_applied", False)),
            "rule_score_components": rerank_meta.get("rule_score_components", []),
            "pre_rerank_rank": rerank_meta.get("pre_rerank_rank", {}),
            "post_rerank_rank": rerank_meta.get("post_rerank_rank", {}),
            "candidates": [
                {
                    "params": item["params"],
                    "note": item.get("note", ""),
                    "source": item.get("source", "agent"),
                }
                for item in candidates
            ],
        }
        return candidates, proposal_log

    def _heuristic_reflection(
        self,
        round_trials: Sequence[Dict[str, Any]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        success = [
            t
            for t in round_trials
            if t.get("status") == "success" and (t.get("metrics") or {}).get("recall") is not None
        ]
        if not success:
            summary = {
                "high_potential": [],
                "failure_modes": ["round_has_no_success"],
                "next_focus": "increase diversity and avoid repeated invalid candidates",
            }
        else:
            best = max(success, key=lambda t: (t["metrics"]["recall"], t["metrics"]["qps"]))
            summary = {
                "high_potential": [best.get("params")],
                "failure_modes": [],
                "next_focus": (
                    f"best round point recall={best['metrics']['recall']:.4f}, "
                    f"qps={best['metrics']['qps']:.2f}; refine near this point"
                ),
            }

        threshold = float(stage_policy["recall_threshold"])
        feasible_points = []
        for trial in stage_trials:
            metrics = trial.get("metrics") or {}
            if trial.get("status") != "success":
                continue
            if metrics.get("recall") is None or metrics.get("qps") is None:
                continue
            if float(metrics["recall"]) < threshold:
                continue
            feasible_points.append(trial)

        if feasible_points:
            fastest = max(feasible_points, key=lambda t: (t["metrics"]["qps"], -t["metrics"]["recall"]))
            summary["threshold_reached"] = True
            summary["fastest_feasible"] = {
                "params": fastest.get("params"),
                "recall": float(fastest["metrics"]["recall"]),
                "qps": float(fastest["metrics"]["qps"]),
            }
            summary["next_focus"] = (
                f"threshold reached; keep recall near {threshold:.4f} while improving qps around the fastest feasible point"
            )
        else:
            summary["threshold_reached"] = False
            summary["fastest_feasible"] = None
        return summary

    def reflect_round(
        self,
        stage: str,
        round_idx: int,
        round_trials: Sequence[Dict[str, Any]],
        stage_trials: Sequence[Dict[str, Any]],
        stage_policy: Dict[str, Any] | None = None,
        selection_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if stage != "unified":
            raise ValueError(f"Only unified strategy is supported, got stage='{stage}'.")
        resolved_stage_policy = self._resolve_unified_policy(stage_policy)
        resolved_selection_context = {
            "selection_mode": "threshold_guided",
            "cold_start_diversity_mode": False,
            "cold_start_diversity_reason": "not_provided",
        }
        if isinstance(selection_context, dict):
            if "selection_mode" in selection_context:
                resolved_selection_context["selection_mode"] = str(selection_context.get("selection_mode"))
            if "cold_start_diversity_mode" in selection_context:
                resolved_selection_context["cold_start_diversity_mode"] = bool(
                    selection_context.get("cold_start_diversity_mode")
                )
            if "cold_start_diversity_reason" in selection_context:
                resolved_selection_context["cold_start_diversity_reason"] = str(
                    selection_context.get("cold_start_diversity_reason")
                )

        round_summary = self._make_history_summary(round_trials, limit=20)
        reflector_result = None

        if self.enable_agentic and round_summary:
            payload = {
                "round_results": round_summary,
                "stage": "unified",
                "round_idx": round_idx,
                "history_summary": self._make_history_summary(stage_trials, limit=30),
                "reflection": self.reflection_memory,
                "objective_text": resolved_stage_policy["objective_text"],
                "recall_threshold": resolved_stage_policy.get("recall_threshold"),
                "recall_slack": resolved_stage_policy.get("recall_slack"),
                "selection_mode": resolved_selection_context["selection_mode"],
                "cold_start_diversity_mode": resolved_selection_context["cold_start_diversity_mode"],
                "cold_start_diversity_reason": resolved_selection_context["cold_start_diversity_reason"],
                "selection_context": resolved_selection_context,
            }
            reflector_result = self._call_role("reflector", payload)

        parsed = reflector_result.get("parsed") if reflector_result else None
        if not isinstance(parsed, dict):
            parsed = self._heuristic_reflection(round_trials, stage_trials, resolved_stage_policy)

        summary = {
            "high_potential": parsed.get("high_potential", [])[:5],
            "failure_modes": parsed.get("failure_modes", [])[:5],
            "next_focus": str(parsed.get("next_focus", "")).strip(),
            "threshold_reached": bool(parsed.get("threshold_reached", False)),
            "fastest_feasible": parsed.get("fastest_feasible"),
        }

        heuristic_summary = self._heuristic_reflection(round_trials, stage_trials, resolved_stage_policy)
        summary["threshold_reached"] = heuristic_summary["threshold_reached"]
        summary["fastest_feasible"] = heuristic_summary["fastest_feasible"]
        if not summary["next_focus"]:
            summary["next_focus"] = heuristic_summary["next_focus"]

        self.reflection_memory = summary

        return {
            "stage": "unified",
            "round_idx": round_idx,
            "summary": summary,
            "reflector_ok": reflector_result["ok"] if reflector_result else False,
            "reflector_error": reflector_result["error"] if reflector_result else None,
        }


__all__ = [
    "PARAM_ORDER",
    "ParameterSpace",
    "RFANNSTuningAgent",
    "compute_pareto_front",
    "key_to_params",
    "params_to_key",
]

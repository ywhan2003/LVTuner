"""DiskANN Filtered-Vamana tuning agent.

Extends RFANNSTuningAgent with DiskANN-specific parameter semantics,
constraint validation, selectivity-aware logic, and skill templates.
"""

import random
from typing import Any, Dict, List, Sequence, Set, Tuple

from agents.rfanns_agent import (
    PARAM_ORDER as _RFANNS_PARAM_ORDER,
    RFANNSTuningAgent,
    params_to_key,
)


DISKANN_BUILD_PARAM_ORDER = ["R", "Lbuild", "FilteredLBuild", "alpha"]
DISKANN_SEARCH_PARAMS = ["L"]
DISKANN_PARAM_ORDER = DISKANN_BUILD_PARAM_ORDER + DISKANN_SEARCH_PARAMS


class DiskANNTuningAgent(RFANNSTuningAgent):
    """Tuning agent for Filtered-DiskANN in-memory index.

    Adds DiskANN-specific:
    - Parameter order (R, Lbuild, FilteredLBuild, alpha, L)
    - Hard constraint validation (FilteredLBuild >= Lbuild, alpha >= 1.0)
    - Selectivity-aware parameter range guidance
    - DiskANN-specific parameter semantics for LLM prompts
    """

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
        super().__init__(
            params_cfg=params_cfg,
            seed=seed,
            param_order=param_order or DISKANN_PARAM_ORDER,
            agentic_cfg=agentic_cfg,
            model_cfg=model_cfg,
            prompt_cfg=prompt_cfg,
            llm_caller=llm_caller,
            objective_preference=objective_preference,
        )
        self.build_order = DISKANN_BUILD_PARAM_ORDER
        self.search_params = DISKANN_SEARCH_PARAMS

    def _build_alias_to_canonical(self) -> Dict[str, str]:
        aliases = super()._build_alias_to_canonical()
        # Remove RFANNS-specific aliases that don't apply
        for key in list(aliases.keys()):
            if key in ("num_slots", "b", "ef_construction", "efConstruction"):
                aliases.pop(key, None)
        return aliases

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate DiskANN parameter constraints. Returns list of error messages."""
        errors: List[str] = []

        if "FilteredLBuild" in params and "Lbuild" in params:
            flb = int(params["FilteredLBuild"])
            lb = int(params["Lbuild"])
            if flb < lb:
                errors.append(
                    f"FilteredLBuild ({flb}) must be >= Lbuild ({lb})"
                )

        if "alpha" in params:
            alpha = float(params["alpha"])
            if alpha < 1.0:
                errors.append(f"alpha ({alpha}) must be >= 1.0")

        if "R" in params:
            r = int(params["R"])
            if r < 2:
                errors.append(f"R ({r}) must be >= 2")

        return errors

    def _candidate_schema(self) -> Dict[str, Any]:
        schema = super()._candidate_schema()
        # Add diskann parameter descriptions to the schema
        param_descriptions = {
            "R": "Maximum graph degree (build param, higher=better recall, lower=better QPS, requires rebuild)",
            "Lbuild": "Build-time search list size (build param, higher=better graph quality, requires rebuild)",
            "FilteredLBuild": "Filter-specific build search width (build param, must be >= Lbuild, higher=better filtered recall, requires rebuild)",
            "alpha": "Vamana pruning parameter (build param, lower=more edges/better recall, higher=faster search, requires rebuild, >=1.0)",
            "L": "Search list size (search param ONLY, higher=better recall, lower=better QPS, NO rebuild needed)",
        }
        for param_name, desc in param_descriptions.items():
            if param_name in schema.get("properties", {}):
                schema["properties"][param_name]["description"] = desc
        return schema

    def build_selectivity_seeds(
        self, selectivity: float | None = None
    ) -> List[Dict[str, Any]]:
        """Generate seed candidates based on selectivity regime.

        These are sensible starting points informed by DiskANN domain knowledge.
        """
        seeds: List[Dict[str, Any]] = []

        if selectivity is not None and selectivity < 0.1:
            # High selectivity: need dense graph for post-filter connectivity
            seeds = [
                {"R": 48, "Lbuild": 80, "FilteredLBuild": 100, "alpha": 1.0, "L": 60},
                {"R": 64, "Lbuild": 100, "FilteredLBuild": 100, "alpha": 1.1, "L": 50},
                {"R": 48, "Lbuild": 60, "FilteredLBuild": 80, "alpha": 1.0, "L": 80},
            ]
        elif selectivity is not None and selectivity < 0.5:
            # Medium selectivity: balanced
            seeds = [
                {"R": 32, "Lbuild": 60, "FilteredLBuild": 70, "alpha": 1.1, "L": 40},
                {"R": 48, "Lbuild": 80, "FilteredLBuild": 90, "alpha": 1.2, "L": 30},
                {"R": 32, "Lbuild": 40, "FilteredLBuild": 50, "alpha": 1.0, "L": 60},
            ]
        else:
            # Low selectivity / unknown: treat like near-unfiltered
            seeds = [
                {"R": 32, "Lbuild": 50, "FilteredLBuild": 60, "alpha": 1.2, "L": 40},
                {"R": 16, "Lbuild": 40, "FilteredLBuild": 50, "alpha": 1.4, "L": 30},
                {"R": 32, "Lbuild": 60, "FilteredLBuild": 70, "alpha": 1.1, "L": 50},
                {"R": 48, "Lbuild": 80, "FilteredLBuild": 90, "alpha": 1.2, "L": 20},
            ]

        # Validate and canonicalize
        validated: List[Dict[str, Any]] = []
        for seed in seeds:
            try:
                canonical = self.canonicalize(seed)
                errors = self.validate_params(canonical)
                if not errors:
                    validated.append(canonical)
            except Exception:
                continue
        return validated

    def initial_design_candidates(
        self,
        target_count: int,
        exclude_param_keys: Set[Tuple[Any, ...]] | None = None,
        selectivity: float | None = None,
        stage_trials: Sequence[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate initial design candidates with diskann-aware seeding."""
        excluded = set(exclude_param_keys or ())

        # Start with selectivity-aware seeds
        seeds = self.build_selectivity_seeds(selectivity=selectivity)
        candidates: List[Dict[str, Any]] = []
        for seed in seeds:
            key = params_to_key(seed, self.space.order)
            if key not in excluded:
                candidates.append({"params": seed, "source": "diskann_selectivity_seed", "note": ""})
                excluded.add(key)
            if len(candidates) >= target_count:
                break

        # Fill remaining with parent's grid-based coverage
        if len(candidates) < target_count:
            remaining = target_count - len(candidates)
            grid_candidates = super().initial_design_candidates(
                target_count=remaining,
                exclude_param_keys=excluded,
                stage_trials=stage_trials,
            )
            candidates.extend(grid_candidates)

        return candidates[:target_count]


__all__ = [
    "DISKANN_BUILD_PARAM_ORDER",
    "DISKANN_SEARCH_PARAMS",
    "DISKANN_PARAM_ORDER",
    "DiskANNTuningAgent",
]

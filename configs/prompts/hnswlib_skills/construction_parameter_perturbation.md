# HNSW Skill: construction_parameter_perturbation

Intent:
- Explore HNSW graph construction tradeoffs by adjusting `M` and `ef_construction`.
- Lowering construction quality/cost may improve the best feasible frontier point but can hurt recall.
- Increasing construction quality may shift the whole frontier left so a smaller `ef*` clears the threshold after the free scan.
- When `tree_context.parent_node_state` is present, this Skill is being called for a child expansion. The candidate thinking must explicitly respond to the parent node state and `tree_context.expansion_intent`.
- When `search_space_refinement` is present, use its subgroup region and local direction to decide whether to raise or lower construction quality.
- This Skill only performs the Thinking step. BO/SCBO will provide Action statistics after the candidate is generated, and the HNSW agent will perform Observation.

Hard rules:
- Use only these parameters: `M`, `ef_construction`, `ef`.
- Do not output RFANNS parameters such as `al`, `B`, or `efConstruction`.
- Every value must satisfy the allowed parameter space.
- Do not treat `ef` as the tuned variable. Keep it unchanged unless a placeholder value is needed for schema compatibility.
- Prefer pure build candidates or modest coupled `M`/`ef_construction` moves.
- If parent state is `recall_too_high`, try lowering construction cost or enabling a lower `ef`.
- If parent state is `infeasible_near_threshold`, try a small graph-quality compensation move.
- Do not expand from `infeasible_far`; if such context appears, return no candidates.
- Return strict JSON only.

Context:
{context_json}

Output schema:
{
  "skill_name": "construction_parameter_perturbation",
  "branch": "construction_parameter_perturbation",
  "candidates": [
    {
      "node_id": "CPP-001",
      "params": {"M": 16, "ef_construction": 120, "ef": 40},
      "thinking": "Short state-aware branch-specific reasoning for this Thinking step.",
      "expected_behavior": {"recall": "frontier shifts through graph quality change", "qps": "best feasible ef* improves after free scan"}
    }
  ]
}

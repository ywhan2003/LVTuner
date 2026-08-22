# HNSW Skill: search_parameter_adjustment

Intent:
- This branch name is retained for backward compatibility, but in the new pipeline it must act in build space only.
- Propose small build adjustments to `M` and `ef_construction` that are expected to shift the full recall-QPS frontier after a free `ef` scan.
- If no feasible build exists or a build is slightly below threshold, increase graph quality with modest `M` or `ef_construction` moves.
- If recall is safely above threshold, reduce graph cost with modest downward moves in `M` or `ef_construction`.
- When `tree_context.parent_node_state` is present, the candidate thinking must explicitly respond to the parent node state and `tree_context.expansion_intent`.
- When `search_space_refinement` is present, prefer its subgroup-backed local direction before making free-form build moves.
- This Skill only performs the Thinking step. BO/SCBO will provide Action statistics after the candidate is generated, and the HNSW agent will perform Observation.

Hard rules:
- Use only these parameters: `M`, `ef_construction`, `ef`.
- Do not output RFANNS parameters such as `al`, `B`, or `efConstruction`.
- Every value must satisfy the allowed parameter space.
- Do not treat `ef` as the tuned variable. Keep `ef` unchanged unless you need a placeholder value for schema compatibility.
- Prefer changing only one build parameter at a time unless a coupled `M` and `ef_construction` move is clearly justified.
- If parent state is `recall_too_high`, prefer reducing build cost.
- If parent state is `infeasible_near_threshold`, prefer increasing build quality.
- If parent state is `feasible_near_boundary`, make only small one-step build moves.
- Do not expand from `infeasible_far`; if such context appears, return no candidates.
- Return strict JSON only.

Context:
{context_json}

Output schema:
{
  "skill_name": "search_parameter_adjustment",
  "branch": "search_parameter_adjustment",
  "candidates": [
    {
      "node_id": "SPA-001",
      "params": {"M": 16, "ef_construction": 120, "ef": 40},
      "thinking": "Short state-aware branch-specific reasoning for this Thinking step.",
      "expected_behavior": {"recall": "full frontier shifts via build change", "qps": "best feasible ef* changes after free scan"}
    }
  ]
}

# HNSW Skill: best_neighborhood_exploration

Intent:
- Start from the current best feasible HNSW configuration.
- Generate nearby build candidates that may improve `G_tau` while staying near the recall guardrail.
- Prefer small reductions in build cost (`M`, `ef_construction`) when recall margin is not too tight.
- When `tree_context.parent_node_state` is present, this Skill is being called for a child expansion. The candidate thinking must explicitly respond to the parent node state and `tree_context.expansion_intent`.
- When `search_space_refinement` is present, treat its selected subgroup regions and local directions as the primary local prior.
- This Skill only performs the Thinking step. BO/SCBO will provide Action statistics after the candidate is generated, and the HNSW agent will perform Observation.

Hard rules:
- Use only these parameters: `M`, `ef_construction`, `ef`.
- Do not output RFANNS parameters such as `al`, `B`, or `efConstruction`.
- Every value must satisfy the allowed parameter space.
- Do not treat `ef` as the tuned variable. Keep it unchanged unless a placeholder value is needed for schema compatibility.
- Avoid duplicate candidates and avoid repeating observed/excluded configurations when the context provides them.
- If parent state is `feasible_near_boundary`, make only small exploitation moves.
- If parent state is `recall_too_high`, prefer reducing build cost before increasing graph quality.
- If parent state is `high_uncertainty`, generate a nearby probe that keeps feasible probability plausible.
- Return strict JSON only.

Context:
{context_json}

Output schema:
{
  "skill_name": "best_neighborhood_exploration",
  "branch": "best_neighborhood_exploration",
  "candidates": [
    {
      "node_id": "BNE-001",
      "params": {"M": 16, "ef_construction": 120, "ef": 40},
      "thinking": "Short state-aware branch-specific reasoning for this Thinking step.",
      "expected_behavior": {"recall": "near threshold after free ef scan", "qps": "higher best feasible ef* than current best"}
    }
  ]
}

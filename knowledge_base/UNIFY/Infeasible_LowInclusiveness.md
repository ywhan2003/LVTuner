Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.LowInclusiveness

Description:
    Inclusiveness (edge overlap of the per-slot subgraphs with the ideal per-range HNSW
    graph) is low — below 70%. The slot granularity is too coarse or the per-slot density
    is too low, so the filtered graphs no longer preserve the neighborhood structure.

Signals:
    - Recall margin < 0
    - inclusiveness_pct is low (below 70)
    - B is large
    - M is low or moderate

Interpretation:
    Large B spreads the data thinly across slots, so each slot's M edges are insufficient to
    maintain graph quality. The fragmentation directly limits achievable recall.

Solution:
    - large recall gap: decrease B by 1-2, then increase M by 10-20, efC by 100-250
    - small recall gap: decrease B by 1, then increase M by 4-8

Cost:
    Fewer slots means coarser filtering — the filter region contains more non-matching
    points, raising search cost per query.

Boundary:
    Do not raise B while inclusiveness is below 70%. B = 1 removes filtering entirely
    (plain HNSW behavior).

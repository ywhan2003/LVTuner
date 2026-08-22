Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.ConstructionQualityEnablesEfReduction

Description:
    The configuration is feasible but relies on a high ef, causing high query-time cost. The
    graph itself is weak (low M, low ef_construction): Recall is being bought with expensive
    query-time search instead of construction quality.

Signals:
    - Recall is feasible but requires high ef
    - QPS is low
    - efC is low
    - M is low

Interpretation:
    Recall is maintained by search work rather than by a good graph. Investing in build quality
    can shift cost from query time to build time: a better graph needs a smaller ef to reach
    the same Recall, so ef* drops and QPS rises.

Solution:
    - large recall margin: increase efC by 100-250, then probe ef down by 20-150
    - small recall margin: increase efC by 20-50, then probe ef down by 5-10

Cost:
    Build time grows roughly linearly with efC; query-time QPS is unaffected by efC directly.

Boundary:
    If raising efC does not lower the smallest feasible efS (U) at the same M, efC is saturated
    for this graph — stop increasing it and adjust ef/M instead.

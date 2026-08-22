Recall Constraint Class:
    Recall-feasible

Pattern ID:
    NHQ.ExcessiveSearchEffort

Description:
    Recall is above the threshold but distance computations per query are high: the search
    beam is larger than the graph needs. The extra search work directly lowers QPS.

Signals:
    - Recall margin > 0 with slack (recall is feasible)
    - dist_comps_per_query is high
    - qps is low

Interpretation:
    The query performs unnecessary distance computations. Since ef is a zero-rebuild lever,
    reducing it is the cheapest and safest QPS improvement.

Solution:
    - large recall margin: decrease ef by 50-500
    - small recall margin: decrease ef by 10-50

Cost:
    Too-large ef reductions drop recall below the threshold.

Boundary:
    Keep the effective beam (ef/2) above the effective_k (10) — ef < 20 is invalid.
    If recall collapses when weight > 0, lower weight before lowering ef.

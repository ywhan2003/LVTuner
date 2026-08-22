Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.GraphUnderConnectivity

Description:
    Recall is below the threshold and the per-slot graphs are under-connected: with low M,
    each node keeps too few edges to route queries to true nearest neighbors. Search width
    alone cannot compensate.

Signals:
    - Recall margin < 0
    - M is low
    - avg_out_degree is low
    - increasing ef alone gives limited improvement

Interpretation:
    The graph lacks the edges needed for navigation. The fix is structural: more edges per
    slot (M), better chosen via a larger build beam (efConstruction).

Solution:
    - large recall gap: increase M by 10-20, then efC by 100-250
    - small recall gap: increase M by 4-8, then efC by 20-50

Cost:
    Larger M grows memory and build time; per-slot edge count scales with M x (dataset/B).

Boundary:
    If B is large and slots are sparse, reduce B instead of raising M alone — sparse slots
    waste edges (see LowInclusiveness).

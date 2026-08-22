Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    NHQ.GraphQualityInsufficient

Description:
    Recall is below the threshold even though the search already performs many distance
    computations: the graph itself is too weak. More search effort cannot compensate — the
    construction (M, efConstruction) must be strengthened.

Signals:
    - Recall margin < 0
    - dist_comps_per_query is high
    - increasing ef gives weak or no recall gain

Interpretation:
    The query explores many candidates but the graph does not route toward true nearest
    neighbors. The bottleneck is construction quality or connectivity.

Solution:
    - large recall gap: increase M by 10-20, then efC by 100-300
    - small recall gap: increase M by 4-8, then efC by 50-100

Cost:
    M grows memory and build time; efConstruction grows build time.

Boundary:
    On SIFT-like datasets with selectivity 0.01, keep efConstruction <= 100 — efConstruction
    >= 150 causes catastrophic recall collapse regardless of M and ef.

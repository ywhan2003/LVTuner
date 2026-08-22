Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    FilterDiskANN.GraphQualityInsufficient

Description:
    Recall is below the threshold even though the search already performs many distance
    computations: the graph navigation is poor. More search effort cannot compensate — the
    construction (R, FilterLBuild) must be strengthened.

Signals:
    - Recall margin < 0
    - dist_comps_per_query is high
    - increasing L gives weak or no recall gain

Interpretation:
    The query evaluates many candidates but the graph does not route toward true nearest
    neighbors. The bottleneck is build quality: R (degree) or FilterLBuild (build beam,
    also the ceiling for L).

Solution:
    - large recall gap: increase FilterLBuild by 80-250, then R by 16-32
    - small recall gap: increase FilterLBuild by 20-80, then R by 4-8

Cost:
    FilterLBuild grows build time; R grows memory.

Boundary:
    Keep alpha >= 1.0 — lowering alpha preserves more edges and helps recall, but at QPS
    cost. Remember L <= FilterLBuild: raising FilterLBuild also raises the usable L range.

Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    FilterDiskANN.InfeasibleLowSearchExploration

Description:
    Recall is below the threshold while the query visits few nodes and performs few
    distance computations: the search list L is too small to find good candidates. This is
    the cheapest kind of recall shortfall — L is a zero-rebuild lever.

Signals:
    - Recall margin < 0 (recall is below the threshold)
    - visited_nodes_per_query is low
    - dist_comps_per_query is low
    - L is low

Interpretation:
    The query terminates before exploring enough candidates. Increasing L is the safest
    first repair action.

Solution:
    - large recall gap: increase L by 10-100
    - small recall gap: increase L by 5-10

Cost:
    Higher L lowers QPS proportionally to the added search work.

Boundary:
    Keep L <= FilterLBuild. If alpha > 1.0, reducing it toward 1.0 gives a free recall
    gain.

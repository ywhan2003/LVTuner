Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    NHQ.LowSearchExploration

Description:
    Recall is below the threshold while distance computations are low: the search beam is
    too small to find good candidates. This is the cheapest kind of recall shortfall — ef is
    a zero-rebuild lever.

Signals:
    - Recall margin < 0 (recall is below the threshold)
    - dist_comps_per_query is low
    - ef is low

Interpretation:
    The query terminates before exploring enough candidates. Increasing ef (effective beam
    = ef/2, propose an even value) is the safest first repair action.

Solution:
    - large recall gap: increase ef by 50-500
    - small recall gap: increase ef by 10-50

Cost:
    Higher ef lowers QPS proportionally to the added search work.

Boundary:
    Keep the effective beam above the effective_k (10). If weight > 0 and recall collapsed,
    reduce weight toward 0 first — see WeightRecallCollapse.

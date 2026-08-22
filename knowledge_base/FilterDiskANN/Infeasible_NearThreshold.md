Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    FilterDiskANN.InfeasibleNearThreshold

Description:
    Recall is only slightly below the threshold: the configuration is nearly feasible and a
    small search-effort compensation is likely to cross the guardrail without a rebuild.

Signals:
    - Recall margin < 0 and near the threshold (margin -0.005 to 0)
    - recall is below the threshold

Interpretation:
    The gap to feasibility is small. A modest L increase on the SAME build is the cheapest
    repair; a rebuild is only justified if L is already at its ceiling (FilterLBuild).

Solution:
    - large recall gap: if the gap grows: increase L by 10-100
    - small recall gap: increase L by 5-10

Cost:
    Small QPS loss from the search-effort increase.

Boundary:
    Change one parameter at a time. Keep L <= FilterLBuild.

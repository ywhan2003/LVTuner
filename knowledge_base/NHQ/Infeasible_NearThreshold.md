Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    NHQ.InfeasibleNearThreshold

Description:
    Recall is only slightly below the threshold: the configuration is nearly feasible and a
    small search-effort compensation is likely to cross the guardrail without a rebuild.

Signals:
    - Recall margin < 0 and near the threshold (margin -0.005 to 0)
    - recall is below the threshold

Interpretation:
    The gap to feasibility is small. A modest ef increase on the SAME build is the cheapest
    repair; a rebuild is only justified if search tuning proves insufficient.

Solution:
    - large recall gap: if the gap grows: increase ef by 50-500
    - small recall gap: increase ef by 10-50

Cost:
    Small QPS loss from the search-effort increase.

Boundary:
    Change one parameter at a time. Keep the effective beam above the effective_k (10).

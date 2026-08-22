Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.InfeasibleNearThreshold

Description:
    Recall is only slightly below the threshold: the configuration is nearly feasible and a
    small compensation move is likely to cross the guardrail. Conservative, search-first
    repair is the right strategy.

Signals:
    - Recall margin < 0 and near the threshold (margin -0.005 to 0)
    - recall is below the threshold

Interpretation:
    The gap to feasibility is small. The cheapest repair is a modest ef/al increase on the
    SAME build (no rebuild); a rebuild is only justified if search tuning proves
    insufficient.

Solution:
    - large recall gap: if the gap grows: increase ef by 20-150, al by 32-64
    - small recall gap: increase ef by 5-10, al by 8-16

Cost:
    Small QPS loss from the search-effort increase.

Boundary:
    Change one parameter at a time. If the same small repair repeatedly fails, escalate to
    ConstructionUnderPrecision.

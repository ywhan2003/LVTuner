Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    HNSW.InfeasibleNearThreshold

Description:
    Recall is only slightly below the threshold: the configuration is nearly feasible and a
    small compensation move is likely to cross the guardrail. Conservative, single-axis repair
    steps are the right strategy; structural changes are unnecessary and wasteful here.

Signals:
    - Recall margin < 0 and near the threshold (margin -0.005 to 0)
    - recall is below the threshold

Interpretation:
    The gap to feasibility is small. The cheapest repair is a modest ef increase (zero-rebuild);
    if that is insufficient, a small construction compensation (M +1 / efC +1) is the fallback.
    Keep the step size in the NEAR tier: |delta ef| no larger than 10.

Solution:
    - large recall gap: if the gap grows: increase ef by 20-150
    - small recall gap: increase ef by 5-10; M by 4-8 if needed

Cost:
    Small QPS loss from the ef increase; a rebuild costs build time if compensation is needed.

Boundary:
    Prefer single-axis moves; change M and ef together only when the interval table supports
    both. If the same repair repeatedly fails, escalate to the far-below structural card.

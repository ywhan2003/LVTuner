Recall Constraint Class:
    Recall-feasible

Pattern ID:
    NHQ.FeasibleFarAboveSlack

Description:
    The latest trial has a large positive recall margin: the configuration is conservative
    and QPS is wasted. Large recall slack can be spent aggressively — recall may drop by
    the entire margin and still meet the target.

Signals:
    - Recall margin > 0 with large slack (margin above 0.02)
    - recall is feasible

Interpretation:
    Recall above the target is wasted in NHQ tuning: the objective is QPS under the recall
    constraint. Search parameters (ef, weight) are zero-rebuild levers; construction
    parameters (M, efConstruction) require rebuilds.

Solution:
    - large recall margin: decrease ef first (zero rebuild) by 50-500; then M by 10-20, efC by 100-300, or increase weight by 20-50
    - small recall margin: decrease ef by 10-50

Cost:
    Aggressive reductions may lose recall without proportional QPS gain; M/efC changes cost
    rebuild time.

Boundary:
    ef below ~20 gives an effective beam below the effective_k (10) — never propose that.
    Weight beyond 100 degrades recall quickly and beyond 1000 collapses it.

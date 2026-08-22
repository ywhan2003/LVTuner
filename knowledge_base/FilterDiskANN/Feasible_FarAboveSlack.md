Recall Constraint Class:
    Recall-feasible

Pattern ID:
    FilterDiskANN.FeasibleFarAboveSlack

Description:
    The latest trial has a large positive recall margin: the configuration is conservative
    and QPS is wasted. Large recall slack can be spent aggressively — recall may drop by
    the entire margin and still meet the target.

Signals:
    - Recall margin > 0 with large slack (margin above 0.02)
    - recall is feasible

Interpretation:
    Recall above the target is wasted in Filter-DiskANN tuning: the objective is QPS under
    the recall constraint. L is the only search knob (zero-rebuild); R, FilterLBuild and
    alpha require rebuilds.

Solution:
    - large recall margin: decrease L first (zero rebuild) by 10-100; then R by 16-32, FilterLBuild by 80-250, or increase alpha by 0.05-0.2
    - small recall margin: decrease L by 5-10

Cost:
    Aggressive reductions may lose recall without proportional QPS gain; rebuilds cost
    build time.

Boundary:
    Keep L <= FilterLBuild and alpha >= 1.0 (Vamana pruning constraint). If alpha was
    raised for QPS and recall later drops, reduce it toward 1.0 for a free recall gain.

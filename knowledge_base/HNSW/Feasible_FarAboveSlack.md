Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.FeasibleFarAboveSlack

Description:
    The latest trial has a large positive Recall margin: the configuration is conservative and
    QPS is leaving performance on the table. Large Recall slack can be spent aggressively on
    QPS optimization.

Signals:
    - Recall margin > 0 with large slack (margin above 0.02)
    - recall is feasible

Interpretation:
    The configuration over-satisfies the target. Recall slack is a resource: spend it on
    reducing search cost first (zero-rebuild) and then on reducing construction cost if more
    QPS is needed. ef is always the first lever; M/efC changes require rebuilds.

Solution:
    - large recall margin: decrease ef first (zero rebuild) by 20-150; then M by 10-20, efC by 100-250
    - small recall margin: decrease ef by 5-10; M by 4-8

Cost:
    Aggressive reductions may lose Recall without proportional QPS gain; M/efC reductions cost
    rebuild time.

Boundary:
    At least one parameter must decrease (all three increasing is not a QPS move). When the
    knowledge cards and the current-task data disagree, trust the data.

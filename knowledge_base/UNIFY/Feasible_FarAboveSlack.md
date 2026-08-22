Recall Constraint Class:
    Recall-feasible

Pattern ID:
    UNIFY.FeasibleFarAboveSlack

Description:
    The latest trial has a large positive recall margin: the configuration is conservative
    and QPS is leaving performance on the table. Large recall slack can be spent aggressively.

Signals:
    - Recall margin > 0 with large slack (margin above 0.02)
    - recall is feasible

Interpretation:
    Recall slack is a spendable resource. Spend it on search parameters first (zero-rebuild:
    ef and al), then on construction parameters if more QPS is needed.

Solution:
    - large recall margin: decrease ef/al first (zero rebuild): ef by 20-150, al by 32-64; then M by 10-20, efC by 100-250
    - small recall margin: decrease ef by 5-10, al by 8-16

Cost:
    Aggressive reductions may lose recall without proportional QPS gain; rebuilds cost build
    time.

Boundary:
    At least one search parameter should decrease (all-increase is not a QPS move). If
    inclusiveness drops below 70% after reducing B, restore the previous B.

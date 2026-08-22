Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    NHQ.InfeasibleFarBelowStructural

Description:
    Recall is far below the threshold: the gap is structural. Search effort alone may not
    close it — repair order matters: weight first, then ef, then construction.

Signals:
    - Recall margin < 0 and far below the threshold (margin below -0.02)
    - recall is below the threshold

Interpretation:
    A large negative margin means small compensation will not suffice. If weight > 0 and
    recall collapsed, weight is the likely cause (pure-L2 GT). Otherwise raise ef first
    (zero-rebuild); if the same build repeatedly fails, the graph is structurally incapable
    and a rebuild is mandatory.

Solution:
    - large recall gap: increase ef first by 50-500; if still failing, rebuild with M +10-20, efC +100-300
    - small recall gap: increase ef by 10-50

Cost:
    Structural repair costs rebuild time; repeated infeasible configurations waste the
    trial budget.

Boundary:
    On SIFT-like selectivity-0.01 datasets, keep efConstruction <= 100 (>= 150 collapses
    recall regardless of M/ef). Watch budget exhaustion: reassess the threshold if one
    rebuild does not close the gap.

Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    FilterDiskANN.InfeasibleFarBelowStructural

Description:
    Recall is far below the threshold: the gap is structural. Search effort alone may not
    close it — repair order matters: L first (zero-rebuild), then construction.

Signals:
    - Recall margin < 0 and far below the threshold (margin below -0.02)
    - recall is below the threshold

Interpretation:
    A large negative margin means small compensation will not suffice. Raise L first; if L
    is at its ceiling (FilterLBuild) or the same build repeatedly fails, the graph is
    structurally incapable and a rebuild is mandatory.

Solution:
    - large recall gap: increase L first by 10-100; if still failing, rebuild with FilterLBuild +80-250, R +16-32
    - small recall gap: increase L by 5-10

Cost:
    Structural repair costs rebuild time; repeated infeasible configurations waste the
    trial budget.

Boundary:
    Keep L <= FilterLBuild and alpha >= 1.0. Watch budget exhaustion: reassess the
    threshold if one rebuild does not close the gap.

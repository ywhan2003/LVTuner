Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.InfeasibleFarBelowStructural

Description:
    Recall is far below the threshold: the gap is structural, not a small search-width
    shortfall. Aggressive repair is needed, combining search effort on the existing build
    with a construction-quality rebuild.

Signals:
    - Recall margin < 0 and far below the threshold (margin below -0.02)
    - recall is below the threshold

Interpretation:
    A large negative margin means small compensation will not suffice. Raise search effort
    first (zero-rebuild); if the same build repeatedly fails across ef/al values, the graph
    itself is structurally incapable and a rebuild is mandatory.

Solution:
    - large recall gap: increase ef/al first: ef by 20-150, al by 32-64; if still failing, rebuild with M +10-20, efC +100-250
    - small recall gap: increase ef by 5-10, al by 8-16

Cost:
    Structural repair costs rebuild time; repeated infeasible configurations waste the
    trial budget.

Boundary:
    Watch budget exhaustion: if the gap stays structural after one rebuild, reassess
    feasibility of the threshold before spending more rounds.

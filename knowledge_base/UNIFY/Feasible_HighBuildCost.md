Recall Constraint Class:
    Recall-feasible

Pattern ID:
    UNIFY.HighBuildCost

Description:
    Recall is above the threshold but build time is excessive. Construction cost scales
    roughly as B x M x efConstruction, so the build parameters — not the search parameters —
    dominate the index time.

Signals:
    - Recall is feasible (margin >= 0)
    - build_time_s is high
    - efConstruction is high

Interpretation:
    The index is over-built for the recall target: construction quality exceeds what the
    query workload needs. Build cost can be reduced without touching query-time behavior.

Solution:
    - large recall margin: decrease efC by 100-250, then M by 10-20
    - small recall margin: decrease efC by 20-50, then M by 4-8

Cost:
    Cutting construction parameters too far lowers graph quality and may require another
    rebuild to repair recall.

Boundary:
    If recall is far below the threshold while build time is high, the direction reverses:
    reduce B (sparser slots waste build effort) but INCREASE M and efConstruction.

Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.ConstructionUnderPrecision

Description:
    Recall is below the threshold because construction quality is insufficient: the graph
    edges (M per slot) or the build-time candidate search (efConstruction) is too weak to
    preserve the per-range neighborhood structure.

Signals:
    - Recall margin < 0 (recall is below the threshold)
    - M is low
    - efConstruction is low
    - inclusiveness_pct is low

Interpretation:
    The build did not invest enough in edge quality, so no search effort can fully
    compensate. Construction precision must be raised before search tuning helps.

Solution:
    - large recall gap: increase M by 10-20, then efC by 100-250
    - small recall gap: increase M by 4-8, then efC by 20-50

Cost:
    Rebuild cost grows with B x M x efConstruction.

Boundary:
    If inclusiveness is below 70%, reduce B FIRST (slot fragmentation) before raising M —
    see the LowInclusiveness card.

Recall Constraint Class:
    Recall-feasible

Pattern ID:
    UNIFY.NearBoundaryUnsafeReduction

Description:
    The latest trial meets the recall threshold with only a small positive margin: the
    configuration sits at the guardrail. Any reduction is risky — small, evidence-based
    moves are the only safe strategy in this band.

Signals:
    - Recall margin > 0 with small slack (margin near the threshold, 0 to 0.005)
    - recall is feasible

Interpretation:
    There is little recall slack to spend. Near-boundary reductions frequently push the
    configuration below the threshold, so aggressive QPS moves (especially rebuilds) are
    not justified here.

Solution:
    - large recall margin: if the margin grows, follow FarAboveSlack: decrease ef by 20-150, al by 32-64
    - small recall margin: single small moves: M by 4-8, ef by 5-10, al by 8-16

Cost:
    Any reduction can violate the threshold; recovery costs at least one round per failed move.

Boundary:
    Reductions are unsafe when |margin| is within 0.01. Do not change M and B together here,
    and do not rebuild for QPS alone.

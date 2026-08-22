Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.FeasibleNearBoundary

Description:
    The latest trial meets the Recall threshold with only a small positive margin — the
    configuration sits right at the guardrail. Small local exploitation around this point is
    the only safe strategy; aggressive moves risk falling below the threshold.

Signals:
    - Recall margin > 0 with small slack (margin near the threshold, 0 to 0.005)
    - recall is feasible

Interpretation:
    The point is feasible but boundary-adjacent. There is little slack to spend, so any
    reduction is risky. The best QPS gains in this state come from careful single-step moves
    and from probing a smaller feasible efS (below U) rather than from structural changes.

Solution:
    - large recall margin: if the margin grows, follow FarAboveSlack: decrease ef by 20-150, M by 10-20
    - small recall margin: single small moves: M by 4-8, efC by 20-50, ef by 5-10

Cost:
    Any reduction can violate the threshold; recovery costs at least one round per failed move.

Boundary:
    Reductions are unsafe when |margin| is within 0.01 — do not cut parameters in this band.
    Do not change M and efC together here, and never move all three parameters at once.

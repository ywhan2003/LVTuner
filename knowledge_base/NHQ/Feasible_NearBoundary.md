Recall Constraint Class:
    Recall-feasible

Pattern ID:
    NHQ.FeasibleNearBoundary

Description:
    The latest trial meets the recall threshold with only a small positive margin. The
    configuration sits at the guardrail: only fine-grained, one-parameter moves are safe.

Signals:
    - Recall margin > 0 with small slack (margin near the threshold, 0 to 0.02)
    - recall is feasible

Interpretation:
    Little recall slack is available to spend. Near-boundary reductions frequently push the
    configuration below the threshold, so rebuilds and multi-parameter changes are
    unjustified.

Solution:
    - large recall margin: if the margin grows, follow FarAboveSlack: decrease ef by 50-500
    - small recall margin: single small moves: ef by 10-50, weight by 5-20

Cost:
    Any reduction can violate the threshold; recovery costs rounds.

Boundary:
    Do not increase M or efConstruction in this band. If weight > 0 and recall is marginal,
    reducing weight toward 0 recovers recall (pure vector search matches pure-L2 GT best).

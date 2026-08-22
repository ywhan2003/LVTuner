Recall Constraint Class:
    Recall-feasible

Pattern ID:
    FilterDiskANN.FeasibleNearBoundary

Description:
    The latest trial meets the recall threshold with only a small positive margin. The
    configuration sits at the guardrail: only fine-grained, one-parameter moves are safe.

Signals:
    - Recall margin > 0 with small slack (margin near the threshold, 0 to 0.02)
    - recall is feasible

Interpretation:
    Little recall slack is available to spend. Near-boundary reductions frequently push the
    configuration below the threshold, so rebuilds are unjustified.

Solution:
    - large recall margin: if the margin grows, follow FarAboveSlack: decrease L by 10-100
    - small recall margin: single small moves: L by 5-10, alpha by +0.05

Cost:
    Any reduction can violate the threshold; recovery costs rounds.

Boundary:
    Do not increase R or FilterLBuild in this band. Keep L <= FilterLBuild and alpha >= 1.0.

Recall Constraint Class:
    Recall-feasible

Pattern ID:
    NHQ.HighBuildCost

Description:
    Recall is above the threshold but build time is excessive: construction parameters are
    over-provisioned. Since the index is cached by (data, attrs, metric, M, efC)
    fingerprint, lowering build cost mostly matters for the build budget itself.

Signals:
    - Recall is feasible (margin >= 0)
    - build_time_s is high
    - M is high
    - efConstruction is high

Interpretation:
    The graph is over-built for the recall target. Cutting M or efConstruction shrinks build
    time and memory without touching query-time recall much.

Solution:
    - large recall margin: decrease M by 10-20, then efC by 100-300
    - small recall margin: decrease M by 4-8, then efC by 50-100

Cost:
    Cutting construction too far lowers graph quality and recall.

Boundary:
    If recall is far below the threshold while build time is high, the direction reverses —
    the graph is weak, not over-built (see GraphQualityInsufficient).

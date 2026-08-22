Recall Constraint Class:
    Recall-feasible

Pattern ID:
    FilterDiskANN.RedundantDensity

Description:
    Recall is above the threshold but the graph is denser than needed: index size is large
    and out-degree is high. The construction over-provisions edges for the recall target.

Signals:
    - Recall is feasible (margin >= 0)
    - index_size_mb is high
    - out_degree_mean is high
    - R is high

Interpretation:
    With a large R, every node keeps many edges (RobustPrune max out-degree), inflating
    memory and per-hop cost. Effective density is the product of R and pruning alpha, so
    density can be reduced via R or raised alpha.

Solution:
    - large recall margin: decrease R by 16-32
    - small recall margin: decrease R by 4-8

Cost:
    Cutting R too far lowers graph connectivity and recall.

Boundary:
    R >= 2 (Vamana minimum). If out-degree is LOW while recall is below the threshold, the
    problem is sparsity, not redundancy — see SparseGraph.

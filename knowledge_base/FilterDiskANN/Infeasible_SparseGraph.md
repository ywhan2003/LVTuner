Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    FilterDiskANN.SparseGraph

Description:
    Recall is below the threshold and the graph is too sparse: index size is small and
    out-degree is low. The graph lacks the edges needed to route queries toward true
    nearest neighbors.

Signals:
    - Recall margin < 0
    - index_size_mb is low
    - out_degree_mean is low
    - R is low

Interpretation:
    A small R (RobustPrune max out-degree) starves the graph of edges. The fix is
    structural: raise R, and if the build beam limits edge quality, raise FilterLBuild.

Solution:
    - large recall gap: increase R by 16-32
    - small recall gap: increase R by 4-8

Cost:
    Larger R grows memory and build time.

Boundary:
    R >= 2. Lowering alpha toward 1.0 keeps more edges (recall up, QPS down) — prefer
    alpha reduction before raising R when alpha > 1.0.

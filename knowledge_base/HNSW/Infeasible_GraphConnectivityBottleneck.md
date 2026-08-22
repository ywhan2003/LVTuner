Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    HNSW.GraphConnectivityBottleneck

Description:
    Recall is below the threshold and graph structure indicators show insufficient connectivity:
    low out-degree or in-degree, and raising ef alone gives limited improvement. The graph lacks
    the edges needed to route queries to high-quality neighbors.

Signals:
    - Recall margin < 0
    - avg_out_degree or avg_in_degree is low
    - ef is moderate or high
    - increasing ef alone gives limited improvement

Interpretation:
    The graph does not contain enough useful edges or long-range connections. Search width
    cannot compensate for a poorly connected graph: the fix is structural (M, then efC).
    When max_recall (the highest Recall any ef achieves at this graph) is below the threshold,
    the graph is structurally incapable — an M rebuild is mandatory, and ef/efC increases
    alone are ineffective.

Solution:
    - large recall gap: increase M by 10-20, then efC by 100-250
    - small recall gap: increase M by 4-8, then efC by 20-50

Cost:
    Index size and build time grow with M; budget exhaustion is a real risk when many
    rebuilds are needed.

Boundary:
    Avoid (M, efC) pairs already explored 3 or more times this run — they are banned.
    Once Recall becomes feasible, switch to QPS optimization (reduce ef).

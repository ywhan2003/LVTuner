Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    HNSW.SearchWidthSaturation

Description:
    Recall remains below the threshold even though ef, visited nodes, and distance computations
    are already high, and recent ef increases brought little or no Recall gain. The bottleneck
    is construction quality or connectivity, not query-time search width.

Signals:
    - Recall margin < 0
    - ef is high
    - visited_nodes_per_query is high
    - dist_comps_count is high
    - recent ef increase gives weak or no Recall gain

Interpretation:
    The search already visits many nodes, but the graph does not provide paths to the true
    nearest neighbors. More search width only adds cost without Recall. The fix must improve
    the graph itself (efC, and M when degree is insufficient).

Solution:
    - large recall gap: increase efC by 100-250; M by 10-20 if needed
    - small recall gap: increase efC by 20-50; M by 4-8 if needed

Cost:
    efC increases cost build time; M increases cost both build time and index size.

Boundary:
    Do not keep raising ef — it only adds query cost without Recall. If out_degree is below
    0.3 x M, treat this as a GraphConnectivityBottleneck instead.

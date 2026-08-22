Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.ExcessiveSearchCostWithSlack

Description:
    Recall is safely above the threshold, but visited nodes and distance computations are high,
    and QPS is lower than expected. The current search width is excessive: query-time search
    is over-exploring the graph while Recall slack goes unused.

Signals:
    - Recall margin > 0 with sufficient slack (recall is feasible)
    - dist_comps_count is high
    - visited_nodes_per_query is high
    - ef is high (above 0.8 x ef_construction)
    - QPS is low

Interpretation:
    The configuration spends more search work than the Recall target requires. The positive
    Recall slack is a resource that can be exchanged for QPS by shrinking the search width.
    ef is the zero-rebuild lever: changing it costs no build time, so it is always the first
    lever for QPS optimization.

Solution:
    - large recall margin: decrease ef (zero rebuild) by 20-150
    - small recall margin: decrease ef by 5-10

Cost:
    Over-reduction drops Recall below the threshold and costs rounds to repair.

Boundary:
    If ef is already high relative to ef_construction and further reduction brings no QPS
    gain, search width is not the bottleneck — see the SearchWidthSaturation card logic.
    Never choose efS below L from the interval table.

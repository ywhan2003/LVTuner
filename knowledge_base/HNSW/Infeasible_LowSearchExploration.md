Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    HNSW.LowSearchExploration

Description:
    Recall is below the threshold while visited nodes and distance computations are low or
    moderate, and ef is not close to its upper bound. The query-time search is not exploring
    enough of the graph to find high-quality nearest neighbors.

Signals:
    - Recall margin < 0 or close to 0 (recall is below the threshold)
    - visited_nodes_per_query is low or moderate
    - dist_comps_count is low or moderate
    - ef is low or moderate

Interpretation:
    The search breadth is insufficient. The query terminates before visiting enough candidates.
    This is the cheapest kind of Recall shortfall: ef is a zero-rebuild lever, so increasing it
    is the safest first repair action.

Solution:
    - large recall gap: increase ef by 20-150
    - small recall gap: increase ef by 5-10

Cost:
    Higher ef lowers QPS proportionally to the added search work.

Boundary:
    If ef is already above 0.8 x ef_construction and Recall is still below the threshold,
    search width is not the bottleneck — see SearchWidthSaturation.

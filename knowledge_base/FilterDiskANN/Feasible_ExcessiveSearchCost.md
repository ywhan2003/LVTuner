Recall Constraint Class:
    Recall-feasible

Pattern ID:
    FilterDiskANN.ExcessiveSearchCost

Description:
    Recall is above the threshold but the query visits many nodes and performs many
    distance computations: the search list L is larger than the graph needs. L is the only
    search knob and changes are zero-rebuild.

Signals:
    - Recall margin > 0 with slack (recall is feasible)
    - visited_nodes_per_query is high
    - dist_comps_per_query is high
    - qps is low

Interpretation:
    The search explores more candidates than necessary for the recall target. Reducing L
    is the cheapest and safest QPS improvement.

Solution:
    - large recall margin: decrease L by 10-100
    - small recall margin: decrease L by 5-10

Cost:
    Too-large L reductions drop recall below the threshold.

Boundary:
    Keep L <= FilterLBuild. If recall stays below the threshold despite a large L, search
    width is not the bottleneck — see GraphQualityInsufficient.

Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.HighIndexCostRedundantDensity

Description:
    Recall is far above the threshold, index size is high, and graph degree is high. The graph
    is denser than the current Recall target requires: construction is over-provisioned and
    costs index memory and query time.

Signals:
    - Recall margin is large and positive (recall margin > 0 with large slack)
    - index_size is high
    - avg_out_degree is high
    - M is high
    - QPS is low

Interpretation:
    The graph is over-built for the current Recall target. Redundant density raises memory and
    per-hop cost without contributing Recall. Either M (index size, requires rebuild) or ef
    (query cost, zero-rebuild) can be reduced, depending on which cost dominates.

Solution:
    - large recall margin: decrease M by 10-20 (rebuild), or decrease ef by 20-150
    - small recall margin: decrease M by 4-8, or decrease ef by 5-10

Cost:
    Every M change costs a full rebuild; cutting M too far drops Recall and requires another
    rebuild to recover.

Boundary:
    Do not reduce M below 12 — empirical recall ceilings collapse at low M. If avg_out_degree
    drops below 0.3 x M, the problem has become connectivity, not redundancy.

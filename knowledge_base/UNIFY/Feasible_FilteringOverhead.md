Recall Constraint Class:
    Recall-feasible

Pattern ID:
    UNIFY.FilteringOverhead

Description:
    Recall meets the threshold but query latency is high while the graph itself is healthy:
    the filter-structure (B slots) adds per-query overhead — every query traverses multiple
    per-slot subgraphs and pays activated-slot search cost (al per slot).

Signals:
    - Recall is feasible (margin >= 0)
    - search latency is high
    - B is large
    - QPS is low

Interpretation:
    With a large B the per-slot data becomes sparser and the query pays for several slot
    traversals. Search cost grows roughly like ef + al x num_activated_slots, so either the
    number of slots (B) or the per-slot depth (al) is oversized for the recall target.

Solution:
    - large recall margin: decrease B by 1-2, then al by 32-64
    - small recall margin: decrease B by 1, then al by 8-16

Cost:
    Lower B weakens filtering accuracy — recall may drop if the query's filter region
    becomes too coarse.

Boundary:
    B = 1 degenerates to plain HNSW (filtering lost entirely). Never raise B while
    inclusiveness is below 70% — the graph is already fragmented.

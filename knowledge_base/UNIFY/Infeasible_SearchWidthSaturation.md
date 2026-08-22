Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    UNIFY.SearchWidthSaturation

Description:
    Recall remains below the threshold even though ef and al are already high and further
    search-effort increases bring little recall gain. The bottleneck is construction quality
    or connectivity, not search width.

Signals:
    - Recall margin < 0
    - ef is high
    - al is high
    - recent ef or al increase gives weak or no recall gain

Interpretation:
    The search already explores enough candidates; the graph itself fails to route queries
    toward true nearest neighbors. More search effort only adds query cost.

Solution:
    - large recall gap: increase efC by 100-250; M by 10-20 if needed
    - small recall gap: increase efC by 20-50; M by 4-8 if needed

Cost:
    efConstruction increases build time; M increases memory.

Boundary:
    Do not keep raising ef/al — it only adds query cost without recall. If out-degree is
    low, treat this as GraphUnderConnectivity instead.

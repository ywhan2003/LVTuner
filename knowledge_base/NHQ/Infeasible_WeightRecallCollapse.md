Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    NHQ.WeightRecallCollapse

Description:
    Recall collapsed below the threshold while weight > 0: the attribute penalty
    (effective distance = vector distance + weight x mismatches) overwhelms vector
    proximity, and the index fails to match the pure-L2 ground truth.

Signals:
    - Recall margin < 0 (recall is below the threshold)
    - weight is high (above 100)
    - the same build reached the threshold at weight = 0

Interpretation:
    Weight trades recall for filtering strength. Beyond ~100 the recall cost accelerates,
    and beyond ~1000 recall collapses entirely. Since ground truth is pure vector distance,
    weight = 0 is the best GT match.

Solution:
    - large recall gap: decrease weight by 20-50 (try weight=0 first)
    - small recall gap: decrease weight by 5-20

Cost:
    Lower weight weakens attribute filtering — more query candidates carry attribute
    mismatches.

Boundary:
    Weight > 1000 causes recall collapse — never propose it. On a given build, tune ef at
    weight = 0 first, then explore weight.

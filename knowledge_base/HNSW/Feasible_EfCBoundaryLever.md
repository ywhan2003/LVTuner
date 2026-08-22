Recall Constraint Class:
    Recall-feasible

Pattern ID:
    HNSW.EfCBoundaryLever

Description:
    efC controls the quality of the edges chosen at build time. A higher efC typically raises
    the baseline Recall of a given (M) graph, which can lower the smallest efS needed to reach
    the threshold — shifting query-time cost into build-time cost. efC is a lever on the
    position of the feasibility boundary U, not just a build-cost knob.

Signals:
    - Recall is feasible (margin >= 0)
    - the current (M, efC) cell has a known U (smallest observed feasible efS) in the interval table
    - QPS improvement from reducing ef further is small or risky
    - build time is acceptable

Interpretation:
    Compare the same-M row across efC cells in the interval table: if a higher-efC cell has a
    LOWER U with competitive QPS, increasing efC is the right move — invest in build quality,
    then re-probe a smaller efS. If higher-efC cells do NOT lower U, efC is saturated for this
    M and must not be changed blindly.

Solution:
    - large recall margin: increase efC by 100-250, then probe a smaller efS
    - small recall margin: increase efC by 20-50, then probe a smaller efS

Cost:
    efC increases add build time only; they do not affect query-time QPS directly.

Boundary:
    Do not raise efC when the same-M row already shows no U improvement at higher efC
    (saturation). Do not use efC to repair a Recall shortfall while Recall still rises with ef
    at the current efC — that gap is a search-width problem, not a boundary problem.

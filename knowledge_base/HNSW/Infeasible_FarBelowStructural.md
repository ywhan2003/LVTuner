Recall Constraint Class:
    Recall-infeasible

Pattern ID:
    HNSW.InfeasibleFarBelowStructural

Description:
    Recall is far below the threshold: the gap is structural, not a small search-width
    shortfall. Aggressive repair is needed, with the choice of lever determined by max_recall
    (the highest Recall any ef achieves at the current graph).

Signals:
    - Recall margin < 0 and far below the threshold (margin below -0.02)
    - recall is below the threshold

Interpretation:
    A large negative margin means small compensation will not suffice. If max_recall >=
    threshold, the graph can reach the target with enough search width — repair with ef first
    (zero-rebuild), and only escalate to M if the same (M, efC) repeatedly fails across ef
    values. If max_recall < threshold, the graph itself is structurally incapable: increasing
    ef or efC is ineffective, and an M rebuild is mandatory.

Solution:
    - large recall gap: increase ef first by 20-150; if still failing, rebuild with M +10-20, efC +100-250
    - small recall gap: increase ef by 5-10

Cost:
    Structural repair costs rebuild time and index size; repeated infeasible configurations
    waste the trial budget.

Boundary:
    Avoid (M, efC) pairs already explored 3 or more times this run. Watch budget exhaustion:
    if the gap stays structural after one M rebuild, reassess the threshold feasibility before
    spending more rounds.

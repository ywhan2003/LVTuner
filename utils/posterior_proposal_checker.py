"""Posterior proposal checker for HNSW tuning proposals.

Given the LLM's raw proposal ``x_raw = (M, efC, efS)`` and the runtime
structural interval table built from current-task memory, check whether the
proposal is consistent with the interval evidence accumulated so far.

The checker is NOT a tuner and does NOT regenerate proposals: it only
reports whether a proposal is ``supported`` / ``too_weak`` /
``too_conservative`` / ``incomparable`` / ``needs_evaluation``, plus an
optional suggested revision.  All judgements are based on interval
evidence only — ``U`` is never treated as an exact ``efS*``, and crossed
construction settings (smaller M with larger efC, or larger M with
smaller efC) are never compared automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Decision labels
DECISION_TOO_WEAK = "too_weak"
DECISION_TOO_CONSERVATIVE = "too_conservative"
DECISION_SUPPORTED = "supported"
DECISION_INCOMPARABLE = "incomparable"
DECISION_NEEDS_EVALUATION = "needs_evaluation"
DECISION_CROSS_UPPER_REFERENCE = "cross_upper_reference"

# Warning labels
WARNING_NEAR_BOUNDARY_LOW_MARGIN = "near_boundary_low_margin"

# Raw proposal field aliases: spec style (M, efC, efS) and pipeline style
# (M, ef_construction, ef).
_M_FIELD_KEYS: Tuple[str, ...] = ("M",)
_EFC_FIELD_KEYS: Tuple[str, ...] = ("efC", "ef_construction", "efConstruction", "efc")
_EFS_FIELD_KEYS: Tuple[str, ...] = ("efS", "ef", "efSearch", "efs")

_DEFAULT_RECALL_MARGIN_EPSILON: float = 0.002


def _pick(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present key from ``d``, or ``None``."""
    for key in keys:
        if key in d:
            return d[key]
    return None


def _as_int(value: Any) -> Optional[int]:
    """Coerce a value to int; return ``None`` when not an integral number."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != int(num):
        return None
    return int(num)


def _as_float(value: Any) -> Optional[float]:
    """Coerce a value to float; return ``None`` when not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _next_grid_value(grid: Sequence[int], above: int) -> Optional[int]:
    """Smallest grid value strictly greater than ``above``, or ``None``."""
    candidates = sorted(g for g in grid if _as_int(g) is not None and g > above)
    return int(candidates[0]) if candidates else None


def parse_interval_table(
    runtime_structural_interval_table: Dict[str, Any],
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], Dict[int, Dict[int, Dict[str, Any]]]]:
    """Parse the interval table JSON into lookup indexes.

    Parameters
    ----------
    runtime_structural_interval_table : dict
        Output of ``build_runtime_structural_interval_table``.

    Returns
    -------
    tuple
        ``(table_index, column_index)`` where ``table_index[M][efC]`` is the
        cell dict and ``column_index[efC][M]`` is the same cell.
    """
    table_index: Dict[int, Dict[int, Dict[str, Any]]] = {}
    column_index: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in (runtime_structural_interval_table or {}).get("interval_table") or []:
        m = _as_int(row.get("M"))
        if m is None:
            continue
        cells: Dict[int, Dict[str, Any]] = {}
        for cell in row.get("efC_cells") or []:
            efc = _as_int(cell.get("efC"))
            if efc is None:
                continue
            cells[efc] = cell
            column_index.setdefault(efc, {})[m] = cell
        table_index[m] = cells
    return table_index, column_index


def _cell_interval(cell: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(L, U)`` of a cell, tolerating missing interval dicts."""
    interval = cell.get("efS_interval") or {}
    return (
        _as_int(interval.get("L")),
        _as_int(interval.get("U")),
    )


def _pick_cover_cell(
    candidates: List[Tuple[Any, Dict[str, Any]]],
    tiebreak_key: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Select a cover reference cell: max qps_at_U, then min U, then min key."""
    def sort_key(item: Tuple[Any, Dict[str, Any]]) -> Tuple[Any, ...]:
        _, cell = item
        qps = _as_float(cell.get("qps_at_U"))
        _, u = _cell_interval(cell)
        # Highest QPS first → negate; then smallest U; then smallest key.
        return (
            -(qps if qps is not None else -1.0),
            u if u is not None else 10 ** 18,
            item[0],
        )

    return min(candidates, key=sort_key)


def _pick_blocker_cell(
    candidates: List[Tuple[Any, Dict[str, Any]]],
    raw_key: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Select a blocker reference cell: closest key to raw, then max L, then min key."""
    def sort_key(item: Tuple[Any, Dict[str, Any]]) -> Tuple[Any, ...]:
        _, cell = item
        l, _ = _cell_interval(cell)
        return (
            abs(item[0] - raw_key),
            -(l if l is not None else -1),
            item[0],
        )

    return min(candidates, key=sort_key)


def check_raw_proposal(
    runtime_structural_interval_table: Dict[str, Any],
    raw_proposal: Dict[str, Any],
    current_state: Optional[Dict[str, Any]] = None,
    efS_grid: Optional[Sequence[int]] = None,
    recall_margin_epsilon: float = _DEFAULT_RECALL_MARGIN_EPSILON,
    m_keys: Sequence[str] = _M_FIELD_KEYS,
    efc_keys: Sequence[str] = _EFC_FIELD_KEYS,
    efs_keys: Sequence[str] = _EFS_FIELD_KEYS,
) -> Dict[str, Any]:
    """Check a raw proposal against the runtime structural interval table.

    Parameters
    ----------
    runtime_structural_interval_table : dict
        The ``runtime_structural_interval_table`` JSON from current-task
        memory.
    raw_proposal : dict
        ``{"M": int, "efC": int, "efS": int}`` — pipeline-style keys
        (``ef_construction`` / ``ef``) are accepted as aliases.
    current_state : dict or None
        Optional ``{"current_configuration", "current_recall",
        "current_qps", "current_feasible"}``.  Used to decide whether a
        "too conservative" revision should be suggested.
    efS_grid : sequence of int or None
        Optional efS grid; without it the checker cannot infer a next grid
        value and will suggest ``null`` where a grid step would be needed.
    recall_margin_epsilon : float
        Margin below which ``recall_at_U - R_tau`` counts as "too small".

    Returns
    -------
    dict
        ``ProposalCheckResult`` — see the module docstring / spec.
    """
    warnings: List[str] = []
    explanation_parts: List[str] = []
    suggested: Optional[Dict[str, int]] = None

    table_index, column_index = parse_interval_table(runtime_structural_interval_table)

    # Objective recall threshold
    objective = (runtime_structural_interval_table or {}).get("objective") or {}
    R_tau = _as_float(objective.get("recall_threshold"))
    if R_tau is None:
        R_tau = 0.0

    m_raw = _as_int(_pick(raw_proposal, m_keys))
    efc_raw = _as_int(_pick(raw_proposal, efc_keys))
    efs_raw = _as_int(_pick(raw_proposal, efs_keys))
    normalized_proposal = {"M": m_raw, "efC": efc_raw, "efS": efs_raw}

    if m_raw is None or efc_raw is None or efs_raw is None:
        return {
            "raw_proposal": normalized_proposal,
            "decision": DECISION_NEEDS_EVALUATION,
            "suggested_proposal": None,
            "exact_cell_check": {
                "exists": False, "interval": None, "status": None,
                "evidence": "Malformed raw proposal: missing M / efC / efS.",
            },
            "same_M_row_check": {
                "row_exists": False, "cover_found": False, "cover_cell": None,
                "blocker_found": False, "blocker_cell": None, "evidence": "",
            },
            "same_efC_column_check": {
                "column_exists": False, "cover_found": False, "cover_cell": None,
                "blocker_found": False, "blocker_cell": None, "evidence": "",
            },
            "incomparable_check": {
                "crossed_cells_exist": False, "num_crossed_cells": 0, "evidence": "",
            },
            "cross_reference_check": {
                "weaker_cell_found": False, "weaker_cell": None, "evidence": "",
            },
            "warnings": warnings,
            "explanation": (
                "The raw proposal is missing M / efC / efS, so it cannot be "
                "checked against interval evidence. Needs evaluation."
            ),
        }

    grid: Optional[Sequence[int]] = (
        [g for g in efS_grid if _as_int(g) is not None] if efS_grid is not None else None
    )

    def _disclaimer() -> str:
        return (
            " This suggestion is based on interval evidence only; it is not "
            "guaranteed optimal, and U is not an exact efS*."
        )

    # ── Step 1: exact cell check ──────────────────────────────────────
    row_cells = table_index.get(m_raw, {})
    exact_cell = row_cells.get(efc_raw)
    exact_decision: Optional[str] = None
    exact_suggested: Optional[Dict[str, int]] = None
    exact_evidence = ""
    if exact_cell is not None:
        L, U = _cell_interval(exact_cell)
        exact_evidence = (
            f"Exact cell (M={m_raw}, efC={efc_raw}) exists with interval "
            f"({L}, {U}], status={exact_cell.get('status')}."
        )
        if L is not None and efs_raw <= L:
            exact_decision = DECISION_TOO_WEAK
            if U is not None:
                exact_suggested = {"M": m_raw, "efC": efc_raw, "efS": U}
                exact_evidence += (
                    f" efS={efs_raw} does not exceed the largest observed "
                    f"infeasible efS (L={L}). Suggest efS={U} (smallest "
                    "observed feasible efS for this cell)."
                )
            elif grid:
                next_val = _next_grid_value(grid, L)
                if next_val is not None:
                    exact_suggested = {"M": m_raw, "efC": efc_raw, "efS": next_val}
                    exact_evidence += (
                        f" efS={efs_raw} does not exceed L={L}; no feasible U "
                        f"known, suggest the next grid value efS={next_val}."
                    )
                else:
                    exact_evidence += (
                        f" efS={efs_raw} does not exceed L={L}; no feasible U "
                        "known and no grid value above L exists — efS must be "
                        "increased."
                    )
            else:
                exact_evidence += (
                    f" efS={efs_raw} does not exceed L={L}; no feasible U "
                    "known — efS must be increased (or a stronger "
                    "construction setting chosen)."
                )
        elif U is not None and efs_raw > U:
            exact_decision = DECISION_TOO_CONSERVATIVE
            exact_evidence += (
                f" efS={efs_raw} is larger than the smallest observed "
                f"feasible efS (U={U}) for this cell — the extra search "
                "effort brings unnecessary query-time cost."
            )
            current_feasible = (current_state or {}).get("current_feasible")
            if current_state is None or current_feasible is True:
                exact_suggested = {"M": m_raw, "efC": efc_raw, "efS": U}
                if current_state is None:
                    exact_evidence += (
                        f" Suggest efS={U} (no current state provided)."
                    )
                else:
                    exact_evidence += (
                        f" Suggest efS={U}: the current state is feasible, "
                        "so the reduction is safe."
                    )
            else:
                exact_evidence += (
                    " Current-state feasibility is false/unknown — do NOT "
                    "force a reduction; only the evidence is reported."
                )
        elif L is not None and U is not None and L < efs_raw <= U:
            exact_decision = DECISION_SUPPORTED
            exact_evidence += (
                f" efS={efs_raw} lies inside the current interval "
                f"(L={L}, U={U}] for this cell."
            )
            if efs_raw == U:
                recall_at_u = _as_float(exact_cell.get("recall_at_U"))
                if recall_at_u is not None and recall_at_u - R_tau < recall_margin_epsilon:
                    warnings.append(WARNING_NEAR_BOUNDARY_LOW_MARGIN)
                    exact_evidence += (
                        f" WARNING: recall_at_U={recall_at_u} leaves a margin "
                        f"< {recall_margin_epsilon} above R_tau={R_tau} — if "
                        "recall safety matters, consider the next larger efS."
                    )
    else:
        exact_evidence = f"No exact cell for (M={m_raw}, efC={efc_raw}) in the table."

    # ── Step 2: same-M row check ──────────────────────────────────────
    row_cover_candidates: List[Tuple[int, Dict[str, Any]]] = []
    row_blocker_candidates: List[Tuple[int, Dict[str, Any]]] = []
    row_evidence_parts: List[str] = []
    efc_sel: Optional[int] = None
    efc_blk: Optional[int] = None
    for efc_hist, cell in row_cells.items():
        if efc_hist == efc_raw:
            continue  # exact cell handled by Step 1
        L, U = _cell_interval(cell)
        if efc_hist <= efc_raw and U is not None and U <= efs_raw:
            row_cover_candidates.append((efc_hist, cell))
        if efc_hist >= efc_raw and L is not None and L >= efs_raw:
            row_blocker_candidates.append((efc_hist, cell))

    row_cover_cell: Optional[Dict[str, Any]] = None
    if row_cover_candidates:
        efc_sel, row_cover_cell = _pick_cover_cell(row_cover_candidates, tiebreak_key="efC")
        _, u_sel = _cell_interval(row_cover_cell)
        row_evidence_parts.append(
            f"Row cover: cell (M={m_raw}, efC={efc_sel}) already reached the "
            f"threshold with U={u_sel} <= efS={efs_raw} — the proposal is "
            "likely too conservative."
        )
    row_blocker_cell: Optional[Dict[str, Any]] = None
    if row_blocker_candidates:
        efc_blk, row_blocker_cell = _pick_blocker_cell(row_blocker_candidates, efc_raw)
        l_blk, _ = _cell_interval(row_blocker_cell)
        row_evidence_parts.append(
            f"Row blocker: cell (M={m_raw}, efC={efc_blk}) is still infeasible "
            f"at L={l_blk} >= efS={efs_raw} — the proposal is likely too weak."
        )
    row_evidence = " ".join(row_evidence_parts)

    # ── Step 3: same-efC column check ─────────────────────────────────
    column_cells = column_index.get(efc_raw, {})
    column_cover_candidates: List[Tuple[int, Dict[str, Any]]] = []
    column_blocker_candidates: List[Tuple[int, Dict[str, Any]]] = []
    column_evidence_parts: List[str] = []
    m_sel: Optional[int] = None
    m_blk: Optional[int] = None
    for m_hist, cell in column_cells.items():
        if m_hist == m_raw:
            continue  # exact cell handled by Step 1
        L, U = _cell_interval(cell)
        if m_hist <= m_raw and U is not None and U <= efs_raw:
            column_cover_candidates.append((m_hist, cell))
        if m_hist >= m_raw and L is not None and L >= efs_raw:
            column_blocker_candidates.append((m_hist, cell))

    column_cover_cell: Optional[Dict[str, Any]] = None
    if column_cover_candidates:
        m_sel, column_cover_cell = _pick_cover_cell(column_cover_candidates, tiebreak_key="M")
        _, u_sel = _cell_interval(column_cover_cell)
        column_evidence_parts.append(
            f"Column cover: cell (M={m_sel}, efC={efc_raw}) already reached "
            f"the threshold with U={u_sel} <= efS={efs_raw} — the proposal is "
            "likely too conservative."
        )
    column_blocker_cell: Optional[Dict[str, Any]] = None
    if column_blocker_candidates:
        m_blk, column_blocker_cell = _pick_blocker_cell(column_blocker_candidates, m_raw)
        l_blk, _ = _cell_interval(column_blocker_cell)
        column_evidence_parts.append(
            f"Column blocker: cell (M={m_blk}, efC={efc_raw}) is still "
            f"infeasible at L={l_blk} >= efS={efs_raw} — the proposal is "
            "likely too weak."
        )
    column_evidence = " ".join(column_evidence_parts)

    # ── Step 4: crossed construction settings ─────────────────────────
    crossed_count = 0
    for m_hist, cells in table_index.items():
        for efc_hist in cells:
            if (m_hist < m_raw and efc_hist > efc_raw) or (
                m_hist > m_raw and efc_hist < efc_raw
            ):
                crossed_count += 1
    crossed_evidence = (
        f"{crossed_count} crossed construction cell(s) exist (M and efC move "
        "in opposite directions relative to the proposal). They cannot be "
        "used for cover / blocker judgements."
        if crossed_count
        else "No crossed construction cells."
    )

    # ── Step 5: component-wise weaker construction upper reference ────
    cross_upper_cell: Optional[Dict[str, Any]] = None
    cross_upper_m: Optional[int] = None
    cross_upper_efc: Optional[int] = None
    cross_upper_u: Optional[int] = None
    for m_hist, cells in table_index.items():
        for efc_hist, cell in cells.items():
            if (
                m_hist <= m_raw
                and efc_hist <= efc_raw
                and (m_hist, efc_hist) != (m_raw, efc_raw)
            ):
                _, u_hist = _cell_interval(cell)
                if u_hist is not None and u_hist <= efs_raw:
                    cross_upper_cell = cell
                    cross_upper_m, cross_upper_efc, cross_upper_u = m_hist, efc_hist, u_hist
                    break
            if cross_upper_cell is not None:
                break
        if cross_upper_cell is not None:
            break
    cross_upper_evidence = (
        f"Weaker construction cell (M={cross_upper_m}, efC={cross_upper_efc}) "
        f"already reached the threshold at U={cross_upper_u} <= efS={efs_raw} — "
        "a stronger construction should need no larger search effort."
        if cross_upper_cell is not None
        else "No component-wise weaker construction cell covers the proposal."
    )

    # ── Final decision (priority P1..P7) ──────────────────────────────
    decision = DECISION_NEEDS_EVALUATION
    if exact_decision == DECISION_TOO_WEAK:
        decision = DECISION_TOO_WEAK
        suggested = exact_suggested
    elif exact_decision == DECISION_TOO_CONSERVATIVE:
        decision = DECISION_TOO_CONSERVATIVE
        suggested = exact_suggested
    elif row_blocker_cell is not None or column_blocker_cell is not None:
        decision = DECISION_TOO_WEAK
        # Row blocker evidence first (same-M row precedes the column check).
        if row_blocker_cell is not None:
            l_blk, _ = _cell_interval(row_blocker_cell)
            if grid:
                next_val = _next_grid_value(grid, l_blk)
                if next_val is not None:
                    suggested = {"M": m_raw, "efC": efc_raw, "efS": next_val}
                    explanation_parts.append(
                        f"Blocker cell (M={m_raw}, efC={efc_blk}) has "
                        f"L={l_blk} >= efS={efs_raw}; suggest the next grid "
                        f"value efS={next_val}."
                    )
                else:
                    explanation_parts.append(
                        f"Blocker cell (M={m_raw}, efC={efc_blk}) has "
                        f"L={l_blk} >= efS={efs_raw}; no grid value above L "
                        "exists — increase efS or choose a stronger "
                        "construction setting."
                    )
            else:
                explanation_parts.append(
                    f"Blocker cell (M={m_raw}, efC={efc_blk}) has "
                    f"L={l_blk} >= efS={efs_raw}; increase efS or choose a "
                    "stronger construction setting (no efS grid provided)."
                )
        else:
            l_blk, _ = _cell_interval(column_blocker_cell)  # type: ignore[arg-type]
            if grid:
                next_val = _next_grid_value(grid, l_blk)
                if next_val is not None:
                    suggested = {"M": m_raw, "efC": efc_raw, "efS": next_val}
                    explanation_parts.append(
                        f"Blocker cell (M={m_blk}, efC={efc_raw}) has "
                        f"L={l_blk} >= efS={efs_raw}; suggest the next grid "
                        f"value efS={next_val}."
                    )
                else:
                    explanation_parts.append(
                        f"Blocker cell (M={m_blk}, efC={efc_raw}) has "
                        f"L={l_blk} >= efS={efs_raw}; no grid value above L "
                        "exists — increase efS or choose a stronger "
                        "construction setting."
                    )
            else:
                explanation_parts.append(
                    f"Blocker cell (M={m_blk}, efC={efc_raw}) has "
                    f"L={l_blk} >= efS={efs_raw}; increase efS or choose a "
                    "stronger construction setting (no efS grid provided)."
                )
    elif row_cover_cell is not None or column_cover_cell is not None:
        decision = DECISION_TOO_CONSERVATIVE
        if row_cover_cell is not None:
            _, u_sel = _cell_interval(row_cover_cell)
            suggested = {"M": m_raw, "efC": efc_sel, "efS": u_sel}
            explanation_parts.append(
                f"Cover cell (M={m_raw}, efC={efc_sel}) reached the threshold "
                f"at U={u_sel} <= efS={efs_raw}."
            )
        else:
            _, u_sel = _cell_interval(column_cover_cell)  # type: ignore[arg-type]
            suggested = {"M": m_sel, "efC": efc_raw, "efS": u_sel}
            explanation_parts.append(
                f"Cover cell (M={m_sel}, efC={efc_raw}) reached the threshold "
                f"at U={u_sel} <= efS={efs_raw}."
            )
    elif exact_cell is None and cross_upper_cell is not None:
        decision = DECISION_CROSS_UPPER_REFERENCE
        suggested = {"M": m_raw, "efC": efc_raw, "efS": cross_upper_u}
        explanation_parts.append(
            f"Weaker construction cell (M={cross_upper_m}, efC={cross_upper_efc}) "
            f"already reached the threshold at U={cross_upper_u} <= efS={efs_raw}; "
            "a stronger construction should need no larger search effort."
        )
    elif exact_decision == DECISION_SUPPORTED:
        decision = DECISION_SUPPORTED
    elif crossed_count:
        decision = DECISION_INCOMPARABLE
        explanation_parts.append(
            "Only crossed construction settings exist — M and efC move in "
            "opposite directions, so no safe comparison is possible. A real "
            "evaluation is needed, or the LLM should give a clearer direction."
        )
    else:
        decision = DECISION_NEEDS_EVALUATION
        explanation_parts.append(
            "No exact / row / column / crossed interval evidence applies to "
            "this proposal. A real evaluation is needed."
        )

    if suggested is not None:
        explanation_parts.append(
            f"Suggested revision (interval evidence only, not guaranteed "
            f"optimal): {suggested}."
        )
    elif decision in (DECISION_TOO_WEAK, DECISION_TOO_CONSERVATIVE):
        explanation_parts.append(
            "No automatic revision suggested — revise manually using the "
            "evidence above."
        )

    # Assemble final explanation
    explanation = " ".join(
        [exact_evidence, row_evidence, column_evidence, crossed_evidence]
        + explanation_parts
    )

    def _cell_ref(cell: Optional[Dict[str, Any]], m: Optional[int], efc: Optional[int]) -> Optional[Dict[str, Any]]:
        if cell is None:
            return None
        ref = dict(cell)
        ref["M"] = m
        ref["efC"] = efc
        return ref

    return {
        "raw_proposal": normalized_proposal,
        "decision": decision,
        "suggested_proposal": suggested,
        "exact_cell_check": {
            "exists": exact_cell is not None,
            "interval": (
                {"L": _cell_interval(exact_cell)[0], "U": _cell_interval(exact_cell)[1]}
                if exact_cell is not None else None
            ),
            "status": exact_cell.get("status") if exact_cell is not None else None,
            "evidence": exact_evidence,
        },
        "same_M_row_check": {
            "row_exists": bool(row_cells),
            "cover_found": row_cover_cell is not None,
            "cover_cell": _cell_ref(row_cover_cell, m_raw, efc_sel if row_cover_cell is not None else None),
            "blocker_found": row_blocker_cell is not None,
            "blocker_cell": _cell_ref(row_blocker_cell, m_raw, efc_blk if row_blocker_cell is not None else None),
            "evidence": row_evidence,
        },
        "same_efC_column_check": {
            "column_exists": bool(column_cells),
            "cover_found": column_cover_cell is not None,
            "cover_cell": _cell_ref(column_cover_cell, m_sel if column_cover_cell is not None else None, efc_raw),
            "blocker_found": column_blocker_cell is not None,
            "blocker_cell": _cell_ref(column_blocker_cell, m_blk if column_blocker_cell is not None else None, efc_raw),
            "evidence": column_evidence,
        },
        "incomparable_check": {
            "crossed_cells_exist": crossed_count > 0,
            "num_crossed_cells": crossed_count,
            "evidence": crossed_evidence,
        },
        "cross_reference_check": {
            "weaker_cell_found": cross_upper_cell is not None,
            "weaker_cell": _cell_ref(cross_upper_cell, cross_upper_m, cross_upper_efc),
            "evidence": cross_upper_evidence,
        },
        "warnings": warnings,
        "explanation": explanation,
    }


def format_check_result_for_llm(result: Dict[str, Any]) -> str:
    """Render a ``ProposalCheckResult`` into LLM-readable feedback.

    Only formats the check result — it does not regenerate proposals or
    alter tuning state.
    """
    proposal = result.get("raw_proposal") or {}
    decision = result.get("decision", DECISION_NEEDS_EVALUATION)
    suggested = result.get("suggested_proposal")
    lines: List[str] = []
    lines.append(
        f"Posterior check on raw proposal (M={proposal.get('M')}, "
        f"efC={proposal.get('efC')}, efS={proposal.get('efS')}): {decision}"
    )

    exact = result.get("exact_cell_check") or {}
    if exact.get("evidence"):
        lines.append(f"- Exact cell: {exact['evidence']}")
    row = result.get("same_M_row_check") or {}
    if row.get("evidence"):
        lines.append(f"- Same-M row: {row['evidence']}")
    column = result.get("same_efC_column_check") or {}
    if column.get("evidence"):
        lines.append(f"- Same-efC column: {column['evidence']}")
    incomparable = result.get("incomparable_check") or {}
    if incomparable.get("crossed_cells_exist"):
        lines.append(f"- Crossed settings: {incomparable['evidence']}")

    if decision == DECISION_TOO_WEAK:
        lines.append(
            "- Verdict: your proposal NEEDS revision — interval evidence says "
            "it cannot be better than what was already run: this (M, efC) "
            "range has been observed infeasible at this efS (or a weaker "
            "setting was already infeasible here)."
        )
    elif decision == DECISION_TOO_CONSERVATIVE:
        lines.append(
            "- Verdict: your proposal NEEDS revision — interval evidence says "
            "it cannot be better than what was already run: a cheaper "
            "configuration already reached the recall threshold at no larger "
            "cost, so your proposal cannot beat it on QPS."
        )
    elif decision == DECISION_CROSS_UPPER_REFERENCE:
        cross_ref = result.get("cross_reference_check") or {}
        if cross_ref.get("evidence"):
            lines.append(f"- Weaker-construction reference: {cross_ref['evidence']}")
        lines.append(
            "- Verdict: your proposal NEEDS revision — a component-wise weaker "
            "construction setting already reached the threshold with no larger "
            "search effort, so this stronger construction should use efS around "
            "or below that U."
        )
    elif decision == DECISION_SUPPORTED:
        lines.append(
            "- Verdict: no revision needed based on interval evidence — the "
            "proposal lies inside the observed feasible interval for this "
            "(M, efC)."
        )
    elif decision == DECISION_INCOMPARABLE:
        lines.append(
            "- Verdict: interval evidence cannot judge this proposal (crossed "
            "construction settings only); keep the direction but expect real "
            "evaluation to decide."
        )
    else:
        lines.append(
            "- Verdict: no interval evidence applies; real evaluation is "
            "needed to judge this proposal."
        )

    if suggested is not None:
        lines.append(
            f"- Suggested revision: (M={suggested.get('M')}, "
            f"efC={suggested.get('efC')}, efS={suggested.get('efS')}) — based "
            "on interval evidence only, not guaranteed optimal."
        )
    for warning in result.get("warnings") or []:
        if warning == WARNING_NEAR_BOUNDARY_LOW_MARGIN:
            lines.append(
                "- Warning: the smallest observed feasible efS leaves a very "
                "small recall margin above the threshold; if recall safety "
                "matters, prefer a slightly larger efS."
            )
    return "\n".join(lines)


def hard_reject(
    runtime_structural_interval_table: Dict[str, Any],
    proposal: Dict[str, Any],
    m_keys: Sequence[str] = _M_FIELD_KEYS,
    efc_keys: Sequence[str] = _EFC_FIELD_KEYS,
    efs_keys: Sequence[str] = _EFS_FIELD_KEYS,
) -> Tuple[bool, str]:
    """Hard repository check following runtime.tex dominance rules.

    Returns ``(rejected, feedback)``. A proposal ``(c, s)`` is REJECTED when
    the exact cell ``(M, efC)`` exists and:

    - ``s <= L`` — the search effort does not exceed the strongest retained
      infeasible evidence, so the proposal is likely still infeasible;
    - ``s > U`` — the effort exceeds the highest-QPS retained feasible
      evidence, adding query cost only.

    Single-sided cells check only the existing side; unresolved cells and an
    empty table always pass.  The component-wise weaker-construction rule is
    guidance only (runtime.tex: "prioritizes search settings around or below
    U(c')") and is reported advisorially by :func:`check_raw_proposal`
    (``cross_upper_reference``) — it is NOT a hard rejection.
    """
    table_index, _ = parse_interval_table(runtime_structural_interval_table)
    m_raw = _as_int(_pick(proposal, m_keys))
    efc_raw = _as_int(_pick(proposal, efc_keys))
    efs_raw = _as_int(_pick(proposal, efs_keys))
    if m_raw is None or efc_raw is None or efs_raw is None:
        return False, ""

    row_cells = table_index.get(m_raw, {})
    exact_cell = row_cells.get(efc_raw)
    if exact_cell is not None:
        L, U = _cell_interval(exact_cell)
        if L is not None and efs_raw <= L:
            return True, (
                f"Repository check REJECTED the proposal (M={m_raw}, efC={efc_raw}, "
                f"efS={efs_raw}): efS does not exceed L={L}, the search setting of the "
                "strongest retained infeasible evidence under this construction setting — "
                "the proposal is likely still infeasible. Propose efS > L, or a stronger "
                "construction setting."
            )
        if U is not None and efs_raw > U:
            return True, (
                f"Repository check REJECTED the proposal (M={m_raw}, efC={efc_raw}, "
                f"efS={efs_raw}): efS exceeds U={U}, the search setting of the highest-QPS "
                "retained feasible evidence under this construction setting — the extra "
                "search effort only adds query-time cost. Propose efS <= U."
            )
    return False, ""

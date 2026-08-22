"""Current-task memory for LLM-based vector search tuning.

Records configuration points and state-action-state transitions from the
*current* tuning task only.  Historical records are used solely for
boundary-width calibration — never turned into action memory.

Provides dual-view retrieval (state-conditioned + config-neighborhood)
for the LLM proposal-generation prompt.

Classes
-------
PointLevelMemory
    Tracks executed configs with quality labels and landscape anchors.
TransitionLevelMemory
    Records real transitions, compresses them into action reflections,
    and supports dual-view retrieval.
CurrentTaskMemory
    Top-level facade that composes the two stores and exposes the public
    API consumed by the tuning pipeline.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Recall zone labels (4-class system with separate lower / upper boundaries)
RECALL_FAR_BELOW = "far-below"
RECALL_NEAR_BELOW = "near-below"
RECALL_NEAR_ABOVE = "near-above"
RECALL_FAR_ABOVE = "far-above"
RECALL_ZONES: Tuple[str, ...] = (
    RECALL_FAR_BELOW,
    RECALL_NEAR_BELOW,
    RECALL_NEAR_ABOVE,
    RECALL_FAR_ABOVE,
)

# Point quality labels
QUALITY_HIGH = "high_quality_feasible"
QUALITY_NEAR = "near_boundary_promising"
QUALITY_LOW = "low_quality_or_dominated"
QUALITY_FAILED = "failed"
QUALITY_LABELS: Tuple[str, ...] = (QUALITY_HIGH, QUALITY_NEAR, QUALITY_LOW, QUALITY_FAILED)

# Outcome labels for transitions
OUTCOME_STRONG = "strong_success"
OUTCOME_WEAK = "weak_success"
OUTCOME_FAIL = "failure"

# Metric pattern labels extracted from state diagnosis
METRIC_PATTERNS: Tuple[str, ...] = (
    "low_search_exploration",
    "search_width_saturation",
    "graph_connectivity_bottleneck",
    "excessive_search_cost_with_slack",
    "near_boundary_unsafe_reduction",
    "redundant_graph_density",
    "unknown",
)

# Evidence roles returned by retrieval
EVIDENCE_SUPPORTING = "supporting"
EVIDENCE_REPAIR = "repair"
EVIDENCE_RISK = "risk"
EVIDENCE_BOUNDARY = "boundary"

# Numeric epsilon for floating-point boundary checks
_CLASSIFY_EPSILON: float = 1e-9

# Default boundary widths (canonical prior before calibration)
_DEFAULT_DELTA_LOW: float = 0.01
_DEFAULT_DELTA_HIGH: float = 0.01

# Minimum number of historical points needed before calibration overrides defaults
_MIN_CALIBRATION_POINTS: int = 10

# The knobs we care about for config-neighbourhood distance
_HNSW_KNOBS: Tuple[str, ...] = ("M", "ef_construction", "ef")
_UNIFY_KNOBS: Tuple[str, ...] = ("M", "B", "efConstruction")
_NHQ_KNOBS: Tuple[str, ...] = ("M", "efConstruction")
_FD_KNOBS: Tuple[str, ...] = ("R", "FilterLBuild", "alpha")

# Default relative weights when no metric-pattern signal exists
_HNSW_DEFAULT_KNOB_WEIGHTS: Dict[str, float] = {"M": 1.0, "ef_construction": 1.0, "ef": 1.0}
_UNIFY_DEFAULT_KNOB_WEIGHTS: Dict[str, float] = {"M": 1.0, "B": 1.0, "efConstruction": 1.0}

# Metric-pattern → knob weight overrides (HNSW)
_HNSW_PATTERN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "low_search_exploration": {"M": 0.5, "ef_construction": 0.5, "ef": 2.0},
    "search_width_saturation": {"M": 0.5, "ef_construction": 0.5, "ef": 2.0},
    "graph_connectivity_bottleneck": {"M": 3.0, "ef_construction": 2.0, "ef": 0.5},
    "excessive_search_cost_with_slack": {"M": 0.5, "ef_construction": 0.5, "ef": 3.0},
    "near_boundary_unsafe_reduction": {"M": 1.0, "ef_construction": 1.0, "ef": 1.5},
    "redundant_graph_density": {"M": 3.0, "ef_construction": 1.5, "ef": 0.5},
}

# Metric-pattern → knob weight overrides (UNIFY)
_UNIFY_PATTERN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "low_search_exploration": {"M": 0.5, "B": 0.5, "efConstruction": 2.0},
    "search_width_saturation": {"M": 0.5, "B": 0.5, "efConstruction": 2.0},
    "graph_connectivity_bottleneck": {"M": 3.0, "B": 2.0, "efConstruction": 0.5},
    "excessive_search_cost_with_slack": {"M": 0.5, "B": 0.5, "efConstruction": 3.0},
    "near_boundary_unsafe_reduction": {"M": 1.0, "B": 1.0, "efConstruction": 1.5},
    "redundant_graph_density": {"M": 3.0, "B": 1.5, "efConstruction": 0.5},
}

# Legacy aliases for backward compatibility
_DEFAULT_KNOB_WEIGHTS = _HNSW_DEFAULT_KNOB_WEIGHTS
_PATTERN_WEIGHTS = _HNSW_PATTERN_WEIGHTS

# Quality classification thresholds
_QPS_COMPETITIVENESS_RATIO: float = 0.85  # qps must be >= 85% of best feasible qps
_QPS_SEVERE_DEGRADATION_RATIO: float = 0.5  # qps <= 50% of best feasible qps → dominated / failed

# Outcome classification thresholds
_STRONG_SUCCESS_QPS_RATIO: float = 1.05  # 5 % qps gain over best feasible
_WEAK_SUCCESS_QPS_GAIN: float = 0.02  # 2 % qps gain


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def _utc_timestamp() -> int:
    return int(time.time())


def _default_boundaries(threshold: float) -> Tuple[float, float]:
    """Return sensible default (delta_low, delta_high) for a given threshold.

    Mirrors the adaptive logic in the agent's ``_near_boundary`` and
    ``_classify_recall_status``.
    """
    if threshold >= 0.99:
        dl = 0.001
    else:
        dl = 0.01
    dh = min(0.03, (1.0 - threshold) * 0.5)
    return (dl, dh)


# ---------------------------------------------------------------------------
# Runtime Structural Interval Table
# ---------------------------------------------------------------------------

# Semantic field → candidate key aliases.  Observations may be flat dicts
# (``{"M": ..., "efC": ..., "efS": ...}``) or point records with the config
# nested under ``"config"``.
_M_FIELD_ALIASES: Tuple[str, ...] = ("M",)
_EFC_FIELD_ALIASES: Tuple[str, ...] = ("efC", "efc", "efConstruction", "ef_construction")
_EFS_FIELD_ALIASES: Tuple[str, ...] = ("efS", "efs", "efSearch", "ef")
_RECALL_FIELD_ALIASES: Tuple[str, ...] = ("recall", "Recall")
_QPS_FIELD_ALIASES: Tuple[str, ...] = ("qps", "QPS", "throughput")


def _pick_obs_field(observation: Dict[str, Any], aliases: Sequence[str]) -> Any:
    """Return the first present alias from an observation, checking both the
    flat dict and the nested ``observation["config"]`` dict."""
    for alias in aliases:
        if alias in observation:
            return observation[alias]
    config = observation.get("config")
    if isinstance(config, dict):
        for alias in aliases:
            if alias in config:
                return config[alias]
    return None


def _as_number(value: Any) -> Optional[float]:
    """Coerce a value to float; return ``None`` when it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """Coerce a value to int; return ``None`` when it is not an integral number."""
    num = _as_number(value)
    if num is None or num != int(num):
        return None
    return int(num)


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text at a paragraph boundary when it exceeds ``max_chars``."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars // 2:
        truncated = truncated[:last_para]
    return truncated + "\n\n... (memory context truncated)"


def build_runtime_structural_interval_table(
    observations: List[Dict[str, Any]],
    R_tau: float,
    m_keys: Sequence[str] | None = None,
    efc_keys: Sequence[str] | None = None,
    efs_keys: Sequence[str] | None = None,
    controlled_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    """Summarize evaluated configurations into an ``M -> efC -> efS``
    feasible-interval table.

    For every construction setting ``c = (M, efC)`` the cell stores the
    evidence interval ``I(M, efC) = (L, U]`` following runtime.tex
    Algorithm 1: ``L`` is the search setting of the infeasible observation
    with the HIGHEST recall (``R_min`` evidence), and ``U`` the search
    setting of the feasible observation with the HIGHEST QPS (``Q_min``
    evidence).  Under monotone recall/QPS in efS this coincides with the
    largest infeasible / smallest feasible efS, but remains correct on
    non-monotone data.  The table records interval evidence only — it
    never claims an exact ``efS*``.

    Parameters
    ----------
    observations : list of dict
        One dict per evaluated configuration.  Each must provide ``M``,
        ``efC`` (aliases: ``efc``, ``efConstruction``, ``ef_construction``),
        ``efS`` (aliases: ``efs``, ``efSearch``, ``ef``), ``recall``
        (alias: ``Recall``) and ``qps`` (aliases: ``QPS``, ``throughput``),
        either flat or nested under a ``"config"`` key.  Observations
        missing any of these are skipped.
    R_tau : float
        Recall threshold.  ``feasible = recall >= R_tau``.

    Returns
    -------
    dict
        The ``runtime_structural_interval_table`` JSON with rows sorted by
        ``M`` ascending, cells by ``efC`` ascending, and ``observed_efS``
        ascending.
    """
    # Group observations by construction setting (M, efC). The dominance
    # relationship applies ONLY to the interval dimension (efS); any other
    # parameter is a CONTROLLED VARIABLE — recorded as evidence, never compared.
    m_aliases = tuple(m_keys) if m_keys is not None else _M_FIELD_ALIASES
    efc_aliases = tuple(efc_keys) if efc_keys is not None else _EFC_FIELD_ALIASES
    efs_aliases = tuple(efs_keys) if efs_keys is not None else _EFS_FIELD_ALIASES
    controlled = tuple(controlled_fields)
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        m = _as_int(_pick_obs_field(obs, m_aliases))
        efc = _as_int(_pick_obs_field(obs, efc_aliases))
        efs = _as_int(_pick_obs_field(obs, efs_aliases))
        recall = _as_number(_pick_obs_field(obs, _RECALL_FIELD_ALIASES))
        qps = _as_number(_pick_obs_field(obs, _QPS_FIELD_ALIASES))
        if m is None or efc is None or efs is None or recall is None or qps is None:
            continue
        record = {"M": m, "efC": efc, "efS": efs, "recall": recall, "qps": qps}
        for field in controlled:
            value = _pick_obs_field(obs, (field,))
            if value is not None:
                record[field] = value
        cells[(m, efc)].append(record)

    # Build one row per M, one cell per efC
    rows_by_m: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (m, efc), cell_obs in sorted(cells.items()):
        infeasible = [o for o in cell_obs if o["recall"] < R_tau]
        feasible = [o for o in cell_obs if o["recall"] >= R_tau]

        # runtime.tex Algorithm 1 evidence semantics: L is the search setting
        # of the infeasible observation with the HIGHEST recall (R_min), U is
        # the search setting of the feasible observation with the HIGHEST QPS
        # (Q_min). Under monotone recall/QPS in efS this coincides with
        # max-infeasible-efS / min-feasible-efS, but remains correct on
        # non-monotone data.
        l_obs = max(infeasible, key=lambda o: o["recall"], default=None)
        u_obs = max(feasible, key=lambda o: o["qps"], default=None)
        L = l_obs["efS"] if l_obs is not None else None
        U = u_obs["efS"] if u_obs is not None else None

        if L is None and U is None:
            status = "unresolved"
        elif L is not None and U is None:
            status = "infeasible_only"
        elif L is None and U is not None:
            status = "feasible_observed"
        else:
            status = "bracketed"

        rows_by_m[m].append(
            {
                "efC": efc,
                "efS_interval": {"L": L, "U": U},
                "status": status,
                "recall_at_L": l_obs["recall"] if l_obs is not None else None,
                "recall_at_U": u_obs["recall"] if u_obs is not None else None,
                "qps_at_L": l_obs["qps"] if l_obs is not None else None,
                "qps_at_U": u_obs["qps"] if u_obs is not None else None,
                "observed_efS": sorted({o["efS"] for o in cell_obs}),
                "controlled": {
                    field: sorted(
                        {o[field] for o in cell_obs if field in o},
                        key=lambda v: (str(type(v)), str(v)),
                    )
                    for field in controlled
                },
                "L_obs_controlled": {
                    field: l_obs.get(field) for field in controlled
                } if l_obs is not None else None,
                "U_obs_controlled": {
                    field: u_obs.get(field) for field in controlled
                } if u_obs is not None else None,
            }
        )

    interval_table = [
        {"M": m, "efC_cells": cells_list}
        for m, cells_list in sorted(rows_by_m.items())
    ]

    return {
        "memory_type": "runtime_structural_interval_table",
        "objective": {
            "maximize": "QPS",
            "constraint": "Recall >= R_tau",
            "recall_threshold": R_tau,
        },
        "table_semantics": {
            "primary_key": "M",
            "secondary_key": "efC",
            "cell_value": "efS_interval",
            "interval_format": "(L, U]",
            "L": "efS of the infeasible observation with the highest recall (R_min evidence)",
            "U": "efS of the feasible observation with the highest QPS (Q_min evidence)",
            "R_min": "recall at the retained lower endpoint L (recall_at_L)",
            "Q_min": "QPS at the retained feasible endpoint U (qps_at_U)",
            "controlled": (
                "parameters other than (M, efC, efS) are CONTROLLED VARIABLES: "
                "recorded per cell as evidence only, never compared by dominance"
            ),
            "important_note": (
                "This table does not store exact efS*. It only stores "
                "interval evidence."
            ),
        },
        "interval_table": interval_table,
    }


def build_current_task_memory_prompt(
    runtime_structural_interval_table: Dict[str, Any],
) -> str:
    """Render the interval table into the prompt block provided to the LLM.

    Parameters
    ----------
    runtime_structural_interval_table : dict
        Output of :func:`build_runtime_structural_interval_table`.

    Returns
    -------
    str
        A text block ready to be concatenated into an LLM prompt, ending
        with the ``json.dumps``-serialized table.
    """
    table_json = json.dumps(
        runtime_structural_interval_table, ensure_ascii=False, indent=2
    )
    return (
        "Current-task memory is provided as a Runtime Structural Interval Table.\n"
        "\n"
        "The table is organized as:\n"
        "\n"
        "    M -> efC -> efS interval\n"
        "\n"
        "Each cell corresponds to one construction setting:\n"
        "\n"
        "    c = (M, efC)\n"
        "\n"
        "Each cell stores:\n"
        "\n"
        "    I(M, efC) = (L, U]\n"
        "\n"
        "where:\n"
        "\n"
        "    L = efS of the retained infeasible observation with the HIGHEST recall\n"
        "    U = efS of the retained feasible observation with the HIGHEST QPS\n"
        "\n"
        "I(M, efC) = (L, U] is an EVIDENCE interval, not an exact mathematical\n"
        "boundary: L summarizes the best infeasible-side evidence and U the best\n"
        "feasible-side evidence under this construction setting.\n"
        "\n"
        "Evidence reading: recall_at_L is the recall evidence at L (the lower\n"
        "recall evidence around the interval); qps_at_U is the conservative QPS\n"
        "evidence at U.\n"
        "\n"
        "This table does not store exact efS*. It only stores interval evidence.\n"
        "\n"
        "Use this table to reason about the next HNSW configuration.\n"
        "\n"
        "When reasoning:\n"
        "1. For the same (M, efC), do not choose efS <= L.\n"
        "2. If U exists, U is the smallest OBSERVED feasible efS — it is NOT a floor.\n"
        "   When the recall margin allows, probe efS BELOW U under the current (M, efC):\n"
        "   a smaller efS that stays feasible improves QPS.\n"
        "3. If efS is much larger than U, it may be conservative and may reduce QPS.\n"
        "4. Along the same M row, observe whether increasing efC reduces U.\n"
        "5. Along the same efC column, observe whether increasing M reduces U.\n"
        "6. Across construction settings: if an explored setting c'=(M', efC') is\n"
        "   component-wise weaker than the current c (M' <= M AND efC' <= efC),\n"
        "   its U(c') is an upper reference for search effort under c — a stronger\n"
        "   construction should need NO LARGER efS. Prefer efS around or below\n"
        "   U(c'). Do NOT compare crossed settings (one parameter larger, the\n"
        "   other smaller).\n"
        "7. Do not assume exact efS*.\n"
        "8. Do not claim any configuration is guaranteed optimal.\n"
        "\n"
        "runtime_structural_interval_table:\n"
        "\n"
        f"{table_json}\n"
    )


def classify_recall_zone(
    recall_margin: float,
    delta_low: float,
    delta_high: float,
) -> str:
    """Map a recall margin to one of the four recall zones.

    Parameters
    ----------
    recall_margin : float
        ``recall - recall_threshold``.
    delta_low : float
        Width of the *lower* near-boundary band (below threshold).
    delta_high : float
        Width of the *upper* near-boundary band (above threshold).

    Returns
    -------
    str
        One of ``"far-below"``, ``"near-below"``, ``"near-above"``,
        ``"far-above"``.
    """
    eps = _CLASSIFY_EPSILON
    if recall_margin < -delta_low - eps:
        return RECALL_FAR_BELOW
    if recall_margin < -eps:
        # -delta_low <= margin < 0
        return RECALL_NEAR_BELOW
    if recall_margin <= delta_high + eps:
        # 0 <= margin <= delta_high
        return RECALL_NEAR_ABOVE
    return RECALL_FAR_ABOVE


def calibrate_boundaries(
    records: Sequence[Dict[str, Any]],
    threshold: float,
    delta_noise: float = 0.001,
    quantile_p: float = 0.2,
) -> Tuple[float, float]:
    """Calibrate ``delta_low`` and ``delta_high`` from historical records.

    Only uses records from the *same* task (same recall threshold).
    Falls back to defaults when too few records exist.

    Parameters
    ----------
    records : sequence of dict
        Each record must have ``"recall"`` (float) and optionally
        ``"recall_threshold"`` (float).
    threshold : float
        The current task's recall threshold.
    delta_noise : float
        Floor value for both boundaries.
    quantile_p : float
        Quantile used for the lower boundary (0 < p <= 1).

    Returns
    -------
    (delta_low, delta_high) : (float, float)
    """
    default_low, default_high = _default_boundaries(threshold)

    # Filter to records for this task (matching threshold)
    margins: List[float] = []
    for rec in records:
        rec_thresh = float(rec.get("recall_threshold", threshold))
        if abs(rec_thresh - threshold) > 1e-9:
            continue
        rec_recall = float(rec.get("recall", 0.0))
        if rec_recall <= 0.0:
            continue
        margins.append(rec_recall - threshold)

    if len(margins) < _MIN_CALIBRATION_POINTS:
        return (default_low, default_high)

    # Separate negative and non-negative margins
    neg_margins = sorted(abs(m) for m in margins if m < -_CLASSIFY_EPSILON)
    pos_margins = sorted(m for m in margins if m >= -_CLASSIFY_EPSILON)

    # delta_low: quantile of |negative margins|
    if neg_margins and quantile_p > 0:
        idx = max(0, min(len(neg_margins) - 1, int(len(neg_margins) * quantile_p)))
        dl = max(delta_noise, neg_margins[idx])
    else:
        dl = default_low

    # delta_high: quantile of positive margins
    if pos_margins and quantile_p > 0:
        idx = max(0, min(len(pos_margins) - 1, int(len(pos_margins) * quantile_p)))
        dh = max(delta_noise, pos_margins[idx])
    else:
        dh = default_high

    return (dl, dh)


def log_ratio_distance(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    knobs: Sequence[str] = _HNSW_KNOBS,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted log-ratio distance between two configs.

    For each knob *k*, ``d_k = |log(a_k) - log(b_k)|``.  The total
    distance is ``sum(w_k * d_k) / sum(w_k)``.

    Returns ``inf`` if no knob values overlap.
    """
    if weights is None:
        weights = _DEFAULT_KNOB_WEIGHTS

    total_weight = 0.0
    total_dist = 0.0

    for k in knobs:
        a_val = config_a.get(k)
        b_val = config_b.get(k)
        if a_val is None or b_val is None:
            continue
        try:
            a_f = float(a_val)
            b_f = float(b_val)
        except (TypeError, ValueError):
            continue
        if a_f <= 0 or b_f <= 0:
            continue

        w = weights.get(k, 1.0)
        d_k = abs(math.log(a_f) - math.log(b_f))
        total_dist += w * d_k
        total_weight += w

    if total_weight <= 0:
        return float("inf")
    return total_dist / total_weight


def extract_metric_pattern(
    structured_state: Dict[str, Any],
    reference: Optional[Dict[str, float]] = None,
) -> str:
    """Extract a metric-pattern label from a structured state.

    Analyses normalized diagnostic metrics and returns one of the
    known ``METRIC_PATTERNS`` strings.

    Parameters
    ----------
    structured_state : dict
        Must contain ``"normalized_diagnostic_metrics"`` and
        ``"recall_margin"``, ``"recall"``.
    reference : dict or None
        Optional reference medians for comparison-based pattern detection.
    """
    metrics = structured_state.get("normalized_diagnostic_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    margin = float(structured_state.get("recall_margin", 0.0))
    recall = float(structured_state.get("recall", 0.0))
    max_recall = float(metrics.get("max_recall", recall))

    # ── Primary signals ──────────────────────────────────────────────
    visited = float(metrics.get("visited_nodes_per_query", 0))
    dist_comps = float(metrics.get("distance_computations", 0))
    out_degree = float(metrics.get("avg_out_degree", 0))
    in_degree = float(metrics.get("avg_in_degree", 0))
    index_size = float(metrics.get("index_size", 0))

    # ── Detect graph connectivity bottleneck ─────────────────────────
    # max_recall < threshold → structural insufficiency (Signal 7 analog)
    # Only trigger when max_recall is explicitly provided and the M value
    # is small enough that the structural ceiling is a plausible concern.
    if max_recall > 0 and recall > 0 and max_recall < 0.99:
        config = structured_state.get("config") or {}
        m_val = float(config.get("M", 0))
        if m_val > 0 and m_val <= 12:
            # For small M, the recall ceiling is empirically constrained
            # M=4: ~0.98, M=8: ~0.998, so M<=12 with max_recall < 0.99 is suspicious
            if max_recall < 0.985:
                return "graph_connectivity_bottleneck"
        elif m_val > 0 and max_recall < 0.95:
            # For larger M, very low max_recall still indicates a problem
            return "graph_connectivity_bottleneck"

    # ── Detect low search exploration ────────────────────────────────
    if margin < 0 and visited > 0:
        # recall below threshold with low visited_nodes → search too shallow
        if reference and "visited_nodes_per_query" in reference:
            ref_visited = reference["visited_nodes_per_query"]
            if ref_visited > 0 and visited < ref_visited * 0.7:
                return "low_search_exploration"
        # heuristic: visited_nodes < 100 suggests light search
        if visited < 100:
            return "low_search_exploration"

    # ── Detect search width saturation ───────────────────────────────
    if margin < 0 and dist_comps > 0 and visited > 0:
        ratio = dist_comps / max(1.0, visited)
        # high dist_comps per visited_node → diminishing returns from more search
        if ratio > 20:
            return "search_width_saturation"

    # ── Detect excessive search cost with slack ──────────────────────
    if margin > 0.02 and dist_comps > 0:
        if reference and "distance_computations" in reference:
            ref_dc = reference["distance_computations"]
            if ref_dc > 0 and dist_comps > ref_dc * 1.5:
                return "excessive_search_cost_with_slack"
        if dist_comps > 500:
            return "excessive_search_cost_with_slack"

    # ── Detect near-boundary unsafe reduction ────────────────────────
    if -0.01 <= margin <= 0.01:
        return "near_boundary_unsafe_reduction"

    # ── Detect redundant graph density ───────────────────────────────
    if out_degree > 0:
        config = structured_state.get("config") or {}
        m_val = float(config.get("M", 0))
        if m_val > 0 and out_degree > m_val * 0.9:
            return "redundant_graph_density"
    if in_degree > 0 and out_degree > 0:
        # in_degree much higher than out_degree → redundant density
        if in_degree > out_degree * 2:
            return "redundant_graph_density"

    return "unknown"


def _knob_weights_for_pattern(
    pattern: str,
    pattern_weights: Optional[Dict[str, Dict[str, float]]] = None,
    default_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return per-knob distance weights for a given metric pattern."""
    pw = pattern_weights if pattern_weights is not None else _HNSW_PATTERN_WEIGHTS
    dw = default_weights if default_weights is not None else _HNSW_DEFAULT_KNOB_WEIGHTS
    return pw.get(pattern, dw)


# ---------------------------------------------------------------------------
# State diagnosis generation (rule-based, no LLM)
# ---------------------------------------------------------------------------


def generate_state_diagnosis(
    structured_state: Dict[str, Any],
    task_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a natural-language diagnosis of the current tuning state.

    This is a pure rule-based function — no LLM calls are made.

    The diagnosis covers:
    1. Current recall zone and what it means
    2. QPS competitiveness in the current task
    3. Likely bottleneck
    4. Recommended next move direction
    5. Primary risks
    """
    zone = structured_state.get("recall_state_class", RECALL_FAR_BELOW)
    margin = float(structured_state.get("recall_margin", 0.0))
    qps = float(structured_state.get("qps", 0.0))
    recall = float(structured_state.get("recall", 0.0))
    config = structured_state.get("config") or {}
    metrics = structured_state.get("normalized_diagnostic_metrics") or {}
    threshold = float((task_context or {}).get("recall_threshold", 0.95))

    # ── Zone description ─────────────────────────────────────────────
    zone_descriptions = {
        RECALL_FAR_BELOW: (
            f"far-below the recall threshold (margin={margin:+.4f}). "
            f"The configuration is significantly infeasible — recall is far "
            f"from the target τ={threshold:.4f}. Structural changes to the "
            f"graph (M, ef_construction) are likely needed."
        ),
        RECALL_NEAR_BELOW: (
            f"near-below the recall threshold (margin={margin:+.4f}). "
            f"The configuration is slightly below τ={threshold:.4f} but close "
            f"enough that a small ef increase or minor construction adjustment "
            f"may cross the boundary. This is a promising repair candidate if "
            f"QPS is competitive."
        ),
        RECALL_NEAR_ABOVE: (
            f"near-above the recall threshold (margin={margin:+.4f}). "
            f"The configuration meets the recall constraint with a small margin. "
            f"QPS optimization available via ef reduction; keep efS above the "
            f"interval table's L (largest observed infeasible efS)."
        ),
        RECALL_FAR_ABOVE: (
            f"far-above the recall threshold (margin={margin:+.4f}). "
            f"The configuration has substantial recall slack. Aggressive QPS "
            f"optimization by reducing ef (no rebuild) or M/ef_construction "
            f"(rebuild) is the primary opportunity."
        ),
    }
    zone_text = zone_descriptions.get(zone, f"in zone '{zone}' (margin={margin:+.4f}).")

    # ── QPS competitiveness ──────────────────────────────────────────
    qps_text = f"Current QPS is {qps:.1f}."
    best_qps = float(metrics.get("_best_feasible_qps", 0))
    if best_qps > 0 and qps > 0:
        ratio = qps / best_qps
        if ratio >= 1.0:
            qps_text += " This is the best feasible QPS observed in this task so far."
        elif ratio >= _QPS_COMPETITIVENESS_RATIO:
            qps_text += (
                f" QPS is {ratio*100:.0f}% of the best feasible QPS ({best_qps:.1f}) "
                f"— competitive."
            )
        else:
            qps_text += (
                f" QPS is only {ratio*100:.0f}% of the best feasible QPS ({best_qps:.1f}) "
                f"— not competitive."
            )

    # ── Bottleneck identification ────────────────────────────────────
    bottleneck = _diagnose_bottleneck(zone, margin, config, metrics)
    bottleneck_text = f"Likely bottleneck: {bottleneck}."

    # ── Recommended next move ────────────────────────────────────────
    next_move = _recommend_next_move(zone, margin, config, metrics, threshold)
    next_text = f"Recommended next direction: {next_move}."

    # ── Risk assessment ──────────────────────────────────────────────
    risks = _assess_risks(zone, margin, config, metrics, threshold)
    risk_text = f"Primary risks: {'; '.join(risks)}."

    return (
        f"Recall zone: {zone} | {zone_text}\n"
        f"QPS assessment: {qps_text}\n"
        f"Bottleneck: {bottleneck_text}\n"
        f"Next direction: {next_text}\n"
        f"Risks: {risk_text}"
    )


def _diagnose_bottleneck(
    zone: str,
    margin: float,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    """Identify the most likely performance bottleneck."""
    visited = float(metrics.get("visited_nodes_per_query", 0))
    dist_comps = float(metrics.get("distance_computations", 0))
    out_degree = float(metrics.get("avg_out_degree", 0))
    max_recall = float(metrics.get("max_recall", 0))

    m_val = float(config.get("M", 0))

    # search width insufficient (visited_nodes low relative to ef)
    if visited > 0 and visited < 50 and margin < 0:
        return "search width insufficient — visited_nodes very low, suggesting ef or graph connectivity is too low to reach the target region"

    # graph quality insufficient (max_recall below threshold)
    if max_recall > 0 and margin < 0 and max_recall < 0.99:
        return "graph quality insufficient — max_recall ceiling is below target, indicating M or ef_construction is structurally too low"

    # search cost excessive
    if dist_comps > 500 and margin > 0:
        return "search cost excessive — distance computations are high while recall slack exists, suggesting ef or M can be reduced for QPS gain"

    # graph density redundant
    if out_degree > 0 and m_val > 0 and out_degree > m_val * 0.9:
        return "graph density redundant — out_degree near or above M, suggesting M is over-provisioned and can be reduced"

    # zone-based defaults
    if zone == RECALL_FAR_BELOW:
        return "graph connectivity or search width — recall is far below threshold, requiring structural parameter increases"
    if zone == RECALL_NEAR_BELOW:
        return "search width marginally low — recall is just below threshold, a small ef increase may suffice"
    if zone == RECALL_NEAR_ABOVE:
        return "search cost slightly high — recall is just above threshold, careful ef reduction may yield QPS without recall violation"
    if zone == RECALL_FAR_ABOVE:
        return "search cost excessively high — large recall slack enables aggressive QPS optimization"

    return "unknown — insufficient diagnostic data"


def _recommend_next_move(
    zone: str,
    margin: float,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    threshold: float,
) -> str:
    """Recommend the next tuning direction based on zone and metrics."""
    max_recall = float(metrics.get("max_recall", 0))

    if zone == RECALL_FAR_BELOW:
        if max_recall > 0 and max_recall < threshold:
            return (
                "recall repair via graph quality improvement — "
                "max_recall < τ, so increase M first (+5~20), "
                "then increase ef_construction (+50~250)"
            )
        return (
            "recall repair via search width — "
            "max_recall ≥ τ, so increase ef first (+20~150, no rebuild), "
            "then increase ef_construction if needed"
        )

    if zone == RECALL_NEAR_BELOW:
        return (
            "recall repair — increase ef first (+2~10, no rebuild) "
            "to cross the recall threshold, then assess QPS competitiveness"
        )

    if zone == RECALL_NEAR_ABOVE:
        if margin > 0.001:
            return (
                "boundary local search — the recall margin "
                f"({margin:+.4f}) gives headroom: select the matching knowledge "
                "(matched cards, diagnostic tree, interval table, cross table) and "
                "let the evidence decide direction and magnitude. Estimate the "
                "recall-vs-ef slope from the CURRENT-TASK interval table's observed "
                "(efS, recall) pairs (other tasks' tables must NOT be used) and take "
                "the largest ef reduction that keeps recall above tau "
                "(no extra buffer) per that slope — avoid needlessly small steps "
                "when the slope is shallow"
            )
        return (
            "boundary local search — derive small evidence-based adjustments "
            "from the knowledge: keep efS > L, use the U-probe rule"
        )

    if zone == RECALL_FAR_ABOVE:
        return (
            "QPS optimization — select the matching knowledge and reduce the "
            "parameters the evidence indicates, to consume recall slack for QPS gain"
        )

    return "exploration — gather more observations to establish baseline"


def _assess_risks(
    zone: str,
    margin: float,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    threshold: float,
) -> List[str]:
    """List the primary risks for the current state."""
    risks: List[str] = []

    if zone in (RECALL_NEAR_ABOVE, RECALL_NEAR_BELOW):
        risks.append("recall violation — verify efS stays above L")

    if zone == RECALL_FAR_ABOVE:
        risks.append("QPS regression — aggressive parameter reduction may lose recall without proportional QPS gain")

    if zone == RECALL_FAR_BELOW:
        risks.append("construction cost increase — increasing M/ef_construction requires index rebuild")
        risks.append("budget exhaustion — repeated infeasible configurations waste the trial budget")

    best_qps = float(metrics.get("_best_feasible_qps", 0))
    current_qps = float(structured_state_get(config, "qps", 0) if isinstance(config, dict) else 0)
    if best_qps > 0 and current_qps > 0 and current_qps < best_qps * _QPS_COMPETITIVENESS_RATIO:
        risks.append("dominated feasible point — QPS significantly below best known feasible QPS")

    out_degree = float(metrics.get("avg_out_degree", 0))
    m_val = float(config.get("M", 0))
    if m_val > 0 and out_degree > m_val * 1.2:
        risks.append("index size increase — high out_degree may inflate memory footprint")

    return risks if risks else ["no significant risks identified"]


def structured_state_get(d: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
    """Safely get a value from a dict that may not contain the key."""
    return d.get(key, default)


# ---------------------------------------------------------------------------
# PointLevelMemory
# ---------------------------------------------------------------------------


class PointLevelMemory:
    """Tracks every configuration point executed in the current task.

    Each point is classified by recall zone and quality label.
    Provides landscape anchors — a compact summary of the best, closest,
    and most representative points — for the LLM prompt.
    """

    def __init__(
        self,
        recall_threshold: float,
        delta_low: float,
        delta_high: float,
        path: Optional[Path] = None,
        knobs: Optional[Sequence[str]] = None,
    ) -> None:
        self._threshold = recall_threshold
        self._delta_low = delta_low
        self._delta_high = delta_high
        self._points: List[Dict[str, Any]] = []
        self._path = path
        self._knobs: Tuple[str, ...] = tuple(knobs) if knobs else _HNSW_KNOBS
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._points.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            self._points = []

    def _flush(self, record: Dict[str, Any]) -> None:
        if self._path is None:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Public API ───────────────────────────────────────────────────

    def add_point(self, full_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add a configuration point and persist it.

        Returns the point record, or ``None`` if the state is invalid.
        """
        ss = full_state.get("structured_state") or full_state
        if not isinstance(ss, dict):
            return None

        config = ss.get("config") or {}
        qps = float(ss.get("qps", 0))
        recall = float(ss.get("recall", 0))
        margin = float(ss.get("recall_margin", 0))
        zone = ss.get("recall_state_class") or classify_recall_zone(
            margin, self._delta_low, self._delta_high
        )
        diag_metrics = ss.get("normalized_diagnostic_metrics") or {}
        diagnosis = full_state.get("state_diagnosis", "")

        # Determine quality label
        quality = self._classify_quality(zone, qps, recall, config)

        point: Dict[str, Any] = {
            "config": config,
            "qps": qps,
            "recall": recall,
            "recall_margin": margin,
            "recall_state_class": zone,
            "normalized_diagnostic_metrics": diag_metrics,
            "quality_label": quality,
            "state_diagnosis": diagnosis,
            "timestamp": _utc_timestamp(),
        }

        self._points.append(point)
        self._flush(point)
        return point

    def contains_config(self, config: Dict[str, Any]) -> bool:
        """Return True when a point with the same parameter values exists."""
        if not isinstance(config, dict):
            return False
        for p in self._points:
            if (p.get("config") or {}) == config:
                return True
        return False

    def classify_point(self, full_state: Dict[str, Any]) -> str:
        """Return the quality label for a full state (without persisting)."""
        ss = full_state.get("structured_state") or full_state
        if not isinstance(ss, dict):
            return QUALITY_FAILED

        config = ss.get("config") or {}
        qps = float(ss.get("qps", 0))
        recall = float(ss.get("recall", 0))
        margin = float(ss.get("recall_margin", 0))
        zone = ss.get("recall_state_class") or classify_recall_zone(
            margin, self._delta_low, self._delta_high
        )
        return self._classify_quality(zone, qps, recall, config)

    def get_landscape_anchors(self, k_per_bucket: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """Return top-k representative points per quality bucket.

        Returns
        -------
        dict
            Keys are the four quality labels; values are lists of up to
            *k_per_bucket* point records.
        """
        buckets: Dict[str, List[Dict[str, Any]]] = {
            QUALITY_HIGH: [],
            QUALITY_NEAR: [],
            QUALITY_LOW: [],
            QUALITY_FAILED: [],
        }

        for p in self._points:
            label = p.get("quality_label", QUALITY_FAILED)
            if label in buckets:
                buckets[label].append(p)

        # Sort each bucket by QPS descending (best first)
        for label in buckets:
            buckets[label].sort(key=lambda p: p.get("qps", 0), reverse=True)

        return {label: pts[:k_per_bucket] for label, pts in buckets.items()}

    @property
    def points(self) -> List[Dict[str, Any]]:
        return list(self._points)

    def count(self) -> int:
        return len(self._points)

    def zone_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {z: 0 for z in RECALL_ZONES}
        for p in self._points:
            zone = p.get("recall_state_class", "")
            if zone in counts:
                counts[zone] += 1
        return counts

    def get_best_feasible(self) -> Optional[Dict[str, Any]]:
        """Return the feasible point with the highest QPS."""
        feasible = [
            p
            for p in self._points
            if p.get("recall_margin", -1.0) >= -_CLASSIFY_EPSILON
        ]
        if not feasible:
            return None
        return max(feasible, key=lambda p: p.get("qps", 0))

    def get_feasible_points(self) -> List[Dict[str, Any]]:
        """Return all feasible points (recall >= threshold)."""
        return [
            p
            for p in self._points
            if p.get("recall_margin", -1.0) >= -_CLASSIFY_EPSILON
        ]

    def failed_config_keys(self) -> List[Tuple[Any, ...]]:
        """Return param keys of failed/low-quality points to avoid re-visiting."""
        keys: List[Tuple[Any, ...]] = []
        for p in self._points:
            if p.get("quality_label") in (QUALITY_FAILED, QUALITY_LOW):
                config = p.get("config") or {}
                key = tuple(config.get(k) for k in self._knobs)
                keys.append(key)
        return keys

    def reference_medians(self, zone: Optional[str] = None) -> Dict[str, float]:
        """Compute median diagnostic metrics, optionally filtered by zone."""
        pts = self._points
        if zone:
            pts = [p for p in pts if p.get("recall_state_class") == zone]
        if not pts:
            pts = self._points
        if not pts:
            return {}

        metrics_keys = {
            "visited_nodes_per_query",
            "distance_computations",
            "avg_out_degree",
            "avg_in_degree",
            "index_size",
        }
        ref: Dict[str, float] = {}
        for k in metrics_keys:
            vals = []
            for p in pts:
                m = (p.get("normalized_diagnostic_metrics") or {}).get(k)
                if m is not None:
                    try:
                        vals.append(float(m))
                    except (TypeError, ValueError):
                        pass
            if vals:
                vals.sort()
                ref[k] = vals[len(vals) // 2]
        return ref

    # ── Internal helpers ─────────────────────────────────────────────

    def _classify_quality(
        self,
        zone: str,
        qps: float,
        recall: float,
        config: Dict[str, Any],
    ) -> str:
        """Determine the quality label for a point (runtime.tex point-memory types).

        The three tex types map onto the quality buckets:
          (1) Best Feasible Configurations      -> QUALITY_HIGH (feasible,
              competitive QPS)
          (2) Near-Threshold Infeasible Configurations -> QUALITY_NEAR
              (infeasible, recall within [R_tau - delta_low, R_tau);
              no QPS gate)
          (3) Failed Configurations             -> QUALITY_FAILED (far below
              the threshold OR poor QPS)
        """
        feasible = recall >= self._threshold - _CLASSIFY_EPSILON
        best = self.get_best_feasible()
        best_qps = float(best.get("qps", 0)) if best else 0.0

        # ── Failed: far below the threshold OR poor QPS (tex type 3) ──
        if zone == RECALL_FAR_BELOW:
            return QUALITY_FAILED
        if feasible and qps <= 0:
            return QUALITY_FAILED
        if feasible and best_qps > 0 and qps < best_qps * _QPS_SEVERE_DEGRADATION_RATIO:
            return QUALITY_FAILED

        # ── Best feasible (tex type 1) ─────────────────────────────
        if feasible:
            if best_qps <= 0 or qps >= best_qps * _QPS_COMPETITIVENESS_RATIO:
                return QUALITY_HIGH
            return QUALITY_LOW

        # ── Near-threshold infeasible (tex type 2): no QPS gate ─────
        if zone == RECALL_NEAR_BELOW:
            return QUALITY_NEAR

        # ── Default ───────────────────────────────────────────────
        return QUALITY_LOW

    def _is_dominated(
        self,
        qps: float,
        recall: float,
    ) -> bool:
        """Check if (qps, recall) is dominated by any stored feasible point."""
        for p in self._points:
            p_recall = float(p.get("recall", 0))
            p_qps = float(p.get("qps", 0))
            if p_recall < self._threshold - _CLASSIFY_EPSILON:
                continue
            # p dominates if p.qps >= qps AND p.recall >= recall (at least one strict)
            if p_qps >= qps and p_recall >= recall:
                if p_qps > qps or p_recall > recall:
                    return True
        return False


# ---------------------------------------------------------------------------
# TransitionLevelMemory
# ---------------------------------------------------------------------------


class TransitionLevelMemory:
    """Records real state-action-state transitions from the current task.

    Compresses raw transitions into action reflections and supports
    dual-view retrieval for the LLM proposal prompt.
    """

    def __init__(
        self,
        parameter_space: Any = None,
        recall_threshold: float = 0.95,
        delta_low: float = 0.01,
        delta_high: float = 0.01,
        path: Optional[Path] = None,
        knobs: Optional[Sequence[str]] = None,
        default_knob_weights: Optional[Dict[str, float]] = None,
        pattern_weights: Optional[Dict[str, Dict[str, float]]] = None,
        algorithm_name: str = "HNSW",
    ) -> None:
        self._space = parameter_space
        self._threshold = recall_threshold
        self._delta_low = delta_low
        self._delta_high = delta_high
        self._transitions: List[Dict[str, Any]] = []
        self._reflections: List[Dict[str, Any]] = []
        self._path = path
        self._knobs: Tuple[str, ...] = tuple(knobs) if knobs else _HNSW_KNOBS
        self._default_knob_weights = default_knob_weights if default_knob_weights is not None else _HNSW_DEFAULT_KNOB_WEIGHTS
        self._pattern_weights = pattern_weights if pattern_weights is not None else _HNSW_PATTERN_WEIGHTS
        self._algorithm_name = algorithm_name
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # Restore state_key from serialised form
                        sk = rec.get("state_key")
                        if isinstance(sk, list):
                            rec["state_key"] = tuple(sk)
                        self._transitions.append(rec)
                        # Also store as reflection if it has reflection_text
                        if rec.get("reflection_text"):
                            self._reflections.append(rec)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            self._transitions = []
            self._reflections = []

    def _flush(self, record: Dict[str, Any]) -> None:
        if self._path is None:
            return
        # Serialise tuple state_key as list for JSON
        serialisable = dict(record)
        sk = serialisable.get("state_key")
        if isinstance(sk, tuple):
            serialisable["state_key"] = list(sk)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serialisable, ensure_ascii=False) + "\n")

    # ── Public API ───────────────────────────────────────────────────

    def add_transition(
        self,
        before_state: Dict[str, Any],
        action: Dict[str, Any],
        after_state: Dict[str, Any],
        outcome_label: str,
        llm_caller: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Record a transition and build its action reflection.

        Uses LLM for reflection text generation when *llm_caller* is provided.

        Returns the reflection dict, or ``None`` if inputs are invalid.
        """
        before_ss = before_state.get("structured_state") or before_state
        after_ss = after_state.get("structured_state") or after_state
        if not isinstance(before_ss, dict) or not isinstance(after_ss, dict):
            return None

        reflection = self.build_action_reflection(
            before_state, action, after_state, outcome_label, llm_caller=llm_caller,
        )
        if reflection is None:
            return None

        self._transitions.append(reflection)
        self._reflections.append(reflection)
        self._flush(reflection)
        return reflection

    def build_action_reflection(
        self,
        before_state: Dict[str, Any],
        action: Dict[str, Any],
        after_state: Dict[str, Any],
        outcome_label: str,
        llm_caller: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a compact action reflection from a raw transition.

        Uses **LLM-based reflection generation** when *llm_caller* is
        provided; falls back to template-based generation otherwise.

        Returns a dict with ``reflection_text``, ``state_key``,
        ``config_signature``, and all raw transition fields.
        """
        before_ss = before_state.get("structured_state") or before_state
        after_ss = after_state.get("structured_state") or after_state
        if not isinstance(before_ss, dict) or not isinstance(after_ss, dict):
            return None

        # Extract state key
        zone_before = before_ss.get("recall_state_class", RECALL_FAR_BELOW)
        pattern = extract_metric_pattern(before_ss)
        state_key: Tuple[str, str] = (zone_before, pattern)

        # Config signature
        from_cfg = action.get("from_config") or {}
        to_cfg = action.get("to_config") or {}
        config_sig = _build_config_signature(from_cfg, to_cfg, knobs=self._knobs)

        # Build reflection text (LLM-based when caller available, else template)
        reflection_text = _build_reflection_text(
            before_state=before_state,
            action=action,
            after_state=after_state,
            outcome_label=outcome_label,
            threshold=self._threshold,
            llm_caller=llm_caller,
            knobs=self._knobs,
            algorithm_name=self._algorithm_name,
        )

        before_qps = float(before_ss.get("qps", 0))
        after_qps = float(after_ss.get("qps", 0))
        before_recall = float(before_ss.get("recall", 0))
        after_recall = float(after_ss.get("recall", 0))

        qps_delta = after_qps - before_qps if before_qps > 0 else 0.0
        recall_delta = after_recall - before_recall

        return {
            "before_state": before_state,
            "action": action,
            "after_state": after_state,
            "outcome_label": outcome_label,
            "reflection_text": reflection_text,
            "state_key": state_key,
            "config_signature": config_sig,
            "before_recall_margin": float(before_ss.get("recall_margin", 0)),
            "after_recall_margin": float(after_ss.get("recall_margin", 0)),
            "qps_delta": round(qps_delta, 2),
            "recall_delta": round(recall_delta, 6),
            "before_qps": before_qps,
            "after_qps": after_qps,
            "before_recall": before_recall,
            "after_recall": after_recall,
            "timestamp": _utc_timestamp(),
        }

    def retrieve_by_state_key(
        self,
        current_full_state: Dict[str, Any],
        k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """State-conditioned retrieval.

        Finds action reflections whose before-state is similar to the
        current state, organised as supporting / repair / risk evidence.
        """
        current_ss = current_full_state.get("structured_state") or current_full_state
        if not isinstance(current_ss, dict):
            return {EVIDENCE_SUPPORTING: [], EVIDENCE_REPAIR: [], EVIDENCE_RISK: []}

        zone = current_ss.get("recall_state_class", RECALL_FAR_BELOW)
        pattern = extract_metric_pattern(current_ss)

        # Exact state_key match first, then zone-only, then any
        exact: List[Dict[str, Any]] = []
        zone_match: List[Dict[str, Any]] = []
        any_match: List[Dict[str, Any]] = []

        for ref in self._reflections:
            sk = ref.get("state_key")
            if isinstance(sk, tuple) and len(sk) == 2:
                ref_zone, ref_pattern = sk
                if ref_zone == zone and ref_pattern == pattern:
                    exact.append(ref)
                elif ref_zone == zone:
                    zone_match.append(ref)
                else:
                    any_match.append(ref)
            else:
                any_match.append(ref)

        # Combine with priority: exact > zone > any
        candidates = exact + zone_match + any_match

        return self._organise_by_outcome(candidates, k)

    def retrieve_by_config_neighborhood(
        self,
        current_config: Dict[str, Any],
        current_metric_pattern: str = "unknown",
        k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Config-neighbourhood retrieval.

        Finds action reflections whose before-config or after-config is
        close to *current_config* in log-ratio distance.
        """
        weights = _knob_weights_for_pattern(
            current_metric_pattern,
            pattern_weights=self._pattern_weights,
            default_weights=self._default_knob_weights,
        )
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for ref in self._reflections:
            action = ref.get("action") or {}
            from_cfg = action.get("from_config") or {}
            to_cfg = action.get("to_config") or {}

            d_from = log_ratio_distance(current_config, from_cfg, knobs=self._knobs, weights=weights)
            d_to = log_ratio_distance(current_config, to_cfg, knobs=self._knobs, weights=weights)
            d = min(d_from, d_to)

            if d < float("inf"):
                scored.append((d, ref))

        scored.sort(key=lambda item: item[0])
        nearest = [ref for _, ref in scored]

        return self._organise_by_outcome(nearest, k)

    def transitions(self) -> List[Dict[str, Any]]:
        return list(self._transitions)

    def count(self) -> int:
        return len(self._transitions)

    def statistics(self) -> Dict[str, Any]:
        """Return lightweight aggregate statistics over all transitions."""
        total = len(self._transitions)
        if total == 0:
            return {
                "total_transitions": 0,
                "strong_success": 0,
                "weak_success": 0,
                "failure": 0,
                "avg_qps_delta": 0.0,
                "avg_recall_delta": 0.0,
                "feasible_crossings": 0,
            }

        strong = sum(
            1 for t in self._transitions if t.get("outcome_label") == OUTCOME_STRONG
        )
        weak = sum(
            1 for t in self._transitions if t.get("outcome_label") == OUTCOME_WEAK
        )
        fail = sum(
            1 for t in self._transitions if t.get("outcome_label") == OUTCOME_FAIL
        )
        qps_deltas = [
            float(t.get("qps_delta", 0)) for t in self._transitions
        ]
        recall_deltas = [
            float(t.get("recall_delta", 0)) for t in self._transitions
        ]

        crossings = 0
        for t in self._transitions:
            bm = float(t.get("before_recall_margin", 0))
            am = float(t.get("after_recall_margin", 0))
            if (bm < -_CLASSIFY_EPSILON) != (am < -_CLASSIFY_EPSILON):
                crossings += 1

        return {
            "total_transitions": total,
            "strong_success": strong,
            "weak_success": weak,
            "failure": fail,
            "avg_qps_delta": round(sum(qps_deltas) / total, 2) if total else 0.0,
            "avg_recall_delta": round(sum(recall_deltas) / total, 6) if total else 0.0,
            "feasible_crossings": crossings,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _organise_by_outcome(
        self,
        reflections: List[Dict[str, Any]],
        k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Organise reflections into supporting / repair / risk buckets."""
        result: Dict[str, List[Dict[str, Any]]] = {
            EVIDENCE_SUPPORTING: [],
            EVIDENCE_REPAIR: [],
            EVIDENCE_RISK: [],
        }

        for ref in reflections:
            label = ref.get("outcome_label", "")
            if label == OUTCOME_STRONG:
                # strong_success → supporting (for any zone)
                if len(result[EVIDENCE_SUPPORTING]) < k:
                    result[EVIDENCE_SUPPORTING].append(ref)
            elif label == OUTCOME_WEAK:
                # weak_success → repair evidence
                if len(result[EVIDENCE_REPAIR]) < k:
                    result[EVIDENCE_REPAIR].append(ref)
            elif label == OUTCOME_FAIL:
                # failure → risk evidence
                if len(result[EVIDENCE_RISK]) < k:
                    result[EVIDENCE_RISK].append(ref)

        return result


# ---------------------------------------------------------------------------
# Reflection text generation (rule-based, no LLM)
# ---------------------------------------------------------------------------


def _build_config_signature(
    from_cfg: Dict[str, Any],
    to_cfg: Dict[str, Any],
    knobs: Optional[Sequence[str]] = None,
) -> str:
    """Build a compact config change signature like 'M:32→32,ef:100→150'."""
    if knobs is None:
        knobs = _HNSW_KNOBS
    parts = []
    for k in knobs:
        fv = from_cfg.get(k)
        tv = to_cfg.get(k)
        if fv is not None and tv is not None:
            parts.append(f"{k}:{fv}→{tv}")
    return ",".join(parts) if parts else "no_change"


def _build_reflection_llm_prompt(
    before_state: Dict[str, Any],
    action: Dict[str, Any],
    after_state: Dict[str, Any],
    outcome_label: str,
    threshold: float,
    knobs: Optional[Sequence[str]] = None,
    algorithm_name: str = "HNSW",
) -> str:
    """Build an LLM prompt for generating an action reflection."""
    if knobs is None:
        knobs = _HNSW_KNOBS
    before_ss = before_state.get("structured_state") or before_state
    after_ss = after_state.get("structured_state") or after_state

    before_zone = before_ss.get("recall_state_class", RECALL_FAR_BELOW)
    after_zone = after_ss.get("recall_state_class", RECALL_FAR_BELOW)
    before_qps = float(before_ss.get("qps", 0))
    after_qps = float(after_ss.get("qps", 0))
    before_recall = float(before_ss.get("recall", 0))
    after_recall = float(after_ss.get("recall", 0))
    before_margin = float(before_ss.get("recall_margin", 0))
    after_margin = float(after_ss.get("recall_margin", 0))
    from_cfg = action.get("from_config") or {}
    to_cfg = action.get("to_config") or {}

    # Parameter changes
    changes = []
    for k in knobs:
        fv = from_cfg.get(k)
        tv = to_cfg.get(k)
        if fv is not None and tv is not None:
            try:
                fvi, tvi = int(fv), int(tv)
                if tvi > fvi:
                    changes.append(f"increased {k} from {fvi} to {tvi}")
                elif tvi < fvi:
                    changes.append(f"decreased {k} from {fvi} to {tvi}")
            except (TypeError, ValueError):
                pass
    action_desc = "; ".join(changes) if changes else "no parameter changes"

    qps_delta = after_qps - before_qps if before_qps > 0 else 0.0
    recall_delta = after_recall - before_recall

    return (
        f"You are an {algorithm_name} tuning reflection writer. Summarise one state-action-state "
        "transition as a compact, actionable reflection for future LLM proposal generation.\n\n"
        "Before action:\n"
        f"  zone: {before_zone} | QPS: {before_qps:.1f} | recall: {before_recall:.4f} "
        f"(margin {before_margin:+.4f} vs τ={threshold:.4f})\n"
        f"  config: {json.dumps(from_cfg)}\n\n"
        "Action:\n"
        f"  {action_desc}\n\n"
        "After action:\n"
        f"  zone: {after_zone} | QPS: {after_qps:.1f} | recall: {after_recall:.4f} "
        f"(margin {after_margin:+.4f})\n"
        f"  QPS delta: {qps_delta:+.1f} | recall delta: {recall_delta:+.6f}\n"
        f"  config: {json.dumps(to_cfg)}\n\n"
        f"Outcome: {outcome_label}\n\n"
        "Write a 2-4 sentence reflection that:\n"
        "1. Characterises the before-state and what it suggested was needed\n"
        "2. Describes the action taken and the observed outcome\n"
        "3. States what lesson this provides for future tuning (supporting, repair, or risk)\n\n"
        "Return ONLY the reflection text, no JSON, no markdown formatting.\n"
    )


def _build_reflection_text(
    before_state: Dict[str, Any],
    action: Dict[str, Any],
    after_state: Dict[str, Any],
    outcome_label: str,
    threshold: float,
    llm_caller: Any = None,
    knobs: Optional[Sequence[str]] = None,
    algorithm_name: str = "HNSW",
) -> str:
    """Generate a natural-language reflection text for a transition.

    Uses **LLM-based generation** when *llm_caller* is provided;
    falls back to template-based generation otherwise.
    """
    if knobs is None:
        knobs = _HNSW_KNOBS

    # ── Try LLM-based generation first ──────────────────────────────
    if llm_caller is not None:
        try:
            prompt = _build_reflection_llm_prompt(
                before_state, action, after_state, outcome_label, threshold,
                knobs=knobs, algorithm_name=algorithm_name,
            )
            raw = llm_caller(prompt)
            if raw and len(raw.strip()) > 20:
                return raw.strip()
        except Exception:
            pass  # fall through to template

    # ── Template-based fallback ──────────────────────────────────────
    before_ss = before_state.get("structured_state") or before_state
    after_ss = after_state.get("structured_state") or after_state

    before_zone = before_ss.get("recall_state_class", RECALL_FAR_BELOW)
    after_zone = after_ss.get("recall_state_class", RECALL_FAR_BELOW)
    before_qps = float(before_ss.get("qps", 0))
    after_qps = float(after_ss.get("qps", 0))
    before_recall = float(before_ss.get("recall", 0))
    after_recall = float(after_ss.get("recall", 0))
    before_margin = float(before_ss.get("recall_margin", 0))
    after_margin = float(after_ss.get("recall_margin", 0))

    from_cfg = action.get("from_config") or {}
    to_cfg = action.get("to_config") or {}

    # ── Describe the starting state ──────────────────────────────────
    zone_descriptions = {
        RECALL_FAR_BELOW: f"far below the recall threshold τ={threshold:.4f} (margin={before_margin:+.4f})",
        RECALL_NEAR_BELOW: f"just below the recall threshold τ={threshold:.4f} (margin={before_margin:+.4f})",
        RECALL_NEAR_ABOVE: f"just above the recall threshold τ={threshold:.4f} (margin={before_margin:+.4f})",
        RECALL_FAR_ABOVE: f"well above the recall threshold τ={threshold:.4f} (margin={before_margin:+.4f})",
    }
    before_desc = zone_descriptions.get(before_zone, f"in zone '{before_zone}'")

    qps_context = (
        f"high QPS ({before_qps:.1f})"
        if before_qps > 5000
        else f"moderate QPS ({before_qps:.1f})"
    )

    # ── Describe the action ──────────────────────────────────────────
    changes = []
    for k in knobs:
        fv = from_cfg.get(k)
        tv = to_cfg.get(k)
        if fv is not None and tv is not None:
            try:
                fvi = int(fv)
                tvi = int(tv)
                if tvi > fvi:
                    changes.append(f"increased {k} from {fvi} to {tvi}")
                elif tvi < fvi:
                    changes.append(f"decreased {k} from {fvi} to {tvi}")
            except (TypeError, ValueError):
                pass

    action_desc = "; ".join(changes) if changes else "no parameter changes"

    # ── Describe the outcome ─────────────────────────────────────────
    after_desc = zone_descriptions.get(after_zone, f"in zone '{after_zone}'")
    qps_delta = after_qps - before_qps if before_qps > 0 else 0.0
    recall_delta = after_recall - before_recall

    qps_change = (
        f"QPS {'increased' if qps_delta >= 0 else 'decreased'} "
        f"by {abs(qps_delta):.1f} (from {before_qps:.1f} to {after_qps:.1f})"
    )
    recall_change = (
        f"recall {'improved' if recall_delta >= 0 else 'declined'} "
        f"by {abs(recall_delta):.4f} (from {before_recall:.4f} to {after_recall:.4f})"
    )

    # ── Build the lesson ─────────────────────────────────────────────
    outcome_descriptions = {
        OUTCOME_STRONG: (
            "This is a strong success. The action moved the configuration "
            "in a productive direction — similar actions under similar "
            "conditions are likely to produce further gains."
        ),
        OUTCOME_WEAK: (
            "This is a weak success. The action made partial progress but "
            "did not achieve a decisive improvement. It provides useful "
            "boundary information and step-size calibration. Similar actions "
            "may help but should be combined with other adjustments."
        ),
        OUTCOME_FAIL: (
            "This action failed to produce meaningful improvement. "
            "The same direction should be avoided under similar conditions. "
            "Consider the opposite direction or a different parameter axis."
        ),
    }
    lesson = outcome_descriptions.get(
        outcome_label,
        f"The outcome was {outcome_label}. Use this to calibrate future decisions.",
    )

    return (
        f"Before action, the configuration was {before_desc} with {qps_context}. "
        f"The action {action_desc}. "
        f"After the action, the configuration was {after_desc} with "
        f"{qps_change} and {recall_change}. "
        f"{lesson}"
    )


# ---------------------------------------------------------------------------
# CurrentTaskMemory (top-level facade)
# ---------------------------------------------------------------------------


class CurrentTaskMemory:
    """Top-level memory facade for a single tuning task.

    Composes ``PointLevelMemory`` and ``TransitionLevelMemory`` and
    exposes the public API consumed by the tuning pipeline.

    Parameters
    ----------
    task_context : dict
        Must include ``"recall_threshold"`` (float) and optionally
        ``"param_order"``, ``"knob_bounds"``.
    parameter_space : object, optional
        A ``ParameterSpace``-compatible object that provides
        ``normalized_position`` and domain metadata.
    delta_noise : float
        Floor value for calibrated boundary widths.
    quantile_p : float
        Quantile used for calibration (0 < p <= 1).
    output_dir : Path or None
        If provided, JSONL persistence directory.
    trials_name : str
        Stem for JSONL files (e.g. ``"sift_95"``).
    """

    def __init__(
        self,
        task_context: Dict[str, Any],
        parameter_space: Any = None,
        delta_noise: float = 0.001,
        quantile_p: float = 0.2,
        output_dir: Optional[Path] = None,
        trials_name: str = "default",
        historical_records: Optional[Sequence[Dict[str, Any]]] = None,
        knobs: Optional[Sequence[str]] = None,
        default_knob_weights: Optional[Dict[str, float]] = None,
        pattern_weights: Optional[Dict[str, Dict[str, float]]] = None,
        algorithm_name: str = "HNSW",
    ) -> None:
        self._task_context = dict(task_context or {})
        self._space = parameter_space
        self._threshold = float(task_context.get("recall_threshold", 0.95))
        # Resolve knobs: constructor arg > task_context > HNSW default
        if knobs is not None:
            self._knobs: Tuple[str, ...] = tuple(knobs)
        elif task_context.get("param_order"):
            self._knobs = tuple(task_context.get("param_order", _HNSW_KNOBS))
        else:
            self._knobs = _HNSW_KNOBS
        # Resolve weights
        self._default_knob_weights = default_knob_weights if default_knob_weights is not None else _HNSW_DEFAULT_KNOB_WEIGHTS
        self._pattern_weights = pattern_weights if pattern_weights is not None else _HNSW_PATTERN_WEIGHTS
        self._algorithm_name = algorithm_name
        self._param_order: Tuple[str, ...] = tuple(
            task_context.get("param_order", self._knobs)
        )
        self._knob_bounds: Dict[str, Any] = dict(
            task_context.get("knob_bounds") or {}
        )

        # Boundary width calibration
        hist_records = list(historical_records or [])
        self._delta_low, self._delta_high = calibrate_boundaries(
            hist_records, self._threshold, delta_noise, quantile_p
        )

        # Persistence paths
        if output_dir is not None:
            mem_dir = output_dir / "current_task_memory"
            mem_dir.mkdir(parents=True, exist_ok=True)
            points_path = mem_dir / f"{trials_name}.points.jsonl"
            transitions_path = mem_dir / f"{trials_name}.transitions.jsonl"
        else:
            points_path = None
            transitions_path = None

        # Sub-stores
        self._point_memory = PointLevelMemory(
            recall_threshold=self._threshold,
            delta_low=self._delta_low,
            delta_high=self._delta_high,
            path=points_path,
            knobs=self._knobs,
        )
        self._transition_memory = TransitionLevelMemory(
            parameter_space=self._space,
            recall_threshold=self._threshold,
            delta_low=self._delta_low,
            delta_high=self._delta_high,
            path=transitions_path,
            knobs=self._knobs,
            default_knob_weights=self._default_knob_weights,
            pattern_weights=self._pattern_weights,
            algorithm_name=self._algorithm_name,
        )

        # Lightweight online statistics
        self._stats: Dict[str, Any] = {
            "total_points": 0,
            "total_transitions": 0,
            "state_action_success": defaultdict(int),
            "state_action_failure": defaultdict(int),
            "total_qps_delta": 0.0,
            "total_recall_delta": 0.0,
            "feasible_crossings": 0,
        }

        # Keep track of the last full state for transition building
        self._last_full_state: Optional[Dict[str, Any]] = None

    @property
    def boundaries(self) -> Tuple[float, float]:
        """Return the current ``(delta_low, delta_high)`` boundary widths."""
        return (self._delta_low, self._delta_high)

    # ── State construction ───────────────────────────────────────────

    def build_structured_state(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a structured numeric state from a raw observation dict.

        Parameters
        ----------
        observation : dict
            Expected keys: ``"config"``, ``"qps"``, ``"recall"``,
            ``"recall_threshold"``, ``"diagnostic_metrics"``.

        Returns
        -------
        dict
            Structured state with ``config``, ``recall_margin``, ``qps``,
            ``recall``, ``recall_state_class``,
            ``normalized_diagnostic_metrics``.
        """
        config = dict(observation.get("config") or {})
        qps = float(observation.get("qps", 0))
        recall = float(observation.get("recall", 0))
        threshold = float(
            observation.get("recall_threshold", self._threshold)
        )
        margin = recall - threshold
        zone = classify_recall_zone(margin, self._delta_low, self._delta_high)

        # Normalize diagnostic metrics
        raw_metrics = dict(observation.get("diagnostic_metrics") or {})
        normalized = self._normalize_diagnostic_metrics(raw_metrics)

        return {
            "config": config,
            "recall_margin": round(margin, 6),
            "qps": qps,
            "recall": recall,
            "recall_state_class": zone,
            "normalized_diagnostic_metrics": normalized,
        }

    def _normalize_diagnostic_metrics(
        self, raw_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Normalise raw diagnostic metrics into a standard float dict."""
        normalized: Dict[str, float] = {}
        # Map known metric keys to normalised names
        key_map = {
            "visited_nodes_per_query": "visited_nodes_per_query",
            "distance_computations": "distance_computations",
            "dist_comps_per_query": "distance_computations",
            "index_size": "index_size",
            "index_size_mb": "index_size",
            "avg_out_degree": "avg_out_degree",
            "out_degree_mean": "avg_out_degree",
            "avg_in_degree": "avg_in_degree",
            "in_degree_mean": "avg_in_degree",
            "build_time_s": "build_time_s",
            "max_recall": "max_recall",
            "candidate_distribution": "candidate_distribution",
            "candidate_distance_stats": "candidate_distance_stats",
        }
        for raw_key, norm_key in key_map.items():
            val = raw_metrics.get(raw_key)
            if val is not None:
                try:
                    normalized[norm_key] = float(val)
                except (TypeError, ValueError):
                    pass

        # Carry through best feasible QPS if available
        best_qps = raw_metrics.get("_best_feasible_qps")
        if best_qps is not None:
            try:
                normalized["_best_feasible_qps"] = float(best_qps)
            except (TypeError, ValueError):
                pass

        return normalized

    def generate_state_diagnosis(
        self,
        structured_state: Dict[str, Any],
    ) -> str:
        """Generate a natural-language diagnosis of the tuning state."""
        return generate_state_diagnosis(structured_state, self._task_context)

    def build_full_state(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a full state (structured + diagnosis) from an observation."""
        ss = self.build_structured_state(observation)
        diagnosis = self.generate_state_diagnosis(ss)
        return {
            "structured_state": ss,
            "state_diagnosis": diagnosis,
        }

    # ── Initialisation ───────────────────────────────────────────────

    def add_initial_observation(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add the very first observation as a point (no transition)."""
        full_state = self.build_full_state(observation)
        self._point_memory.add_point(full_state)
        self._last_full_state = full_state
        self._stats["total_points"] += 1
        return full_state

    def warm_up(
        self,
        existing_trials: Sequence[Dict[str, Any]],
    ) -> None:
        """Replay existing (same-task, resumed) trials through the memory.

        This restores point-level and transition-level state after a
        pipeline resume so that anchors and reflections survive restarts.
        """
        prev_full: Optional[Dict[str, Any]] = None
        for trial in existing_trials:
            if not isinstance(trial, dict):
                continue
            if trial.get("status") != "success":
                continue
            metrics = trial.get("metrics") or {}
            params = trial.get("params") or {}
            if not isinstance(metrics, dict) or not isinstance(params, dict):
                continue
            recall = float(metrics.get("recall", 0))
            if recall <= 0:
                continue

            obs = {
                "config": params,
                "qps": float(metrics.get("qps", 0)),
                "recall": recall,
                "recall_threshold": self._threshold,
                "diagnostic_metrics": self._extract_diagnostic_metrics(
                    metrics, params
                ),
            }
            fs = self.build_full_state(obs)
            # Dedup: persisted points.jsonl is already loaded at init, so
            # replaying the same trials on resume must not double-count.
            if self._point_memory.contains_config(params):
                prev_full = fs
                continue
            self._point_memory.add_point(fs)
            self._stats["total_points"] += 1

            if prev_full is not None:
                action = self.build_action(
                    prev_full["structured_state"]["config"],
                    fs["structured_state"]["config"],
                )
                label = self.assign_outcome_label(prev_full, action, fs)
                self._transition_memory.add_transition(
                    prev_full, action, fs, label
                )
                self._stats["total_transitions"] += 1

            prev_full = fs

        if prev_full is not None:
            self._last_full_state = prev_full

    def _extract_diagnostic_metrics(
        self,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract diagnostic metrics from a trial's metrics dict."""
        diag: Dict[str, Any] = {}
        for key in (
            "visited_nodes_per_query",
            "dist_comps_per_query",
            "distance_computations",
            "index_size_mb",
            "out_degree_mean",
            "in_degree_mean",
            "in_degree_std",
            "in_degree_max",
            "build_time_s",
            "candidate_distance_stats",
            "candidate_distribution",
        ):
            val = metrics.get(key)
            if val is not None:
                diag[key] = val
        # max_recall from frontier_summary if available
        fs = metrics.get("frontier_summary") or {}
        if isinstance(fs, dict) and fs.get("max_recall") is not None:
            diag["max_recall"] = float(fs["max_recall"])
        # selected_ef
        if metrics.get("selected_ef") is not None:
            diag["selected_ef"] = metrics["selected_ef"]
        return diag

    # ── Action construction ──────────────────────────────────────────

    def build_action(
        self,
        from_config: Dict[str, Any],
        to_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an action dict from two configs."""
        return {
            "from_config": dict(from_config or {}),
            "to_config": dict(to_config or {}),
        }

    # ── Online update ────────────────────────────────────────────────

    def update_memory(
        self,
        before_state: Optional[Dict[str, Any]],
        action: Optional[Dict[str, Any]],
        after_observation: Dict[str, Any],
        llm_caller: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Full online update after executing a new configuration.

        Uses **LLM-based reflection generation** when *llm_caller* is provided.

        1. Build after_full_state from observation
        2. Add to point-level memory
        3. Build transition if before_state exists
        4. Assign outcome label
        5. Compress to action reflection (LLM or template)
        6. Update lightweight statistics

        Parameters
        ----------
        before_state : dict or None
            The full state before the action (may be None for first point).
        action : dict or None
            The action dict with ``from_config`` / ``to_config``.
        after_observation : dict
            Raw observation from the executed configuration.
        llm_caller : callable or None
            Optional LLM caller for reflection text generation.

        Returns
        -------
        dict or None
            The action reflection if a transition was built, else None.
        """
        # 1-2. Build after-state and add point
        after_fs = self.build_full_state(after_observation)
        self._point_memory.add_point(after_fs)
        self._stats["total_points"] += 1

        reflection = None

        # 3-7. Build transition if we have a before-state
        if before_state is not None and action is not None:
            # 5. Assign outcome label
            label = self.assign_outcome_label(before_state, action, after_fs)

            # 6-7. Add transition and compress to reflection
            reflection = self._transition_memory.add_transition(
                before_state, action, after_fs, label, llm_caller=llm_caller,
            )

            # 9. Update lightweight statistics
            if reflection is not None:
                self._stats["total_transitions"] += 1
                self._stats["total_qps_delta"] += float(
                    reflection.get("qps_delta", 0)
                )
                self._stats["total_recall_delta"] += float(
                    reflection.get("recall_delta", 0)
                )

                sk = reflection.get("state_key")
                if isinstance(sk, tuple):
                    sk_str = str(sk)
                    if label == OUTCOME_FAIL:
                        self._stats["state_action_failure"][sk_str] += 1
                    else:
                        self._stats["state_action_success"][sk_str] += 1

                bm = float(reflection.get("before_recall_margin", 0))
                am = float(reflection.get("after_recall_margin", 0))
                if (bm < -_CLASSIFY_EPSILON) != (am < -_CLASSIFY_EPSILON):
                    self._stats["feasible_crossings"] += 1

        self._last_full_state = after_fs
        return reflection

    # ── Outcome labelling ────────────────────────────────────────────

    def assign_outcome_label(
        self,
        before_state: Dict[str, Any],
        action: Dict[str, Any],
        after_state: Dict[str, Any],
    ) -> str:
        """Assign an outcome label to a state-action-state transition.

        Returns one of ``"strong_success"``, ``"weak_success"``,
        ``"failure"``.
        """
        before_ss = before_state.get("structured_state") or before_state
        after_ss = after_state.get("structured_state") or after_state

        before_qps = float(before_ss.get("qps", 0))
        after_qps = float(after_ss.get("qps", 0))
        before_recall = float(before_ss.get("recall", 0))
        after_recall = float(after_ss.get("recall", 0))
        before_margin = float(before_ss.get("recall_margin", 0))
        after_margin = float(after_ss.get("recall_margin", 0))
        before_zone = before_ss.get("recall_state_class", RECALL_FAR_BELOW)
        after_zone = after_ss.get("recall_state_class", RECALL_FAR_BELOW)

        before_feasible = before_margin >= -_CLASSIFY_EPSILON
        after_feasible = after_margin >= -_CLASSIFY_EPSILON

        qps_ratio = after_qps / max(1.0, before_qps) if before_qps > 0 else 1.0
        best_qps = self._get_best_feasible_qps()
        after_competitive = (
            best_qps <= 0 or after_qps >= best_qps * _QPS_COMPETITIVENESS_RATIO
        )

        # ── strong_success conditions ────────────────────────────────

        # Case 1: crossed from infeasible to feasible with competitive QPS
        if not before_feasible and after_feasible and after_competitive:
            return OUTCOME_STRONG

        # Case 2: stayed feasible, consumed recall slack for significant QPS gain
        if (
            before_feasible
            and after_feasible
            and qps_ratio >= _STRONG_SUCCESS_QPS_RATIO
        ):
            return OUTCOME_STRONG

        # Case 3: near-below with high QPS became feasible and still competitive
        if (
            before_zone == RECALL_NEAR_BELOW
            and after_feasible
            and after_competitive
            and before_qps >= best_qps * _QPS_COMPETITIVENESS_RATIO
        ):
            return OUTCOME_STRONG

        # Case 4: after QPS exceeds or is close to best known feasible QPS
        if after_feasible and best_qps > 0 and after_qps >= best_qps:
            return OUTCOME_STRONG

        # ── failure conditions ────────────────────────────────────────

        # Case 1: stayed far-below without meaningful recall improvement
        recall_improvement = after_recall - before_recall
        if (
            after_zone == RECALL_FAR_BELOW
            and recall_improvement < 0.005
        ):
            return OUTCOME_FAIL

        # Case 1b: stayed far-below and QPS also dropped → definite failure
        if after_zone == RECALL_FAR_BELOW and qps_ratio < 0.95:
            return OUTCOME_FAIL

        # Case 2: fell from feasible to infeasible
        if before_feasible and not after_feasible:
            return OUTCOME_FAIL

        # Case 3: stayed feasible but QPS dropped without enough recall gain
        if (
            before_feasible
            and after_feasible
            and qps_ratio < 0.95
            and after_recall - before_recall < 0.005
        ):
            return OUTCOME_FAIL

        # Case 4: feasible but severely dominated
        if after_feasible and best_qps > 0 and after_qps < best_qps * 0.5:
            return OUTCOME_FAIL

        # Case 5: severe QPS regression
        if before_qps > 0 and qps_ratio < 0.7:
            return OUTCOME_FAIL

        # ── weak_success (everything else that isn't clearly a failure) ──

        # near-below with high QPS → repair potential
        if after_zone == RECALL_NEAR_BELOW and after_competitive:
            return OUTCOME_WEAK

        # recall margin meaningfully improved toward threshold
        if after_margin - before_margin >= 0.001:
            return OUTCOME_WEAK

        # successful recall repair but QPS not competitive
        if not before_feasible and after_feasible:
            return OUTCOME_WEAK

        # small QPS gain while staying near boundary
        if after_feasible and qps_ratio >= _WEAK_SUCCESS_QPS_GAIN:
            return OUTCOME_WEAK

        # default: no clear success signal → failure
        return OUTCOME_FAIL

    def _get_best_feasible_qps(self) -> float:
        best = self._point_memory.get_best_feasible()
        return float(best.get("qps", 0)) if best else 0.0

    # ── Memory retrieval ─────────────────────────────────────────────

    def retrieve_memory_context(
        self,
        current_full_state: Optional[Dict[str, Any]] = None,
        current_config: Optional[Dict[str, Any]] = None,
        k: int = 8,
        ablation: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Dual-view memory retrieval for LLM proposal generation.

        Parameters
        ----------
        ablation : dict or None
            Optional ablation flags: ``no_point_memory``, ``no_action_memory``,
            ``no_state_conditioned``, ``no_config_neighborhood``.
            When True, the corresponding section is skipped.
        """
        abl = ablation or {}
        fs = current_full_state or self._last_full_state
        if fs is None:
            return _empty_memory_context()

        cfg = current_config
        if cfg is None and fs is not None:
            ss = fs.get("structured_state") or fs
            cfg = ss.get("config") or {}

        current_diag = fs.get("state_diagnosis", "")

        # ── Landscape anchors (point-level memory) ────────────────────
        if abl.get("no_point_memory"):
            anchors = {q: [] for q in QUALITY_LABELS}
        else:
            anchors = self._point_memory.get_landscape_anchors(k_per_bucket=2)

        # ── State-conditioned reflections ─────────────────────────────
        if abl.get("no_action_memory") or abl.get("no_state_conditioned"):
            state_refs = {EVIDENCE_SUPPORTING: [], EVIDENCE_REPAIR: [], EVIDENCE_RISK: []}
        else:
            state_refs = self._transition_memory.retrieve_by_state_key(fs, k=max(1, k // 2))

        # ── Config-neighborhood reflections ───────────────────────────
        if abl.get("no_action_memory") or abl.get("no_config_neighborhood"):
            config_refs = {EVIDENCE_SUPPORTING: [], EVIDENCE_BOUNDARY: [], EVIDENCE_RISK: []}
        else:
            ss = fs.get("structured_state") or fs
            pattern = extract_metric_pattern(ss)
            config_refs = self._transition_memory.retrieve_by_config_neighborhood(
                cfg, current_metric_pattern=pattern, k=max(1, k // 2)
            )

        # ── Contrastive reflections ───────────────────────────────────
        contrastive: List[Dict[str, Any]] = []
        if not abl.get("no_action_memory"):
            for ref in self._transition_memory.transitions():
                if ref.get("outcome_label") == OUTCOME_FAIL:
                    contrastive.append(ref)
                    if len(contrastive) >= max(1, k // 4):
                        break

        stats = self._transition_memory.statistics() if not abl.get("no_action_memory") else {}
        zone_counts = self._point_memory.zone_counts()

        context: Dict[str, Any] = {
            "current_state_diagnosis": current_diag,
            "landscape_anchors": anchors,
            "state_conditioned_reflections": state_refs,
            "config_neighborhood_reflections": config_refs,
            "contrastive_reflections": contrastive,
            "zone_counts": zone_counts,
            "statistics": stats,
        }

        # ── Runtime structural interval table ────────────────────────
        # Dominance applies to (M, efC) -> ef only (FilterDiskANN: R/FilterLBuild
        # -> L); other parameters are controlled variables recorded as evidence.
        if self._algorithm_name in {"HNSW", "UNIFY", "NHQ", "FilterDiskANN"} and not abl.get("no_point_memory"):
            build_kwargs: Dict[str, Any] = {}
            if self._algorithm_name == "UNIFY":
                build_kwargs["controlled_fields"] = ("B", "al")
            elif self._algorithm_name == "NHQ":
                build_kwargs["controlled_fields"] = ("weight",)
            elif self._algorithm_name == "FilterDiskANN":
                build_kwargs.update(
                    m_keys=("R",),
                    efc_keys=("FilterLBuild", "FilteredLBuild"),
                    efs_keys=("L",),
                    controlled_fields=("alpha",),
                )
            context["runtime_structural_interval_table"] = (
                build_runtime_structural_interval_table(
                    self._point_memory.points, self._threshold, **build_kwargs
                )
            )

        return context

    # ── Runtime Structural Interval Table ─────────────────────────────

    def build_runtime_structural_interval_table(self) -> Dict[str, Any]:
        """Build the interval table from all current-task points.

        Returns
        -------
        dict
            The ``runtime_structural_interval_table`` JSON summarizing the
            ``M -> efC -> efS`` feasible-interval evidence observed so far.
        """
        build_kwargs: Dict[str, Any] = {}
        if self._algorithm_name == "UNIFY":
            build_kwargs["controlled_fields"] = ("B", "al")
        elif self._algorithm_name == "NHQ":
            build_kwargs["controlled_fields"] = ("weight",)
        elif self._algorithm_name == "FilterDiskANN":
            build_kwargs.update(
                m_keys=("R",),
                efc_keys=("FilterLBuild", "FilteredLBuild"),
                efs_keys=("L",),
                controlled_fields=("alpha",),
            )
        return build_runtime_structural_interval_table(
            self._point_memory.points, self._threshold, **build_kwargs
        )

    def build_interval_memory_context(
        self,
        current_full_state: Optional[Dict[str, Any]] = None,
        current_config: Optional[Dict[str, Any]] = None,
        max_chars: int = 4000,
    ) -> Dict[str, Any]:
        """Main entry: return the interval table and its prompt context.

        Returns
        -------
        dict
            ``{"runtime_structural_interval_table": dict,
            "prompt_context": str}`` — the second value is ready to be
            concatenated into an LLM prompt.
        """
        table = self.build_runtime_structural_interval_table()
        memory_context = self.retrieve_memory_context(
            current_full_state=current_full_state,
            current_config=current_config,
        )
        memory_context["runtime_structural_interval_table"] = table
        prompt_context = self.format_memory_context(
            memory_context, max_chars=max_chars
        )
        return {
            "runtime_structural_interval_table": table,
            "prompt_context": prompt_context,
        }

    # ── Boundary width management ────────────────────────────────────

    def update_boundary_widths(self) -> None:
        """Recalibrate ``delta_low`` and ``delta_high`` from current-task points.

        Uses the points already stored in ``_point_memory``.
        """
        records: List[Dict[str, Any]] = []
        for p in self._point_memory.points:
            records.append(
                {
                    "recall": float(p.get("recall", 0)),
                    "recall_threshold": self._threshold,
                }
            )
        if records:
            self._delta_low, self._delta_high = calibrate_boundaries(
                records, self._threshold
            )

    # ── Formatting (for LLM prompts) ─────────────────────────────────

    @staticmethod
    def format_memory_context(
        memory_context: Dict[str, Any],
        max_chars: int = 4000,
        knobs: Optional[Sequence[str]] = None,
        ablation: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Render a memory context dict into LLM-readable markdown.

        Parameters
        ----------
        knobs : sequence of str or None
            Parameter knob names for display (defaults to HNSW knobs).
        ablation : dict or None
            Optional ablation flags to skip specific sections.
        """
        if knobs is None:
            knobs = _HNSW_KNOBS
        abl = ablation or {}
        if not memory_context or not memory_context.get("current_state_diagnosis"):
            return ""

        lines: List[str] = []
        lines.append("## Current Task Memory (this run only)")
        lines.append("")

        # ── Zone counts ────────────────────────────────────────────
        zc = memory_context.get("zone_counts") or {}
        if zc:
            parts = [f"{zone}×{zc.get(zone, 0)}" for zone in RECALL_ZONES]
            lines.append(f"**Zones**: {' | '.join(parts)}")
            lines.append("")

        # ── State diagnosis ────────────────────────────────────────
        diag = memory_context.get("current_state_diagnosis", "")
        if diag:
            lines.append("### Current State Diagnosis")
            lines.append(diag)
            lines.append("")

        # ── Statistics ─────────────────────────────────────────────
        stats = memory_context.get("statistics") or {}
        if stats and not abl.get("no_action_memory"):
            lines.append("### Online Statistics")
            total = stats.get("total_transitions", 0)
            strong = stats.get("strong_success", 0)
            weak = stats.get("weak_success", 0)
            fail = stats.get("failure", 0)
            avg_qps = stats.get("avg_qps_delta", 0)
            avg_rec = stats.get("avg_recall_delta", 0)
            crossings = stats.get("feasible_crossings", 0)
            lines.append(
                f"Transitions: {total} total ({strong} strong, {weak} weak, "
                f"{fail} failure) | avg ΔQPS: {avg_qps:+.1f} | "
                f"avg Δrecall: {avg_rec:+.6f} | crossings: {crossings}"
            )
            lines.append("")

        # ── Landscape anchors ──────────────────────────────────────
        if not abl.get("no_point_memory"):
            anchors = memory_context.get("landscape_anchors") or {}
            if anchors:
                lines.append("### Landscape Anchors")
                lines.append("")
                bucket_names = {
                    QUALITY_HIGH: "Best Feasible Configurations",
                    QUALITY_NEAR: "Near-Threshold Infeasible Configurations",
                    QUALITY_LOW: "Low Quality / Dominated",
                    QUALITY_FAILED: "Failed Configurations (avoid)",
                }
                for label in QUALITY_LABELS:
                    pts = anchors.get(label) or []
                    if not pts:
                        continue
                    lines.append(f"**{bucket_names.get(label, label)}**:")
                    for p in pts:
                        config = p.get("config") or {}
                        params_str = ", ".join(
                            f"{k}={config.get(k, '?')}" for k in knobs
                        )
                        zone = p.get("recall_state_class", "?")
                        margin = p.get("recall_margin", 0)
                        lines.append(
                            f"  - [{zone}] {params_str} | "
                            f"QPS={p.get('qps', 0):.1f} | "
                            f"recall={p.get('recall', 0):.4f} | "
                            f"margin={margin:+.4f}"
                        )
                    lines.append("")

        # ── State-conditioned reflections ──────────────────────────
        if not abl.get("no_action_memory") and not abl.get("no_state_conditioned"):
            state_refs = memory_context.get("state_conditioned_reflections") or {}
            if state_refs:
                lines.append("### Similar Past Experiences (State-Conditioned)")
                _format_reflection_group(lines, state_refs, EVIDENCE_SUPPORTING, "Supporting Evidence")
                _format_reflection_group(lines, state_refs, EVIDENCE_REPAIR, "Repair Evidence")
                _format_reflection_group(lines, state_refs, EVIDENCE_RISK, "Risk Evidence")
                lines.append("")

        # ── Config-neighborhood reflections ────────────────────────
        if not abl.get("no_action_memory") and not abl.get("no_config_neighborhood"):
            config_refs = memory_context.get("config_neighborhood_reflections") or {}
            if config_refs:
                lines.append("### Local Configuration Response (Config-Neighborhood)")
                _format_reflection_group(lines, config_refs, EVIDENCE_SUPPORTING, "Supporting")
                _format_reflection_group(lines, config_refs, EVIDENCE_REPAIR, "Boundary")
                _format_reflection_group(lines, config_refs, EVIDENCE_RISK, "Risk")
                lines.append("")

        # ── Contrastive reflections ────────────────────────────────
        if not abl.get("no_action_memory"):
            contrast = memory_context.get("contrastive_reflections") or []
            if contrast:
                lines.append("### Contrastive Risk Examples (What to Avoid)")
                for ref in contrast[:3]:
                    sig = ref.get("config_signature", "?")
                    text = ref.get("reflection_text", "")
                    if len(text) > 200:
                        text = text[:197] + "..."
                    lines.append(f"  - {sig}: {text}")
                lines.append("")

        return _truncate_text("\n".join(lines), max_chars)

    # ── Delegate properties ──────────────────────────────────────────

    @property
    def point_memory(self) -> PointLevelMemory:
        return self._point_memory

    @property
    def transition_memory(self) -> TransitionLevelMemory:
        return self._transition_memory

    @property
    def statistics(self) -> Dict[str, Any]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_reflection_group(
    lines: List[str],
    refs_by_role: Dict[str, List[Dict[str, Any]]],
    role: str,
    heading: str,
) -> None:
    """Format a group of reflections for the prompt."""
    refs = refs_by_role.get(role) or []
    if not refs:
        return
    lines.append(f"**{heading}** ({len(refs)}):")
    for ref in refs:
        sig = ref.get("config_signature", "?")
        label = ref.get("outcome_label", "?")
        text = ref.get("reflection_text", "")
        if len(text) > 180:
            text = text[:177] + "..."
        lines.append(f"  - [{label}] {sig}: {text}")


def _empty_memory_context() -> Dict[str, Any]:
    """Return an empty memory context."""
    return {
        "current_state_diagnosis": "",
        "landscape_anchors": {
            QUALITY_HIGH: [],
            QUALITY_NEAR: [],
            QUALITY_LOW: [],
            QUALITY_FAILED: [],
        },
        "state_conditioned_reflections": {
            EVIDENCE_SUPPORTING: [],
            EVIDENCE_REPAIR: [],
            EVIDENCE_RISK: [],
        },
        "config_neighborhood_reflections": {
            EVIDENCE_SUPPORTING: [],
            EVIDENCE_REPAIR: [],
            EVIDENCE_RISK: [],
        },
        "contrastive_reflections": [],
        "zone_counts": {},
        "statistics": {},
    }

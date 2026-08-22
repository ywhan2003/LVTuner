"""Static Knowledge module for LLM-based vector search tuning.

Provides recall-constraint-aware selection of pre-defined metric pattern
cards.  These cards encode stable domain knowledge (mechanism interpretation,
tuning direction, cost/risk) and are matched to the current tuning state
*before* proposal generation.

Key design principles
---------------------
* Recall-constraint branch routing — cards are partitioned into
  ``recall-infeasible`` and ``recall-feasible``; only cards from the
  active branch are considered.
* Rule-based signal matching — fast, deterministic, no LLM dependency.
* LLM-based description matching — optional, for higher precision when
  a ``llm_caller`` is available.
* Static knowledge **never** records task data — it only provides
  mechanism-level direction priors.

Classes
-------
MetricPatternCard    — dataclass for a single pattern card.
StaticKnowledgeBase  — registry of cards, organised by algorithm and branch.
StaticKnowledgeSelector — selects relevant cards for the current tuning state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECALL_INFEASIBLE = "recall-infeasible"
RECALL_FEASIBLE = "recall-feasible"

_DEFAULT_HNSW_CARDS_DIR = "knowledge_base/HNSW"
_DEFAULT_UNIFY_CARDS_DIR = "knowledge_base/UNIFY"

# Numeric thresholds for signal matching (HNSW)
_LOW_VISITED_NODES = 100
_HIGH_DIST_COMPS = 500
_LOW_OUT_DEGREE_RATIO = 0.3  # out_degree / M
_HIGH_EF_RATIO = 0.8  # ef / efC
_SMALL_MARGIN = 0.005
_LARGE_MARGIN = 0.02

# Numeric thresholds for UNIFY signal matching
_LOW_INCLUSIVENESS_PCT = 70.0  # inclusiveness below 70% is concerning
_HIGH_BUILD_TIME_S = 300.0  # build time above 300s is high
_LOW_AL_RATIO = 0.3  # al / efC ratio


# ---------------------------------------------------------------------------
# MetricPatternCard
# ---------------------------------------------------------------------------


@dataclass
class MetricPatternCard:
    """A single static-knowledge pattern card.

    Each card describes one recognisable metric pattern, its mechanism
    interpretation, the recommended tuning direction, and its
    applicability boundary.
    """

    id: str  # e.g. "HNSW.LowSearchExploration"
    algorithm: str  # e.g. "HNSW"
    recall_constraint_class: str  # "recall-infeasible" | "recall-feasible"
    description: str  # NL description for matching
    signals: List[str] = field(default_factory=list)  # Observable signal bullets
    interpretation: str = ""  # Mechanism explanation
    solution: str = ""  # Recommended tuning direction
    cost: str = ""  # Expected cost / downside
    boundary: str = ""  # Applicability boundary / risk
    confidence: float = 1.0
    source: str = "domain_knowledge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "algorithm": self.algorithm,
            "recall_constraint_class": self.recall_constraint_class,
            "description": self.description,
            "signals": self.signals,
            "interpretation": self.interpretation,
            "solution": self.solution,
            "cost": self.cost,
            "boundary": self.boundary,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPatternCard":
        return cls(
            id=str(data.get("id", "")),
            algorithm=str(data.get("algorithm", "")),
            recall_constraint_class=str(data.get("recall_constraint_class", "")),
            description=str(data.get("description", "")),
            signals=list(data.get("signals") or []),
            interpretation=str(data.get("interpretation", "")),
            solution=str(data.get("solution", "")),
            cost=str(data.get("cost", "")),
            boundary=str(data.get("boundary", "")),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "domain_knowledge")),
        )

    @classmethod
    def from_markdown(cls, text: str) -> Optional["MetricPatternCard"]:
        """Parse a card from the markdown format used in ``knowledge_base/HNSW/*.md``.

        Expected format::

            Recall Constraint Class: Recall-infeasible
            Pattern ID: HNSW.LowSearchExploration
            Description:
                ...
            Signals:
                - ...
            Interpretation:
                ...
            Solution:
                ...

        Returns ``None`` if the text cannot be parsed.
        """
        data: Dict[str, Any] = {}

        # ── Recall Constraint Class ────────────────────────────────────
        m = re.search(
            r"Recall\s*Constraint\s*Class\s*:?\s*\n?\s*(.+)",
            text,
            re.IGNORECASE,
        )
        if m:
            raw = m.group(1).strip().lower()
            if "infeasible" in raw:
                data["recall_constraint_class"] = RECALL_INFEASIBLE
            elif "feasible" in raw:
                data["recall_constraint_class"] = RECALL_FEASIBLE
            else:
                data["recall_constraint_class"] = raw

        # ── Pattern ID ─────────────────────────────────────────────────
        m = re.search(r"Pattern\s*ID\s*:?\s*\n?\s*(.+)", text, re.IGNORECASE)
        if m:
            data["id"] = m.group(1).strip()

        # ── Multi-line fields ──────────────────────────────────────────
        _extract_field(text, "Description", data, "description")
        _extract_field(text, "Interpretation", data, "interpretation")
        _extract_field(text, "Solution", data, "solution")
        _extract_field(text, "Cost", data, "cost")
        _extract_field(text, "Boundary", data, "boundary")

        # ── Signals (bullet list) ──────────────────────────────────────
        signals: List[str] = []
        in_signals = False
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^Signals?\s*:", stripped, re.IGNORECASE):
                in_signals = True
                continue
            if in_signals:
                if stripped.startswith("-"):
                    signals.append(stripped[1:].strip())
                elif re.match(r"^[A-Za-z]+:", stripped):
                    # Next field header
                    break
                elif stripped == "":
                    continue
                else:
                    # Continuation line or end of signals
                    if not stripped.startswith(("Recall", "Pattern", "Description", "Interpretation", "Solution", "Cost", "Boundary")):
                        signals.append(stripped)
                    else:
                        break
        data["signals"] = signals

        # ── Extract algorithm from ID ──────────────────────────────────
        if data.get("id") and not data.get("algorithm"):
            prefix = data["id"].split(".")[0] if "." in data["id"] else ""
            if prefix:
                data["algorithm"] = prefix
        if not data.get("algorithm"):
            data["algorithm"] = "HNSW"

        # ── Validate ───────────────────────────────────────────────────
        if not data.get("id"):
            return None
        if not data.get("recall_constraint_class"):
            return None
        if not data.get("description"):
            return None

        return cls.from_dict(data)


def _extract_field(
    text: str,
    field_name: str,
    data: Dict[str, Any],
    key: str,
) -> None:
    """Extract a multi-line text field from markdown.

    Supports two formats:
    1. ``Field: value on same line``
    2. ``Field:\\n    indented value on next line(s)``

    Captures everything until the next recognised field header or end of text.
    """
    # Try same-line format first: "Field: value"
    m = re.search(
        rf"{field_name}\s*:\s*(.+?)(?=\n\s*(?:Recall\s*Constraint|Pattern\s*ID|Description|Signals?|Interpretation|Solution|Cost|Boundary)\s*:?|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        value = m.group(1).strip()
        if value:
            # Clean up: collapse multiline indented content
            lines = value.splitlines()
            cleaned = " ".join(line.strip() for line in lines if line.strip())
            if cleaned:
                data[key] = cleaned
            return

    # Try multiline format: "Field:\n    indented content"
    pattern = (
        rf"{field_name}\s*:?\s*\n"
        rf"((?:\s{{4,}}[^\n]+\n?)+)"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        value = m.group(1).strip()
        if value:
            lines = value.splitlines()
            cleaned = " ".join(line.strip() for line in lines if line.strip())
            if cleaned:
                data[key] = cleaned


# ---------------------------------------------------------------------------
# StaticKnowledgeBase
# ---------------------------------------------------------------------------


class StaticKnowledgeBase:
    """Registry of metric pattern cards, organised by algorithm and branch.

    Parameters
    ----------
    cards_dir : str or Path, optional
        Directory containing ``*.md`` card files.  Passed through to
        ``load_default_hnsw_cards``.
    """

    def __init__(self, cards_dir: str | Path = _DEFAULT_HNSW_CARDS_DIR) -> None:
        self._cards: Dict[str, Dict[str, List[MetricPatternCard]]] = {}
        # _cards[algorithm][recall_constraint_class] = [card, ...]
        if cards_dir:
            self.load_default_hnsw_cards(cards_dir)

    # ── Card management ────────────────────────────────────────────────

    def add_card(self, card: MetricPatternCard) -> None:
        """Register a single metric pattern card."""
        alg = card.algorithm
        branch = card.recall_constraint_class
        if alg not in self._cards:
            self._cards[alg] = {}
        if branch not in self._cards[alg]:
            self._cards[alg][branch] = []
        # Avoid duplicates
        existing_ids = {c.id for c in self._cards[alg][branch]}
        if card.id not in existing_ids:
            self._cards[alg][branch].append(card)

    def load_default_hnsw_cards(
        self,
        cards_dir: str | Path = _DEFAULT_HNSW_CARDS_DIR,
    ) -> None:
        """Load all HNSW metric pattern cards from a directory of ``.md`` files.

        Each ``.md`` file should contain one card in the format expected
        by ``MetricPatternCard.from_markdown``.
        """
        self.load_cards("HNSW", cards_dir)

    def load_cards(
        self,
        algorithm: str,
        cards_dir: str | Path,
    ) -> None:
        """Load metric pattern cards for *algorithm* from a directory of ``.md`` files.

        Each ``.md`` file should contain one card in the format expected
        by ``MetricPatternCard.from_markdown``.
        """
        base = Path(cards_dir)
        if not base.is_dir():
            return

        for fpath in sorted(base.glob("*.md")):
            try:
                text = fpath.read_text(encoding="utf-8")
                card = MetricPatternCard.from_markdown(text)
                if card is not None:
                    # Override algorithm if the card doesn't have it set
                    if not card.algorithm or card.algorithm == "HNSW":
                        card.algorithm = algorithm
                    self.add_card(card)
            except Exception:
                pass

    @classmethod
    def for_algorithm(
        cls,
        algorithm: str,
        cards_dir: str | Path | None = None,
    ) -> "StaticKnowledgeBase":
        """Create a ``StaticKnowledgeBase`` pre-loaded with cards for *algorithm*.

        Parameters
        ----------
        algorithm : str
            Algorithm identifier (e.g. ``"HNSW"``, ``"UNIFY"``).
        cards_dir : str or Path, optional
            Directory containing ``*.md`` card files.  If None, a default
            directory is inferred from the algorithm name.

        Returns
        -------
        StaticKnowledgeBase
        """
        if cards_dir is None:
            if algorithm.upper() == "UNIFY":
                cards_dir = _DEFAULT_UNIFY_CARDS_DIR
            elif algorithm.upper() == "HNSW":
                cards_dir = _DEFAULT_HNSW_CARDS_DIR
            else:
                cards_dir = f"knowledge_base/{algorithm}"
        kb = cls(cards_dir=None)  # don't auto-load HNSW cards
        kb.load_cards(algorithm, cards_dir)
        return kb

    # ── Queries ────────────────────────────────────────────────────────

    def get_cards_by_branch(
        self,
        algorithm: str,
        recall_constraint_class: str,
    ) -> List[MetricPatternCard]:
        """Return all cards for *algorithm* in the given recall branch."""
        alg_cards = self._cards.get(algorithm, {})
        return list(alg_cards.get(recall_constraint_class, []))

    def get_all_cards(self, algorithm: str | None = None) -> List[MetricPatternCard]:
        """Return all registered cards, optionally filtered by algorithm."""
        result: List[MetricPatternCard] = []
        for alg, branches in self._cards.items():
            if algorithm and alg != algorithm:
                continue
            for cards in branches.values():
                result.extend(cards)
        return result

    @property
    def algorithms(self) -> List[str]:
        return sorted(self._cards.keys())

    def branch_counts(self, algorithm: str) -> Dict[str, int]:
        """Return ``{branch: card_count}`` for *algorithm*."""
        counts: Dict[str, int] = {}
        for branch, cards in self._cards.get(algorithm, {}).items():
            counts[branch] = len(cards)
        return counts


# ---------------------------------------------------------------------------
# StaticKnowledgeSelector
# ---------------------------------------------------------------------------


class StaticKnowledgeSelector:
    """Selects relevant metric pattern cards for the current tuning state.

    Parameters
    ----------
    knowledge_base : StaticKnowledgeBase
        The card registry to select from.
    llm_caller : callable or None
        Optional LLM caller for description-based matching.  When
        provided, ``select_knowledge`` can use it to refine the match.
        Expected signature: ``llm_caller(prompt: str) -> str``.
    """

    def __init__(
        self,
        knowledge_base: StaticKnowledgeBase,
        llm_caller: Any = None,
    ) -> None:
        self._kb = knowledge_base
        self._llm_caller = llm_caller

    # ── State construction ─────────────────────────────────────────────

    def build_tuning_state(
        self,
        task_descriptor: Dict[str, Any],
        observation: Dict[str, Any],
        full_state: Dict[str, Any] | None = None,
        interval_table: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build a normalised tuning state dict from raw inputs.

        Parameters
        ----------
        task_descriptor : dict
            Must contain ``"algorithm"``, ``"recall_threshold"``.
        observation : dict
            Must contain ``"config"``, ``"qps"``, ``"recall"``,
            ``"diagnostic_metrics"``.
        full_state : dict or None
            Optional full state from ``CurrentTaskMemory``.  If provided,
            its ``state_diagnosis`` and ``structured_state`` are used.

        Returns
        -------
        dict
            Normalised tuning state with keys: ``algorithm``,
            ``config``, ``recall``, ``qps``, ``recall_threshold``,
            ``recall_margin``, ``recall_constraint_class``,
            ``diagnostic_metrics``, ``state_diagnosis``.
        """
        algorithm = str(task_descriptor.get("algorithm", "HNSW"))
        threshold = float(
            observation.get("recall_threshold", task_descriptor.get("recall_threshold", 0.95))
        )
        config = dict(observation.get("config") or {})
        qps = float(observation.get("qps", 0))
        recall = float(observation.get("recall", 0))
        margin = recall - threshold
        diag = dict(observation.get("diagnostic_metrics") or {})

        # Pull in extra diagnostic fields from the observation config
        if full_state and isinstance(full_state, dict):
            ss = full_state.get("structured_state") or {}
            ndm = ss.get("normalized_diagnostic_metrics") or {}
            for k, v in ndm.items():
                if k not in diag:
                    diag[k] = v

        branch = self.determine_recall_branch({"recall_margin": margin})

        interval_summary = ""
        if isinstance(interval_table, dict):
            rows = interval_table.get("interval_table") or []
            parts = []
            for row in rows:
                m_val = row.get("M")
                for cell in row.get("efC_cells") or []:
                    iv = cell.get("efS_interval") or {}
                    parts.append(
                        f"(M={m_val}, efC={cell.get('efC')}) "
                        f"[L={iv.get('L')}, U={iv.get('U')}] "
                        f"qps_at_U={cell.get('qps_at_U')}"
                    )
            interval_summary = "; ".join(parts)

        return {
            "algorithm": algorithm,
            "config": config,
            "recall": recall,
            "qps": qps,
            "recall_threshold": threshold,
            "recall_margin": round(margin, 6),
            "recall_constraint_class": branch,
            "diagnostic_metrics": diag,
            "interval_table_summary": interval_summary,
            "state_diagnosis": (
                full_state.get("state_diagnosis", "")
                if isinstance(full_state, dict)
                else ""
            ),
        }

    # ── Branch routing ─────────────────────────────────────────────────

    @staticmethod
    def determine_recall_branch(tuning_state: Dict[str, Any]) -> str:
        """Determine the recall constraint branch from a tuning state.

        Returns ``"recall-infeasible"`` or ``"recall-feasible"``.
        """
        margin = float(tuning_state.get("recall_margin", 0))
        return RECALL_INFEASIBLE if margin < 0 else RECALL_FEASIBLE

    # ── Signal matching (rule-based) ───────────────────────────────────

    def match_signals(
        self,
        card: MetricPatternCard,
        tuning_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check how many of *card*'s signals are present in the tuning state.

        This is a **rule-based fast filter** — no LLM is involved.

        Returns
        -------
        dict
            ``{"signal_support_level": "strong"|"moderate"|"weak",
            "matched_signals": [...], "missed_signals": [...],
            "total_signals": int, "match_count": int}``.
        """
        signals = card.signals
        if not signals:
            return {
                "signal_support_level": "weak",
                "matched_signals": [],
                "missed_signals": [],
                "total_signals": 0,
                "match_count": 0,
            }

        config = tuning_state.get("config") or {}
        diag = tuning_state.get("diagnostic_metrics") or {}
        margin = float(tuning_state.get("recall_margin", 0))
        recall = float(tuning_state.get("recall", 0))
        qps = float(tuning_state.get("qps", 0))

        matched: List[str] = []
        missed: List[str] = []

        for signal in signals:
            s = signal.lower().strip()
            if self._check_signal(s, margin, recall, qps, config, diag):
                matched.append(signal)
            else:
                missed.append(signal)

        total = len(signals)
        match_count = len(matched)
        if total == 0:
            level = "weak"
        elif match_count >= total * 0.75:
            level = "strong"
        elif match_count >= total * 0.4:
            level = "moderate"
        else:
            level = "weak"

        return {
            "signal_support_level": level,
            "matched_signals": matched,
            "missed_signals": missed,
            "total_signals": total,
            "match_count": match_count,
        }

    def _check_signal(
        self,
        signal: str,
        margin: float,
        recall: float,
        qps: float,
        config: Dict[str, Any],
        diag: Dict[str, Any],
    ) -> bool:
        """Check a single signal string against observed values.

        The matching uses keyword-based heuristics over the signal text.
        """
        # ── Recall margin signals ───────────────────────────────────
        if "recall_margin < 0" in signal or "recall margin < 0" in signal:
            return margin < 0
        if ("recall_margin > 0" in signal or "recall margin > 0" in signal
                or "recall margin is positive" in signal):
            if "sufficient slack" in signal or "large" in signal:
                return margin > _LARGE_MARGIN
            if "small" in signal:
                return 0 <= margin <= _SMALL_MARGIN
            return margin >= 0
        if "recall is feasible" in signal:
            return margin >= 0
        if "recall is below" in signal:
            return margin < 0
        if "near the threshold" in signal or "close to 0" in signal:
            return abs(margin) <= _SMALL_MARGIN

        # ── visited_nodes signals ───────────────────────────────────
        visited = _get_metric(diag, "visited_nodes_per_query")
        if visited is not None:
            if "visited_nodes_per_query is low" in signal or ("visited nodes" in signal and "low" in signal and "moderate" not in signal):
                return visited < _LOW_VISITED_NODES
            if "visited_nodes_per_query is high" in signal or ("visited nodes" in signal and "high" in signal):
                return visited > _LOW_VISITED_NODES * 3  # > 300
            if "visited_nodes" in signal and ("moderate" in signal or "low or moderate" in signal):
                return visited <= _LOW_VISITED_NODES * 2  # <= 200

        # ── distance_computations signals ───────────────────────────
        dist = _get_metric(diag, "distance_computations", "dist_comps_per_query", "dist_comps_count")
        if dist is not None:
            if "distance_computations is low" in signal or "dist_comps_count is low" in signal:
                return dist < 200
            if "distance_computations is high" in signal or "dist_comps_count is high" in signal:
                return dist > _HIGH_DIST_COMPS
            if "distance_computations" in signal and "moderate" in signal:
                return 200 <= dist <= _HIGH_DIST_COMPS

        # ── ef signals ──────────────────────────────────────────────
        ef_val = _get_config_value(config, "ef")
        efC_val = _get_config_value(config, "ef_construction", "efC", "efConstruction")
        if ef_val is not None:
            if "ef is high" in signal:
                if efC_val and efC_val > 0:
                    return ef_val > efC_val * _HIGH_EF_RATIO
                return ef_val > 100
            if "ef is low" in signal or "ef is not close to its upper bound" in signal:
                if efC_val and efC_val > 0:
                    return ef_val <= efC_val * _HIGH_EF_RATIO
                return ef_val <= 100
            if "ef is moderate" in signal:
                return 40 <= ef_val <= 120
            if "ef is low or moderate" in signal:
                if efC_val and efC_val > 0:
                    return ef_val <= efC_val * _HIGH_EF_RATIO
                return ef_val <= 120
            if "requires high ef" in signal:
                if efC_val and efC_val > 0:
                    return ef_val > efC_val * 0.7
                return ef_val > 80

        # ── efC signals ─────────────────────────────────────────────
        efC_val = _get_config_value(config, "ef_construction", "efC")
        if efC_val is not None:
            if "efc is low" in signal or "efc is low or moderate" in signal:
                return efC_val <= 300
            if "efc is high" in signal:
                return efC_val > 500

        # ── M signals ───────────────────────────────────────────────
        m_val = _get_config_value(config, "M")
        if m_val is not None:
            if "m is high" in signal:
                return m_val > 32
            if "m is low" in signal:
                return m_val < 16

        # ── Out-degree signals ──────────────────────────────────────
        out_deg = _get_metric(diag, "avg_out_degree", "out_degree_mean")
        if out_deg is not None:
            if "avg_out_degree" in signal and "low" in signal:
                if m_val and m_val > 0:
                    return out_deg < m_val * _LOW_OUT_DEGREE_RATIO
                return out_deg < 8
            if "avg_out_degree" in signal and "high" in signal:
                if m_val and m_val > 0:
                    return out_deg > m_val * 0.7
                return out_deg > 16
            if "out_degree_mean" in signal and "low" in signal:
                if m_val and m_val > 0:
                    return out_deg < m_val * _LOW_OUT_DEGREE_RATIO
                return out_deg < 8

        # ── In-degree signals ───────────────────────────────────────
        in_deg = _get_metric(diag, "avg_in_degree", "in_degree_mean")
        if in_deg is not None:
            if "avg_in_degree" in signal and "low" in signal:
                return in_deg < 4

        # ── QPS signals ─────────────────────────────────────────────
        if "qps is low" in signal or "qps is lower than expected" in signal:
            return qps < 5000
        if "qps is not optimal" in signal:
            return True  # always true in an optimisation context

        # ── UNIFY: B (num_slots) signals ────────────────────────────
        b_val = _get_config_value(config, "B", "num_slots")
        if b_val is not None:
            if "b is high" in signal or "b is large" in signal or "num_slots is high" in signal:
                return b_val > 8
            if "b is low" in signal or "b is small" in signal or "num_slots is low" in signal:
                return b_val < 4

        # ── UNIFY: al signals ───────────────────────────────────────
        al_val = _get_config_value(config, "al")
        if al_val is not None:
            if "al is high" in signal or "al is large" in signal:
                return al_val > 64
            if "al is low" in signal or "al is small" in signal:
                return al_val < 16
            if "al is moderate" in signal:
                return 16 <= al_val <= 64

        # ── UNIFY: efConstruction signals ───────────────────────────
        efC_u_val = _get_config_value(config, "efConstruction", "ef_construction")
        if efC_u_val is not None:
            if "efconstruction is low" in signal or "efc is low" in signal:
                return efC_u_val <= 300
            if "efconstruction is high" in signal or "efc is high" in signal:
                return efC_u_val > 500

        # ── UNIFY: inclusiveness signals ────────────────────────────
        incl_val = _get_metric(diag, "inclusiveness_pct")
        if incl_val is not None:
            if "inclusiveness is low" in signal or "inclusiveness_pct is low" in signal:
                return incl_val < _LOW_INCLUSIVENESS_PCT
            if "inclusiveness is high" in signal:
                return incl_val >= 80.0

        # ── UNIFY: build_time signals ───────────────────────────────
        bt_val = _get_metric(diag, "build_time_s")
        if bt_val is not None:
            if "build_time is high" in signal or "build time is high" in signal:
                return bt_val > _HIGH_BUILD_TIME_S
            if "build_time is low" in signal or "build time is low" in signal:
                return bt_val < 60.0

        # ── Index size signals ──────────────────────────────────────
        idx_size = _get_metric(diag, "index_size", "index_size_mb")
        if idx_size is not None:
            if "index_size is high" in signal:
                return idx_size > 1.0  # GB

        # ── Edge-case / contextual signals ──────────────────────────
        if "recent ef increase gives weak" in signal:
            # We can't observe this without history; soft-match
            return margin < 0
        if "increasing ef alone gives limited improvement" in signal:
            return margin < 0  # Soft match — requires transition history
        if "ef reduction is being considered" in signal:
            return margin >= 0  # Soft match — plausible in feasible state

        # ── Graph quality signals ───────────────────────────────────
        if "graph-quality indicators are not strong" in signal:
            if m_val and m_val < 24:
                return True
            if efC_val and efC_val < 300:
                return True
            return False

        # ── Recall variance / noise signals ─────────────────────────
        if "recall variance" in signal or "measurement noise" in signal:
            return abs(margin) <= _SMALL_MARGIN  # Soft match

        # ── Default: unknown signal → skip (don't count as missed) ──
        return True  # lenient: unknown signal patterns don't block

    # ── Boundary checking (rule-based) ─────────────────────────────────

    def check_boundary(
        self,
        card: MetricPatternCard,
        tuning_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check whether the card's applicability boundary is violated.

        Returns
        -------
        dict
            ``{"boundary_valid": bool, "violations": [str, ...]}``.
        """
        violations: List[str] = []
        config = tuning_state.get("config") or {}
        margin = float(tuning_state.get("recall_margin", 0))
        diag = tuning_state.get("diagnostic_metrics") or {}

        boundary = card.boundary.lower()
        description = card.description.lower()

        # ── ef upper-bound checks ──────────────────────────────────
        ef_val = _get_config_value(config, "ef")
        if ef_val is not None:
            if "ef is not close to its upper bound" in description:
                # If ef IS at or near upper bound → violation
                efC_val = _get_config_value(config, "ef_construction", "efC")
                if efC_val and efC_val > 0 and ef_val >= efC_val:
                    violations.append(
                        f"ef ({ef_val}) is at ef_construction ({efC_val}) — "
                        "card assumes ef is not at upper bound"
                    )

            if "ef is already high" in boundary or "ef already high" in description:
                if ef_val < 100:
                    violations.append(
                        f"ef ({ef_val}) is not high — "
                        "card's boundary expects ef already high"
                    )

        # ── Recall margin mismatch ──────────────────────────────────
        if ("far above" in description and margin < _LARGE_MARGIN):
            violations.append(
                f"recall_margin ({margin:+.4f}) is not 'far above' — "
                "card expects large positive slack"
            )

        if ("recall is below" in description or "recall below" in description) and margin >= 0:
            violations.append(
                f"recall_margin ({margin:+.4f}) is non-negative — "
                "card expects recall below threshold"
            )

        # ── M boundary ──────────────────────────────────────────────
        m_val = _get_config_value(config, "M")
        if m_val is not None:
            if "m is not at lower bound" in boundary and m_val <= 4:
                violations.append(f"M ({m_val}) is at lower bound")
            if "m not at upper bound" in boundary and m_val >= 128:
                violations.append(f"M ({m_val}) is at upper bound")

        # ── efC boundary ────────────────────────────────────────────
        efC_val = _get_config_value(config, "ef_construction", "efC")
        if efC_val is not None:
            if "efc not at upper bound" in boundary and efC_val >= 800:
                violations.append(f"efC ({efC_val}) is at upper bound")

        # ── UNIFY: B boundary ───────────────────────────────────────
        b_val = _get_config_value(config, "B", "num_slots")
        if b_val is not None:
            if "b not at lower bound" in boundary and b_val <= 2:
                violations.append(f"B ({b_val}) is at lower bound")
            if "b not at upper bound" in boundary and b_val >= 16:
                violations.append(f"B ({b_val}) is at upper bound")

        # ── UNIFY: al boundary ──────────────────────────────────────
        al_val = _get_config_value(config, "al")
        if al_val is not None:
            if "al not at lower bound" in boundary and al_val <= 4:
                violations.append(f"al ({al_val}) is at lower bound")
            if "al not at upper bound" in boundary and al_val >= 128:
                violations.append(f"al ({al_val}) is at upper bound")

        # ── UNIFY: efConstruction boundary ──────────────────────────
        efC_u_val = _get_config_value(config, "efConstruction", "ef_construction")
        if efC_u_val is not None:
            if "efconstruction not at upper bound" in boundary and efC_u_val >= 800:
                violations.append(f"efConstruction ({efC_u_val}) is at upper bound")
            if "efconstruction not at lower bound" in boundary and efC_u_val <= 100:
                violations.append(f"efConstruction ({efC_u_val}) is at lower bound")

        # ── UNIFY: inclusiveness boundary ───────────────────────────
        incl_val = _get_metric(diag, "inclusiveness_pct")
        if incl_val is not None:
            if "inclusiveness above" in boundary and incl_val < 70.0:
                violations.append(
                    f"inclusiveness ({incl_val:.1f}%) is below card's assumed threshold"
                )

        return {
            "boundary_valid": len(violations) == 0,
            "violations": violations,
        }

    # ── LLM-based description matching (optional) ──────────────────────

    def build_description_match_prompt(
        self,
        card: MetricPatternCard,
        tuning_state: Dict[str, Any],
    ) -> str:
        """Build a prompt for LLM-based description matching.

        The LLM is asked whether the current tuning state matches the
        card's description — it does NOT generate a proposal.
        """
        return (
            "You are matching the current vector search tuning state to "
            "one metric pattern card.\n\n"
            "Current tuning state:\n"
            f"- Algorithm: {tuning_state.get('algorithm', 'HNSW')}\n"
            f"- Config: {json.dumps(tuning_state.get('config', {}))}\n"
            f"- Recall: {tuning_state.get('recall', 0):.4f}\n"
            f"- Recall threshold: {tuning_state.get('recall_threshold', 0):.4f}\n"
            f"- Recall margin: {tuning_state.get('recall_margin', 0):+.4f}\n"
            f"- QPS: {tuning_state.get('qps', 0):.1f}\n"
            f"- Diagnostic metrics: {json.dumps(tuning_state.get('diagnostic_metrics', {}))}\n"
            f"- State diagnosis: {tuning_state.get('state_diagnosis', '')}\n"
            + (
                f"- Interval table (current task, M->efC->(L,U] with qps_at_U): "
                f"{tuning_state.get('interval_table_summary', '')}\n"
                if tuning_state.get("interval_table_summary") else ""
            )
            + f"\n"
            f"Metric pattern card:\n"
            f"- Pattern ID: {card.id}\n"
            f"- Recall constraint class: {card.recall_constraint_class}\n"
            f"- Description: {card.description}\n"
            f"- Signals: {json.dumps(card.signals)}\n"
            f"- Interpretation: {card.interpretation}\n"
            f"- Boundary: {card.boundary}\n"
            f"\n"
            f"Question:\n"
            f"Does the current tuning state match this metric pattern card?\n"
            f"\n"
            f'Please return JSON:\n'
            f'{{\n'
            f'  "match": true/false,\n'
            f'  "match_strength": "strong" | "moderate" | "weak" | "none",\n'
            f'  "matched_reasons": [...],\n'
            f'  "boundary_violations": [...],\n'
            f'  "rationale": "..."\n'
            f'}}\n'
        )

    def _llm_match_cards(
        self,
        pre_filtered: List[Dict[str, Any]],
        tuning_state: Dict[str, Any],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Use LLM to judge whether each card's description matches the current state.

        Sends one prompt per card; parses the JSON response to get match/mismatch
        and match_strength.  Falls back to rule-based scoring on parse failure.
        """
        scored: List[Dict[str, Any]] = []
        for item in pre_filtered:
            card = item["card"]
            prompt = self.build_description_match_prompt(card, tuning_state)
            try:
                raw = self._llm_caller(prompt)
                parsed = self._parse_llm_match_result(raw)
            except Exception:
                parsed = None

            if parsed and parsed.get("match"):
                strength = parsed.get("match_strength", "moderate")
                score = {"strong": 3.0, "moderate": 2.0, "weak": 0.5}.get(strength, 2.0)
                item["signal_result"]["llm_match_strength"] = strength
                item["signal_result"]["llm_rationale"] = parsed.get("rationale", "")
                item["signal_result"]["llm_matched_reasons"] = parsed.get("matched_reasons", [])
            elif parsed is not None and not parsed.get("match"):
                # LLM explicitly rejected this card
                continue
            else:
                # LLM call failed — fall back to rule-based score
                score = _score_card(item["signal_result"], card)

            scored.append({**item, "score": score})

        return scored

    @staticmethod
    def _parse_llm_match_result(raw: str) -> Dict[str, Any] | None:
        """Parse LLM's JSON response from description matching."""
        import json as _json

        if not raw:
            return None
        raw = raw.strip()
        # Try direct parse
        try:
            return _json.loads(raw)
        except Exception:
            pass
        # Try to find JSON block
        import re as _re
        m = _re.search(r"\{[^{}]*\"match\"[^{}]*\}", raw, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except Exception:
                pass
        return None

    # ── Main selection logic ───────────────────────────────────────────

    def select_knowledge(
        self,
        task_descriptor: Dict[str, Any],
        observation: Dict[str, Any],
        full_state: Dict[str, Any] | None = None,
        k: int = 3,
        interval_table: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Select at most *k* relevant metric pattern cards.

        Uses **LLM-based description matching** when ``self._llm_caller``
        is available; falls back to rule-based signal matching otherwise.

        Parameters
        ----------
        task_descriptor : dict
            Task metadata (algorithm, recall_threshold, etc.).
        observation : dict
            Current observation (config, qps, recall, diagnostic_metrics).
        full_state : dict or None
            Optional full state from ``CurrentTaskMemory``.
        k : int
            Maximum number of cards to return.

        Returns
        -------
        dict
            ``{"recall_constraint_class": str,
            "selected_cards": [...],
            "matched_interpretations": [...],
            "recommended_directions": [...],
            "costs_and_risks": [...],
            "applicability_boundaries": [...],
            "selection_rationale": str}``
        """
        # Step 1-2: Build state and determine branch
        tuning_state = self.build_tuning_state(
            task_descriptor, observation, full_state, interval_table
        )
        branch = tuning_state["recall_constraint_class"]
        algorithm = tuning_state["algorithm"]

        # Step 3: Get candidate cards from this branch only
        candidates = self._kb.get_cards_by_branch(algorithm, branch)
        if not candidates:
            other = RECALL_FEASIBLE if branch == RECALL_INFEASIBLE else RECALL_INFEASIBLE
            candidates = self._kb.get_cards_by_branch(algorithm, other)
            if candidates:
                branch = other

        # Step 4: Rule-based pre-filter (fast) — signal match + boundary check
        pre_filtered: List[Dict[str, Any]] = []
        for card in candidates:
            signal_result = self.match_signals(card, tuning_state)
            boundary_result = self.check_boundary(card, tuning_state)
            if not boundary_result["boundary_valid"]:
                continue
            level = signal_result.get("signal_support_level", "weak")
            if level in ("strong", "moderate"):
                pre_filtered.append({
                    "card": card,
                    "signal_result": signal_result,
                    "boundary_result": boundary_result,
                })

        # Step 5: LLM-based description matching (when caller available)
        if self._llm_caller is not None and pre_filtered:
            scored = self._llm_match_cards(pre_filtered, tuning_state, k)
        else:
            scored = [
                {**item, "score": _score_card(item["signal_result"], item["card"])}
                for item in pre_filtered
            ]

        # Step 6: Sort by score, select diverse top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        selected = _choose_representative_cards(scored, k)

        # Step 7: Build return context
        return {
            "recall_constraint_class": branch,
            "selected_cards": [
                {
                    "card": item["card"].to_dict(),
                    "signal_result": item["signal_result"],
                    "boundary_result": item["boundary_result"],
                }
                for item in selected
            ],
            "matched_interpretations": [
                item["card"].interpretation for item in selected
            ],
            "recommended_directions": [
                item["card"].solution for item in selected
            ],
            "costs_and_risks": [
                item["card"].cost for item in selected
            ],
            "applicability_boundaries": [
                item["card"].boundary for item in selected
            ],
            "selection_rationale": _build_selection_rationale(selected, tuning_state),
        }

    # ── Best-match selection (single closest card, one LLM call) ──────

    def build_best_match_prompt(
        self,
        pre_filtered: List[Dict[str, Any]],
        tuning_state: Dict[str, Any],
    ) -> str:
        """Build the comparative prompt: one LLM call over all candidate cards.

        Unlike :meth:`build_description_match_prompt`, the LLM sees every
        boundary-valid card — including Solution and Cost — plus the full
        last-round metric block, and must pick the single closest card.
        """
        lines: List[str] = [
            "You are choosing the single most relevant metric pattern card for "
            "the current HNSW tuning state.\n",
            "Current tuning state (all metrics collected from the last round):",
            f"- Algorithm: {tuning_state.get('algorithm', 'HNSW')}",
            f"- Config: {json.dumps(tuning_state.get('config', {}))}",
            f"- Recall: {tuning_state.get('recall', 0):.4f}",
            f"- Recall threshold: {tuning_state.get('recall_threshold', 0):.4f}",
            f"- Recall margin: {tuning_state.get('recall_margin', 0):+.4f}",
            f"- QPS: {tuning_state.get('qps', 0):.1f}",
            f"- Diagnostic metrics: {json.dumps(tuning_state.get('diagnostic_metrics', {}))}",
            f"- State diagnosis: {tuning_state.get('state_diagnosis', '')}",
        ]
        if tuning_state.get("interval_table_summary"):
            lines.append(
                f"- Interval table (current task, M->efC->(L,U] with qps_at_U): "
                f"{tuning_state.get('interval_table_summary')}"
            )
        lines.append("\nCandidate metric pattern cards:")
        for idx, item in enumerate(pre_filtered, start=1):
            card = item["card"]
            lines.append(
                f"\nCard {idx}: {card.id} ({card.recall_constraint_class})\n"
                f"- Description: {card.description}\n"
                f"- Signals: {json.dumps(card.signals)}\n"
                f"- Interpretation: {card.interpretation}\n"
                f"- Solution: {card.solution}\n"
                f"- Cost: {card.cost}\n"
                f"- Boundary: {card.boundary}"
            )
        lines.append(
            "\nQuestion:\n"
            "Which single card is CLOSEST to the current tuning state? "
            "Rank all cards from most to least relevant.\n"
            "\nPlease return strict JSON:\n"
            "{\n"
            '  "best_card_id": "<Pattern ID of the single closest card>",\n'
            '  "ranked_card_ids": ["<most relevant>", "...", "<least relevant>"],\n'
            '  "rationale": "..."\n'
            "}"
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_best_card_result(raw: str, valid_ids: List[str]) -> Dict[str, Any] | None:
        """Parse the LLM's best-card JSON and validate ids against the presented set."""
        import json as _json
        import re as _re

        if not raw:
            return None
        raw = raw.strip()
        parsed: Any = None
        try:
            parsed = _json.loads(raw)
        except Exception:
            m = _re.search(r"\{[^{}]*\"best_card_id\"[^{}]*\}", raw, _re.DOTALL)
            if m:
                try:
                    parsed = _json.loads(m.group())
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            return None
        best = parsed.get("best_card_id")
        if not isinstance(best, str) or best not in valid_ids:
            return None
        ranked = parsed.get("ranked_card_ids")
        if not isinstance(ranked, list):
            ranked = [best]
        ranked = [str(x) for x in ranked if isinstance(x, str) and x in valid_ids]
        if best not in ranked:
            ranked.insert(0, best)
        # Append any presented cards the LLM omitted, preserving input order.
        for vid in valid_ids:
            if vid not in ranked:
                ranked.append(vid)
        return {
            "best_card_id": best,
            "ranked_card_ids": ranked,
            "rationale": str(parsed.get("rationale", "")),
        }

    def _llm_rank_cards(
        self,
        pre_filtered: List[Dict[str, Any]],
        tuning_state: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """One comparative LLM call ranking all boundary-valid cards.

        Returns ``(reordered_items, llm_ok)``. On any failure the input order
        is preserved (``llm_ok=False``) so the caller falls back to rule
        scoring.
        """
        valid_ids = [item["card"].id for item in pre_filtered]
        prompt = self.build_best_match_prompt(pre_filtered, tuning_state)
        try:
            raw = self._llm_caller(prompt)
            parsed = self._parse_best_card_result(raw, valid_ids)
        except Exception:
            parsed = None
        if parsed is None:
            return list(pre_filtered), False
        rank = {cid: idx for idx, cid in enumerate(parsed["ranked_card_ids"])}
        reordered = sorted(pre_filtered, key=lambda item: rank.get(item["card"].id, len(rank)))
        for item in reordered:
            item["signal_result"]["llm_rank"] = rank.get(item["card"].id, -1)
            item["signal_result"]["llm_best_card"] = parsed["best_card_id"]
            item["signal_result"]["llm_rationale"] = parsed.get("rationale", "")
        return reordered, True

    def select_best_match(
        self,
        task_descriptor: Dict[str, Any],
        observation: Dict[str, Any],
        full_state: Dict[str, Any] | None = None,
        interval_table: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Select the single closest knowledge card for the current state.

        Same input contract and return shape as :meth:`select_knowledge`, but
        ``selected_cards`` holds exactly one card: the LLM's best match from a
        single comparative LLM call over all boundary-valid branch cards.
        Rule-based scoring is used only as fallback ordering when the LLM
        call fails or is unavailable.
        """
        tuning_state = self.build_tuning_state(
            task_descriptor, observation, full_state, interval_table
        )
        branch = tuning_state["recall_constraint_class"]
        algorithm = tuning_state["algorithm"]

        candidates = self._kb.get_cards_by_branch(algorithm, branch)
        if not candidates:
            other = RECALL_FEASIBLE if branch == RECALL_INFEASIBLE else RECALL_INFEASIBLE
            candidates = self._kb.get_cards_by_branch(algorithm, other)
            if candidates:
                branch = other

        # Boundary check only — weak signal support does NOT drop a card here;
        # the LLM sees every boundary-valid card and decides.
        pre_filtered: List[Dict[str, Any]] = []
        for card in candidates:
            signal_result = self.match_signals(card, tuning_state)
            boundary_result = self.check_boundary(card, tuning_state)
            if not boundary_result["boundary_valid"]:
                continue
            pre_filtered.append({
                "card": card,
                "signal_result": signal_result,
                "boundary_result": boundary_result,
            })

        if not pre_filtered:
            return {
                "recall_constraint_class": branch,
                "selected_cards": [],
                "matched_interpretations": [],
                "recommended_directions": [],
                "costs_and_risks": [],
                "applicability_boundaries": [],
                "selection_rationale": "No boundary-valid knowledge cards for this state.",
            }

        llm_ok = False
        if self._llm_caller is not None:
            pre_filtered, llm_ok = self._llm_rank_cards(pre_filtered, tuning_state)
        if not llm_ok:
            pre_filtered.sort(
                key=lambda item: _score_card(item["signal_result"], item["card"]),
                reverse=True,
            )

        best = pre_filtered[0]
        return {
            "recall_constraint_class": branch,
            "selected_cards": [
                {
                    "card": best["card"].to_dict(),
                    "signal_result": best["signal_result"],
                    "boundary_result": best["boundary_result"],
                }
            ],
            "matched_interpretations": [best["card"].interpretation],
            "recommended_directions": [best["card"].solution],
            "costs_and_risks": [best["card"].cost],
            "applicability_boundaries": [best["card"].boundary],
            "selection_rationale": (
                "Best match via single LLM comparison over all boundary-valid cards."
                if llm_ok
                else "Best match via rule-based scoring (LLM unavailable or failed)."
            ),
        }

    # ── Formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_knowledge_context_for_llm(
        selected_context: Dict[str, Any],
    ) -> str:
        """Render selected knowledge into LLM-readable markdown.

        Parameters
        ----------
        selected_context : dict
            The output of ``select_knowledge``.

        Returns
        -------
        str
            Formatted markdown ready for injection into an LLM prompt.
        """
        if not selected_context:
            return ""

        branch = selected_context.get("recall_constraint_class", "unknown")
        cards = selected_context.get("selected_cards") or []

        if not cards:
            return (
                "## Selected Static Knowledge\n\n"
                f"**Recall constraint branch**: {branch}\n\n"
                "_No metric pattern cards matched the current tuning state._\n\n"
            )

        lines: List[str] = []
        lines.append("## Selected Static Knowledge")
        lines.append("")
        lines.append(f"**Recall constraint branch**: `{branch}`")
        lines.append("")

        for i, item in enumerate(cards, start=1):
            card = item.get("card") or item
            signal_result = item.get("signal_result") or {}
            boundary_result = item.get("boundary_result") or {}

            pid = card.get("id", "?")
            desc = card.get("description", "")
            interp = card.get("interpretation", "")
            solution = card.get("solution", "")
            cost = card.get("cost", "")
            boundary = card.get("boundary", "")

            # Why matched
            matched = signal_result.get("matched_signals") or []
            why = ", ".join(matched[:3]) if matched else "description matched"

            lines.append(f"### Matched Pattern Card {i}: {pid}")
            lines.append("")
            lines.append(f"**Why matched**: {why}")
            lines.append("")
            if desc:
                lines.append(f"**Description**: {desc}")
                lines.append("")
            if interp:
                lines.append(f"**Interpretation**: {interp}")
                lines.append("")
            if solution:
                lines.append(f"**Relevant direction**: {solution}")
                lines.append("")
            if cost:
                lines.append(f"**Cost**: {cost}")
                lines.append("")
            if boundary:
                lines.append(f"**Boundary / risk**: {boundary}")
                lines.append("")
            if boundary_result.get("violations"):
                viols = boundary_result["violations"]
                lines.append(f"**Boundary violations**: {', '.join(viols)}")
                lines.append("")

        # Usage guidance
        lines.append("---")
        lines.append("")
        lines.append("**How to use this knowledge**:")
        lines.append("- Treat these cards as **mechanism-level direction priors**.")
        lines.append("- Do **not** treat them as current-task empirical evidence.")
        lines.append("- Do **not** treat them as final actions.")
        lines.append("- Combine them with **Current-task Memory**")
        lines.append("  before choosing the next action.")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_metric(diag: Dict[str, Any], *keys: str) -> Optional[float]:
    """Get the first available metric from *diag*."""
    for k in keys:
        v = diag.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _get_config_value(config: Dict[str, Any], *keys: str) -> Optional[float]:
    """Get the first available config value."""
    for k in keys:
        v = config.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _score_card(
    signal_result: Dict[str, Any],
    card: MetricPatternCard,
) -> float:
    """Score a matched card for ranking.

    Higher = better.  Factors: signal support level, match ratio,
    card confidence, and a small bonus for having cost/boundary info.
    """
    level = signal_result.get("signal_support_level", "weak")
    match_count = int(signal_result.get("match_count", 0))
    total = max(1, int(signal_result.get("total_signals", 1)))

    base = {"strong": 3.0, "moderate": 2.0, "weak": 0.5}.get(level, 0.5)
    ratio = match_count / total
    conf = float(card.confidence)

    score = base * (0.5 + 0.5 * ratio) * conf
    if card.cost:
        score += 0.1
    if card.boundary:
        score += 0.1
    return score


def _choose_representative_cards(
    scored: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Select up to *k* cards, preferring diverse mechanism perspectives.

    Already sorted by score descending.  We take the top-scoring card
    and then select remaining cards that cover different solution
    directions.
    """
    if len(scored) <= k:
        return scored

    selected: List[Dict[str, Any]] = [scored[0]]
    seen_directions: set = {
        _direction_key(scored[0]["card"].solution)
    }

    for item in scored[1:]:
        if len(selected) >= k:
            break
        dk = _direction_key(item["card"].solution)
        if dk not in seen_directions:
            selected.append(item)
            seen_directions.add(dk)

    # If we still have room, fill from remaining top-scored
    if len(selected) < k:
        for item in scored[1:]:
            if len(selected) >= k:
                break
            if item not in selected:
                selected.append(item)

    return selected


def _direction_key(solution: str) -> str:
    """Extract a short direction key from a solution string."""
    s = solution.lower()
    # ef / efConstruction
    if "increase efconstruction" in s or "increase ef_construction" in s:
        return "efc_up"
    if "decrease efconstruction" in s or "decrease ef_construction" in s or "reduce efconstruction" in s:
        return "efc_down"
    if "increase ef" in s or "ef increase" in s or "larger ef" in s:
        return "ef_up"
    if "decrease ef" in s or "reduce ef" in s or "ef reduction" in s:
        return "ef_down"
    # M
    if "increase m" in s or "larger m" in s:
        return "m_up"
    if "decrease m" in s or "reduce m" in s or "smaller m" in s:
        return "m_down"
    # B (UNIFY)
    if "increase b" in s or "larger b" in s or "increase num_slots" in s:
        return "b_up"
    if "decrease b" in s or "reduce b" in s or "smaller b" in s:
        return "b_down"
    # al (UNIFY)
    if "increase al" in s or "larger al" in s:
        return "al_up"
    if "decrease al" in s or "reduce al" in s or "smaller al" in s:
        return "al_down"
    # efC (legacy HNSW)
    if "increase efc" in s:
        return "efc_up"
    if "decrease efc" in s or "reduce efc" in s:
        return "efc_down"
    return "other"


def _build_selection_rationale(
    selected: List[Dict[str, Any]],
    tuning_state: Dict[str, Any],
) -> str:
    """Build a concise rationale for why these cards were selected."""
    if not selected:
        return (
            f"No metric pattern cards matched the current tuning state "
            f"(margin={tuning_state.get('recall_margin', 0):+.4f})."
        )

    card_ids = [item["card"].id for item in selected]
    margin = float(tuning_state.get("recall_margin", 0))
    branch = tuning_state.get("recall_constraint_class", "unknown")

    return (
        f"Selected {len(selected)} card(s) for branch '{branch}' "
        f"(recall_margin={margin:+.4f}): {', '.join(card_ids)}. "
        f"Cards were chosen based on signal support level and "
        f"diverse mechanism perspectives."
    )

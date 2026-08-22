#!/usr/bin/env python3
"""Load and format knowledge_base/*.md documents for LLM consumption.

Replaces the subgroup-mining-based knowledge pipeline with direct
loading of curated domain knowledge documents.

Usage:
    from utils.knowledge_loader import load_knowledge_base, format_knowledge_for_llm

    kb = load_knowledge_base("knowledge_base")
    text = format_knowledge_for_llm(kb, max_chars=40000)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


# File priority: higher = more important, loaded first and truncated last.
# 07 (empirical) is the primary operational reference.
# 01 (semantics) and 02 (dependencies) provide foundational understanding.
# 04 (strategies) provides action frameworks.
# 03 (dataset features) and 05 (filtered) are contextual/background.
_DEFAULT_PRIORITY: List[str] = [
    "07_empirical_quantitative_insights.md",
    "08_diskann_empirical_insights.md",
    "09_diskann_parameter_dependencies.md",
    "01_parameter_semantics.md",
    "02_parameter_dependencies.md",
    "04_tuning_strategies.md",
    "03_dataset_features.md",
    "05_filtered_anns.md",
]

# UNIFY-specific priority: unify docs (semantics, dependencies, strategies) first,
# followed by general empirical insights for cross-algorithm reference.
_UNIFY_PRIORITY: List[str] = [
    "unify_01_parameter_semantics.md",
    "unify_02_parameter_dependencies.md",
    "unify_03_tuning_strategies.md",
    "07_empirical_quantitative_insights.md",
    "01_parameter_semantics.md",
    "02_parameter_dependencies.md",
    "04_tuning_strategies.md",
]

# Algorithm → priority list mapping
_ALGORITHM_PRIORITIES: Dict[str, List[str]] = {
    "unify": _UNIFY_PRIORITY,
    "hnswlib": _DEFAULT_PRIORITY,
    "hnsw": _DEFAULT_PRIORITY,
}

# Files to exclude from loading
_EXCLUDE: set = {"README.md"}


def load_knowledge_base(
    base_dir: str | Path = "knowledge_base",
    priority: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Load all .md knowledge documents from the knowledge base directory.

    Args:
        base_dir: Path to the knowledge_base directory.
        priority: Ordered list of filenames (highest priority first).
                  If None, uses _DEFAULT_PRIORITY.

    Returns:
        Dict mapping filename (without .md) to markdown content.
        Files are ordered by priority.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {base_dir}")

    if priority is None:
        priority = _DEFAULT_PRIORITY

    result: Dict[str, str] = {}

    # Load files in priority order
    for fname in priority:
        fpath = base / fname
        if fpath.exists() and fpath.suffix == ".md":
            key = fpath.stem  # filename without .md
            result[key] = fpath.read_text(encoding="utf-8")

    # Load any remaining .md files not in the priority list
    for fpath in sorted(base.glob("*.md")):
        if fpath.name in _EXCLUDE:
            continue
        key = fpath.stem
        if key not in result:
            result[key] = fpath.read_text(encoding="utf-8")

    return result


def format_knowledge_for_llm(
    knowledge: Dict[str, str],
    max_chars: int = 40000,
    priority_keys: Optional[List[str]] = None,
) -> str:
    """Format knowledge documents into a single LLM-readable text block.

    Documents are concatenated in priority order. If the total exceeds
    max_chars, lower-priority documents are truncated from the end.

    Args:
        knowledge: Dict from load_knowledge_base().
        max_chars: Maximum total characters in the output.
        priority_keys: Ordered list of keys (highest priority first).
                       If None, uses insertion order of `knowledge`.

    Returns:
        A single string with all knowledge documents formatted for LLM.
    """
    if priority_keys is None:
        priority_keys = list(knowledge.keys())

    sections: List[str] = []
    total = 0

    for key in priority_keys:
        if key not in knowledge:
            continue
        content = knowledge[key]
        header = f"## {_doc_title(key)}\n\n"
        section = header + content
        section_len = len(section)

        if total + section_len <= max_chars:
            sections.append(section)
            total += section_len
        else:
            # Truncate this section to fit remaining space
            remaining = max_chars - total
            if remaining > len(header) + 200:
                # Worth including a truncated version
                truncated = header + content[: remaining - len(header) - 50] + "\n\n[... truncated ...]"
                sections.append(truncated)
            break

    return "\n\n---\n\n".join(sections)


def _doc_title(key: str) -> str:
    """Extract a human-readable title from a knowledge doc filename key."""
    # Strip leading number prefix like "01_"
    parts = key.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        return parts[1].replace("_", " ").title()
    return key.replace("_", " ").title()


def build_knowledge_context(
    base_dir: str | Path = "knowledge_base",
    max_chars: int = 40000,
    algorithm: str = "hnswlib",
) -> Dict[str, object]:
    """High-level entry point: load knowledge and build context dict.

    Returns a dict suitable for insertion into the LLM prompt context as
    the `base_knowledge_full` or equivalent field.

    Args:
        base_dir: Path to knowledge_base directory.
        max_chars: Max characters for the formatted knowledge text.
        algorithm: Algorithm identifier (e.g. ``"hnswlib"``, ``"unify"``).
                   Determines the priority ordering of knowledge documents.

    Returns:
        Dict with keys: mode, base_knowledge_full, doc_keys_loaded.
    """
    algo_key = str(algorithm).lower()
    priority = _ALGORITHM_PRIORITIES.get(algo_key, _DEFAULT_PRIORITY)
    kb = load_knowledge_base(base_dir, priority=priority)
    formatted = format_knowledge_for_llm(kb, max_chars=max_chars)

    return {
        "mode": "knowledge_base_driven",
        "base_knowledge_full": formatted,
        "doc_keys_loaded": list(kb.keys()),
        "total_chars": len(formatted),
        "algorithm": algo_key,
    }

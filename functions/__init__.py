"""Public tuning pipeline entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Callable, Mapping, Tuple


@dataclass(frozen=True)
class TuningEntrypoint:
    name: str
    module: str
    default_config: str
    description: str


_PIPELINES: Mapping[str, TuningEntrypoint] = MappingProxyType(
    {
        "hnswlib": TuningEntrypoint(
            name="hnswlib",
            module="functions.hnswlib_tune",
            default_config="configs/hnswlib_tune.yaml",
            description="Run native hnswlib recall/QPS tuning.",
        ),
        "diskann-filter": TuningEntrypoint(
            name="diskann-filter",
            module="functions.diskann_filter_tune",
            default_config="configs/diskann_filter_tune.yaml",
            description="Run filtered DiskANN / AF-ANNS tuning.",
        ),
        "unify": TuningEntrypoint(
            name="unify",
            module="functions.unify_tune",
            default_config="configs/unify_tune.yaml",
            description="Run UNIFY/HSIG range-filtered ANN tuning with agentic diagnosis.",
        ),
        "filter-diskann": TuningEntrypoint(
            name="filter-diskann",
            module="functions.filter_diskann_tune",
            default_config="configs/filter_diskann_tune.yaml",
            description="Run Filter-DiskANN (Vamana) tuning with agentic diagnosis (independent unify architecture).",
        ),
        "nhq": TuningEntrypoint(
            name="nhq",
            module="functions.nhq_tune",
            default_config="configs/nhq_tune.yaml",
            description="Run NHQ (Native Hybrid Query) recall/QPS tuning with agentic diagnosis.",
        ),
    }
)


def available_pipelines() -> Tuple[TuningEntrypoint, ...]:
    return tuple(_PIPELINES.values())


def get_tuning_entrypoint(name: str) -> TuningEntrypoint:
    key = str(name or "").strip()
    try:
        return _PIPELINES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_PIPELINES))
        raise ValueError(f"Unknown tuning pipeline '{name}'. Available pipelines: {known}") from exc


def load_run_pipeline(name: str) -> Callable[..., int]:
    entrypoint = get_tuning_entrypoint(name)
    module = import_module(entrypoint.module)
    run_pipeline = getattr(module, "run_pipeline", None)
    if not callable(run_pipeline):
        raise AttributeError(f"Module '{entrypoint.module}' does not expose callable run_pipeline.")
    return run_pipeline


__all__ = [
    "TuningEntrypoint",
    "available_pipelines",
    "get_tuning_entrypoint",
    "load_run_pipeline",
]

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_DATASETS = ("train", "test", "neighbors")


def _load_hdf5_dependencies():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read HDF5 benchmark data. Install with: pip install h5py") from exc

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for this benchmark. Install with: pip install numpy") from exc

    return h5py, np


def _load_hnswlib():
    try:
        import hnswlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hnswlib is required for this benchmark. Install the local package with: pip install -e hnswlib"
        ) from exc
    return hnswlib


def _read_hdf5(data_path: Path, max_train: int | None, max_query: int | None):
    if not data_path.exists():
        raise FileNotFoundError(f"HDF5 data path does not exist: {data_path}")

    h5py, np = _load_hdf5_dependencies()
    with h5py.File(data_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(
                f"HDF5 file must contain datasets {list(REQUIRED_DATASETS)}; missing: {missing}"
            )

        train = np.asarray(handle["train"][:], dtype=np.float32)
        test = np.asarray(handle["test"][:], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"][:])

    if train.ndim != 2:
        raise ValueError(f"HDF5 dataset 'train' must be 2D, got shape={train.shape}")
    if test.ndim != 2:
        raise ValueError(f"HDF5 dataset 'test' must be 2D, got shape={test.shape}")
    if neighbors.ndim != 2:
        raise ValueError(f"HDF5 dataset 'neighbors' must be 2D, got shape={neighbors.shape}")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"train/test dimensions differ: train={train.shape}, test={test.shape}")

    if max_train is not None:
        if max_train <= 0:
            raise ValueError("--max-train must be > 0 when provided")
        train = train[:max_train]
    if max_query is not None:
        if max_query <= 0:
            raise ValueError("--max-query must be > 0 when provided")
        test = test[:max_query]
        neighbors = neighbors[:max_query]

    if len(train) == 0:
        raise ValueError("HDF5 dataset 'train' is empty after applying limits")
    if len(test) == 0:
        raise ValueError("HDF5 dataset 'test' is empty after applying limits")
    if len(neighbors) < len(test):
        raise ValueError(
            f"HDF5 dataset 'neighbors' has fewer rows than test queries: neighbors={neighbors.shape}, test={test.shape}"
        )
    return train, test, neighbors


def _compute_recall(labels, neighbors, k: int) -> float:
    if neighbors.shape[1] < k:
        raise ValueError(f"HDF5 dataset 'neighbors' must have at least k={k} columns")
    total = labels.shape[0] * k
    if total <= 0:
        return 0.0

    matches = 0
    for predicted, truth in zip(labels[:, :k], neighbors[:, :k]):
        truth_set = set(int(item) for item in truth)
        matches += sum(1 for item in predicted if int(item) in truth_set)
    return float(matches) / float(total)


def _index_size_mb(index: Any) -> float:
    if hasattr(index, "index_file_size"):
        try:
            return float(index.index_file_size()) / (1024.0 * 1024.0)
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "hnsw.index"
        index.save_index(str(path))
        return float(path.stat().st_size) / (1024.0 * 1024.0)


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return stem.strip("._") or "dataset"


def _resolve_thread_count(explicit: int | None, fallback: int) -> int:
    value = fallback if explicit is None else explicit
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("thread counts must be > 0")
    return resolved


def _resolve_ef_list(args: argparse.Namespace) -> List[int]:
    """Return an ordered list of ef values to scan.

    When ``--ef-scan-start`` and ``--ef-scan-end`` are both provided, the list
    covers every integer from *start* to *end* (step = 1).  Otherwise the
    explicit ``--ef-list`` is used, falling back to the single ``--ef`` value.
    """
    ef_scan_start = getattr(args, "ef_scan_start", None)
    ef_scan_end = getattr(args, "ef_scan_end", None)
    if ef_scan_start is not None and ef_scan_end is not None:
        lo = int(ef_scan_start)
        hi = int(ef_scan_end)
        if lo > hi:
            lo, hi = hi, lo
        values = list(range(lo, hi + 1))
    else:
        raw_list = getattr(args, "ef_list", None)
        if raw_list:
            values = sorted({int(value) for value in raw_list})
        else:
            values = [int(args.ef)]
    if not values:
        raise ValueError("At least one ef value is required.")
    if any(value <= 0 for value in values):
        raise ValueError("All ef values must be > 0.")
    if any(value < int(args.k) for value in values):
        raise ValueError(f"All ef values must be >= --k; got ef_list={values}, k={args.k}")
    return values


def _select_frontier_point(
    frontier: List[Dict[str, Any]],
    *,
    default_ef: int,
    recall_threshold: float | None,
    recall_slack: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_ef = {int(point["ef"]): point for point in frontier}
    default_point = by_ef.get(int(default_ef), frontier[0])

    if recall_threshold is None:
        return dict(default_point), {
            "mode": "explicit_ef",
            "selected_by": "requested_ef",
            "requested_ef": int(default_ef),
            "recall_threshold": None,
            "recall_slack": float(recall_slack),
            "best_feasible": None,
            "best_within_slack": None,
        }

    feasible = [point for point in frontier if float(point["recall"]) >= float(recall_threshold)]
    within_slack = [
        point
        for point in frontier
        if float(point["recall"]) >= float(recall_threshold) - float(recall_slack)
    ]

    def _pick_best(points: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not points:
            return None
        return max(points, key=lambda point: (float(point["qps"]), -int(point["ef"]), float(point["recall"])))

    best_feasible = _pick_best(feasible)
    if best_feasible is not None:
        return dict(best_feasible), {
            "mode": "frontier_threshold_feasible_then_qps",
            "selected_by": "best_feasible_qps",
            "requested_ef": int(default_ef),
            "recall_threshold": float(recall_threshold),
            "recall_slack": float(recall_slack),
            "best_feasible": dict(best_feasible),
            "best_within_slack": dict(_pick_best(within_slack) or best_feasible),
        }

    best_within_slack = _pick_best(within_slack)
    if best_within_slack is not None:
        return dict(best_within_slack), {
            "mode": "frontier_threshold_feasible_then_qps",
            "selected_by": "best_within_slack_qps",
            "requested_ef": int(default_ef),
            "recall_threshold": float(recall_threshold),
            "recall_slack": float(recall_slack),
            "best_feasible": None,
            "best_within_slack": dict(best_within_slack),
        }

    best_recall = max(frontier, key=lambda point: (float(point["recall"]), float(point["qps"]), -int(point["ef"])))
    return dict(best_recall), {
        "mode": "frontier_threshold_feasible_then_qps",
        "selected_by": "max_recall_fallback",
        "requested_ef": int(default_ef),
        "recall_threshold": float(recall_threshold),
        "recall_slack": float(recall_slack),
        "best_feasible": None,
        "best_within_slack": None,
    }


def _index_cache_paths(args: argparse.Namespace, data_path: Path, dim: int, train_count: int) -> Tuple[Path, Path]:
    cache_dir = Path(args.index_cache_dir).expanduser()
    dataset = _safe_stem(data_path.stem)
    data_id = hashlib.sha1(str(data_path.resolve()).encode("utf-8")).hexdigest()[:10]
    max_train = "all" if args.max_train is None else str(int(args.max_train))
    build_num_threads = _resolve_thread_count(args.build_num_threads, args.num_threads)
    name = (
        f"hnswlib_{dataset}_{data_id}_{args.space}_seed{int(args.seed)}_"
        f"dim{int(dim)}_n{int(train_count)}_maxtrain{max_train}_"
        f"M{int(args.M)}_efC{int(args.ef_construction)}_buildT{int(build_num_threads)}.index"
    )
    index_path = cache_dir / name
    return index_path, Path(str(index_path) + ".meta.json")


def _read_index_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_index_meta(meta_path: Path, payload: Dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cached_index(index: Any, index_path: Path, train_count: int) -> None:
    if hasattr(index, "load_index"):
        try:
            index.load_index(str(index_path), max_elements=int(train_count))
        except TypeError:
            index.load_index(str(index_path))
        return
    raise RuntimeError("hnswlib Index object does not support load_index; cannot reuse cached index")


def _cached_index_size_mb(index: Any, index_path: Path) -> float:
    if index_path.exists():
        return float(index_path.stat().st_size) / (1024.0 * 1024.0)
    return _index_size_mb(index)


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.data_path).expanduser()
    train, test, neighbors = _read_hdf5(
        data_path=data_path,
        max_train=args.max_train,
        max_query=args.max_query,
    )
    hnswlib = _load_hnswlib()
    _, np = _load_hdf5_dependencies()
    build_num_threads = _resolve_thread_count(getattr(args, "build_num_threads", None), args.num_threads)
    query_num_threads = _resolve_thread_count(getattr(args, "query_num_threads", None), args.num_threads)

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.ef < args.k:
        raise ValueError(f"--ef must be >= --k; got ef={args.ef}, k={args.k}")
    if args.M <= 0:
        raise ValueError("--M must be > 0")
    if args.ef_construction <= 0:
        raise ValueError("--ef-construction must be > 0")
    ef_list = _resolve_ef_list(args)
    select_recall_threshold = getattr(args, "select_recall_threshold", None)
    if select_recall_threshold is not None:
        select_recall_threshold = float(select_recall_threshold)
        if not (0.0 < select_recall_threshold <= 1.0):
            raise ValueError("--select-recall-threshold must be in (0, 1].")
    select_recall_slack = float(getattr(args, "select_recall_slack", 0.0) or 0.0)
    if select_recall_slack < 0.0:
        raise ValueError("--select-recall-slack must be >= 0.")

    index = hnswlib.Index(space=args.space, dim=int(train.shape[1]))

    index_reused = False
    index_load_time_s = 0.0
    original_build_time_s = 0.0
    index_path_text = ""
    index_path: Path | None = None
    meta_path: Path | None = None

    if getattr(args, "index_cache_dir", ""):
        index_path, meta_path = _index_cache_paths(
            args=args,
            data_path=data_path,
            dim=int(train.shape[1]),
            train_count=int(train.shape[0]),
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path_text = str(index_path)

    if index_path is not None and index_path.exists():
        meta = _read_index_meta(meta_path or Path(str(index_path) + ".meta.json"))
        original_build_time_s = float(meta.get("original_build_time_s", 0.0) or 0.0)
    if index_path is not None and index_path.exists() and original_build_time_s > 0.0:
        load_start = time.perf_counter()
        _load_cached_index(index, index_path=index_path, train_count=int(train.shape[0]))
        index_load_time_s = time.perf_counter() - load_start
        index_reused = True
        build_time_s = float(original_build_time_s + index_load_time_s)
    else:
        ids = np.arange(train.shape[0])
        build_start = time.perf_counter()
        index.init_index(
            max_elements=int(train.shape[0]),
            ef_construction=int(args.ef_construction),
            M=int(args.M),
            random_seed=int(args.seed),
        )
        index.add_items(train, ids, num_threads=int(build_num_threads))
        if index_path is not None:
            index.save_index(str(index_path))
        build_time_s = time.perf_counter() - build_start
        original_build_time_s = float(build_time_s)
        if index_path is not None and meta_path is not None:
            _write_index_meta(
                meta_path,
                {
                    "index_path": str(index_path),
                    "data_path": str(data_path),
                    "space": args.space,
                    "seed": int(args.seed),
                    "dim": int(train.shape[1]),
                    "train_count": int(train.shape[0]),
                    "max_train": args.max_train,
                    "M": int(args.M),
                    "ef_construction": int(args.ef_construction),
                    "build_num_threads": int(build_num_threads),
                    "original_build_time_s": float(original_build_time_s),
                },
            )

    frontier: List[Dict[str, Any]] = []
    # Early-stop: once recall reaches the threshold, scan at most
    # MAX_AFTER_THRESHOLD more ef values then break.  Because QPS is
    # monotonic-decreasing in ef, the first feasible ef gives the
    # fastest QPS; a few extra points provide a noise margin.
    _MAX_AFTER_THRESHOLD = 3
    _after_threshold_count = 0
    _threshold_hit = False
    for ef in ef_list:
        index.set_ef(int(ef))
        # Collect per-query diagnostic stats (hnswlib native counters).
        index.reset_distance_computations()
        index.reset_visited_nodes()
        query_start = time.perf_counter()
        labels, distances = index.knn_query(test, k=int(args.k), num_threads=int(query_num_threads))
        query_time_s = time.perf_counter() - query_start
        dist_comps = index.get_distance_computations()
        visited = index.get_visited_nodes()
        qps = float(test.shape[0]) / max(query_time_s, 1e-12)
        recall = _compute_recall(labels, neighbors, int(args.k))
        frontier.append(
            {
                "ef": int(ef),
                "recall": float(recall),
                "qps": float(qps),
                "search_time_s": float(query_time_s),
                "dist_comps_per_query": float(dist_comps) / max(1, test.shape[0]),
                "visited_nodes_per_query": float(visited) / max(1, test.shape[0]),
                "candidate_distance_stats": {
                    "mean": float(distances.mean()),
                    "std": float(distances.std()),
                    "min": float(distances.min()),
                    "max": float(distances.max()),
                },
            }
        )
        if select_recall_threshold is not None and float(recall) >= float(select_recall_threshold):
            if not _threshold_hit:
                _threshold_hit = True
                _after_threshold_count = 0
            else:
                _after_threshold_count += 1
        elif _threshold_hit:
            _after_threshold_count += 1
        if _threshold_hit and _after_threshold_count >= _MAX_AFTER_THRESHOLD:
            break

    selected_point, selection = _select_frontier_point(
        frontier,
        default_ef=int(args.ef),
        recall_threshold=select_recall_threshold,
        recall_slack=select_recall_slack,
    )
    feasible_frontier = (
        [
            point
            for point in frontier
            if select_recall_threshold is not None and float(point["recall"]) >= float(select_recall_threshold)
        ]
        if select_recall_threshold is not None
        else []
    )
    # ── Graph topology metrics (approximated from per-query diagnostics) ──
    # out_degree_mean ≈ 2 * dist_comps / visited_nodes  (from signal S2: ratio ≈ M/2)
    # clamped to nominal M since actual out-degree cannot exceed configured M.
    dc = float(selected_point.get("dist_comps_per_query", 0.0))
    vn = float(selected_point.get("visited_nodes_per_query", 1.0))
    approx_out_degree = max(0.0, 2.0 * dc / max(1.0, vn))
    out_degree_mean = min(float(args.M), round(approx_out_degree, 1))

    # In-degree stats are not available without C++ instrumentation.
    # The benchmark reports null; the LLM prompt notes this is unavailable.
    in_degree_mean = None
    in_degree_std = None
    in_degree_max = None

    # Attempt native C++ collection if the .so exposes the methods.
    try:
        native_out = getattr(index, "compute_out_degree_mean", None)
        native_in = getattr(index, "compute_in_degree_stats", None)
        if callable(native_out):
            out_degree_mean = float(native_out())
        if callable(native_in):
            in_deg = native_in()
            if isinstance(in_deg, dict):
                in_degree_mean = float(in_deg.get("mean", 0))
                in_degree_std = float(in_deg.get("std", 0))
                in_degree_max = int(in_deg.get("max", 0))
    except Exception:
        pass

    return {
        "recall": float(selected_point["recall"]),
        "qps": float(selected_point["qps"]),
        "selected_ef": int(selected_point["ef"]),
        "selected_recall": float(selected_point["recall"]),
        "selected_qps": float(selected_point["qps"]),
        "selection": selection,
        "frontier": frontier,
        "frontier_summary": {
            "point_count": len(frontier),
            "min_ef": min(point["ef"] for point in frontier),
            "max_ef": max(point["ef"] for point in frontier),
            "max_recall": max(float(point["recall"]) for point in frontier),
            "max_qps": max(float(point["qps"]) for point in frontier),
            "best_feasible_qps": (
                max(feasible_frontier, key=lambda point: (float(point["qps"]), -int(point["ef"])))
                if feasible_frontier
                else None
            ),
        },
        "build_time_s": float(build_time_s),
        "dist_comps_per_query": float(selected_point.get("dist_comps_per_query", 0.0)),
        "visited_nodes_per_query": float(selected_point.get("visited_nodes_per_query", 0.0)),
        "candidate_distance_stats": selected_point.get("candidate_distance_stats", {}),
        "index_size_mb": float(
            _cached_index_size_mb(index, index_path) if index_path is not None else _index_size_mb(index)
        ),
        "index_reused": bool(index_reused),
        "index_load_time_s": float(index_load_time_s),
        "original_build_time_s": float(original_build_time_s),
        "index_path": index_path_text,
        "build_num_threads": int(build_num_threads),
        "query_num_threads": int(query_num_threads),
        "out_degree_mean": out_degree_mean,
        "in_degree_mean": in_degree_mean,
        "in_degree_std": in_degree_std,
        "in_degree_max": in_degree_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an hnswlib recall/QPS benchmark.")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--ef-construction", dest="ef_construction", type=int, required=True)
    parser.add_argument("--ef", type=int, required=True)
    parser.add_argument("--ef-list", dest="ef_list", type=int, nargs="+", default=None)
    parser.add_argument("--ef-scan-start", dest="ef_scan_start", type=int, default=None)
    parser.add_argument("--ef-scan-end", dest="ef_scan_end", type=int, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--metrics-output", type=str, required=True)
    parser.add_argument("--space", type=str, default="l2", choices=["l2", "ip", "cosine"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--build-num-threads", type=int, default=None)
    parser.add_argument("--query-num-threads", type=int, default=None)
    parser.add_argument("--select-recall-threshold", type=float, default=None)
    parser.add_argument("--select-recall-slack", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument("--index-cache-dir", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = run_benchmark(args)
        output_path = Path(args.metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"hnswlib benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

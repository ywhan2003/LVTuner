import argparse
import csv
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DTYPE_TO_NUMPY = {
    "float": "float32",
    "int8": "int8",
    "uint8": "uint8",
}


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


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return stem.strip("._") or "dataset"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _normalize_recall(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 100.0 if float(value) > 1.0 else float(value)


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_diskann_bin(path: Path, vectors: Any, dtype_name: str) -> Tuple[int, int]:
    _, np = _load_hdf5_dependencies()
    if vectors.ndim != 2:
        raise ValueError(f"DiskANN vectors must be 2D, got shape={vectors.shape}")
    if len(vectors) == 0:
        raise ValueError("DiskANN vectors are empty")

    cast = np.asarray(vectors, dtype=np.dtype(DTYPE_TO_NUMPY[dtype_name]), order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.asarray([cast.shape[0], cast.shape[1]], dtype=np.int32).tofile(handle)
        cast.tofile(handle)
    return int(cast.shape[0]), int(cast.shape[1])


def _read_hdf5_vectors(data_path: Path, max_train: int | None, max_query: int | None):
    if not data_path.exists():
        raise FileNotFoundError(f"HDF5 data path does not exist: {data_path}")

    h5py, np = _load_hdf5_dependencies()
    with h5py.File(data_path, "r") as handle:
        missing = [name for name in ("train", "test") if name not in handle]
        if missing:
            raise ValueError(f"HDF5 file must contain datasets ['train', 'test']; missing: {missing}")

        train = np.asarray(handle["train"][:])
        test = np.asarray(handle["test"][:])

    if train.ndim != 2:
        raise ValueError(f"HDF5 dataset 'train' must be 2D, got shape={train.shape}")
    if test.ndim != 2:
        raise ValueError(f"HDF5 dataset 'test' must be 2D, got shape={test.shape}")
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

    if len(train) == 0:
        raise ValueError("HDF5 dataset 'train' is empty after applying limits")
    if len(test) == 0:
        raise ValueError("HDF5 dataset 'test' is empty after applying limits")
    return train, test


def _read_bin_metadata(path: Path) -> Tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"DiskANN vector bin path does not exist: {path}")
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"DiskANN vector bin has an invalid header: {path}")
    npts, dim = struct.unpack("<ii", header)
    if npts <= 0 or dim <= 0:
        raise ValueError(f"DiskANN vector bin metadata must be positive, got npts={npts}, dim={dim}")
    return npts, dim


def _prepare_vector_bins(args: argparse.Namespace) -> Tuple[Path, Path, int, int]:
    if getattr(args, "query_file", ""):
        if not args.data_path:
            raise ValueError("--data_path is required when --query_file is provided")
        if args.base_bin or args.query_bin:
            raise ValueError("Provide either native --data_path/--query_file or --base-bin/--query-bin, not both")
        base_bin = Path(args.data_path).expanduser()
        query_bin = Path(args.query_file).expanduser()
    elif args.data_path:
        if args.base_bin or args.query_bin:
            raise ValueError("Provide either --data-path HDF5 or --base-bin/--query-bin, not both")
        work_dir = Path(args.work_dir).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)

        data_path = Path(args.data_path).expanduser()
        train, test = _read_hdf5_vectors(data_path, args.max_train, args.max_query)
        data_id = _sha1_text(
            json.dumps(
                {
                    "data": _file_fingerprint(data_path),
                    "dtype": args.data_type,
                    "max_train": args.max_train,
                    "max_query": args.max_query,
                },
                sort_keys=True,
            )
        )[:12]
        stem = _safe_stem(data_path.stem)
        base_bin = work_dir / f"{stem}_{data_id}_base.bin"
        query_bin = work_dir / f"{stem}_{data_id}_query.bin"
        if not base_bin.exists():
            _write_diskann_bin(base_bin, train, args.data_type)
        if not query_bin.exists():
            _write_diskann_bin(query_bin, test, args.data_type)
    elif not (args.base_bin and args.query_bin):
        raise ValueError(
            "Provide native --data_path/--query_file, --data-path HDF5, or both --base-bin and --query-bin"
        )
    else:
        base_bin = Path(args.base_bin).expanduser()
        query_bin = Path(args.query_bin).expanduser()

    n_base, dim = _read_bin_metadata(base_bin)
    n_query, query_dim = _read_bin_metadata(query_bin)
    if dim != query_dim:
        raise ValueError(f"base/query dimensions differ: base={dim}, query={query_dim}")
    return base_bin, query_bin, n_base, n_query


def _run_command(cmd: Sequence[str], description: str) -> Tuple[str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        command = " ".join(cmd)
        raise RuntimeError(
            f"{description} failed with exit code {proc.returncode}\n"
            f"Command: {command}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), elapsed


def _resolve_executable(path_text: str, name: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} binary does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} path is not a file: {path}")
    return path


def _build_fingerprint(args: argparse.Namespace, base_bin: Path, label_file: Path) -> Dict[str, Any]:
    return {
        "benchmark": "diskann_memory_filtered",
        "data_type": args.data_type,
        "dist_fn": args.dist_fn,
        "base_bin": _file_fingerprint(base_bin),
        "label_file": _file_fingerprint(label_file),
        "R": int(args.R),
        "Lbuild": int(args.Lbuild),
        "filtered_Lbuild": int(args.filtered_Lbuild),
        "alpha": float(args.alpha),
        "build_pq_bytes": int(args.build_pq_bytes),
        "use_opq": bool(args.use_opq),
        "universal_label": args.universal_label,
        "label_type": args.label_type,
    }


def _index_cache_paths(args: argparse.Namespace, fingerprint: Dict[str, Any], source_name: str) -> Tuple[Path, Path]:
    digest = _sha1_text(json.dumps(fingerprint, sort_keys=True))[:12]
    suffix = (
        f"{digest}_R{int(args.R)}_LB{int(args.Lbuild)}_"
        f"FL{int(args.filtered_Lbuild)}_a{_float_token(float(args.alpha))}"
    )
    if getattr(args, "index_path_prefix", ""):
        base_prefix = Path(args.index_path_prefix).expanduser()
        base_prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix = Path(f"{base_prefix}_{suffix}")
    else:
        cache_dir = Path(args.index_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        prefix = cache_dir / f"diskann_memory_filtered_{_safe_stem(source_name)}_{suffix}"
    return prefix, Path(str(prefix) + ".meta.json")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _index_files_exist(prefix: Path) -> bool:
    files = [path for path in prefix.parent.glob(prefix.name + "*") if path.name != prefix.name + ".meta.json"]
    return any(path.is_file() and path.stat().st_size > 0 for path in files)


def _build_index(
    args: argparse.Namespace,
    build_binary: Path,
    base_bin: Path,
    label_file: Path,
    index_prefix: Path,
    meta_path: Path,
    fingerprint: Dict[str, Any],
) -> Tuple[bool, float, float]:
    meta = _read_json(meta_path)
    original_build_time_s = float(meta.get("original_build_time_s", 0.0) or 0.0)
    if meta.get("fingerprint") == fingerprint and original_build_time_s > 0.0 and _index_files_exist(index_prefix):
        return True, 0.0, original_build_time_s

    cmd = [
        str(build_binary),
        "--data_type",
        args.data_type,
        "--dist_fn",
        args.dist_fn,
        "--data_path",
        str(base_bin),
        "--index_path_prefix",
        str(index_prefix),
        "-R",
        str(int(args.R)),
        "-L",
        str(int(args.Lbuild)),
        "--FilteredLbuild",
        str(int(args.filtered_Lbuild)),
        "--alpha",
        str(float(args.alpha)),
        "-T",
        str(int(args.num_threads)),
        "--label_file",
        str(label_file),
        "--universal_label",
        args.universal_label,
        "--label_type",
        args.label_type,
    ]
    if int(args.build_pq_bytes) > 0:
        cmd.extend(["--build_PQ_bytes", str(int(args.build_pq_bytes))])
    if args.use_opq:
        cmd.append("--use_opq")

    _, build_time_s = _run_command(cmd, "DiskANN build_memory_index")
    _write_json(
        meta_path,
        {
            "fingerprint": fingerprint,
            "index_path_prefix": str(index_prefix),
            "original_build_time_s": float(build_time_s),
        },
    )
    return False, build_time_s, build_time_s


def _compute_groundtruth(
    args: argparse.Namespace,
    gt_binary: Path,
    base_bin: Path,
    query_bin: Path,
    label_file: Path,
    gt_file: Path,
) -> float:
    cmd = [
        str(gt_binary),
        "--data_type",
        args.data_type,
        "--dist_fn",
        args.dist_fn,
        "--base_file",
        str(base_bin),
        "--query_file",
        str(query_bin),
        "--gt_file",
        str(gt_file),
        "--K",
        str(int(args.gt_k)),
        "--label_file",
        str(label_file),
        "--universal_label",
        args.universal_label,
    ]
    if args.filter_label:
        cmd.extend(["--filter_label", args.filter_label])
    else:
        cmd.extend(["--filter_label_file", str(Path(args.query_filters_file).expanduser())])
    _, elapsed = _run_command(cmd, "DiskANN compute_groundtruth_for_filters")
    return elapsed


def _parse_search_rows(text: str, k: int, print_all_recalls: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not re.match(r"^\d+\s+", line):
            continue
        values = line.split()
        if len(values) < 5:
            continue
        try:
            search_list = int(values[0])
            qps = float(values[1])
            numeric_tail = [float(item) for item in values[2:]]
        except ValueError:
            continue

        # New format (with enhanced diagnostics): L, QPS, Avg dist cmps, Visited nodes, Cand dist mean, Mean Latency, 99.9 Latency, [recalls...]
        # Old format (tags mode): L, QPS, Mean Latency, 99.9 Latency, [recalls...]
        # Old format (no-tags): L, QPS, Avg dist cmps, Mean Latency, 99.9 Latency, [recalls...]
        # Distinguish by checking if the 4th value looks like latency (> 100 typically) vs visited_nodes (integer, usually < 10000)
        if len(numeric_tail) >= 5 and numeric_tail[1] < 100000 and numeric_tail[1] == int(numeric_tail[1]):
            # New format with visited nodes and cand dist mean
            avg_dist_cmps = numeric_tail[0]
            visited_nodes = numeric_tail[1]
            cand_dist_mean = numeric_tail[2]
            latency_mean_us = numeric_tail[3]
            latency_999_us = numeric_tail[4]
            recall_values = numeric_tail[5:]
        elif len(numeric_tail) >= 3 and numeric_tail[0] > 100:
            # Old format no-tags: avg_dist_cmps, latency, 99.9, [recalls...]
            avg_dist_cmps = numeric_tail[0]
            visited_nodes = None
            cand_dist_mean = None
            latency_mean_us = numeric_tail[1]
            latency_999_us = numeric_tail[2]
            recall_values = numeric_tail[3:]
        else:
            # Old format tags mode: latency, 99.9, [recalls...]
            avg_dist_cmps = None
            visited_nodes = None
            cand_dist_mean = None
            latency_mean_us = numeric_tail[0]
            latency_999_us = numeric_tail[1] if len(numeric_tail) > 1 else None
            recall_values = numeric_tail[2:]

        recall = None
        if recall_values:
            recall = _normalize_recall(recall_values[-1] if print_all_recalls else recall_values[0])
        rows.append(
            {
                "search_list": search_list,
                "qps": qps,
                "avg_dist_cmps": avg_dist_cmps,
                "visited_nodes": visited_nodes,
                "cand_dist_mean": cand_dist_mean,
                "latency_mean_us": latency_mean_us,
                "latency_999_us": latency_999_us,
                "recall": recall,
            }
        )
    return rows


def _parse_search_stats_csv(path: Path, print_all_recalls: bool) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    parsed_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        qps_raw = row.get("QPS")
        if qps_raw in (None, ""):
            qps_raw = row.get("QPS/thread")
        recall_keys = sorted(
            [key for key in row if key.startswith("recall_")],
            key=lambda key: int(key.split("_", 1)[1]) if key.split("_", 1)[1].isdigit() else 0,
        )
        recall = row.get("recall")
        if recall in (None, "") and recall_keys:
            recall = row[recall_keys[-1]] if print_all_recalls else row[recall_keys[0]]

        parsed_rows.append(
            {
                "search_list": int(float(row["L"])),
                "qps": float(qps_raw) if qps_raw not in (None, "") else None,
                "avg_dist_cmps": float(row["Avg dist cmps"]) if row.get("Avg dist cmps") not in (None, "") else None,
                "visited_nodes": float(row["Visited nodes"]) if row.get("Visited nodes") not in (None, "") else None,
                "cand_dist_mean": float(row["Cand dist mean"]) if row.get("Cand dist mean") not in (None, "") else None,
                "latency_mean_us": float(row["Mean Latency (mus)"])
                if row.get("Mean Latency (mus)") not in (None, "")
                else None,
                "latency_999_us": float(row["99.9 Latency"])
                if row.get("99.9 Latency") not in (None, "")
                else None,
                "recall": _normalize_recall(float(recall)) if recall not in (None, "") else None,
            }
        )
    return parsed_rows


def _read_search_metrics(result_prefix: Path, stdout: str, k: int, print_all_recalls: bool) -> List[Dict[str, Any]]:
    stats_csv_path = Path(str(result_prefix) + "_stats.csv")
    if stats_csv_path.exists():
        rows = _parse_search_stats_csv(stats_csv_path, print_all_recalls)
        if rows:
            return rows

    meta_path = Path(str(result_prefix) + "meta.txt")
    if meta_path.exists():
        rows = _parse_search_rows(meta_path.read_text(encoding="utf-8", errors="replace"), k, print_all_recalls)
        if rows:
            return rows
    return _parse_search_rows(stdout, k, print_all_recalls)


def _search_index(
    args: argparse.Namespace,
    search_binary: Path,
    query_bin: Path,
    gt_file: Path | None,
    index_prefix: Path,
) -> Tuple[List[Dict[str, Any]], Path, float]:
    result_fingerprint = {
        "index_path_prefix": str(index_prefix),
        "query_bin": _file_fingerprint(query_bin),
        "gt_file": _file_fingerprint(gt_file) if gt_file is not None else None,
        "filter_label": args.filter_label,
        "query_filters_file": _file_fingerprint(Path(args.query_filters_file).expanduser())
        if args.query_filters_file
        else None,
        "k": int(args.k),
        "search_list": [int(item) for item in args.search_list],
        "label_type": args.label_type,
    }
    result_digest = _sha1_text(json.dumps(result_fingerprint, sort_keys=True))[:12]
    l_token = "_".join(str(int(item)) for item in args.search_list)
    if getattr(args, "result_path", ""):
        result_base = Path(args.result_path).expanduser()
        result_base.parent.mkdir(parents=True, exist_ok=True)
        result_prefix = Path(f"{result_base}_{result_digest}_L{l_token}")
    else:
        result_dir = Path(args.result_dir).expanduser()
        result_dir.mkdir(parents=True, exist_ok=True)
        result_prefix = result_dir / f"{index_prefix.name}_search_{result_digest}_L{l_token}"
    cmd = [
        str(search_binary),
        "--data_type",
        args.data_type,
        "--dist_fn",
        args.dist_fn,
        "--index_path_prefix",
        str(index_prefix),
        "--result_path",
        str(result_prefix),
        "--query_file",
        str(query_bin),
        "--gt_file",
        str(gt_file) if gt_file is not None else "null",
        "-K",
        str(int(args.k)),
        "-L",
        *[str(int(item)) for item in args.search_list],
        "-T",
        str(int(args.search_num_threads if args.search_num_threads is not None else args.num_threads)),
        "--label_type",
        args.label_type,
        "--fail_if_recall_below",
        str(float(args.fail_if_recall_below)),
    ]
    if args.filter_label:
        cmd.extend(["--filter_label", args.filter_label])
    else:
        cmd.extend(["--query_filters_file", str(Path(args.query_filters_file).expanduser())])
    if args.print_all_recalls:
        cmd.append("--print_all_recalls")
    if args.print_qps_per_thread:
        cmd.append("--print_qps_per_thread")

    stdout, elapsed = _run_command(cmd, "DiskANN search_memory_index")
    rows = _read_search_metrics(result_prefix, stdout, int(args.k), bool(args.print_all_recalls))
    if not rows:
        raise RuntimeError(f"Could not parse search metrics from search output or {result_prefix}meta.txt")
    return rows, result_prefix, elapsed


def _validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.gt_k <= 0:
        raise ValueError("--gt-k must be > 0")
    if args.gt_k < args.k:
        raise ValueError(f"--gt-k must be >= --k; got gt_k={args.gt_k}, k={args.k}")
    if args.R <= 0:
        raise ValueError("--R must be > 0")
    if args.Lbuild <= 0:
        raise ValueError("--Lbuild must be > 0")
    if args.filtered_Lbuild <= 0:
        raise ValueError("--filtered-Lbuild must be > 0")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be > 0")
    if any(item < args.k for item in args.search_list):
        raise ValueError(f"all --search-list values must be >= --k={args.k}")
    if not args.label_file:
        raise ValueError("--label-file is required for Filtered-DiskANN")
    if bool(args.filter_label) == bool(args.query_filters_file):
        raise ValueError("Provide exactly one of --filter-label or --query-filters-file")
    if args.compute_gt and args.gt_file:
        raise ValueError("Use either --compute-gt or --gt-file, not both")
    if not args.compute_gt and not args.gt_file:
        raise ValueError("Provide --gt-file or use --compute-gt")


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    build_binary = _resolve_executable(args.build_memory_index, "build_memory_index")
    search_binary = _resolve_executable(args.search_memory_index, "search_memory_index")

    label_file = Path(args.label_file).expanduser()
    if not label_file.exists():
        raise FileNotFoundError(f"--label-file does not exist: {label_file}")
    if args.query_filters_file and not Path(args.query_filters_file).expanduser().exists():
        raise FileNotFoundError(f"--query-filters-file does not exist: {args.query_filters_file}")

    base_bin, query_bin, n_base, n_query = _prepare_vector_bins(args)
    fingerprint = _build_fingerprint(args, base_bin, label_file)
    source_name = Path(args.data_path or args.base_bin).expanduser().stem
    index_prefix, meta_path = _index_cache_paths(args, fingerprint, source_name)

    index_reused, build_time_s, original_build_time_s = _build_index(
        args=args,
        build_binary=build_binary,
        base_bin=base_bin,
        label_file=label_file,
        index_prefix=index_prefix,
        meta_path=meta_path,
        fingerprint=fingerprint,
    )

    gt_compute_time_s = 0.0
    if args.compute_gt:
        gt_binary = _resolve_executable(args.compute_groundtruth_for_filters, "compute_groundtruth_for_filters")
        gt_dir = Path(args.gt_cache_dir).expanduser()
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_key = _sha1_text(
            json.dumps(
                {
                    "base": _file_fingerprint(base_bin),
                    "query": _file_fingerprint(query_bin),
                    "labels": _file_fingerprint(label_file),
                    "filter_label": args.filter_label,
                    "query_filters_file": _file_fingerprint(Path(args.query_filters_file).expanduser())
                    if args.query_filters_file
                    else None,
                    "universal_label": args.universal_label,
                    "dist_fn": args.dist_fn,
                    "data_type": args.data_type,
                    "gt_k": int(args.gt_k),
                },
                sort_keys=True,
            )
        )[:12]
        gt_file = gt_dir / f"diskann_filtered_gt_{_safe_stem(source_name)}_{gt_key}.bin"
        if not gt_file.exists():
            gt_compute_time_s = _compute_groundtruth(args, gt_binary, base_bin, query_bin, label_file, gt_file)
    else:
        gt_file = Path(args.gt_file).expanduser()
        if not gt_file.exists():
            raise FileNotFoundError(f"--gt-file does not exist: {gt_file}")

    per_search_list, result_prefix, search_time_s = _search_index(
        args=args,
        search_binary=search_binary,
        query_bin=query_bin,
        gt_file=gt_file,
        index_prefix=index_prefix,
    )
    best = max(per_search_list, key=lambda row: ((row.get("recall") or 0.0), row.get("qps") or 0.0))
    return {
        "recall": best.get("recall"),
        "qps": best.get("qps"),
        "latency_mean_us": best.get("latency_mean_us"),
        "latency_999_us": best.get("latency_999_us"),
        "avg_dist_cmps": best.get("avg_dist_cmps"),
        "visited_nodes": best.get("visited_nodes"),
        "cand_dist_mean": best.get("cand_dist_mean"),
        "build_time_s": float(build_time_s),
        "search_time_s": float(search_time_s),
        "gt_compute_time_s": float(gt_compute_time_s),
        "index_reused": bool(index_reused),
        "original_build_time_s": float(original_build_time_s),
        "index_path_prefix": str(index_prefix),
        "result_prefix": str(result_prefix),
        "gt_file": str(gt_file),
        "base_bin": str(base_bin),
        "query_bin": str(query_bin),
        "base_count": int(n_base),
        "query_count": int(n_query),
        "best_search_list": best.get("search_list"),
        "per_search_list": per_search_list,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a Filtered-DiskANN in-memory filtered-vamana benchmark.")

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--data-path",
        "--data_path",
        dest="data_path",
        type=str,
        default="",
        help="HDF5 train/test file, or DiskANN base .bin when --query_file is provided",
    )
    inputs.add_argument("--base-bin", type=str, default="", help="DiskANN .bin file containing base vectors")
    inputs.add_argument("--query-bin", type=str, default="", help="DiskANN .bin file containing query vectors")
    inputs.add_argument("--query-file", "--query_file", dest="query_file", type=str, default="", help="DiskANN query .bin file")
    inputs.add_argument("--label-file", "--label_file", dest="label_file", type=str, required=True, help="DiskANN base labels txt file")
    inputs.add_argument("--filter-label", "--filter_label", dest="filter_label", type=str, default="", help="Single filter label for every query")
    inputs.add_argument("--query-filters-file", "--query_filters_file", dest="query_filters_file", type=str, default="", help="One filter label per query")
    inputs.add_argument("--gt-file", "--gt_file", dest="gt_file", type=str, default="", help="Filtered groundtruth file")
    inputs.add_argument("--compute-gt", action="store_true", help="Compute filtered groundtruth before searching")
    inputs.add_argument("--max-train", type=int, default=None)
    inputs.add_argument("--max-query", type=int, default=None)

    paths = parser.add_argument_group("paths")
    paths.add_argument("--metrics-output", "--metrics_output", dest="metrics_output", type=str, required=True)
    paths.add_argument("--index-path-prefix", "--index_path_prefix", dest="index_path_prefix", type=str, default="")
    paths.add_argument("--result-path", "--result_path", dest="result_path", type=str, default="")
    paths.add_argument("--index-cache-dir", "--index_cache_dir", dest="index_cache_dir", type=str, default=str(repo_root / "results" / "diskann" / "indexes"))
    paths.add_argument("--work-dir", "--work_dir", dest="work_dir", type=str, default=str(repo_root / "results" / "diskann" / "work"))
    paths.add_argument("--result-dir", "--result_dir", dest="result_dir", type=str, default=str(repo_root / "results" / "diskann" / "search"))
    paths.add_argument("--gt-cache-dir", "--gt_cache_dir", dest="gt_cache_dir", type=str, default=str(repo_root / "results" / "diskann" / "groundtruth"))
    paths.add_argument("--build-memory-index", "--build_memory_index", dest="build_memory_index", type=str, default=str(repo_root / "DiskANN" / "build" / "apps" / "build_memory_index"))
    paths.add_argument("--search-memory-index", "--search_memory_index", dest="search_memory_index", type=str, default=str(repo_root / "DiskANN" / "build" / "apps" / "search_memory_index"))
    paths.add_argument(
        "--compute-groundtruth-for-filters",
        "--compute_groundtruth_for_filters",
        dest="compute_groundtruth_for_filters",
        type=str,
        default=str(repo_root / "DiskANN" / "build" / "apps" / "utils" / "compute_groundtruth_for_filters"),
    )

    build = parser.add_argument_group("filtered-vamana build")
    build.add_argument("--data-type", "--data_type", dest="data_type", type=str, default="float", choices=["float", "int8", "uint8"])
    build.add_argument("--dist-fn", "--dist_fn", dest="dist_fn", type=str, default="l2", choices=["l2", "mips", "cosine"])
    build.add_argument("--R", type=int, required=True)
    build.add_argument("--Lbuild", type=int, required=True)
    build.add_argument("--filtered-Lbuild", "--FilteredLbuild", dest="filtered_Lbuild", type=int, required=True)
    build.add_argument("--alpha", type=float, default=1.2)
    build.add_argument("--build-pq-bytes", "--build_PQ_bytes", dest="build_pq_bytes", type=int, default=0)
    build.add_argument("--use-opq", action="store_true")
    build.add_argument("--universal-label", "--universal_label", dest="universal_label", type=str, default="")
    build.add_argument("--label-type", "--label_type", dest="label_type", type=str, default="uint", choices=["uint", "ushort"])

    search = parser.add_argument_group("search")
    search.add_argument("--search-list", "--search_list", dest="search_list", type=int, nargs="+", required=True)
    search.add_argument("--k", type=int, default=10)
    search.add_argument("--gt-k", "--gt_k", dest="gt_k", type=int, default=100)
    search.add_argument("--num-threads", "--num_threads", dest="num_threads", type=int, default=1)
    search.add_argument("--search-num-threads", dest="search_num_threads", type=int, default=None,
                        help="Threads for search only (default: same as --num-threads)")
    search.add_argument("--print-all-recalls", "--print_all_recalls", dest="print_all_recalls", action="store_true")
    search.add_argument("--print-qps-per-thread", "--print_qps_per_thread", dest="print_qps_per_thread", action="store_true")
    search.add_argument("--fail-if-recall-below", "--fail_if_recall_below", dest="fail_if_recall_below", type=float, default=0.0)
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
        print(f"Filtered in-memory DiskANN benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

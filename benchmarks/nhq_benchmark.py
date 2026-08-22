#!/usr/bin/env python3
"""NHQ (Native Hybrid Query) benchmark wrapper.

Wraps the NHQ-NPG_nsw C++ binaries (index + search) and produces a
metrics.json file consumable by the tuning pipeline.

Usage (standalone smoke test):
  python benchmarks/nhq_benchmark.py \
      --M 64 --ef-construction 200 --ef 400 --weight 0 \
      --data-file ~/data/nhq_data/sift_sel05/base.fvecs \
      --attr-file ~/data/nhq_data/sift_sel05/base_attr.txt \
      --query-file ~/data/nhq_data/sift_sel05/query.fvecs \
      --query-attr-file ~/data/nhq_data/sift_sel05/query_attr.txt \
      --gt-file ~/data/nhq_data/sift_sel05/groundtruth.ivecs \
      --metric L2 \
      --metrics-output /tmp/nhq_metrics.json
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return stem.strip("._") or "dataset"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


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


def _resolve_executable(path_text: str, name: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} binary does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} path is not a file: {path}")
    return path


def _run_command(cmd: List[str], description: str) -> Tuple[str, float]:
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
    return proc.stdout, proc.stderr, elapsed


def _read_fvecs_metadata(path: Path) -> Tuple[int, int]:
    """Read (num, dim) from fvecs file header."""
    with path.open("rb") as f:
        dim = struct.unpack_from("<i", f.read(4), 0)[0]
    record_size = 4 + dim * 4
    file_size = path.stat().st_size
    num = file_size // record_size
    return num, dim


# ---------------------------------------------------------------------------
# Index fingerprint & caching
# ---------------------------------------------------------------------------

def _build_fingerprint(
    data_file: Path,
    attr_file: Path,
    metric: str,
    M: int,
    ef_construction: int,
) -> Dict[str, Any]:
    return {
        "benchmark": "nhq",
        "metric": metric,
        "data_file": _file_fingerprint(data_file),
        "attr_file": _file_fingerprint(attr_file),
        "M": int(M),
        "efConstruction": int(ef_construction),
    }


def _index_cache_paths(
    index_cache_dir: Path,
    fingerprint: Dict[str, Any],
    source_name: str,
    M: int,
    ef_construction: int,
) -> Tuple[Path, Path]:
    digest = _sha1_text(json.dumps(fingerprint, sort_keys=True))[:12]
    prefix = index_cache_dir / f"nhq_{_safe_stem(source_name)}_{digest}_M{int(M)}_EFC{int(ef_construction)}"
    meta_path = Path(str(prefix) + ".meta.json")
    return prefix, meta_path


def _index_graph_exists(prefix: Path) -> bool:
    graph = Path(str(prefix) + ".graph")
    table = Path(str(prefix) + ".attrtable")
    return graph.is_file() and graph.stat().st_size > 0 and table.is_file() and table.stat().st_size > 0


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def _build_index(
    index_binary: Path,
    data_file: Path,
    attr_file: Path,
    metric: str,
    M: int,
    ef_construction: int,
    index_prefix: Path,
    meta_path: Path,
    fingerprint: Dict[str, Any],
) -> Tuple[bool, float, float]:
    """Build NHQ index. Returns (index_reused, build_time_s, original_build_time_s)."""
    meta = _read_json(meta_path)
    original_build_time_s = float(meta.get("original_build_time_s", 0.0) or 0.0)
    if meta.get("fingerprint") == fingerprint and original_build_time_s > 0.0 and _index_graph_exists(index_prefix):
        return True, 0.0, original_build_time_s

    graph_path = str(index_prefix) + ".graph"
    table_path = str(index_prefix) + ".attrtable"

    cmd = [
        str(index_binary),
        str(data_file),
        str(attr_file),
        graph_path,
        table_path,
        str(int(M)),
        str(int(ef_construction)),
        metric,
    ]

    stdout, stderr, build_time_s = _run_command(cmd, "NHQ index build")

    # Parse build time from stdout: "Build time: <float>"
    build_match = re.search(r"Build time:\s*([\d.e+-]+)", stdout)
    if build_match:
        build_time_s = float(build_match.group(1))

    _write_json(
        meta_path,
        {
            "fingerprint": fingerprint,
            "index_path_prefix": str(index_prefix),
            "original_build_time_s": float(build_time_s),
        },
    )
    return False, build_time_s, build_time_s


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_NHQ_METRICS_RE = re.compile(
    r"Search Time:\s*([\d.e+-]+)\s+\d+NN accuracy:\s*([\d.e+-]+)\s+Distcount:\s*(\d+)"
)


def _search_index(
    search_binary: Path,
    graph_file: Path,
    table_file: Path,
    query_file: Path,
    gt_file: Path,
    query_attr_file: Path,
    ef: int,
    weight: int,
) -> Tuple[Dict[str, Any], float]:
    """Run NHQ search. Returns (metrics_dict, search_time_s)."""

    cmd = [
        str(search_binary),
        str(graph_file),
        str(table_file),
        str(query_file),
        str(gt_file),
        str(query_attr_file),
        str(int(ef)),
        str(int(weight)),
    ]

    stdout, stderr, wall_time_s = _run_command(cmd, "NHQ search")

    # Check for attribute table mismatch
    if "wrong attributes" in stdout.lower():
        raise RuntimeError(
            f"NHQ search: attribute table mismatch. The query attributes don't match "
            f"the attribute table built during indexing.\nstdout:\n{stdout}"
        )

    # Parse metrics from stderr
    match = _NHQ_METRICS_RE.search(stderr)
    if not match:
        # Also try stdout
        match = _NHQ_METRICS_RE.search(stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse NHQ search metrics from output.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    search_time_s = float(match.group(1))
    accuracy = float(match.group(2))
    distcount = int(match.group(3))

    return {
        "recall": accuracy,
        "search_time_s": search_time_s,
        "distcount": distcount,
    }, wall_time_s


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    # Resolve paths
    data_file = Path(args.data_file).expanduser()
    attr_file = Path(args.attr_file).expanduser()
    query_file = Path(args.query_file).expanduser()
    query_attr_file = Path(args.query_attr_file).expanduser()
    gt_file = Path(args.gt_file).expanduser()
    index_binary = _resolve_executable(args.index_binary, "index")
    search_binary = _resolve_executable(args.search_binary, "search")

    # Validate input files
    for path, name in [
        (data_file, "data_file"),
        (attr_file, "attr_file"),
        (query_file, "query_file"),
        (query_attr_file, "query_attr_file"),
        (gt_file, "gt_file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"--{name} does not exist: {path}")

    metric = args.metric
    M = int(args.M)
    ef_construction = int(args.ef_construction)
    ef = int(args.ef)
    weight = int(args.weight)

    # Read metadata
    n_base, dim = _read_fvecs_metadata(data_file)
    n_query, query_dim = _read_fvecs_metadata(query_file)
    if dim != query_dim:
        raise ValueError(f"base/query dimensions differ: base={dim}, query={query_dim}")

    # Index caching
    fingerprint = _build_fingerprint(data_file, attr_file, metric, M, ef_construction)
    source_name = data_file.stem
    index_cache_dir = Path(args.index_cache_dir).expanduser()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    index_prefix, meta_path = _index_cache_paths(index_cache_dir, fingerprint, source_name, M, ef_construction)

    index_reused, build_time_s, original_build_time_s = _build_index(
        index_binary=index_binary,
        data_file=data_file,
        attr_file=attr_file,
        metric=metric,
        M=M,
        ef_construction=ef_construction,
        index_prefix=index_prefix,
        meta_path=meta_path,
        fingerprint=fingerprint,
    )

    graph_file = Path(str(index_prefix) + ".graph")
    table_file = Path(str(index_prefix) + ".attrtable")

    search_metrics, _ = _search_index(
        search_binary=search_binary,
        graph_file=graph_file,
        table_file=table_file,
        query_file=query_file,
        gt_file=gt_file,
        query_attr_file=query_attr_file,
        ef=ef,
        weight=weight,
    )

    recall = search_metrics["recall"]
    search_time = search_metrics["search_time_s"]
    distcount = search_metrics["distcount"]

    qps = round(n_query / search_time, 3) if search_time > 0 else 0.0

    per_search_list = [{
        "search_list": ef,
        "recall": recall,
        "qps": qps,
        "dist_comps_per_query": round(distcount / n_query, 1) if n_query > 0 else 0.0,
    }]

    return {
        "recall": recall,
        "qps": qps,
        "search_time_s": float(search_time),
        "build_time_s": float(build_time_s),
        "latency_mean_us": float(search_time / n_query * 1e6) if n_query > 0 else 0.0,
        "distcount": int(distcount),
        "dist_comps_per_query": round(distcount / n_query, 1) if n_query > 0 else 0.0,
        "index_reused": bool(index_reused),
        "original_build_time_s": float(original_build_time_s),
        "index_path_prefix": str(index_prefix),
        "base_bin": str(data_file),
        "query_bin": str(query_file),
        "gt_file": str(gt_file),
        "attr_file": str(attr_file),
        "query_attr_file": str(query_attr_file),
        "base_count": int(n_base),
        "query_count": int(n_query),
        "best_search_list": int(ef),
        "selected_ef": int(ef),
        "per_search_list": per_search_list,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run an NHQ (Native Hybrid Query) benchmark.")

    # Tunable params (mapped from pipeline config param_args)
    tunable = parser.add_argument_group("tunable parameters")
    tunable.add_argument("--M", type=int, required=True, help="Max graph degree (MaxM0)")
    tunable.add_argument("--ef-construction", "--ef_construction", dest="ef_construction",
                         type=int, required=True, help="Build-time beam width")
    tunable.add_argument("--ef", type=int, required=True, help="Search-time beam width (halved internally)")
    tunable.add_argument("--weight", type=int, required=True, help="Attribute-mismatch penalty weight")

    # Data paths
    data = parser.add_argument_group("data paths")
    data.add_argument("--data-file", "--data_file", dest="data_file", type=str, required=True,
                      help="Base vectors (fvecs format)")
    data.add_argument("--attr-file", "--attr_file", dest="attr_file", type=str, required=True,
                      help="Base attribute text file")
    data.add_argument("--query-file", "--query_file", dest="query_file", type=str, required=True,
                      help="Query vectors (fvecs format)")
    data.add_argument("--query-attr-file", "--query_attr_file", dest="query_attr_file", type=str, required=True,
                      help="Query attribute text file")
    data.add_argument("--gt-file", "--gt_file", dest="gt_file", type=str, required=True,
                      help="Ground truth (ivecs format)")
    data.add_argument("--metric", type=str, default="L2", choices=["L2", "euclidean", "angular"],
                      help="Distance metric (default: L2)")

    # Binary paths
    binaries = parser.add_argument_group("binary paths")
    binaries.add_argument("--index-binary", "--index_binary", dest="index_binary", type=str,
                          default=str(repo_root / "NHQ" / "NHQ-NPG_nsw" / "examples" / "cpp" / "index"),
                          help="Path to NHQ index build binary")
    binaries.add_argument("--search-binary", "--search_binary", dest="search_binary", type=str,
                          default=str(repo_root / "NHQ" / "NHQ-NPG_nsw" / "examples" / "cpp" / "search"),
                          help="Path to NHQ search binary")

    # Output
    output = parser.add_argument_group("output")
    output.add_argument("--metrics-output", "--metrics_output", dest="metrics_output", type=str, required=True,
                        help="Path to write metrics.json")
    output.add_argument("--index-cache-dir", "--index_cache_dir", dest="index_cache_dir", type=str,
                        default=str(repo_root / "results" / "nhq" / "indexes"),
                        help="Directory for cached index files")

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
        print(f"NHQ benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

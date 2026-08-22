import argparse
import json
import os
from typing import Any


def _load_data_generator():
    try:
        from benchmark.utils import DataGenerator
    except ModuleNotFoundError as exc:
        if exc.name not in {"benchmark", "benchmark.utils"}:
            raise
        from data_generation.benchmark.utils import DataGenerator
    return DataGenerator


def generate_filter_diskann_data(args: argparse.Namespace) -> dict[str, Any]:
    DataGenerator = _load_data_generator()
    output_dir_arg = getattr(args, "output_dir", None) or getattr(args, "output", None)
    if not output_dir_arg:
        raise ValueError("--output_dir is required")
    dist_fn = getattr(args, "dist_fn", None) or getattr(args, "metric", None) or "l2"
    output_dir = os.path.expanduser(output_dir_arg)
    data_root = os.path.dirname(os.path.abspath(output_dir)) or "."
    generator = DataGenerator(data_root, num_threads=getattr(args, "num_threads", 10))
    return generator.run_filter_diskann(
        args.input_hdf5,
        output_dir,
        args.selectivity,
        k=args.k,
        nq=args.nq,
        seed=args.seed,
        dist_fn=dist_fn,
        diskann_path=getattr(args, "diskann_path", None),
        base_file=getattr(args, "base_file", None),
        query_file=getattr(args, "query_file", None),
        gt_file=getattr(args, "gt_file", None),
        label_file=getattr(args, "label_file", None),
        filter_label=getattr(args, "filter_label", None),
        universal_label=getattr(args, "universal_label", None),
        overwrite=getattr(args, "overwrite", False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Filtered-DiskANN data files with controlled attribute selectivity."
    )
    parser.add_argument("--input_hdf5", required=True, help="Input HDF5 containing train/test datasets.")
    parser.add_argument(
        "--diskann_path",
        required=True,
        help="DiskANN repository root containing build/apps/utils/compute_groundtruth_for_filters.",
    )
    parser.add_argument(
        "--output_dir",
        "--output",
        dest="output_dir",
        required=True,
        help="Output directory for DiskANN files.",
    )
    parser.add_argument(
        "--selectivity",
        type=float,
        nargs="+",
        required=True,
        help="One or more target attribute selectivities in (0, 1].",
    )
    parser.add_argument("--k", type=int, default=100, help="Filtered groundtruth K.")
    parser.add_argument("--nq", type=int, default=1000, help="Number of queries to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label assignment.")
    parser.add_argument(
        "--dist_fn",
        "--metric",
        dest="dist_fn",
        choices=["l2", "mips", "cosine"],
        default="l2",
        help="Distance function passed to DiskANN compute_groundtruth_for_filters.",
    )
    parser.add_argument("--base_file", default=None, help="Output DiskANN base .bin path. Defaults to output_dir/base.bin.")
    parser.add_argument(
        "--query_file",
        default=None,
        help="Output DiskANN query .bin path. Defaults to output_dir/query.bin.",
    )
    parser.add_argument(
        "--gt_file",
        default=None,
        help="Output DiskANN groundtruth .bin path. Defaults to selectivity_<s>/groundtruth.bin.",
    )
    parser.add_argument(
        "--label_file",
        default=None,
        help="Output DiskANN base labels file. Defaults to selectivity_<s>/base_labels.txt.",
    )
    parser.add_argument("--filter_label", required=True, help="Filter label assigned to selected base points and queries.")
    parser.add_argument(
        "--universal_label",
        required=True,
        help="Universal label assigned to unselected base points and passed to DiskANN.",
    )
    parser.add_argument(
        "--num_threads",
        "--num-threads",
        dest="num_threads",
        type=int,
        default=10,
        help="Number of query-level worker threads for filtered groundtruth.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = generate_filter_diskann_data(args)
    except Exception as exc:
        print(f"FilterDiskANN data generation failed: {exc}")
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

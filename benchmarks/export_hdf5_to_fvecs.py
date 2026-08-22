import argparse
from pathlib import Path


def _load_dependencies():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required. Install with: pip install h5py") from exc
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required. Install with: pip install numpy") from exc
    return h5py, np


def _write_fvecs(path: Path, array, np) -> None:
    if array.ndim != 2:
        raise ValueError(f"fvecs export requires a 2D array, got shape={array.shape}")
    dim = int(array.shape[1])
    header = np.full((array.shape[0], 1), dim, dtype=np.int32)
    payload = np.concatenate([header.view(np.float32), array.astype(np.float32, copy=False)], axis=1)
    payload.view(np.int32).tofile(path)


def _write_ivecs(path: Path, array, np) -> None:
    if array.ndim != 2:
        raise ValueError(f"ivecs export requires a 2D array, got shape={array.shape}")
    k = int(array.shape[1])
    header = np.full((array.shape[0], 1), k, dtype=np.int32)
    payload = np.concatenate([header, array.astype(np.int32, copy=False)], axis=1)
    payload.tofile(path)


def _write_bvecs(path: Path, array, np) -> None:
    if array.ndim != 2:
        raise ValueError(f"bvecs export requires a 2D array, got shape={array.shape}")
    if not np.all(np.equal(array, np.round(array))):
        raise ValueError("bvecs export requires integer-valued input.")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("bvecs export requires values in [0, 255].")
    dim = int(array.shape[1])
    with path.open("wb") as f:
        for row in array.astype(np.uint8, copy=False):
            f.write(np.int32(dim).tobytes())
            f.write(row.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Export train/test/neighbors from HDF5 to fvecs/ivecs files.")
    parser.add_argument("--input-hdf5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-name", default="sift")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument(
        "--vector-format",
        choices=["fvecs", "bvecs", "both"],
        default="fvecs",
        help="Export train/test vectors as float vectors, byte vectors, or both.",
    )
    args = parser.parse_args()

    h5py, np = _load_dependencies()
    input_path = Path(args.input_hdf5).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as handle:
        train = np.asarray(handle["train"][:], dtype=np.float32)
        test = np.asarray(handle["test"][:], dtype=np.float32)
        neighbors = np.asarray(handle["neighbors"][:], dtype=np.int32)

    if args.max_train is not None:
        train = train[: int(args.max_train)]
    if args.max_query is not None:
        test = test[: int(args.max_query)]
        neighbors = neighbors[: int(args.max_query)]

    base_stem = str(args.base_name).strip() or "sift"
    vector_paths = []
    if args.vector_format in {"fvecs", "both"}:
        base_path = output_dir / f"{base_stem}_base.fvecs"
        query_path = output_dir / f"{base_stem}_query.fvecs"
        _write_fvecs(base_path, train, np)
        _write_fvecs(query_path, test, np)
        vector_paths.extend([base_path, query_path])
    if args.vector_format in {"bvecs", "both"}:
        base_path = output_dir / f"{base_stem}_base.bvecs"
        query_path = output_dir / f"{base_stem}_query.bvecs"
        _write_bvecs(base_path, train, np)
        _write_bvecs(query_path, test, np)
        vector_paths.extend([base_path, query_path])
    gt_path = output_dir / f"{base_stem}_groundtruth.ivecs"
    _write_ivecs(gt_path, neighbors, np)
    for path in vector_paths:
        print(path)
    print(gt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import logging
import os
import subprocess
from glob import glob
from multiprocessing.pool import ThreadPool

import numpy as np
import h5py

logger = logging.getLogger(__name__)

class HDF5Handler:
    """A class to handle reading and writing HDF5 files."""

    @staticmethod
    def read_hdf5_file(file_path):
        """Reads an HDF5 file and returns its contents as a dictionary.

        The HDF5 file from ../data/sift-128-euclidean_with_scalar.hdf5 contains:
            - 'base': vectors used to construct the index
            - 'base_scalars': sequential ids for the BASE vectors
            - 'test': vectors used as queries
            - 'test_hybrid_knn': ground truth hybrid knn for the TEST vectors (corresponding to 'test_ranges')
            - 'test_knn': ground truth knn for the TEST vectors
            - 'test_ranges': range information for the TEST vectors

        Args:
            file_path (str): The path to the HDF5 file.
        Returns:
            dict: A dictionary containing the datasets in the HDF5 file.
        """
        data = {}
        with h5py.File(file_path, 'r') as hdf5_file:
            def recursively_load(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    for key, val in obj.items():
                        recursively_load(f"{name}/{key}", val)

            for key, val in hdf5_file.items():
                recursively_load(key, val)
        return data

    @staticmethod
    def write_hdf5_file(file_path, data):
        """Writes a dictionary to an HDF5 file.

        Args:
            file_path (str): The path to the HDF5 file.
            data (dict): A dictionary containing the datasets to write.
        """
        with h5py.File(file_path, 'w') as hdf5_file:
            def recursively_save(name, obj):
                if isinstance(obj, dict):
                    group = hdf5_file.create_group(name)
                    for key, val in obj.items():
                        recursively_save(f"{name}/{key}", val)
                else:
                    hdf5_file.create_dataset(name, data=obj)

            for key, val in data.items():
                recursively_save(key, val)

    @staticmethod
    def _resolve_single_file(input_dir, explicit_name, pattern, logical_name):
        if explicit_name:
            candidate = explicit_name
            if not os.path.isabs(candidate):
                candidate = os.path.join(input_dir, candidate)
            candidate = os.path.abspath(candidate)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(
                    f"{logical_name} file does not exist: {candidate}"
                )
            return candidate

        matches = sorted(glob(os.path.join(input_dir, pattern)))
        if len(matches) != 1:
            matched = ", ".join(matches) if matches else "none"
            raise ValueError(
                f"Expected exactly one {logical_name} file matching "
                f"{pattern!r} in {input_dir}, found {len(matches)}: {matched}"
            )
        return os.path.abspath(matches[0])

    @staticmethod
    def read_fvecs_file(file_path: str) -> np.ndarray:
        raw = np.fromfile(file_path, dtype=np.int32)
        if raw.size == 0:
            raise ValueError(f"fvecs file is empty: {file_path}")

        dim = int(raw[0])
        if dim <= 0:
            raise ValueError(f"Invalid fvecs dimension header {dim} in {file_path}")

        record_len = dim + 1
        if raw.size % record_len != 0:
            file_size = os.path.getsize(file_path)
            raise ValueError(
                f"Corrupted fvecs file {file_path}: {raw.size} int32 values "
                f"(bytes={file_size}) is not divisible by record length {record_len}"
            )

        records = raw.reshape(-1, record_len)
        dim_headers = records[:, 0]
        invalid_idx = np.where(dim_headers != dim)[0]
        if invalid_idx.size:
            first_bad = int(invalid_idx[0])
            got_dim = int(dim_headers[first_bad])
            raise ValueError(
                f"Inconsistent fvecs headers in {file_path}: row {first_bad} "
                f"has dim={got_dim}, expected {dim}"
            )

        vectors = records.view(np.float32)[:, 1:]
        return vectors.astype(np.float32, copy=False)

    @staticmethod
    def read_ivecs_file(file_path: str) -> np.ndarray:
        raw = np.fromfile(file_path, dtype=np.int32)
        if raw.size == 0:
            raise ValueError(f"ivecs file is empty: {file_path}")

        k = int(raw[0])
        if k <= 0:
            raise ValueError(f"Invalid ivecs dimension header {k} in {file_path}")

        record_len = k + 1
        if raw.size % record_len != 0:
            file_size = os.path.getsize(file_path)
            raise ValueError(
                f"Corrupted ivecs file {file_path}: {raw.size} int32 values "
                f"(bytes={file_size}) is not divisible by record length {record_len}"
            )

        records = raw.reshape(-1, record_len)
        dim_headers = records[:, 0]
        invalid_idx = np.where(dim_headers != k)[0]
        if invalid_idx.size:
            first_bad = int(invalid_idx[0])
            got_k = int(dim_headers[first_bad])
            raise ValueError(
                f"Inconsistent ivecs headers in {file_path}: row {first_bad} "
                f"has k={got_k}, expected {k}"
            )

        vectors = records[:, 1:].astype(np.int64, copy=False)
        return vectors

    @staticmethod
    def convert_folder_to_hdf5(
        input_dir: str,
        output_hdf5: str,
        *,
        base_file: str | None = None,
        query_file: str | None = None,
        groundtruth_file: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        input_dir = os.path.abspath(input_dir)
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

        output_hdf5 = os.path.abspath(output_hdf5)
        if os.path.exists(output_hdf5) and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_hdf5}. "
                "Use overwrite=True to replace it."
            )

        base_path = HDF5Handler._resolve_single_file(
            input_dir, base_file, "*base.fvecs", "base"
        )
        query_path = HDF5Handler._resolve_single_file(
            input_dir, query_file, "*query.fvecs", "query"
        )
        groundtruth_path = HDF5Handler._resolve_single_file(
            input_dir, groundtruth_file, "*groundtruth.ivecs", "groundtruth"
        )

        train = HDF5Handler.read_fvecs_file(base_path)
        test = HDF5Handler.read_fvecs_file(query_path)
        neighbors = HDF5Handler.read_ivecs_file(groundtruth_path)

        if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
            raise ValueError(
                "Converted arrays must be 2D: "
                f"train.ndim={train.ndim}, test.ndim={test.ndim}, neighbors.ndim={neighbors.ndim}"
            )

        if train.shape[1] != test.shape[1]:
            raise ValueError(
                f"Dimension mismatch between base and query: "
                f"train.shape={train.shape}, test.shape={test.shape}"
            )

        if neighbors.shape[0] != test.shape[0]:
            raise ValueError(
                f"Row mismatch between groundtruth and query: "
                f"neighbors.shape={neighbors.shape}, test.shape={test.shape}"
            )

        HDF5Handler.write_hdf5_file(
            output_hdf5,
            {
                "train": train,
                "test": test,
                "neighbors": neighbors,
            },
        )

        summary = {
            "output_path": output_hdf5,
            "train_shape": tuple(train.shape),
            "test_shape": tuple(test.shape),
            "neighbors_shape": tuple(neighbors.shape),
            "dtypes": {
                "train": str(train.dtype),
                "test": str(test.dtype),
                "neighbors": str(neighbors.dtype),
            },
        }
        logger.info("Converted folder %s to %s", input_dir, output_hdf5)
        logger.info("Conversion summary: %s", summary)
        return summary

class RangeHandler:
    """A class to handle generation, saving, and loading of index ranges."""

    @staticmethod
    def generate_test_ranges(
        n,
        min_value,
        max_value,
        selectivity,
        *,
        rng=None,
        dtype="int64",
    ):
        """Generate random test ranges based on selectivity (closed interval).

        Assumes scalar values are consecutive and the domain is [min_value, max_value]
        (both ends inclusive). The returned range is [start, end] (end inclusive).

        Args:
            n (int): Number of unique ranges to generate.
            min_value (int): Inclusive minimum scalar value.
            max_value (int): Inclusive maximum scalar value.
            selectivity (float): Fixed selectivity s in (0, 1]. Selectivity is
                interpreted as covered fraction of the domain cardinality
                (max_value - min_value + 1).
            rng (np.random.Generator | None): RNG for reproducibility.
            dtype (str | np.dtype): Output dtype.

        Returns:
            np.ndarray: shape (n, 2), each row is [start, end] (both inclusive).
        """

        if rng is None:
            rng = np.random.default_rng()

        if n <= 0:
            return np.empty((0, 2), dtype=dtype)

        min_v = int(min_value)
        max_v = int(max_value)
        domain_count = max_v - min_v + 1    # The number of distinct values
        if domain_count <= 0:
            raise ValueError(
                f"Invalid domain: require max_value >= min_value, got {min_value=} {max_value=}"
            )

        s = float(selectivity)
        if not np.isfinite(s):
            raise ValueError("selectivity must be finite")
        if s <= 0 or s > 1:
            raise ValueError("selectivity must be in (0, 1]")

        # Convert selectivity to inclusive sizes (#elements covered)
        size_count_value = int(
            np.clip(np.rint(s * domain_count).astype(np.int64), 1, domain_count)
        )

        # Sample start so that end = start + size_count - 1 never exceeds max_value.
        max_start_offset = domain_count - size_count_value
        possible_count = max_start_offset + 1
        if n > possible_count:
            raise ValueError(
                f"Requested {n} unique ranges but only {possible_count} distinct ranges exist"
            )

        offsets = rng.permutation(possible_count)[:n]  # distinct start offsets
        start = offsets + min_v
        end = start + size_count_value - 1

        ranges = np.empty((n, 2), dtype=dtype)
        ranges[:, 0] = start
        ranges[:, 1] = end
        return ranges

    @staticmethod
    def generate_fraction_ranges(
        n,
        min_value,
        max_value,
        fraction,
        *,
        rng=None,
        dtype="int64",
    ):
        if not fraction == 17:
            selectivity = 2 ** (-fraction) 
            return RangeHandler.generate_test_ranges(
                n,
                min_value,
                max_value,
                selectivity,
                rng=rng,
                dtype=dtype,
            )
        
        # When fraction is 17, we have to generate ranges with varying selectivities
        ranges = []
        frac = 1
        for _ in range(n):
            selectivity = 2 ** (-frac)
            range_ = RangeHandler.generate_test_ranges(
                1,
                min_value,
                max_value,
                selectivity,
                rng=rng,
                dtype=dtype,
            )
            ranges.append(range_[0])
            frac += 1
            if frac > 10:
                frac = 1

        return np.array(ranges, dtype=dtype)

    @staticmethod
    def save_ranges_to_bin(ranges, out_dir, basename="ranges"):
        """Persist ranges to a binary file (little-endian 4-byte ints per bound).

        Args:
            ranges (array-like): Shape (n, 2) of [start, end] pairs.
            out_dir (str): Directory to write the file.
            basename (str): Base filename (without extension) for the .bin output.
        """

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{basename}.bin")
        arr = np.asarray(ranges, dtype=np.int64)
        with open(out_file, "wb") as fh:
            for l, r in arr:
                fh.write(int(l).to_bytes(4, "little", signed=True))
                fh.write(int(r).to_bytes(4, "little", signed=True))
        logger.info("index ranges saved to %s", out_file)

    @staticmethod
    def load_ranges_from_bin(file_path, *, dtype="int64"):
        """Load ranges from a binary file written by save_ranges_to_bin."""

        data = np.fromfile(file_path, dtype="<i4")
        if data.size % 2 != 0:
            raise ValueError("Corrupted ranges file: element count is not even")
        ranges = data.reshape(-1, 2).astype(dtype, copy=False)
        return ranges


class AttributeHandler:
    """Generate DiskANN filter labels and filtered groundtruth data."""

    def __init__(self, *, num_threads=10, rng=None):
        self.num_threads = max(1, int(num_threads))
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def _format_selectivity(value: float) -> str:
        text = f"{value:.6f}".replace(".", "p")
        return text.rstrip("0").rstrip("p") if "p" in text else text

    @classmethod
    def _selectivity_label(cls, value: float) -> str:
        return f"sel_{cls._format_selectivity(value)}"

    @classmethod
    def _selectivity_dir_name(cls, value: float) -> str:
        return f"selectivity_{cls._format_selectivity(value)}"

    @staticmethod
    def _validate_selectivities(selectivities):
        values = [float(value) for value in selectivities]
        if not values:
            raise ValueError("--selectivity must contain at least one value")
        for value in values:
            if not np.isfinite(value):
                raise ValueError(f"selectivity must be finite, got {value}")
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"selectivity must be in (0, 1], got {value}")
        return values

    @staticmethod
    def _prepare_output_dir(path, overwrite: bool):
        if os.path.exists(path) and not os.path.isdir(path):
            raise NotADirectoryError(f"Output path exists and is not a directory: {path}")
        if os.path.isdir(path) and os.listdir(path) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite to replace generated files."
            )
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _ensure_parent_dir(path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _write_diskann_vector_bin(path, vectors):
        arr = np.asarray(vectors, dtype=np.float32, order="C")
        if arr.ndim != 2:
            raise ValueError(f"DiskANN vector data must be 2D, got shape={arr.shape}")
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError(f"DiskANN vector data must be non-empty, got shape={arr.shape}")

        AttributeHandler._ensure_parent_dir(path)
        with open(path, "wb") as handle:
            np.asarray([arr.shape[0], arr.shape[1]], dtype=np.int32).tofile(handle)
            arr.tofile(handle)

    @staticmethod
    def _write_lines(path, values):
        AttributeHandler._ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(values) + "\n")

    @staticmethod
    def _resolve_output_path(path, default_path):
        resolved = default_path if path is None else path
        return os.path.realpath(os.path.abspath(os.path.expanduser(str(resolved))))

    @staticmethod
    def _resolve_diskann_groundtruth_exe(diskann_path):
        if not diskann_path:
            raise ValueError("--diskann_path is required to generate filtered DiskANN groundtruth")
        diskann_root = os.path.realpath(os.path.abspath(os.path.expanduser(str(diskann_path))))
        exe_path = os.path.join(diskann_root, "build", "apps", "utils", "compute_groundtruth_for_filters")
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(
                f"DiskANN filtered groundtruth tool not found: {exe_path}. "
                "Build DiskANN first or pass the correct --diskann_path."
            )
        if not os.access(exe_path, os.X_OK):
            raise PermissionError(f"DiskANN filtered groundtruth tool is not executable: {exe_path}")
        return diskann_root, exe_path

    @staticmethod
    def _read_truthset_header(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DiskANN groundtruth was not created: {path}")
        with open(path, "rb") as handle:
            header = np.fromfile(handle, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Invalid DiskANN groundtruth header in {path}")
        return int(header[0]), int(header[1])

    @staticmethod
    def _run_diskann_groundtruth(
        *,
        diskann_exe,
        dist_fn,
        base_file,
        query_file,
        gt_file,
        k,
        label_file,
        filter_label,
        universal_label,
    ):
        command = [
            diskann_exe,
            "--data_type",
            "float",
            "--dist_fn",
            str(dist_fn),
            "--base_file",
            str(base_file),
            "--query_file",
            str(query_file),
            "--gt_file",
            str(gt_file),
            "--K",
            str(int(k)),
            "--label_file",
            str(label_file),
            "--filter_label",
            str(filter_label),
            "--universal_label",
            str(universal_label),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            details = "\n".join(
                part
                for part in (
                    f"Command: {' '.join(command)}",
                    f"stdout:\n{result.stdout.strip()}" if result.stdout else "",
                    f"stderr:\n{result.stderr.strip()}" if result.stderr else "",
                )
                if part
            )
            raise RuntimeError(f"DiskANN filtered groundtruth command failed with code {result.returncode}\n{details}")
        return command, result.stdout, result.stderr

    @staticmethod
    def _read_hdf5_vectors(input_hdf5, nq):
        input_hdf5 = os.path.expanduser(str(input_hdf5))
        if not os.path.exists(input_hdf5):
            raise FileNotFoundError(f"Input HDF5 does not exist: {input_hdf5}")

        data = HDF5Handler.read_hdf5_file(input_hdf5)
        try:
            train = data["train"]
            test = data["test"]
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing required dataset '{missing}' in {input_hdf5}") from exc

        if train.ndim != 2:
            raise ValueError(f"HDF5 dataset 'train' must be 2D, got shape={train.shape}")
        if test.ndim != 2:
            raise ValueError(f"HDF5 dataset 'test' must be 2D, got shape={test.shape}")
        if train.shape[1] != test.shape[1]:
            raise ValueError(f"train/test dimensions differ: train={train.shape}, test={test.shape}")

        if nq is not None:
            if nq <= 0:
                raise ValueError("--nq must be > 0 when provided")
            test = test[:nq]
        if train.shape[0] == 0:
            raise ValueError("HDF5 dataset 'train' is empty")
        if test.shape[0] == 0:
            raise ValueError("HDF5 dataset 'test' is empty after applying --nq")
        return train, test

    def _generate_one_selectivity(
        self,
        *,
        train,
        test,
        selectivity: float,
        output_dir,
        k: int,
        dist_fn: str,
        seed: int,
        input_hdf5,
        overwrite: bool,
        diskann_root,
        diskann_exe,
        base_file,
        query_file,
        gt_file=None,
        label_file=None,
        filter_label=None,
        universal_label=None,
    ):
        n_base = int(train.shape[0])
        n_query = int(test.shape[0])
        target_count = int(np.rint(selectivity * n_base))
        target_count = int(np.clip(target_count, 1, n_base))
        if n_base < k:
            raise ValueError(
                f"base dataset has {n_base} points, which is smaller than k={k}. Lower --k."
            )

        label = str(filter_label) if filter_label is not None else self._selectivity_label(selectivity)
        unselected_label = str(universal_label) if universal_label is not None else "background"
        selectivity_dir = os.path.join(output_dir, self._selectivity_dir_name(selectivity))
        if os.path.isdir(selectivity_dir) and os.listdir(selectivity_dir) and not overwrite:
            raise FileExistsError(f"Selectivity output directory is not empty: {selectivity_dir}. Use --overwrite.")
        os.makedirs(selectivity_dir, exist_ok=True)

        rng = np.random.default_rng(seed)
        selected_ids = np.sort(rng.choice(n_base, size=target_count, replace=False))
        labels = np.full(n_base, unselected_label, dtype=object)
        labels[selected_ids] = label

        base_labels_path = self._resolve_output_path(label_file, os.path.join(selectivity_dir, "base_labels.txt"))
        query_filters_path = os.path.join(selectivity_dir, "query_filters.txt")
        gt_path = self._resolve_output_path(gt_file, os.path.join(selectivity_dir, "groundtruth.bin"))
        manifest_path = os.path.join(selectivity_dir, "manifest.json")

        self._write_lines(base_labels_path, labels.astype(str).tolist())
        self._write_lines(query_filters_path, [label] * n_query)
        command, stdout, stderr = self._run_diskann_groundtruth(
            diskann_exe=diskann_exe,
            dist_fn=dist_fn,
            base_file=base_file,
            query_file=query_file,
            gt_file=gt_path,
            k=k,
            label_file=base_labels_path,
            filter_label=label,
            universal_label=unselected_label,
        )
        gt_queries, gt_k = self._read_truthset_header(gt_path)
        if gt_queries != n_query or gt_k != int(k):
            raise ValueError(
                f"DiskANN groundtruth header mismatch in {gt_path}: "
                f"got [{gt_queries}, {gt_k}], expected [{n_query}, {int(k)}]"
            )

        actual_selectivity = target_count / float(n_base)
        manifest = {
            "input_hdf5": os.path.abspath(input_hdf5),
            "diskann_path": str(diskann_root),
            "diskann_groundtruth_exe": str(diskann_exe),
            "groundtruth_command": command,
            "groundtruth_stdout": stdout,
            "groundtruth_stderr": stderr,
            "target_selectivity": float(selectivity),
            "actual_selectivity": float(actual_selectivity),
            "target_count": int(target_count),
            "base_count": int(n_base),
            "query_count": int(n_query),
            "dimension": int(train.shape[1]),
            "k": int(k),
            "data_type": "float",
            "dist_fn": dist_fn,
            "metric": dist_fn,
            "seed": int(seed),
            "target_label": label,
            "filter_label": label,
            "background_label": unselected_label,
            "universal_label": unselected_label,
            "unselected_points_label": unselected_label,
            "unselected_points_use_universal_label": True,
            "base_file": str(base_file),
            "query_file": str(query_file),
            "base_labels": str(base_labels_path),
            "query_filters": str(query_filters_path),
            "groundtruth": str(gt_path),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return manifest

    def generate_filter_diskann_data(
        self,
        input_hdf5,
        output_dir,
        selectivities,
        *,
        k=100,
        nq=1000,
        seed=42,
        metric="l2",
        dist_fn=None,
        diskann_path=None,
        base_file=None,
        query_file=None,
        gt_file=None,
        label_file=None,
        filter_label=None,
        universal_label=None,
        overwrite=False,
    ):
        selectivities = self._validate_selectivities(selectivities)
        if k <= 0:
            raise ValueError("--k must be > 0")
        resolved_dist_fn = str(dist_fn or metric or "l2")
        if resolved_dist_fn not in {"l2", "mips", "cosine"}:
            raise ValueError("--dist_fn must be one of: l2, mips, cosine")
        explicit_filter_config = (
            filter_label is not None
            or universal_label is not None
            or label_file is not None
            or gt_file is not None
        )
        if explicit_filter_config and len(selectivities) != 1:
            raise ValueError(
                "Explicit filter_label/universal_label/label_file/gt_file requires exactly one selectivity"
            )

        input_hdf5 = os.path.realpath(os.path.abspath(os.path.expanduser(str(input_hdf5))))
        output_dir = os.path.realpath(os.path.abspath(os.path.expanduser(str(output_dir))))
        diskann_root, diskann_exe = self._resolve_diskann_groundtruth_exe(diskann_path)
        self._prepare_output_dir(output_dir, overwrite)

        train, test = self._read_hdf5_vectors(input_hdf5, nq)
        base_bin_path = self._resolve_output_path(base_file, os.path.join(output_dir, "base.bin"))
        query_bin_path = self._resolve_output_path(query_file, os.path.join(output_dir, "query.bin"))
        if overwrite or not os.path.exists(base_bin_path):
            self._write_diskann_vector_bin(base_bin_path, train)
        if overwrite or not os.path.exists(query_bin_path):
            self._write_diskann_vector_bin(query_bin_path, test)

        manifests = []
        for offset, selectivity in enumerate(selectivities):
            manifests.append(
                self._generate_one_selectivity(
                    train=train,
                    test=test,
                    selectivity=float(selectivity),
                    output_dir=output_dir,
                    k=int(k),
                    dist_fn=resolved_dist_fn,
                    seed=int(seed) + offset,
                    input_hdf5=input_hdf5,
                    overwrite=bool(overwrite),
                    diskann_root=diskann_root,
                    diskann_exe=diskann_exe,
                    base_file=base_bin_path,
                    query_file=query_bin_path,
                    gt_file=gt_file,
                    label_file=label_file,
                    filter_label=filter_label,
                    universal_label=universal_label,
                )
            )

        summary = {
            "input_hdf5": os.path.abspath(input_hdf5),
            "output_dir": os.path.abspath(output_dir),
            "base_bin": str(base_bin_path),
            "query_bin": str(query_bin_path),
            "base_shape": [int(train.shape[0]), int(train.shape[1])],
            "query_shape": [int(test.shape[0]), int(test.shape[1])],
            "diskann_path": str(diskann_root),
            "dist_fn": resolved_dist_fn,
            "selectivities": manifests,
        }
        with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return summary


class DataGenerator:
    """A class to generate synthetic data for benchmarking."""
    def __init__(
        self,
        data_root,
        *,
        knn_fn=None,
        num_threads=10,
        rng=None,
    ):
        self.data_root = data_root
        self.knn_fn = knn_fn or self._brute_force_knn
        self.num_threads = max(1, int(num_threads))
        self.rng = rng or np.random.default_rng()

    def _cos_normalize(self, x, _mean=None, _std=None, *, eps=1e-12):
        norm = np.linalg.norm(x, axis=1)
        norm.resize((len(norm), 1))
        ret = x / norm
        ret[np.isnan(ret)] = 100
        return ret

    def _brute_force_knn(self, base, query, k):
        if base.size == 0:
            return (
                np.empty((query.shape[0], 0), dtype=np.float32),
                np.empty((query.shape[0], 0), dtype=np.int64),
            )

        k = min(int(k), base.shape[0])
        base = base.astype(np.float32, copy=False)
        query = query.astype(np.float32, copy=False)

        base_sq = np.sum(base ** 2, axis=1)
        query_sq = np.sum(query ** 2, axis=1, keepdims=True)
        distances = query_sq + base_sq[None, :] - 2.0 * np.dot(query, base.T)

        idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(distances.shape[0])[:, None]
        order = np.argsort(distances[row, idx], axis=1)
        I = idx[row, order]
        D = distances[row, I]
        return D, I

    def _bf_hybrid_search(self, thread_id, q, query_range, *, base_scalars, train, k):
        logger.info("Thread %s, searching range %s...", thread_id, query_range)
        low, high = query_range
        mask = (base_scalars >= low) & (base_scalars <= high)
        id_map = np.arange(0, len(train))[mask]
        base = train[mask]
        D, I = self.knn_fn(base, q.reshape(1, len(q)), k)
        logger.info("Thread %s, searching done.", thread_id)
        return D, id_map[I]

    def run(self, dataset, filepath, *, nq=1000, k=100, fraction=None):
        data = HDF5Handler.read_hdf5_file(filepath)
        try:
            train = data["train"]
            test = data["test"]
            test_knn = data["neighbors"]
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Missing required dataset '{missing}' in {filepath}"
            ) from exc

        test = test[:nq]
        test_knn = test_knn[:nq]

        if "angular" in dataset.lower():
            train = self._cos_normalize(train)
            test = self._cos_normalize(test)

        n_train = train.shape[0]
        n_test = test.shape[0]

        scalar_min, scalar_max = 0, n_train
        base_scalars = np.arange(scalar_min, scalar_max, dtype="int64")
        if fraction is None:
            raise ValueError("fraction must be provided")

        test_ranges = RangeHandler.generate_fraction_ranges(
            n_test,
            scalar_min,
            scalar_max,
            fraction,
            rng=self.rng,
            dtype="int64",
        )

        test_hybrid_knn = np.zeros((n_test, k), dtype="uint64")

        pool = ThreadPool(self.num_threads)
        try:
            results = pool.map(
                lambda q: self._bf_hybrid_search(
                    q[0], q[1], q[2], base_scalars=base_scalars, train=train, k=k
                ),
                zip(range(n_test), test, test_ranges),
            )
        finally:
            pool.close()
            pool.join()

        for i, (_D, I) in enumerate(results):
            if I.shape[1] < k:
                padded = np.full((1, k), -1, dtype=I.dtype)
                padded[:, : I.shape[1]] = I
                I = padded
            test_hybrid_knn[i, :] = I

        os.makedirs(self.data_root, exist_ok=True)
        output_path = os.path.join(self.data_root, f"{dataset}_fraction_{fraction}.hdf5")
        HDF5Handler.write_hdf5_file(
            output_path,
            {
                "base": train,
                "base_scalars": base_scalars,
                "test": test,
                "test_ranges": test_ranges,
                "test_knn": test_knn,
                "test_hybrid_knn": test_hybrid_knn,
            },
        )

    def run_filter_diskann(
        self,
        input_hdf5,
        output_dir,
        selectivities,
        *,
        k=100,
        nq=1000,
        seed=42,
        metric="l2",
        dist_fn=None,
        diskann_path=None,
        base_file=None,
        query_file=None,
        gt_file=None,
        label_file=None,
        filter_label=None,
        universal_label=None,
        overwrite=False,
    ):
        handler = AttributeHandler(num_threads=self.num_threads, rng=self.rng)
        return handler.generate_filter_diskann_data(
            input_hdf5,
            output_dir,
            selectivities,
            k=k,
            nq=nq,
            seed=seed,
            metric=metric,
            dist_fn=dist_fn,
            diskann_path=diskann_path,
            base_file=base_file,
            query_file=query_file,
            gt_file=gt_file,
            label_file=label_file,
            filter_label=filter_label,
            universal_label=universal_label,
            overwrite=overwrite,
        )
    

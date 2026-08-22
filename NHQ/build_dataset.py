#!/usr/bin/env python3
"""
build_dataset.py — General-purpose hybrid query dataset constructor.

Converts a vector dataset (HDF5 or raw fvecs) into the binary/text formats
required by NHQ-NPG_nsw, and generates synthetic structured attributes with
controllable selectivity.

Selectivity
-----------
Selectivity is the fraction of the dataset that matches a single attribute-value
filter.  For example, if selectivity = 0.1 and attribute "color" has values
{"red","blue","green",...,"purple"} (10 values), each value is assigned to ~10%
of the objects, so `WHERE color='red'` matches ~10% of the data.

For C attributes, the combined selectivity is roughly selectivity^C.

Usage
-----
  python3 build_dataset.py \
      --input       ~/data/sift-128-euclidean.hdf5 \
      --output      data/my_dataset \
      --selectivity 0.3 \
      --num_attrs   5 \
      --weight      140000
"""

import argparse
import h5py
import numpy as np
import os
import struct
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Build a hybrid-query dataset from vectors",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # From HDF5 (keys: train, test, neighbors)
  python3 build_dataset.py -i data.hdf5 -o out/ -s 0.1 -n 5

  # From raw fvecs file
  python3 build_dataset.py -i base.fvecs -o out/ --query query.fvecs --gt gt.ivecs -s 0.2

  # High selectivity (broad filters)
  python3 build_dataset.py -i data.hdf5 -o out/ -s 0.5 -n 3
""",
)
parser.add_argument("-i", "--input", required=True,
                    help="Path to input: HDF5 file or fvecs binary file")
parser.add_argument("-o", "--output", required=True,
                    help="Output directory")
parser.add_argument("-s", "--selectivity", type=float, default=0.3,
                    help="Per-attribute selectivity: fraction of data "
                         "matching a single attribute value (default: 0.3)")
parser.add_argument("-n", "--num_attrs", type=int, default=5,
                    help="Number of synthetic attribute dimensions (default: 5)")
parser.add_argument("-w", "--weight", type=float, default=140000,
                    help="weight_search for hybrid groundtruth computation "
                         "(default: 140000)")
parser.add_argument("--query", help="Path to query fvecs file (if input is fvecs)")
parser.add_argument("--gt", help="Path to groundtruth ivecs file (if input is fvecs)")
parser.add_argument("--train-rows", type=int, default=0,
                    help="HDF5 dataset key for train vectors (default: train)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
parser.add_argument("--no-hybrid-gt", action="store_true",
                    help="Skip hybrid groundtruth computation (keep pure-L2 GT)")

# ---------------------------------------------------------------------------
# Helpers — binary I/O
# ---------------------------------------------------------------------------

def read_fvecs(path):
    """Read fvecs file → (num, dim, float32 array)."""
    with open(path, "rb") as f:
        data = f.read()
    dim = struct.unpack_from("<i", data, 0)[0]
    record_size = 4 + dim * 4
    num = len(data) // record_size
    arr = np.empty((num, dim), dtype=np.float32)
    for i in range(num):
        off = i * record_size + 4
        arr[i] = np.frombuffer(data, off, dim * 4, dtype="<f4")
    return num, dim, arr


def read_ivecs(path):
    """Read ivecs file → (num, dim, int32 array)."""
    with open(path, "rb") as f:
        data = f.read()
    dim = struct.unpack_from("<i", data, 0)[0]
    record_size = 4 + dim * 4
    num = len(data) // record_size
    arr = np.empty((num, dim), dtype=np.int32)
    for i in range(num):
        off = i * record_size + 4
        arr[i] = np.frombuffer(data, off, dim * 4, dtype="<i4")
    return num, dim, arr


def write_fvecs(path, arr):
    num, dim = arr.shape
    with open(path, "wb") as f:
        for i in range(num):
            f.write(struct.pack("<i", dim))
            f.write(arr[i].astype("<f4").tobytes())
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [fvecs] {path}  ({num}×{dim}, {size_mb:.1f} MB)")


def write_ivecs(path, arr):
    num, dim = arr.shape
    with open(path, "wb") as f:
        for i in range(num):
            f.write(struct.pack("<i", dim))
            f.write(arr[i].astype("<i4").tobytes())
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [ivecs] {path}  ({num}×{dim}, {size_mb:.1f} MB)")


def write_attributes(path, attr):
    """attr: 2D array-like of strings, shape (num, dim)."""
    num, dim = attr.shape
    with open(path, "w") as f:
        f.write(f"{num} {dim}\n")
        for i in range(num):
            f.write(" ".join(str(v) for v in attr[i]) + "\n")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [attr] {path}  ({num}×{dim}, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path, query_path, gt_path):
    """Unified loader: HDF5 or raw fvecs/ivecs."""
    if input_path.endswith(".hdf5") or input_path.endswith(".h5"):
        print(f"[load] HDF5: {input_path}")
        with h5py.File(input_path, "r") as f:
            ks = list(f.keys())
            print(f"       keys: {ks}")
            # Heuristic key matching
            train_key = next((k for k in ks if k.lower() in ("train", "base", "data")), ks[0])
            test_key  = next((k for k in ks if k.lower() in ("test", "query")), None)
            gt_key    = next((k for k in ks if k.lower() in ("neighbors", "groundtruth", "gt", "gnd")), None)

            train = f[train_key][:].astype(np.float32)
            print(f"       train  ← '{train_key}' {train.shape}")
            if test_key:
                test = f[test_key][:].astype(np.float32)
                print(f"       test   ← '{test_key}' {test.shape}")
            else:
                test = None
            if gt_key:
                gt = f[gt_key][:].astype(np.int32)
                print(f"       gt     ← '{gt_key}' {gt.shape}")
            else:
                gt = None
    else:
        print(f"[load] fvecs: {input_path}")
        _, _, train = read_fvecs(input_path)
        test = None
        gt = None
        if query_path:
            _, _, test = read_fvecs(query_path)
        if gt_path:
            _, _, gt = read_ivecs(gt_path)
    return train, test, gt


# ---------------------------------------------------------------------------
# Attribute generation with controlled selectivity
# ---------------------------------------------------------------------------

def generate_attributes(num_objects, selectivity, num_attrs, rng):
    """
    Generate synthetic categorical attributes with controlled per-value
    selectivity.

    Each attribute dimension has K = round(1 / selectivity) distinct values.
    Values are bucket IDs like "a0_0", "a0_1", ..., "a1_0", ...

    Assignment strategy:
      - For each attribute, shuffle object indices and assign values in
        round-robin fashion.  This guarantees every value appears in
        exactly ~(selectivity * num_objects) objects.

    Returns a (num_objects, num_attrs) array of strings.
    """
    n = num_objects
    d = num_attrs
    # Number of distinct values per attribute dimension
    num_values_per_attr = max(2, int(round(1.0 / selectivity)))

    attr = np.empty((n, d), dtype=object)

    for j in range(d):
        values = [f"a{j}_{v}" for v in range(num_values_per_attr)]
        # Shuffle indices, then assign round-robin
        indices = rng.permutation(n)
        col = np.empty(n, dtype=object)
        for idx_pos, obj_idx in enumerate(indices):
            col[obj_idx] = values[idx_pos % num_values_per_attr]
        attr[:, j] = col

    # Report actual selectivity
    counts = defaultdict(int)
    for v in attr[:, 0]:
        counts[v] += 1
    fractions = np.array(list(counts.values())) / n
    actual_sel = fractions.mean()
    print(f"\n[attr] Generated {d} attributes, "
          f"target selectivity={selectivity:.4f}, "
          f"actual per-value selectivity={actual_sel:.4f} (±{fractions.std():.4f})")
    print(f"       Each attr has {num_values_per_attr} values: "
          f"{', '.join(f'a{j}_0..a{j}_{num_values_per_attr-1}' for j in range(min(3, d)))}"
          + ("..." if d > 3 else ""))

    return attr


# ---------------------------------------------------------------------------
# Hybrid groundtruth computation
# ---------------------------------------------------------------------------

def compute_hybrid_groundtruth(train, test, train_attr, test_attr, k, weight):
    """
    Compute hybrid-query groundtruth efficiently using attribute buckets.

    Hybrid distance = L2_dist + weight * attribute_mismatch_count

    Since `weight` is typically large (e.g., 140000), candidates with fewer
    attribute mismatches ALWAYS rank above those with more mismatches,
    regardless of L2 distance.  This means:

        • 0-mismatch candidates always rank above 1-mismatch
        • 1-mismatch always above 2-mismatch, etc.

    Strategy:
      1. Group all training indices by their attribute value tuple.
      2. For each query, probe buckets in increasing Hamming-distance order
         (0 mismatches → 1 mismatch → 2 mismatches → ...) until we collect
         ≥ k candidates.
      3. Compute L2 distances ONLY for the collected candidates, sort, and
         keep top-k.

    Complexity: O(Q * (B + C*D)) where
        B = avg bucket size scanned,
        C = avg candidates per query,
        D = vector dimension.

    Returns (num_queries, k) array of neighbor IDs.
    """
    n_train, dim = train.shape
    n_test = test.shape[0]
    num_attrs = train_attr.shape[1]

    # -- 1. Build bucket index: attribute tuple → list of training indices --
    print(f"\n[hybrid-gt] Building attribute bucket index ...")
    buckets = defaultdict(list)
    for i in range(n_train):
        sig = tuple(train_attr[i])
        buckets[sig].append(i)

    bucket_sizes = np.array([len(v) for v in buckets.values()])
    print(f"            {len(buckets)} unique attribute tuples, "
          f"avg bucket size={bucket_sizes.mean():.0f}, "
          f"min={bucket_sizes.min()}, max={bucket_sizes.max()}")

    # -- 2. For each query, collect candidates & rank by hybrid distance --
    print(f"            Computing hybrid GT (weight={weight}, k={k}) "
          f"for {n_test} queries ...")

    # Pre-compute possible values per attribute dimension for hamming expansion
    attr_values = [sorted(set(train_attr[:, j])) for j in range(num_attrs)]
    for j, vals in enumerate(attr_values):
        print(f"            attr[{j}]: {len(vals)} values {vals[:5]}"
              + ("..." if len(vals) > 5 else ""))

    # -- 2. Precompute contiguous bucket data & L2 norms --
    print(f"            Precomputing bucket data for {len(buckets)} buckets ...")
    train_f64 = train.astype(np.float64)
    bucket_data = {}   # sig → contiguous (n_bucket, dim) float64 array
    bucket_ids = {}    # sig → (n_bucket,) int64 original indices
    bucket_norms = {}  # sig → (n_bucket,) float64 ||v||²
    for sig, idxs in buckets.items():
        arr = np.array(idxs, dtype=np.int64)
        data = train_f64[arr]  # copy once — avoids per-query fancy indexing
        bucket_data[sig] = data
        bucket_ids[sig] = arr
        bucket_norms[sig] = np.sum(data ** 2, axis=1)
    total_gb = sum(d.nbytes for d in bucket_data.values()) / 1e9
    print(f"            Precomputed ({total_gb:.1f} GB total).")

    # Precompute query norms
    query_norms = np.sum(test.astype(np.float64) ** 2, axis=1)  # shape (n_test,)

    # -- 3. For each query, find hybrid top-k --
    gt_indices = np.empty((n_test, k), dtype=np.int32)

    for q_idx in range(n_test):
        q_vec = test[q_idx].astype(np.float64)
        q_sq = query_norms[q_idx]
        q_sig = tuple(test_attr[q_idx])

        result = []  # list of (idx, l2_sq, mismatch_level)
        mismatch_level = 0

        while len(result) < k and mismatch_level <= num_attrs:
            layer_candidates = []
            if mismatch_level == 0:
                layer_candidates = [q_sig]
            else:
                from itertools import combinations, product
                for diff_positions in combinations(range(num_attrs), mismatch_level):
                    alt_values = []
                    for j in diff_positions:
                        alt_values.append([v for v in attr_values[j] if v != q_sig[j]])
                    for alt_vals in product(*alt_values):
                        new_sig = list(q_sig)
                        for pos, val in zip(diff_positions, alt_vals):
                            new_sig[pos] = val
                        layer_candidates.append(tuple(new_sig))

            for sig in layer_candidates:
                if sig not in bucket_data:
                    continue
                cand_data = bucket_data[sig]           # (N, D) — no copy
                cand_ids = bucket_ids[sig]             # (N,) original indices
                cand_norms_sq = bucket_norms[sig]       # (N,)
                # Fast L2 via GEMV: ||q-c||² = ||q||² + ||c||² - 2·q·c
                cross = cand_data @ q_vec                # GEMV (N,)
                l2_sq = q_sq + cand_norms_sq - 2 * cross

                need = k - len(result)
                n_cand = len(cand_data)
                if n_cand > need:
                    top_n = min(n_cand, need)
                    top_idx = np.argpartition(l2_sq, top_n)[:top_n]
                    sort_order = np.argsort(l2_sq[top_idx])
                    for local_i in top_idx[sort_order]:
                        result.append((int(cand_ids[local_i]),
                                       float(np.sqrt(l2_sq[local_i])),
                                       mismatch_level))
                else:
                    for local_i in np.argsort(l2_sq):
                        result.append((int(cand_ids[local_i]),
                                       float(np.sqrt(l2_sq[local_i])),
                                       mismatch_level))

            # Only proceed to higher mismatch levels if 0-mismatch didn't fill k
            if mismatch_level == 0 and len(result) >= k:
                break
            mismatch_level += 1

        # Sort result by (mismatch_level, L2)
        result.sort(key=lambda x: (x[2], x[1]))
        gt_indices[q_idx] = [r[0] for r in result[:k]]

        if (q_idx + 1) % 2000 == 0:
            print(f"  ... {q_idx + 1}/{n_test} queries done")

    print(f"  Done: hybrid GT shape={gt_indices.shape}")
    return gt_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hybrid Query Dataset Builder")
    print(f"  selectivity = {args.selectivity}")
    print(f"  num_attrs   = {args.num_attrs}")
    print(f"  weight      = {args.weight}")
    print("=" * 60)

    rng = np.random.RandomState(args.seed)

    # ---- 1. Load data ----
    train, test, gt_l2 = load_data(args.input, args.query, args.gt)

    n_train, dim = train.shape
    print(f"\n[data] train: {train.shape}, dtype={train.dtype}")
    if test is not None:
        n_test = test.shape[0]
        print(f"       test:  {test.shape}, dtype={test.dtype}")
    if gt_l2 is not None:
        print(f"       gt:    {gt_l2.shape}, dtype={gt_l2.dtype}")

    # ---- 2. Create output dir ----
    os.makedirs(args.output, exist_ok=True)

    # ---- 3. Generate attributes ----
    train_attr = generate_attributes(n_train, args.selectivity,
                                     args.num_attrs, rng)
    if test is not None:
        # Query attributes: use the same value set, randomly assigned
        # (queries represent real filters users would ask)
        query_attr = generate_attributes(n_test, args.selectivity,
                                         args.num_attrs, rng)
    else:
        query_attr = None

    # ---- 4. Write binary files ----
    print(f"\n[write] Output → {args.output}/")
    write_fvecs(os.path.join(args.output, "base.fvecs"), train)
    if test is not None:
        write_fvecs(os.path.join(args.output, "query.fvecs"), test)

    # ---- 5. Write attributes ----
    write_attributes(os.path.join(args.output, "base_attr.txt"), train_attr)
    if query_attr is not None:
        write_attributes(os.path.join(args.output, "query_attr.txt"), query_attr)

    # ---- 6. Groundtruth ----
    if not args.no_hybrid_gt and test is not None:
        k = min(100, n_train)
        if gt_l2 is not None:
            k = min(gt_l2.shape[1], n_train)
        hybrid_gt = compute_hybrid_groundtruth(
            train, test, train_attr, query_attr, k, args.weight)
        write_ivecs(os.path.join(args.output, "groundtruth.ivecs"), hybrid_gt)
    elif gt_l2 is not None:
        write_ivecs(os.path.join(args.output, "groundtruth.ivecs"), gt_l2)
        print("  (using pure-L2 groundtruth from input)")

    # ---- 7. Summary ----
    print(f"\n{'=' * 60}")
    print("Done.  Run NHQ-NPG_nsw:")
    graph   = os.path.join(args.output, "graph.bin")
    table   = os.path.join(args.output, "table.bin")
    print(f"  ./index {args.output}/base.fvecs {args.output}/base_attr.txt "
          f"{graph} {table} <MaxM0> <efConstruction>")
    print(f"  ./search {graph} {table} {args.output}/query.fvecs "
          f"{args.output}/groundtruth.ivecs {args.output}/query_attr.txt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parser.parse_args()
    main()

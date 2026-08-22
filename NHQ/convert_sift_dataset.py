#!/usr/bin/env python3
"""
convert_sift_dataset.py

Convert SIFT-128-Euclidean HDF5 dataset to binary/text formats required by
NHQ-NPG_nsw algorithms.

Input:  ~/data/sift-128-euclidean.hdf5
Output: data/nhq_npg_nsw/  (under current directory)

Output files:
  - sift_base.fvecs          : training vectors in fvecs binary format
  - sift_query.fvecs         : query vectors in fvecs binary format
  - sift_groundtruth.ivecs   : groundtruth neighbor IDs in ivecs binary format
  - sift_attributes.txt      : synthetic structured attributes for base vectors
  - sift_query_attributes.txt: synthetic structured attributes for query vectors

fvecs format (per vector):
  [4 bytes: int32 dimension][dim * 4 bytes: float32 values]

ivecs format (per vector):
  [4 bytes: int32 dimension][dim * 4 bytes: int32 values]

Attributes text format:
  Line 1: <num_vectors> <num_attribute_dims>
  Lines 2+: <attr1> <attr2> ... <attrN>   (space-separated string tokens)
"""

import h5py
import numpy as np
import os
import struct
import sys
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HDF5_PATH    = os.path.expanduser("~/data/sift-128-euclidean.hdf5")
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nhq_npg_nsw")

# Number of synthetic attribute dimensions to generate
NUM_ATTR_DIMS = 10
# Number of clusters for generating categorical attributes from raw vectors
NUM_CLUSTERS  = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_fvecs(path, data):
    """
    Write float32 vectors in fvecs format.
    data shape: (num_vectors, dim)
    """
    num, dim = data.shape
    with open(path, "wb") as f:
        for i in range(num):
            f.write(struct.pack("<i", dim))               # dimension as int32 LE
            f.write(data[i].astype("<f4").tobytes())      # vector as float32 LE
    print(f"  Wrote {path}  ({num} x {dim}, float32)")


def write_ivecs(path, data):
    """
    Write int32 vectors in ivecs format.
    data shape: (num_vectors, dim)
    """
    num, dim = data.shape
    with open(path, "wb") as f:
        for i in range(num):
            f.write(struct.pack("<i", dim))               # dimension as int32 LE
            f.write(data[i].astype("<i4").tobytes())      # vector as int32 LE
    print(f"  Wrote {path}  ({num} x {dim}, int32)")


def write_attributes(path, attr_data):
    """
    Write string-categorical attributes in the text format expected by
    load_data_txt().

    attr_data shape: (num_vectors, num_dims)
    attr_data values are strings (category labels).

    Format:
      <num_vectors> <num_dims>
      <attr0_0> <attr0_1> ... <attr0_{d-1}>
      ...
    """
    num, dim = attr_data.shape
    with open(path, "w") as f:
        f.write(f"{num} {dim}\n")
        for i in range(num):
            f.write(" ".join(str(v) for v in attr_data[i]) + "\n")
    print(f"  Wrote {path}  ({num} x {dim}, text)")


def generate_synthetic_attributes(vectors, num_attr_dims, num_clusters, random_seed=42):
    """
    Generate synthetic structured/categorical attributes from raw float vectors.

    Strategy (hybrid):
      1. Run k-means on a *subset* to get cluster centroids, then assign every
         vector to its nearest centroid → one categorical attribute ("cluster_id").
      2. For remaining dims, use L2-norm bucketing and random-projection
         sign patterns to produce diverse categorical features.

    Returns a (num_vectors, num_attr_dims) array of string category labels.
    """
    rng = np.random.RandomState(random_seed)
    n, d = vectors.shape

    attrs = np.empty((n, num_attr_dims), dtype=object)

    # --- Attribute 0: k-means cluster id (runs on a sample for speed) ---
    sample_size = min(n, 50000)
    idx = rng.choice(n, sample_size, replace=False)
    sample = vectors[idx].astype(np.float64)

    print(f"  Running KMeans (k={num_clusters}, sample={sample_size}) ...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed,
                    n_init=3, max_iter=100)
    kmeans.fit(sample)
    print(f"  KMeans done. Assigning all {n} vectors ...")
    labels = kmeans.predict(vectors.astype(np.float64))
    attrs[:, 0] = [f"c{lab}" for lab in labels]

    # --- Attribute 1: L2-norm bucket (10 buckets) ---
    norms = np.linalg.norm(vectors, axis=1)
    buckets = np.digitize(norms, np.percentile(norms, np.linspace(0, 100, 11)[1:-1]))
    attrs[:, 1] = [f"n{b}" for b in buckets]

    # --- Attributes 2..num_attr_dims-1: random projection sign patterns ---
    for j in range(2, num_attr_dims):
        proj = rng.randn(d)
        dots = vectors.dot(proj)
        median = np.median(dots)
        signs = (dots >= median).astype(int)
        attrs[:, j] = [f"rp{j}_{s}" for s in signs]

    return attrs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SIFT-128-Euclidean → NHQ-NPG_nsw Dataset Converter")
    print("=" * 60)

    # 1. Load HDF5
    print(f"\n[1/4] Loading HDF5: {HDF5_PATH}")
    with h5py.File(HDF5_PATH, "r") as f:
        train      = f["train"][:]      # (1000000, 128)
        test       = f["test"][:]       # (10000,   128)
        neighbors  = f["neighbors"][:]  # (10000,   100)
        distances  = f["distances"][:]  # (10000,   100)

    print(f"  train:      {train.shape}  dtype={train.dtype}")
    print(f"  test:       {test.shape}   dtype={test.dtype}")
    print(f"  neighbors:  {neighbors.shape}  dtype={neighbors.dtype}")
    print(f"  distances:  {distances.shape}  dtype={distances.dtype}")

    # 2. Create output directory
    print(f"\n[2/4] Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Write binary files (fvecs / ivecs)
    print(f"\n[3/4] Writing binary vector files ...")
    write_fvecs(os.path.join(OUTPUT_DIR, "sift_base.fvecs"), train)
    write_fvecs(os.path.join(OUTPUT_DIR, "sift_query.fvecs"), test)
    write_ivecs(os.path.join(OUTPUT_DIR, "sift_groundtruth.ivecs"), neighbors)

    # 4. Generate & write synthetic attributes
    print(f"\n[4/4] Generating synthetic attributes ...")
    print(f"  → base attributes  ({train.shape[0]} x {NUM_ATTR_DIMS})")
    base_attrs = generate_synthetic_attributes(train, NUM_ATTR_DIMS, NUM_CLUSTERS)
    write_attributes(os.path.join(OUTPUT_DIR, "sift_attributes.txt"), base_attrs)

    print(f"  → query attributes ({test.shape[0]} x {NUM_ATTR_DIMS})")
    query_attrs = generate_synthetic_attributes(test, NUM_ATTR_DIMS, NUM_CLUSTERS)
    write_attributes(os.path.join(OUTPUT_DIR, "sift_query_attributes.txt"), query_attrs)

    # Summary
    print(f"\n{'=' * 60}")
    print("Done!  Output files:")
    print(f"  {OUTPUT_DIR}/")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fn)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fn:30s}  ({size_mb:8.2f} MB)")
    print(f"\nUsage with NHQ-NPG_nsw:")
    print(f"  cd NHQ-NPG_nsw/examples/cpp/")
    print(f"  ./index {OUTPUT_DIR}/sift_base.fvecs \\")
    print(f"          {OUTPUT_DIR}/sift_attributes.txt \\")
    print(f"          {OUTPUT_DIR}/sift_graph.bin \\")
    print(f"          {OUTPUT_DIR}/sift_attributetable.bin \\")
    print(f"          <MaxM0> <efConstruction>")
    print(f"  ./search {OUTPUT_DIR}/sift_graph.bin \\")
    print(f"           {OUTPUT_DIR}/sift_attributetable.bin \\")
    print(f"           {OUTPUT_DIR}/sift_query.fvecs \\")
    print(f"           {OUTPUT_DIR}/sift_groundtruth.ivecs \\")
    print(f"           {OUTPUT_DIR}/sift_query_attributes.txt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

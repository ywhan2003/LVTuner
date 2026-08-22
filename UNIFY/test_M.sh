#!/usr/bin/env bash
set -euo pipefail

cd benchmark/code

PYTHON=/home/hyw/miniconda3/envs/unify/bin/python

M_LIST=(2 4 8 16 32 64)

for M in "${M_LIST[@]}"; do
    echo "==== Running M=${M} ===="
    rm -rf ../index/sift
    rm -rf ../index/sift.time
    
    $PYTHON search_hsig.py \
            --use_mbv_hnsw true \
            --M "$M" \
            --data_path ../data/hybrid_anns/sift-128-euclidean_with_scalar.hdf5 \
            --index_cache_path ../index/sift \
            --result_save_path ../result/sift_M${M}.csv
    echo "==== Finished M=${M} ===="
done

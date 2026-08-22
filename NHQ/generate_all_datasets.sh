#!/bin/bash
# Generate all NHQ datasets sequentially
OUT_BASE="$HOME/data/nhq_data"
SCRIPT="$HOME/coding/RFANNSTuner/NHQ/build_dataset.py"
LOG="$OUT_BASE/build.log"

mkdir -p "$OUT_BASE"
echo "Starting at $(date)" | tee "$LOG"

run_one() {
    local name="$1"
    local hdf5="$2"
    local sel="$3"
    local sdir="${sel/./}"
    local out="$OUT_BASE/${name}_sel${sdir}"

    if [ -f "$out/groundtruth.ivecs" ] && [ -f "$out/base.fvecs" ]; then
        echo "===== $name selectivity=$sel (SKIP, exists) =====" | tee -a "$LOG"
        return
    fi

    echo "===== $name selectivity=$sel [$(date)] =====" | tee -a "$LOG"
    /home/ywhan/coding/RFANNSTuner/.venv/bin/python3 "$SCRIPT" \
        -i "$hdf5" \
        -o "$out" \
        -s "$sel" \
        -n 3 \
        -w 140000 \
        --no-hybrid-gt \
        2>&1 | tee -a "$LOG"
    echo "Done: $out [$(date)]" | tee -a "$LOG"
}

# 1. SIFT (1M, 128) — fast
run_one "sift"  "$HOME/data/sift-128-euclidean.hdf5"  0.5
run_one "sift"  "$HOME/data/sift-128-euclidean.hdf5"  0.01

# 2. GloVe (1.18M, 100) — fast
run_one "glove" "$HOME/data/glove-100-angular.hdf5"   0.5
run_one "glove" "$HOME/data/glove-100-angular.hdf5"   0.01

# 3. GIST (1M, 960) — medium
run_one "gist"  "$HOME/data/gist-960-euclidean.hdf5"  0.5
run_one "gist"  "$HOME/data/gist-960-euclidean.hdf5"  0.01

# 4. Deep (10M, 96) — heavy, may need patience
run_one "deep"  "$HOME/data/deep-image-96-angular.hdf5" 0.5
run_one "deep"  "$HOME/data/deep-image-96-angular.hdf5" 0.01

echo "All done at $(date)" | tee -a "$LOG"

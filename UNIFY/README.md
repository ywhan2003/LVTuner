# UNIFY - Unified Index for Range Filtered Approximate Nearest Neighbors Search
This repo is the implementation of UNIFY: Unified Index for Range Filtered Approximate Nearest Neighbors Search.

Header-only C++ implementation for HSIG, with python bindings.

## Quick Start

### Compile

```bash
cd python_bindings
python setup.py install
``` 

### Datasets
The datasets can be downloaded from their official websites, and users can generate attributes and query ranges as the following: 
```bash
cd benchmark
cd code
python preprocessing.py
``` 

### Running example benchmark:
```bash
cd benchmark
cd code
python search_hsig.py --use_mbv_hnsw true --data_path YOUR_DATA_DIR --index_cache_path YOUR_INDEX_DIR --result_save_path YOUR_RESULT_PATH
```

For example, you can use the following command
```bash
python search_hsig.py --use_mbv_hnsw true --data_path ../data/hybrid_anns/sift-128-euclidean_with_scalar.hdf5 --index_cache_path ../index/sift --result_save_path ../result/sift.csv
```

```bash
nohup python search_hsig.py --use_mbv_hnsw true --data_path ../data/hybrid_anns/sift-128-euclidean_with_scalar.hdf5 --index_cache_path ../index/sift --result_save_path ../result/sift.csv > hsig.log 2>&1 &
```

```bash
nohup python search_hsig.py --use_mbv_hnsw true --data_path ../data/hybrid_anns/glove-100-angular_with_scalar.hdf5 --index_cache_path ../index/glove --result_save_path ../result/glove.csv > hsig.log 2>&1 &
```
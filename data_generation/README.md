```sh
python data_generation/filter_data_generation.py \
  --input_hdf5 /home/hyw/data/sift-128-euclidean.hdf5 \
  --diskann_path /home/hyw/RFANNSTuner/DiskANN \
  --output_dir /home/hyw/data/sift/selectivity_0p50 \
  --selectivity 0.5 \
  --k 10 \
  --nq 10000 \
  --dist_fn l2 \
  --base_file /home/hyw/data/sift/base.bin \
  --query_file /home/hyw/data/sift/query.bin \
  --gt_file /home/hyw/data/sift/selectivity_0p50/groundtruth.bin \
  --label_file /home/hyw/data/sift/selectivity_0p50/base_labels.txt \
  --filter_label sel_0p50 \
  --universal_label background \
  --overwrite
```

```sh
python data_generation/filter_data_generation.py \
  --input_hdf5 /home/hyw/data/glove-100-angular.hdf5 \
  --diskann_path /home/hyw/RFANNSTuner/DiskANN \
  --output_dir /home/hyw/data/glove/selectivity_0p50 \
  --selectivity 0.5 \
  --k 10 \
  --nq 10000 \
  --dist_fn cosine \
  --base_file /home/hyw/data/glove/base.bin \
  --query_file /home/hyw/data/glove/query.bin \
  --gt_file /home/hyw/data/glove/selectivity_0p50/groundtruth.bin \
  --label_file /home/hyw/data/glove/selectivity_0p50/base_labels.txt \
  --filter_label sel_0p50 \
  --universal_label background \
  --overwrite
```
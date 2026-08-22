import argparse
import json
import os
import time
from typing import List

import h5py
import numpy as np
import pandas as pd
import hannlib
import utils
from alg_base import BaseANN
from alg_hannlib import HannLib
from alg_hnswlib import HnswLib


class SearchStategy:
    HYBRID_FITERING = 0
    PRE_FILTERING = 1
    POST_FILTERING = 2
    CBO = 3


def read_hdf5_dataset(filepath, keys: List[str]):
    with h5py.File(filepath, "r") as f:
        ret = []
        for k in keys:
            ret.append(f[k][:])
    return ret


def build_contiguous_slot_combinations(num_slots: int):
    combinations = []
    for left in range(num_slots):
        for right in range(left, num_slots):
            combinations.append((left, right))
    return combinations


def compute_scalar_slot_ids(scalars: np.ndarray, slot_ranges: np.ndarray):
    upper_bounds = slot_ranges[:, 1]
    slot_ids = np.searchsorted(upper_bounds, scalars, side="right")
    slot_ids = np.clip(slot_ids, 0, len(upper_bounds) - 1)
    return slot_ids.astype(np.int64)


def edge_view(edges: np.ndarray):
    dtype = np.dtype([("u", np.uint64), ("v", np.uint64)])
    if edges.size == 0:
        return np.empty((0,), dtype=dtype)
    edges = np.ascontiguousarray(edges.astype(np.uint64, copy=False))
    return edges.view(dtype).reshape(-1)


def count_common_edges(edges_a: np.ndarray, edges_b: np.ndarray):
    if edges_a.size == 0 or edges_b.size == 0:
        return 0
    view_a = edge_view(edges_a)
    view_b = edge_view(edges_b)
    return np.intersect1d(view_a, view_b, assume_unique=True).size


def compute_inclusiveness(index: HannLib, base, base_scalars, M, ef_construction):
    if not hasattr(index, "p"):
        raise RuntimeError("Inclusiveness computation requires HannLib index.")

    print("Computing inclusiveness...")
    base = np.asarray(base, dtype=np.float32)
    base_scalars = np.asarray(base_scalars, dtype=np.int64).reshape(-1)
    slot_ranges = np.asarray(index.p.get_slot_ranges(), dtype=np.int64)
    if slot_ranges.ndim != 2 or slot_ranges.shape[1] != 2:
        raise RuntimeError("Invalid slot ranges from HSIG index.")

    slot_ids = compute_scalar_slot_ids(base_scalars, slot_ranges)
    num_slots = slot_ranges.shape[0]
    combinations = build_contiguous_slot_combinations(num_slots)

    total_common_edges = 0
    total_hnsw_edges = 0
    detail_rows = []

    for combo_id, (left, right) in enumerate(combinations):
        combo_slots = np.arange(left, right + 1, dtype=np.uint32)
        active_labels = np.where((slot_ids >= left) & (slot_ids <= right))[0].astype(
            np.uint64
        )

        if combo_id % 5 == 0:
            print(
                f"Evaluating slot combo [{left}, {right}] "
                f"({combo_id + 1}/{len(combinations)})..."
            )

        if len(active_labels) <= 1:
            detail_rows.append(
                {
                    "slot_left": left,
                    "slot_right": right,
                    "num_slots": right - left + 1,
                    "num_points": len(active_labels),
                    "common_edge": 0,
                    "hnsw_edge": 0,
                    "inclusiveness(%)": 100.0,
                }
            )
            continue

        hsig_edges = np.asarray(
            index.p.get_undirected_edges(combo_slots, active_labels),
            dtype=np.uint64,
        )

        subset_vectors = np.asarray(base[active_labels], dtype=np.float32)
        subset_scalars = np.asarray(base_scalars[active_labels], dtype=np.int64)
        subset_slot_ranges = np.array(
            [[subset_scalars.min(), subset_scalars.max()]], dtype=np.int64
        )

        ref_index = hannlib.HybridIndex(space="l2", dim=subset_vectors.shape[1])
        ref_index.init_index(
            slot_ranges=subset_slot_ranges,
            max_elements=len(active_labels),
            M=M,
            ef_construction=ef_construction,
        )
        ref_index.add_items(subset_vectors, subset_scalars, active_labels)
        ref_index.set_num_threads(1)

        hnsw_edges = np.asarray(
            ref_index.get_undirected_edges(np.array([0], dtype=np.uint32)),
            dtype=np.uint64,
        )

        hnsw_edge_count = len(hnsw_edges)
        common_edge_count = count_common_edges(hsig_edges, hnsw_edges)
        if hnsw_edge_count > 0:
            inclusiveness_pct = common_edge_count / hnsw_edge_count * 100
            total_common_edges += common_edge_count
            total_hnsw_edges += hnsw_edge_count
        else:
            inclusiveness_pct = 100.0

        detail_rows.append(
            {
                "slot_left": left,
                "slot_right": right,
                "num_slots": right - left + 1,
                "num_points": len(active_labels),
                "common_edge": common_edge_count,
                "hnsw_edge": hnsw_edge_count,
                "inclusiveness(%)": inclusiveness_pct,
            }
        )

    if total_hnsw_edges > 0:
        overall_inclusiveness = total_common_edges / total_hnsw_edges * 100
    else:
        overall_inclusiveness = 100.0

    return overall_inclusiveness, pd.DataFrame(detail_rows)


def bench_hybrid_query(
    index: BaseANN,
    search_strategy: int,
    optimizer_conf_dir,
    k,
    ef_list,
    al_list,
    low_range,
    high_range,
    query_vectors,
    query_ranges,
    gt,
    inclusiveness_pct=np.nan,
):
    print("Searching index...")
    time_list = []
    recall_list = []

    base_params = {
        "search_strategy": search_strategy,
        "optimizer_conf_dir": optimizer_conf_dir,
        "target_recall": 0.9,
        "ef_factor": 1,
        "low_range": low_range,
        "high_range": high_range,
        "ef": k,
        "al": 16,
    }

    if search_strategy != SearchStategy.CBO:
        ef_data = []
        al_data = []
        for ef in ef_list:
            for al in al_list:
                ef_data.append(ef)
                al_data.append(al)
                base_params["ef"] = ef
                base_params["al"] = al
                index.set_query_arguments(base_params)
                total_time = 0
                i = 0
                results = np.zeros(
                    [len(query_vectors), k], dtype="int64"
                )  

                for q, r in zip(query_vectors, query_ranges):
                    if i % 100 == 0:
                        print(f"Executiong query {i} under ef={ef}, al={al}...")
                    start = time.time()
                    I = index.hybrid_query(q, r, k)
                    end = time.time()
                    total_time += end - start
                    results[i, : len(I)] = I
                    i += 1
                recall = utils.compute_recall(results, gt, k, k)
                avg_time = total_time / len(query_ranges) * 1000
                time_list.append(avg_time)
                recall_list.append(recall)

        df = pd.DataFrame(
            {
                "ef": ef_data,
                "al": al_data,
                "recall": recall_list,
                "latency(ms)": time_list,
            }
        )
        df["QPS"] = 1000 / df["latency(ms)"]
    else:
        ef_data = []
        al_data = []
        for ef in ef_list:
            for al in al_list:
                ef_data.append(ef)
                al_data.append(al)
                base_params["ef"] = ef
                base_params["al"] = al
                base_params["low_range"] = low_range
                base_params["high_range"] = high_range
                index.set_query_arguments(base_params)
                total_time = 0
                i = 0
                results = np.zeros(
                    [len(query_vectors), k], dtype="int64"
                )  # 存储每个query_vector的knn

                for q, r in zip(query_vectors, query_ranges):
                    if i % 100 == 0:
                        print(
                            f"Executiong query {i} under ef={ef}, al={al}..."
                        )
                    start = time.time()
                    I = index.hybrid_query(q, r, k)
                    end = time.time()
                    total_time += end - start
                    results[i, : len(I)] = I
                    i += 1
                recall = utils.compute_recall(results, gt, k, k)
                avg_time = total_time / len(query_ranges) * 1000
                time_list.append(avg_time)
                recall_list.append(recall)
        df = pd.DataFrame(
            {
                "ef": ef_data,
                "al": al_data,
                "recall": recall_list,
                "latency(ms)": time_list,
            }
        )
        df["QPS"] = 1000 / df["latency(ms)"]
    if not np.isnan(inclusiveness_pct):
        df["inclusiveness(%)"] = inclusiveness_pct
    return df


def build_or_load_index(name: str, params, base, base_scalars, index_save_path):
    index = None
    if name == "HNSW":
        index = HnswLib(metric="euclidean", method_param=params)
    else:
        index = HannLib(metric="euclidean", method_param=params)
    
    if os.path.exists(index_save_path):
        print(f"Reading index from {index_save_path} ...")
        start, end = 0, 0
        index.loadIndex(base, index_save_path)
        index.scalars = base_scalars
    else:
        print(f"Building index: {index.name}...")
        start = time.time()
        index.fit(base, base_scalars)
        end = time.time()
        index.saveIndex(index_save_path)

        print(f"Index built: {index.name}, duration: {end-start}.")
        with open(index_save_path + ".time", "w") as f:
            f.write(f"{end - start}")

    return index, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--k", type=int, default=10, help="For kNN search")
    parser.add_argument(
        "--n_query_to_use", type=int, default=1000, help="Number of queries to use"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the hdf5 data file"
    )
    parser.add_argument(
        "--use_mbv_hnsw",
        action="store_true",
        default=False,
        help="Whether to use MBV-HNSW index",
    )
    parser.add_argument("--M", type=int, default=16, help="Number of graph connections")
    parser.add_argument(
        "--efConstruction", type=int, default=500, help="Parameter for HNSW index"
    )
    parser.add_argument("--B", type=int, default=8, help="Number of buckets")
    parser.add_argument("--low_range", type=float, default=0.1, help="The lower threshold for query range")
    parser.add_argument("--high_range", type=float, default=0.5, help="The upper threshold for query range")
    parser.add_argument(
        "--index_cache_path",
        type=str,
        required=True,
        help="Directory to load and save the index",
    )
    parser.add_argument(
        "--optimizer_conf_dir",
        type=str,
        required=False,
        help="Directory of optimizer configuration files, only effective if rso is enabled",
    )
    parser.add_argument(
        "--plan",
        type=int,
        default=0,
        help="Search plan: 0 (hybrid filtering), 1 (pre fitering), 2 (post filtering), 3 (CBO)",
    )
    parser.add_argument(
        "--target_recall_list",
        type=float,
        nargs="+",
        default=[0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
        help="List of target recall values, only effective if rso is enabled",
    )
    parser.add_argument(
        "--ef_factor_list",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 2.0],
        help="List of ef_factor values, only effective if rso is enabled",
    )
    # 候选邻居的数量
    parser.add_argument(
        "--ef_list",
        type=int,
        nargs="+",
        default=list(range(10, 200, 10)),
        help="List of EF values",
    )
    # 在搜索的过程中，不是每一个 segment 都去找 top k 个邻居，而是[M/segment_size]*k 个邻居，然后再从这些邻居中选出最终的 k 个邻居
    # 这里的 al 就是干这件事
    parser.add_argument(
        "--al_list", type=int, nargs="+", default=[8, 16, 32, 48, 64, 80, 96, 112, 128], help="List of AL values"
    )
    parser.add_argument(
        "--result_save_path",
        type=str,
        required=True,
        help="File path to save the benchmark results",
    )
    parser.add_argument(
        "--compute_inclusiveness",
        action="store_true",
        default=False,
        help="Whether to compute inclusiveness for MBV-HNSW benchmark",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Path to write metrics JSON (recall, qps, selected_ef, selected_al, etc.)",
    )
    parser.add_argument(
        "--select-recall-threshold",
        type=float,
        default=None,
        help="Recall threshold for selecting best (ef, al) pair",
    )
    parser.add_argument(
        "--select-recall-slack",
        type=float,
        default=0.005,
        help="Slack for recall feasibility check (default: 0.005)",
    )

    args = parser.parse_args()

    if os.path.exists(args.result_save_path):
        print(f"Result file exists, skip: {args.result_save_path}")
        exit(0)

    if args.use_mbv_hnsw:
        name = "MBV-HNSW"
        params = {
            "num_slots": args.B,
            "M": args.M,
            "efConstruction": args.efConstruction,
        }
    else:
        name = "HNSW"
        params = {"M": args.M, "efConstruction": args.efConstruction}

    (
        base,
        base_scalars,
        test,
        test_ranges,
        test_hybrid_knn,
    ) = read_hdf5_dataset(
        args.data_path,
        ["base", "base_scalars", "test", "test_ranges", "test_hybrid_knn"],
    )
    index, _ = build_or_load_index(
        name, params, base, base_scalars, args.index_cache_path
    )

    inclusiveness_pct = np.nan
    if args.use_mbv_hnsw and args.compute_inclusiveness:
        inclusiveness_pct, inclusiveness_detail_df = compute_inclusiveness(
            index,
            base,
            base_scalars,
            args.M,
            args.efConstruction,
        )
        inclusiveness_detail_path = args.result_save_path + ".inclusiveness.csv"
        inclusiveness_detail_df.to_csv(inclusiveness_detail_path, index=False)
        print(f"Inclusiveness(%): {inclusiveness_pct:.4f}")
        print(
            f"Inclusiveness detail were saved to {inclusiveness_detail_path}"
        )

    nq = args.n_query_to_use
    results = bench_hybrid_query(
        index,
        args.plan,
        args.optimizer_conf_dir,
        args.k,
        args.ef_list,
        args.al_list,
        args.low_range,
        args.high_range,
        test[:nq],
        test_ranges[:nq],
        test_hybrid_knn[:nq],
        inclusiveness_pct=inclusiveness_pct,
    )
    results.to_csv(args.result_save_path, index=False)
    print(results)
    print(f"Results were saved to {args.result_save_path}")

    # ── Select best (ef, al) pair and write metrics JSON ────────────────
    if args.metrics_output:
        threshold = args.select_recall_threshold
        slack = args.select_recall_slack

        # Find the best feasible (ef, al) pair.
        if threshold is not None and not results.empty:
            feasible_threshold = threshold - slack
            feasible = results[results["recall"] >= feasible_threshold]
            if not feasible.empty:
                best_row = feasible.loc[feasible["QPS"].idxmax()]
                selection_mode = "feasible"
            else:
                # No feasible pair: pick the one with highest recall.
                best_row = results.loc[results["recall"].idxmax()]
                selection_mode = "best_recall_fallback"

            selected_recall = float(best_row["recall"])
            selected_qps = float(best_row["QPS"])
            selected_ef = int(best_row["ef"])
            selected_al = int(best_row["al"])
            latency_ms = float(best_row.get("latency(ms)", 0))
        else:
            selected_recall = float(results["recall"].iloc[0])
            selected_qps = float(results["QPS"].iloc[0])
            selected_ef = int(results["ef"].iloc[0])
            selected_al = int(results["al"].iloc[0])
            latency_ms = float(results.get("latency(ms)", [0]).iloc[0] if "latency(ms)" in results.columns else 0)
            selection_mode = "first"

        # Read build time from index cache time file.
        build_time_s = 0.0
        index_time_path = args.index_cache_path + ".time"
        if os.path.exists(index_time_path):
            try:
                with open(index_time_path, "r") as f:
                    build_time_s = float(f.read().strip())
            except Exception:
                pass

        search_time_s = float(results["latency(ms)"].sum() / 1000.0) if "latency(ms)" in results.columns else 0.0

        feasible = selected_recall >= (threshold if threshold is not None else 0)

        metrics = {
            "recall": selected_recall,
            "qps": selected_qps,
            "selected_ef": selected_ef,
            "selected_al": selected_al,
            "build_time_s": build_time_s,
            "search_time_s": search_time_s,
            "latency_ms_p95": latency_ms,
            "feasible": feasible,
            "selection_mode": selection_mode,
            "inclusiveness_pct": float(inclusiveness_pct) if not np.isnan(inclusiveness_pct) else None,
        }

        metrics_dir = os.path.dirname(args.metrics_output)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.metrics_output}")
        print(f"  Selected: ef={selected_ef}, al={selected_al}, recall={selected_recall:.4f}, QPS={selected_qps:.1f}, feasible={feasible}")

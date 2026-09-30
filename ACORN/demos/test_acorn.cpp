#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <cerrno>
#include <omp.h>

#include "utils.cpp"

// create a directory if it does not exist (for ./tmp/<dataset> index outputs)
static void ensure_dir(const std::string& path) {
    if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "could not create directory %s: %s\n",
                path.c_str(), strerror(errno));
        exit(1);
    }
}

// build the base HNSW + filtered ACORN indices, write them to file, and get
// recall stats for all queries. Data is read from $ACORN_DATA_ROOT (default
// $HOME/data/acorn_data) for datasets [sift, glove, gist, deep].
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "====================\nSTART: running TEST_ACORN --" << nthreads << " cores\n" << std::endl;
    double t0 = elapsed();

    int efc = -1; // efConstruction; -1 = upstream defaults (40 for HNSW, M*gamma for ACORN)
    int efs = 16; //  default is 16
    int build_threads = 32; // omp threads for index build
    int search_threads = 1;  // omp threads for search (single-thread QPS measurement)
    int k = 10; // search parameter
    size_t d = 128; // will be overwritten by the query vector dimension
    int M;
    int M_beta; // param for compression
    int gamma;
    std::string dataset;

    size_t N = 0; // number of base vectors to use (0 = all rows in the file)

    {   // parse arguments
        if (argc < 6 || argc > 10) {
            fprintf(stderr, "Syntax: %s <number vecs|0 for all> <gamma> <dataset> <M> <M_beta> [<efSearch>] [<efConstruction>] [<build_threads>] [<search_threads>]\n", argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);

        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        if (!is_acorn_dataset(dataset)) {
            printf("got dataset: %s\n", dataset.c_str());
            fprintf(stderr, "Invalid <dataset>; must be one of [sift, glove, gist, deep]\n");
            exit(1);
        }

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);

        if (argc >= 7) {
            efs = atoi(argv[6]);
        }
        if (argc >= 8) {
            efc = atoi(argv[7]);
        }
        if (argc >= 9) {
            build_threads = atoi(argv[8]);
        }
        if (argc >= 10) {
            search_threads = atoi(argv[9]);
        }
        printf("efSearch: %d, efConstruction: %d, build_threads: %d, search_threads: %d\n",
               efs, efc, build_threads, search_threads);
    }

    size_t nq;
    float* xq;
    {   // load query vectors
        printf("[%.3f s] Loading query vectors\n", elapsed() - t0);

        size_t d2;
        std::string filename = get_file_name(dataset, false);
        xq = fvecs_read(filename.c_str(), &d2, &nq);
        assert(d == d2 || !"query dimension does not match");
        d = d2;

        std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" << nq << std::endl;
        printf("[%.3f s] Loaded query vectors from %s\n", elapsed() - t0, filename.c_str());
    }

    const int gt_size = 100;
    std::vector<faiss::idx_t> gt;

    float* xb = NULL;
    {   // load database vectors (before index creation so N can default to all rows)
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        std::string filename = get_file_name(dataset, true);
        xb = fvecs_read(filename.c_str(), &d2, &nb);
        assert(d == d2 || !"base dimension does not match the query dimension");
        if (N == 0) {
            N = nb;
        }
        assert(N <= nb);
        printf("[%.3f s] Loaded base vectors from file: %s, dim %ld, nb %ld\n",
               elapsed() - t0, filename.c_str(), d2, nb);
    }

    // synthetic labels and query filters, generated in-driver with a fixed
    // seed (gamma is a pure runtime parameter, no data files involved)
    std::vector<int> metadata = generate_metadata(N, gamma);
    std::vector<int> aq = generate_query_filters(nq, gamma);
    printf("[%.3f s] Generated metadata (%ld labels) and query filters (%ld) for gamma=%d\n",
           elapsed() - t0, metadata.size(), aq.size(), gamma);

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d, M_beta: %d, efSearch: %d, efConstruction: %d\n",
           elapsed() - t0, d, M, N, gamma, M_beta, efs, efc);
    // base HNSW index
    faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
    base_index.hnsw.efConstruction = efc > 0 ? efc : 40; // default is 40 in HNSW.capp
    base_index.hnsw.efSearch = efs; // default is 16 in HNSW.capp

    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16
    if (efc > 0) {
        hybrid_index.acorn.efConstruction = efc; // ctor default is M*gamma
    }

    omp_set_num_threads(build_threads);
    printf("[%.3f s] Building indices with %d omp thread(s)\n",
           elapsed() - t0, build_threads);

    {   // exact filtered ground truth, computed in-driver for the requested gamma
        double t_gt0 = elapsed();
        gt = compute_filtered_gt(xb, N, xq, nq, d, metadata, aq, gt_size);
        printf("[%.3f s] Computed filtered ground truth (gamma=%d, gt_size=%d) in %.3f s\n",
               elapsed() - t0, gamma, gt_size, elapsed() - t_gt0);
    }

    {   // populating the indices
        std::cout << "====================Vectors====================\n" << std::endl;

        printf("[%.3f s] Adding the vectors to the indices, size %ld*%ld\n",
               elapsed() - t0, N, d);

        base_index.add(N, xb);
        printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);

        delete[] xb;
    }

    // write indices to files
    {
        std::cout << "====================Write Index====================\n" << std::endl;
        {
            std::stringstream filepath_stream;
            filepath_stream << "./tmp/" << dataset << "/hybrid" << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta << "_gamma=" << gamma << ".json";
            std::string filepath = filepath_stream.str();
            ensure_dir("./tmp");
            ensure_dir("./tmp/" + dataset);
            write_index(&hybrid_index, filepath.c_str());
            printf("[%.3f s] Wrote hybrid index to file: %s\n", elapsed() - t0, filepath.c_str());
        }
        {
            std::stringstream filepath_stream;
            filepath_stream << "./tmp/" << dataset << "/base" << "_M=" << M << "_efc=" << efc << ".json";
            std::string filepath = filepath_stream.str();
            ensure_dir("./tmp");
            ensure_dir("./tmp/" + dataset);
            write_index(&base_index, filepath.c_str());
            printf("[%.3f s] Wrote base index to file: %s\n", elapsed() - t0, filepath.c_str());
        }
    }

    {   // print out stats
        printf("====================================\n");
        printf("============ BASE INDEX =============\n");
        printf("====================================\n");
        base_index.printStats(false);
        printf("====================================\n");
        printf("============ ACORN INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats(false);
    }

    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    omp_set_num_threads(search_threads);
    printf("[%.3f s] Searching with %d omp thread(s)\n",
           elapsed() - t0, search_threads);

    {   // searching the base database
        printf("====================HNSW INDEX====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0, k, nq, base_index.hnsw.efSearch);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        double t1 = elapsed();
        base_index.search(nq, xq, k, dis.data(), nns.data());
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        int nq_print = std::min(5, (int)nq);
        for (int i = 0; i < nq_print; i++) {
            printf("query %2d nn's: ", i);
            for (int j = 0; j < k; j++) {
                printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2 - t1);
        printf("[%.3f s] *** Base QPS: %f\n", elapsed() - t0, nq / (t2 - t1));
    }

    {   // look at stats
        const faiss::HNSWStats& stats = faiss::hnsw_stats;

        std::cout << "============= BASE HNSW QUERY PROFILING STATS =============" << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0, k, nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        std::cout << "n_visited: " << stats.n_visited << std::endl;
        std::cout << "n_steps: " << stats.n_steps << std::endl;
        printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
    }

    {   // searching the hybrid database
        printf("==================== ACORN INDEX ====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0, k, nq, hybrid_index.acorn.efSearch);

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        // create filter_ids_map, ie a bitmap of the ids that are in the filter
        std::vector<char> filter_ids_map(nq * N);
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                filter_ids_map[xq * N + xb] = (bool)(metadata[xb] == aq[xq]);
            }
        }

        double t1_x = elapsed();
        hybrid_index.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());
        double t2_x = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        int nq_print = std::min(5, (int)nq);
        for (int i = 0; i < nq_print; i++) {
            printf("query %2d nn's (%d): ", i, aq[i]);
            for (int j = 0; j < k; j++) {
                printf("%7ld (%d) ", nns2[j + i * k], metadata[nns2[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis2[j + i * k]);
            }
            printf("\n");
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2_x - t1_x);
        printf("[%.3f s] *** Hybrid QPS: %f\n", elapsed() - t0, nq / (t2_x - t1_x));

        {   // distance distribution of the returned candidates (k per query)
            double cds_sum = 0.0, cds_sq = 0.0, cds_min = 1e30, cds_max = -1e30;
            for (float v : dis2) {
                cds_sum += v;
                cds_sq += (double)v * v;
                if (v < cds_min) cds_min = v;
                if (v > cds_max) cds_max = v;
            }
            double cds_mean = cds_sum / dis2.size();
            double cds_std = std::sqrt(cds_sq / dis2.size() - cds_mean * cds_mean);
            printf("[%.3f s] *** candidate_distance_stats: mean=%f std=%f min=%f max=%f\n",
                   elapsed() - t0, cds_mean, cds_std, cds_min, cds_max);
        }

        // compute_recall returns the average number of hits in the top-k
        // (max k, i.e. recall*k); normalize by k to get recall in [0, 1]
        float hybrid_recall = compute_recall(gt, gt_size, nns2, (int)nq, k) / k;
        printf("[%.3f s] *** Hybrid Recall@%d: %f\n",
               elapsed() - t0, k, hybrid_recall);
    }

    {   // look at stats
        const faiss::ACORNStats& stats = faiss::acorn_stats;

        std::cout << "============= ACORN QUERY PROFILING STATS =============" << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0, k, nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        std::cout << "n_visited: " << stats.n_visited << std::endl;
        std::cout << "n_steps: " << stats.n_steps << std::endl;
        printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}

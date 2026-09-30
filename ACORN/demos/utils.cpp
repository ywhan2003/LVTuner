#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>

#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>

/*****************************************************
 * I/O functions for fvecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

bool is_acorn_dataset(const std::string& dataset) {
    return dataset == "sift" || dataset == "glove" ||
           dataset == "gist" || dataset == "deep";
}

// Directory holding the base/query vectors for one of our datasets. The
// vectors are gamma-independent (identical across all per-selectivity dirs),
// so any of the generated run dirs works; prefer the gamma=2 one.
std::string acorn_vector_dir(const std::string& dataset) {
    const char* root = std::getenv("ACORN_DATA_ROOT");
    std::string base = root && root[0] ? std::string(root)
                                       : std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/data/acorn_data";
    std::string dir05 = base + "/" + dataset + "_sel05";
    struct stat st;
    if (stat(dir05.c_str(), &st) == 0) {
        return dir05;
    }
    return base + "/" + dataset + "_sel001";
}

// get file name to load data vectors from
std::string get_file_name(std::string dataset, bool is_base) {
    if (is_acorn_dataset(dataset)) {
        return acorn_vector_dir(dataset) + (is_base ? "/base.fvecs" : "/query.fvecs");
    }
    std::cerr << "Invalid dataset in get_file_name: " << dataset << std::endl;
    return "";
}

// static: faiss/impl/ACORN.cpp defines its own global elapsed,
// which would collide at link time when the demo links against libfaiss.
static double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*****************************************************
 * Synthetic labels + filtered ground truth (gamma-independent)
 *
 * Labels and query filters are generated here with a fixed seed instead of
 * being loaded from pre-generated JSON files, so gamma is a pure runtime
 * parameter with no data dependency. Both the old pre-generated scheme
 * (numpy PCG64(42), i.i.d. uniform) and this splitmix64 scheme produce
 * i.i.d. uniform labels in [1..gamma], so results are statistically
 * equivalent across schemes.
 *****************************************************/

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static const uint64_t SEED_LABEL = 0x2C1B3C6D5E4F7081ULL;
static const uint64_t SEED_QUERY = 0x9A8B7C6D5E4F3012ULL;

std::vector<int> generate_metadata(size_t n, int gamma) {
    std::vector<int> labels(n);
    for (size_t i = 0; i < n; i++) {
        labels[i] = (int)(splitmix64(i ^ SEED_LABEL) % (uint64_t)gamma) + 1;
    }
    return labels;
}

std::vector<int> generate_query_filters(size_t nq, int gamma) {
    std::vector<int> aq(nq);
    for (size_t q = 0; q < nq; q++) {
        aq[q] = (int)(splitmix64(q ^ SEED_QUERY) % (uint64_t)gamma) + 1;
    }
    return aq;
}

// Exact filtered ground truth: per query, the top-gt_size base ids among
// labels == aq[q], ranked by squared L2 distance (nearest first, matching
// the ordering of the previously pre-generated gt.json files). Padded with
// -1 if a query's matched subset has fewer than gt_size points.
std::vector<faiss::idx_t> compute_filtered_gt(
        const float* base, size_t nb,
        const float* query, size_t nq,
        size_t d,
        const std::vector<int>& labels,
        const std::vector<int>& aq,
        int gt_size) {
    std::vector<float> base_sq(nb);
#pragma omp parallel for
    for (size_t i = 0; i < nb; i++) {
        const float* v = base + i * d;
        float s = 0.0f;
        for (size_t j = 0; j < d; j++) {
            s += v[j] * v[j];
        }
        base_sq[i] = s;
    }

    std::vector<faiss::idx_t> gt((size_t)nq * gt_size, -1);
#pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        int filter = aq[q];
        const float* qvec = query + q * d;
        float q_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            q_sq += qvec[j] * qvec[j];
        }

        // max-heap on distance, capped at gt_size
        std::priority_queue<std::pair<float, faiss::idx_t>> heap;
        for (size_t i = 0; i < nb; i++) {
            if (labels[i] != filter) {
                continue;
            }
            const float* v = base + i * d;
            float dot = 0.0f;
            for (size_t j = 0; j < d; j++) {
                dot += v[j] * qvec[j];
            }
            float dist = q_sq + base_sq[i] - 2.0f * dot;
            if ((int)heap.size() < gt_size) {
                heap.emplace(dist, (faiss::idx_t)i);
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.emplace(dist, (faiss::idx_t)i);
            }
        }

        // heap pops farthest-first; store nearest-first to match the
        // pre-generated gt.json ordering (compute_recall uses the first k)
        std::vector<std::pair<float, faiss::idx_t>> tmp;
        while (!heap.empty()) {
            tmp.push_back(heap.top());
            heap.pop();
        }
        for (int k = 0; k < gt_size; k++) {
            gt[q * gt_size + k] = (k < (int)tmp.size()) ? tmp[tmp.size() - 1 - k].second : -1;
        }
    }
    return gt;
}

// ground truth labels @gt (gt_size entries per query), results to evaluate @I
// with @nq queries: returns the average number of hits in the top-k (max k).
// The caller divides by k to get recall in [0, 1].
float compute_recall(std::vector<faiss::idx_t>& gt, int gt_size, std::vector<faiss::idx_t>& I, int nq, int k) {
    int n_hits = 0;
    for (int i = 0; i < nq; i++) { // loop over all queries
        std::vector<faiss::idx_t>::const_iterator first = gt.begin() + i * gt_size;
        std::vector<faiss::idx_t>::const_iterator last = gt.begin() + i * gt_size + k;
        std::set<faiss::idx_t> gt_nns(first, last);

        for (int j = 0; j < k; j++) { // iterate over returned nn results
            if (gt_nns.count(I[i * k + j]) != 0) {
                n_hits++;
            }
        }
    }
    return n_hits / float(nq);
}

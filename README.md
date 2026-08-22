# Light

A unified tuning framework for four ANN tuning pipelines: **hnswlib**, **UNIFY**, **DiskANN** (diskann-filter / filter-diskann), and **NHQ**. Every pipeline shares the same architecture:

- **Knowledge-driven**: static diagnostic knowledge cards under `knowledge_base/<ALGORITHM>/` (Signals / Interpretation / margin-tiered Intervention). Each round, the closest single card is retrieved by an LLM from the last round's full metrics (`select_best_match`) and injected into the diagnosis prompt.
- **Subgroup-mining initialization**: `agentic.subgroup_init.json_path` points to a historical-task subgroup-mining JSON — its mined range becomes the execution-time tuning constraint and its mined best point becomes the initial seed (seed rounds do not consume the budget).
- **Dominance repository**: each construction setting maintains a `(L, U]` evidence interval (L = search param of the highest-recall infeasible point, U = search param of the highest-QPS feasible point; all other parameters are recorded as controlled variables). A hard checker rejects `s <= L` / `s > U` and triggers LLM re-proposal.
- **Mandatory LLM proposals**: every round's proposal must be LLM-generated — failures raise errors. All model configuration is read from `.env`.

---

## 1. Repository Structure

```
Light/
├── main.py                  # Unified entrypoint: python main.py <pipeline> --config <yaml>
├── functions/               # Tuning pipeline implementations (hnswlib/unify/nhq/filter_diskann/diskann_filter + shared driver)
├── agents/                  # Per-algorithm LLM tuning agents (diagnose + propose)
├── utils/                   # Shared utilities: static knowledge cards, current-task memory, interval table / hard checker, benchmark runner, subgroup mining
├── configs/                 # Default tuning configs (one per pipeline) + prompts/templates
├── knowledge_base/          # Diagnostic knowledge cards (HNSW/UNIFY/NHQ/FilterDiskANN/) + parameter-semantics docs + diagnostic trees
├── llm_agent/               # LLM call infrastructure
├── benchmarks/              # Benchmark scripts per algorithm (hnswlib/nhq/diskann)
├── data_generation/         # Data generation tools (HDF5 conversion, fraction/filter-label generation)
├── data/                    # Runtime data directory (yaml defaults use relative data/... paths)
├── hnswlib/                 # git submodule (upstream hnswlib, installed editable by uv)
├── UNIFY/                   # UNIFY source (vendored; build the hannlib extension locally)
├── DiskANN/                 # DiskANN source (vendored; build the python bindings locally)
├── NHQ/                     # NHQ source and dataset scripts (vendored; cmake build required)
├── pyproject.toml / uv.lock # Python dependencies (managed by uv, python 3.13)
└── .env.example             # LLM configuration template (copy to .env and fill in)
```

## 2. Environment Setup (uv)

```bash
# 1. Install dependencies (python 3.13; hnswlib installed editable automatically)
uv sync

# 2. Configure the LLM (all LLM calls read from .env)
cp .env.example .env
# Edit .env: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL_NAME

# 3. Build the algorithm libraries locally (as needed)
#   UNIFY (vendored; build the hannlib python extension):
uv pip install -e UNIFY          # or: cd UNIFY && python setup.py install
#   DiskANN (python bindings; see DiskANN/python/README.md):
cd DiskANN/python && pip install build && python -m build && pip install dist/*.whl
#   NHQ (C++ index binaries invoked by the nhq benchmark):
bash NHQ/build.sh
```

## 3. Data Generation (data_generation/)

All four algorithms build on **ann-benchmarks-style HDF5** files (`train` / `test` / `neighbors` groups) from public datasets: **SIFT-128** (L2), **GIST-960** (L2), **GloVe-100** (angular), **Deep-1M-96** (angular).

| Algorithm | Data format | How to generate |
|---|---|---|
| **hnswlib** | HDF5 (e.g. `sift-128-euclidean.hdf5`) | Use directly; `data_generation/folder_to_hdf5.py` converts from fvecs directories; `data_generation/check_data.py` validates the format |
| **UNIFY** | Range-filtered fraction HDF5 (e.g. `sift-128-euclidean_fraction_2.hdf5`) | `python data_generation/data_generation.py --dataset sift-128-euclidean --fraction 2` (or 8) |
| **DiskANN** (filter-diskann / diskann-filter) | `base.bin` / `query.bin` / `groundtruth.bin` + filter labels `base_labels.txt` (generated per selectivity) | `python data_generation/filter_data_generation.py --input_hdf5 <hdf5> --selectivity 0.5 --diskann_path DiskANN/ --output_dir <out> ...` (full examples in `data_generation/README.md`) |
| **NHQ** | `base.fvecs` / `base_attr.txt` / `query.fvecs` / `query_attr.txt` / `groundtruth.ivecs` (synthetic attributes with controllable selectivity) | `python NHQ/convert_sift_dataset.py` (SIFT-specific conversion), `python NHQ/build_dataset.py` (generic, synthetic attributes); batch: `bash NHQ/generate_all_datasets.sh` |

Place generated artifacts under `data/` (the default yaml paths are relative `data/...`), or adjust the benchmark data paths in the corresponding yaml.

## 4. Running Tuning

Unified entrypoint (default yamls are registered per pipeline, so `--config` is optional):

```bash
uv run python main.py <pipeline> [--config configs/<x>.yaml] [--no-resume] [--dry-run]
```

| pipeline | default yaml | description |
|---|---|---|
| `hnswlib` | `configs/hnswlib_tune.yaml` | native hnswlib (M / ef_construction / ef) |
| `unify` | `configs/unify_tune.yaml` | UNIFY/HSIG range filtering (M / B / efConstruction / ef / al) |
| `diskann-filter` | `configs/diskann_filter_tune.yaml` | Filtered-DiskANN (R / Lbuild / FilteredLBuild / alpha / L, shared driver) |
| `filter-diskann` | `configs/filter_diskann_tune.yaml` | Filter-DiskANN Vamana (R / FilterLBuild / alpha / L, independent unify architecture) |
| `nhq` | `configs/nhq_tune.yaml` | NHQ hybrid query (M / efConstruction / ef / weight) |

Example (SIFT at τ=0.95 with subgroup-mining initialization — `configs/hnswlib_tune.yaml` is this run; its data paths are machine-specific absolute paths, adjust them to your own):

```bash
uv run python main.py hnswlib --config configs/hnswlib_tune.yaml --no-resume
```

Key configuration options (the `agentic` block of each yaml is minimal: `enabled` + optional `subgroup_init` + `logging`):

- `agentic.subgroup_init.json_path`: historical-task subgroup-mining JSON — its mined range becomes the execution-time constraint and its best point the initial seed; changing the path requires `--no-resume`
- `search.recall_threshold` / `search.budget`: target recall / tuning budget (**initialization seed rounds are excluded**)
- LLM configuration: always read from `.env` (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME`) — no yaml key needed
- Static knowledge cards: loaded automatically from `knowledge_base/<ALGORITHM>/` — no yaml key needed

Run artifacts (under `output.dir`): `trials/<name>.jsonl` (compact format: params + metrics{recall,qps} only), `stage_a_plan.json`, `current_task_memory/` (point memory, transitions, interval tables). Runs resume by default; `--dry-run` only generates candidates and commands without executing benchmarks.

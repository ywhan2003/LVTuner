# LVTuner

## 1. Repository Structure

```
LVTuner-VLDB/
├── main.py                  # Unified entrypoint: python main.py <pipeline> --config <yaml>
├── functions/               # Tuning pipeline implementations (hnswlib/unify/nhq/filter_diskann/diskann_filter + shared driver)
├── agents/                  # Per-algorithm LLM tuning agents (diagnose + propose)
├── conditional_policy/      # Offline policy-validation pipeline (python -m conditional_policy.cli) + runtime.py (deterministic per-round matcher used by hnswlib)
├── utils/                   # Shared utilities: static knowledge cards, regression-tree init, current-task memory, interval table / hard checker, benchmark runner, subgroup mining
├── configs/                 # Default tuning configs (one per pipeline) + prompts/templates
├── knowledge_base/          # Diagnostic knowledge cards (HNSW/UNIFY/NHQ/FilterDiskANN/) + conditional_policy/policies.json + diagnostic_tree.json
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

Example (SIFT at τ=0.95 with regression-tree initialization + conditional policies — `configs/hnswlib_tune.yaml` is this run; its data paths are machine-specific absolute paths, adjust them to your own):

```bash
uv run python main.py hnswlib --config configs/hnswlib_tune.yaml --no-resume
```

Key configuration options (the `agentic` block of each yaml):

- `agentic.initial_design.mode: regression_tree` + `agentic.initial_design.regression_tree.*` (hnswlib): `trials_path` must point at a **prior** trials file (this run writes its own — do not self-reference); missing/empty input fails open and the run continues on the full YAML space. `seed_count` is a pool — exactly one seed executes as the round-1 cold start; all seeds are budget-exempt.
- `agentic.conditional_policy.*` (hnswlib): deterministic per-round policy matching (`enabled`, `policy_file`, `max_policies_per_round`, `max_context_chars`, `min_history_for_matching`). Below `min_history_for_matching` trials (cold start / resume with minimal rows) the full accepted set is injected instead; hnswlib does not instrument `expansion_cnt`/`traversal_effectiveness`, so those symptom dimensions are wildcards. Fail-open: a missing policy file leaves the prompt unchanged.
- `agentic.subgroup_init.json_path` (unify / nhq / filter-diskann / diskann-filter): historical-task subgroup-mining JSON — its mined range becomes the execution-time constraint and its best point the initial seed; changing the path requires `--no-resume`
- `search.recall_threshold` / `search.budget`: target recall / tuning budget (**initialization seed rounds are excluded**)
- LLM configuration: always read from `.env` (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME`) — no yaml key needed
- Static knowledge cards (unify / nhq / filter-diskann / diskann-filter): loaded automatically from `knowledge_base/<ALGORITHM>/` — no yaml key needed

Rebuilding the conditional-policy book (offline, `python -m conditional_policy.cli`): the default synthetic validation run writes `results/conditional_policy/` (`policies.json` etc.); real-data runs use `--real-points results/hnswlib/current_task_memory/<name>.points.jsonl:<task>:<tau> --real-transitions ...transitions.jsonl:<task>:<tau>`. Point the runtime at the resulting file via `agentic.conditional_policy.policy_file`.

Run artifacts (under `output.dir`): `trials/<name>.jsonl` (compact format: params + metrics{recall,qps} only), `stage_a_plan.json` / `stage_report.json` (incl. `regression_tree_init`, `regression_tree_freeze`, and the per-run `conditional_policy` match summary), `current_task_memory/` (point memory, transitions, interval tables), `regression_tree_init/` (regtree report + prompt). Dry runs additionally write `dry_run_plan.json` with `conditional_policy_rounds`. Runs resume by default; `--dry-run` only generates candidates and commands without executing benchmarks.

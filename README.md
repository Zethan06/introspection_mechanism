# A Mechanistic Study of Language Model Introspection

Research code for studying how language models detect and localize injected
concept representations. The experiments investigate the roles of attention
heads in introspective reports through activation interventions, head selection,
and query–key (QK) and output–value (OV) analyses.

## Overview

A concept vector is injected into the hidden state at one of ten candidate
tokens. The model reports the affected position's label, or `none` when no
injection is detected. The experimental pipeline:

1. Calibrates injection layers and strengths and constructs balanced token clusters.
2. Screens concepts and creates fixed training, validation, and test splits.
3. Selects gate heads using straight-through Top-k masks and analyzes router
   heads through activation patching.
4. Evaluates causal interventions, label transfer, and lexical controls.
5. Analyzes how the selected heads read concept injections through QK and OV circuits.

## Repository Structure

```text
configs/              Model-specific experiment settings
introspection_core/   Reusable model, intervention, scoring, and analysis code
  assets/             Static visualization assets
scripts/              Python experiment and analysis entry points
run_sh/               Staged experiment workflows
data/dataset/        Calibration concepts and generated model-specific datasets
tests/               CPU-friendly unit tests
```

`results/`, `logs/`, and `tmp/` contain generated outputs and are excluded from
version control. Model checkpoints and the local `.venv` are not included.

## Installation

The configured environment uses Python 3.10, PyTorch 2.6.0, and TransformerLens
3.5.x. GPU experiments default to CUDA and `bfloat16`. Linux and Windows use
the CUDA 12.4 PyTorch build; macOS resolves PyTorch from PyPI. The shell workflows
require Bash and have been verified on Linux.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv sync --locked
uv run --locked python scripts/search_vocab_injection_words.py --help
```

`pyproject.toml` declares the dependencies, `uv.lock` fixes their resolved
versions, and `.python-version` selects the interpreter. Synchronization creates
an isolated `.venv` and installs `introspection_core` in editable mode, including
its visualization assets. Compatibility constraints retain prebuilt dependencies
for the experiment server's glibc 2.17.

The shell workflows use `.venv/bin/python` automatically; activation is optional.
Set `UV_PROJECT_ENVIRONMENT` to use another environment directory, or
`PYTHON_BIN` to select an existing interpreter explicitly. Synchronize the
environment before launching jobs.

## Model Configuration

The repository includes configurations for:

| Model | Configuration |
| --- | --- |
| Qwen3-4B-Instruct-2507 | `configs/qwen3-4b-instruct-2507.env` |
| Llama-3.1-8B-Instruct | `configs/llama3.1-8b-instruct.env` |
| Gemma-3-12B-IT | `configs/gemma3-12b-it.env` |

Before running an experiment, review the model configuration:

- `MODEL_ID`: Hugging Face model ID or local checkpoint path.
- `GPU_IDS`, `STE_GPU`, and `VISUALIZATION_GPU`: available GPU assignments.
- `BATCH_SIZE` and `EXTRACTION_BATCH_SIZE`: batch sizes for the available memory.
- `DATASET_ROOT`, `RESULTS_ROOT`, `LOGS_ROOT`, and `TMP_ROOT`: storage locations.
- `STE_LAYERS`, `ROUTER_LAYER`, and `ROUTER_HEADS`: model-specific gate search
  window and router heads.

Use `configs/model.env.example` as a template for another model. Stage 00 writes
the selected injection setting to `results/<model>/calibration/selection.json`.
Nonempty `INJECTION_LAYER` and `INJECTION_STRENGTH` values in a configuration
act as consistency checks against that selection; leave them blank to use the
newly calibrated values.

## Data and Experimental Protocol

The bundled calibration vocabulary is
`data/dataset/simple_1000_concepts.json`. The pipeline generates cluster banks,
concept splits, and the English-token vocabulary under `data/dataset/<model>/`.
Model weights are loaded separately through `MODEL_ID`.

Stage 00 constructs four disjoint banks of 30 clusters for calibration, training,
validation, and testing. Coarse screening produces a shortlist of 3,000 concepts;
fine screening evaluates that shortlist on all 30 calibration clusters and
selects the top 300, split into groups of 100 for training, validation, and testing.

Stage 10 compares two fixed populations:

- **C_intro:** the 100 validation concepts.
- **C_nonintro:** a seeded random sample of 100 from the lowest-ranked 300
  concepts within the 3,000-concept shortlist, using fine-screening accuracy.

The nonintro sampler validates shortlist membership and population separation.
Neither population is re-ranked using test outcomes.

### Prompts

`introspection_core/prompts.py` defines the prompt families:

- `semantic_highinj_posref_gate_balanced_disrupts`: the evaluation prompt, with
  ten `TOKEN i: <candidate>` entries and answers `0`–`9` or `none`. The
  `_letters_a_j` and `_numwords_one_ten` variants change the displayed labels;
  shuffled-label conditions use a fixed derangement for each cluster.
- `token_localization`: the clean position-prior prompt, prefilled with
  `It is located in TOKEN `, used during token and cluster screening.

## Running Experiments

Run commands from the repository root. Most per-model workflows accept a
configuration file, for example:

```bash
bash run_sh/00_cluster_calibration_top300.sh configs/qwen3-4b-instruct-2507.env
```

The following entry points use different arguments:

```bash
# Stage 04e: one worker on GPU 0; model selection is controlled by MODELS.
bash run_sh/04e_gate_label_transfer.sh 0

# Stage 11: aggregate figures and tables after the required model runs finish.
bash run_sh/11_paper_figures_and_tables.sh
```

For Stage 04e, `MODEL_ROOT` can point to a directory containing local checkpoints;
otherwise, the workflow loads models from Hugging Face. GPU runs require access
to the selected checkpoints and sufficient device memory.

### Experiment Stages

The table follows the reproduction workflow for the supplied model configurations.
Downstream stages consume earlier outputs; the two Stage 08 sweep scripts are
alternative execution modes. Run Stage 11 after the required per-model results
are available.

| Stage | What it produces |
|---|---|
| `00_cluster_calibration_top300` | clean token prior, four disjoint 30-cluster banks, layer/strength calibration, concept screening, the 300 selected concepts split 100/100/100 |
| `01_validation_diagnostics` | concept-vector payload; K-means position clustering by layer (Figure 2b) |
| `02_validation_visualizations` | attention browser and residual PCA across layers (latent PCA figures) |
| `02b_position_none_direction` | position--none direction fit on one validation half, scored on the other (Figure 2a) |
| `025_router_head_visualization` | per-head output PCA at the router layer (head PCA figures) |
| `03_prepare_splits` | injected outcomes for the train/validation/test banks |
| `04b_ste_topk_sweep` | gate-on / gate-off STE masks for k = 1, 4, 8, 16, 32, 48, 64, scored on validation; the Top-32 pair is the gate-head set used everywhere else |
| `04d_random_topk_control` | random-k control for the Top-k figure |
| `04f_gate_router_test` | Top-32 test grid under ordered digits: gate patches, router heads pinned clean, gate donor i x router donor j (Figure 3, Table 2, Table 1 digits column) |
| `04e_gate_label_transfer` | the same Figure 3 runs under the other five label settings |
| `05_lexical_replacement_control` | lexical replacement table (all six label settings) |
| `06_router_redirection` | one-hot attention redirection of the router heads, all six label settings |
| `08c` (one GPU) / `08d` (four GPUs) | head-wise clean patching over every head (router-head heatmaps); both call `08_validation_clean_head_patch` per shard |
| `09_label_accuracy_splits` | Table 1 letters/words and shuffled columns |
| `10_concept_readout` | Section 4: C_nonintro sample, QK key-change SVD, query/key term ablation and KL, SVD of M = W_O dV^T, OV causal ablations |
| `11_paper_figures_and_tables` | Figures 2 and 3, Top-k figure, Table 1, Table 2 and redirection-table intervals (all models) |

During gate-on training, the router heads' attention is forced to the target
successor token. The supplied configurations specify the router heads and gate
search windows used in the paper.

## Results and Paper Artifacts

Paths below use the default `RESULTS_ROOT=results`.

| Paper item | Output |
|---|---|
| Injection calibration table | `results/<model>/calibration/selection.json` |
| Table 1 | `results/paper/task_performance.{csv,tex}` |
| Lexical replacement table | `results/lexical_replacement/summary.csv` |
| Figure 2 | `results/paper/manuscript_panels/fig2_layerwise.*` |
| Figure 3 and its intervals | `results/paper/manuscript_panels/fig3_interventions_mean6.*`, `fig3_intervals.json` |
| Table 2 and its intervals | `results/paper/router_label_transfer/`, `results/paper/uncertainty/` |
| Top-k selection figure | `results/paper/ste_topk_gate_effect/` |
| Gate-head table | `results/<model>/ste_topk_sweep/top32/train_{on,off}/selected_heads.json`, `selection_summary.json` |
| Head-wise patching heatmaps | `results/<model>/validation_clean_head_model_sweep{,_parallel}/` |
| Redirection table and intervals | `results/<model>/router_sweeps/`, `router_label_sweeps/`, `results/paper/uncertainty/` |
| Latent and head PCA | `results/<model>/visualizations/*.vector.html` |
| Router attention patterns | `results/<model>/visualizations/validation_attention.html` |
| QK term ablation, KL | `concept_readout/qk_score_ablation/summary/`, `concept_readout/qk_kl/summary/` |
| QK first-mode table | `concept_readout/qk_successor/first_mode_summary/`, query norms in `qk_successor/context_summary/` |
| OV statistics | `concept_readout/ov/output_svd/summary.csv` |
| OV causal ablations | `concept_readout/ov/causal_terms/`, `causal_top5/`, `causal_ksweep/` |

`concept_readout/` lives under `results/<model>/`.

## Development and Validation

Run the CPU-friendly unit suite and syntax checks without loading a checkpoint:

```bash
uv run --locked python -m unittest discover -s tests -p "test_*.py"
uv run --locked python -m compileall -q introspection_core scripts
uv lock --check
```

Manage dependencies through uv:

```bash
uv add PACKAGE                     # Add a dependency and update the lock file
uv lock --upgrade-package PACKAGE  # Update one dependency within its constraints
uv sync --locked                   # Install the locked environment
```

Commit `pyproject.toml` and `uv.lock` together when dependencies change. Update
`.python-version` when intentionally changing the project interpreter.
`requirements.txt` is a compatibility entry point for pip and reads dependencies
from `pyproject.toml`; it does not apply uv's lock file or CUDA index selection.

For details, see the [uv environment documentation](https://docs.astral.sh/uv/concepts/projects/sync/)
and [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

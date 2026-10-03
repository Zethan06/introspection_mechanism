# A Mechanistic Study of Language Model Introspection

This repository contains the code for the paper [**A mechanistic study of language model introspection**](https://arxiv.org/abs/2609.35108).

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zou2026mechanistic,
  title={A mechanistic study of language model introspection},
  author={Zou, Jiahong and Sun, Xiangkun and Kong, Lingkai and Wang, Tonghan},
  journal={arXiv preprint arXiv:2609.35108},
  year={2026}
}
```

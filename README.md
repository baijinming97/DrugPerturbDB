# DrugPerturbDB

Code, numerical results and frozen models accompanying *Reproducibility and
cross-dataset concordance of drug-induced transcriptional responses*.

The study examines the repeatability, cross-source agreement and component
structure of drug responses in Tahoe-100M, L1000 L3, DRUG-seq, CIGS, sciPlex3
and MIX-seq, and evaluates which response components a control-expression
baseline recovers.

Licensed under [CC BY-NC 4.0](code/common/docs/LICENSING.md).

## Analyses and results

| Analysis | Code | Results |
| :--- | :--- | :--- |
| Expression and response repeatability | [01_repeatability](code/01_repeatability/) | [Data S1](result/source_tables/Data_S1_measurements.zip) |
| Cross-dataset agreement and drug retrieval | [02_cross_dataset](code/02_cross_dataset/) | [Data S1](result/source_tables/Data_S1_measurements.zip) |
| Response-component decomposition | [03_components](code/03_components/) | [Data S1](result/source_tables/Data_S1_measurements.zip) |
| Baseline errors and drug-average–interaction mapping | [04_prediction](code/04_prediction/) | [Data S2](result/source_tables/Data_S2_predictions.zip) |

```text
code/     Analysis programs, shared functions and software checks
data/     Data preparation, source metadata and frozen models
result/   Numerical results and plotting inputs
```

## Installation

Use Python 3.12. Source-data processing requires Linux or WSL; DRUG-seq also
requires R. Numerical checks and plotting also run on Windows.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/common/environment/requirements-tested-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Frozen-model inference additionally requires CPU PyTorch:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, create the environment with `conda env create -f environment.yml`.
On Windows, activate the virtual environment with `.venv\Scripts\Activate.ps1`.

## Reproduction

Run from the repository root:

```bash
drugperturbdb --repo . verify         # Check supplied files and result archives
drugperturbdb --repo . example-check  # Recalculate retained measurement examples
drugperturbdb --repo . models-check   # Verify the ten frozen model files
python data/prepare.py plan          # List source-data preparation stages
```

For source-data recomputation, follow [data preparation](data/README.md) and
[analysis workflows](code/common/docs/WORKFLOW.md). Each experiment provides
`run.py`, `config.json` and optional `plot.py`. Downloaded data go to `data/raw/`,
prepared inputs to `data/prepared/`, and new results to
`result/<experiment>/computed/`. Rendered figures are generated locally.

The five control-expression-only 3H2048 models and five drug-average-to-interaction
(G5) decoders are included in `data/models/`. Descriptive Tahoe analyses cover 379 eligible drugs;
predictive analyses use the separate 153-drug development panel.

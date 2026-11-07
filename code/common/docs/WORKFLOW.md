# Analysis workflows

Run commands from the repository root after [installation](../../../README.md#installation).
Source files and prepared inputs are described in [data preparation](../../../data/README.md).
Analytical definitions, preprocessing, and statistical procedures are described in [Methods](METHODS.md).

## Data preparation and analysis

1. `python data/prepare.py plan` lists the ordered preparation stages.
2. Download a source, initialize the workspace, bind verified inputs and run
   its preparation stages using `data/prepare.py`.
3. Run the corresponding analysis stage with `code/<experiment>/run.py`.

For example, after preparing sciPlex3 profiles:

```bash
python code/01_repeatability/run.py --stage sciPlex3 --check
python code/01_repeatability/run.py --stage sciPlex3 --execute --cpus 4
python code/03_components/run.py --stage sciPlex3 --execute --cpus 4
```

Entry points list dependencies by default. Computation requires `--execute`
and a declared stage; missing inputs return exit code 2. Analysis stages do not
implicitly download sources. Generated inputs are shared in `data/prepared/`;
new tables are written to the experiment's `computed/` directory.

| Experiment | Inputs and purpose |
| --- | --- |
| `01_repeatability` | Prepared expression and response profiles; independent replicate comparisons |
| `02_cross_dataset` | Six prepared condition panels and fixed identity/matching tables; agreement and retrieval |
| `03_components` | Prepared profiles; decomposition on native and shared supports |
| `04_prediction` | Predictive panels, frozen models and supplied split metadata; baseline errors and G5 analyses |

After preparing all six datasets, run the `02_cross_dataset` stages in order:
`support`, `concordance`, then `matched_repeats`. The support stage checks the
reconstructed condition keys against the frozen matching selections; concordance
exports and verifies the shared gene axis. Then run the `03_components`
`intersection` stage.

## Baseline inference

Inspect `code/04_prediction/config.json` and `data/config/models.json` before
running the control-expression-only 3H2048 baseline:

```bash
drugperturbdb --repo . models-check
drugperturbdb --repo . baseline --threads 4 --include-decoder
```

Run Tahoe's `predictive_inputs` preparation stage first. The command reads the
registered prepared inputs and writes `result/04_prediction/computed/baseline/`.
Use `--workspace PATH` if preparation used a non-default location.

The optional `--include-decoder` flag also recomputes the measured-input G5
diagnostic and its chemical-group bootstrap intervals. Baseline reconstruction uses equal-physical-plate
aggregation; substituting cell-count weighting changes its inputs.
Predictive evaluation uses TRAIN/VALID roles, with no TEST evaluation command.

For the cross-plate comparison, run the `04_prediction` stages in order:
`cross_plate_G5`, `cross_plate_summary`, then `cross_plate_common_support`.
The final stage produces the manuscript estimates with equal chemical-group
weights on the shared plate support.

## Fixed exploratory comparisons

Figure S2 compares ten fixed configurations and three matched training-label
permutations per configuration. Prepare the Tahoe `predictive_inputs` stage, then:

```bash
python data/prepare.py download --dataset Prediction_features --max-gib 0.1
python data/prepare.py bind --dataset Prediction_features
python data/prepare.py build --dataset Prediction_features --stage features --cpus 1
python code/04_prediction/run.py --stage exploratory --execute --cpus 1
```

The annotated-target input averages L1000 genetic-perturbation signatures for
annotated drug targets (GSE106127); it is not a same-drug response measurement.
The predicted-target input uses a filtered ChEMBL activity reference.
The versioned feature archive contains external measurements and molecular
representations, with source provenance and checksums. It contains no Tahoe
response targets. Targets are constructed from the prepared TRAIN panels.
This command reproduces the displayed comparisons, including the alternative
held-block coordinate analysis; it does not repeat the historical model search.

## Inspecting results and figures

[Result indexes](../../../result/README.md) locate the numerical source tables.
Figure generation uses supplied tables and is independent of raw-data processing:

```bash
python code/02_cross_dataset/plot.py
drugperturbdb --repo . verify
drugperturbdb --repo . example-check
python -m pytest -q
```

Software checks are in `code/common/checks/`. They test implementation correctness;
they do not access held-out model TEST observations. See
[reproduction checks](VALIDATION.md) for the scope of the verified computations.

After completing all preparation and analysis stages, compare newly computed
endpoints with the manuscript's supplied tables:

```bash
drugperturbdb --repo . results-check
```

This comparison includes native/common repeatability, response composition,
matched-drug agreement and retrieval, matched repeatability, component prediction,
the G5 diagnostic and cross-plate comparison, and the ten Figure S2 configurations.
Missing outputs return `INCOMPLETE`; differences beyond the declared numerical
tolerance return `FAIL`. Reference tables are never overwritten.

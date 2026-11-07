# Numerical results

| Analysis | Table index | Source archive |
| --- | --- | --- |
| Within-source repeatability | [01_repeatability](01_repeatability/TABLE_INDEX.json) | Data S1 |
| Cross-dataset concordance | [02_cross_dataset](02_cross_dataset/TABLE_INDEX.json) | Data S1 |
| Response components | [03_components](03_components/TABLE_INDEX.json) | Data S1 |
| Prediction analyses | [04_prediction](04_prediction/TABLE_INDEX.json) | Data S2 |

Download [Data S1](source_tables/Data_S1_measurements.zip) or
[Data S2](source_tables/Data_S2_predictions.zip) and extract it to inspect the
tables. Each index records the archive member path, file size and SHA256.
The archives also contain their own `CONTENTS.json` manifests.

[figure_inputs/](figure_inputs/) holds the numerical tables used by the plotting
programs; [captions](../code/common/docs/FIGURE_LEGENDS.json) describe their support
and interpretation. Figures can be generated locally with each experiment's
`plot.py`.

New calculations write to `result/<experiment>/computed/`, separately from
these reference results.

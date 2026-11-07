# Reproduction checks

The source-data workflows were run in an isolated Linux CPU environment using
fresh source downloads, the supplied identity/split metadata and the ten frozen
models. Newly computed endpoints were compared with the manuscript tables.

| Scope | Recomputed checks |
| :--- | :--- |
| Six data sources | Native/common-gene repeatability and response composition |
| Cross-source comparisons | Matched-drug agreement, retrieval, shared-support composition and matched repeatability |
| Prediction | Five-fold baseline, component errors, G5 diagnostic and cross-plate mapping |
| Fixed exploratory comparisons | Ten Figure S2 configurations, each with three training-label permutations |
| Software | Checksums, weighted decomposition, constrained projection, correlation, bootstrap and path handling |

```bash
drugperturbdb --repo . verify
drugperturbdb --repo . models-check
drugperturbdb --repo . example-check
python -m pytest -q
# After source preparation and all analysis stages:
drugperturbdb --repo . results-check
```

`results-check` compares numerical endpoints at an absolute tolerance of 1e-8;
the baseline's rounded four-metric reference additionally uses 1e-6. Missing
outputs return `INCOMPLETE`, and discrepancies return `FAIL`. Figure rendering
uses the supplied result tables and does not substitute for recomputation.

The supplied drug identities, sample annotations, gene axes and split assignments
are versioned analysis inputs. This verification begins with those annotations;
it does not recreate historical manual curation. The ten exploratory comparisons
are reproduced, not the broader historical search that preceded them.

Prediction results remain development evidence. G5 uses measured drug-average
responses; its association with interaction responses is distinct from a
zero-shot prediction gain. The cross-plate results retain both directions in
Data S2.

# Data sources and conventions

[data/sources.json](../../../data/sources.json) records identifiers, URLs, file
sizes and SHA256 hashes. Downloads must match this registry. The two CIGS
20 μM count matrices are available in the [included input archive](../../../data/inputs/CIGS_TCM20_counts_v1.tar.gz),
because the source website has changed the contents served at those URLs.
The snapshot preserves unnormalized counts and the original source terms.

Source processing requires Linux/WSL and R for DRUG-seq. The verified environment
uses Python 3.12, CPU PyTorch 2.5.1 and R 4.6.1. Package versions are recorded in
[requirements-tested-lock.txt](../environment/requirements-tested-lock.txt).

- L1000 L3 analysis uses all 978 directly measured landmark genes and the shared
  gene subset. The 12,328-column source contains additional inferred genes;
  they are not included in the measured-gene endpoint. Gene and sample IDs are
  checked against the original GCTX axes before extraction.
- Tahoe descriptive analyses use the full eligible source panel. Prediction uses
  separate frozen TRAIN/VALID roles. Equal-physical-plate aggregation reconstructs
  the baseline inputs; cell-count weighting is not interchangeable with it.
- Drug and cell identity tables, gene axes, sample annotations and predictive
  splits are supplied in `data/metadata/`. Missing genes remain masked.
- Frozen model paths and hashes are in `data/config/models.json`. G5 files use
  Python pickle serialization; load only verified, trusted files.
- The fixed exploratory feature archive contains molecular representations and
  external measurements. Tahoe response targets are rebuilt from the prepared
  TRAIN inputs, not stored in that archive.

Follow [data preparation](../../../data/README.md) and the
[ordered workflows](WORKFLOW.md). See [reproduction checks](VALIDATION.md) for
the verified scope.

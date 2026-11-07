# Data preparation

`prepare.py` manages downloads and preparation for the six data sources.
Source-specific programs and ordered stages are in `datasets/<source>/config.json`.
[sources.json](sources.json) records source identifiers, URLs, expected sizes
and SHA256 hashes. [metadata/](metadata/) contains drug and cell identities,
gene axes, condition annotations and predictive splits.

## Example: sciPlex3

Run from the repository root after installing the environment:

```bash
python data/prepare.py plan --dataset sciPlex3
python data/prepare.py download --dataset sciPlex3 --max-gib 3
python data/prepare.py init
python data/prepare.py bind --dataset sciPlex3
python data/prepare.py check --dataset sciPlex3
python data/prepare.py build --dataset sciPlex3 --stage profiles --cpus 4
```

Use `plan` without a dataset argument to inspect all sources and their required
stages. The complete registered source downloads total approximately 147.6 GiB;
extracted working arrays require additional storage. Downloads require an explicit
size limit and a matching checksum.
Missing inputs, checksum mismatches and existing outputs stop the affected stage.

The two CIGS 20 μM count matrices use the fixed
[input archive](inputs/CIGS_TCM20_counts_v1.tar.gz)
included with the repository. It contains unnormalized counts, sample/gene identifiers and their
original source terms. The other CIGS inputs are downloaded from the source
study. The fixed exploratory feature archive is also included in [inputs/](inputs/).
All downloads are checked against the same SHA256 registry.

## Prepared inputs

Downloads are stored in `data/raw/<dataset>/`; generated analysis inputs are
stored in `data/prepared/_runtime/`. Binding uses hard links on the same filesystem.
All four analyses use these prepared inputs.

Expression profiles pair an HDF5 matrix (`post`, `delta`, `gene`, `entrez`) with
a Parquet condition table containing drug, cell, dose in micromolar, time in
hours, plate or replicate, normalization and row index. Missing measurements
remain masked. Native, common-gene and predictive gene axes are recorded
separately in the supplied metadata.

## Models and dependencies

[models/](models/) contains five control-expression-only 3H2048 models and five
G5 decoders. Paths, sizes and checksums are recorded in [config/models.json](config/models.json).
Check them with `drugperturbdb --repo . models-check` before inference.

Source processing requires Linux/WSL, with R for DRUG-seq and additional packages
listed in [data dependencies](../code/common/docs/DATA_RECOVERY.md).
See [reproduction checks](../code/common/docs/VALIDATION.md) for the verified scope.

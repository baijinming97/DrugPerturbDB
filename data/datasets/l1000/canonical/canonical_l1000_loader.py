from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pyarrow.parquet as pq


DEFAULT_ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000")
DATASETS = {
    "l3_treated": ("l3_treated_full_*-of-*.h5", "treatment_row"),
    "l3_control": ("l3_control_full_*-of-*.h5", "control_row"),
    "l5": ("l5_signatures_full_*-of-*.h5", "signature_row"),
}


class CanonicalL1000:
    """Random-access reader for the canonical, unpooled 978-gene L3/L5 assets."""

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.data = self.root / "data"
        self._shards: dict[str, list[tuple[int, int, Path]]] = {}

    def _index(self, dataset: str) -> list[tuple[int, int, Path]]:
        if dataset not in DATASETS:
            raise KeyError(dataset)
        if dataset not in self._shards:
            pattern, _ = DATASETS[dataset]
            indexed = []
            for path in sorted(self.data.glob(pattern)):
                with h5py.File(path, "r") as handle:
                    indexed.append(
                        (
                            int(handle.attrs["canonical_start"]),
                            int(handle.attrs["canonical_stop"]),
                            path,
                        )
                    )
            if not indexed:
                raise FileNotFoundError(f"No canonical shards for {dataset}")
            indexed.sort()
            if indexed[0][0] != 0 or any(a[1] != b[0] for a, b in zip(indexed, indexed[1:])):
                raise RuntimeError(f"Non-contiguous shard coverage for {dataset}")
            self._shards[dataset] = indexed
        return self._shards[dataset]

    def read(self, dataset: str, rows: Iterable[int], genes: Iterable[int] | None = None) -> np.ndarray:
        requested = np.asarray(list(rows), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("rows must be one-dimensional")
        if not len(requested):
            n_genes = 978 if genes is None else len(list(genes))
            return np.empty((0, n_genes), dtype=np.float32)
        shards = self._index(dataset)
        if requested.min() < 0 or requested.max() >= shards[-1][1]:
            raise IndexError(f"{dataset} row outside 0..{shards[-1][1] - 1}")
        gene_index = None if genes is None else np.asarray(list(genes), dtype=np.int64)
        if gene_index is not None and (gene_index.min(initial=0) < 0 or gene_index.max(initial=0) >= 978):
            raise IndexError("gene position outside 0..977")
        width = 978 if gene_index is None else len(gene_index)
        result = np.empty((len(requested), width), dtype=np.float32)
        for start, stop, path in shards:
            target_positions = np.flatnonzero((requested >= start) & (requested < stop))
            if not len(target_positions):
                continue
            local = requested[target_positions] - start
            unique_local, inverse = np.unique(local, return_inverse=True)
            with h5py.File(path, "r") as handle:
                values = np.asarray(handle["expression"][unique_local], dtype=np.float32)
            values = values[inverse]
            if gene_index is not None:
                values = values[:, gene_index]
            result[target_positions] = values
        return result

    def read_l3_treated(self, rows: Iterable[int], genes: Iterable[int] | None = None) -> np.ndarray:
        return self.read("l3_treated", rows, genes)

    def read_l3_controls(self, rows: Iterable[int], genes: Iterable[int] | None = None) -> np.ndarray:
        return self.read("l3_control", rows, genes)

    def read_l5(self, rows: Iterable[int], genes: Iterable[int] | None = None) -> np.ndarray:
        return self.read("l5", rows, genes)

    def metadata(self, dataset: str, columns: list[str] | None = None):
        files = {
            "l3_treated": "l3_treatment_manifest.parquet",
            "l3_control": "l3_control_manifest.parquet",
            "l5": "l5_signature_manifest.parquet",
        }
        return pq.read_table(self.data / files[dataset], columns=columns)

    def control_rows_for_treatments(self, treatment_rows: Iterable[int]) -> list[list[int]]:
        rows = np.asarray(list(treatment_rows), dtype=np.int64)
        treatment = pq.read_table(
            self.data / "l3_treatment_manifest.parquet",
            columns=["treatment_row", "control_context_id"],
        ).to_pandas()
        if len(rows) and (rows.min() < 0 or rows.max() >= len(treatment)):
            raise IndexError("treatment row outside manifest")
        contexts = pq.read_table(
            self.data / "l3_control_context_manifest.parquet",
            columns=["control_context_id", "control_rows"],
        ).to_pandas().set_index("control_context_id")["control_rows"].to_dict()
        return [list(map(int, contexts.get(int(treatment.iloc[row].control_context_id), []))) for row in rows]


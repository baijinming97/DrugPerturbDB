from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


class TahoeCanonicalDataset:
    """Read compact canonical Tahoe treated/control pseudobulk subsets."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.conditions = pq.read_table(self.root / "data/condition_manifest.parquet").to_pandas()
        self.gene_axis = pq.read_table(self.root / "data/gene_axis.parquet").to_pandas()
        self.control_map = pq.read_table(self.root / "data/condition_control_map.parquet").to_pandas()
        self._gene_position = dict(
            zip(self.gene_axis["feature"].astype(str), self.gene_axis["gene_position"].astype(int))
        )

    def gene_positions(self, genes: list[str]) -> np.ndarray:
        missing = [gene for gene in genes if gene not in self._gene_position]
        if missing:
            raise KeyError(f"Genes absent from Tahoe axis: {missing[:20]}")
        return np.asarray([self._gene_position[gene] for gene in genes], dtype=np.int64)

    def read(
        self,
        condition_rows: np.ndarray | list[int],
        gene_positions: np.ndarray | list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        rows = np.asarray(condition_rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(self.conditions)):
            raise IndexError("condition_rows must be a one-dimensional in-range integer array")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("condition_rows must not contain duplicates")
        if gene_positions is None:
            positions = np.arange(len(self.gene_axis), dtype=np.int64)
        else:
            positions = np.asarray(gene_positions, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= len(self.gene_axis)):
            raise IndexError("gene_positions must be a one-dimensional in-range integer array")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("gene_positions must not contain duplicates")
        sort_order = np.argsort(positions)
        sorted_positions = positions[sort_order]
        restore_order = np.argsort(sort_order)

        treated = np.empty((len(rows), len(positions)), dtype=np.float32)
        requested = self.conditions.iloc[rows].copy()
        requested["request_row"] = np.arange(len(rows), dtype=np.int64)
        for shard, frame in requested.groupby("treated_shard", sort=False):
            path = self.root / "data" / f"treated_{int(shard):02d}-of-16.h5"
            with h5py.File(path, "r") as handle:
                for record in frame.to_dict("records"):
                    values = np.asarray(
                        handle["treated"][int(record["treated_row_in_shard"]), sorted_positions],
                        dtype=np.float32,
                    )
                    treated[int(record["request_row"])] = values[restore_order]

        condition_to_request = {int(row): i for i, row in enumerate(rows)}
        relevant = self.control_map[self.control_map["condition_row"].isin(condition_to_request)].copy()
        control_rows = np.sort(relevant["control_row"].unique().astype(np.int64))
        with h5py.File(self.root / "data/controls.h5", "r") as handle:
            controls_sorted = np.asarray(handle["controls"][control_rows][:, sorted_positions], dtype=np.float32)
        controls_subset = controls_sorted[:, restore_order]
        control_lookup = {int(row): i for i, row in enumerate(control_rows)}
        control = np.zeros_like(treated, dtype=np.float64)
        weight_sums = np.zeros(len(rows), dtype=np.float64)
        for record in relevant.to_dict("records"):
            request_row = condition_to_request[int(record["condition_row"])]
            weight = float(record["control_weight"])
            control[request_row] += weight * controls_subset[control_lookup[int(record["control_row"])]].astype(
                np.float64
            )
            weight_sums[request_row] += weight
        if not np.allclose(weight_sums, 1.0, atol=1e-7):
            raise RuntimeError("Condition control weights do not sum to one")
        control32 = control.astype(np.float32)
        return {
            "condition_rows": rows,
            "gene_positions": positions,
            "treated": treated,
            "control": control32,
            "delta": treated - control32,
        }

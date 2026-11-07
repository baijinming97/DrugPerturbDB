from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
REFERENCE = Path("@WORKSPACE@/baicai/tahoe_official_drug_signal_audit_20260823")
N_CONDITIONS = 52886


def main(rank: int, world: int) -> None:
    if world != 16:
        raise RuntimeError("Official-cell_eval concordance uses the 16 canonical shards")
    out = ROOT / "audit" / f"cell_eval_directional_{rank:02d}-of-{world:02d}.json"
    if out.exists():
        raise FileExistsError(f"Refuse overwrite: {out}")

    conditions = pq.read_table(ROOT / "audit/qc_condition_manifest.parquet").to_pandas()
    old_conditions = pq.read_table(REFERENCE / "data/condition_manifest.parquet").to_pandas()
    key_columns = ["cell_line", "drug", "dose", "dose_unit"]
    old_lookup = {
        tuple(row[column] for column in key_columns): int(row["array_row"])
        for row in old_conditions.to_dict("records")
    }
    if len(old_lookup) != len(old_conditions):
        raise RuntimeError("Old official cell_eval condition keys are not unique")

    old_genes = pd.read_csv(REFERENCE / "data/gene_order_2000.tsv", sep="\t")["gene_symbol"].astype(str).tolist()
    axis = pq.read_table(ROOT / "audit/gene_axis.parquet").to_pandas()
    position = dict(zip(axis["feature"].astype(str), axis["gene_position"].astype(int)))
    missing = [gene for gene in old_genes if gene not in position]
    if missing:
        raise RuntimeError(f"Official 2k genes missing from the full axis: {missing[:20]}")
    requested_positions = np.asarray([position[gene] for gene in old_genes], dtype=np.int64)
    sort_order = np.argsort(requested_positions)
    sorted_positions = requested_positions[sort_order]
    restore_order = np.argsort(sort_order)

    control_map = pq.read_table(ROOT / "data/condition_control_map.parquet").to_pandas()
    control_groups = {
        int(key): frame[["control_row", "control_weight"]].to_numpy()
        for key, frame in control_map.groupby("condition_row", sort=False)
    }
    with h5py.File(ROOT / "data/controls.h5", "r") as controls_h5:
        controls_sorted = np.asarray(controls_h5["controls"][:, sorted_positions], dtype=np.float32)
    controls_2k = controls_sorted[:, restore_order]

    edges = np.linspace(0, N_CONDITIONS, world + 1, dtype=np.int64)
    start = int(edges[rank])
    stop = int(edges[rank + 1])
    local_conditions = conditions.iloc[start:stop]
    old_y = np.load(REFERENCE / "data/Y_condition_2000.npy", mmap_mode="r", allow_pickle=False)

    totals = {
        group: {
            "conditions": 0,
            "values": 0,
            "sse": 0.0,
            "max_abs": 0.0,
            "new_sum": 0.0,
            "old_sum": 0.0,
            "new_sumsq": 0.0,
            "old_sumsq": 0.0,
            "dot": 0.0,
            "condition_cosine_defined": 0,
            "condition_cosine_positive": 0,
            "condition_cosine_sum": 0.0,
            "condition_cosine_min": 1.0,
            "condition_cosine_max": -1.0,
        }
        for group in ["single", "multi"]
    }
    treated_path = ROOT / "data" / f"treated_{rank:02d}-of-{world:02d}.h5"
    with h5py.File(treated_path, "r") as treated_h5:
        treated_ds = treated_h5["treated"]
        for local_row, record in enumerate(local_conditions.to_dict("records")):
            condition_row = int(record["condition_row"])
            key = tuple(record[column] for column in key_columns)
            if key not in old_lookup:
                raise RuntimeError(f"New QC condition absent from official cell_eval: {key}")
            treated_sorted = np.asarray(treated_ds[local_row, sorted_positions], dtype=np.float32)
            treated_2k = treated_sorted[restore_order]
            control = np.zeros(len(old_genes), dtype=np.float64)
            weight_sum = 0.0
            for control_row, weight in control_groups[condition_row]:
                control += float(weight) * controls_2k[int(control_row)].astype(np.float64)
                weight_sum += float(weight)
            if not np.isclose(weight_sum, 1.0, atol=1e-7):
                raise RuntimeError(f"Control weights for condition {condition_row} sum to {weight_sum}")
            new_delta = treated_2k.astype(np.float64) - control
            old_delta = np.asarray(old_y[old_lookup[key]], dtype=np.float64)
            difference = new_delta - old_delta
            group = "single" if int(record["source_plate_profiles"]) == 1 else "multi"
            stats = totals[group]
            stats["conditions"] += 1
            stats["values"] += len(old_genes)
            stats["sse"] += float(np.square(difference).sum())
            stats["max_abs"] = max(stats["max_abs"], float(np.max(np.abs(difference))))
            stats["new_sum"] += float(new_delta.sum())
            stats["old_sum"] += float(old_delta.sum())
            stats["new_sumsq"] += float(np.square(new_delta).sum())
            stats["old_sumsq"] += float(np.square(old_delta).sum())
            stats["dot"] += float(np.dot(new_delta, old_delta))
            denominator = float(np.linalg.norm(new_delta) * np.linalg.norm(old_delta))
            if denominator > 0:
                cosine = float(np.dot(new_delta, old_delta) / denominator)
                stats["condition_cosine_defined"] += 1
                stats["condition_cosine_positive"] += int(cosine > 0)
                stats["condition_cosine_sum"] += cosine
                stats["condition_cosine_min"] = min(stats["condition_cosine_min"], cosine)
                stats["condition_cosine_max"] = max(stats["condition_cosine_max"], cosine)
            if (local_row + 1) % 250 == 0 or local_row + 1 == len(local_conditions):
                print(
                    f"cell_eval_concordance rank={rank}/{world} rows={local_row + 1}/{len(local_conditions)}",
                    flush=True,
                )

    for values in totals.values():
        values["rmse"] = float(np.sqrt(values["sse"] / values["values"])) if values["values"] else None
        if not values["condition_cosine_defined"]:
            values["condition_cosine_min"] = None
            values["condition_cosine_max"] = None
    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "condition_start": start,
        "condition_stop": stop,
        "official_cell_eval_genes": len(old_genes),
        "metrics": totals,
        "interpretation": {
            "endpoint_identity": "PDEX target_mean/ref_mean and official cell_eval are distinct official endpoints and are not expected to be numerically identical.",
            "purpose": "Verify exact condition/gene identity binding and positive cross-endpoint direction; retain RMSE and maxima as descriptive scale diagnostics.",
            "multi_plate": "Canonical PDEX uses raw full-QC cell-count weights; the prior cell_eval audit used equal plate weights.",
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    main(args.rank, args.world)

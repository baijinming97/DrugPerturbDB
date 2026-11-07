from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
URL = str(Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1/raw/all_plates_pdex.parquet'))
if not Path(URL).is_file():
    raise FileNotFoundError('Download and bind the registered Tahoe pdex source first')

COLUMNS = [
    "target",
    "feature",
    "target_mean",
    "ref_mean",
    "target_membership",
    "ref_membership",
    "plate",
    "cell_line",
]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_mask(table: pa.Table, record: dict) -> pa.Array:
    return pc.and_(
        pc.and_(
            pc.equal(table["plate"], pa.scalar(int(record["plate"]), type=table["plate"].type)),
            pc.equal(table["cell_line"], pa.scalar(str(record["cell_line"]), type=table["cell_line"].type)),
        ),
        pc.equal(table["target"], pa.scalar(str(record["target"]), type=table["target"].type)),
    )


def main(rank: int, world: int, tag: str, limit_profiles: int) -> None:
    gene_axis = pq.read_table(ROOT / "audit/gene_axis.parquet").to_pandas().sort_values("gene_position")
    features = gene_axis["feature"].astype(str).tolist()
    n_genes = len(features)
    if n_genes != 62710 or len(set(features)) != n_genes:
        raise RuntimeError("Frozen full-gene axis is not 62,710 unique entries")
    feature_position = {feature: i for i, feature in enumerate(features)}

    profile_manifest = pq.read_table(ROOT / "audit/qc_plate_profile_manifest.parquet").to_pandas()
    if not (profile_manifest["row_group_count"] == 1).all():
        bad = profile_manifest.loc[profile_manifest["row_group_count"] != 1]
        raise RuntimeError(f"Profiles split across PDEX row groups are unsupported: {bad.head().to_dict('records')}")
    profile_manifest["source_row_group"] = profile_manifest["row_groups"].astype(int)
    row_group_edges = np.linspace(0, 13045, world + 1, dtype=np.int64)
    row_group_start = int(row_group_edges[rank])
    row_group_stop = int(row_group_edges[rank + 1])
    selected_manifest = profile_manifest.loc[
        (profile_manifest["source_row_group"] >= row_group_start)
        & (profile_manifest["source_row_group"] < row_group_stop)
    ].copy()
    selected_manifest = selected_manifest.sort_values(
        ["source_row_group", "profile_row"], kind="mergesort"
    ).reset_index(drop=True)
    if limit_profiles:
        selected_manifest = selected_manifest.iloc[:limit_profiles].copy()

    by_row_group = defaultdict(list)
    for record in selected_manifest.to_dict("records"):
        by_row_group[int(record["source_row_group"])].append(record)

    stem = f"pdex_values_{tag}_{rank:02d}-of-{world:02d}"
    out_h5 = ROOT / "tmp" / f"{stem}.h5"
    out_meta = ROOT / "tmp" / f"{stem}.parquet"
    out_receipt = ROOT / "tmp" / f"{stem}.json"
    partial_h5 = out_h5.with_name(out_h5.name + ".part")
    if any(path.exists() for path in [out_h5, out_meta, out_receipt, partial_h5]):
        raise FileExistsError(f"Refuse overwrite of extraction shard {stem}")

    metadata = []
    context_to_hash = {}
    hash_to_control_row = {}
    control_records = []
    output_row = 0
    treated_min = np.inf
    treated_max = -np.inf
    control_min = np.inf
    control_max = -np.inf

    with h5py.File(partial_h5, "w") as h5:
        h5.attrs["format"] = "Tahoe-PDEX-QC-plate-shard-v1"
        h5.attrs["pdex_revision"] = "c7963cf334bec0683225d41c9586d900ca6303a2"
        h5.attrs["rank"] = rank
        h5.attrs["world"] = world
        treated_ds = h5.create_dataset(
            "treated",
            shape=(len(selected_manifest), n_genes),
            dtype="f4",
            chunks=(1, n_genes),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )
        control_ds = h5.create_dataset(
            "controls",
            shape=(0, n_genes),
            maxshape=(None, n_genes),
            dtype="f4",
            chunks=(1, n_genes),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )

        with fsspec.open(
            URL,
            "rb",
            block_size=16 * 1024 * 1024,
            cache_type="readahead",
            client_kwargs={"trust_env": True},
        ) as handle:
            pf = pq.ParquetFile(handle)
            for rg_pos, row_group in enumerate(sorted(by_row_group)):
                table = pf.read_row_group(row_group, columns=COLUMNS, use_threads=False)
                for record in by_row_group[row_group]:
                    selected = table.filter(make_mask(table, record))
                    if selected.num_rows != n_genes:
                        raise RuntimeError(
                            f"Incomplete profile {record['profile_row']} in row group {row_group}: "
                            f"{selected.num_rows} != {n_genes}"
                        )

                    selected_features = selected["feature"].combine_chunks()
                    expected_features = pa.array(features, type=selected_features.type)
                    if selected_features.equals(expected_features):
                        order = None
                    else:
                        positions = np.asarray(
                            [feature_position.get(str(value), -1) for value in selected_features.to_pylist()],
                            dtype=np.int64,
                        )
                        if np.any(positions < 0) or len(np.unique(positions)) != n_genes:
                            raise RuntimeError(f"Non-canonical feature set for profile {record['profile_row']}")
                        order = np.argsort(positions)

                    treated = np.asarray(
                        selected["target_mean"].combine_chunks().to_numpy(zero_copy_only=False),
                        dtype=np.float32,
                    )
                    control = np.asarray(
                        selected["ref_mean"].combine_chunks().to_numpy(zero_copy_only=False),
                        dtype=np.float32,
                    )
                    if order is not None:
                        treated = treated[order]
                        control = control[order]
                    if not np.isfinite(treated).all() or not np.isfinite(control).all():
                        raise RuntimeError(f"Non-finite expression for profile {record['profile_row']}")

                    target_membership = np.unique(
                        np.asarray(
                            selected["target_membership"].combine_chunks().to_numpy(zero_copy_only=False),
                            dtype=np.int64,
                        )
                    )
                    ref_membership = np.unique(
                        np.asarray(
                            selected["ref_membership"].combine_chunks().to_numpy(zero_copy_only=False),
                            dtype=np.int64,
                        )
                    )
                    if len(target_membership) != 1 or int(target_membership[0]) != int(record["target_membership"]):
                        raise RuntimeError(f"target_membership mismatch for profile {record['profile_row']}")
                    if len(ref_membership) != 1 or int(ref_membership[0]) != int(record["ref_membership"]):
                        raise RuntimeError(f"ref_membership mismatch for profile {record['profile_row']}")

                    treated_ds[output_row] = treated
                    control_hash = hashlib.sha256(control.tobytes(order="C")).hexdigest()
                    context = (str(record["cell_line"]), int(record["plate"]))
                    prior_hash = context_to_hash.setdefault(context, control_hash)
                    if prior_hash != control_hash:
                        raise RuntimeError(f"Control vector varies within cell/plate context {context}")
                    if control_hash not in hash_to_control_row:
                        control_row = len(hash_to_control_row)
                        hash_to_control_row[control_hash] = control_row
                        control_ds.resize((control_row + 1, n_genes))
                        control_ds[control_row] = control
                        control_records.append(
                            {
                                "temp_control_row": control_row,
                                "control_hash": control_hash,
                                "cell_line": context[0],
                                "plate": context[1],
                                "ref_membership": int(record["ref_membership"]),
                            }
                        )
                    control_row = hash_to_control_row[control_hash]

                    metadata.append(
                        {
                            "temp_value_row": output_row,
                            "profile_row": int(record["profile_row"]),
                            "condition_row": int(record["condition_row"]),
                            "source_row_group": int(row_group),
                            "cell_line": str(record["cell_line"]),
                            "drug": str(record["drug"]),
                            "dose": float(record["dose"]),
                            "plate": int(record["plate"]),
                            "target_membership": int(record["target_membership"]),
                            "ref_membership": int(record["ref_membership"]),
                            "aggregation_weight": float(record["aggregation_weight"]),
                            "temp_control_row": int(control_row),
                            "control_hash": control_hash,
                        }
                    )
                    output_row += 1
                    treated_min = min(treated_min, float(treated.min()))
                    treated_max = max(treated_max, float(treated.max()))
                    control_min = min(control_min, float(control.min()))
                    control_max = max(control_max, float(control.max()))

                if (rg_pos + 1) % 25 == 0 or rg_pos + 1 == len(by_row_group):
                    print(
                        f"rank={rank}/{world} row_groups={rg_pos + 1}/{len(by_row_group)} "
                        f"profiles={output_row}/{len(selected_manifest)} controls={len(hash_to_control_row)}",
                        flush=True,
                    )

        if output_row != len(selected_manifest):
            raise RuntimeError(f"Extraction count mismatch {output_row} != {len(selected_manifest)}")
        h5.attrs["profiles"] = output_row
        h5.attrs["controls"] = len(hash_to_control_row)
        h5.flush()

    os.replace(partial_h5, out_h5)
    meta_frame = pd.DataFrame(metadata)
    control_frame = pd.DataFrame(control_records)
    meta_table = pa.Table.from_pandas(meta_frame, preserve_index=False)
    control_table = pa.Table.from_pandas(control_frame, preserve_index=False)
    combined = pa.table({}) if len(metadata) == 0 else meta_table
    if len(control_records):
        # Store control metadata as a nested list in a separate schema-compatible sidecar table.
        control_sidecar = out_meta.with_name(out_meta.stem + "_controls.parquet")
        control_tmp = control_sidecar.with_name(control_sidecar.name + ".part")
        pq.write_table(control_table, control_tmp, compression="zstd")
        os.replace(control_tmp, control_sidecar)
    else:
        control_sidecar = out_meta.with_name(out_meta.stem + "_controls.parquet")
        pq.write_table(control_table, control_sidecar, compression="zstd")
    meta_tmp = out_meta.with_name(out_meta.name + ".part")
    pq.write_table(combined, meta_tmp, compression="zstd")
    os.replace(meta_tmp, out_meta)

    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "tag": tag,
        "profiles": output_row,
        "controls_local_unique": len(hash_to_control_row),
        "genes": n_genes,
        "row_groups": len(by_row_group),
        "row_group_start": row_group_start,
        "row_group_stop": row_group_stop,
        "treated_range": [treated_min, treated_max] if output_row else None,
        "control_range": [control_min, control_max] if output_row else None,
        "h5": str(out_h5),
        "h5_bytes": out_h5.stat().st_size,
        "h5_sha256": sha256sum(out_h5),
        "metadata": str(out_meta),
        "metadata_sha256": sha256sum(out_meta),
        "control_metadata": str(control_sidecar),
        "control_metadata_sha256": sha256sum(control_sidecar),
    }
    out_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit-profiles", type=int, default=0)
    args = parser.parse_args()
    main(args.rank, args.world, args.tag, args.limit_profiles)

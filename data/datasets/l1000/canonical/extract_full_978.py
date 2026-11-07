from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000")
CONFIG = {
    "l3_treated": {
        "source": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_trt_cp_n1805898x12328.gctx"),
        "manifest": ROOT / "data/l3_treatment_manifest.parquet",
        "row_column": "treatment_row",
        "gene_column": "l3_treated_source_gene_position",
        "stem": "l3_treated",
    },
    "l3_control": {
        "source": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_ctl_n188708x12328.gctx"),
        "manifest": ROOT / "data/l3_control_manifest.parquet",
        "row_column": "control_row",
        "gene_column": "l3_control_source_gene_position",
        "stem": "l3_control",
    },
    "l5": {
        "source": Path("@WORKSPACE@/baicai/l3_l5_978_prep_20260822/raw/level5_beta_trt_cp_n720216x12328.gctx"),
        "manifest": ROOT / "data/l5_signature_manifest.parquet",
        "row_column": "signature_row",
        "gene_column": "l5_source_gene_position",
        "stem": "l5_signatures",
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(dataset: str, rank: int, world: int, tag: str, limit_rows: int) -> None:
    cfg = CONFIG[dataset]
    manifest = pq.read_table(
        cfg["manifest"], columns=[cfg["row_column"], "source_profile_index"]
    ).to_pandas()
    n_total = len(manifest)
    expected_rows = np.arange(n_total, dtype=np.int64)
    if not np.array_equal(manifest[cfg["row_column"]].to_numpy(), expected_rows):
        raise RuntimeError(f"{dataset} canonical rows are not exact and contiguous")
    source_indices_all = manifest["source_profile_index"].to_numpy(dtype=np.int64)
    if np.any(np.diff(source_indices_all) <= 0):
        raise RuntimeError(f"{dataset} source profile indices are not strictly increasing")

    edges = np.linspace(0, n_total, world + 1, dtype=np.int64)
    canonical_start = int(edges[rank])
    canonical_stop = int(edges[rank + 1])
    if limit_rows:
        canonical_stop = min(canonical_stop, canonical_start + limit_rows)
    source_indices = source_indices_all[canonical_start:canonical_stop]
    n_local = len(source_indices)
    if not n_local:
        raise RuntimeError(f"Empty extraction partition {dataset} {rank}/{world}")

    axis = pq.read_table(
        ROOT / "data/gene_axis_978.parquet",
        columns=["axis_position", cfg["gene_column"]],
    ).to_pandas().sort_values("axis_position")
    gene_positions = axis[cfg["gene_column"]].to_numpy(dtype=np.int64)
    if len(gene_positions) != 978 or len(np.unique(gene_positions)) != 978:
        raise RuntimeError(f"Invalid source gene binding for {dataset}")
    gene_order = np.argsort(gene_positions)
    sorted_gene_positions = gene_positions[gene_order]
    restore_gene_order = np.argsort(gene_order)

    directory = ROOT / ("data" if tag == "full" else "tmp")
    stem = f"{cfg['stem']}_{tag}_{rank:02d}-of-{world:02d}"
    out = directory / f"{stem}.h5"
    receipt = ROOT / ("audit" if tag == "full" else "tmp") / f"{stem}.json"
    partial = out.with_name(out.name + ".part")
    if any(path.exists() for path in [out, receipt, partial]):
        raise FileExistsError(f"Refuse overwrite: {stem}")

    value_min = np.inf
    value_max = -np.inf
    write_position = 0
    block_width = 2048
    with h5py.File(cfg["source"], "r") as source, h5py.File(partial, "w") as target:
        matrix = source["0/DATA/0/matrix"]
        if matrix.shape[1] <= int(sorted_gene_positions[-1]):
            raise RuntimeError(f"Source gene position exceeds matrix width for {dataset}")
        target.attrs["format"] = "LINCS-L1000-all-QC-pass-978-v1"
        target.attrs["dataset"] = dataset
        target.attrs["tag"] = tag
        target.attrs["canonical_start"] = canonical_start
        target.attrs["canonical_stop"] = canonical_stop
        target.attrs["source_gctx"] = str(cfg["source"])
        target.create_dataset(
            "canonical_row",
            data=np.arange(canonical_start, canonical_stop, dtype=np.int64),
        )
        target.create_dataset("source_profile_index", data=source_indices)
        expression = target.create_dataset(
            "expression",
            shape=(n_local, 978),
            dtype="f4",
            chunks=(min(256, n_local), 978),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )

        pointer = 0
        while pointer < n_local:
            source_block_start = int(source_indices[pointer] // block_width * block_width)
            source_block_stop = min(source_block_start + block_width, matrix.shape[0])
            end_pointer = int(np.searchsorted(source_indices, source_block_stop, side="left"))
            relative_rows = source_indices[pointer:end_pointer] - source_block_start
            block = np.asarray(
                matrix[source_block_start:source_block_stop, sorted_gene_positions]
            )
            selected = np.asarray(block[relative_rows][:, restore_gene_order], dtype=np.float32)
            if not np.isfinite(selected).all():
                raise RuntimeError(
                    f"Non-finite {dataset} values near canonical row {canonical_start + pointer}"
                )
            expression[write_position : write_position + len(selected)] = selected
            value_min = min(value_min, float(selected.min()))
            value_max = max(value_max, float(selected.max()))
            write_position += len(selected)
            pointer = end_pointer
            if write_position % 10000 < len(selected) or write_position == n_local:
                print(
                    f"extract {dataset} rank={rank}/{world} rows={write_position}/{n_local}",
                    flush=True,
                )
        if write_position != n_local:
            raise RuntimeError(f"Extraction row count mismatch {write_position} != {n_local}")
        target.flush()

    os.replace(partial, out)
    payload = {
        "status": "PASS",
        "dataset": dataset,
        "tag": tag,
        "rank": rank,
        "world": world,
        "canonical_start": canonical_start,
        "canonical_stop": canonical_stop,
        "rows": n_local,
        "genes": 978,
        "source_profile_index_min": int(source_indices.min()),
        "source_profile_index_max": int(source_indices.max()),
        "value_range": [value_min, value_max],
        "file": str(out),
        "bytes": out.stat().st_size,
        "sha256": sha256sum(out),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CONFIG), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()
    main(args.dataset, args.rank, args.world, args.tag, args.limit_rows)

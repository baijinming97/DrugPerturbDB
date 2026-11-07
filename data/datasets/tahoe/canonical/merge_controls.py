from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
N_CONDITIONS = 52886
N_GENES = 62710


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        # Supplied fixed annotations are checked against the reconstructed values.
        expected = pd.read_parquet(path)
        columns = [c for c in frame.columns if not c.startswith("source_temp_")]
        if not set(columns).issubset(expected.columns):
            raise ValueError(f"Incompatible registered control metadata: {path}")
        pd.testing.assert_frame_equal(
            frame[columns].reset_index(drop=True), expected[columns].reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10,
        )
        return
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)


def main(extraction_tag: str, extraction_world: int) -> None:
    main_frames = []
    control_frames = []
    main_pattern = re.compile(rf"pdex_values_{re.escape(extraction_tag)}_(\d+)-of-{extraction_world:02d}\.parquet$")
    main_paths = sorted(ROOT.glob(f"tmp/pdex_values_{extraction_tag}_*-of-{extraction_world:02d}.parquet"))
    control_paths = sorted(
        ROOT.glob(f"tmp/pdex_values_{extraction_tag}_*-of-{extraction_world:02d}_controls.parquet")
    )
    if len(main_paths) != extraction_world or len(control_paths) != extraction_world:
        raise RuntimeError(
            f"Expected {extraction_world} main/control shards, found {len(main_paths)}/{len(control_paths)}"
        )
    for path in main_paths:
        match = main_pattern.search(path.name)
        if not match:
            raise RuntimeError(f"Unexpected metadata filename: {path}")
        frame = pq.read_table(path).to_pandas()
        frame["temp_rank"] = int(match.group(1))
        main_frames.append(frame)
    for path in control_paths:
        rank = int(path.name.split(f"{extraction_tag}_", 1)[1].split("-of-", 1)[0])
        frame = pq.read_table(path).to_pandas()
        frame["temp_rank"] = rank
        control_frames.append(frame)

    values = pd.concat(main_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True)
    if len(values) == 0 or len(controls) == 0:
        raise RuntimeError("Extraction metadata is empty")
    if values["profile_row"].duplicated().any():
        raise RuntimeError("Duplicate profile_row across extraction metadata")
    if values["condition_row"].nunique() != N_CONDITIONS:
        raise RuntimeError(f"Condition coverage {values['condition_row'].nunique()} != {N_CONDITIONS}")

    context_hashes = values.groupby(["cell_line", "plate"])["control_hash"].nunique()
    if (context_hashes != 1).any():
        raise RuntimeError("A cell/plate context maps to multiple control vectors")
    hash_sources = (
        controls.sort_values(["control_hash", "temp_rank", "temp_control_row"], kind="mergesort")
        .drop_duplicates("control_hash", keep="first")
        .reset_index(drop=True)
    )
    hash_sources.insert(0, "control_row", np.arange(len(hash_sources), dtype=np.int64))
    hash_to_row = dict(zip(hash_sources["control_hash"], hash_sources["control_row"]))

    out_h5 = ROOT / "data/controls.h5"
    partial = out_h5.with_name(out_h5.name + ".part")
    if out_h5.exists() or partial.exists():
        raise FileExistsError("Refuse overwrite of canonical controls.h5")
    value_min = np.inf
    value_max = -np.inf
    handles = {}
    with h5py.File(partial, "w") as out:
        out.attrs["format"] = "Tahoe-PDEX-full-QC-controls-v1"
        out.attrs["pdex_revision"] = "c7963cf334bec0683225d41c9586d900ca6303a2"
        ds = out.create_dataset(
            "controls",
            shape=(len(hash_sources), N_GENES),
            dtype="f4",
            chunks=(1, N_GENES),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )
        try:
            for row in hash_sources.to_dict("records"):
                rank = int(row["temp_rank"])
                if rank not in handles:
                    path = ROOT / "tmp" / (
                        f"pdex_values_{extraction_tag}_{rank:02d}-of-{extraction_world:02d}.h5"
                    )
                    handles[rank] = h5py.File(path, "r")
                vector = np.asarray(
                    handles[rank]["controls"][int(row["temp_control_row"])], dtype=np.float32
                )
                digest = hashlib.sha256(vector.tobytes(order="C")).hexdigest()
                if digest != row["control_hash"]:
                    raise RuntimeError(f"Control hash mismatch at canonical row {row['control_row']}")
                if not np.isfinite(vector).all():
                    raise RuntimeError(f"Non-finite control at canonical row {row['control_row']}")
                ds[int(row["control_row"])] = vector
                value_min = min(value_min, float(vector.min()))
                value_max = max(value_max, float(vector.max()))
        finally:
            for handle in handles.values():
                handle.close()
        out.flush()
    os.replace(partial, out_h5)

    control_manifest = hash_sources[
        ["control_row", "control_hash", "temp_rank", "temp_control_row"]
    ].rename(columns={"temp_rank": "source_temp_rank", "temp_control_row": "source_temp_control_row"})
    context_manifest = values[
        ["cell_line", "plate", "ref_membership", "control_hash"]
    ].drop_duplicates()
    if context_manifest.duplicated(["cell_line", "plate"]).any():
        raise RuntimeError("Control context metadata is not unique")
    context_manifest["control_row"] = context_manifest["control_hash"].map(hash_to_row).astype(np.int64)
    context_manifest = context_manifest.sort_values(["cell_line", "plate"], kind="mergesort")

    condition_control = (
        values.groupby(["condition_row", "control_hash"], sort=True, as_index=False)["aggregation_weight"]
        .sum()
        .rename(columns={"aggregation_weight": "control_weight"})
    )
    condition_control["control_row"] = condition_control["control_hash"].map(hash_to_row).astype(np.int64)
    sums = condition_control.groupby("condition_row")["control_weight"].sum().to_numpy()
    if len(sums) != N_CONDITIONS or not np.allclose(sums, 1.0, atol=1e-7):
        raise RuntimeError("Condition-to-control weights are incomplete or do not sum to one")
    condition_control = condition_control[
        ["condition_row", "control_row", "control_hash", "control_weight"]
    ].sort_values(["condition_row", "control_row"], kind="mergesort")

    control_manifest_path = ROOT / "data/control_manifest.parquet"
    context_manifest_path = ROOT / "data/control_context_manifest.parquet"
    condition_control_path = ROOT / "data/condition_control_map.parquet"
    write_parquet_atomic(control_manifest, control_manifest_path)
    write_parquet_atomic(context_manifest, context_manifest_path)
    write_parquet_atomic(condition_control, condition_control_path)

    payload = {
        "status": "PASS",
        "genes": N_GENES,
        "unique_control_vectors": int(len(hash_sources)),
        "cell_plate_contexts": int(len(context_manifest)),
        "condition_control_map_rows": int(len(condition_control)),
        "control_range": [value_min, value_max],
        "controls_h5": str(out_h5),
        "controls_h5_bytes": out_h5.stat().st_size,
        "controls_h5_sha256": sha256sum(out_h5),
        "control_manifest_sha256": sha256sum(control_manifest_path),
        "control_context_manifest_sha256": sha256sum(context_manifest_path),
        "condition_control_map_sha256": sha256sum(condition_control_path),
    }
    audit = ROOT / "audit/control_merge_audit.json"
    if audit.exists():
        raise FileExistsError(f"Refuse overwrite: {audit}")
    audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-tag", default="full32")
    parser.add_argument("--extraction-world", type=int, default=32)
    args = parser.parse_args()
    main(args.extraction_tag, args.extraction_world)

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


def load_temp_metadata(tag: str, extraction_world: int) -> pd.DataFrame:
    frames = []
    pattern = re.compile(rf"pdex_values_{re.escape(tag)}_(\d+)-of-{extraction_world:02d}\.parquet$")
    paths = sorted(ROOT.glob(f"tmp/pdex_values_{tag}_*-of-{extraction_world:02d}.parquet"))
    if len(paths) != extraction_world:
        raise RuntimeError(f"Expected {extraction_world} extraction metadata shards, found {len(paths)}")
    for path in paths:
        match = pattern.search(path.name)
        if not match:
            raise RuntimeError(f"Unexpected extraction metadata name: {path}")
        frame = pq.read_table(path).to_pandas()
        frame["temp_rank"] = int(match.group(1))
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata["profile_row"].duplicated().any():
        raise RuntimeError("Duplicate profile_row across extraction shards")
    return metadata


def main(rank: int, world: int, extraction_tag: str, extraction_world: int) -> None:
    if world != 16:
        raise RuntimeError("Canonical v1 fixes 16 contiguous treated shards")
    edges = np.linspace(0, N_CONDITIONS, world + 1, dtype=np.int64)
    start = int(edges[rank])
    stop = int(edges[rank + 1])
    n_local = stop - start
    out = ROOT / "data" / f"treated_{rank:02d}-of-{world:02d}.h5"
    receipt = ROOT / "audit" / f"treated_{rank:02d}-of-{world:02d}.json"
    partial = out.with_name(out.name + ".part")
    if any(path.exists() for path in [out, receipt, partial]):
        raise FileExistsError(f"Refuse overwrite of final treated shard {rank}")

    metadata = load_temp_metadata(extraction_tag, extraction_world)
    local = metadata.loc[
        (metadata["condition_row"] >= start) & (metadata["condition_row"] < stop)
    ].copy()
    groups = {int(key): frame for key, frame in local.groupby("condition_row", sort=True)}
    expected_rows = set(range(start, stop))
    if set(groups) != expected_rows:
        missing = sorted(expected_rows - set(groups))
        raise RuntimeError(f"Missing contributing profiles for condition rows: {missing[:20]}")

    handles = {}
    value_min = np.inf
    value_max = -np.inf
    with h5py.File(partial, "w") as out_h5:
        out_h5.attrs["format"] = "Tahoe-PDEX-full-QC-treated-shard-v1"
        out_h5.attrs["pdex_revision"] = "c7963cf334bec0683225d41c9586d900ca6303a2"
        out_h5.attrs["condition_start"] = start
        out_h5.attrs["condition_stop"] = stop
        out_h5.create_dataset("condition_row", data=np.arange(start, stop, dtype=np.int64))
        treated_ds = out_h5.create_dataset(
            "treated",
            shape=(n_local, N_GENES),
            dtype="f4",
            chunks=(1, N_GENES),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
            fletcher32=True,
        )
        try:
            for local_row, condition_row in enumerate(range(start, stop)):
                accumulator = np.zeros(N_GENES, dtype=np.float64)
                weight_sum = 0.0
                for record in groups[condition_row].to_dict("records"):
                    temp_rank = int(record["temp_rank"])
                    if temp_rank not in handles:
                        path = ROOT / "tmp" / (
                            f"pdex_values_{extraction_tag}_{temp_rank:02d}-of-{extraction_world:02d}.h5"
                        )
                        handles[temp_rank] = h5py.File(path, "r")
                    vector = np.asarray(
                        handles[temp_rank]["treated"][int(record["temp_value_row"])], dtype=np.float64
                    )
                    weight = float(record["aggregation_weight"])
                    accumulator += weight * vector
                    weight_sum += weight
                if not np.isclose(weight_sum, 1.0, atol=1e-7):
                    raise RuntimeError(f"Aggregation weights for condition {condition_row} sum to {weight_sum}")
                vector32 = accumulator.astype(np.float32)
                if not np.isfinite(vector32).all():
                    raise RuntimeError(f"Non-finite final treated vector for condition {condition_row}")
                treated_ds[local_row] = vector32
                value_min = min(value_min, float(vector32.min()))
                value_max = max(value_max, float(vector32.max()))
                if (local_row + 1) % 250 == 0 or local_row + 1 == n_local:
                    print(
                        f"treated_merge rank={rank}/{world} rows={local_row + 1}/{n_local}", flush=True
                    )
        finally:
            for handle in handles.values():
                handle.close()
        out_h5.flush()

    os.replace(partial, out)
    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "condition_start": start,
        "condition_stop": stop,
        "conditions": n_local,
        "genes": N_GENES,
        "value_range": [value_min, value_max],
        "file": str(out),
        "bytes": out.stat().st_size,
        "sha256": sha256sum(out),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--extraction-tag", default="full32")
    parser.add_argument("--extraction-world", type=int, default=32)
    args = parser.parse_args()
    main(args.rank, args.world, args.extraction_tag, args.extraction_world)

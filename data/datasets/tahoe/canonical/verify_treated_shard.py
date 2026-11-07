from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
N_CONDITIONS = 52886
N_GENES = 62710


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(rank: int, world: int) -> None:
    if world != 16:
        raise RuntimeError("Canonical treated layout requires 16 shards")
    path = ROOT / "data" / f"treated_{rank:02d}-of-{world:02d}.h5"
    receipt_path = ROOT / "audit" / f"treated_{rank:02d}-of-{world:02d}.json"
    out = ROOT / "audit" / f"verify_treated_{rank:02d}-of-{world:02d}.json"
    if out.exists():
        raise FileExistsError(f"Refuse overwrite: {out}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256sum(path) != receipt["sha256"]:
        raise RuntimeError(f"SHA256 mismatch for {path}")

    edges = np.linspace(0, N_CONDITIONS, world + 1, dtype=np.int64)
    expected_start = int(edges[rank])
    expected_stop = int(edges[rank + 1])
    value_min = np.inf
    value_max = -np.inf
    zero_count = 0
    value_sum = 0.0
    value_sumsq = 0.0
    with h5py.File(path, "r") as h5:
        ds = h5["treated"]
        rows = np.asarray(h5["condition_row"][:], dtype=np.int64)
        if ds.shape != (expected_stop - expected_start, N_GENES):
            raise RuntimeError(f"Shape mismatch for {path}: {ds.shape}")
        if not np.array_equal(rows, np.arange(expected_start, expected_stop, dtype=np.int64)):
            raise RuntimeError(f"Condition-row binding mismatch for {path}")
        for start in range(0, ds.shape[0], 32):
            block = np.asarray(ds[start : start + 32], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError(f"Non-finite treated values in {path} near row {start}")
            value_min = min(value_min, float(block.min()))
            value_max = max(value_max, float(block.max()))
            zero_count += int(np.count_nonzero(block == 0))
            block64 = block.astype(np.float64)
            value_sum += float(block64.sum())
            value_sumsq += float(np.square(block64).sum())

    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "condition_start": expected_start,
        "condition_stop": expected_stop,
        "conditions": expected_stop - expected_start,
        "genes": N_GENES,
        "value_range": [value_min, value_max],
        "zero_count": zero_count,
        "value_sum": value_sum,
        "value_sumsq": value_sumsq,
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": receipt["sha256"],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    main(args.rank, args.world)

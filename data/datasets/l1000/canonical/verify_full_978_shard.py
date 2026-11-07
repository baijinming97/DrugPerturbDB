from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from extract_full_978 import CONFIG, ROOT


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(dataset: str, rank: int, world: int, tag: str) -> None:
    cfg = CONFIG[dataset]
    stem = f"{cfg['stem']}_{tag}_{rank:02d}-of-{world:02d}"
    directory = ROOT / ("data" if tag == "full" else "tmp")
    audit_directory = ROOT / ("audit" if tag == "full" else "tmp")
    path = directory / f"{stem}.h5"
    receipt_path = audit_directory / f"{stem}.json"
    out = audit_directory / f"verify_{stem}.json"
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or sha256sum(path) != receipt["sha256"]:
        raise RuntimeError(f"Extraction receipt/hash failure: {path}")

    manifest = pq.read_table(
        cfg["manifest"], columns=[cfg["row_column"], "source_profile_index"]
    ).to_pandas()
    start = int(receipt["canonical_start"])
    stop = int(receipt["canonical_stop"])
    expected_rows = manifest[cfg["row_column"]].to_numpy(dtype=np.int64)[start:stop]
    expected_source = manifest["source_profile_index"].to_numpy(dtype=np.int64)[start:stop]
    axis = pq.read_table(
        ROOT / "data/gene_axis_978.parquet",
        columns=["axis_position", cfg["gene_column"]],
    ).to_pandas().sort_values("axis_position")
    gene_positions = axis[cfg["gene_column"]].to_numpy(dtype=np.int64)
    order = np.argsort(gene_positions)
    sorted_positions = gene_positions[order]
    restore = np.argsort(order)

    with h5py.File(path, "r") as canonical, h5py.File(cfg["source"], "r") as source:
        expression = canonical["expression"]
        rows = np.asarray(canonical["canonical_row"][:], dtype=np.int64)
        source_indices = np.asarray(canonical["source_profile_index"][:], dtype=np.int64)
        if expression.shape != (stop - start, 978):
            raise RuntimeError(f"Canonical shape mismatch: {path} {expression.shape}")
        if not np.array_equal(rows, expected_rows) or not np.array_equal(source_indices, expected_source):
            raise RuntimeError(f"Canonical/source row binding mismatch: {path}")
        sample_local_rows = sorted(set([0, len(rows) // 2, len(rows) - 1]))
        matrix = source["0/DATA/0/matrix"]
        sample_checks = []
        for local_row in sample_local_rows:
            source_row = int(source_indices[local_row])
            expected = np.asarray(matrix[source_row, sorted_positions], dtype=np.float32)[restore]
            observed = np.asarray(expression[local_row], dtype=np.float32)
            if not np.array_equal(observed, expected):
                raise RuntimeError(
                    f"Exact source-value mismatch: {dataset} shard {rank} local row {local_row}"
                )
            sample_checks.append(
                {
                    "canonical_row": int(rows[local_row]),
                    "source_profile_index": source_row,
                    "value_min": float(observed.min()),
                    "value_max": float(observed.max()),
                }
            )

    payload = {
        "status": "PASS",
        "dataset": dataset,
        "rank": rank,
        "world": world,
        "rows": stop - start,
        "genes": 978,
        "sha256": receipt["sha256"],
        "exact_source_value_checks": sample_checks,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CONFIG), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    main(args.dataset, args.rank, args.world, args.tag)

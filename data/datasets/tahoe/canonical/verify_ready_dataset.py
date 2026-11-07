from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
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


ready_path = ROOT / "READY.json"
ready = json.loads(ready_path.read_text(encoding="utf-8"))
if ready.get("status") != "READY":
    raise RuntimeError("READY marker is absent or invalid")
if ready.get("conditions") != N_CONDITIONS or ready.get("genes") != N_GENES:
    raise RuntimeError("READY dimensions are not the frozen canonical dimensions")

manifest_path = ROOT / ready["file_manifest"]
if sha256sum(manifest_path) != ready["file_manifest_sha256"]:
    raise RuntimeError("File-manifest SHA256 differs from READY")
manifest = pd.read_csv(manifest_path, sep="\t")
if len(manifest) != int(ready["data_files"]):
    raise RuntimeError("File-manifest row count differs from READY")

verified_bytes = 0
for record in manifest.to_dict("records"):
    path = ROOT / str(record["relative_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != int(record["bytes"]):
        raise RuntimeError(f"Byte-size mismatch: {path}")
    if sha256sum(path) != str(record["sha256"]):
        raise RuntimeError(f"SHA256 mismatch: {path}")
    verified_bytes += size
if verified_bytes != int(ready["data_bytes"]):
    raise RuntimeError("Verified data bytes differ from READY")

if list(ROOT.rglob("*.part")):
    raise RuntimeError("Partial files remain under the canonical root")

conditions = pq.read_table(ROOT / "data/condition_manifest.parquet").to_pandas()
if len(conditions) != N_CONDITIONS:
    raise RuntimeError("Condition manifest row count mismatch")
if not np.array_equal(conditions["condition_row"].to_numpy(), np.arange(N_CONDITIONS)):
    raise RuntimeError("Condition-row binding is not contiguous and exact")
if conditions["cell_line"].nunique() != 47 or conditions["drug"].nunique() != 379:
    raise RuntimeError("Condition manifest cell-line/drug cardinality mismatch")
if conditions[["drug", "dose"]].drop_duplicates().shape[0] != 1135:
    raise RuntimeError("Condition manifest drug-dose cardinality mismatch")

axis = pq.read_table(ROOT / "data/gene_axis.parquet").to_pandas()
if len(axis) != N_GENES or axis["feature"].nunique() != N_GENES:
    raise RuntimeError("Gene axis is not 62,710 unique features")
if not np.array_equal(axis["gene_position"].to_numpy(), np.arange(N_GENES)):
    raise RuntimeError("Gene-position binding is not contiguous and exact")

edges = np.linspace(0, N_CONDITIONS, 17, dtype=np.int64)
for rank in range(16):
    path = ROOT / f"data/treated_{rank:02d}-of-16.h5"
    with h5py.File(path, "r") as handle:
        expected_rows = int(edges[rank + 1] - edges[rank])
        if handle["treated"].shape != (expected_rows, N_GENES):
            raise RuntimeError(f"Treated HDF5 shape mismatch: {path}")
        rows = np.asarray(handle["condition_row"][:], dtype=np.int64)
        if not np.array_equal(rows, np.arange(edges[rank], edges[rank + 1])):
            raise RuntimeError(f"Treated HDF5 row binding mismatch: {path}")

with h5py.File(ROOT / "data/controls.h5", "r") as handle:
    if handle["controls"].shape != (658, N_GENES):
        raise RuntimeError("Deduplicated control HDF5 shape mismatch")

control_map = pq.read_table(ROOT / "data/condition_control_map.parquet").to_pandas()
sums = control_map.groupby("condition_row")["control_weight"].sum()
if len(sums) != N_CONDITIONS or not np.allclose(sums.to_numpy(), 1.0, atol=1e-7):
    raise RuntimeError("Condition-to-control map is incomplete or weights do not sum to one")

verification_path = ROOT / ready["final_verification"]
if sha256sum(verification_path) != ready["final_verification_sha256"]:
    raise RuntimeError("Final-verification SHA256 differs from READY")
verification = json.loads(verification_path.read_text(encoding="utf-8"))
if verification.get("status") != "PASS":
    raise RuntimeError("Final verification is not PASS")
for group, metrics in verification["official_cell_eval_2k_directional_crosscheck"].items():
    if metrics["flat_pearson"] < 0.5:
        raise RuntimeError(f"Weak cross-endpoint flat Pearson for {group}")
    if metrics["mean_condition_cosine"] < 0.5:
        raise RuntimeError(f"Weak cross-endpoint mean condition cosine for {group}")
    if metrics["positive_condition_cosine_fraction"] < 0.8:
        raise RuntimeError(f"Weak positive cross-endpoint fraction for {group}")

payload = {
    "status": "PASS",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "ready_sha256": sha256sum(ready_path),
    "files_verified": int(len(manifest)),
    "bytes_verified": int(verified_bytes),
    "conditions": N_CONDITIONS,
    "genes": N_GENES,
    "treated_shards": 16,
    "unique_controls": 658,
    "checks": [
        "READY dimensions and manifest binding",
        "all data file sizes and SHA256 hashes",
        "no partial files",
        "condition and gene row-key continuity",
        "all treated HDF5 shapes and condition-row bindings",
        "control HDF5 shape and condition-control weight sums",
        "final verification hash and cross-endpoint directional gates",
    ],
}
parser = argparse.ArgumentParser()
parser.add_argument("--output-name", default="independent_ready_verification.json")
args = parser.parse_args()
if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json"):
    raise RuntimeError("output-name must be a plain JSON filename")
out = ROOT / "audit" / args.output_name
if out.exists():
    raise FileExistsError(f"Refuse overwrite: {out}")
tmp = out.with_name(out.name + ".part")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, out)
print(json.dumps(payload, indent=2, sort_keys=True))

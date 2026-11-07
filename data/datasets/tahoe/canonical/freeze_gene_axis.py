from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
URL = (
    "https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister/resolve/"
    "c7963cf334bec0683225d41c9586d900ca6303a2/pdex/all_plates_pdex.parquet?download=true"
)
OUT = ROOT / "audit/gene_axis.parquet"
AUDIT = ROOT / "audit/gene_axis_audit.json"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if OUT.exists() or AUDIT.exists():
    raise FileExistsError("Refuse overwrite of frozen gene-axis outputs")

with fsspec.open(
    URL,
    "rb",
    block_size=8 * 1024 * 1024,
    cache_type="readahead",
    client_kwargs={"trust_env": True},
) as handle:
    pf = pq.ParquetFile(handle)
    table = pf.read_row_group(0, columns=["feature"], use_threads=False)
    features = [str(x) for x in pc.unique(table["feature"]).to_pylist()]

if len(features) != 62710 or len(set(features)) != 62710:
    raise RuntimeError(f"Official full feature axis is not 62,710 unique entries: {len(features)}")

axis = pa.table(
    {
        "gene_position": pa.array(range(len(features)), type=pa.int32()),
        "feature": pa.array(features, type=pa.string()),
    }
)
tmp = OUT.with_name(OUT.name + ".part")
pq.write_table(axis, tmp, compression="zstd")
os.replace(tmp, OUT)
payload = {
    "status": "PASS",
    "features": len(features),
    "source_row_group": 0,
    "pdex_revision": "c7963cf334bec0683225d41c9586d900ca6303a2",
    "gene_axis_sha256": sha256sum(OUT),
    "first_features": features[:10],
    "last_features": features[-10:],
}
AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))

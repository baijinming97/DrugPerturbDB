from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tahoe_canonical_io import TahoeCanonicalDataset


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
dataset = TahoeCanonicalDataset(ROOT)
rows = np.asarray([0, 1, 52885], dtype=np.int64)
positions = np.asarray([0, 100, 62709], dtype=np.int64)
batch = dataset.read(rows, positions)
for key in ["treated", "control", "delta"]:
    if batch[key].shape != (3, 3) or not np.isfinite(batch[key]).all():
        raise RuntimeError(f"Loader smoke failed for {key}")
if not np.allclose(batch["delta"], batch["treated"] - batch["control"], atol=0, rtol=0):
    raise RuntimeError("Loader delta is not exactly treated minus weighted control in float32")
payload = {
    "status": "PASS",
    "condition_rows": rows.tolist(),
    "gene_positions": positions.tolist(),
    "treated": batch["treated"].tolist(),
    "control": batch["control"].tolist(),
    "delta": batch["delta"].tolist(),
}
out = ROOT / "audit/loader_smoke.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))

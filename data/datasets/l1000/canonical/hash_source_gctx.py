from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000")
SOURCES = {
    "l3_treated": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_trt_cp_n1805898x12328.gctx"),
    "l3_control": Path("@WORKSPACE@/Noise/LINCS2020/raw/level3/level3_beta_ctl_n188708x12328.gctx"),
    "l5": Path("@WORKSPACE@/baicai/l3_l5_978_prep_20260822/raw/level5_beta_trt_cp_n720216x12328.gctx"),
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(dataset: str) -> None:
    source = SOURCES[dataset]
    out = ROOT / "audit" / f"source_{dataset}_sha256.json"
    if out.exists():
        raise FileExistsError(out)
    payload = {
        "status": "PASS",
        "dataset": dataset,
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256sum(source),
    }
    partial = out.with_name(out.name + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    args = parser.parse_args()
    main(args.dataset)


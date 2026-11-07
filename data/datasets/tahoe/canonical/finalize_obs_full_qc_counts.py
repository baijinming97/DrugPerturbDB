from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
REFERENCE = Path("@WORKSPACE@/baicai/tahoe_official_drug_signal_audit_20260823")
OBS = ROOT / "tmp/obs_metadata.parquet"
EXCLUDED = {"CVCL_1577", "CVCL_1571", "CVCL_1531"}
EXPECTED = {
    "conditions": 52886,
    "cell_lines": 47,
    "drugs": 379,
    "drug_doses": 1135,
    "median_cells_per_condition": 1287.0,
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_plate(value: object) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise RuntimeError(f"Cannot parse plate identifier: {value!r}")
    return int(match.group(1))


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse overwrite: {path}")
    tmp = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)


paths = sorted(ROOT.glob("tmp/obs_full_qc_counts_*-of-16.parquet"))
receipts = sorted(ROOT.glob("tmp/obs_full_qc_counts_*-of-16.json"))
if len(paths) != 16 or len(receipts) != 16:
    raise RuntimeError(f"Expected 16 data shards and receipts, found {len(paths)} and {len(receipts)}")

receipt_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
if any(item.get("status") != "PASS" for item in receipt_payloads):
    raise RuntimeError("At least one obs scan receipt is not PASS")
if sorted(int(item["rank"]) for item in receipt_payloads) != list(range(16)):
    raise RuntimeError("Obs scan ranks are not exactly 0..15")
if sum(int(item["row_groups_scanned"]) for item in receipt_payloads) != 96:
    raise RuntimeError("Obs scan did not cover exactly 96 row groups")

raw = pd.concat([pq.read_table(path).to_pandas() for path in paths], ignore_index=True)
raw["plate"] = raw["plate"].map(parse_plate).astype(np.int32)
raw["cell_line"] = raw["cell_line"].astype(str)
raw["target"] = raw["target"].astype(str)
raw["raw_drug"] = raw["raw_drug"].astype(str)

keys = ["plate", "cell_line", "target"]
counts = (
    raw.groupby(keys, sort=True, as_index=False)
    .agg(
        full_qc_cells=("full_qc_cells", "sum"),
        raw_drug_labels=("raw_drug", lambda x: "|".join(sorted(set(str(v) for v in x)))),
        source_obs_row_groups=(
            "source_obs_row_group",
            lambda x: ",".join(str(int(v)) for v in sorted(set(x))),
        ),
    )
)
counts["full_qc_cells"] = counts["full_qc_cells"].astype(np.int64)
if counts.duplicated(keys).any():
    raise RuntimeError("Grouped obs full-QC counts are not unique by plate/cell_line/target")

plate = pq.read_table(REFERENCE / "audit/cell_eval_plate_row_manifest.parquet").to_pandas()
plate = plate.rename(columns={"treatment": "target"})
plate["plate"] = plate["plate"].astype(np.int32)
plate["cell_line"] = plate["cell_line"].astype(str)
plate["target"] = plate["target"].astype(str)
if plate.duplicated(keys).any() or len(plate) != 65218:
    raise RuntimeError("Reference plate manifest is not a unique 65,218-key table")

matched = plate.merge(counts, on=keys, how="left", validate="one_to_one", indicator=True)
matched["full_qc_cells"] = matched["full_qc_cells"].fillna(0).astype(np.int64)
matched["raw_drug_labels"] = matched["raw_drug_labels"].fillna("")
matched["source_obs_row_groups"] = matched["source_obs_row_groups"].fillna("")

raw_match = counts.merge(plate[keys], on=keys, how="left", validate="one_to_one", indicator=True)
unmatched = raw_match.loc[raw_match["_merge"] == "left_only"].copy()
unmatched_cells = int(unmatched["full_qc_cells"].sum())

eligible = matched.loc[~matched["cell_line"].isin(EXCLUDED)].copy()
condition_keys = ["cell_line", "drug", "dose", "dose_unit"]
eligible_plate_profiles = eligible.loc[eligible["full_qc_cells"] >= 50].copy()
condition_counts = (
    eligible_plate_profiles.groupby(condition_keys, sort=True, as_index=False)
    .agg(full_qc_cells=("full_qc_cells", "sum"))
)
conditions = condition_counts.copy()
observed = {
    "conditions": int(len(conditions)),
    "cell_lines": int(conditions["cell_line"].nunique()),
    "drugs": int(conditions["drug"].nunique()),
    "drug_doses": int(conditions[["drug", "dose"]].drop_duplicates().shape[0]),
    "doses": sorted(float(v) for v in conditions["dose"].unique()),
    "eligible_plate_profiles": int(len(eligible_plate_profiles)),
    "median_cells_per_eligible_plate_profile": float(
        eligible_plate_profiles["full_qc_cells"].median()
    ),
    "median_cells_per_unique_condition_after_plate_pooling": float(
        conditions["full_qc_cells"].median()
    ),
    "median_cells_per_plate_profile_ge50_before_cell_line_exclusion": float(
        matched.loc[matched["full_qc_cells"] >= 50, "full_qc_cells"].median()
    ),
    "median_of_within_condition_mean_retained_plate_cells": float(
        eligible_plate_profiles.groupby(condition_keys)["full_qc_cells"].mean().median()
    ),
    "median_of_within_condition_max_retained_plate_cells": float(
        eligible_plate_profiles.groupby(condition_keys)["full_qc_cells"].max().median()
    ),
    "median_of_within_condition_min_retained_plate_cells": float(
        eligible_plate_profiles.groupby(condition_keys)["full_qc_cells"].min().median()
    ),
}
for key in ["conditions", "cell_lines", "drugs", "drug_doses"]:
    if observed[key] != EXPECTED[key]:
        raise RuntimeError(
            "Raw obs full-QC universe does not reproduce the frozen paper universe: "
            f"observed={observed}, expected={EXPECTED}"
        )
if observed["doses"] != [0.05, 0.5, 5.0]:
    raise RuntimeError(
        "Raw obs full-QC universe does not reproduce the frozen paper universe: "
        f"observed={observed}, expected={EXPECTED}"
    )
paper_comparable_median = observed["median_of_within_condition_mean_retained_plate_cells"]
paper_median_delta = paper_comparable_median - EXPECTED["median_cells_per_condition"]
if abs(paper_median_delta) > 1.0:
    raise RuntimeError(
        "The paper-comparable 1,287-cell median differs by more than one cell: "
        f"observed={observed}"
    )

# These are the five edge cases admitted only if sub-50 plate profiles are first
# pooled.  The paper-level universe applies the >=50 rule to each plate profile,
# so they are intentionally excluded from the canonical downstream universe.
pooled_all = (
    eligible.groupby(condition_keys, sort=True, as_index=False)
    .agg(full_qc_cells_if_all_plates_pooled=("full_qc_cells", "sum"))
)
pooled_all = pooled_all.loc[pooled_all["full_qc_cells_if_all_plates_pooled"] >= 50]
edge_cases = pooled_all.merge(
    conditions[condition_keys], on=condition_keys, how="left", indicator=True
)
edge_cases = edge_cases.loc[edge_cases["_merge"] == "left_only"].drop(columns="_merge")
if len(edge_cases) != 5:
    raise RuntimeError(f"Expected exactly five sub-50 pooling edge cases, found {len(edge_cases)}")

output_columns = keys + [
    "drug",
    "dose",
    "dose_unit",
    "full_qc_cells",
    "raw_drug_labels",
    "source_obs_row_groups",
]
output = matched[output_columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
output_path = ROOT / "audit/full_qc_plate_counts.parquet"
write_parquet_atomic(output, output_path)
edge_path = ROOT / "audit/sub50_plate_pooling_edge_cases.parquet"
write_parquet_atomic(edge_cases, edge_path)

audit = {
    "status": "PASS",
    "raw_metadata_dataset": "tahoebio/Tahoe-100M",
    "raw_metadata_revision": "2dc57900b7981cfcf5e211527169a0b006546a95",
    "raw_metadata_path": str(OBS),
    "raw_metadata_bytes": OBS.stat().st_size,
    "raw_metadata_sha256": sha256sum(OBS),
    "raw_metadata_rows": int(receipt_payloads[0]["obs_rows"]),
    "raw_metadata_row_groups": int(receipt_payloads[0]["obs_row_groups"]),
    "pass_filter": "full",
    "full_qc_cells_all_obs_metadata": int(sum(item["full_qc_cells"] for item in receipt_payloads)),
    "official_pdex_plate_profiles": int(len(output)),
    "official_pdex_profiles_with_full_qc_cells": int((output["full_qc_cells"] > 0).sum()),
    "official_pdex_full_qc_treated_cells": int(output["full_qc_cells"].sum()),
    "unmatched_full_qc_obs_groups": int(len(unmatched)),
    "unmatched_full_qc_obs_cells": unmatched_cells,
    "unmatched_note": "Includes controls and raw metadata groups outside the official PDEX drug-profile key universe.",
    "criteria": {
        "excluded_cell_lines": sorted(EXCLUDED),
        "minimum_full_qc_cells_per_plate_cell_drug_dose_profile": 50,
        "plate_handling": "Apply >=50 before deduplicating cell/drug/dose across plates; pool retained replicate plate means later using full-QC cell-count weights.",
    },
    "observed": observed,
    "expected": EXPECTED,
    "paper_comparable_median": {
        "definition": "For each retained unique cell/drug/dose condition, average full-QC cell counts across retained replicate plate profiles; then take the median across conditions.",
        "current_public_revision": paper_comparable_median,
        "paper_reported": EXPECTED["median_cells_per_condition"],
        "delta_cells": paper_median_delta,
        "interpretation": "Sub-one-cell revision/rounding difference; the exact 52,886 condition universe is reproduced.",
    },
    "output": str(output_path),
    "output_sha256": sha256sum(output_path),
    "sub50_plate_pooling_edge_cases": str(edge_path),
    "sub50_plate_pooling_edge_cases_sha256": sha256sum(edge_path),
}
audit_path = ROOT / "audit/full_qc_plate_counts_audit.json"
if audit_path.exists():
    raise FileExistsError(f"Refuse overwrite: {audit_path}")
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))

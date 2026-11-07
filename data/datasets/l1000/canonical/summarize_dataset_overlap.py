from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


T = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe/data")
L = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000/data")
conditions = pq.read_table(T / "condition_manifest.parquet").to_pandas()
drugs = pq.read_table(T / "drug_metadata.parquet").to_pandas()
tgenes = pq.read_table(T / "gene_axis.parquet").to_pandas()
lgenes = pq.read_table(L / "gene_axis_978.parquet").to_pandas()

tset = set(tgenes["feature"].astype(str))
symbols = lgenes["gene_symbol"].astype(str)
missing = lgenes.loc[~symbols.isin(tset), ["gene_id", "gene_symbol"]].to_dict("records")

payload = {
    "tahoe": {
        "conditions_by_dose": {
            str(key): int(value) for key, value in conditions["dose"].value_counts().sort_index().items()
        },
        "single_plate_conditions": int((conditions["source_plate_profiles"] == 1).sum()),
        "multi_plate_conditions": int((conditions["source_plate_profiles"] > 1).sum()),
        "target_cells_median": float(conditions["target_cells"].median()),
        "target_cells_min": int(conditions["target_cells"].min()),
        "target_cells_max": int(conditions["target_cells"].max()),
        "drugs_with_nonempty_canonical_smiles": int(drugs["canonical_smiles"].fillna("").astype(str).str.len().gt(0).sum()),
        "drugs_structure_parse_ok": int(drugs["structure_parse_ok"].fillna(False).astype(bool).sum()),
        "unique_parent_connectivity_keys": int(drugs["parent_connectivity_key"].fillna("").replace("", pd.NA).nunique()),
        "unique_murcko_scaffold_ids": int(drugs["murcko_scaffold_id"].fillna("").replace("", pd.NA).nunique()),
        "unique_chemistry_components": int(drugs["chemistry_component"].fillna("").replace("", pd.NA).nunique()),
    },
    "gene_overlap": {
        "l1000_landmarks": len(lgenes),
        "direct_symbol_overlap_with_tahoe_full_axis": int(symbols.isin(tset).sum()),
        "missing_l1000_symbols": missing,
    },
}
print(json.dumps(payload, indent=2))

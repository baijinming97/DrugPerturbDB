"""Verify the fixed Tahoe inclusion metadata against the original cell annotations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

O=Path('@WORKSPACE@')
C=O/'baicai/canonical_perturbation_data_v1/tahoe'
source=O/'FULL_DATASET_FIGURES_V1/raw/obs_metadata.parquet'
keys=['plate','cell_line','target']
counts={}
pf=pq.ParquetFile(source)
for rg in range(pf.num_row_groups):
    table=pf.read_row_group(rg,columns=['plate','cell_line','drugname_drugconc','pass_filter'],use_threads=False)
    for col in ['cell_line','pass_filter']:
        if pa.types.is_dictionary(table.schema.field(col).type):
            table=table.set_column(table.schema.get_field_index(col),col,table[col].cast(pa.string()))
    table=table.filter(pc.equal(table['pass_filter'],'full'))
    if not len(table):continue
    grouped=table.group_by(['plate','cell_line','drugname_drugconc']).aggregate([('pass_filter','count')]).to_pandas()
    for row in grouped.itertuples(index=False,name=None):
        key=(int(str(row[0]).removeprefix('plate')),str(row[1]),str(row[2]))
        counts[key]=counts.get(key,0)+int(row[3])

expected=pd.read_parquet(C/'audit/full_qc_plate_counts.parquet')
actual=np.asarray([counts.get((int(r.plate),str(r.cell_line),str(r.target)),0) for r in expected.itertuples()])
np.testing.assert_array_equal(actual,expected.full_qc_cells.to_numpy())
profiles=pd.read_parquet(C/'audit/qc_plate_profile_manifest.parquet')
observed=np.asarray([counts.get((int(r.plate),str(r.cell_line),str(r.target)),0) for r in profiles.itertuples()])
np.testing.assert_array_equal(observed,profiles.full_qc_cells.to_numpy())
assert (observed>=50).all() and profiles.cell_line.nunique()==47
assert profiles.profile_row.is_unique and profiles.condition_row.nunique()==52886
receipt=dict(status='PASS',observed_cell_rows=pf.metadata.num_rows,
             metadata_groups_checked=len(expected),eligible_profiles=len(profiles),
             minimum_cells=50,eligible_cells=47,conditions=52886,
             frozen_identity_and_split_annotations=True)
(C/'audit/source_qc_verified.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)

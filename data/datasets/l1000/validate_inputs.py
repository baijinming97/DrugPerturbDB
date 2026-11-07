"""Verify frozen gene/sample bindings against the two original L3 GCTX files."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

W=Path('@WORKSPACE@')
C=W/'baicai/canonical_perturbation_data_v1/l1000'
raw=W/'Noise/LINCS2020/raw/level3'
axis=pd.read_parquet(C/'data/gene_axis_978.parquet').sort_values('axis_position')
records=[]
for kind,filename,manifest in [
    ('control','level3_beta_ctl_n188708x12328.gctx','l3_control_manifest.parquet'),
    ('treated','level3_beta_trt_cp_n1805898x12328.gctx','l3_treatment_manifest.parquet'),
]:
    meta=pd.read_parquet(C/'data'/manifest)
    gp=axis[f'l3_{kind}_source_gene_position'].to_numpy(int)
    sp=meta.source_profile_index.to_numpy(int)
    with h5py.File(raw/filename,'r') as h:
        genes=h['0/META/ROW/id'][:].astype(str)
        samples=h['0/META/COL/id'][:].astype(str)
        assert h['0/DATA/0/matrix'].shape==(len(samples),len(genes))
        assert np.array_equal(genes[gp],axis.gene_id.astype(str).to_numpy()),'L3 gene identifiers do not match the registered axis'
        assert np.array_equal(samples[sp],meta.sample_id.astype(str).to_numpy()),'L3 sample identifiers do not match the registered profile indices'
        records.append(dict(source=kind,registered_profiles=len(meta),measured_genes=len(gp),source_profiles=len(samples),source_genes=len(genes),gene_IDs_equal=True,sample_IDs_equal=True))
out=C/'audit/source_axes_verified.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='PASS',sources=records),indent=2))
print(out.read_text())

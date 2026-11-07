"""Export the fixed shared gene axis from complete, source-specific consensus profiles."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd

F=Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
axis=json.loads((F/'COMMON_GENES.json').read_text())['axis']
assert len(axis)==len(set(axis))==869
records=[]
for name in ['L1000_L3','DRUGseq','CIGS','sciPlex3','MIXseq']:
    source=F/'data'/f'{name}_CONDITIONS.h5';target=F/'data'/f'{name}_COMMON_CONDITIONS.h5'
    existing=target.exists()
    meta=pd.read_parquet(F/'data'/f'{name}_CONDITION_META.parquet')
    with h5py.File(source,'r') as src,h5py.File(target,'r' if existing else 'w') as out:
        gene=src['entrez'].asstr()[:];positions={v:i for i,v in enumerate(gene) if v}
        assert all(g in positions for g in axis),name
        ix=np.array([positions[g] for g in axis]);n=src['delta'].shape[0]
        assert len(meta)==n and np.array_equal(meta.row.to_numpy(),np.arange(n)),name
        if existing:
            assert out['entrez'].asstr()[:].tolist()==axis and out['delta'].shape==(n,len(axis)),name
            ds=out['delta']
        else:
            out.create_dataset('entrez',data=np.asarray(axis,object),dtype=h5py.string_dtype())
            ds=out.create_dataset('delta',shape=(n,len(axis)),dtype='f4',chunks=(min(256,n),len(axis)),compression='lzf')
        for lo in range(0,n,256):
            values=src['delta'][lo:lo+256][:,ix]
            assert np.isfinite(values).all(),name
            if not existing:ds[lo:lo+256]=values
            assert np.array_equal(ds[lo:lo+256],values),name
    records.append(dict(dataset=name,rows=n,genes=len(axis),exact_subset=True))
    print('SHARED_AXIS',name,n,flush=True)
(F/'receipts/SHARED_AXIS_EXPORT.json').write_text(json.dumps(dict(status='PASS',datasets=records),indent=2))

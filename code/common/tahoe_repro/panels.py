"""Reconstruct a model panel from canonical author-normalized Tahoe data.

Only explicitly listed TRAIN/VALID condition rows are read. No TEST identifiers
are accepted by this entry point. The response is constructed after the frozen
control aggregation, on the frozen ordered 978-gene axis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse


def build_panel(canonical, identities, gene_axis, role, multiplate_manifest=None, multiplate_values=None):
    if role not in ['TRAIN','VALID']:
        raise ValueError('This model reproduction entry point permits TRAIN/VALID only')
    root=Path(canonical)
    ids=identities.copy() if isinstance(identities,pd.DataFrame) else pd.read_csv(identities,dtype=str)
    axis=pd.read_csv(gene_axis,sep='\t',dtype=str)
    pos=axis.tahoe_gene_position.to_numpy(int)
    if len(pos)!=978 or len(np.unique(pos))!=978:
        raise ValueError('Expected the exact 978-gene model axis')
    cm=pd.read_parquet(root/'condition_manifest.parquet')
    cm=cm[cm.dose.eq(5)].set_index(['drug','cell_line'])
    index=pd.MultiIndex.from_arrays([ids.drug_id,ids.cell_id],names=['drug','cell_line'])
    chosen=cm.loc[index].reset_index()
    if len(chosen)!=len(ids) or chosen.condition_row.duplicated().any():
        raise ValueError('The frozen panel does not map one-to-one to canonical conditions')
    mapping=pd.read_parquet(root/'condition_control_map.parquet')
    mapping=mapping[mapping.condition_row.isin(chosen.condition_row)]
    loc={int(v):i for i,v in enumerate(chosen.condition_row)}
    with h5py.File(root/'controls.h5','r') as h:
        order=np.argsort(pos);inv=np.argsort(order)
        controls=h['controls'][:,pos[order]][:,inv]
    W=sparse.csr_matrix((mapping.control_weight.to_numpy(),([loc[int(v)] for v in mapping.condition_row],mapping.control_row.to_numpy(int))),shape=(len(chosen),len(controls)))
    if not np.allclose(np.asarray(W.sum(1)).ravel(),1):
        raise ValueError('Control weights do not sum to one')
    x=np.asarray(W@controls,np.float32)
    treated=np.empty_like(x);seen=np.zeros(len(x),bool)
    for shard,part in chosen.groupby('treated_shard'):
        with h5py.File(root/f'treated_{int(shard):02d}-of-16.h5','r') as h:
            part=part.sort_values('treated_row_in_shard')
            for k in range(0,len(part),64):
                ss=part.iloc[k:k+64];rr=ss.treated_row_in_shard.to_numpy(int)
                # h5py only supports one fancy axis; read only allowed rows,
                # then select the ordered gene positions in memory.
                treated[ss.index]=h['treated'][rr][:,pos];seen[ss.index]=True
    if not seen.all():
        raise ValueError('Incomplete canonical shard coverage')
    # The predictive benchmark uses equal physical-plate weights. Canonical
    # descriptive profiles use cell-count weights. They agree for single-plate
    # conditions, but must never be interchanged for repeated conditions.
    repeated=chosen[chosen.source_plate_profiles.gt(1)]
    if len(repeated):
        if multiplate_manifest is None or multiplate_values is None:
            raise ValueError('Equal-plate benchmark reconstruction requires per-plate profiles')
        m=pd.read_parquet(multiplate_manifest)
        m=m[m.condition_row.isin(repeated.condition_row)]
        with h5py.File(multiplate_values,'r') as h:
            if not np.array_equal(h['tahoe_gene_position'][:],pos):
                raise ValueError('Per-plate gene axis differs from the frozen model axis')
            for idx,row in repeated.iterrows():
                v=m[m.condition_row.eq(row.condition_row)].sort_values('plate')
                if len(v)!=int(row.source_plate_profiles):
                    raise ValueError('Missing physical-plate profile')
                rr=v.value_row.to_numpy(int);order=np.argsort(rr)
                x[idx]=h['X1_mean'][rr[order]].astype(np.float64).mean(0).astype(np.float32)
                treated[idx]=h['X2'][rr[order]].astype(np.float64).mean(0).astype(np.float32)
    return dict(x=x,y=(treated-x).astype(np.float32),drug_id=ids.drug_id.to_numpy(str),
                cell_id=ids.cell_id.to_numpy(str),component=ids.component.to_numpy(str))

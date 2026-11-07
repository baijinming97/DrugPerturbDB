"""Build physical-plate and TRAIN/VALID model inputs from reconstructed PDEX."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
from tahoe_repro.panels import build_panel

O=Path('@WORKSPACE@')
C=O/'baicai/canonical_perturbation_data_v1/tahoe'
A=O/'4MAN/minimum_restore_v1/retained/orion'
P=A/'baicai/new_audit/tahoe_l3_x1_noise_audit_v1/results/tahoe/plate978'
G=A/'baicai/gamma_empirical_v1_20261008T190322Z/data'
axis=G/'GENE_AXIS.tsv'
pos=pd.read_csv(axis,sep='\t').tahoe_gene_position.to_numpy(int)
assert len(pos)==978 and len(set(pos))==978
meta=pd.concat([pd.read_parquet(C/'tmp'/f'pdex_values_paper4_{i:02d}-of-04.parquet').assign(extraction_rank=i) for i in range(4)],ignore_index=True)
meta=meta.sort_values('profile_row').reset_index(drop=True)
assert meta.profile_row.is_unique
meta['value_row']=np.arange(len(meta))
P.mkdir(parents=True,exist_ok=True)
manifest=P/'tahoe_multiplate_plate_profile_manifest_978_v1.parquet'
values=P/'tahoe_multiplate_plate_profiles_978_v1.h5'
if manifest.exists() or values.exists():raise FileExistsError('Predictive physical profiles already exist')
with h5py.File(values,'w') as out:
    out.create_dataset('tahoe_gene_position',data=pos)
    x1=out.create_dataset('X1_mean',shape=(len(meta),978),dtype='f4',chunks=(128,978),compression='lzf')
    x2=out.create_dataset('X2',shape=(len(meta),978),dtype='f4',chunks=(128,978),compression='lzf')
    for rank,part in meta.groupby('extraction_rank'):
        with h5py.File(C/'tmp'/f'pdex_values_paper4_{rank:02d}-of-04.h5') as h:
            controls=h['controls'][:][:,pos]
            part=part.sort_values('temp_value_row')
            for start in range(0,len(part),64):
                batch=part.iloc[start:start+64]
                xx=h['treated'][batch.temp_value_row.to_numpy(int)][:,pos]
                order=np.argsort(batch.value_row.to_numpy())
                rows=batch.value_row.to_numpy(int)[order]
                x2[rows]=xx[order]
                x1[rows]=controls[batch.temp_control_row.to_numpy(int)[order]]
meta.to_parquet(manifest,index=False)
audit=json.loads((O/'4MAN/PAPER_TWO_CHECKS_V1/exp2/G5_LINEAGE_AUDIT.json').read_text().replace('@'+'WORKSPACE@',O.as_posix()))
receipt=[]
with h5py.File(values) as physical:
    for rec in audit['folds']:
        fold=rec['fold'];cells=list(rec['cells'])
        groups=dict(zip(rec['fit_drug'],rec['fit_group']))
        groups.update(rec['VALID_group'])
        for role,drugs in [('TRAIN',rec['fit_drug']),('VALID',rec['VALID_drug'])]:
            ids=pd.DataFrame([dict(cell_id=c,drug_id=d,component=groups[d]) for c in cells for d in drugs])
            panel=build_panel(C/'data',ids,axis,role,manifest,values)
            dest=G/f'A_f{fold}_{role}.npz'
            if dest.exists():raise FileExistsError(dest)
            np.savez_compressed(dest,**panel)
            receipt.append(dict(fold=fold,role=role,dose=5,drugs=len(drugs),conditions=len(ids)))
        for entry in rec['lowdose']:
            drugs=list(entry['drugs']);dose=float(entry['dose'])
            assert set(drugs)<=set(rec['fit_drug'])
            selected=meta[meta.drug.isin(drugs)&meta.dose.eq(dose)&meta.cell_line.isin(cells)]
            lookup={key:part.value_row.to_numpy(int) for key,part in selected.groupby(['cell_line','drug'])}
            X=[];Y=[]
            for c in cells:
                xc=[];yc=[]
                for d in drugs:
                    rows=lookup[c,d]
                    control=physical['X1_mean'][rows].astype(np.float64)
                    treated=physical['X2'][rows].astype(np.float64)
                    xc.append(control.mean(0));yc.append((treated-control).mean(0))
                X.append(xc);Y.append(yc)
            dest=Path(entry['path']);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise FileExistsError(dest)
            np.savez_compressed(dest,X=np.asarray(X,np.float32),Y=np.asarray(Y,np.float64),drug=np.array(drugs),cell=np.array(cells),group=np.array([groups[d] for d in drugs]))
            receipt.append(dict(fold=fold,role='TRAIN',dose=dose,drugs=len(drugs),conditions=len(cells)*len(drugs)))
(C/'audit/predictive_inputs.json').write_text(json.dumps(dict(status='PASS',physical_profiles=len(meta),model_inputs=receipt,TEST_evaluated=False),indent=2))

from ff_common import *
start=time.time();name='Tahoe';D=C/'tahoe/data';owners=json.loads((A/'DRUG_ROLE_OWNER.json').read_text());legal={d for d,v in owners.items() if v['role']=='TRAIN'}
mt=pd.read_parquet(D/'condition_manifest.parquet');selected=mt[mt.drug.isin(legal)].copy().reset_index(drop=True);assert set(selected.drug)<=legal
mp=pd.read_parquet(D/'condition_control_map.parquet');mp=mp[mp.condition_row.isin(selected.condition_row)];g=pd.read_parquet(D/'gene_axis.parquet').feature.astype(str).to_numpy()
meta=selected.rename(columns={'cell_line':'cell'}).copy();meta['time']=24.;meta['plate']=meta.plates;meta['control_key']=meta.plates+'|'+meta.cell;meta['physical_unit']=meta.condition_row.astype(str);meta['normalization']='author_logCPM';meta['repeat_type']='canonical_plate_aggregate';meta['is_control']=False;meta['role']='TRAIN_owner';meta['group']=meta.drug.map(lambda d:owners[d]['chemical_group'])
meta.to_parquet(R/'data/Tahoe_ROLE_READ_LEDGER.parquet',index=False)
with newh5(name,meta,g,'symbol') as out,h5py.File(D/'controls.h5','r') as ctrl:
 controls=ctrl['controls'][:];loc={int(v):i for i,v in enumerate(selected.condition_row)};weights=sparse.csr_matrix((mp.control_weight.to_numpy(),([loc[int(v)] for v in mp.condition_row],mp.control_row.to_numpy(int))),shape=(len(selected),len(controls)))
 assert np.allclose(np.asarray(weights.sum(1)).ravel(),1,atol=1e-8)
 for p in sorted(D.glob('treated_*-of-16.h5')):
  with h5py.File(p,'r') as h:
   rows=h['condition_row'][:];sel=np.asarray([i for i,v in enumerate(rows) if int(v) in loc],int)
   for lo in range(0,len(sel),32):
    ii=sel[lo:lo+32];dest=np.asarray([loc[int(v)] for v in rows[ii]]);x=h['treated'][ii];assert np.isfinite(x).all();ref=weights[dest]@controls
    order=np.argsort(dest);out['post'][dest[order]]=x[order];out['delta'][dest[order]]=(x-ref)[order]
  print(p.name,'legal rows',len(sel),flush=True)
finish(name,start,dict(canonical_source_conditions=len(mt),read_conditions=len(meta),read_drugs=meta.drug.nunique(),role_rule='only DRUG_ROLE_OWNER TRAIN; TEST response rows never indexed',TEST_response_reads=0,cells=meta.cell.nunique(),doses=sorted(meta.dose.unique()),native_genes=len(g),repeatability_eligible=False,repeatability_source='separately verified single-plate native shards'))

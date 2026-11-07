from ff_common import *
start=time.time();D=C/'l1000/data';mode=sys.argv[1];name='L1000_'+mode
if (R/'receipts'/f'{name}_PREPARED.json').exists():raise SystemExit('ALREADY_PREPARED')
g=pd.read_parquet(D/'gene_axis_978.parquet').gene_id.astype(str).to_numpy()
def readshards(pattern,n):
 x=np.empty((n,978),np.float32);seen=np.zeros(n,bool)
 for p in sorted(D.glob(pattern)):
  with h5py.File(p,'r') as h:
   ix=h['canonical_row'][:];assert not seen[ix].any();x[ix]=h['expression'][:];seen[ix]=True
 assert seen.all() and np.isfinite(x).all();return x
if mode=='L3':
 mt=pd.read_parquet(D/'l3_treatment_manifest.parquet');cm=pd.read_parquet(D/'l3_control_manifest.parquet');cx=pd.read_parquet(D/'l3_control_context_manifest.parquet')
 cv=readshards('l3_control_full_*.h5',len(cm));refs={};controlsets={}
 cm=cm.set_index('control_row')
 for r in cx.itertuples():
  ids=np.asarray(r.control_rows,int);cc=cm.loc[ids];u=(cc.rna_plate.astype(str)+'|'+cc.rna_well.astype(str)).to_numpy();uu,ix=np.unique(u,return_inverse=True);mat=sparse.csr_matrix((1/np.bincount(ix)[ix],(ix,np.arange(len(ix)))),shape=(len(uu),len(ix)))
  refs[int(r.control_context_id)]=(mat@cv[ids]).mean(0);controlsets[int(r.control_context_id)]=';'.join(uu)
 del cv
 tv=readshards('l3_treated_full_*.h5',len(mt));keep=mt.has_exact_qc_vehicle_control.to_numpy(bool)&(mt.dose_uM.to_numpy(float)>0)
 excluded=mt.loc[~keep,['sample_id','pert_id','cell_iname','dose_uM','duration_h','has_exact_qc_vehicle_control']];excluded.to_parquet(R/'data/L3_EXCLUDED.parquet',index=False)
 mt=mt.loc[keep].reset_index(drop=True);tv=tv[keep];dv=np.empty_like(tv)
 for k,ix in mt.groupby('control_context_id',sort=False).indices.items():dv[ix]=tv[ix]-refs[int(k)]
 mt['physical_unit']=mt.rna_plate.astype(str)+'|'+mt.rna_well.astype(str)
 # Exactly matching RNA wells are technical readouts; never call them independent repeats.
 keys=['pert_id','cell_iname','dose_uM','duration_h','physical_unit'];codes=mt.groupby(keys,sort=False).ngroup().to_numpy();ng=int(codes.max())+1;n=np.bincount(codes);first=np.full(ng,len(mt),int);np.minimum.at(first,codes,np.arange(len(mt)))
 avg=sparse.csr_matrix((1/n[codes],(codes,np.arange(len(mt)))),shape=(ng,len(mt)))
 meta=mt.iloc[first][['pert_id','cmap_name','cell_iname','dose_uM','duration_h','rna_plate','physical_unit','control_context_id']].copy().reset_index(drop=True)
 meta=meta.rename(columns={'pert_id':'drug','cmap_name':'drug_name','cell_iname':'cell','dose_uM':'dose','duration_h':'time','rna_plate':'plate'})
 # Union of all physical control wells used by the technical readouts.
 bindings=mt[['control_context_id']].copy();bindings['code']=codes
 unions={int(k):';'.join(sorted(set(v for cid in ss.control_context_id.unique() for v in controlsets[int(cid)].split(';')))) for k,ss in bindings.groupby('code',sort=False)}
 meta['control_key']=[unions[i] for i in range(ng)];meta['n_readouts']=n;meta['normalization']='native_L3';meta['repeat_type']='RNA_well';meta['is_control']=False
 with newh5(name,meta,g,'entrez') as h:
  for lo in range(0,ng,16384):
   sl=slice(lo,min(ng,lo+16384));h['post'][sl]=avg[sl]@tv;h['delta'][sl]=avg[sl]@dv
 finish(name,start,dict(source_QC_profiles=int(len(keep)),excluded_no_control_or_nonpositive_dose=len(excluded),physical_profiles=ng,pert_ids=meta.drug.nunique(),cells=meta.cell.nunique(),all_QC_drugs_no_Tahoe_intersection=True,technical_readouts_collapsed=int(len(mt)-ng)))
elif mode=='L5':
 mt=pd.read_parquet(D/'l5_signature_manifest.parquet');x=readshards('l5_signatures_full_*.h5',len(mt));meta=mt[['pert_id','cmap_name','cell_iname','dose_uM','duration_h','sig_id','nsample']].rename(columns={'pert_id':'drug','cmap_name':'drug_name','cell_iname':'cell','dose_uM':'dose','duration_h':'time','sig_id':'physical_unit'});meta['plate']=meta.physical_unit;meta['control_key']='consensus_not_independent';meta['repeat_type']='MODZ_consensus';meta['normalization']='L5_standardized_MODZ';meta['is_control']=False
 with newh5(name,meta,g,'entrez',post=False) as h:h['delta'][:]=x
 finish(name,start,dict(profiles=len(meta),pert_ids=meta.drug.nunique(),cells=meta.cell.nunique(),repeatability_eligible=False,reason='MODZ already aggregates replicates; standardized response not absolute expression'))
else:raise ValueError(mode)

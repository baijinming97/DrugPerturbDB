from ff_common import *
start=time.time();name='sciPlex3';raw=R/'raw/sciplex3.h5ad'
with h5py.File(raw,'r') as h:
 md=pd.DataFrame({k:read_cat(h['obs'],v) for k,v in [('cell','cell_line'),('drug','perturbation'),('dose','dose_value'),('time','time'),('plate','replicate')]});md['dose']=md.dose.astype(float)/1000;md['time']=md.time.astype(float);md['is_control']=md.drug.astype(str).str.lower().isin(['control','dmso']);md.loc[md.is_control,'dose']=0
 cols=['cell','drug','dose','time','plate','is_control'];valid=md[cols].notna().all(axis=1)&np.isfinite(md.dose)&np.isfinite(md.time);code=np.full(len(md),-1,int);code[valid]=md.loc[valid].groupby(cols,sort=True).ngroup().to_numpy(int);n=int(code.max())+1;first=np.full(n,len(md),int);np.minimum.at(first,code[valid],np.flatnonzero(valid));meta=md.iloc[first].reset_index(drop=True).copy();nc=np.bincount(code[valid]);dump(R/'receipts/sciPlex3_METADATA_QC.json',dict(source_cells=len(md),missing_metadata_cells=int((~valid).sum()),missing_by_column=md[cols].isna().sum().to_dict()))
 original=np.asarray(read_cat(h['var'],'ensembl_id'),str);genes,inv=np.unique(np.asarray([g.split('.')[0] for g in original]),return_inverse=True);human=np.char.startswith(genes,'ENSG');hgenes=genes[human];hm=np.full(len(genes),-1,int);hm[human]=np.arange(human.sum());gmap=hm[inv];validgene=gmap>=0
 counts=np.zeros((n,int(human.sum())),np.int64);total=np.zeros(n,np.int64);X=h['X'];ptr=X['indptr'][:];geneproj=sparse.csr_matrix((np.ones(validgene.sum(),np.int8),(np.flatnonzero(validgene),gmap[validgene])),shape=(len(original),len(hgenes)))
 for lo in range(0,len(md),4096):
  hi=min(len(md),lo+4096);a,b=int(ptr[lo]),int(ptr[hi]);v=X['data'][a:b];assert (v>=0).all() and (v==np.rint(v)).all();mat=sparse.csr_matrix((v,X['indices'][a:b],ptr[lo:hi+1]-a),shape=(hi-lo,len(original)))
  grp=code[lo:hi];active=grp>=0;agg=sparse.csr_matrix((np.ones(active.sum()),(grp[active],np.flatnonzero(active))),shape=(n,len(grp)));part=agg@mat;total+=np.asarray(part.sum(1)).ravel().astype(np.int64);counts+=(part@geneproj).toarray().astype(np.int64)
  if lo%40960==0:print('cells',hi,len(md),flush=True)
meta['n_cells']=nc;meta['library_total']=total;meta['control_key']=meta.cell.astype(str)+'|'+meta.plate.astype(str)+'|'+meta.time.astype(str);meta['physical_unit']=meta.index.astype(str);meta['repeat_type']='biological_replicate_pseudobulk';meta['panel']='sciPlex3';meta['valid_profile']=(nc>=20)&(total>0)
with h5py.File(R/'data/sciPlex3_COUNTS.h5','w') as h:
 h.create_dataset('counts',data=counts,compression='gzip',compression_opts=1);h.create_dataset('total_full_library',data=total);h.create_dataset('gene',data=hgenes.astype(object),dtype=h5py.string_dtype())
outmeta=pd.concat([meta.assign(normalization='logCPM1'),meta.assign(normalization='logCPM100')],ignore_index=True)
with newh5(name,outmeta,hgenes,'ensembl') as h:
 for a,pc in enumerate([1,100]):
  expr=np.log2(counts/np.maximum(total,1)[:,None]*1e6+pc);delta=np.empty_like(expr)
  for key,ix in meta.groupby('control_key').indices.items():
   ci=ix[meta.iloc[ix].is_control.to_numpy()&meta.iloc[ix].valid_profile.to_numpy()];assert len(ci)==1,(key,len(ci));delta[ix]=expr[ix]-expr[ci[0]]
  h['post'][a*n:(a+1)*n]=expr;h['delta'][a*n:(a+1)*n]=delta
finish(name,start,dict(raw_sha256=sha(raw),single_cells=len(md),original_features=len(original),unique_Ensembl=len(genes),native_human_genes=len(hgenes),duplicate_Ensembl_columns=len(original)-len(genes),pseudobulks=n,QC_pseudobulks=int(meta.valid_profile.sum()),drug_names=meta.loc[~meta.is_control,'drug'].nunique(),minimum_cells=20,denominator='all original reference features; same across native and common axes'))

from ff_statistics import *
from bulk_repeats import bulk_repeats
import gc
start=time.time();quota();name=sys.argv[1];meta=pd.read_parquet(R/'data'/f'{name}_META.parquet');h=h5py.File(R/'data'/f'{name}.h5','r');g=h['gene'].asstr()[:];ent=h['entrez'].asstr()[:]
if 'is_control' in meta:eligible=~meta.is_control.astype(bool)
else:eligible=np.ones(len(meta),bool)
if 'valid_profile' in meta:eligible=eligible&meta.valid_profile.astype(bool)
meta=meta[eligible].copy();source_rows=meta.row.to_numpy(int);meta=meta.reset_index(drop=True)
missing={k:int(pd.to_numeric(meta[k],errors='coerce').isna().sum()) for k in ['dose','time']};dump(R/'receipts'/f'{name}_MISSING_CONDITION_METADATA.json',missing)
for k in ['dose','time']:meta[k]=pd.to_numeric(meta[k],errors='coerce').fillna(-1).map(lambda v:float(format(float(v),'.6g')))
meta['normalization']=meta.get('normalization','native')
if 'panel' not in meta:meta['panel']='all'
conditions=['panel','normalization','drug','cell','dose','time'];codes=meta.groupby(conditions,sort=True).ngroup().to_numpy();ng=int(codes.max())+1;first=np.full(ng,len(meta),int);np.minimum.at(first,codes,np.arange(len(meta)));n=np.bincount(codes)
agg=sparse.csr_matrix((1/n[codes],(codes,np.arange(len(meta)))),shape=(ng,len(meta)));cm=meta.iloc[first].reset_index(drop=True).copy();cm['row']=np.arange(ng);cm['n_repeats']=n;cm.to_parquet(R/'data'/f'{name}_CONDITION_META.parquet',index=False)
maskfile=R/'COMMON_GENES.json';common=json.loads(maskfile.read_text())['axis'] if maskfile.exists() else COMMON
emap={e:i for i,e in enumerate(ent) if e};common=[e for e in common if e in emap];cp=np.array([emap[e] for e in common]);assert len(cp)>=500,(name,len(cp))
replicate_allowed=not meta.repeat_type.isin(['MODZ_consensus','canonical_plate_aggregate','author_QC_replacements_not_independent']).all()
condgroups=[(int(k),np.asarray(ix)) for k,ix in meta.groupby(codes).indices.items() if len(ix)>1 and meta.iloc[ix].plate.nunique()>1] if replicate_allowed and name!='L1000_L3' else []
stats=[];rep=[];checks=[]
# Aggregate native-gene consensus only once; retain 978-axis masked response for cross-dataset comparisons.
ah=h5py.File(R/'data'/f'{name}_CONDITIONS.h5','w');ah.create_dataset('gene',data=np.asarray(g,object),dtype=h5py.string_dtype());ah.create_dataset('entrez',data=np.asarray(ent,object),dtype=h5py.string_dtype())
for mode in ['post','delta']:
 if mode not in h:continue
 ah.create_dataset(mode,shape=(ng,len(g)),dtype='f4',chunks=(min(64,ng),min(4096,len(g))),compression='lzf')
 # Accumulate by physical rows to avoid repeated decompression/partial chunk rewrites.
 xall=np.zeros((ng,len(g)),np.float64)
 for lo in range(0,len(meta),256):
  hi=min(len(meta),lo+256);rr=source_rows[lo:hi];xx=h[mode][int(rr[0]):int(rr[-1])+1][rr-rr[0]].astype(np.float64) if rr[-1]-rr[0]<2*len(rr) else h[mode][rr].astype(np.float64)
  if len(np.unique(codes[lo:hi]))==len(rr):xall[codes[lo:hi]]+=xx
  else:np.add.at(xall,codes[lo:hi],xx)
 xall/=n[:,None]
 for lo in range(0,ng,128):ah[mode][lo:lo+128]=xall[lo:lo+128]
 del xall;gc.collect()
 print(name,mode,'consensus ready',ng,flush=True)
 # Repeatability exact mean of all cross-plate pairwise PCCs, no pair subsampling.
 if name=='L1000_L3':rep.extend(bulk_repeats(h[mode],meta,cm,codes,source_rows,cp,mode,name))
 elif condgroups:
  used=np.sort(np.unique(np.concatenate([ix for k,ix in condgroups])));lu=np.full(len(meta),-1,int);lu[used]=np.arange(len(used));xs=np.empty((len(used),len(g)),np.float32)
  for lo in range(0,len(used),256):
   rr=source_rows[used[lo:lo+256]];xs[lo:lo+256]=h[mode][int(rr[0]):int(rr[-1])+1][rr-rr[0]] if rr[-1]-rr[0]<2*len(rr) else h[mode][rr]
  for axis,positions in [('native',np.arange(len(g))),('common',cp)]:
   for code,ii in condgroups:
    xx=xs[lu[ii]][:,positions];v,npairs=all_cross_plate_pcc(xx,meta.iloc[ii].plate.to_numpy())
    if npairs:
     rr=cm.iloc[code];rep.append(dict(dataset=name,mode=mode,axis=axis,genes=len(positions),drug=rr.drug,cell=rr.cell,dose=rr.dose,time=rr.time,panel=rr.panel,normalization=rr.normalization,pcc=v,pairs=npairs,physical_units=len(ii),plates=meta.iloc[ii].plate.nunique()))
  del xs;gc.collect()
 # Complete globally drug-equal decomposition is performed in compose_full.py.
 gc.collect()
ah.close();h.close()
pd.DataFrame(rep).to_csv(R/'results'/f'{name}_REPEAT_CONDITIONS.csv.gz',index=False) if name!='Tahoe' else None
summary=[]
if rep:
 t=pd.DataFrame(rep)
 for keys,ss in t.groupby(['mode','axis','normalization']):
  d=ss.groupby('drug').pcc.mean();mean,lo,hi=boot_mean(d.to_numpy());summary.append(dict(dataset=name,mode=keys[0],axis=keys[1],normalization=keys[2],pcc=mean,low=lo,high=hi,drugs=len(d),conditions=len(ss),pairs=int(ss.pairs.sum()),genes=int(ss.genes.iloc[0]),bootstrap='drug; conditions and pair dependence retained within drug',all_cross_plate_pairs_exact=True))
pd.DataFrame(summary).to_csv(R/'results'/f'{name}_REPEAT_SUMMARY.csv',index=False) if name!='Tahoe' else None
dump(R/'receipts'/f'{name}_ANALYZED.json',dict(status='PASS',seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID'),profiles=len(meta),conditions=ng,source_drugs=meta.drug.nunique(),cells=meta.cell.nunique(),native_genes=len(g),common_genes=len(cp),repeatability_eligible=replicate_allowed,composition_method='weighted sparse additive LS; order-symmetric Shapley main-effect energy; balanced support reduces to ordinary orthogonal decomposition',no_missing_zero_fill=True))

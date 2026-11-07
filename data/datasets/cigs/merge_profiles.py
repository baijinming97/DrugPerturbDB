from ff_common import *
if not (R/'receipts/CIGS_PREPARED.json').exists():
 start=time.time();name='CIGS';names=['CIGS_'+str(i) for i in range(6)];meta=[];g=None;total=0;parts=[]
 for nm in names:
  assert (R/'receipts'/f'{nm}_PREPARED.json').exists(),nm
  t=pd.read_parquet(R/'data'/f'{nm}_META.parquet');meta.append(t)
  with h5py.File(R/'data'/f'{nm}.h5','r') as h:
   gg=h['gene'].asstr()[:];gg=np.asarray([({'45542':'CIGS_DATE_SEP7_UNRESOLVED','X7.Sep':'CIGS_DATE_SEP7_UNRESOLVED'}.get(v,v.replace('.','-') if v.startswith(('HLA.','NKX')) else v)) for v in gg]);assert len(set(gg))==len(gg)
   if g is None:g=gg
   assert set(gg)==set(g);parts.append((nm,len(t),pd.Index(gg).get_indexer(g)));total+=len(t)
 md=pd.concat(meta,ignore_index=True)
 with newh5(name,md,g,'symbol') as h:
  off=0
  for nm,n,order in parts:
   with h5py.File(R/'data'/f'{nm}.h5','r') as src:
    for lo in range(0,n,512):
     hi=min(n,lo+512)
     for mode in ['post','delta']:h[mode][off+lo:off+hi]=src[mode][lo:hi][:,order]
   off+=n
 finish(name,start,dict(physical_sources=names,profiles=len(md),drugs_by_library=md[~md.is_control].groupby('panel').drug.nunique().to_dict(),genes=len(g)))

# The registered common gene axis is frozen metadata; it is not reselected here.

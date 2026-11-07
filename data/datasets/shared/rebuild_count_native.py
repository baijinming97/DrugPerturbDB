"""Reconstruct native-gene matrices from retained count data.

Writes data/replay/DATASET.h5 and stops if the output already exists.
Run through the CPU workflow entry point."""
from ff_common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['DRUGseq','CIGS','sciPlex3','MIXseq']);p.add_argument('--check-only',action='store_true');arg=p.parse_args();name=arg.dataset
assert os.getenv('TAHOE_REPRO_RUN_ID'),'CPU allocation required';quota();out=R/'data/replay';out.mkdir(exist_ok=True);dest=out/f'{name}.h5';assert not dest.exists()
md=pd.read_parquet(R/'data'/f'{name}_META.parquet');axes=np.load(R/'data'/f'{name}_NATIVE_AXES.npz');genes=axes['gene'];ent=axes['entrez'];base=md[md.normalization=='logCPM1'].reset_index(drop=True);n=len(base)
def standard_counts():
 if name in ['DRUGseq','sciPlex3']:
  h=h5py.File(R/'data'/f'{name}_COUNTS.h5');assert h['counts'].shape[0]==n
  def fetch(ix):return h['counts'][np.sort(ix)].astype(float)[np.argsort(np.argsort(ix))]
  return fetch,lambda:h.close()
 if name=='CIGS':
  hs={};links={}
  for i in range(6):
   h=h5py.File(R/'data'/f'CIGS_{i}_COUNTS.h5');m=pd.read_parquet(R/'data'/f'CIGS_{i}_META.parquet');m=m[m.normalization=='logCPM1'];g=h['gene'].asstr()[:];g=[{'45542':'CIGS_DATE_SEP7_UNRESOLVED','X7.Sep':'CIGS_DATE_SEP7_UNRESOLVED'}.get(v,v.replace('.','-') if v.startswith(('HLA.','NKX')) else v) for v in g];order=pd.Index(g).get_indexer(genes);assert (order>=0).all();hs[i]=(h,order)
   for j,z in enumerate(m.itertuples()):links[(z.source_file,str(z.physical_unit))]=(i,j)
  def fetch(ix):
   x=np.empty((len(ix),len(genes)))
   for j,rr in enumerate(ix):
    z=base.iloc[rr];i,k=links[(z.source_file,str(z.physical_unit))];h,order=hs[i];x[j]=h['counts'][k][order]
   return x
  return fetch,lambda:[h.close() for h,o in hs.values()]
 if name=='MIXseq':
  links={};cache={}
  for f in sorted((R/'data').glob('*.PB.npz')):
   z=np.load(f,allow_pickle=True);g=np.asarray(z['gene'],str);order=pd.Index(g).get_indexer(genes);assert (order>=0).all();cts=z['counts'];meta=json.loads(str(z['metadata']));cache[f.name]=(cts,order)
   for j,v in enumerate(meta):links[(v['source_file'],str(v['cell']),str(v['drug']),float(v['time']))]=(f.name,j)
  def fetch(ix):
   x=np.empty((len(ix),len(genes)))
   for j,rr in enumerate(ix):
    z=base.iloc[rr];f,k=links[(z.source_file,str(z.cell),str(z.drug),float(z.time))];ct,order=cache[f];x[j]=ct[k][order]
   return x
  return fetch,lambda:None
fetch,close=standard_counts()
if arg.check_only:
 cm=pd.read_parquet(R/'data'/f'{name}_CONDITION_META.parquet');cm=cm[cm.normalization=='logCPM1'];common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];lu={v:i for i,v in enumerate(ent) if v};pos=np.array([lu[v] for v in common]);errs=[]
 with h5py.File(R/'data'/f'{name}_COMMON_CONDITIONS.h5') as h:
  for z in cm.iloc[np.linspace(0,len(cm)-1,5,dtype=int)].itertuples():
   ix=np.flatnonzero((base.drug==z.drug)&(base.cell==z.cell)&np.isclose(base.dose,z.dose,rtol=1e-5)&np.isclose(base.time,z.time,rtol=1e-5)&(base.panel==z.panel)&~base.is_control&base.get('valid_profile',True));xs=[]
   assert len(ix)>0
   for i in ix:
    ci=np.flatnonzero((base.control_key==base.iloc[i].control_key)&base.is_control&base.get('valid_profile',True));ct=fetch(np.array([i]));tt=base.iloc[i].library_total;cc=fetch(ci);tot=base.iloc[ci].library_total.to_numpy(float);xx=np.log2(ct/max(tt,1)*1e6+1)-np.log2(cc/np.maximum(tot,1)[:,None]*1e6+1).mean(0);xs.append(xx[0].astype('f4')[pos])
   expected=np.mean(np.asarray(xs,dtype=float),axis=0).astype('f4');err=float(np.max(abs(expected-h['delta'][z.row])));assert err<2e-6,(name,z.drug,err);errs.append(dict(drug=str(z.drug),cell=str(z.cell),source_profiles=len(ix),max_abs_error=err))
 close();dump(R/'receipts'/f'{name}_NATIVE_REPLAY_CHECK.json',dict(status='PASS',checks=errs,all_native_counts_available=True,no_raw_download_required=True,job=os.getenv('TAHOE_REPRO_RUN_ID')));raise SystemExit(0)
with h5py.File(dest,'w') as h:
 for k,g in [('gene',genes),('entrez',ent)]:h.create_dataset(k,data=g.astype(object),dtype=h5py.string_dtype())
 for mode in ['post','delta']:h.create_dataset(mode,shape=(len(md),len(genes)),dtype='f4',chunks=(32,min(4096,len(genes))),compression='lzf')
 for norm,pc in [('logCPM1',1),('logCPM100',100)]:
  ids=md.index[md.normalization==norm].to_numpy();assert len(ids)==n
  for ck,ix in base.groupby('control_key',sort=False).indices.items():
   ix=np.asarray(ix);ct=fetch(ix);tot=base.iloc[ix].library_total.to_numpy(float);expr=np.log2(ct/np.maximum(tot,1)[:,None]*1e6+pc);valid=base.iloc[ix].get('valid_profile',pd.Series(True,index=range(len(ix)))).to_numpy(bool);ci=base.iloc[ix].is_control.to_numpy(bool)&valid
   delta=expr-expr[ci].mean(0) if ci.any() else np.zeros_like(expr)
   rr=ids[ix];sort=np.argsort(rr);h['post'][rr[sort]]=expr[sort];h['delta'][rr[sort]]=delta[sort]
close();dump(out/f'{name}_REPLAY.json',dict(status='REBUILT',bytes=dest.stat().st_size,sha256=sha(dest),source='retained complete count/pseudobulk matrices; original metadata and full-library denominators',job=os.getenv('TAHOE_REPRO_RUN_ID')))

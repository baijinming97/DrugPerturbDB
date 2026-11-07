from ff_statistics import *
from scipy.stats import pearsonr
start=time.time();checks=[];rng=np.random.default_rng(20261029)
for nm in ['DRUGseq']+['CIGS_'+str(i) for i in range(6)]+['sciPlex3']:
 assert (R/'receipts'/f'{nm}_PREPARED.json').exists()
 meta=pd.read_parquet(R/'data'/f'{nm}_META.parquet');h=h5py.File(R/'data'/f'{nm}.h5');p=R/'data'/f'{nm}_COUNTS.h5'
 if not p.exists():
  checks.append(dict(dataset=nm,status='STRUCTURE_ONLY',reason='count container differs; separate parser receipt'))
  h.close();continue
 with h5py.File(p) as hc:
  n=hc['counts'].shape[0];genes=hc['gene'].asstr()[:];assert np.array_equal(genes,h['gene'].asstr()[:]);total=hc['total_full_library' if nm=='sciPlex3' else 'total'][:]
  errs=[]
  for i in rng.choice(n,min(n,24),replace=False):
   ct=hc['counts'][i].astype(float);assert (ct>=0).all();assert abs(ct.sum()-total[i])<1 if nm!='sciPlex3' else total[i]>=ct.sum()
   for a,pc in enumerate([1,100]):
    expected=np.log2(ct/max(total[i],1)*1e6+pc);actual=h['post'][i+a*n].astype(float);errs.append(float(np.max(abs(expected-actual))))
   ci=np.flatnonzero((meta.iloc[:n].control_key==meta.iloc[i].control_key).to_numpy()&meta.iloc[:n].is_control.to_numpy()&(total>0))
   assert len(ci)>0
   for a,pc in enumerate([1,100]):
    ctr=np.log2(hc['counts'][ci].astype(float)/total[ci,None]*1e6+pc).mean(0);exp=np.log2(ct/max(total[i],1)*1e6+pc)-ctr;errs.append(float(np.max(abs(exp-h['delta'][i+a*n]))))
  assert max(errs)<2e-6,(nm,max(errs));checks.append(dict(dataset=nm,status='PASS',max_independent_post_error=max(errs),samples=min(n,24),all_native_genes=len(genes)))
 h.close()
# Verify CIGS concatenation against original panels, including explicit source-specific gene labels.
md=pd.read_parquet(R/'data/CIGS_META.parquet');h=h5py.File(R/'data/CIGS.h5');off=0;errs=[]
for i in range(6):
 with h5py.File(R/'data'/f'CIGS_{i}.h5') as src:
  n=src['post'].shape[0];gg=src['gene'].asstr()[:];gg=[{'45542':'CIGS_DATE_SEP7_UNRESOLVED','X7.Sep':'CIGS_DATE_SEP7_UNRESOLVED'}.get(v,v.replace('.','-') if v.startswith(('HLA.','NKX')) else v) for v in gg];order=pd.Index(gg).get_indexer(h['gene'].asstr()[:]);assert (order>=0).all()
  for j in [0,n//2,n-1]:
   # Source gene order is conserved after alias normalization; merger additionally checks set equality.
   for mode in ['post','delta']:errs.append(float(np.max(abs(src[mode][j][order]-h[mode][off+j]))))
  off+=n
assert max(errs)==0;h.close();checks.append(dict(dataset='CIGS_merge',status='PASS',max_abs=max(errs),rows=off))

with h5py.File(R/'data/MIXseq.h5') as h:
 md=pd.read_parquet(R/'data/MIXseq_META.parquet');n=len(md)//2;rowmap={(z.source_file,z.cell,z.drug,float(z.time)):i for i,z in enumerate(md.iloc[:n].itertuples())};errors=[]
 for p in sorted((R/'data').glob('*.zip.PB.npz')):
  z=np.load(p,allow_pickle=True);mm=json.loads(str(z['metadata']));order=pd.Index(z['gene']).get_indexer(h['gene'].asstr()[:]);assert (order>=0).all()
  if not mm:continue
  j=len(mm)//2;i=rowmap[(mm[j]['source_file'],mm[j]['cell'],mm[j]['drug'],float(mm[j]['time']))];ct=z['counts'][j][order].astype(float);expected=np.log2(ct/max(mm[j]['library_total'],1)*1e6+1);errors.append(float(np.max(abs(expected-h['post'][i]))))
 assert max(errors)<2e-6;checks.append(dict(dataset='MIXseq',status='PASS',samples=len(errors),max_independent_post_error=max(errors)))

# Independent numerical verification of decomposition by explicit weighted linear least squares.
test=[]
for sparse_case in [False,True]:
 pairs=[(d,c) for d in range(7) for c in range(5) if not sparse_case or (d+c)%4!=0];m=pd.DataFrame(pairs,columns=['drug','cell']);x=rng.normal(size=(len(m),11));E,com,info=decompose(x,m)
 d,c,w,D,Cc,_=weighted_design(m);X=np.c_[np.ones(len(m)),np.eye(7)[d],np.eye(5)[c]];fit=X@np.linalg.lstsq(X*np.sqrt(w[:,None]),x*np.sqrt(w[:,None]),rcond=None)[0]
 err=float(np.max(abs((x-fit)-com['residual'])));closure=abs(float(E.sum()-np.sum(w[:,None]*x*x)));assert err<1e-8 and closure<1e-8
 test.append(dict(sparse=sparse_case,explicit_lstsq_max_error=err,energy_closure=closure))
dump(R/'receipts/SOURCE_ACCEPTANCE.json',dict(status='PASS',checks=checks,decomposition_independent=test,seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID')));print('SOURCE_ACCEPTANCE PASS',flush=True)

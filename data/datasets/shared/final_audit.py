from ff_statistics_global import *
from scipy.stats import pearsonr
import itertools
start=time.time();names=['Tahoe','L1000_L3','L1000_L5','DRUGseq','CIGS','sciPlex3','MIXseq','NCI_TPW'];common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];checks=[];inventory=[];summaries=[];compositions=[];sources=[]
for name in names:
 assert (R/'receipts'/f'{name}_ANALYZED.json').exists(),name
 assert (R/'receipts'/f'{name}_COMPOSITION_FINAL.json').exists(),name
 md=pd.read_parquet(R/'data'/f'{name}_META.parquet');cm=pd.read_parquet(R/'data'/f'{name}_CONDITION_META.parquet');firstnorm=next(x for x in cm.normalization.unique() if x!='logCPM100');m=md[md.normalization==firstnorm].copy();eligible=~m.is_control.astype(bool)
 if 'valid_profile' in m:eligible&=m.valid_profile.astype(bool)
 e=m[eligible];f=pd.read_csv(R/'results'/f'{name}_FULL_COMPOSITION.csv');compositions.append(f);assert np.allclose(f[[x+'_fraction' for x in ['mean','cell','drug','interaction_residual']]].sum(1),1,atol=1e-10)
 inventory.append(dict(dataset=name,source_profiles=len(m),analysis_profiles=len(e),source_drug_ids=e.drug.nunique(),cells=e.cell.nunique(),native_genes=int(f.genes.max()),common_genes=len(common),conditions=int(f.conditions.iloc[0]),normalization=firstnorm,doses=';'.join(map(str,sorted(e.dose.dropna().unique()))),times=';'.join(map(str,sorted(e.time.dropna().unique()))),scope='sealed role-owner TRAIN only, not whole public Tahoe' if name=='Tahoe' else 'entire fixed public release after stated QC/metadata eligibility',source_QC_flow=json.dumps(json.loads((R/'receipts'/f'{name}_PREPARED.json').read_text()),ensure_ascii=False)))
 rp=R/'results'/f'{name}_REPEAT_SUMMARY.csv'
 if rp.stat().st_size>5:summaries.append(pd.read_csv(rp))
 if name in ['Tahoe','L1000_L5','NCI_TPW']:continue
 m=e.copy();rr=pd.read_csv(R/'results'/f'{name}_REPEAT_CONDITIONS.csv.gz');base=rr[(rr['mode']=='delta')&(rr.axis=='native')&(rr.normalization==firstnorm)]
 # Independent pair-by-pair scipy replay, rather than the bulk algebra used by the production calculation.
 h=h5py.File(R/'data'/f'{name}.h5');ent=h['entrez'].asstr()[:];cp=[list(ent).index(g) for g in common];m=m.copy()
 for key in ['dose','time']:m[key]=m[key].map(lambda v:float(format(float(v),'.6g')))
 done=0
 for z in base.itertuples():
  ss=m[(m.drug==z.drug)&(m.cell==z.cell)&(m.dose==z.dose)&(m.time==z.time)&(m.panel==z.panel)] if 'panel' in m else m[(m.drug==z.drug)&(m.cell==z.cell)&(m.dose==z.dose)&(m.time==z.time)]
  if len(ss)<2 or len(ss)>8:continue
  x=h['delta'][ss.row.to_numpy(int)].astype(float);ok=(np.linalg.norm(x-x.mean(1,keepdims=True),axis=1)>1e-12)&(np.linalg.norm(x[:,cp]-x[:,cp].mean(1,keepdims=True),axis=1)>1e-12);ss=ss.iloc[np.where(ok)[0]];x=x[ok];vals=[];vcommon=[];pairs=0
  for i,j in itertools.combinations(range(len(ss)),2):
   if ss.iloc[i].plate==ss.iloc[j].plate:continue
   if name=='L1000_L3':assert set(ss.iloc[i].control_key.split(';')).isdisjoint(ss.iloc[j].control_key.split(';'))
   else:assert ss.iloc[i].control_key!=ss.iloc[j].control_key
   vals.append(pearsonr(x[i],x[j]).statistic);vcommon.append(pearsonr(x[i,cp],x[j,cp]).statistic);pairs+=1
  if not vals:continue
  oldcommon=rr[(rr['mode']=='delta')&(rr.axis=='common')&(rr.normalization==firstnorm)&(rr.drug==z.drug)&(rr.cell==z.cell)&(rr.dose==z.dose)&(rr.time==z.time)&(rr.panel==z.panel)].iloc[0]
  err=max(abs(np.mean(vals)-z.pcc),abs(np.mean(vcommon)-oldcommon.pcc));assert err<1e-8,(name,err);assert pairs==z.pairs
  checks.append(dict(dataset=name,drug=z.drug,cell=z.cell,pairs=pairs,max_scipy_error=err,disjoint_recorded_controls=True));done+=1
  if done==3:break
 assert done>0,name
 # Native and common analyses must use the same valid pair support.
 a=rr[rr.axis=='native'];b=rr[rr.axis=='common'];join=a.merge(b,on=['dataset','mode','normalization','panel','drug','cell','dose','time'],suffixes=('_native','_common'));assert len(join)==len(a)==len(b);assert (join.pairs_native==join.pairs_common).all();h.close()
# Whole L3 recorded control pools: verify no control RNA plate is borrowed across distinct treatment RNA plates.
l3=pd.read_parquet(R/'data/L1000_L3_META.parquet',columns=['plate','control_key']).drop_duplicates();bad=[]
for z in l3.itertuples():
 plates={v.split('|')[0] for v in z.control_key.split(';')}
 if plates!={z.plate}:bad.append(dict(plate=z.plate,controls=sorted(plates)))
assert not bad,bad[:3]
# Explicit least squares on a weighted incomplete design independently checks the final direct projection.
rng=np.random.default_rng(427);pairs=[(d,c) for d in range(9) for c in range(4) if (d+2*c)%5!=0];m=pd.DataFrame(pairs,columns=['drug','cell']);m['_weight']=rng.uniform(.1,1,len(m));x=rng.normal(size=(len(m),13));E,co,inf=decompose(x,m);d,c,w,D,Cc,_=weighted_design(m);X=np.c_[np.ones(len(m)),np.eye(9)[d],np.eye(4)[c]];pred=X@np.linalg.lstsq(X*np.sqrt(w[:,None]),x*np.sqrt(w[:,None]),rcond=None)[0];err=float(np.max(abs(x-pred-co['residual'])));assert err<1e-8
pd.DataFrame(inventory).to_csv(R/'results/DATASET_INVENTORY.csv',index=False);pd.concat(summaries,ignore_index=True).to_csv(R/'results/FIG1_FULL_SUMMARY.csv',index=False);pd.concat(compositions,ignore_index=True).to_csv(R/'results/FIG3_FULL_COMPOSITION.csv',index=False)
assert json.loads((R/'receipts/INTERSECTIONS.json').read_text())['status']=='PASS'
dump(R/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',repeated_pair_scipy_checks=checks,whole_L3_control_pool_bindings_checked=len(l3),whole_L3_cross_plate_control_overlap=0,weighted_sparse_explicit_lstsq_error=err,energy_closure=inf['closure'],Tahoe_independent_replay=json.loads((R/'receipts/Tahoe_REPEATS.json').read_text()),TEST_response_reads=0,seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID')))
for p in sorted((R/'code').glob('*.py')):sources.append(dict(kind='executed_code',path=str(p),sha256=sha(p),bytes=p.stat().st_size))
for p in [R/'PROTOCOL.json',R/'COMMON_GENES.json',A/'DRUG_ROLE_OWNER.json',C/'tahoe/data/condition_manifest.parquet',C/'l1000/data/l3_treatment_manifest.parquet',C/'l1000/data/l3_control_manifest.parquet']:
 sources.append(dict(kind='protocol_or_metadata',path=str(p),sha256=sha(p),bytes=p.stat().st_size))
for p in (R/'receipts').glob('*PREPARED.json'):
 z=json.loads(p.read_text());sources.append(dict(kind='prepared_native_matrix',path=str(R/'data'/(p.name.replace('_PREPARED.json','.h5'))),sha256=z['sha256'],bytes=z['bytes']))
pd.DataFrame(sources).to_csv(R/'INPUTS_AND_CODE_SHA256.csv',index=False)
print('FINAL_AUDIT PASS',flush=True)

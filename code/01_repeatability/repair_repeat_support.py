from ff_statistics import *
import shutil
assert not (R/'receipts/REPEAT_SUPPORT_REPAIR.json').exists(), 'Already repaired: do not overwrite standalone results'
start=time.time();name='DRUGseq';path=R/'results'/f'{name}_REPEAT_CONDITIONS.csv.gz';t=pd.read_csv(path);keys=['dataset','mode','normalization','panel','drug','cell','dose','time'];a=t[t.axis=='native'];b=t[t.axis=='common'];join=a.merge(b,on=keys,how='outer',suffixes=('_n','_c'));bad=join[join.pairs_n.ne(join.pairs_c)|join.pairs_n.isna()|join.pairs_c.isna()];rows=[]
shutil.copyfile(path,R/'results/DRUGseq_NATIVE_UNMATCHED_AXIS_SUPPORT.csv.gz');shutil.copyfile(R/'results/DRUGseq_REPEAT_SUMMARY.csv',R/'results/DRUGseq_UNMATCHED_AXIS_SUMMARY.csv')
md=pd.read_parquet(R/'data/DRUGseq_META.parquet');md=md[~md.is_control&md.valid_profile].copy();h=h5py.File(R/'data/DRUGseq.h5');ent=h['entrez'].asstr()[:];common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];cp=np.array([list(ent).index(g) for g in common]);removed=[]
for z in bad.to_dict('records'):
 keep=np.ones(len(t),bool)
 for k in keys:keep&=t[k].eq(z[k]).to_numpy()
 t=t[~keep].copy();ss=md[(md.normalization==z['normalization'])&(md.panel==z['panel'])&(md.drug==z['drug'])&(md.cell==z['cell'])&np.isclose(md.dose,z['dose'])&np.isclose(md.time,z['time'])];x=h[z['mode']][ss.row.to_numpy(int)].astype(float);ok=(np.linalg.norm(x-x.mean(1,keepdims=True),axis=1)>1e-12)&(np.linalg.norm(x[:,cp]-x[:,cp].mean(1,keepdims=True),axis=1)>1e-12);removed.append(dict(**{k:z[k] for k in keys},source_profiles=len(ss),joint_variable_profiles=int(ok.sum()),reason='PCC undefined on at least one gene axis; joint mathematical support, not activity filtering'))
 for axis,pos in [('native',np.arange(x.shape[1])),('common',cp)]:
  v,pairs=all_cross_plate_pcc(x[ok][:,pos],ss.plate.to_numpy()[ok])
  if pairs:rows.append(dict(**{k:z[k] for k in keys},axis=axis,genes=len(pos),pcc=v,pairs=pairs,physical_units=int(ok.sum()),plates=ss.plate[ok].nunique()))
t=pd.concat([t,pd.DataFrame(rows)],ignore_index=True);t.to_csv(path,index=False);summary=[]
for keys2,ss in t.groupby(['mode','axis','normalization']):
 d=ss.groupby('drug').pcc.mean();mean,lo,hi=boot_mean(d.to_numpy());summary.append(dict(dataset=name,mode=keys2[0],axis=keys2[1],normalization=keys2[2],pcc=mean,low=lo,high=hi,drugs=len(d),conditions=len(ss),pairs=int(ss.pairs.sum()),genes=int(ss.genes.iloc[0]),bootstrap='drug clusters',all_cross_plate_pairs_exact=True,joint_native_common_validity=True))
pd.DataFrame(summary).to_csv(R/'results/DRUGseq_REPEAT_SUMMARY.csv',index=False);pd.DataFrame(removed).to_csv(R/'results/DRUGseq_JOINT_AXIS_EXCLUSIONS.csv',index=False);dump(R/'receipts/REPEAT_SUPPORT_REPAIR.json',dict(status='PASS',affected_condition_normalizations=len(bad),seconds=time.time()-start,standalone_native_full_results_retained=True,job=os.getenv('TAHOE_REPRO_RUN_ID')))

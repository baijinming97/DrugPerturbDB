from ff_statistics import *
start=time.time();S=O/'REPEATABILITY_NATIVE_GENES_V1';common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];pos=GENE.set_index('l1000_entrez_gene_id').loc[common,'tahoe_gene_position'].astype(int).to_numpy()
old=pd.read_csv(S/'results/Tahoe_PAIR_RESULTS.csv.gz');handles=[h5py.File(p,'r') for p in sorted((S/'subsets').glob('Tahoe_native_shard*.h5'))];lookup={int(a):(h,k) for h in handles for k,a in enumerate(h['analysis_indices'][:]) if a>=0};rows=[];verify=[]
for r in old.itertuples():
 ha,ia=lookup[int(r.i)];hb,ib=lookup[int(r.j)]
 for mode in ['post','delta']:
  a=ha[mode][ia].astype(float);b=hb[mode][ib].astype(float);v=float(pearson_rows(a,b));previous=getattr(r,mode+'_full');assert abs(v-previous)<1e-8,(r.i,r.j,mode,v,previous)
  for axis,cc,gg in [('native',v,len(a)),('common',float(pearson_rows(a[pos],b[pos])),len(pos))]:rows.append(dict(dataset='Tahoe',mode=mode,axis=axis,normalization='author_logCPM',drug=r.drug,group=r.group,cell=r.cell,dose=r.dose,time=r.time,pcc=cc,pairs=1,genes=gg))
  if len(verify)<10:
   from scipy.stats import pearsonr
   err=abs(float(pearsonr(a[pos],b[pos]).statistic)-float(pearson_rows(a[pos],b[pos])));assert err<1e-10;verify.append(dict(i=int(r.i),j=int(r.j),mode=mode,error=err))
for h in handles:h.close()
t=pd.DataFrame(rows);t.to_csv(R/'results/Tahoe_REPEAT_CONDITIONS.csv.gz',index=False);summary=[]
for keys,ss in t.groupby(['mode','axis','normalization']):
 d=ss.groupby(['group','drug']).pcc.mean().reset_index();groups=d.groupby('group').pcc.mean();mean,lo,hi=boot_mean(groups.to_numpy());summary.append(dict(dataset='Tahoe',mode=keys[0],axis=keys[1],normalization=keys[2],pcc=mean,low=lo,high=hi,drugs=d.drug.nunique(),conditions=len(ss),pairs=int(ss.pairs.sum()),genes=int(ss.genes.iloc[0]),bootstrap='frozen chemical supercomponent; same repeats as full native validated recovery',scope='all currently verified role-safe native single-plate pairs; plate 6/14; 5 uM; missing unsafe source row groups excluded'))
pd.DataFrame(summary).to_csv(R/'results/Tahoe_REPEAT_SUMMARY.csv',index=False);dump(R/'receipts/Tahoe_REPEATS.json',dict(status='PASS',seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID'),n_pairs=len(old),source_exclusions=str(S/'results/Tahoe_EXCLUSIONS.csv'),independent_checks=verify,new_TEST_response_reads=0))

from ff_statistics import *
import itertools
start=time.time();identity=json.loads((R/'IDENTITY_MAP.json').read_text());common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];datasets={};flows=[]
for name in identity['drug']:
 assert (R/'receipts'/f'{name}_ANALYZED.json').exists(),name
 md=pd.read_parquet(R/'data'/f'{name}_CONDITION_META.parquet');md=md[(md.normalization!='logCPM100')].copy();md['source_drug']=md.drug;md['drug']=md.drug.map(identity['drug'][name]);md['cell']=md.cell.astype(str).map(identity['cell'][name]);md=md[md.drug.notna()&md.cell.fillna('').ne('')].copy()
 with h5py.File(R/'data'/f'{name}_CONDITIONS.h5') as h:
  ent=h['entrez'].asstr()[:];lu={v:i for i,v in enumerate(ent) if v};pos=np.array([lu[v] for v in common]);sort=np.argsort(pos);rows=md.row.to_numpy(int);values=np.empty((len(rows),len(pos)),np.float64)
  # Read only verified identity rows, retaining all genes in the common measured axis.
  for lo in range(0,len(rows),256):
   rr=rows[lo:lo+256];values[lo:lo+256]=(h['delta'][int(rr[0]):int(rr[-1])+1][rr-rr[0]] if rr[-1]-rr[0]<2*len(rr) else h['delta'][rr])[:,pos]
 md=md.reset_index(drop=True);md['value_row']=np.arange(len(md));groups=md.groupby(['drug','cell','dose','time'],sort=True).indices;ms=[];xs=[]
 for k,ii in groups.items():ms.append(dict(drug=k[0],cell=k[1],dose=k[2],time=k[3]));xs.append(values[ii].mean(0))
 datasets[name]=(pd.DataFrame(ms),np.array(xs));flows.append(dict(dataset=name,confirmed_conditions=len(ms),confirmed_drugs=len(set(z['drug'] for z in ms)),confirmed_cells=len(set(z['cell'] for z in ms))))
 print('loaded',name,len(ms),flush=True)
pd.DataFrame(flows).to_csv(R/'results/INTERSECTION_ELIGIBILITY.csv',index=False)
owners=json.loads((A/'DRUG_ROLE_OWNER.json').read_text());cm=pd.read_parquet(C/'tahoe/data/condition_manifest.parquet');tgroups=cm.drop_duplicates('drug').set_index('drug').chemistry_component.to_dict();clusters={}
for drug,k in identity['drug']['Tahoe'].items():clusters[k]=tgroups.get(drug,k)
results=[];shares=[];support=[];skips=[]
for na,nb in itertools.combinations(datasets,2):
 if na.startswith('L1000') and nb.startswith('L1000'):continue  # Same source assays, not an independent cross-platform comparison.
 ma,xa=datasets[na];mb,xb=datasets[nb]
 for tm in sorted(set(ma.time)&set(mb.time)):
  aa={d:z for d,z in ma[ma.time==tm].groupby('drug')};bb={d:z for d,z in mb[mb.time==tm].groupby('drug')};pairs=[]
  for drug in sorted(set(aa)&set(bb)):
   da=aa[drug];db=bb[drug];choices=[]
   for dosea,ga in da.groupby('dose'):
    for doseb,gb in db.groupby('dose'):
     if min(dosea,doseb)<=0 or max(dosea,doseb)/min(dosea,doseb)>2+1e-8:continue
     cells=sorted(set(ga.cell)&set(gb.cell))
     if cells:choices.append((abs(np.log(dosea/doseb)),-len(cells),dosea,doseb,cells,ga,gb))
   if not choices:continue
   z=min(choices,key=lambda z:z[:4]);_,_,dosea,doseb,cells,ga,gb=z
   for cell in cells:pairs.append(dict(drug=drug,cell=cell,dose_a=dosea,dose_b=doseb,row_a=int(ga[ga.cell==cell].index[0]),row_b=int(gb[gb.cell==cell].index[0])))
  if not pairs:continue
  m=pd.DataFrame(pairs);nc=m.cell.nunique();nd=m.drug.nunique()
  if nd<3:skips.append(dict(a=na,b=nb,time=tm,drugs=nd,cells=nc,reason='fewer than 3 drugs; no substantive component comparison'));continue
  x=xa[m.row_a];y=xb[m.row_b];ea,ca,ia=decompose(x,m);eb,cb,ib=decompose(y,m);pid=f'{na}__{nb}__{tm:g}h';m.assign(panel_id=pid,a=na,b=nb,time=tm).to_csv(R/'results'/f'MATCHED_{pid}.csv.gz',index=False)
  support.append(dict(panel_id=pid,a=na,b=nb,time=tm,**ia,max_dose_ratio=float(np.maximum(m.dose_a/m.dose_b,m.dose_b/m.dose_a).max()),shared_genes=len(common),same_platform=na.startswith('L1000') and nb.startswith('L1000')))
  for name,e in [(na,ea),(nb,eb)]:
   for j,c in enumerate(['mean','cell','drug','interaction_residual']):shares.append(dict(panel_id=pid,a=na,b=nb,dataset=name,time=tm,component=c,fraction=e[j]/e.sum(),energy=e[j],drugs=nd,cells=nc,observed=len(m),identifiable=ia['interaction_identifiable'],connected_components=ia['connected_components']))
  for c in ['complete','mean','cell','drug','interaction_residual']:
   key='residual' if c=='interaction_residual' else c
   if c in ['cell','interaction_residual'] and not ia['interaction_identifiable'] or c in ['cell','drug'] and ia['connected_components']!=1:
    results.append(dict(panel_id=pid,a=na,b=nb,time=tm,component=c,pcc=np.nan,low=np.nan,high=np.nan,drugs=nd,cells=nc,status='not identifiable on this design'));continue
   u,v=(x,y) if c=='complete' else (ca['mu'][None],cb['mu'][None]) if c=='mean' else (ca[key],cb[key])
   if c=='mean':mean=float(pearson_rows(u,v)[0]);lo=hi=np.nan
   else:
    vals=pearson_rows(u,v);per=pd.DataFrame({'drug':m.drug,'pcc':vals}).groupby('drug').pcc.mean();cl=[clusters.get(d,d) for d in per.index];mean,lo,hi=boot_mean(per.to_numpy(),cl if 'Tahoe' in [na,nb] else None)
    if c=='cell':lo=hi=np.nan
   results.append(dict(panel_id=pid,a=na,b=nb,time=tm,component=c,pcc=mean,low=lo,high=hi,drugs=nd,cells=nc,status='descriptive',bootstrap='chemical groups' if 'Tahoe' in [na,nb] else 'drug clusters',normalization='source native R1; counts logCPM1',shared_genes=len(common)))
  print(pid,nd,nc,len(m),flush=True)
pd.DataFrame(results).to_csv(R/'results/INTERSECTION_CONSISTENCY.csv',index=False);pd.DataFrame(shares).to_csv(R/'results/INTERSECTION_COMPOSITION.csv',index=False);pd.DataFrame(support).to_csv(R/'results/INTERSECTION_SUPPORT.csv',index=False);pd.DataFrame(skips).to_csv(R/'results/INTERSECTION_SKIPPED.csv',index=False)
dump(R/'receipts/INTERSECTIONS.json',dict(status='PASS',seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID'),panels=len(support),TEST_reads=0,selection='exact time; dose ratio<=2; closest log dose then largest cell support, all metadata only; no response-selected panels',limitations='Identity unresolved sources retained in own-full but excluded from cross-source; sparse coefficients not reported when disconnected; single-cell interaction not estimable; intervals conditional on fitted descriptive components'))

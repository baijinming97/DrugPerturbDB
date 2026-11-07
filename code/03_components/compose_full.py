from ff_statistics_global import *
start=time.time();name=sys.argv[1];assert (R/'receipts'/f'{name}_ANALYZED.json').exists();meta=pd.read_parquet(R/'data'/f'{name}_CONDITION_META.parquet');common=json.loads((R/'COMMON_GENES.json').read_text())['axis'];rows=[];summary=[]
with h5py.File(R/'data'/f'{name}_CONDITIONS.h5') as h:
 g=h['gene'].asstr()[:];ent=h['entrez'].asstr()[:];emap={e:i for i,e in enumerate(ent) if e};cp=np.array([emap[e] for e in common]);x=h['delta'][:]
 for norm,mm in meta.groupby('normalization'):
  d,_=pd.factorize(mm.drug);w=1/np.bincount(d)[d];w/=w.sum();mm=mm.copy();mm['_weight']=w
  for axis,pos in [('native',np.arange(len(g))),('common',cp)]:
   total=np.zeros(4);ident=0.;connected=0.;closure=0.;strata=0
   for keys,sm in mm.groupby(['panel','dose','time'],sort=True):
    mass=sm._weight.sum();e=np.zeros(4)
    for lo in range(0,len(pos),256):
     ec,co,inf=decompose(x[np.ix_(sm.row.to_numpy(int),pos[lo:lo+256])],sm);e+=ec;closure=max(closure,inf['closure'])
    total+=mass*e;ident+=mass*inf['interaction_identifiable'];connected+=mass*(inf['connected_components']==1);strata+=1
    rows.append(dict(dataset=name,normalization=norm,axis=axis,panel=keys[0],dose=keys[1],time=keys[2],stratum_weight=mass,genes=len(pos),**inf,**{k+'_energy':e[j] for j,k in enumerate(['mean','cell','drug','interaction_residual'])}))
   rr=dict(dataset=name,normalization=norm,axis=axis,drugs=mm.drug.nunique(),cells=mm.cell.nunique(),conditions=len(mm),genes=len(pos),strata=strata,interaction_identifiable_weight=ident,connected_design_weight=connected,max_closure=closure,weighting='each drug equal globally; its observed cell/dose/time/library conditions equal; project separately within each dose/time/library stratum')
   for j,k in enumerate(['mean','cell','drug','interaction_residual']):rr[k+'_energy']=total[j];rr[k+'_fraction']=total[j]/total.sum()
   summary.append(rr);print(name,norm,axis,flush=True)
pd.DataFrame(rows).to_csv(R/'results'/f'{name}_FULL_STRATA.csv.gz',index=False);pd.DataFrame(summary).to_csv(R/'results'/f'{name}_FULL_COMPOSITION.csv',index=False);dump(R/'receipts'/f'{name}_COMPOSITION_FINAL.json',dict(status='PASS',seconds=time.time()-start,job=os.getenv('TAHOE_REPRO_RUN_ID'),full_observed_support=True,globally_drug_equal=True))

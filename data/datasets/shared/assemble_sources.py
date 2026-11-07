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
# Recompute the common measured Entrez axis after all full data sources are present.
sets={}
for nm in ['Tahoe','L1000_L3','L1000_L5','DRUGseq','CIGS','sciPlex3','MIXseq','NCI_TPW']:
 assert (R/'data'/f'{nm}.h5').exists(),nm
 if nm=='DRUGseq':ids=gene_ids((R/'data/DRUGseq_GENE_IDS.txt').read_text().splitlines(),'drugseq')
 else:
  with h5py.File(R/'data'/f'{nm}.h5','r') as h:ids=h['entrez'].asstr()[:]
 ids=ids[ids!=''];vc=pd.Series(ids).value_counts();sets[nm]=set(vc[vc==1].index)
dump(R/'receipts/CIGS_GENE_ALIAS_AUDIT.json',dict(punctuation_aliases='HLA/NKX dots to hyphens',unresolved_excel_date='45542 and X7.Sep retained as unassigned measured feature; excluded from Entrez common genes',no_response_based_mapping=True))
axis=[g for g in GENE.l1000_entrez_gene_id if all(g in s for s in sets.values())]
dump(R/'COMMON_GENES.json',dict(axis=axis,n=len(axis),previous_882=COMMON,lost_from_previous=[g for g in COMMON if g not in axis],gained=[g for g in axis if g not in COMMON],coverage={nm:len(s) for nm,s in sets.items()},rule='intersection of actually measured, uniquely mapped Entrez genes; no missing-zero fill'))
print('COMMON_GENES',len(axis),flush=True)

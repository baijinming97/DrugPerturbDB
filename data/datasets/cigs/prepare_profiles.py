from ff_common import *
from ff_xlsx import frame,rows
import re
start=time.time();which=int(sys.argv[1]);labels=['MCE_Bioactive_Compounds_HEK293T_10uM','MCE_Bioactive_Compounds_MDA_MB_231_10uM','TCM_Compounds_HEK293T_10','TCM_Compounds_HEK293T_20','TCM_Compounds_MDA_MB_231_10','TCM_Compounds_MDA_MB_231_20'];label=labels[which];p=R/'raw'/f'{label}_Counts.xlsx';name='CIGS_'+str(which)
if (R/'receipts'/f'{name}_PREPARED.json').exists():raise SystemExit('ALREADY_PREPARED')
md=frame(p.with_name(p.name.replace('_Counts','_MetaData')))
if 'unique_ID' in md:
 md=md.rename(columns={'unique_ID':'physical_unit','sample_plate':'plate','sample':'drug','pert_idose':'dose_text','pert_itime':'time_text'});md['panel']='MCE'
 md['drug']=md.treatment;md['metadata_eligible']=~md.treatment.str.lower().isin(['blank','rna'])
else:
 md=md.rename(columns={'Sample_unique_id':'physical_unit','Plate':'plate','Treat':'drug','Dose':'dose_text','Time':'time_text'});md['panel']='TCM'
 md['metadata_eligible']=True
md['cell']='CVCL_0063' if 'HEK293T' in label else 'CVCL_0062';md['is_control']=md.drug.str.upper().str.contains('DMSO');md['dose']=md.dose_text.str.extract(r'([0-9.]+)')[0].astype(float);md.loc[md.is_control,'dose']=0.;md['time']=md.time_text.str.extract(r'([0-9.]+)')[0].astype(float)
assert md.loc[~md.is_control&md.metadata_eligible,'dose_text'].str.lower().str.contains('um').all();assert md.physical_unit.is_unique
lookup={v:i for i,v in enumerate(md.physical_unit)};head=None;seen=np.zeros(len(md),bool)
count_source=p
if which in (3,5):
 count_source=R/'raw/cigs_snapshots'/f'{label}_Counts.h5'
 with h5py.File(count_source) as hs:
  genes=hs['gene'].asstr()[:].tolist();ids=hs['physical_unit'].asstr()[:]
  assert len(ids)==len(md) and set(ids)==set(lookup)
  ct=np.empty((len(md),len(genes)),np.float32)
  loc=np.array([lookup[uid] for uid in ids]);ct[loc]=hs['counts'][:];seen[loc]=True
else:
 for rr in rows(p):
  if head is None:
   if len(rr)<1000:continue
   head=rr;idcol=min(rr);gcols=sorted(k for k in rr if k!=idcol);genes=[rr[k] for k in gcols];assert len(set(genes))==len(genes);ct=np.zeros((len(md),len(genes)),np.float32)
  else:
   uid=rr.get(idcol)
   if not uid:continue
   assert uid in lookup,(p.name,uid);i=lookup[uid];assert not seen[i];seen[i]=True;ct[i]=[float(rr.get(j,0) or 0) for j in gcols]
assert seen.all() and np.isfinite(ct).all() and (ct>=0).all();total=ct.sum(1,dtype=np.float64);zero=total<=0;md['library_total']=total
eligible=~zero&md.metadata_eligible.to_numpy();md.loc[~eligible].to_csv(R/'data'/f'{name}_EXCLUDED.csv',index=False);ct=ct[eligible];total=total[eligible];md=md.loc[eligible].reset_index(drop=True)
with h5py.File(R/'data'/f'{name}_COUNTS.h5','w') as h:
 h.create_dataset('counts',data=ct,compression='gzip',compression_opts=1);h.create_dataset('total',data=total);h.create_dataset('gene',data=np.asarray(genes,object),dtype=h5py.string_dtype());h.create_dataset('physical_unit',data=md.physical_unit.to_numpy(object),dtype=h5py.string_dtype())
outmeta=pd.concat([md.assign(normalization='logCPM1'),md.assign(normalization='logCPM100')],ignore_index=True);outmeta['plate']=label+'|'+outmeta.plate;outmeta['control_key']=outmeta.plate+'|'+outmeta.cell;outmeta['repeat_type']='cross_plate';outmeta['source_file']=p.name
with newh5(name,outmeta,genes,'symbol') as h:
 for a,pc in enumerate([1,100]):
  expr=np.log2(ct.astype(np.float64)/total[:,None]*1e6+pc);delta=expr.copy()
  for pl,ix in md.groupby('plate').indices.items():
   ci=ix[md.iloc[ix].is_control.to_numpy()];assert len(ci)>0,(p.name,pl,'no controls');delta[ix]-=expr[ci].mean(0)
  sl=slice(a*len(md),(a+1)*len(md));h['post'][sl]=expr;h['delta'][sl]=delta
finish(name,start,dict(source=count_source.name,raw_sha256=sha(count_source),source_rows=len(seen),excluded_zero_library=int(zero.sum()),physical_profiles=len(md),drugs=md.loc[~md.is_control,'drug'].nunique(),genes=len(genes),dose_unit='uM verified from metadata',all_source_rows_matched=True))

from ff_common import *
import re,zipfile
from scipy.io import mmread
start=time.time();name='MIXseq';S=O/'4MAN/TAHOE_GAMMA_PARALLEL_V1/ws3_multicell_external/v2';dosetable=pd.read_csv(S/'MIX_META_2.csv');clean=lambda s:re.sub('[^a-z0-9]','',str(s).lower());doses={clean(r['Drug']):float(r['Concentration (uM)']) for _,r in dosetable.iterrows()}
# Publisher file is misspelled AZD5591; retain this source ID, explicitly link only its experimental dose to the metadata entry.
doses['azd5591']=doses['azd5991'];parts=[];meta=[];axis=None;sourceaudit=[]
for p in sorted((R/'raw').glob('*.zip')):
 ex=re.search(r'expt(\d+)',p.name)
 if ex is None:continue
 ex='expt'+ex.group(1);cache=R/'data'/(p.name+'.PB.npz')
 if cache.exists():
  z=np.load(cache);rows=json.loads(str(z['metadata']));g=z['gene'];cts=z['counts'];sourceaudit.append(dict(file=p.name,reused_pseudobulk=True))
 else:
  with zipfile.ZipFile(p) as z:
   members={Path(v).name:v for v in z.namelist() if not v.endswith('/') and not v.startswith('__MACOSX/') and not Path(v).name.startswith('._')}
   g=pd.read_csv(z.open(members['genes.tsv']),sep='\t',header=None)[0].astype(str).str.split('.').str[0].to_numpy();assert len(g)==len(set(g));bar=pd.read_csv(z.open(members['barcodes.tsv']),sep='\t',header=None)[0].astype(str);cl=pd.read_csv(z.open(members['classifications.csv']));cl=cl.set_index('barcode').reindex(bar)
   ok=cl.cell_quality.eq('normal')&cl.DepMap_ID.notna();cl['drug']=p.name.split('_')[0];cl['time']=float(re.search(r'_(\d+)hr',p.name).group(1)) if re.search(r'_(\d+)hr',p.name) else 24.
   if 'tc_expt5' in p.name:
    tag=cl.hash_assignment.astype(str);ok&=tag.str.match(r'^(DMSO|Tram)_\d+hr$');cl['drug']=np.where(tag.str.startswith('Tram_'),'Trametinib','DMSO');cl['time']=pd.to_numeric(tag.str.extract(r'_(\d+)hr')[0],errors='coerce').to_numpy()
   if p.name.startswith('Untreated'):ok&=False
   mat=mmread(z.open(members['matrix.mtx'])).tocsr();assert mat.shape==(len(g),len(bar));assert (mat.data>=0).all() and (mat.data==np.rint(mat.data)).all();rows=[];ls=[]
   for key,ix in cl[ok].groupby(['DepMap_ID','drug','time'],sort=True).indices.items():
    # groupby indices address the filtered frame; map back to matrix barcode positions.
    ids=np.flatnonzero(ok.to_numpy())[ix];cell,drug,tm=key;ct=np.asarray(mat[:,ids].sum(1)).ravel().astype(np.int64)
    rows.append(dict(cell=cell,cell_name=str(cl.iloc[ids[0]].singlet_ID),drug=drug,time=float(tm),dose=0. if drug=='DMSO' else doses[clean(drug)],plate=ex,n_cells=len(ids),library_total=int(ct.sum()),source_file=p.name,is_control=drug=='DMSO',physical_unit=p.stem+'|'+cell+'|'+str(tm)));ls.append(ct)
   cts=np.stack(ls) if ls else np.zeros((0,len(g)),np.int64);sourceaudit.append(dict(file=p.name,raw_sha256=sha(p),source_cells=len(cl),QC_cells=int(ok.sum()),pseudobulks=len(rows)))
  np.savez_compressed(cache,counts=cts,gene=g,metadata=json.dumps(rows))
 if axis is None:axis=g
 assert set(axis)==set(g);order=pd.Index(g).get_indexer(axis);parts.append(cts[:,order]);meta.extend(rows);print('parsed',p.name,len(rows),flush=True)
counts=np.concatenate(parts);md=pd.DataFrame(meta);assert len(counts)==len(md);md['valid_profile']=(md.n_cells>=20)&(md.library_total>0);md['control_key']=md.plate+'|'+md.cell+'|'+md.time.astype(str);md['repeat_type']='independent_experiment';md['panel']='MIXseq';total=md.library_total.to_numpy()
controls={k:ix[md.iloc[ix].is_control.to_numpy()&md.iloc[ix].valid_profile.to_numpy()] for k,ix in md.groupby('control_key').indices.items()}
md['control_available']=md.control_key.map(lambda k:len(controls[k])>0);md['valid_profile']&=md.control_available;outmeta=pd.concat([md.assign(normalization='logCPM1'),md.assign(normalization='logCPM100')],ignore_index=True)
with newh5(name,outmeta,axis,'ensembl') as h:
 for a,pc in enumerate([1,100]):
  expr=np.log2(counts/np.maximum(total,1)[:,None]*1e6+pc);delta=np.zeros_like(expr)
  for k,ix in md.groupby('control_key').indices.items():
   ci=controls[k]
   if len(ci):delta[ix]=expr[ix]-expr[ci].mean(0)
  h['post'][a*len(md):(a+1)*len(md)]=expr;h['delta'][a*len(md):(a+1)*len(md)]=delta
dump(R/'receipts/MIXseq_SOURCE_AUDIT.json',sourceaudit);md[~md.valid_profile].to_csv(R/'data/MIXseq_EXCLUDED.csv',index=False)
finish(name,start,dict(source_archives=len(sourceaudit),pseudobulks=len(md),QC_pseudobulks=int(md.valid_profile.sum()),drugs=md.loc[~md.is_control,'drug'].nunique(),cells=md.cell.nunique(),minimum_cells=20,exact_time_control_required=True,native_genes=len(axis),timecourse_channels_not_independent_repeats=True))

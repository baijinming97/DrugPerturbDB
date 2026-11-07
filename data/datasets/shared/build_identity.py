from ff_common import *
import re
names=['Tahoe','L1000_L3','L1000_L5','DRUGseq','CIGS','sciPlex3','MIXseq','NCI_TPW'];parents={};evidence=[]
clean=lambda x:re.sub('[^A-Z0-9]','',str(x).upper())
metas={nm:pd.read_parquet(R/'data'/f'{nm}_META.parquet') for nm in names}
def node(n,d):return n+'|'+str(d)
def root(x):
 parents.setdefault(x,x)
 if parents[x]!=x:parents[x]=root(parents[x])
 return parents[x]
def link(a,b,why):
 a0,b0=root(a),root(b)
 if a0!=b0:parents[max(a0,b0)]=min(a0,b0)
 evidence.append(dict(left=a,right=b,evidence=why))
observed={nm:set(m.drug.astype(str)) for nm,m in metas.items()};lookup={nm:{clean(d):d for d in ds} for nm,ds in observed.items()}
def extlink(n,external,tahoe,why,key=None):
 d=lookup[n].get(clean(external))
 if d is None or tahoe not in observed['Tahoe']:return
 link(node(n,d),node('Tahoe',tahoe),why)
 if key and re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',str(key)):link(node(n,d),'KEY|'+key,why)
# Entire disclosed external structure tables, not restricted to the Tahoe drug list.
for nm,path,idcol in [('L1000_L3',R/'raw/compoundinfo_beta.txt','pert_id'),('L1000_L5',R/'raw/compoundinfo_beta.txt','pert_id'),('DRUGseq',R/'raw/MoABox_compounds_metadata.txt','cmpd_sample_id')]:
 tab=pd.read_csv(path,sep='\t',dtype=str).fillna('')
 for z in tab.to_dict('records'):
  if z[idcol] in observed[nm] and re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',z['inchi_key']):link(node(nm,z[idcol]),'KEY|'+z['inchi_key'],'official full InChIKey')
# Frozen structure-confirmed mappings retain stereochemistry and explicit parent decisions.
t=pd.read_csv(O/'4MAN/CIGS_BETAW_PILOT_V1/inputs/CIGS_STRICT_MAP.csv').fillna('')
for z in t.to_dict('records'):
 if z['chemical_identity_status'] in ['EXACT_FULL_INCHIKEY','PARENT_FULL_INCHIKEY']:
  for ext in [z['external_id'],z['external_name']]:extlink('CIGS',ext,z['tahoe_drug'],z['chemical_identity_status'],z.get('tahoe_exact_key'))
t=pd.read_csv(O/'4MAN/DRUGSEQ_BETAW_PILOT_V1/inputs/STRICT_MAP.csv').fillna('')
for z in t.to_dict('records'):extlink('DRUGseq',z['external_id'],z['tahoe_drug'],z['match_level'],z['external_key'])
t=pd.read_csv(A/'SCIPLEX_DRUG_IDENTITY.csv').fillna('')
for z in t.to_dict('records'):
 if str(z['accepted']).lower()=='true':
  for ext in [z['source_name'],z['catalog']]:extlink('sciPlex3',ext,z['tahoe_drug'],z['status'],z['pubchem_keys'] if z['status']=='EXACT_FULL_INCHIKEY' else z['restored_key'])
t=pd.read_csv(O/'4MAN/TAHOE_GAMMA_PARALLEL_V1/ws3_multicell_external/v2/DRUG_IDENTITY_AUDIT.csv').fillna('')
for z in t.to_dict('records'):
 if str(z['accepted']).lower()=='true':
  for nm in ['MIXseq','NCI_TPW']:extlink(nm,z['source_name'],z['Tahoe_drug'],z['match'],z['source_parent'])
p=O/'4MAN/GAMMA_PRISM_INC_V1/features/LINCS_DRUG_MATCHES.csv'
if p.exists():
 t=pd.read_csv(p).fillna('');dump(R/'inventory/LINCS_MATCH_SCHEMA.json',dict(columns=list(t),head=t.head(3).to_dict('records')))
 for z in t.to_dict('records'):
  if 'inchi' in str(z.get('method','')).lower():
   for nm in ['L1000_L3','L1000_L5']:extlink(nm,z['pert_id'],z['drug'],z['method'])
groups={}
for nm,ds in observed.items():
 for d in ds:
  key=node(nm,d)
  if key in parents:groups.setdefault(root(key),[]).append((nm,d))
mapping={nm:{} for nm in names}
for k,members in groups.items():
 if len(set(n for n,d in members))<2:continue
 for n,d in members:mapping[n][d]=k
cm=json.loads((A/'CELL_IDENTITY_MAP.json').read_text());official=pd.read_csv(R/'raw/cellinfo_beta.txt',sep='\t').fillna('')
for z in official.to_dict('records'):
 if str(z['cellosaurus_id']).startswith('CVCL_'):cm[clean(z['cell_iname'])]=z['cellosaurus_id']
cm.update({'U2OS':'CVCL_0042','HEK293T':'CVCL_0063','MDAMB231':'CVCL_0062','K562':'CVCL_0004','MCF7':'CVCL_0031','A549':'CVCL_0023'})
mx=pd.read_csv(O/'4MAN/TAHOE_GAMMA_PARALLEL_V1/ws3_multicell_external/v2/MIX_META_0.csv').fillna('')
dep={z['DepMap ID']:cm.get(clean(z['CCLE Name'].split('_')[0]),'') for z in mx.to_dict('records')}
cellmaps={};coverage=[]
for nm,m in metas.items():
 cellmaps[nm]={str(c):(str(c) if str(c).startswith('CVCL_') else dep.get(str(c),'') or cm.get(clean(c),'')) for c in m.cell.unique()}
 coverage.append(dict(dataset=nm,source_drug_ids=len(observed[nm]),drug_ids_linked_to_other_sources=len(mapping[nm]),source_cells=m.cell.nunique(),cells_resolved=sum(bool(v) for v in cellmaps[nm].values())))
pd.DataFrame(evidence).to_csv(R/'results/CHEMICAL_IDENTITY_EVIDENCE.csv',index=False);pd.DataFrame(coverage).to_csv(R/'results/IDENTITY_COVERAGE.csv',index=False)
dump(R/'IDENTITY_MAP.json',dict(drug=mapping,cell=cellmaps,scope='full publicly disclosed exact InChIKeys plus frozen structure-confirmed mappings; unresolved identities remain in own-full results but are excluded from cross-source intersections; no response-based or name-only drug matching'))
print(pd.DataFrame(coverage).to_string(index=False),flush=True)

from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];I=R.parent/'FIG3_NATIVE_INTERSECTION_V1/inventory'
files=[]
for x in json.loads((I/'SIZE_CHECKS.json').read_text(encoding='utf-8')):
 u=x.get('url','')
 if 'docs/cigs/' in u and ('MCE_' in u or 'TCM_' in u):
  n=u.rsplit('/',1)[-1].replace('%CE%BC','u')
  files.append(dict(dataset='CIGS',url=u,name=n,bytes=x.get('bytes',534846609 if 'MDA_MB_231_10%CE%BCM_Counts' in u else None)))
mix=json.loads((I/'MIX_api.txt').read_text())
for f in mix['files']:
 if f['name'].startswith('sg') or f['name']=='all_CL_features.rds':continue
 files.append(dict(dataset='MIXseq',url=f['download_url'],name=f['name'].replace(' ','_'),bytes=f['size'],md5=f['computed_md5']))
files.extend([
dict(dataset='DRUGseq',url='https://zenodo.org/records/14291446/files/Exp_gzip.RData?download=1',name='Exp_gzip.RData',bytes=2068039088,sha256='4bdad59bc37518c036340114257f9b5e298d5eb2d269a2201ba0c8c20ba7d0a1'),
dict(dataset='sciPlex3',url='https://zenodo.org/api/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad/content',name='sciplex3.h5ad',bytes=2526631614,sha256='603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a'),
dict(dataset='NCI_TPW',url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116436/matrix/GSE116436_series_matrix.txt.gz',name='GSE116436_series_matrix.txt.gz',max_bytes=1100000000),
])
base='https://raw.githubusercontent.com/Novartis/DRUG-seq/main/data/Novartis_drugseq_U2OS_MoABox/'
for n in ['DRUGseq_U2OS_MoABox_plate_wells_metadata_public.txt','MoABox_compounds_metadata.txt','MoABox_compounds_target_pairs_public.txt']:
 files.append(dict(dataset='DRUGseq',url=base+n,name=n,max_bytes=30*2**20))
(R/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
protocol = {'task': 'FULL_DATASET_FIGURES_V1', 'GPU': 0, 'public_scope': 'fixed releases; all QC-eligible compounds/cells/doses/times; no Tahoe identity-intersection filtering', 'Tahoe_scope': 'existing role owner TRAIN-only, excluded TEST; disclose total/retained and dose coverage', 'fig1': 'independent repeat per exact drug/cell/dose/time and disjoint recorded controls; absolute post-treatment vs control-difference; all native measured genes and recomputed common axis; split-cell technical repeat separate; L5 consensus not called individual replicate', 'fig3': 'complete balanced panels have orthogonal mu/alpha/beta/gamma; whole sparse panels use additive least-squares with order-symmetric main-effect energy allocation, explicitly distinct from orthogonal factorial identity; show identifiable subset, uncovered edges and design connectivity; no zero fill', 'intersections': 'pairwise exact chemical identity and Cellosaurus; exact time and dose ratio <=2; re-decompose each matched panel; gene masks, not zero imputation', 'weights': 'full native descriptive drug equal, cell equal conditional on available support; pairwise matched identical support weights; chemical cluster bootstrap when metadata permits, otherwise explicitly drug-cluster bootstrap', 'bootstrap_replicates': 2000, 'no_training': True, 'frozen_assets_readonly': True, 'normalization': 'RNA count: log2(CPM+1), sensitivity +100; library denominator native measured counts; L1000 L3 native normalized values; L5 separately standardized consensus; NCI author normalized log2 array expression; cannot compare absolute units across technologies', 'scope_integrity': 'source->QC->analysis counts and exclusions; report all failed sources; avoid silently labeling retained Tahoe-matched subsets full'}
(R/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(files),sum(x.get('bytes',x.get('max_bytes',0)) for x in files)/2**30)

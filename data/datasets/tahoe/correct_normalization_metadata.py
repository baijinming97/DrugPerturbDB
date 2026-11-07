from pathlib import Path
import hashlib,json,os,subprocess,sys
import pandas as pd
R=Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2')
assert os.environ.get('TAHOE_REPRO_RUN_ID')
records=[]
for rel in ['data/Tahoe_CONDITION_META.parquet','results/Tahoe_FULL_COMPOSITION.csv']:
 p=R/rel;before=hashlib.sha256(p.read_bytes()).hexdigest()
 table=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,float_precision='round_trip')
 original=table.drop(columns='normalization').copy()
 old=sorted(table.normalization.unique());assert set(old)<=set(['author_logCPM','author_linear_normalized'])
 table['normalization']='author_linear_normalized'
 if p.suffix=='.parquet':table.to_parquet(p,index=False)
 else:table.to_csv(p,index=False)
 check=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,float_precision='round_trip')
 pd.testing.assert_frame_equal(original,check.drop(columns='normalization'),check_exact=True)
 records.append(dict(path=str(p),old=old,new='author_linear_normalized',before_sha256=before,after_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'receipts/NORMALIZATION_METADATA_CORRECTION.json').write_text(json.dumps(dict(status='PASS',records=records,numeric_values_changed=False,reason='Inherited label author_logCPM was inaccurate; retained source arrays are author-provided linear normalized expression. No transform or estimate changed.',frozen_sources_modified=False),indent=2))
subprocess.run([sys.executable,str(R/'scripts/final_verify.py')],check=True)


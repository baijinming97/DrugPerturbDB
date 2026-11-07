from pathlib import Path
import os,json,hashlib,time
import numpy as np,pandas as pd
R=Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2');F=R.parent.parent/'FULL_DATASET_FIGURES_V1';assert os.environ.get('TAHOE_REPRO_RUN_ID')
out=[]
for n in ['L1000_L3','DRUGseq','CIGS','sciPlex3','MIXseq']:
 p=F/'data'/f'{n}_META.parquet';m=pd.read_parquet(p);cm=pd.read_parquet(F/'data'/f'{n}_CONDITION_META.parquet')
 m=m[m.normalization.ne('logCPM100')].copy();cm=cm[cm.normalization.ne('logCPM100')].copy();keep=~m.is_control.astype(bool)
 if 'valid_profile' in m:keep&=m.valid_profile.astype(bool)
 good=m[keep].copy()
 for k in ['dose','time']:good[k]=pd.to_numeric(good[k],errors='coerce').fillna(-1).map(lambda v:float(format(float(v),'.6g')))
 keys=['normalization','drug','cell','dose','time']
 # 'panel' is a derived stratum in aggregated outputs; compare physical
 # condition keys common to source and aggregate, without a fabricated label.
 if 'panel' in good and 'panel' in cm:keys=['panel']+keys
 expected=set(map(tuple,good[keys].to_numpy()));actual=set(map(tuple,cm[keys].to_numpy()))
 assert expected==actual,(n,'missing',len(expected-actual),'extra',len(actual-expected))
 rolecols=[c for c in m if any(x in c.lower() for x in ['role','outer_fold','owner_fold'])]
 prep=F/'code'/{'L1000_L3':'prep_l1000.py','DRUGseq':'prep_drugseq.py','CIGS':'prep_cigs.py','sciPlex3':'prep_sciplex.py','MIXseq':'prep_mix.py'}[n]
 text=prep.read_text();assert 'DRUG_ROLE_OWNER' not in text
 out.append(dict(dataset=n,source_profiles=len(m),control_profiles=int(m.is_control.sum()),QC_treatment_profiles=len(good),eligible_drug_IDs=good.drug.nunique(),prepared_drug_IDs=cm.drug.nunique(),expected_conditions=len(expected),retained_conditions=len(actual),omitted_eligible_conditions=0,extra_conditions=0,model_role_columns=';'.join(rolecols),preparation_code=str(prep),code_sha256=hashlib.sha256(prep.read_bytes()).hexdigest(),metadata_path=str(p),metadata_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='All eligible profiles in retained public source releases; no Tahoe overlap or model-role gate',status='PASS'))
pd.DataFrame(out).to_csv(R/'results/EXTERNAL_SCOPE_AUDIT.csv',index=False)
(R/'receipts/EXTERNAL_SCOPE_AUDIT.json').write_text(json.dumps(dict(status='PASS',datasets=out,model_response_reads=0,new_expression_reads=0),indent=2))
print(pd.DataFrame(out).to_string(index=False))

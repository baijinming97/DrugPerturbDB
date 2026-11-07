from ff_common import *
paths=[A/'SCIPLEX_DRUG_IDENTITY.csv',O/'4MAN/CIGS_BETAW_PILOT_V1/inputs/CIGS_STRICT_MAP.csv',O/'4MAN/DRUGSEQ_BETAW_PILOT_V1/inputs/STRICT_MAP.csv',O/'4MAN/TAHOE_GAMMA_PARALLEL_V1/ws3_multicell_external/v2/DRUG_IDENTITY_AUDIT.csv',R/'raw/compoundinfo_beta.txt',R/'raw/cellinfo_beta.txt']
out={}
for p in paths:
 t=pd.read_csv(p,sep='\t' if p.suffix=='.txt' else ',');out[str(p)]={'shape':t.shape,'columns':list(t),'head':t.head(3).astype(str).to_dict('records')}
out['native_receipts']={p.stem:json.loads(p.read_text()) for p in (R/'receipts').glob('*PREPARED.json')}
dump(R/'inventory/IDENTITY_SCHEMAS.json',out)

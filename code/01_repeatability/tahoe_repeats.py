from pathlib import Path
import os, sys, json, time, hashlib, threading, itertools, resource
assert os.environ.get('TAHOE_REPRO_RUN_ID')
O = Path('@WORKSPACE@')
R = O / '4MAN/TAHOE_DESCRIPTIVE_FULL_V2'
C = O / 'baicai/canonical_perturbation_data_v1/tahoe'
F = O / 'FULL_DATASET_FIGURES_V1'
N = O / 'REPEATABILITY_NATIVE_GENES_V1'
T = time.time()
import numpy as np, pandas as pd, h5py, fsspec, pyarrow.parquet as pq, pyarrow.compute as pc
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import pearsonr

def dump(n, v):
    (R / n).write_text(json.dumps(v, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
q = pd.read_parquet(R / 'metadata/Tahoe_REPLICATE_ELIGIBLE.parquet').sort_values(['row_groups', 'profile_row']).reset_index(drop=True)
q['row'] = np.arange(len(q))
g = pd.read_parquet(C / 'data/gene_axis.parquet')
genes = g.feature.astype(str).to_numpy()
ng = len(genes)
gt = pd.read_csv(O / '4MAN/minimum_restore_v1/retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data/GENE_AXIS.tsv', sep='\t', dtype=str)
p978 = gt.tahoe_gene_position.astype(int).to_numpy()
axis869 = json.loads((F / 'COMMON_GENES.json').read_text())['axis']
p869 = gt.set_index('l1000_entrez_gene_id').loc[axis869, 'tahoe_gene_position'].astype(int).to_numpy()
post = np.full((len(q), ng), np.nan, 'f4')
delta = np.full_like(post, np.nan)
seen = np.zeros(len(q), bool)
key = lambda drug, cell, dose, plate: (str(drug), str(cell), float(dose), int(plate))
lookup = {key(z.drug, z.cell_line, z.dose, z.plate): z.Index for z in q.itertuples()}
# Read only expression reconstructed from the registered public PDEX file.
index_by_profile = {int(p): i for i, p in enumerate(q.profile_row)}
parts = sorted((C / 'tmp').glob('pdex_values_paper4_*-of-04.parquet'))
assert len(parts) == 4, 'Complete the four canonical extraction stages first'
for part in parts:
    meta = pd.read_parquet(part)
    selected = meta[meta.profile_row.isin(index_by_profile)]
    with h5py.File(part.with_suffix('.h5'), 'r') as h:
        controls = h['controls'][:]
        for lo in range(0, len(selected), 128):
            block = selected.iloc[lo:lo+128]
            rows = block.temp_value_row.to_numpy(int)
            order = np.argsort(rows)
            values = h['treated'][rows[order]][np.argsort(order)]
            dest = np.array([index_by_profile[int(p)] for p in block.profile_row])
            assert not seen[dest].any()
            post[dest] = values
            delta[dest] = values-controls[block.temp_control_row.to_numpy(int)]
            seen[dest] = True
assert seen.all() and np.isfinite(post).all() and np.isfinite(delta).all()
downloaded = 0
dump('receipts/REPEAT_SOURCE.json', dict(status='PASS', profiles=len(q),
    source='registered PDEX reconstruction', historical_response_arrays_used=False))
with h5py.File(R / 'data/Tahoe_REPLICATE_978.h5', 'w') as h:
    h.create_dataset('post', data=post[:, p978], compression='gzip', compression_opts=1)
    h.create_dataset('delta', data=delta[:, p978], compression='gzip', compression_opts=1)
    h.create_dataset('entrez', data=gt.l1000_entrez_gene_id.to_numpy(object), dtype=h5py.string_dtype())
chem = pd.read_parquet(C / 'data/condition_manifest.parquet').drop_duplicates('drug').set_index('drug').chemistry_component.to_dict()
q['group'] = q.drug.map(chem)
q['cell'] = q.cell_line
q['time'] = 24.0
q['normalization'] = 'author_linear_normalized'
q['physical_unit'] = 'plate' + q.plate.astype(str)
q['control_key'] = q.cell_line + '|plate' + q.plate.astype(str)
q.to_parquet(R / 'metadata/Tahoe_REPLICATE_META.parquet', index=False)

def corr(a, b):
    a = a.astype(float)
    b = b.astype(float)
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else np.nan
rows = []
for (drug, cell, dose), z in q.groupby(['drug', 'cell', 'dose']):
    for i, j in itertools.combinations(z.index, 2):
        assert q.loc[i, 'plate'] != q.loc[j, 'plate']
        for mode, a in [('post', post), ('delta', delta)]:
            for axis, pos in [('native', None), ('common', p869)]:
                x = a[i] if pos is None else a[i, pos]
                y = a[j] if pos is None else a[j, pos]
                rows.append(dict(dataset='Tahoe', mode=mode, axis=axis, normalization='author_linear_normalized', drug=drug, group=chem[drug], cell=cell, dose=dose, time=24, i=i, j=j, plate_a=int(q.loc[i, 'plate']), plate_b=int(q.loc[j, 'plate']), pcc=corr(x, y), genes=len(x), pairs=1))
df = pd.DataFrame(rows)
df.to_csv(R / 'results/Tahoe_REPEAT_CONDITIONS.csv.gz', index=False)
summ = []
for ds, ss in [('all_doses', df)] + [(str(d), df[df.dose == d]) for d in sorted(df.dose.unique())]:
    for (mode, axis, norm), z in ss.groupby(['mode', 'axis', 'normalization']):
        dd = z.groupby(['group', 'drug', 'cell', 'dose']).pcc.mean().groupby(['group', 'drug']).mean().reset_index()
        v = dd.groupby('group').pcc.mean().dropna().to_numpy()
        rng = np.random.default_rng(20261029)
        bs = v[rng.integers(len(v), size=(2000, len(v)))].mean(1)
        summ.append(dict(dataset='Tahoe', mode=mode, axis=axis, normalization=norm, dose_scope=ds, pcc=v.mean(), low=np.quantile(bs, 0.025), high=np.quantile(bs, 0.975), drugs=dd.drug.nunique(), groups=len(v), conditions=len(z.groupby(['drug', 'cell', 'dose'])), pairs=len(z), genes=z.genes.iloc[0], scope='All eligible independent physical-plate pairs; all model roles; >=50 full-QC cells per plate; author-qualified 47-cell panel'))
pd.DataFrame(summ).to_csv(R / 'results/Tahoe_REPEAT_SUMMARY.csv', index=False)
i, j = (int(df.i.iloc[0]), int(df.j.iloc[0]))
ve = abs(corr(delta[i, p869], delta[j, p869]) - pearsonr(delta[i, p869].astype(float), delta[j, p869].astype(float)).statistic)
assert ve < 1e-10
dump('receipts/REPEAT_FULL.json', dict(status='PASS', seconds=time.time() - T, downloaded_bytes=downloaded, n_profiles=len(q), n_drugs=q.drug.nunique(), pairs=len(df) // 4, independent_pearson_error=ve, allocated_core_hours=(time.time() - T) * int(os.environ['TAHOE_REPRO_CPUS']) / 3600, MaxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, model_training=0, model_evaluation=0, descriptive_all_roles=True))
print(pd.DataFrame(summ).to_string(index=False), flush=True)

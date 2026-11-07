"""Prepare all QC-eligible Tahoe conditions for descriptive analysis."""
from pathlib import Path
import os, sys, json, time, hashlib, resource
assert os.environ.get('TAHOE_REPRO_RUN_ID') and (not os.environ.get('CUDA_VISIBLE_DEVICES'))
O = Path('@WORKSPACE@')
R = O / '4MAN/TAHOE_DESCRIPTIVE_FULL_V2'
F = O / 'FULL_DATASET_FIGURES_V1'
C = O / 'baicai/canonical_perturbation_data_v1/tahoe'
start = time.time()
import numpy as np, pandas as pd, h5py
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def dump(n, x):
    (R / n).write_text(json.dumps(x, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
src = (F / 'code/ff_statistics_global.py').read_text()
exec(src.replace('from ff_common import *', ''), globals())
m = pd.read_parquet(C / 'data/condition_manifest.parquet')
mp = pd.read_parquet(C / 'data/condition_control_map.parquet')
genes = pd.read_parquet(C / 'data/gene_axis.parquet').feature.astype(str).to_numpy()
owners = json.loads((O / '4MAN/CROSS_DATASET_ATLAS_V1/DRUG_ROLE_OWNER.json').read_text())
m['legacy_owner_role'] = m.drug.map(lambda d: owners.get(d, {}).get('role', 'OUTSIDE_OLD_PANEL'))
assert len(m) == 52886 and m.drug.nunique() == 379 and m.condition_row.is_unique
m = m.sort_values('condition_row').reset_index(drop=True)
m['row'] = np.arange(len(m))
m['cell'] = m.cell_line
m['time'] = 24.0
m['panel'] = 'Tahoe'
m['normalization'] = 'author_linear_normalized'
m['group'] = m.chemistry_component
m.to_parquet(R / 'data/Tahoe_CONDITION_META.parquet', index=False)
gmap = pd.read_csv(O / '4MAN/minimum_restore_v1/retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data/GENE_AXIS.tsv', sep='\t', dtype=str)
axis869 = json.loads((F / 'COMMON_GENES.json').read_text())['axis']
pos869 = gmap.set_index('l1000_entrez_gene_id').loc[axis869, 'tahoe_gene_position'].astype(int).to_numpy()
pos978 = gmap.tahoe_gene_position.astype(int).to_numpy()
assert len(pos978) == 978
loc = {int(r): i for i, r in enumerate(m.condition_row)}
with h5py.File(C / 'data/controls.h5') as h:
    ctrl = h['controls'][:]
W = sparse.csr_matrix((mp.control_weight.to_numpy(), ([loc[int(v)] for v in mp.condition_row], mp.control_row.to_numpy(int))), shape=(len(m), len(ctrl)))
assert np.allclose(np.asarray(W.sum(1)).ravel(), 1)
Y = np.empty((len(m), len(genes)), np.float32)
seen = np.zeros(len(m), bool)
io = []
for p in sorted((C / 'data').glob('treated_*-of-16.h5')):
    tick = time.time()
    with h5py.File(p) as h:
        rr = h['condition_row'][:]
        for lo in range(0, len(rr), 128):
            dest = np.array([loc[int(v)] for v in rr[lo:lo + 128]])
            x = h['treated'][lo:lo + 128]
            assert np.isfinite(x).all() and (not seen[dest].any())
            Y[dest] = x - W[dest] @ ctrl
            seen[dest] = True
    io.append(dict(path=str(p), bytes=p.stat().st_size, seconds=time.time() - tick))
    print('READ', p.name, round(time.time() - tick, 2), flush=True)
assert seen.all()
del ctrl
with h5py.File(R / 'data/Tahoe_COMMON_CONDITIONS.h5', 'w') as h:
    h.create_dataset('delta', data=Y[:, pos869], compression='gzip', compression_opts=1)
    h.create_dataset('entrez', data=np.asarray(axis869, object), dtype=h5py.string_dtype())
with h5py.File(R / 'data/Tahoe_978_CONDITIONS.h5', 'w') as h:
    h.create_dataset('delta', data=Y[:, pos978], compression='gzip', compression_opts=1)
    h.create_dataset('entrez', data=gmap.l1000_entrez_gene_id.to_numpy(object), dtype=h5py.string_dtype())
np.savez_compressed(R / 'data/Tahoe_NATIVE_AXES.npz', gene=genes, common_positions=pos869)
summary = []
strata = []
checks = []
for label, idx in [('ALL_ELIGIBLE', np.arange(len(m))), ('LEGACY_SUBSET', np.flatnonzero(m.legacy_owner_role.eq('TRAIN')))]:
    mm = m.iloc[idx].copy()
    d, _ = pd.factorize(mm.drug)
    mm['_weight'] = 1 / np.bincount(d)[d]
    mm['_weight'] /= mm['_weight'].sum()
    for axis, pos in [('native', np.arange(len(genes))), ('common', pos869)]:
        total = np.zeros(4)
        ident = conn = 0.0
        closure = 0.0
        for dose, sm in mm.groupby('dose', sort=True):
            mass = sm._weight.sum()
            e = np.zeros(4)
            for lo in range(0, len(pos), 256):
                xx = Y[np.ix_(sm.row.to_numpy(), pos[lo:lo + 256])]
                ec, co, info = decompose(xx, sm)
                e += ec
                closure = max(closure, info['closure'])
            total += mass * e
            ident += mass * info['interaction_identifiable']
            conn += mass * (info['connected_components'] == 1)
            strata.append(dict(scope=label, axis=axis, dose=dose, mass=mass, **info, energy=e.tolist()))
            print('COMPOSED', label, axis, dose, round(time.time() - start, 1), flush=True)
        rr = dict(dataset='Tahoe', scope=label, normalization='author_linear_normalized', axis=axis, drugs=mm.drug.nunique(), cells=mm.cell.nunique(), conditions=len(mm), genes=len(pos), strata=3, interaction_identifiable_weight=ident, connected_design_weight=conn, max_closure=closure)
        for j, k in enumerate(['mean', 'cell', 'drug', 'interaction_residual']):
            rr[k + '_energy'] = total[j]
            rr[k + '_fraction'] = total[j] / total.sum()
        summary.append(rr)
sm = m[m.dose.eq(5)].copy()
ds = sorted(sm.drug.unique())[:8]
cs = sorted(sm.cell.unique())[:6]
sm = sm[sm.drug.isin(ds) & sm.cell.isin(cs)].copy()
xx = Y[np.ix_(sm.row, pos869[:20])].astype(float)
ec, co, info = decompose(xx, sm)
d, c, w, _, _, _ = weighted_design(sm)
D = np.c_[np.ones(len(sm)), np.eye(len(set(c)))[c], np.eye(len(set(d)))[d]]
coef = np.linalg.lstsq(D * np.sqrt(w[:, None]), xx * np.sqrt(w[:, None]), rcond=None)[0]
er = float(np.max(abs(xx - D @ coef - co['residual'])))
assert er < 1e-08
pd.DataFrame(summary).to_csv(R / 'results/Tahoe_FULL_COMPOSITION.csv', index=False)
dump('results/Tahoe_COMPOSITION_STRATA.json', strata)
pd.DataFrame(io).to_csv(R / 'results/IO_TIMING.csv', index=False)
dump('receipts/PREPARE_FULL.json', dict(status='PASS', job=os.environ['TAHOE_REPRO_RUN_ID'], seconds=time.time() - start, allocated_core_hours=(time.time() - start) * int(os.environ['TAHOE_REPRO_CPUS']) / 3600, MaxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, conditions=len(m), drugs=m.drug.nunique(), cells=m.cell.nunique(), scope_counts=m.groupby('legacy_owner_role').agg(conditions=('drug', 'size'), drugs=('drug', 'nunique')).reset_index().to_dict('records'), independent_residual_max_error=er, descriptive_all_roles=True, model_training=0, model_evaluation=0))
print(json.dumps(summary), flush=True)

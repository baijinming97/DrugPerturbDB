from pathlib import Path
import os, sys, json, hashlib, time, itertools
import numpy as np, pandas as pd, h5py
from scipy import sparse
from scipy.stats import pearsonr
O = Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2/pairs')
F = Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
assert os.getenv('TAHOE_REPRO_RUN_ID')
start = time.time()
assert json.loads((O / 'ANALYSIS_RECEIPT.json').read_text())['status'] == 'ANALYZED_PENDING_INDEPENDENT_VERIFICATION'
source = (F / 'code/ff_statistics_global.py').read_text()
exec(source.replace('from ff_common import *', ''), globals())
protocol = dict(identity_sha256=hashlib.sha256((O / 'IDENTITY_MAP_FINAL.json').read_bytes()).hexdigest(), selection='All prelocked STRICT_NEAREST panels, one nearest dose pair per drug, every common exact time; >=3 drugs', axis=869, display='All 24h panels; composition differences only on multi-cell identifiable panels', bootstrap=2000, intervals_conditional_on_fitted_decomposition=True, selected_without_response_values=True)
(O / 'COMPONENT_PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
sel = [z for z in json.loads((O / 'LOCKED_SELECTIONS.json').read_text()) if z['mode'] == 'STRICT_NEAREST']
identity = json.loads((O / 'IDENTITY_MAP_FINAL.json').read_text())
vectors = {}
for n in identity['drug']:
    ids = sorted({i for z in sel for i in (z['rows_a'] if z['a'] == n else z['rows_b'] if z['b'] == n else [])})
    k = pd.read_parquet(O / 'metadata' / f'{n}_VECTOR_KEYS.parquet').iloc[ids]
    raw = sorted({int(i) for s in k.source_rows for i in s.split(',')})
    loc = {v: i for i, v in enumerate(raw)}
    with h5py.File((O.parent if n == 'Tahoe' else F) / 'data' / f'{n}_COMMON_CONDITIONS.h5', 'r') as h:
        x = h['delta'][raw].astype(float)
    vectors[n] = {int(r.value_row): x[[loc[int(i)] for i in r.source_rows.split(',')]].mean(0) for r in k.itertuples()}
cm = pd.read_parquet(F.parent / 'baicai/canonical_perturbation_data_v1/tahoe/data/condition_manifest.parquet')
groups = cm.drop_duplicates('drug').set_index('drug').chemistry_component.to_dict()
clusters = {v: groups[d] for d, v in identity['drug']['Tahoe'].items()}
out = []
shares = []
supports = []
skips = []
checks = []
key = lambda z: (z['a'], z['b'], z['time'])
for (a, b, tm), zz in itertools.groupby(sorted(sel, key=key), key=key):
    pairs = []
    xx = []
    yy = []
    for z in zz:
        assert z['cells_a'] == z['cells_b']
        for c, ra, rb in zip(z['cells_a'], z['rows_a'], z['rows_b']):
            pairs.append(dict(drug=z['drug'], cell=c, dose_a=z['dose_a'], dose_b=z['dose_b']))
            xx.append(vectors[a][ra])
            yy.append(vectors[b][rb])
    m = pd.DataFrame(pairs)
    nd = m.drug.nunique()
    nc = m.cell.nunique()
    pid = f'{a}__{b}__{tm:g}h'
    if nd < 3:
        skips.append(dict(panel_id=pid, drugs=nd, cells=nc, reason='fewer_than_3_drugs'))
        continue
    x = np.stack(xx)
    y = np.stack(yy)
    ea, ca, ia = decompose(x, m)
    eb, cb, ib = decompose(y, m)
    supports.append(dict(panel_id=pid, a=a, b=b, time=tm, genes=869, **ia))
    for name, z, e, co in [(a, x, ea, ca), (b, y, eb, cb)]:
        d, c, w, D, Cc, inf = weighted_design(m)
        design = np.c_[np.ones(len(m)), np.eye(nd)[d], np.eye(nc)[c]]
        sw = np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(design * sw, z * sw, rcond=None)[0]
        res = z - design @ coef
        err = float(np.max(abs(res - co['residual'])))
        assert err < 1e-08, (pid, name, err)
        total = float(np.sum(w[:, None] * z * z))
        closure = abs(total - float(e.sum()))
        assert closure < 1e-08, (pid, closure)
        checks.append(dict(panel_id=pid, dataset=name, independent_residual_max_error=err, energy_closure=closure))
        for j, component in enumerate(['mean', 'cell', 'drug', 'interaction_residual']):
            shares.append(dict(panel_id=pid, a=a, b=b, dataset=name, time=tm, component=component, fraction=e[j] / e.sum(), energy=e[j], drugs=nd, cells=nc, observed=len(m), identifiable=ia['interaction_identifiable'], connected_components=ia['connected_components']))
    for component in ['complete', 'mean', 'cell', 'drug', 'interaction_residual']:
        if component in ['cell', 'interaction_residual'] and (not ia['interaction_identifiable']) or (component in ['cell', 'drug'] and ia['connected_components'] != 1):
            out.append(dict(panel_id=pid, a=a, b=b, time=tm, component=component, pcc=np.nan, low=np.nan, high=np.nan, drugs=nd, cells=nc, status='not identifiable'))
            continue
        ck = 'residual' if component == 'interaction_residual' else component
        u, v = (x, y) if component == 'complete' else (ca['mu'][None], cb['mu'][None]) if component == 'mean' else (ca[ck], cb[ck])
        pv = pearson_rows(u, v)
        for i in range(min(3, len(pv))):
            if np.isfinite(pv[i]):
                assert abs(float(pearsonr(u[i], v[i]).statistic) - pv[i]) < 1e-12
        if component == 'mean':
            point = float(pv[0])
            lo = hi = np.nan
        else:
            drug = pd.DataFrame(dict(drug=m.drug, pcc=pv)).groupby('drug').pcc.mean()
            cl = [clusters.get(d, d) for d in drug.index] if 'Tahoe' in [a, b] else None
            point, lo, hi = boot_mean(drug.to_numpy(), cl, seed=20261030)
            if component == 'cell' or (len(set(cl)) if cl else len(drug)) < 3:
                lo = hi = np.nan
        out.append(dict(panel_id=pid, a=a, b=b, time=tm, component=component, pcc=point, low=lo, high=hi, drugs=nd, cells=nc, status='single_cell_mixed_drug_effect' if component == 'drug' and nc == 1 else 'descriptive', genes=869))
for fn, rows in [('INTERSECTION_CONSISTENCY', out), ('INTERSECTION_COMPOSITION', shares), ('INTERSECTION_SUPPORT', supports), ('INTERSECTION_SKIPPED', skips), ('COMPONENT_REPLAY_CHECKS', checks)]:
    pd.DataFrame(rows).to_csv(O / 'results' / f'{fn}.csv', index=False)
receipt = dict(status='PASS', job=os.getenv('TAHOE_REPRO_RUN_ID'), seconds=time.time() - start, panels=len(supports), independent_design_replays=len(checks), max_residual_error=max((z['independent_residual_max_error'] for z in checks)), max_energy_closure=max((z['energy_closure'] for z in checks)), model_TEST_reads=0, descriptive_Tahoe_all_roles=True, training=0)
(O / 'COMPONENT_REPLAY_VERIFICATION.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)

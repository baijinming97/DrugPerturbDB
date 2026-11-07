"""Independent enumeration and numerical replay. Does not import the analysis module."""
from pathlib import Path
import os, sys, json, math, itertools, hashlib, collections, time
import pandas as pd, numpy as np, h5py
from scipy.stats import pearsonr
O = Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2/pairs')
F = Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
assert os.getenv('TAHOE_REPRO_RUN_ID')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
start = time.time()
pr = json.loads((O / 'PROTOCOL.json').read_text())
sel = json.loads((O / 'LOCKED_SELECTIONS.json').read_text())
identity = json.loads((O / 'IDENTITY_MAP_FINAL.json').read_text())
names = list(identity['drug'])
assert sha(O / 'LOCKED_SELECTIONS.json') == pr['selection_sha256'] and sha(O / 'IDENTITY_MAP_FINAL.json') == pr['identity_sha256']
kt = {n: pd.read_parquet(O / 'metadata' / f'{n}_VECTOR_KEYS.parquet') for n in names}
actual = collections.defaultdict(set)
used = collections.defaultdict(set)
for z in sel:
    if z['mode'] not in ['STRICT_ALL_NEAR', 'MARGINAL_ALL_NEAR']:
        continue
    k = (z['a'], z['b'], z['mode'])
    tag = (z['drug'], z['time'], z['dose_a'], z['dose_b'], z['cells_a'][0] if z['mode'] == 'STRICT_ALL_NEAR' else 'marginal')
    assert tag not in actual[k], ('duplicate selection', k, tag)
    actual[k].add(tag)
    if z['time'] == 24:
        for n, p, rs in [(z['a'], z['b'], z['rows_a']), (z['b'], z['a'], z['rows_b'])]:
            for sr in kt[n].iloc[rs].source_rows:
                used[n, p, z['mode']].update((int(i) for i in sr.split(',')))
checks = []
for a, b in itertools.combinations(names, 2):
    ta = kt[a]
    tb = kt[b]
    da = ta[['drug', 'time', 'dose']].drop_duplicates()
    db = tb[['drug', 'time', 'dose']].drop_duplicates()
    j = da.merge(db, on=['drug', 'time'], suffixes=('_a', '_b'))
    j = j[np.maximum(j.dose_a, j.dose_b) / np.minimum(j.dose_a, j.dose_b) <= 2 + 1e-08]
    expected = {(r.drug, float(r.time), float(r.dose_a), float(r.dose_b), 'marginal') for r in j.itertuples()}
    assert expected == actual[a, b, 'MARGINAL_ALL_NEAR'], ('marginal omitted/extra', a, b, len(expected), len(actual[a, b, 'MARGINAL_ALL_NEAR']))
    ca = ta[ta.cvcl.ne('')][['drug', 'time', 'dose', 'cell']]
    cb = tb[tb.cvcl.ne('')][['drug', 'time', 'dose', 'cell']]
    j = ca.merge(cb, on=['drug', 'time', 'cell'], suffixes=('_a', '_b'))
    j = j[np.maximum(j.dose_a, j.dose_b) / np.minimum(j.dose_a, j.dose_b) <= 2 + 1e-08]
    expected = {(r.drug, float(r.time), float(r.dose_a), float(r.dose_b), r.cell) for r in j.itertuples()}
    assert expected == actual[a, b, 'STRICT_ALL_NEAR'], ('strict omitted/extra', a, b)
    checks.append(dict(a=a, b=b, strict_all_times=len(expected), marginal_all_times=len(actual[a, b, 'MARGINAL_ALL_NEAR']), omitted_eligible_pairs=0, extra_pairs=0))
flow = pd.read_csv(O / 'results/PAIR_EXCLUSION_FLOW.csv')
fc = []
for k, g in flow.groupby(['dataset', 'partner', 'mode']):
    assert g.conditions.sum() == g.total_conditions.iloc[0]
    eligible = int(g.loc[g.reason == 'eligible_all_near', 'conditions'].sum())
    assert eligible == len(used[k]), ('eligible raw-condition coverage', k, eligible, len(used[k]))
    fc.append(dict(dataset=k[0], partner=k[1], mode=k[2], eligible_source_conditions=eligible, actually_used=len(used[k]), omitted=0))
pd.DataFrame(fc).to_csv(O / 'results/EXHAUSTIVE_COVERAGE_VERIFICATION.csv', index=False)

def mean_finite(v):
    v = [float(x) for x in v if np.isfinite(x)]
    return math.fsum(v) / len(v) if v else float('nan')

def recompute(g, col):
    dc = collections.defaultdict(list)
    for r in g.to_dict('records'):
        dc[r['drug'], r['cluster'], r['cell']].append(r[col])
    drug = collections.defaultdict(list)
    for (d, c, cell), v in dc.items():
        drug[d, c].append(mean_finite(v))
    cluster = collections.defaultdict(list)
    for (d, c), v in drug.items():
        cluster[c].append(mean_finite(v))
    v = np.array([mean_finite(cluster[c]) for c in sorted(cluster)], float)
    v = v[np.isfinite(v)]
    pt = mean_finite(v)
    if len(v) < 3:
        return (pt, float('nan'), float('nan'))
    rng = np.random.default_rng(20261030)
    vals = []
    for _ in range(2000):
        vals.append(float(np.mean(v[rng.integers(0, len(v), len(v))])))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return (pt, float(lo), float(hi))
raw = pd.read_csv(O / 'results/CONDITION_RESULTS.csv.gz')
summary = pd.read_csv(O / 'results/RESULTS.csv')
errors = []
for r in summary.itertuples():
    g = raw[(raw.a == r.a) & (raw.b == r.b) & (raw['mode'] == r.mode) & (raw.time == r.time) & (raw.coordinate == r.coordinate)]
    v = recompute(g, 'pcc')
    for a, b in zip(v, [r.estimate, r.low, r.high]):
        if np.isnan(a) and np.isnan(b):
            continue
        errors.append(abs(a - b))
        assert abs(a - b) < 1e-10, (r.a, r.b, r.mode, r.time, a, b)
ret = pd.read_csv(O / 'results/RETRIEVAL_CONDITIONS.csv.gz')
rs = pd.read_csv(O / 'results/RETRIEVAL_RESULTS.csv')
for r in rs.itertuples():
    g = ret[(ret.a == r.a) & (ret.b == r.b) & (ret['mode'] == r.mode) & (ret.time == r.time) & (ret.direction == r.direction)]
    v = recompute(g, r.metric)
    for a, b in zip(v, [r.estimate, r.low, r.high]):
        if np.isnan(a) and np.isnan(b):
            continue
        errors.append(abs(a - b))
        assert abs(a - b) < 1e-10
examples = json.loads((O / 'results/REPLAY_EXAMPLES.json').read_text())
arrays = np.load(O / 'results/REPLAY_EXAMPLES.npz')
replay = []
for i, e in enumerate(examples):
    z = e['selection']
    v = []
    for n, rs in [(z['a'], z['rows_a']), (z['b'], z['rows_b'])]:
        if z['mode'].startswith('STRICT'):
            rs = rs[:1]
        vv = []
        with h5py.File((O.parent if n == 'Tahoe' else F) / 'data' / f'{n}_COMMON_CONDITIONS.h5', 'r') as h:
            for row in kt[n].iloc[rs].itertuples():
                ids = sorted((int(x) for x in row.source_rows.split(',')))
                vv.append(np.asarray(h['delta'][ids], np.float64).mean(0))
        v.append(np.stack(vv).mean(0))
    assert np.max(np.abs(v[0] - arrays['x'][i])) < 1e-12 and np.max(np.abs(v[1] - arrays['y'][i])) < 1e-12
    r = float(pearsonr(*v).statistic)
    assert abs(r - e['pcc']) < 1e-12
    replay.append(dict(a=z['a'], b=z['b'], mode=z['mode'], time=z['time'], drug=z['drug'], pcc=r, error=abs(r - e['pcc'])))
pd.DataFrame(replay).to_csv(O / 'results/INDEPENDENT_RAW_REPLAY.csv', index=False)
tm = pd.read_parquet(O / 'metadata/Tahoe_ANNOTATED_CONDITIONS.parquet')
assert len(tm) == 52886 and tm.source_drug.nunique() == 379
prior = O.parent.parent / 'PREPRINT_PAIR_EXHAUSTIVENESS_V1_20261030'
prev = pd.read_csv(prior / 'results/RESULTS.csv')
now = summary
keys = ['a', 'b', 'mode', 'time', 'coordinate', 'metric']
j = prev[prev.a.ne('Tahoe') & prev.b.ne('Tahoe')].merge(now, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
externalerr = max((float((j[k + '_old'] - j[k + '_new']).abs().max()) for k in ['estimate', 'low', 'high']))
assert externalerr < 1e-10
out = dict(status='PASS', verified_at_unix=time.time(), job=os.getenv('TAHOE_REPRO_RUN_ID'), elapsed_seconds=time.time() - start, all_15_pairs_checked=checks, selection_enumeration='independent metadata merge equals full locked selection set', omitted_eligible_conditions=0, source_flow_closes=True, summary_rows_recomputed=len(summary) + len(rs), max_summary_or_CI_error=max(errors, default=0), raw_examples_recomputed=len(replay), raw_PCC_max_error=max((x['error'] for x in replay)), Tahoe_drugs=tm.source_drug.nunique(), Tahoe_conditions=len(tm), model_TEST_response_reads=0, descriptive_Tahoe_all_roles=True, training=0, GPU=0, external_only_result_max_change=externalerr, protocol_sha256=sha(O / 'PROTOCOL.json'))
(O / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out), flush=True)

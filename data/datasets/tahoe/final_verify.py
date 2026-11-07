from pathlib import Path
import os, sys, json, hashlib, itertools, time
import numpy as np, pandas as pd
R = Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2')
import h5py
assert os.environ.get('TAHOE_REPRO_RUN_ID')
T = time.time()
for p in ['receipts/PREPARE_FULL.json', 'receipts/REPEAT_FULL.json', 'receipts/EXTERNAL_SCOPE_AUDIT.json', 'pairs/INDEPENDENT_VERIFICATION.json']:
    assert json.loads((R / p).read_text())['status'] == 'PASS', p
assert json.loads((R / 'exp1/EXP1_RECEIPT.json').read_text())['status'] == 'COMPLETE'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
rep = pd.read_csv(R / 'results/Tahoe_REPEAT_CONDITIONS.csv.gz')
assert np.isfinite(rep.pcc).all(), 'Undefined repeat PCC; shared validity filter required'
for mode, z in rep.groupby('mode'):
    assert set(map(tuple, z[z.axis.eq('native')][['i', 'j']].to_numpy())) == set(map(tuple, z[z.axis.eq('common')][['i', 'j']].to_numpy()))
with h5py.File(R / 'data/Tahoe_COMMON_CONDITIONS.h5', 'r') as h:
    print('COMMON_KEYS', list(h), flush=True)
    axis = h['entrez'].asstr()[:] if 'entrez' in h else h['gene'].asstr()[:]
with h5py.File(R / 'data/Tahoe_REPLICATE_978.h5', 'r') as h:
    pos = pd.Index(h['entrez'].asstr()[:]).get_indexer(axis)
    assert min(pos) >= 0
    repeaterr = 0.0
    for mode, z in rep[rep.axis.eq('common')].groupby('mode'):
        v = h[mode][:].astype(float)[:, pos]
        v -= v.mean(1, keepdims=True)
        v /= np.linalg.norm(v, axis=1)[:, None]
        pcc = np.einsum('ij,ij->i', v[z.i.to_numpy(int)], v[z.j.to_numpy(int)])
        repeaterr = max(repeaterr, float(np.max(abs(pcc - z.pcc))))
assert repeaterr < 1e-10
raw = np.load(R / 'exp1/verification_pair.npz')
A = raw['A']
B = raw['B']
pa = raw['AA_pairs']
pb = raw['BB_pairs']
aa = np.mean([np.corrcoef(A[i], A[j])[0, 1] for i, j in pa])
bb = np.mean([np.corrcoef(B[i], B[j])[0, 1] for i, j in pb])
ab = np.mean([np.corrcoef(a, b)[0, 1] for a in A for b in B])
rawerr = max(abs(aa - raw['claimed_rAA']), abs(bb - raw['claimed_rBB']), abs(ab - raw['claimed_rAB']))
assert rawerr < 1e-10
for side, pairs in [('A', pa), ('B', pb)]:
    units = raw[side + '_unit']
    sets = [set(json.loads(s)) for s in raw[side + '_control_sets']]
    for i, j in pairs:
        assert units[i] != units[j] and (not sets[i].intersection(sets[j]))
cc = pd.read_csv(R / 'exp1/CONDITION_RESULTS.csv')
rr = pd.read_csv(R / 'exp1/MATCHED_RELIABILITY.csv')
err = 0.0
for (a, b), z in cc.groupby(['A', 'B']):
    metrics = ['rAA', 'rBB', 'rAB', 'gap_A', 'gap_B', 'gap_mean']
    dc = z.groupby(['group', 'drug', 'cell'])[metrics].mean().groupby(['group', 'drug']).mean().groupby('group').mean()
    arr = dc.to_numpy()
    idx = np.random.default_rng(20261103).integers(0, len(arr), (2000, len(arr)))
    boot = arr[idx].mean(1)
    for j, k in enumerate(metrics):
        q = rr[rr.A.eq(a) & rr.B.eq(b) & rr.metric.eq(k)].iloc[0]
        got = [arr[:, j].mean(), *(np.quantile(boot[:, j], [0.025, 0.975]) if len(arr) >= 3 else [np.nan, np.nan])]
        expected = [q.estimate, q.ci_low, q.ci_high]
        assert np.array_equal(np.isnan(got), np.isnan(expected))
        err = max(err, float(np.nanmax(abs(np.asarray(got) - expected))))
assert err < 1e-10
rec = dict(status='PASS', job=os.environ['TAHOE_REPRO_RUN_ID'], all_native_common_repeat_support_equal=True, undefined_repeat_PCC=0, common_axis_all_pair_replay_max_error=repeaterr, S1B_independent_raw_pair_max_error=float(rawerr), S1B_summary_and_2000bootstrap_max_error=err, independent_control_pools=True, seconds=time.time() - T, model_training=0, model_TEST_evaluation=0, descriptive_Tahoe_all_roles=True)
(R / 'receipts/FINAL_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(rec, indent=2))
paths = [p for p in R.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and (not any((x in p.parts for x in ['logs', '__pycache__'])))]
(R / 'SHA256SUMS').write_text(''.join((sha(p) + '  ' + str(p.relative_to(R)) + '\n' for p in sorted(paths))))
print(json.dumps(rec, indent=2))

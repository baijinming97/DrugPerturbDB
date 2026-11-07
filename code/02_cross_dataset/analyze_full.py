from pathlib import Path
import os, sys, json, time, hashlib, resource, itertools, subprocess, collections
import numpy as np, pandas as pd, h5py
from scipy.stats import pearsonr
O = Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2/pairs')
F = Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
CAN = F.parent / 'baicai/canonical_perturbation_data_v1/tahoe/data/condition_manifest.parquet'
assert os.getenv('TAHOE_REPRO_RUN_ID') and (not os.getenv('CUDA_VISIBLE_DEVICES', '')) and (not (O / 'results/RESULTS.csv').exists())
start = time.time()
protocol = json.loads((O / 'PROTOCOL.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(O / 'IDENTITY_MAP_FINAL.json') == protocol['identity_sha256']
assert sha(O / 'LOCKED_SELECTIONS.json') == protocol['selection_sha256']
import shutil
free = shutil.disk_usage(O).free / 2**30
assert free >= 15

def guard(event, args):
    if event != 'open' or not args or (not isinstance(args[0], (str, bytes, os.PathLike))):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    fl = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    if 'TEST' in p.name.upper():
        raise RuntimeError('TEST file forbidden')
    if str(p).startswith('@WORKSPACE@') and fl & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        assert p.is_relative_to(O), str(p)
sys.addaudithook(guard)

def corr(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.divide(np.sum(x * y, axis=-1), den, out=np.full(den.shape, np.nan), where=den > 1e-12)

def bootstrap(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if not n:
        return (np.nan, np.nan, np.nan)
    if n < 3:
        return (float(v.mean()), np.nan, np.nan)
    rng = np.random.default_rng(20261030)
    b = np.array([v[rng.integers(n, size=n)].mean() for _ in range(2000)])
    return (float(v.mean()), *np.quantile(b, [0.025, 0.975]).tolist())
identity = json.loads((O / 'IDENTITY_MAP_FINAL.json').read_text())
selections = json.loads((O / 'LOCKED_SELECTIONS.json').read_text())
names = list(identity['drug'])
axis = json.loads((F / 'COMMON_GENES.json').read_text())['axis']
assert len(axis) == 869
need = {n: set() for n in names}
for z in selections:
    need[z['a']].update(z['rows_a'])
    need[z['b']].update(z['rows_b'])
vectors = {}
keytables = {}
sourcecounts = []
for n in names:
    keys = pd.read_parquet(O / 'metadata' / f'{n}_VECTOR_KEYS.parquet')
    keytables[n] = keys
    kk = keys.iloc[sorted(need[n])]
    rawids = {int(i) for s in kk.source_rows for i in s.split(',')}
    rr = np.array(sorted(rawids), int)
    loc = {int(v): i for i, v in enumerate(rr)}
    with h5py.File((O.parent if n == 'Tahoe' else F) / 'data' / f'{n}_COMMON_CONDITIONS.h5', 'r') as h:
        assert h['entrez'].asstr()[:].tolist() == axis
        v = np.empty((len(rr), 869), np.float64)
        for lo in range(0, len(rr), 512):
            v[lo:lo + 512] = h['delta'][rr[lo:lo + 512]]
    assert np.isfinite(v).all(), n
    vectors[n] = {int(z.value_row): v[[loc[int(i)] for i in z.source_rows.split(',')]].mean(0) for z in kk.itertuples()}
    sourcecounts.append(dict(dataset=n, source_conditions_read=len(rr), collapsed_identity_conditions=len(kk), genes=869))
    print('LOADED', n, len(rr), flush=True)
    del v
cm = pd.read_parquet(CAN)
chem = cm.drop_duplicates('drug').set_index('drug').chemistry_component.to_dict()
clusters = {}
for d, g in identity['drug']['Tahoe'].items():
    if g in clusters:
        assert clusters[g] == chem[d], ('merged Tahoe identity crosses chemistry groups', g, d)
    clusters[g] = chem[d]
rows = []
retrieval = []
examples = []
exx = []
exy = []
blocks_summary = []
selkey = lambda z: (z['a'], z['b'], z['mode'], z['time'])
for (a, b, mode, tm), zz in itertools.groupby(sorted(selections, key=selkey), key=selkey):
    zz = list(zz)
    X = []
    Y = []
    records = []
    for j, z in enumerate(zz):
        xx = np.stack([vectors[a][i] for i in z['rows_a']])
        yy = np.stack([vectors[b][i] for i in z['rows_b']])
        cell = z['cells_a'][0] if mode == 'STRICT_ALL_NEAR' else 'shared_cells' if mode == 'STRICT_NEAREST' else 'own_cell_mean'
        if mode.startswith('STRICT'):
            assert z['cells_a'] == z['cells_b'] and xx.shape == yy.shape
            score = float(np.nanmean(corr(xx, yy)))
        else:
            xx = xx.mean(0, keepdims=True)
            yy = yy.mean(0, keepdims=True)
            score = float(corr(xx, yy)[0])
        r = dict(a=a, b=b, mode=mode, time=tm, drug=z['drug'], cluster=clusters.get(z['drug'], z['drug']) if 'Tahoe' in [a, b] else z['drug'], cell=cell, dose_a=z['dose_a'], dose_b=z['dose_b'], dose_ratio=z['dose_ratio'], cells_a=len(z['rows_a']), cells_b=len(z['rows_b']), pcc=score, coordinate='R1')
        rows.append(r)
        records.append(r)
        X.append(xx.mean(0))
        Y.append(yy.mean(0))
        if j == 0 and np.isfinite(corr(xx[:1], yy[:1])[0]):
            e = dict(r)
            e['pcc'] = float(corr(xx[:1], yy[:1])[0])
            e['raw_rows_a'] = keytables[a].iloc[z['rows_a'][0]].source_rows
            e['raw_rows_b'] = keytables[b].iloc[z['rows_b'][0]].source_rows
            e['selection'] = z
            examples.append(e)
            exx.append(xx[0])
            exy.append(yy[0])
            assert abs(float(pearsonr(xx[0], yy[0]).statistic) - e['pcc']) < 1e-12
    if mode not in ['STRICT_ALL_NEAR', 'MARGINAL_ALL_NEAR']:
        continue
    X = np.stack(X)
    Y = np.stack(Y)
    dm = pd.DataFrame(records)
    for group, idx in dm.groupby(['cell', 'dose_a', 'dose_b'], sort=True).indices.items():
        if len(idx) < 2:
            continue
        assert dm.iloc[idx].drug.nunique() == len(idx)
        x = X[idx]
        y = Y[idx]
        xc = x - x.mean(1, keepdims=True)
        yc = y - y.mean(1, keepdims=True)
        nx = np.linalg.norm(xc, axis=1)
        ny = np.linalg.norm(yc, axis=1)
        good = (nx > 1e-12) & (ny > 1e-12)
        ids = np.asarray(idx)[good]
        xc = xc[good]
        yc = yc[good]
        nx = nx[good]
        ny = ny[good]
        n = len(ids)
        if n < 2:
            continue
        similarity = xc / nx[:, None] @ (yc / ny[:, None]).T
        diag = np.diag(similarity)
        wrong = (similarity.sum(1) - diag) / (n - 1)
        auc = ((similarity < diag[:, None]).sum(1) + 0.5 * (similarity == diag[:, None]).sum(1) - 0.5) / (n - 1)
        wrong_r = (similarity.sum(0) - diag) / (n - 1)
        auc_r = ((similarity < diag[None, :]).sum(0) + 0.5 * (similarity == diag[None, :]).sum(0) - 0.5) / (n - 1)
        for p, i in enumerate(ids):
            r = records[i]
            for direction, w, ac in [('a_to_b', wrong[p], auc[p]), ('b_to_a', wrong_r[p], auc_r[p])]:
                retrieval.append(dict(r, direction=direction, candidates=n, same_pcc=float(diag[p]), wrong_pcc=float(w), pcc_difference=float(diag[p] - w), auc=float(ac)))
        blocks_summary.append(dict(a=a, b=b, mode=mode, time=tm, cell=group[0], dose_a=group[1], dose_b=group[2], drugs=n))
        if len(idx) >= 3:
            xr = x - x.mean(0)
            yr = y - y.mean(0)
            scores = corr(xr, yr)
            for i, score in zip(idx, scores):
                rows.append(dict(records[i], coordinate='R3_common_drug_background', pcc=float(score)))
    print('SCORED', a, b, mode, tm, len(zz), flush=True)
raw = pd.DataFrame(rows)
raw.to_csv(O / 'results/CONDITION_RESULTS.csv.gz', index=False)
ret = pd.DataFrame(retrieval)
ret.to_csv(O / 'results/RETRIEVAL_CONDITIONS.csv.gz', index=False)
pd.DataFrame(blocks_summary).to_csv(O / 'results/RETRIEVAL_BLOCKS.csv', index=False)

def summarize(df, metrics, groupcols):
    out = []
    perdrug = []
    for key, g in df.groupby(groupcols, sort=True):
        key = (key,) if not isinstance(key, tuple) else key
        labels = dict(zip(groupcols, key))
        for metric in metrics:
            u = g.groupby(['drug', 'cluster', 'cell'])[metric].mean().groupby(['drug', 'cluster']).mean().reset_index()
            u['metric'] = metric
            for r in u.to_dict('records'):
                perdrug.append(dict(labels, **r))
            c = u.groupby('cluster')[metric].mean()
            point, lo, hi = bootstrap(c)
            out.append(dict(labels, metric=metric, estimate=point, low=lo, high=hi, drugs=int(u[metric].notna().sum()), groups=int(c.notna().sum()), condition_pairs=len(g), simple_drug_mean=float(u[metric].mean()), undefined_condition_pairs=int(g[metric].isna().sum()), genes=869, low_support=int(u[metric].notna().sum()) < 5))
    return (pd.DataFrame(out), pd.DataFrame(perdrug))
results, perdrug = summarize(raw, ['pcc'], ['a', 'b', 'mode', 'time', 'coordinate'])
results.to_csv(O / 'results/RESULTS.csv', index=False)
perdrug.to_csv(O / 'results/PER_DRUG_RESULTS.csv.gz', index=False)
if len(ret):
    r, p = summarize(ret, ['same_pcc', 'wrong_pcc', 'pcc_difference', 'auc'], ['a', 'b', 'mode', 'time', 'direction'])
    r.to_csv(O / 'results/RETRIEVAL_RESULTS.csv', index=False)
    p.to_csv(O / 'results/RETRIEVAL_PER_DRUG.csv.gz', index=False)
np.savez_compressed(O / 'results/REPLAY_EXAMPLES.npz', x=np.stack(exx), y=np.stack(exy))
(O / 'results/REPLAY_EXAMPLES.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2))
pd.DataFrame(sourcecounts).to_csv(O / 'results/SOURCE_READ_COUNTS.csv', index=False)
receipt = dict(status='ANALYZED_PENDING_INDEPENDENT_VERIFICATION', job=os.getenv('TAHOE_REPRO_RUN_ID'), elapsed_seconds=time.time() - start, process_cpu_hours=(resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime) / 3600, MaxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, GPU=0, model_TEST_response_reads=0, descriptive_Tahoe_all_roles=True, training=0, free_scratch_GiB_before=free, output_bytes=sum((p.stat().st_size for p in O.rglob('*') if p.is_file())))
(O / 'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
print(results.query('time==24 and coordinate=="R1"').to_string(index=False), flush=True)

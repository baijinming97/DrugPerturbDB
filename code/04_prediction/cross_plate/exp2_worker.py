"""Evaluate the G5 drug-average-to-interaction mapping within and across plates.

Select folds with --fold; --finalize aggregates saved sufficient statistics
with chemical-group panel recentering. Only TRAIN/VALID responses are read.
Use the CPU workflow entry point to configure the execution environment."""
from pathlib import Path
import os, sys, json, gzip, pickle, hashlib, time, collections, resource, ast, argparse, copy
sys.dont_write_bytecode = True
CPUS = int(os.environ.get('TAHOE_REPRO_CPUS', '0'))
assert os.environ.get('TAHOE_REPRO_RUN_ID') and 1 <= CPUS <= 8, 'CPU workflow entry point required'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = str(CPUS)
O = Path('@WORKSPACE@')
R = O / '4MAN/PAPER_TWO_CHECKS_V1/exp2'
A = O / '4MAN/minimum_restore_v1'
CAN = O / 'baicai/canonical_perturbation_data_v1/tahoe/data'
PL = A / 'retained/orion/baicai/new_audit/tahoe_l3_x1_noise_audit_v1/results/tahoe/plate978'
GE = A / 'retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data'
BASE = A / 'retained/orion/tahoe_x1only_3h2048_baseline_20261007/cpu_v2'
BD = O / '4MAN/BETA_DISCOVERY_V1'
SH = O / '4MAN/TAHOE_GAMMA_PARALLEL_V1/shared'
import numpy as np
import pandas as pd
import h5py
from scipy.linalg import solve, eigh
sys.path.insert(0, str(SH))
from tp_projection import Projector
START = time.monotonic()
READS = {}
ACCESS = []
HASHES = []
SEEDS = [20261103, 20261104, 20261105]
CFG = dict(name='G5', kind='dose', K=20, alpha=100, beta='within')
PCC_VAR_MIN = 1e-12 / 978

def guard(event, args):
    if event != 'open' or not args or (not isinstance(args[0], (str, bytes, os.PathLike))):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    if not str(p).startswith(str(O)):
        return
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        assert p.is_relative_to(R), 'write outside new exp2: ' + str(p)
    elif 'TEST' in p.name and p.suffix in ['.npz', '.h5', '.parquet']:
        raise RuntimeError('TEST file forbidden: ' + str(p))
sys.addaudithook(guard)

def dump(name, x):
    (R / name).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf-8')

def reg(path, whole=False):
    p = Path(str(path).replace('@' + 'WORKSPACE@', str(O)))
    s = p.stat()
    if str(p) not in READS:
        z = dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
        if whole:
            with p.open('rb') as h:
                z['sha256'] = hashlib.file_digest(h, 'sha256').hexdigest()
        else:
            z['hash_scope'] = 'selected legal arrays; no whole mixed-role response-file hash'
        READS[str(p)] = z
    return p

def ah(name, x, **meta):
    a = np.ascontiguousarray(x)
    HASHES.append(dict(name=name, shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest(), **meta))

def budget():
    cap = float(os.environ.get('EXP2_JOB_COREH_CAP', '4.7'))
    if (time.monotonic() - START) * CPUS / 3600 > cap:
        raise RuntimeError('Exp2 allocated core-hour cap reached')

def weights(g):
    ct = collections.Counter(g)
    return np.array([1 / (len(ct) * ct[x]) for x in g])

def dec(y, w):
    cell = np.einsum('cdg,d->cg', y, w)
    mu = cell.mean(0)
    b = y.mean(0) - mu
    return dict(mu=mu, alpha=cell - mu, beta=b, gamma=y - cell[:, None] - b[None])

def pb(b, x, w):
    flat = np.ascontiguousarray(x.transpose(1, 0, 2).reshape(x.shape[1], -1))
    _, ids = np.unique(flat.view(np.dtype((np.void, flat.shape[1] * flat.dtype.itemsize))).ravel(), return_inverse=True)
    out = b.copy()
    for k in np.unique(ids):
        ix = ids == k
        out[ix] -= np.average(b[ix], axis=0, weights=w[ix])
    return out

def raw(h, b):
    x = (b - h['mu']) @ h['v'].T / h['scale']
    x = np.c_[x, np.ones(len(x))]
    z = (x - h['xm']) @ h['coef'] + h['ym']
    return (z @ h['U']).reshape(len(b), 47, 978).transpose(1, 0, 2)

def safe_pcc(x, y):
    a = x - x.mean(-1, keepdims=True)
    b = y - y.mean(-1, keepdims=True)
    aa = np.mean(a * a, -1)
    bb = np.mean(b * b, -1)
    den = np.sqrt(aa * bb)
    return np.divide(np.mean(a * b, -1), den, out=np.full(den.shape, np.nan), where=(aa > PCC_VAR_MIN) & (bb > PCC_VAR_MIN))

def mean_metric(v, w):
    ww = np.broadcast_to(w[None] / v.shape[0], v.shape)
    ok = np.isfinite(v)
    return float(np.sum(np.where(ok, v, 0) * ww) / np.sum(ww * ok)) if np.sum(ww * ok) > 0 else np.nan

def direct_metrics(y, b, d, w):
    yt = dec(y, w)['gamma']
    new = b + d
    return dict(full_PCC=mean_metric(safe_pcc(new, y), w), B_full_PCC=mean_metric(safe_pcc(b, y), w), full_MSE=float(np.einsum('cdg,d->', (new - y) ** 2, w) / (47 * 978)), B_full_MSE=float(np.einsum('cdg,d->', (b - y) ** 2, w) / (47 * 978)), gamma_PCC=mean_metric(safe_pcc(d, yt), w), B_gamma_PCC=np.nan, gamma_MSE=float(np.einsum('cdg,d->', (d - yt) ** 2, w) / (47 * 978)), B_gamma_MSE=float(np.einsum('cdg,d->', yt * yt, w) / (47 * 978)))

def load_train(f, rec):
    p = reg(GE / f'A_f{f}_TRAIN.npz')
    with np.load(p, allow_pickle=False) as z:
        ds = z['drug_id'].astype(str)
        cs = z['cell_id'].astype(str)
        assert set(ds) == set(rec['fit_drug'])
        ys = z['y']
        xs = z['x']
    drug = np.array(rec['fit_drug'])
    cells = rec['cells']
    lookup = {(c, d): i for i, (c, d) in enumerate(zip(cs, ds))}
    ix = np.array([[lookup[c, d] for d in drug] for c in cells])
    y = ys[ix].astype(float)
    x = xs[ix]
    assert set(ds) == set(drug)
    ah('TRAIN_Y', y, fold=f, role='TRAIN', drugs=drug.tolist())
    ACCESS.append(dict(fold=f, role='TRAIN', asset=str(p), response_drugs=drug.tolist(), TEST_rows=0))
    out = dict(drug=drug, group=np.array(rec['fit_group']), X=x, Y=y)
    lows = []
    for row in rec['lowdose']:
        assert row['exists'], 'Required low-dose input is missing for this configuration'
        path = reg(row['path'])
        with np.load(path, allow_pickle=False) as z:
            ds = z['drug'].astype(str)
            gs = z['group'].astype(str)
            assert set(ds) <= set(drug) and (not set(gs) & set(rec['VALID_group'].values()))
            lp = {k: z[k] for k in ['drug', 'group', 'X', 'Y']}
        lp['drug'] = lp['drug'].astype(str)
        lp['group'] = lp['group'].astype(str)
        lp['Y'] = lp['Y'].astype(float)
        assert set(lp['drug']) <= set(drug)
        assert not set(lp['group']) & set(rec['VALID_group'].values())
        ah('LOWDOSE_TRAIN_Y', lp['Y'], fold=f, dose=row['dose'], drugs=lp['drug'].tolist())
        ACCESS.append(dict(fold=f, role='TRAIN', dose=row['dose'], asset=str(path), response_drugs=lp['drug'].tolist(), TEST_rows=0))
        lows.append((lp, float(row['dose'])))
    return (out, lows)

def fit_controls(f, h, p, lows):
    """Fixed historical G5 basis/preprocessing; refit only linear input mapping.

    Unpermuted reconstruction is checked against the frozen model first. All
    doses then share one TRAIN-drug permutation within joint coverage pattern.
    """
    panels = [(p, 5.0)] + lows
    parts = []
    rebuild = {}
    for ip, (pp, dose) in enumerate(panels):
        budget()
        w = weights(pp['group'])
        proj = Projector(pp['X'], w)
        y = proj(pp['Y'])
        b = pb(dec(pp['Y'], w)['beta'], pp['X'], w)
        x = (b - h['mu']) @ h['v'].T / h['scale']
        flat = y.transpose(1, 0, 2).reshape(len(b), -1)
        if ip == 0:
            obs = np.repeat(proj.eligible.T, 978, axis=1)
            z = np.zeros((len(b), len(h['U'])))
            filled = np.where(obs, flat, 0)
            sw = np.sqrt(w / w.sum())
            previous = None
            for it in range(250 if not obs.all() else 1):
                budget()
                S = filled * sw[:, None]
                ev, vec = eigh(S @ S.T, check_finite=False)
                ix = np.argsort(ev)[::-1][:min(20, len(flat) - 1)]
                ev = ev[ix]
                vec = vec[:, ix]
                keep = ev > max(ev.max(initial=0) * 1e-12, 1e-20)
                U = vec[:, keep].T @ S / np.sqrt(ev[keep])[:, None]
                if obs.all():
                    break
                recon = filled @ U.T @ U
                obj = np.sum((recon - flat) ** 2 * obs * w[:, None])
                filled[~obs] = recon[~obs]
                if previous is not None and abs(obj - previous) / max(abs(previous), 1e-20) < 1e-05:
                    break
                previous = obj
            assert U.shape == h['U'].shape
            basis_error = float(np.max(abs(U - U @ h['U'].T @ h['U'])))
            assert basis_error < 1e-06, ('Frozen G5 basis does not reproduce from legal TRAIN', basis_error)
            centered = b - b.mean(0)
            ev, vec = eigh(centered @ centered.T, check_finite=False)
            ix = np.argsort(ev)[::-1][:min(32, len(b) - 1, 978)]
            ev = ev[ix]
            vec = vec[:, ix]
            keep = ev > max(ev.max(initial=0) * 1e-12, 1e-20)
            V = vec[:, keep].T @ centered / np.sqrt(ev[keep])[:, None]
            pca_error = float(np.max(abs(V - V @ h['v'].T @ h['v'])))
            mean_error = float(np.max(abs(b.mean(0) - h['mu'])))
            scale_error = float(np.max(abs(np.maximum((centered @ h['v'].T).std(0), 1e-08) - h['scale'])))
            assert max(pca_error, mean_error, scale_error) < 1e-06
            rebuild = dict(gamma_basis_subspace_maxabs=basis_error, beta_PCA_subspace_maxabs=pca_error, beta_mean_maxabs=mean_error, beta_scale_maxabs=scale_error, basis_iterations=it + 1, source='Only this outer-fold TRAIN response arrays and metadata; no VALID/TEST response read before these checks.')
            for j in range(len(b)):
                z[j] = np.linalg.lstsq(h['U'][:, obs[j]].T, flat[j, obs[j]], rcond=1e-10)[0]
        else:
            z = flat @ h['U'].T
        parts.append(dict(drug=pp['drug'], x=x, z=z, w=w, dose=dose, beta=b))

    def refit(mapping):
        xx = []
        zz = []
        ww = []
        for pp in parts:
            ix = {d: i for i, d in enumerate(pp['drug'])}
            order = np.array([ix[mapping[d]] for d in pp['drug']])
            xx.append(np.c_[pp['x'][order], np.full(len(order), np.log10(pp['dose'] / 5) + 1)])
            zz.append(pp['z'])
            ww.append(pp['w'])
        x = np.concatenate(xx)
        z = np.concatenate(zz)
        w = np.concatenate(ww)
        w = w / w.sum() * len(w)
        xm = np.average(x, axis=0, weights=w)
        ym = np.average(z, axis=0, weights=w)
        a = x - xm
        coef = solve(a.T @ (a * w[:, None]) + 100 * np.eye(x.shape[1]), a.T @ ((z - ym) * w[:, None]), assume_a='pos', check_finite=False)
        hh = h.copy()
        hh.update(xm=xm, ym=ym, coef=coef)
        return hh
    identity = {d: d for d in p['drug']}
    replay = refit(identity)
    b = parts[0]['beta']
    check = raw(replay, b) - raw(h, b)
    error = float(np.max(abs(check)))
    assert error < 1e-06, ('Unshuffled fixed-recipe replay failed', f, error)
    coverage = {d: tuple((d in set(pp['drug']) for pp in parts)) for d in p['drug']}
    buckets = collections.defaultdict(list)
    for d, pattern in coverage.items():
        buckets[pattern].append(d)
    controls = []
    audits = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mapping = {}
        for pattern, ds in sorted(buckets.items()):
            ds = sorted(ds)
            dest = rng.permutation(ds)
            mapping.update(zip(ds, map(str, dest)))
        change = {}
        for pp in parts:
            ix = {d: i for i, d in enumerate(pp['drug'])}
            order = [ix[mapping[d]] for d in pp['drug']]
            change[str(pp['dose'])] = dict(changed_drugs=sum((mapping[d] != d for d in pp['drug'])), changed_feature_rows=int(np.any(pp['x'] != pp['x'][order], axis=1).sum()), total=len(order))
        assert change['5.0']['changed_feature_rows'] > 0, 'Degenerate null'
        controls.append((f'shuffle_{seed}', refit(mapping)))
        audits.append(dict(seed=seed, mapping=mapping, per_dose=change, coverage_buckets={str(k): len(v) for k, v in buckets.items()}, fixed_drugs_reason='Singleton joint-dose-coverage patterns cannot be permuted without inventing missing measurements.'))
    dump(f'NULL_AUDIT_f{f}.json', dict(fold=f, unshuffled_maxabs_replay=error, TRAIN_basis_rebuild=rebuild, models=audits, scientific_difference_from_legacy='Legacy separately shuffled each dose by row position; this run uses the same whole-drug mapping at all doses.', fitting='Only fixed G5 linear input-to-code map is refit for controls; original basis, rank, alpha, and TRAIN-only preprocessing are fixed.'))
    return [('G5_true', h)] + controls

def load_paired(f, rec):
    cm = pd.read_parquet(reg(CAN / 'condition_manifest.parquet', True))
    mm = pd.read_parquet(reg(PL / 'tahoe_multiplate_plate_profile_manifest_978_v1.parquet', True))
    cc = pd.read_parquet(reg(CAN / 'control_context_manifest.parquet', True))
    cells = rec['cells']
    drugs = rec['shared_drug']
    pos = pd.read_csv(reg(GE / 'GENE_AXIS.tsv', True), sep='\t', dtype=str).tahoe_gene_position.to_numpy(int)
    assert len(pos) == 978 and len(set(pos)) == 978
    sort = np.argsort(pos)
    inv = np.argsort(sort)
    ctr = {}
    with h5py.File(reg(CAN / 'controls.h5'), 'r') as stream:
        for row in cc[cc.plate.isin([6, 14]) & cc.cell_line.isin(cells)].itertuples():
            ctr[int(row.plate), str(row.cell_line)] = stream['controls'][int(row.control_row), pos[sort]][inv].astype(float)
    base = cm[cm.drug.isin(drugs) & (cm.dose == 5.0)]
    one = base[base.source_plate_profiles == 1].copy()
    one['plate'] = one.plates.astype(int)
    one['kind'] = 'single'
    one['value_row'] = -1
    multi = mm[mm.condition_row.isin(base.loc[base.source_plate_profiles > 1, 'condition_row'])].copy()
    multi['kind'] = 'multi'
    multi['treated_shard'] = -1
    multi['treated_row_in_shard'] = -1
    cols = ['condition_row', 'cell_line', 'drug', 'plate', 'kind', 'value_row', 'treated_shard', 'treated_row_in_shard']
    meta = pd.concat([one[cols], multi[cols]], ignore_index=True)
    meta = meta[meta.plate.isin([6, 14])].reset_index(drop=True)
    assert len(meta) == len(drugs) * 47 * 2 and set(meta.drug) <= set(rec['VALID_drug'])
    values = np.empty((len(meta), 978))
    for (kind, shard), ss in meta.groupby(['kind', 'treated_shard']):
        path = CAN / f'treated_{int(shard):02d}-of-16.h5' if kind == 'single' else PL / 'tahoe_multiplate_plate_profiles_978_v1.h5'
        rc = 'treated_row_in_shard' if kind == 'single' else 'value_row'
        ss = ss.sort_values(rc)
        with h5py.File(reg(path), 'r') as stream:
            for start in range(0, len(ss), 64):
                batch = ss.iloc[start:start + 64]
                ix = batch[rc].to_numpy(int)
                assert len(ix) == len(set(ix))
                rawx = stream['treated' if kind == 'single' else 'X2'][ix]
                if kind == 'single':
                    rawx = rawx[:, pos]
                for (ii, t), v in zip(batch.iterrows(), rawx):
                    values[ii] = v - ctr[int(t.plate), str(t.cell_line)]
    lookup = {(int(t.plate), str(t.cell_line), str(t.drug)): i for i, t in enumerate(meta.itertuples())}
    ys = {plate: values[[lookup[plate, c, d] for c in cells for d in drugs]].reshape(47, len(drugs), 978) for plate in [6, 14]}
    xs = {plate: np.broadcast_to(np.array([ctr[plate, c] for c in cells])[:, None], ys[plate].shape).copy().astype('float32') for plate in [6, 14]}
    for plate in [6, 14]:
        assert np.array_equal(xs[plate], np.broadcast_to(xs[plate][:, :1], xs[plate].shape)), 'Gram shortcut requires exactly one context per cell on each plate'
        ah('VALID_single_plate_response', ys[plate], fold=f, plate=plate, drugs=drugs)
        ah('VALID_single_plate_control', xs[plate][:, 0], fold=f, plate=plate)
    ACCESS.append(dict(fold=f, role='VALID', drugs=drugs, cells=cells, physical_rows=len(meta), dose=5.0, TEST_rows=0, own_plate_controls=True))
    return (ys, xs)

def infer_B(f, xs):
    import torch
    from torch import nn
    torch.set_num_threads(CPUS)
    src = reg(BASE / 'scripts/tb_train.py', True)
    nodes = [n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'BasalMLP']
    assert len(nodes) == 1
    ns = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec'), ns)
    ckpath = reg(BASE / f'runs/A_f{f}_3H2048/selected.pt', True)
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    assert ck['seed'] == 1337
    net = ns['BasalMLP']()
    net.load_state_dict(ck['model'])
    net.eval()
    out = {}
    with torch.inference_mode():
        for plate, x in xs.items():
            v = net(torch.from_numpy(np.ascontiguousarray(x[:, 0]))).numpy()
            out[plate] = np.broadcast_to(v[:, None], x.shape).copy().astype(float)
    return out

def gram(a, b):
    return np.einsum('cdg,ceg->cde', a, b, optimize=True) / a.shape[-1]

def sufficient(y, b, u):
    """u is cell-centered but drug centering is re-estimated for every draw."""
    assert np.max(abs(b - b[:, :1])) < 1e-12
    v = y - y.mean(0, keepdims=True)
    uc = u - u.mean(-1, keepdims=True)
    vc = v - v.mean(-1, keepdims=True)
    yc = y - y.mean(-1, keepdims=True)
    bc = b[:, 0] - b[:, 0].mean(-1, keepdims=True)
    return dict(uu=gram(u, u), vv=gram(v, v), uv=gram(u, v), uuc=gram(uc, uc), vvc=gram(vc, vc), uvc=gram(uc, vc), uy=gram(uc, yc), ub=np.einsum('cdg,cg->cd', uc, bc) / 978, by=np.einsum('cg,cdg->cd', bc, yc) / 978, bn=np.mean(bc * bc, -1), yn=np.mean(yc * yc, -1), bm=np.mean((b - y) ** 2, -1))

def stats(s, W):
    """Vectorized exact re-centering with normalized per-draw drug weights."""
    n, d = W.shape
    num = {k: np.zeros(n) for k in ['full_PCC', 'B_full_PCC', 'gamma_PCC', 'full_MSE', 'B_full_MSE', 'gamma_MSE', 'B_gamma_MSE']}
    pden = {k: np.zeros(n) for k in ['full_PCC', 'B_full_PCC', 'gamma_PCC']}
    enough = (W > 0).sum(1) >= 3
    for c in range(47):

        def centered(k):
            a = s[k][c]
            left = W @ a
            right = W @ a.T
            mid = np.sum(left * W, 1)
            return np.diag(a)[None] - left - right + mid[:, None]
        uu = centered('uu')
        vv = centered('vv')
        uv = centered('uv')
        uu[~enough] = 0
        uv[~enough] = 0
        gm = np.sum((uu + vv - 2 * uv) * W, 1)
        bm = np.sum(s['bm'][c][None] * W, 1)
        num['gamma_MSE'] += gm / 47
        num['B_gamma_MSE'] += np.sum(vv * W, 1) / 47
        num['B_full_MSE'] += bm / 47
        num['full_MSE'] += (bm + np.sum((uu - 2 * uv) * W, 1)) / 47
        uuc = centered('uuc')
        vvc = centered('vvc')
        uvc = centered('uvc')
        uuc[~enough] = 0
        uvc[~enough] = 0
        ub = s['ub'][c][None] - W @ s['ub'][c][:, None]
        ut = np.diag(s['uy'][c])[None] - W @ s['uy'][c]
        ub[~enough] = 0
        ut[~enough] = 0
        fn = s['bn'][c] + uuc + 2 * ub
        fc = s['by'][c][None] + ut
        for key, dot, xn, yn in [('full_PCC', fc, fn, s['yn'][c][None]), ('B_full_PCC', s['by'][c][None], np.full((n, d), s['bn'][c]), s['yn'][c][None]), ('gamma_PCC', uvc, uuc, vvc)]:
            ok = (xn > PCC_VAR_MIN) & (yn > PCC_VAR_MIN)
            den = np.sqrt(np.maximum(xn, 0) * np.maximum(yn, 0))
            rho = np.divide(dot, den, out=np.zeros((n, d)), where=ok)
            num[key] += np.sum(rho * W, 1) / 47
            pden[key] += np.sum(ok * W, 1) / 47
    for key in pden:
        num[key] = np.divide(num[key], pden[key], out=np.full(n, np.nan), where=pden[key] > 0)
        num[key + '_defined_weight'] = pden[key]
    num['B_gamma_PCC'] = np.full(n, np.nan)
    num['full_PCC_gain'] = num['full_PCC'] - num['B_full_PCC']
    num['full_MSE_gain'] = num['B_full_MSE'] - num['full_MSE']
    num['gamma_MSE_gain'] = num['B_gamma_MSE'] - num['gamma_MSE']
    return num

def run_fold(f, audit):
    start = time.monotonic()
    rec = next((v for v in audit['folds'] if v['fold'] == f))
    assert rec['lineage_status'] == 'PASS_OUTER_TRAIN_WHITELIST_ALL_DOSE_CACHE'
    modelpath = reg(rec['path'], True)
    assert READS[str(modelpath)]['sha256'] == rec['sha256']
    with gzip.open(modelpath, 'rb') as stream:
        h, oldlam = pickle.load(stream)
    assert h['cfg'] == CFG and list(map(str, h['train_drug'])) == rec['fit_drug']
    p, lows = load_train(f, rec)
    assert np.array_equal(h['train_X'], p['X'])
    models = fit_controls(f, h, p, lows)
    del p, lows
    budget()
    ys, xs = load_paired(f, rec)
    bs = infer_B(f, xs)
    rolecounts = audit['valid_role_counts']
    gg = audit['global_drug_group']
    allvalid = list(rolecounts)
    size = collections.Counter((gg[d] for d in allvalid))
    ng = len(size)
    q = np.array([1 / (ng * size[gg[d]] * rolecounts[d]) for d in rec['shared_drug']])
    w = q / q.sum()
    for plate in [6, 14]:
        P = Projector(xs[plate], w, 3)
        assert np.max(abs(P(ys[plate]) - dec(ys[plate], w)['gamma'])) < 1e-08
        assert np.max(abs(dec(bs[plate], w)['gamma'])) < 1e-10
    saved = dict(drug=np.array(rec['shared_drug']), group=np.array(rec['shared_group']), q=q)
    rows = []
    pure = []
    check = []
    config = []
    for mi, (name, m) in enumerate(models):
        for source in [6, 14]:
            bet = dec(ys[source], w)['beta']
            unprojected = raw(m, bet)
            u = unprojected - unprojected.mean(0, keepdims=True)
            for target in [6, 14]:
                budget()
                direction = f'{source}_to_{target}'
                delta = u - np.einsum('cdg,d->cg', u, w)[:, None]
                s = sufficient(ys[target], bs[target], u)
                key = f'm{mi}_s{source}_t{target}'
                for k, v in s.items():
                    saved[key + '__' + k] = v
                config.append(dict(key=key, model=name, source=source, target=target, direction=direction))
                v = {k: float(z[0]) for k, z in stats(s, w[None]).items()}
                direct = direct_metrics(ys[target], bs[target], delta, w)
                err = max((abs(v[k] - direct[k]) for k in direct if np.isfinite(direct[k])))
                assert err < 1e-08, err
                dd = dec(delta, w)
                errm = max((float(np.max(abs(dd[k]))) for k in ['mu', 'alpha', 'beta']))
                closure = abs(v['full_MSE_gain'] - v['gamma_MSE_gain'])
                assert errm < 1e-08 and closure < 1e-08
                pure.append(dict(fold=f, model=name, direction=direction, pure_gamma_mean_maxabs=errm, MSE_gain_closure=closure))
                rows.append(dict(fold=f, model=name, direction=direction, setting='within' if source == target else 'cross', drugs=len(q), groups=len(set(rec['shared_group'])), weight=float(q.sum()), lambda_used=1.0, **v))
                if name == 'G5_true' and source == 6 and (target == 14):
                    wb = w * np.arange(1, len(w) + 1)
                    wb /= wb.sum()
                    db = Projector(xs[target], wb, 3)(raw(m, dec(ys[source], wb)['beta']))
                    ds = direct_metrics(ys[target], bs[target], db, wb)
                    fast = {k: float(z[0]) for k, z in stats(s, wb[None]).items()}
                    drift = max((abs(ds[k] - fast[k]) for k in ds if np.isfinite(ds[k])))
                    assert drift < 1e-08, drift
                    check.append(dict(fold=f, nonuniform_recenter_repredict_KKT_error=drift, point_Gram_direct_error=err))
                    if f == 0:
                        np.savez_compressed(R / 'verification_inputs.npz', X=xs[target], Y=ys[target], B=bs[target], delta=delta, w=w, drug=np.array(rec['shared_drug']), group=np.array(rec['shared_group']), role=np.array(['VALID'] * len(w)), train_drug=np.array(rec['fit_drug']), train_group=np.array(rec['fit_group']), claimed_full_pcc_gain=v['full_PCC_gain'], claimed_full_mse_gain=v['full_MSE_gain'], claimed_gamma_mse_gain=v['gamma_MSE_gain'])
    np.savez_compressed(R / f'SUFFICIENT_f{f}.npz', **saved)
    dump(f'CONFIG_f{f}.json', config)
    pd.DataFrame(rows).to_csv(R / f'POINT_f{f}.csv', index=False)
    dump(f'PURE_f{f}.json', pure)
    dump(f'GRAM_VERIFICATION_f{f}.json', check)
    dump(f'TIMING_f{f}.json', dict(fold=f, elapsed_seconds=time.monotonic() - start, allocated_core_hours=(time.monotonic() - start) * CPUS / 3600, CPU=CPUS, GPU=0, trained_neural_models=0, frozen_models_mutated=0, fixed_linear_null_refits=3))
    print('EXP2_FOLD_DONE', f, 'seconds', time.monotonic() - start, 'coreh', (time.monotonic() - start) * CPUS / 3600, flush=True)

def finalize():
    configs = json.loads((R / 'CONFIG_f0.json').read_text())
    panels = []
    groups = set()
    points = []
    for f in range(5):
        with np.load(R / f'SUFFICIENT_f{f}.npz', allow_pickle=False) as z:
            panels.append({k: z[k] for k in z.files})
        groups.update(panels[-1]['group'])
        points.append(pd.read_csv(R / f'POINT_f{f}.csv'))
    groups = sorted(groups)
    ix = {g: i for i, g in enumerate(groups)}
    counts = np.random.default_rng(20261103).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    full = []
    bsall = {}
    pointall = {}
    for cfg in configs:
        budget()
        sums = {}
        dens = {}
        den = np.zeros(2001)
        foldvals = []
        for f, p in enumerate(panels):
            mult = np.r_[np.ones((1, len(p['group']))), counts[:, [ix[g] for g in p['group']]]]
            qw = mult * p['q'][None]
            mass = qw.sum(1)
            W = np.divide(qw, mass[:, None], out=np.zeros_like(qw), where=mass[:, None] > 0)
            s = {k.split('__', 1)[1]: v for k, v in p.items() if k.startswith(cfg['key'] + '__')}
            st = stats(s, W)
            den += mass
            for k, v in st.items():
                if k not in sums:
                    sums[k] = np.zeros(2001)
                    dens[k] = np.zeros(2001)
                metric_mass = mass * st[k + '_defined_weight'] if k + '_defined_weight' in st else mass
                sums[k] += np.nan_to_num(v, nan=0) * metric_mass
                dens[k] += np.isfinite(v) * metric_mass
            foldvals.append(float(st['full_PCC_gain'][0]))
        values = {k: np.divide(v, dens[k], out=np.full(2001, np.nan), where=dens[k] > 0) for k, v in sums.items()}
        values['B_gamma_PCC'][:] = np.nan
        row = dict(model=cfg['model'], direction=cfg['direction'], setting='within' if cfg['source'] == cfg['target'] else 'cross', drugs=len({d for p in panels for d in p['drug']}), chemical_groups=len(groups), lambda_used=1.0, folds_improved=int(np.sum(np.array(foldvals) > 1e-08)), folds_tied=int(np.sum(abs(np.array(foldvals)) <= 1e-08)))
        for k, v in values.items():
            row[k] = float(v[0])
            row[k + '_lo'] = float(np.quantile(v[1:], 0.025)) if np.isfinite(v[1:]).all() else np.nan
            row[k + '_hi'] = float(np.quantile(v[1:], 0.975)) if np.isfinite(v[1:]).all() else np.nan
        full.append(row)
        bsall[cfg['model'], cfg['direction']] = values
    for name in ['G5_true'] + [f'shuffle_{s}' for s in SEEDS]:
        for label, directions in [('cross_mean', ['6_to_14', '14_to_6']), ('within_mean', ['6_to_6', '14_to_14'])]:
            value = {k: np.mean([bsall[name, d][k] for d in directions], axis=0) for k in bsall[name, directions[0]]}
            bsall[name, label] = value
            row = dict(model=name, direction=label, setting=label, drugs=len({d for p in panels for d in p['drug']}), chemical_groups=len(groups), lambda_used=1.0)
            for k, v in value.items():
                row[k] = float(v[0])
                row[k + '_lo'] = float(np.quantile(v[1:], 0.025)) if np.isfinite(v[1:]).all() else np.nan
                row[k + '_hi'] = float(np.quantile(v[1:], 0.975)) if np.isfinite(v[1:]).all() else np.nan
            full.append(row)
    table = pd.DataFrame(full)
    table.to_csv(R / 'WITHIN_CROSS_PLATE.csv', index=False)
    table.to_csv(R / 'FULL_RESPONSE_RESULTS.csv', index=False)
    pd.concat(points).to_csv(R / 'PER_FOLD.csv', index=False)
    contrasts = []
    for direction in ['6_to_6', '6_to_14', '14_to_6', '14_to_14', 'cross_mean', 'within_mean']:
        true = bsall['G5_true', direction]
        for compare in ['mean_three_shuffles'] + [f'shuffle_{s}' for s in SEEDS]:
            nulls = [bsall[f'shuffle_{s}', direction] for s in SEEDS] if compare == 'mean_three_shuffles' else [bsall[compare, direction]]
            for metric in ['full_PCC', 'full_MSE', 'gamma_PCC', 'gamma_MSE']:
                diff = true[metric] - np.mean([v[metric] for v in nulls], axis=0)
                contrasts.append(dict(direction=direction, comparison='G5_true_minus_' + compare, metric=metric, delta=diff[0], lo=np.quantile(diff[1:], 0.025), hi=np.quantile(diff[1:], 0.975), positive_is_better=metric.endswith('PCC')))
    pd.DataFrame(contrasts).to_csv(R / 'SHUFFLE_CONTROLS.csv', index=False)
    t = bsall['G5_true', 'cross_mean']
    ns = [bsall[f'shuffle_{s}', 'cross_mean'] for s in SEEDS]
    advantage = t['gamma_PCC'] - np.mean([n['gamma_PCC'] for n in ns], axis=0)
    relation = bool(t['gamma_PCC'][0] > 0 and all((bsall['G5_true', d]['gamma_PCC'][0] > 0 for d in ['6_to_14', '14_to_6'])) and (np.quantile(advantage[1:], 0.025) > 0))
    complete = bool(relation and t['gamma_MSE_gain'][0] > 0 and (t['full_PCC_gain'][0] > 0) and (t['full_MSE_gain'][0] > 0) and (np.quantile(t['full_PCC_gain'][1:], 0.025) > 0) and (np.quantile(t['full_MSE_gain'][1:], 0.025) > 0))
    dump('PRIMARY_ENDPOINTS.json', dict(cross_plate_relationship_evidence=relation, complete_prediction_improvement_evidence=complete, gamma_PCC=float(t['gamma_PCC'][0]), true_minus_mean_shuffle_gamma_PCC=float(advantage[0]), true_minus_mean_shuffle_gamma_PCC_CI=list(map(float, np.quantile(advantage[1:], [0.025, 0.975]))), mathematical_clarification='Single-plate X1-only gamma is identically zero, hence its PCC is undefined. Positive predicted-gamma PCC plus superiority to shuffle replaces a meaningless PCC-versus-zero comparison; gamma MSE still compares against the exact zero baseline.', gain_sign='All *_gain columns mean improvement: PCC new minus baseline; MSE baseline minus new.', complete_rule_scope='Pre-specified cross-direction mean; per-direction and per-fold results are reported separately and may differ.', scope='Two directions of one measured plate pair, development VALID, not an independent new test set or causal beta-to-gamma evidence.'))
    pure = sum([json.loads((R / f'PURE_f{f}.json').read_text()) for f in range(5)], [])
    checks = sum([json.loads((R / f'GRAM_VERIFICATION_f{f}.json').read_text()) for f in range(5)], [])
    dump('PURE_GAMMA_CHECK.json', dict(status='PASS', checks=pure))
    dump('GRAM_REPLAY_VERIFICATION.json', dict(status='PASS', checks=checks, method='Every bootstrap resample re-estimates fold centering and the pure-gamma projection; frozen linear G5 and exactly one X1 context per physical plate make the Gram formula algebraically equivalent.'))
    dump('BOOTSTRAP_PROTOCOL.json', dict(draws=2000, seed=20261103, unit='chemical supercomponent; joint repeated roles and both directions', reestimate_panel_means=True, reestimate_projection=True, refit_G5=False, small_sample='Interaction correction fixed zero for draws with fewer than 3 unique sampled drugs in a fold.', metric_zero_norm='PCC undefined (NA); no synthetic B gamma PCC of zero', support='Historical development VALID only; one plate pair; external source-plate beta is a measured input.'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', nargs='+', type=int)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    reg(Path(__file__).resolve(), True)
    auditpath = R / 'G5_LINEAGE_AUDIT.json'
    if not auditpath.exists():
        auditpath = R.parent / 'audit/G5_LINEAGE_AUDIT.json'
    audit = json.loads(auditpath.read_text().replace('@WORKSPACE@', O.as_posix()))
    assert audit['status'] == 'PASS_METADATA'
    if args.fold:
        for f in args.fold:
            run_fold(f, audit)
    if args.finalize:
        finalize()
    tag = 'finalize' if args.finalize else '_'.join(map(str, args.fold or []))
    for path, rec in READS.items():
        st = Path(path).stat()
        assert st.st_size == rec['bytes'] and st.st_mtime_ns == rec['mtime_ns'], 'Frozen source changed'
    dump(f'READS_{tag}.json', list(READS.values()))
    dump(f'READ_ROLES_{tag}.json', ACCESS)
    dump(f'ARRAY_HASHES_{tag}.json', HASHES)
    dump(f'RESOURCE_{tag}.json', dict(status='COMPLETE', job_id=os.environ['TAHOE_REPRO_RUN_ID'], CPUs=CPUS, GPU=0, elapsed_seconds=time.monotonic() - START, allocated_core_hours=(time.monotonic() - START) * CPUS / 3600, process_cpu_seconds=time.process_time(), maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, TEST_response_reads=0, frozen_assets_modified=0, lambda_used=1.0))
if __name__ == '__main__':
    main()

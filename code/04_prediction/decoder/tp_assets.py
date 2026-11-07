from pathlib import Path
import os, sys, json, hashlib, time, collections, ast
import numpy as np, pandas as pd, h5py
R = Path(os.environ.get('TAHOE_OUTPUT_DIR', str(Path(__file__).resolve().parent)))
O = Path('@WORKSPACE@')
A = O / '4MAN/minimum_restore_v1'
CAN = A / 'retained/orion/baicai/canonical_perturbation_data_v1/tahoe/data'
MECH = O / '4MAN/GAMMA_W_MECH_V1'
FEW = O / '4MAN/TAHOE_FEWSHOT_L1000_V1'
BASE = A / 'retained/orion/tahoe_x1only_3h2048_baseline_20261007/cpu_v2'
GE = A / 'retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data'
VEC = A / 'retained/orion/TAHOE_CHAMPION_REGISTRY/assets/TAHOE_ALPHA_BETA_X3_ASSEMBLY_V1_20261008_220046/vectors'
PL = A / 'retained/orion/baicai/new_audit/tahoe_l3_x1_noise_audit_v1/results/tahoe/plate978'
START = time.monotonic()
CPUS = int(os.environ.get('TAHOE_REPRO_CPUS', '2'))
READS = {}
ACCESS = []
assert os.environ.get('TAHOE_REPRO_RUN_ID') and CPUS in (4, 8)

def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if str(p).startswith(str(O)):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                assert p.is_relative_to(R), str(p)
            elif 'TEST' in p.name and p.suffix in ['.npz', '.h5', '.parquet']:
                raise RuntimeError('TEST asset blocked: ' + str(p))
sys.addaudithook(guard)

def dump(name, value):
    (R / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))

def reg(p):
    p = Path(p)
    if str(p) not in READS:
        READS[str(p)] = {'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns_before': p.stat().st_mtime_ns}
    return p

def finish_reads(prefix='ANALYSIS'):
    for x in READS.values():
        p = Path(x['path'])
        assert p.stat().st_size == x['bytes'] and p.stat().st_mtime_ns == x['mtime_ns_before']
        with p.open('rb') as h:
            x['sha256'] = hashlib.file_digest(h, 'sha256').hexdigest()
    dump(prefix + '_INPUT_PATHS_SHA256.json', list(READS.values()))
    dump(prefix + '_READ_ROLE_AUDIT.json', ACCESS)

def budget():
    elapsed = (time.monotonic() - START) * CPUS / 3600
    if elapsed > float(os.environ.get('TAHOE_STAGE_CPU_CAP', '15')):
        raise RuntimeError('STOP_REPORT_BUDGET: current workflow allocation reached the registered cap; no further steps')

def nz(p, keys=None):
    with np.load(reg(p), allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}

def weights(g):
    cnt = collections.Counter(g)
    return np.array([1 / (len(cnt) * cnt[x]) for x in g])
MET = []
roles = {r: collections.Counter() for r in ['TRAIN', 'VALID']}
groups = {}
for f in range(5):
    for role in roles:
        z = nz(GE / f'A_f{f}_{role}.npz', ['drug_id', 'component'])
        dd = dict(zip(z['drug_id'].astype(str), z['component'].astype(str)))
        for d, g in dd.items():
            roles[role][d] += 1
            if d in groups:
                assert groups[d] == g
            groups[d] = g
sizes = {r: collections.Counter((groups[d] for d in roles[r])) for r in roles}
NG = {r: len(sizes[r]) for r in roles}
ALLGROUPS = sorted(set(groups.values()))
GIX = {g: i for i, g in enumerate(ALLGROUPS)}

def qweights(drugs, role='VALID', simple=False):
    return np.array([(1 if simple else 1 / (NG[role] * sizes[role][groups[d]])) / roles[role][d] for d in drugs], float)

def panel(f, role, need_y=True):
    keys = ['drug_id', 'cell_id', 'component', 'x'] + (['y'] if need_y else [])
    z = nz(GE / f'A_f{f}_{role}.npz', keys)
    drugs = np.array(sorted(set(z['drug_id'].astype(str))))
    cells = np.array(sorted(set(z['cell_id'].astype(str))))
    lookup = {(str(c), str(d)): i for i, (c, d) in enumerate(zip(z['cell_id'], z['drug_id']))}
    ix = np.array([[lookup[c, d] for d in drugs] for c in cells])
    out = dict(drug=drugs, cell=cells, group=np.array([groups[d] for d in drugs]), X=z['x'][ix])
    if need_y:
        out['Y'] = z['y'][ix].astype(float)
    if role == 'VALID':
        v = nz(VEC / f'fold{f}_VALID.npz', ['B', 'drug', 'cells'] + (['Y'] if need_y else []))
        di = [list(v['drug']).index(d) for d in drugs]
        ci = [list(v['cells']).index(c) for c in cells]
        out['B'] = v['B'][np.ix_(ci, di)]
        if need_y:
            assert np.array_equal(v['Y'][np.ix_(ci, di)], out['Y'])
    ACCESS.append(dict(fold=f, role=role, asset='unified_panel', drugs=drugs.tolist(), TEST_rows=0, response=need_y))
    return out

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt(np.sum(x * x, -1) * np.sum(y * y, -1))
    return np.divide(np.sum(x * y, -1), den, out=np.full(den.shape, np.nan), where=den > 1e-24)

def dec(x, w):
    a = np.einsum('cdg,d->cg', x, w)
    mu = a.mean(0)
    b = x.mean(0) - mu
    return {'mu': mu, 'alpha': a - mu, 'beta': b, 'gamma': x - a[:, None] - b[None]}

def full_context(X):
    flat = np.ascontiguousarray(X.transpose(1, 0, 2).reshape(X.shape[1], -1))
    _, inv = np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel(), return_inverse=True)
    return inv

def beta_split(beta, X, w):
    ids = full_context(X)
    ctx = np.empty_like(beta)
    for k in np.unique(ids):
        ix = ids == k
        ctx[ix] = np.average(beta[ix], axis=0, weights=w[ix])
    return (ctx, beta - ctx)

def six_components(Y, X, w):
    sys.path.insert(0, str(MECH))
    from w_projection_spectral import WProjection
    d = dec(Y, w)
    gw = WProjection(X, w)(Y)
    bc, bw = beta_split(d['beta'], X, w)
    return {'mu': np.broadcast_to(d['mu'], Y.shape), 'alpha': np.broadcast_to(d['alpha'][:, None], Y.shape), 'beta_ctx': np.broadcast_to(bc[None], Y.shape), 'beta_w': np.broadcast_to(bw[None], Y.shape), 'gamma_pc': d['gamma'] - gw, 'gamma_w': gw}

def bootstrap_linear(t, columns, draws=2000):
    g = sorted(set(t.group))
    den = []
    num = []
    for group in g:
        s = t[t.group == group]
        den.append(s.weight.sum())
        num.append([np.sum(s.weight * s[k]) for k in columns])
    den = np.array(den)
    num = np.array(num)
    draw = np.random.default_rng(20261021).multinomial(len(g), np.ones(len(g)) / len(g), size=draws)
    val = num.sum(0) / den.sum()
    bs = draw @ num / (draw @ den)[:, None]
    return (dict(zip(columns, val)), dict(zip(columns, np.quantile(bs, [0.025, 0.975], axis=0).T)), bs)

def tier(r):
    return '稳定' if r >= 0.6 else '部分稳定' if r >= 0.3 else '基本不稳定'

def moments(x, y):
    return dict(PCC=pcc(x, y), energy_6=np.mean(x * x, -1), energy_14=np.mean(y * y, -1), cross=np.mean(x * y, -1), half_difference=np.mean((x - y) ** 2, -1) / 2, mean_6=x.mean(-1), mean_14=y.mean(-1))

def summarize_pairs(t, labels):
    cols = ['PCC', 'energy_6', 'energy_14', 'cross', 'half_difference', 'mean_6', 'mean_14']
    val, ci, bs = bootstrap_linear(t, cols)
    e1, e2, c = (val['energy_6'], val['energy_14'], val['cross'])
    rho = c / np.sqrt(e1 * e2)
    rc = bs[:, 3] / np.sqrt(bs[:, 1] * bs[:, 2])
    gc = (c - val['mean_6'] * val['mean_14']) / np.sqrt((e1 - val['mean_6'] ** 2) * (e2 - val['mean_14'] ** 2))
    gb = (bs[:, 3] - bs[:, 5] * bs[:, 6]) / np.sqrt((bs[:, 1] - bs[:, 5] ** 2) * (bs[:, 2] - bs[:, 6] ** 2))
    return dict(**labels, drugs=t.drug.nunique(), groups=t.group.nunique(), **val, **{k + '_lo': float(v[0]) for k, v in ci.items()}, **{k + '_hi': float(v[1]) for k, v in ci.items()}, global_PCC=gc, global_PCC_lo=np.quantile(gb, 0.025), global_PCC_hi=np.quantile(gb, 0.975), repeatability=rho, repeatability_lo=np.quantile(rc, 0.025), repeatability_hi=np.quantile(rc, 0.975), energy_ratio=e1 / e2, tier=tier(rho), energy_identity_error=abs((e1 + e2) / 2 - c - val['half_difference']))
CM = pd.read_parquet(reg(CAN / 'condition_manifest.parquet'))
MM = pd.read_parquet(reg(PL / 'tahoe_multiplate_plate_profile_manifest_978_v1.parquet'))
CC = pd.read_parquet(reg(CAN / 'control_context_manifest.parquet'))
GT = pd.read_csv(reg(GE / 'GENE_AXIS.tsv'), sep='\t', dtype=str)
POS = GT.tahoe_gene_position.to_numpy(int)
SORT = np.argsort(POS)
INV = np.argsort(SORT)
CACHE = {}
CONTROL = {}

def controls():
    if not CONTROL:
        with h5py.File(reg(CAN / 'controls.h5'), 'r') as h:
            for t in CC.itertuples():
                CONTROL[int(t.plate), str(t.cell_line)] = h['controls'][int(t.control_row), POS[SORT]][INV].astype(float)
    return CONTROL

def physical_metadata(drugs, dose=None):
    s = CM[CM.drug.isin(drugs)].copy()
    if dose is not None:
        s = s[s.dose == dose]
    one = s[s.source_plate_profiles == 1].copy()
    one['plate'] = one.plates.astype(int)
    one['source_kind'] = 'single'
    one['value_row'] = -1
    multi = MM[MM.condition_row.isin(s.loc[s.source_plate_profiles > 1, 'condition_row'])].copy()
    multi['source_kind'] = 'multi'
    multi['treated_shard'] = -1
    multi['treated_row_in_shard'] = -1
    cols = ['condition_row', 'cell_line', 'drug', 'dose', 'plate', 'source_kind', 'value_row', 'treated_shard', 'treated_row_in_shard']
    return pd.concat([one[cols], multi[cols]], ignore_index=True)

def physical_values(meta, f, role, allowed):
    assert set(meta.drug) <= set(allowed)
    need = meta.copy()
    need['key'] = [(r.source_kind, int(r.value_row) if r.source_kind == 'multi' else int(r.condition_row)) for r in need.itertuples()]
    missing = need[~need.key.isin(CACHE)]
    for kind in ['single', 'multi']:
        z = missing[missing.source_kind == kind]
        shards = sorted(z.treated_shard.unique()) if kind == 'single' else [-1]
        for shard in shards:
            ss = z[z.treated_shard == shard] if kind == 'single' else z
            if ss.empty:
                continue
            path = CAN / f'treated_{int(shard):02d}-of-16.h5' if kind == 'single' else PL / 'tahoe_multiplate_plate_profiles_978_v1.h5'
            rowcol = 'treated_row_in_shard' if kind == 'single' else 'value_row'
            ss = ss.sort_values(rowcol)
            with h5py.File(reg(path), 'r') as h:
                for start in range(0, len(ss), 64):
                    batch = ss.iloc[start:start + 64]
                    rows = batch[rowcol].to_numpy(int)
                    assert len(rows) == len(np.unique(rows))
                    vals = h['treated' if kind == 'single' else 'X2'][rows]
                    if kind == 'single':
                        vals = vals[:, POS]
                    for row, v in zip(batch.itertuples(), vals):
                        CACHE[row.key] = v.astype(np.float64)
            budget()
    ACCESS.append(dict(fold=f, role=role, asset='physical_plate', drugs=sorted(set(meta.drug)), rows=len(meta), new_rows=len(missing), TEST_rows=0))
    ctr = controls()
    return np.array([CACHE[r.key] - ctr[int(r.plate), str(r.cell_line)] for r in need.itertuples()])

def paired(f, role, drugs, cells, dose=5.0):
    meta = physical_metadata(drugs, dose)
    shared = []
    for d in sorted(drugs):
        if all((set(meta.loc[(meta.drug == d) & (meta.plate == p), 'cell_line']) == set(cells) for p in [6, 14])):
            shared.append(d)
    if not shared:
        return None
    meta = meta[meta.drug.isin(shared) & meta.plate.isin([6, 14])]
    values = physical_values(meta, f, role, drugs)
    look = {(int(r.plate), r.cell_line, r.drug): i for i, r in enumerate(meta.itertuples())}
    ys = {p: values[[look[p, c, d] for c in cells for d in shared]].reshape(len(cells), len(shared), 978) for p in [6, 14]}
    xs = {p: np.broadcast_to(np.array([controls()[p, c] for c in cells])[:, None], ys[p].shape).copy().astype('float32') for p in [6, 14]}
    return (np.array(shared), ys, xs)

def infer_B(f, X):
    import torch
    from torch import nn
    torch.set_num_threads(CPUS)
    src = reg(BASE / 'scripts/tb_train.py')
    nodes = [n for n in ast.parse(src.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'BasalMLP']
    ns = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec'), ns)
    ck = torch.load(reg(BASE / f'runs/A_f{f}_3H2048/selected.pt'), map_location='cpu', weights_only=False)
    assert ck['seed'] == 1337
    net = ns['BasalMLP']()
    net.load_state_dict(ck['model'])
    net.eval()
    flat = X.reshape(-1, 978).astype('float32')
    unique, ix = np.unique(flat, axis=0, return_inverse=True)
    with torch.inference_mode():
        out = net(torch.from_numpy(unique)).numpy()[ix].reshape(X.shape).astype(float)
    return out

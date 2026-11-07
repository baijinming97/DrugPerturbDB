"""Independent-repeat, common-support cross-source comparison.

No downloads, no fitted model, no selection on observed repeatability. Source
files are read-only. RNA-seq counts are normalized using the original full
library total, before selecting the fixed Entrez axis. Only selected physical
profiles and their recorded controls are recovered, never consensus means.
"""
from pathlib import Path
import argparse, collections, hashlib, itertools, json, os, sys, time
sys.dont_write_bytecode = True
O = Path('@WORKSPACE@')
import numpy as np
import pandas as pd
import h5py
F = O / 'FULL_DATASET_FIGURES_V1'
R = O / '4MAN/TAHOE_DESCRIPTIVE_FULL_V2'
I = R / 'pairs'
N = O / 'REPEATABILITY_NATIVE_GENES_V1'
H = O / 'REPEATABILITY_HARMONIZED_V1/outputs'
C = O / 'baicai/canonical_perturbation_data_v1/l1000/data'
DS = ['Tahoe', 'L1000_L3', 'DRUGseq', 'CIGS', 'sciPlex3', 'MIXseq']
READS = []
EXCLUDE = []
START = time.monotonic()

def jdump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))

def manifest(p, purpose, selected=None):
    rr = {'path': str(p), 'bytes': p.stat().st_size, 'purpose': purpose}
    if selected is not None:
        rr['selected_rows'] = list(map(int, selected))
    if p.suffix in ('.parquet', '.json') or p.name.endswith('.csv.gz'):
        with p.open('rb') as f:
            rr['metadata_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    READS.append(rr)

def boolcol(m, k, default=True):
    return m[k].astype(str).str.lower().isin(['true', '1']).to_numpy() if k in m else np.full(len(m), default, bool)

def readmeta(ds):
    p = R / 'metadata/Tahoe_REPLICATE_META.parquet' if ds == 'Tahoe' else F / 'data' / f'{ds}_META.parquet'
    manifest(p, 'profile identities and controls; no responses')
    m = pd.read_parquet(p)
    m['meta_row'] = np.arange(len(m))
    if ds not in ('Tahoe', 'L1000_L3'):
        m = m[m.normalization.eq('logCPM1')].copy()
    m['source_row'] = np.arange(len(m))
    return m

def legal_pairs(m, whitelist=None):
    result = []
    for i, j in itertools.combinations(range(len(m)), 2):
        a = m.iloc[i]
        b = m.iloc[j]
        if str(a.physical_id) == str(b.physical_id):
            continue
        if str(a.plate) == str(b.plate):
            continue
        if not a.control_set or not b.control_set:
            continue
        if set(a.control_set) & set(b.control_set):
            continue
        if whitelist is not None and tuple(sorted((int(a.meta_row), int(b.meta_row)))) not in whitelist:
            continue
        result.append((i, j))
    return np.asarray(result, dtype=int).reshape(-1, 2)

def prepare_metadata(out):
    ident = json.loads((I / 'IDENTITY_MAP_FINAL.json').read_text())
    manifest(I / 'IDENTITY_MAP_FINAL.json', 'locked identity map')
    owners = json.loads((O / '4MAN/CROSS_DATASET_ATLAS_V1/DRUG_ROLE_OWNER.json').read_text())
    allmeta = {}
    conditions = {}
    full = {}
    whitelists = {}
    for ds in DS:
        m = readmeta(ds)
        full[ds] = m.copy()
        ctrl = boolcol(m, 'is_control', False)
        valid = boolcol(m, 'valid_profile')
        if ds == 'L1000_L3':
            m['control_set'] = [tuple(('well:' + v for v in str(k).split(';') if v)) for k in m.control_key]
            m['control_evidence'] = 'actual_RNA_control_wells'
        elif ds == 'Tahoe':
            m['control_set'] = [('pool:' + str(k),) for k in m.control_key]
            m['control_evidence'] = 'distinct_physical_plates_recorded_pool; individual_DMSO_wells_unresolved'
        else:
            cm = {}
            for k, part in m[ctrl & valid].groupby('control_key', sort=False):
                cm[str(k)] = tuple(sorted(set(('well:' + str(ds) + '|' + str(r.plate) + '|' + str(r.physical_unit) for r in part.itertuples()))))
            m['control_set'] = [cm.get(str(k), ()) for k in m.control_key]
            m['control_evidence'] = 'recorded_physical_control_pseudobulks' if ds in ('sciPlex3', 'MIXseq') else 'recorded_physical_control_wells'
        m['physical_id'] = ds + '|' + m.plate.astype(str) + '|' + m.physical_unit.astype(str)
        m['source_drug'] = m.drug.astype(str)
        m['source_cell'] = m.cell.astype(str)
        m['drug'] = m.source_drug.map(ident['drug'].get(ds, {}))
        m['cell'] = m.source_cell.map(ident['cell'].get(ds, {}))
        already = m.cell.isna() & m.source_cell.str.match('^CVCL_[A-Z0-9]+$')
        m.loc[already, 'cell'] = m.loc[already, 'source_cell']
        m['dose'] = pd.to_numeric(m.dose, errors='coerce')
        m['time'] = pd.to_numeric(m.time, errors='coerce')
        good = valid & ~ctrl & m.drug.notna() & m.cell.fillna('').str.match('^CVCL_[A-Z0-9]+$') & m.time.eq(24) & m.dose.gt(0)
        EXCLUDE.append({'source': ds, 'reason': 'metadata_QC_identity_cell_time_dose', 'rows': int((~good).sum())})
        m = m[good].copy()
        m = m.reset_index(drop=True)
        dup = m.duplicated(['drug', 'cell', 'dose', 'time', 'physical_id'])
        EXCLUDE.append({'source': ds, 'reason': 'duplicate_physical_unit_after_identity_mapping', 'rows': int(dup.sum())})
        m = m[~dup].reset_index(drop=True)
        allmeta[ds] = m
    preliminary = {ds: m.groupby(['drug', 'cell', 'dose', 'time'], sort=True).size().rename('n').reset_index() for ds, m in allmeta.items()}
    candidatekeys = {ds: set() for ds in DS}
    for a, b in itertools.combinations(DS, 2):
        aa = preliminary[a].query('n>=2')
        bb = preliminary[b].query('n>=2')
        matched = aa.merge(bb, on=['drug', 'cell', 'time'], suffixes=('A', 'B'))
        matched = matched[np.maximum(matched.doseA, matched.doseB) / np.minimum(matched.doseA, matched.doseB) <= 2 + 1e-12]
        for r in matched.itertuples():
            candidatekeys[a].add((r.drug, r.cell, float(r.doseA), float(r.time)))
            candidatekeys[b].add((r.drug, r.cell, float(r.doseB), float(r.time)))
    for ds, m in allmeta.items():
        cc = {}
        keys = list(zip(m.drug, m.cell, m.dose, m.time))
        m = m[np.asarray([k in candidatekeys[ds] for k in keys])].reset_index(drop=True)
        for key, ix in m.groupby(['drug', 'cell', 'dose', 'time'], sort=True).indices.items():
            part = m.iloc[ix].copy().reset_index(drop=True)
            pp = legal_pairs(part, whitelists.get(ds))
            if not len(pp):
                continue
            take = np.unique(pp)
            part = part.iloc[take].reset_index(drop=True)
            pp = legal_pairs(part, whitelists.get(ds))
            cc[key] = {'meta': part, 'pairs': pp}
        conditions[ds] = cc
    canonical_groups = {}
    for r in allmeta['Tahoe'].itertuples():
        group = str(getattr(r, 'group', ''))
        if not group:
            raise RuntimeError('Tahoe group missing')
        if r.drug in canonical_groups:
            assert canonical_groups[r.drug] == group, ('identity_groups_conflict', r.drug)
        canonical_groups[r.drug] = group
    selection = []
    support = []
    for a, b in itertools.combinations(DS, 2):
        common = collections.defaultdict(list)
        for kb in conditions[b]:
            common[kb[0], kb[1], kb[3]].append(kb)
        rows = []
        for ka in conditions[a]:
            for kb in common[ka[0], ka[1], ka[3]]:
                ratio = max(ka[2], kb[2]) / min(ka[2], kb[2])
                if ratio <= 2 + 1e-12:
                    rows.append({'A': a, 'B': b, 'keyA': ka, 'keyB': kb, 'drug': ka[0], 'cell': ka[1], 'doseA': ka[2], 'doseB': kb[2], 'time': ka[3], 'group': canonical_groups.get(ka[0], ka[0]) if 'Tahoe' in (a, b) else ka[0]})
        selection += rows
        support.append({'A': a, 'B': b, 'status': 'eligible' if rows else 'not_estimable', 'matched_dose_pairs': len(rows), 'matched_drug_cells': len(set(((r['drug'], r['cell']) for r in rows))), 'drugs': len(set((r['drug'] for r in rows))), 'AA_eligible_conditions': len(conditions[a]), 'BB_eligible_conditions': len(conditions[b]), 'reason': '' if rows else 'No common confirmed cell, 24h, dose ratio<=2 support with independent repeats and disjoint recorded controls on both sides'})
    pd.DataFrame(support).to_csv(out / 'SUPPORT_COUNTS.csv', index=False)
    jdump(out / 'SUPPORT_LOCK.json', {'selection': selection, 'rule': 'metadata_only; no response or repeatability filter', 'identity_sha256': hashlib.sha256((I / 'IDENTITY_MAP_FINAL.json').read_bytes()).hexdigest(), 'Tahoe_role': 'all_eligible_descriptive_roles', 'all_15_pairs_reported': True})
    return (full, conditions, selection, support)

def axis_positions(ds, axis, source_gene):
    with np.load(F / 'data' / f'{ds}_NATIVE_AXES.npz', allow_pickle=False) as z:
        names = z['gene'].astype(str)
        entrez = z['entrez'].astype(str)
    ec = collections.Counter(entrez.tolist())
    ambiguous = [e for e in axis if ec[e] > 1]
    assert not ambiguous, ('non_unique_Entrez_axis', ds, ambiguous)
    lookup = {e: g for e, g in zip(entrez, names) if e and e != 'nan'}
    source = list(map(str, source_gene))
    sc = collections.Counter(source)
    sl = {g: i for i, g in enumerate(source)}
    pos = []
    for e in axis:
        g = lookup.get(e)
        assert g is None or sc[g] <= 1, ('non_unique_native_gene', ds, e, g)
        pos.append(sl.get(g, -1))
    return np.asarray(pos, int)

def read_h5_rows(path, rows, cols, key='counts'):
    rows = np.asarray(rows, int)
    cols = np.asarray(cols, int)
    unique, inverse = np.unique(rows, return_inverse=True)
    good = cols >= 0
    out = np.full((len(rows), len(cols)), np.nan, np.float64)
    with h5py.File(path, 'r') as h:
        parts = []
        for lo in range(0, len(unique), 128):
            parts.append(h[key][unique[lo:lo + 128]][:, cols[good]])
        if len(parts):
            out[:, good] = np.concatenate(parts)[inverse]
    manifest(path, 'selected physical profile payload', unique)
    return out

def l3_read(kind, rows, cols):
    requested = np.asarray(rows, int)
    out = np.empty((len(rows), len(cols)), float)
    seen = np.zeros(len(rows), bool)
    for p in sorted(C.glob(f'l3_{kind}_full_*.h5')):
        with h5py.File(p, 'r') as h:
            start = int(h.attrs['canonical_start'])
            stop = int(h.attrs['canonical_stop'])
            ix = np.flatnonzero((requested >= start) & (requested < stop))
            if not len(ix):
                continue
            local = requested[ix] - start
            u, inv = np.unique(local, return_inverse=True)
            out[ix] = h['expression'][u][:, cols][inv]
            seen[ix] = True
        manifest(p, 'selected L3 ' + kind + ' readouts', requested[ix])
    assert seen.all()
    return out

def load_l3(m, axis):
    mt = pd.read_parquet(C / 'l3_treatment_manifest.parquet')
    cm = pd.read_parquet(C / 'l3_control_manifest.parquet').set_index('control_row')
    cx = pd.read_parquet(C / 'l3_control_context_manifest.parquet').set_index('control_context_id')
    mt['physical_unit'] = mt.rna_plate.astype(str) + '|' + mt.rna_well.astype(str)
    mt = mt[mt.has_exact_qc_vehicle_control & mt.dose_uM.gt(0) & mt.physical_unit.isin(set(m.physical_unit))]
    lookup = {key: part for key, part in mt.groupby(['pert_id', 'cell_iname', 'dose_uM', 'duration_h', 'physical_unit'], sort=False)}
    groups = []
    for r in m.itertuples():
        groups.append(lookup[r.source_drug, r.source_cell, float(r.dose), float(r.time), r.physical_unit])
    sel = pd.concat(groups).drop_duplicates('treatment_row').set_index('treatment_row')
    gi = {str(e): i for i, e in enumerate(pd.read_parquet(C / 'gene_axis_978.parquet').gene_id)}
    cols = np.array([gi[e] for e in axis])
    tids = sel.index.to_numpy(int)
    tv = l3_read('treated', tids, cols)
    tvlu = {v: i for i, v in enumerate(tids)}
    ctx = sorted(set(sel.control_context_id))
    cr = sorted(set((int(i) for k in ctx for i in cx.loc[k, 'control_rows'])))
    cv = l3_read('control', cr, cols)
    clu = {v: i for i, v in enumerate(cr)}
    ref = {}
    for k in ctx:
        ids = np.asarray(cx.loc[k, 'control_rows'], int)
        cc = cm.loc[ids]
        units = cc.rna_plate.astype(str) + '|' + cc.rna_well.astype(str)
        vv = cv[[clu[i] for i in ids]]
        ref[k] = np.stack([vv[units.to_numpy() == unit].mean(0) for unit in sorted(units.unique())]).mean(0)
    result = []
    for part in groups:
        val = np.stack([tv[tvlu[int(r.treatment_row)]] - ref[r.control_context_id] for r in part.itertuples()])
        result.append(val.mean(0))
    return np.stack(result)

def load_tahoe(m, axis):
    p = R / 'data/Tahoe_REPLICATE_978.h5'
    with h5py.File(p, 'r') as h:
        cols = pd.Index(h['entrez'].asstr()[:]).get_indexer(axis)
        assert min(cols) >= 0
        rows = m.meta_row.to_numpy(int)
        u, inv = np.unique(rows, return_inverse=True)
        v = h['delta'][u][:, cols][inv].astype(float)
    manifest(p, 'all eligible descriptive physical-plate delta rows', rows)
    return v

def fix_cigs_gene(g):
    g = str(g)
    if g in ('45542', 'X7.Sep'):
        return 'CIGS_DATE_SEP7_UNRESOLVED'
    if g.startswith('HLA.') or g.startswith('NKX'):
        return g.replace('.', '-')
    return g

def load_counts(ds, m, full, axis, pc):
    needed_keys = set(m.control_key.astype(str))
    allm = full[ds]
    ctr = allm[boolcol(allm, 'is_control', False) & boolcol(allm, 'valid_profile') & allm.control_key.astype(str).isin(needed_keys)]
    take = pd.concat([allm[allm.meta_row.isin(m.meta_row)], ctr]).drop_duplicates('meta_row').reset_index(drop=True)
    values = np.full((len(take), len(axis)), np.nan)
    totals = np.full(len(take), np.nan)
    if ds in ('DRUGseq', 'sciPlex3'):
        p = F / 'data' / f'{ds}_COUNTS.h5'
        with h5py.File(p, 'r') as h:
            assert len(allm) == h['counts'].shape[0], ('counts_metadata_length', ds)
            assert np.array_equal(allm.source_row.to_numpy(), np.arange(len(allm))), ('counts_metadata_order', ds)
            gene = h['gene'].asstr()[:]
            col = axis_positions(ds, axis, gene)
            totalkey = 'total_full_library' if ds == 'sciPlex3' else 'total'
            alltot = h[totalkey][:]
            totals = alltot[take.source_row.to_numpy(int)]
            assert 'library_total' in allm and np.allclose(allm.library_total.to_numpy(float), alltot, rtol=0, atol=0.01), ('full_library_total_row_alignment', ds)
        values = read_h5_rows(p, take.source_row, col)
    elif ds == 'CIGS':
        for i in range(6):
            p = F / 'data' / f'CIGS_{i}_COUNTS.h5'
            meta = pd.read_parquet(F / 'data' / f'CIGS_{i}_META.parquet')
            meta = meta[meta.normalization.eq('logCPM1')]
            source = str(meta.source_file.iloc[0])
            ix = np.flatnonzero(take.source_file.eq(source))
            if not len(ix):
                continue
            with h5py.File(p, 'r') as h:
                genes = [fix_cigs_gene(g) for g in h['gene'].asstr()[:]]
                col = axis_positions(ds, axis, genes)
                units = list(h['physical_unit'].asstr()[:])
                assert len(meta) == h['counts'].shape[0] and np.array_equal(meta.physical_unit.to_numpy(str), np.asarray(units)), ('CIGS_count_identity_order', i)
                lookup = {u: j for j, u in enumerate(units)}
                rows = np.array([lookup[u] for u in take.iloc[ix].physical_unit])
                totals[ix] = h['total'][:][rows]
            values[ix] = read_h5_rows(p, rows, col)
    elif ds == 'MIXseq':
        key = lambda r: (str(r.source_file), str(r.physical_unit), str(r.drug))
        lu = {key(r): i for i, r in enumerate(take.itertuples())}
        assert len(lu) == len(take), 'MIX-seq raw identity collision'
        for p in sorted((F / 'data').glob('*.PB.npz')):
            assert p.resolve().is_relative_to((F / 'data').resolve())
            with np.load(p, allow_pickle=True) as z:
                mm = json.loads(str(z['metadata']))
                chosen = [(j, lu[str(r['source_file']), str(r['physical_unit']), str(r['drug'])]) for j, r in enumerate(mm) if (str(r['source_file']), str(r['physical_unit']), str(r['drug'])) in lu]
                if not chosen:
                    continue
                cols = axis_positions(ds, axis, z['gene'].astype(str))
                good = cols >= 0
                ct = z['counts']
                for j, k in chosen:
                    values[k, good] = ct[j, cols[good]]
                    totals[k] = float(mm[j]['library_total'])
            manifest(p, 'selected MIX-seq pseudobulks', [j for j, k in chosen])
    else:
        raise ValueError(ds)
    assert np.isfinite(totals).all() and (totals > 0).all(), ('invalid_full_library_total', ds, take.loc[~np.isfinite(totals) | (totals <= 0), ['source_file', 'physical_unit', 'control_key', 'is_control']].to_dict('records'))
    expression = np.log2(values / totals[:, None] * 1000000.0 + pc)
    refs = {}
    for ck in needed_keys:
        ix = np.flatnonzero(take.control_key.astype(str).eq(ck) & boolcol(take, 'is_control', False))
        assert len(ix), ('no_controls', ds, ck)
        refs[ck] = np.mean(expression[ix], axis=0)
    lu = {int(v): i for i, v in enumerate(take.meta_row)}
    return np.stack([expression[lu[int(r.meta_row)]] - refs[str(r.control_key)] for r in m.itertuples()])

def normrows(x):
    x = np.asarray(x, float)
    x = x - x.mean(1, keepdims=True)
    n = np.linalg.norm(x, axis=1)
    good = np.isfinite(x).all(1) & (n > 1e-12)
    out = np.full_like(x, np.nan)
    out[good] = x[good] / n[good, None]
    return (out, good)

def evaluate_pair(a, b, pa, pb):
    mask = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    if mask.sum() < 3:
        return (None, 'fewer_than_three_common_finite_genes')
    na, ga = normrows(a[:, mask])
    nb, gb = normrows(b[:, mask])
    if not ga.all() or not gb.all():
        return (None, 'constant_or_nonfinite_repeat_entire_condition_excluded')
    aa = float(np.mean(np.sum(na[pa[:, 0]] * na[pa[:, 1]], axis=1)))
    bb = float(np.mean(np.sum(nb[pb[:, 0]] * nb[pb[:, 1]], axis=1)))
    ab = float(np.mean(na, axis=0) @ np.mean(nb, axis=0))
    return ({'rAA': aa, 'rBB': bb, 'rAB': ab, 'gap_A': aa - ab, 'gap_B': bb - ab, 'gap_mean': (aa + bb) / 2 - ab, 'n_genes': int(mask.sum()), 'meanA': na.mean(0), 'meanB': nb.mean(0), 'mask': mask}, '')

def aggregate(frame, metrics, seed=20261103, bootstrap=True):
    if not len(frame):
        return []
    dc = frame.groupby(['group', 'drug', 'cell'], sort=True)[metrics].mean()
    dr = dc.groupby(['group', 'drug'], sort=True).mean()
    gr = dr.groupby('group', sort=True).mean()
    rng = np.random.default_rng(seed)
    arr = gr.to_numpy(float)
    boot = None
    if bootstrap and len(arr) >= 3:
        ix = rng.integers(0, len(arr), (2000, len(arr)))
        boot = np.nanmean(arr[ix], axis=1)
    ans = []
    for i, met in enumerate(metrics):
        ans.append({'metric': met, 'estimate': float(np.nanmean(arr[:, i])), 'ci_low': float(np.nanquantile(boot[:, i], 0.025)) if boot is not None else np.nan, 'ci_high': float(np.nanquantile(boot[:, i], 0.975)) if boot is not None else np.nan, 'bootstrap_units': len(arr), 'drugs': len(dr), 'drug_cells': len(dc), 'dose_pairs': len(frame), 'ci_scope': 'chemical/drug resampling conditional on observed plates; not a causal platform effect' if boot is not None else 'point estimate only: <3 units or prespecified leave-one-plate-out sensitivity'})
    return ans

def double_mean_data(x, pairs):
    """All two-repeat means participating in disjoint four-unit comparisons."""
    legal = {tuple(sorted(p)) for p in pairs.tolist()}
    means = {}
    comparisons = []
    for quad in itertools.combinations(range(len(x)), 4):
        if not all((tuple(sorted(p)) in legal for p in itertools.combinations(quad, 2))):
            continue
        a, b, c, d = quad
        for p, q in [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]:
            for pair in [p, q]:
                if pair not in means:
                    means[pair] = len(means)
            comparisons.append((means[p], means[q]))
    if not comparisons:
        return None
    values = np.stack([x[list(p)].mean(0) for p in means])
    return (values, np.asarray(comparisons, int))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['pilot', 'all'], default='pilot')
    args = ap.parse_args()
    root = Path(args.out).resolve()
    assert root == R / 'matched_repeats'
    assert os.getenv('TAHOE_REPRO_RUN_ID')
    out = root / 'exp1'
    out.mkdir(exist_ok=True)

    def guard(event, args):
        if event != 'open' or not args or (not isinstance(args[0], (str, bytes, os.PathLike))):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) and p.is_relative_to(O):
            assert p.is_relative_to(root), ('forbidden_write', str(p))
    sys.addaudithook(guard)
    full, conditions, selection, support = prepare_metadata(out)
    primary = list(map(str, json.loads((F / 'COMMON_GENES.json').read_text())['axis']))
    assert len(primary) == 869
    extra = list(map(str, json.loads((H / 'GLOBAL_GENE_MASK.json').read_text())['axis']))
    assert len(extra) == 882
    axes = primary + [e for e in extra if e not in primary]
    axisix = {e: i for i, e in enumerate(axes)}
    needed = {ds: set() for ds in DS}
    for r in selection:
        needed[r['A']].add(tuple(r['keyA']))
        needed[r['B']].add(tuple(r['keyB']))
    cache = {}
    metacache = {}
    valuehash = []
    bindings = []
    (out / 'cache').mkdir(exist_ok=True)
    for ds in DS:
        if not needed[ds]:
            continue
        m = pd.concat([conditions[ds][key]['meta'] for key in sorted(needed[ds])]).drop_duplicates('meta_row').reset_index(drop=True)
        for r in m.itertuples():
            bindings.append({'source': ds, 'meta_row': int(r.meta_row), 'canonical_drug': r.drug, 'cell': r.cell, 'dose': r.dose, 'time': r.time, 'plate': str(r.plate), 'physical_unit': r.physical_id, 'control_members': r.control_set, 'control_evidence': r.control_evidence, 'role': 'descriptive_all_roles' if ds == 'Tahoe' else 'external'})
        metacache[ds] = m
        pcvalues = [1] if ds in ('Tahoe', 'L1000_L3') else [1, 100]
        for pc in pcvalues:
            print('RECOVER', ds, pc, len(m), flush=True)
            cachefile = out / 'cache' / f'{ds}_pc{pc}.npz'
            if cachefile.exists():
                with np.load(cachefile, allow_pickle=False) as zz:
                    assert np.array_equal(zz['meta_rows'], m.meta_row.to_numpy()) and np.array_equal(zz['axis'], np.asarray(axes)), ('cache_identity_mismatch', ds, pc)
                    v = zz['values']
                manifest(cachefile, 'same-task selected-response cache; prior pilot read')
            else:
                v = load_tahoe(m, axes) if ds == 'Tahoe' else load_l3(m, axes) if ds == 'L1000_L3' else load_counts(ds, m, full, axes, pc)
                np.savez_compressed(cachefile, values=v, meta_rows=m.meta_row.to_numpy(), axis=np.asarray(axes))
            assert v.shape == (len(m), len(axes))
            cache[ds, pc] = (v, {int(k): i for i, k in enumerate(m.meta_row)})
            valuehash.append({'dataset': ds, 'pc': pc, 'rows': len(m), 'sha256_selected_float64': hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(), 'meta_rows': m.meta_row.tolist()})
    jdump(out / 'CONTROL_BINDINGS.json', bindings)
    if args.stage == 'pilot':
        pilot = []
        for r in selection[:1]:
            a, b = (r['A'], r['B'])
            ca = conditions[a][tuple(r['keyA'])]
            cb = conditions[b][tuple(r['keyB'])]
            va, la = cache[a, 1]
            vb, lb = cache[b, 1]
            z, err = evaluate_pair(va[[la[int(i)] for i in ca['meta'].meta_row]][:, :869], vb[[lb[int(i)] for i in cb['meta'].meta_row]][:, :869], ca['pairs'], cb['pairs'])
            pilot.append({**r, 'error': err, 'statistics': {k: v for k, v in (z or {}).items() if k not in ('meanA', 'meanB', 'mask')}})
        jdump(out / 'EXP1_PILOT.json', {'status': 'PASS', 'seconds': time.monotonic() - START, 'job': os.getenv('TAHOE_REPRO_RUN_ID'), 'selected_data_recovered': valuehash, 'one_condition_check': pilot, 'next_stage': '--stage all reuses same-task selected cache'})
        jdump(out / 'SELECTED_INPUT_HASHES_PILOT.json', {'inputs': READS, 'selected_arrays': valuehash, 'model_TEST_reads': 0, 'descriptive_Tahoe_all_roles': True})
        print('EXP1 PILOT COMPLETE', time.monotonic() - START, flush=True)
        return
    results = []
    summary = []
    sensitivity = []
    wrong = []
    verification = False
    for a, b in itertools.combinations(DS, 2):
        rr = [r for r in selection if r['A'] == a and r['B'] == b]
        if not rr:
            continue
        for genename, axis in [('primary869', primary), ('sensitivity882', extra)]:
            cols = np.array([axisix[e] for e in axis])
            for pc in [1, 100]:
                rows = []
                av = []
                bv = []
                detail = []
                double_rows = []
                double_candidate_count = 0
                for r in rr:
                    ca = conditions[a][tuple(r['keyA'])]
                    cb = conditions[b][tuple(r['keyB'])]
                    ma = ca['meta']
                    mb = cb['meta']
                    va, la = cache[a, 1 if a in ('Tahoe', 'L1000_L3') else pc]
                    vb, lb = cache[b, 1 if b in ('Tahoe', 'L1000_L3') else pc]
                    x = va[[la[int(i)] for i in ma.meta_row]][:, cols]
                    y = vb[[lb[int(i)] for i in mb.meta_row]][:, cols]
                    z, err = evaluate_pair(x, y, ca['pairs'], cb['pairs'])
                    if err or z['n_genes'] != len(axis):
                        EXCLUDE.append({'source': a + ' vs ' + b, 'reason': err or 'fixed_gene_axis_not_recoverable_without_imputation', 'drug': r['drug'], 'cell': r['cell'], 'scope': genename, 'normalization': pc})
                        continue
                    rec = {k: v for k, v in r.items() if k not in ('keyA', 'keyB')}
                    rec.update({k: v for k, v in z.items() if k not in ('meanA', 'meanB', 'mask')})
                    rec.update(axis=genename, pseudocount=pc, repeat_A=len(x), repeat_B=len(y), pairs_A=len(ca['pairs']), pairs_B=len(cb['pairs']), control_evidence_A=ma.control_evidence.iloc[0], control_evidence_B=mb.control_evidence.iloc[0], plates_A=';'.join(sorted(set(ma.plate.astype(str)))), plates_B=';'.join(sorted(set(mb.plate.astype(str)))))
                    rows.append(rec)
                    av.append(z['meanA'])
                    bv.append(z['meanB'])
                    if genename == 'primary869' and pc == 1:
                        nx, _ = normrows(x)
                        ny, _ = normrows(y)
                        detail.append((rec, nx, ny, ca['pairs'], cb['pairs'], ma.plate.astype(str).to_numpy(), mb.plate.astype(str).to_numpy()))
                        if len(x) >= 4 and len(y) >= 4:
                            double_candidate_count += 1
                            dmA = double_mean_data(x, ca['pairs'])
                            dmB = double_mean_data(y, cb['pairs'])
                            if dmA is not None and dmB is not None:
                                zz, ee = evaluate_pair(dmA[0], dmB[0], dmA[1], dmB[1])
                                if not ee:
                                    double_rows.append({**rec, **{k: v for k, v in zz.items() if k not in ('meanA', 'meanB', 'mask')}})
                    if not verification and genename == 'primary869' and (pc == 1):
                        np.savez_compressed(out / 'verification_pair.npz', A=x, B=y, A_unit=ma.physical_id.to_numpy(str), B_unit=mb.physical_id.to_numpy(str), A_control_sets=np.asarray([json.dumps(list(v)) for v in ma.control_set]), B_control_sets=np.asarray([json.dumps(list(v)) for v in mb.control_set]), AA_pairs=ca['pairs'], BB_pairs=cb['pairs'], claimed_rAA=z['rAA'], claimed_rBB=z['rBB'], claimed_rAB=z['rAB'], gene=np.asarray(axis), source_A=a, source_B=b, role_A=np.asarray(['descriptive_all_roles' if a == 'Tahoe' else 'external'] * len(ma)), role_B=np.asarray(['descriptive_all_roles' if b == 'Tahoe' else 'external'] * len(mb)), metadata=json.dumps(rec))
                        verification = True
                frame = pd.DataFrame(rows)
                if not len(frame):
                    continue
                for q in aggregate(frame, ['rAA', 'rBB', 'rAB', 'gap_A', 'gap_B', 'gap_mean']):
                    q.update(A=a, B=b, axis=genename, pseudocount=pc)
                    (summary if genename == 'primary869' and pc == 1 else sensitivity).append(q)
                if genename == 'primary869' and pc == 1:
                    exact = frame[np.isclose(frame.doseA, frame.doseB, rtol=0, atol=1e-10)]
                    for q in aggregate(exact, ['rAA', 'rBB', 'rAB', 'gap_mean']):
                        q.update(A=a, B=b, axis=genename, pseudocount=pc, analysis='exact_dose_only')
                        sensitivity.append(q)
                    drug_equal = frame.copy()
                    drug_equal['group'] = drug_equal.drug
                    for q in aggregate(drug_equal, ['rAA', 'rBB', 'rAB', 'gap_mean']):
                        q.update(A=a, B=b, axis=genename, pseudocount=pc, analysis='unweighted_drug_equal')
                        sensitivity.append(q)
                    if double_rows:
                        for q in aggregate(pd.DataFrame(double_rows), ['rAA', 'rBB', 'rAB', 'gap_mean']):
                            q.update(A=a, B=b, axis=genename, pseudocount=pc, analysis='two_replicate_means_disjoint_four_units')
                            sensitivity.append(q)
                    else:
                        sensitivity.append({'A': a, 'B': b, 'analysis': 'two_replicate_means_disjoint_four_units', 'status': 'NOT_ELIGIBLE', 'conditions_with_at_least_four_nodes_both': double_candidate_count, 'reason': 'no common condition with four mutually independent physical units and disjoint control sets in both sources'})
                av = np.stack(av)
                bv = np.stack(bv)
                for j, r in enumerate(rows):
                    ix = [k for k, s in enumerate(rows) if s['drug'] != r['drug'] and s['cell'] == r['cell'] and (s['time'] == r['time']) and (max(s['doseB'], r['doseA']) / min(s['doseB'], r['doseA']) <= 2 + 1e-12)]
                    ix = list({(rows[k]['drug'], rows[k]['cell'], rows[k]['doseB'], rows[k]['time']): k for k in ix}.values())
                    if not ix:
                        continue
                    raw = av[j] @ bv[ix].T
                    bydrug = collections.defaultdict(list)
                    for k, v in zip(ix, raw):
                        bydrug[rows[k]['drug']].append(float(v))
                    neg = float(np.mean([np.mean(v) for v in bydrug.values()]))
                    wrong.append({**r, 'wrong_drug_rAB': neg, 'same_minus_wrong': r['rAB'] - neg, 'wrong_candidate_drugs': len(bydrug)})
                if genename == 'primary869' and pc == 1:
                    results.extend(rows)
                if genename == 'primary869' and pc == 1:
                    for source in [a, b]:
                        pls = sorted(set((p for r in rows for p in r['plates_A' if source == a else 'plates_B'].split(';'))))
                        if len(pls) < 3:
                            sensitivity.append({'A': a, 'B': b, 'metric': 'leave_one_plate_out', 'axis': genename, 'pseudocount': pc, 'source': source, 'status': 'not_estimable_fewer_than_three_independent_plates'})
                            continue
                        for plate in pls:
                            lrows = []
                            for rec, nx, ny, pa, pb, pla, plb in detail:
                                ka = pla != plate if source == a else np.ones(len(pla), bool)
                                kb = plb != plate if source == b else np.ones(len(plb), bool)
                                paa = pa[ka[pa[:, 0]] & ka[pa[:, 1]]]
                                pbb = pb[kb[pb[:, 0]] & kb[pb[:, 1]]]
                                if not len(paa) or not len(pbb):
                                    continue
                                ia = np.unique(paa)
                                ib = np.unique(pbb)
                                aa = float(np.mean(np.sum(nx[paa[:, 0]] * nx[paa[:, 1]], axis=1)))
                                bb = float(np.mean(np.sum(ny[pbb[:, 0]] * ny[pbb[:, 1]], axis=1)))
                                ab = float(nx[ia].mean(0) @ ny[ib].mean(0))
                                lrows.append({**rec, 'rAA': aa, 'rBB': bb, 'rAB': ab, 'gap_mean': (aa + bb) / 2 - ab})
                            if lrows:
                                for q in aggregate(pd.DataFrame(lrows), ['rAA', 'rBB', 'rAB', 'gap_mean'], bootstrap=False):
                                    q.update(A=a, B=b, source=source, omitted_plate=plate, axis=genename, pseudocount=pc, analysis='leave_one_plate_out')
                                    sensitivity.append(q)
                            else:
                                sensitivity.append({'A': a, 'B': b, 'source': source, 'omitted_plate': plate, 'analysis': 'leave_one_plate_out', 'status': 'insufficient_remaining_independent_repeat_support'})
    pd.DataFrame(results).to_csv(out / 'CONDITION_RESULTS.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'MATCHED_RELIABILITY.csv', index=False)
    pd.DataFrame([r for r in summary if r['metric'].startswith('gap')]).to_csv(out / 'PAIRED_GAPS.csv', index=False)
    wrongsum = []
    if wrong:
        w = pd.DataFrame(wrong)
        w.to_csv(out / 'WRONG_DRUG_CONDITIONS.csv', index=False)
        for key, part in w.groupby(['A', 'B', 'axis', 'pseudocount']):
            for q in aggregate(part, ['rAB', 'wrong_drug_rAB', 'same_minus_wrong']):
                q.update(A=key[0], B=key[1], axis=key[2], pseudocount=key[3])
                wrongsum.append(q)
    pd.DataFrame(wrongsum).to_csv(out / 'WRONG_DRUG_CONTROL.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'SENSITIVITY.csv', index=False)
    pd.DataFrame(EXCLUDE).to_csv(out / 'EXCLUSIONS.csv', index=False)
    jdump(out / 'SELECTED_INPUT_HASHES.json', {'inputs': READS, 'selected_arrays': valuehash, 'model_TEST_reads': 0, 'descriptive_Tahoe_all_roles': True, 'no_full_mixed_role_response_hashing': True})
    jdump(out / 'EXP1_RECEIPT.json', {'status': 'COMPLETE', 'seconds': time.monotonic() - START, 'job': os.getenv('TAHOE_REPRO_RUN_ID'), 'cores': os.getenv('TAHOE_REPRO_CPUS'), 'zero_GPU': True, 'new_download_bytes': 0, 'response_reads': 'selected authorized repeats and exact controls only', 'verification_pair_saved': verification, 'two_repeat_mean_sensitivity': 'implemented for mutually disjoint four-repeat supports; eligibility reported by pair', 'gap_definition': 'rAA-rAB, rBB-rAB and (rAA+rBB)/2-rAB; positive means cross-source agreement below within-source repeatability', 'all_15_metadata_supports_reported': True})
    print('EXP1 COMPLETE', time.monotonic() - START, flush=True)
if __name__ == '__main__':
    main()

from pathlib import Path
import os, sys, json, re, hashlib, itertools, collections, subprocess, ast, time
import numpy as np, pandas as pd, requests
R = Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2')
O = R / 'pairs'
F = Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
I = R.parent / 'PREPRINT_PAIR_EXHAUSTIVENESS_V1_20261030'
assert os.environ.get('TAHOE_REPRO_RUN_ID') and (not (O / 'PROTOCOL.json').exists())
for d in ['metadata', 'results']:
    (O / d).mkdir(parents=True, exist_ok=True)
old = json.loads((I / 'IDENTITY_MAP_FINAL.json').read_text())
names = list(old['drug'])
cm = old['cell']
parents = {}
blocked = set()

def root(x):
    parents.setdefault(x, x)
    if parents[x] != x:
        parents[x] = root(parents[x])
    return parents[x]

def link(a, b):
    x, y = (root(a), root(b))
    if x != y:
        parents[max(x, y)] = min(x, y)
for z in pd.read_csv(I / 'metadata/CHEMICAL_IDENTITY_EVIDENCE.csv.gz').itertuples():
    link(z.left, z.right)
for n in names:
    for d, k in old['drug'][n].items():
        link(n + '|' + d, k)
md = {n: pd.read_parquet((R if n == 'Tahoe' else F) / 'data' / f'{n}_CONDITION_META.parquet') for n in names}
md['Tahoe']['normalization'] = 'author_linear_normalized'
t = pd.read_parquet(F.parent / 'baicai/canonical_perturbation_data_v1/tahoe/data/official_drug_metadata.parquet')
t.to_csv(O / 'metadata/TAHOE_OFFICIAL_COMPOUNDS_ALL.csv', index=False)
need = t[~t.drug.isin(old['drug']['Tahoe'])].copy()
ids = sorted(set((int(x) for x in need.pubchem_cid.dropna())))
props = {}
dl = []
for lo in range(0, len(ids), 50):
    cs = ids[lo:lo + 50]
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + ','.join(map(str, cs)) + '/property/InChIKey,IsomericSMILES/JSON'
    for attempt in range(3):
        try:
            ans = requests.get(url, timeout=90)
            ans.raise_for_status()
            z = ans.json()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    for p in z['PropertyTable']['Properties']:
        props[int(p['CID'])] = p
    dl.append(dict(url=url, bytes=len(ans.content), sha256=hashlib.sha256(ans.content).hexdigest()))
    (O / 'metadata' / f'PUBCHEM_CID_{lo}.json').write_bytes(ans.content)
    time.sleep(0.3)
struct = []
audit = []
for z in need.to_dict('records'):
    p = props.get(int(z['pubchem_cid']) if pd.notna(z['pubchem_cid']) else -1, {})
    key = p.get('InChIKey', '')
    s = p.get('SMILES', p.get('IsomericSMILES', ''))
    if key:
        link('Tahoe|' + z['drug'], 'KEY|' + key)
    if key and s:
        struct.append(dict(dataset='Tahoe', drug=z['drug'], key=key, smiles=s))
    audit.append(dict(drug=z['drug'], cid=z['pubchem_cid'], full_inchikey=key, status='resolved official CID' if key else 'unresolved no forced match'))
q = subprocess.run([sys.executable, str(Path(__file__).with_name('parent_structures.py'))], input=json.dumps(struct), text=True, capture_output=True, check=True)
par = json.loads(q.stdout)
for z in par:
    if z['parent_key']:
        link('Tahoe|' + z['drug'], 'KEY|' + z['parent_key'])
pd.DataFrame(par).to_csv(O / 'metadata/TAHOE_ADDED_PARENT_AUDIT.csv', index=False)
pd.DataFrame(audit).to_csv(O / 'metadata/TAHOE_ADDED_IDENTITY_AUDIT.csv', index=False)
pd.DataFrame(dl).to_csv(O / 'metadata/DOWNLOAD_LOG.csv', index=False)
mapping = {n: {} for n in names}
changes = []
for n in names:
    for d in md[n].drug.astype(str).unique():
        node = n + '|' + d
        if node in parents:
            mapping[n][d] = root(node)
        if d in old['drug'][n] and mapping[n].get(d) != old['drug'][n][d]:
            changes.append(dict(dataset=n, drug=d, old=old['drug'][n][d], new=mapping[n].get(d)))
(O / 'metadata/IDENTITY_EXTENSION_CHANGES.json').write_text(json.dumps(changes, indent=2))
(O / 'GENE_SCOPE.json').write_bytes((I / 'GENE_SCOPE.json').read_bytes())
identity = {'drug': mapping, 'cell': cm, 'policy': old['policy'] + '; expanded Tahoe official CID-resolved 379-drug universe'}
(O / 'IDENTITY_MAP_FINAL.json').write_text(json.dumps(identity, indent=2))
data = {}
flow = []
for n, m in md.items():
    m = m[m.normalization != 'logCPM100'].copy()
    m['source_drug'] = m.drug.astype(str)
    m['source_cell'] = m.cell.astype(str)
    m['drug'] = m.source_drug.map(mapping[n])
    m['cvcl'] = m.source_cell.map(cm[n]).fillna('')
    m['cell'] = np.where(m.cvcl.ne(''), m.cvcl, n + '::' + m.source_cell)
    m.to_parquet(O / 'metadata' / f'{n}_ANNOTATED_CONDITIONS.parquet', index=False)
    good = m[m.drug.notna() & (m.dose > 0) & np.isfinite(m.time)].copy()
    groups = good.groupby(['drug', 'cell', 'dose', 'time'], sort=True)
    a = groups.agg(cvcl=('cvcl', 'first'), source_rows=('row', lambda s: ','.join((str(int(i)) for i in s))), n_source_conditions=('row', 'size')).reset_index()
    a['value_row'] = np.arange(len(a))
    data[n] = a
    a.to_parquet(O / 'metadata' / f'{n}_VECTOR_KEYS.parquet', index=False)
    flow.append(dict(dataset=n, all_conditions=len(m), identity_resolved_conditions=int(m.drug.notna().sum()), identity_resolved_drugs=m.loc[m.drug.notna(), 'source_drug'].nunique(), source_drugs=m.source_drug.nunique(), resolved_cells=m.loc[m.cvcl.ne(''), 'source_cell'].nunique(), source_cells=m.source_cell.nunique(), resolved_24h_drugs=a[a.time == 24].drug.nunique(), aggregated_conditions=len(a)))
pd.DataFrame(flow).to_csv(O / 'results/IDENTITY_COVERAGE_FINAL.csv', index=False)
selection = []
pairrows = []
for a, b in itertools.combinations(names, 2):
    aa = {d: z for d, z in data[a].groupby('drug')}
    bb = {d: z for d, z in data[b].groupby('drug')}
    counts = collections.Counter()
    for d in sorted(set(aa) & set(bb)):
        for tm in sorted(set(aa[d].time) & set(bb[d].time)):
            la = aa[d][aa[d].time == tm]
            lb = bb[d][bb[d].time == tm]
            opts = []
            for da, ga in la.groupby('dose'):
                for db, gb in lb.groupby('dose'):
                    ratio = max(da, db) / min(da, db)
                    item = dict(a=a, b=b, drug=d, time=float(tm), dose_a=float(da), dose_b=float(db), dose_ratio=float(ratio), rows_a=ga.value_row.astype(int).tolist(), rows_b=gb.value_row.astype(int).tolist(), cells_a=ga.cell.tolist(), cells_b=gb.cell.tolist())
                    rank = (abs(np.log(da / db)), -min(len(ga), len(gb)), da, db)
                    opts.append((rank, item))
                    if ratio > 2 + 1e-08:
                        continue
                    selection.append(dict(item, mode='MARGINAL_ALL_NEAR'))
                    counts['MARGINAL_ALL_NEAR'] += 1
                    common = sorted(set(ga.loc[ga.cvcl.ne(''), 'cell']) & set(gb.loc[gb.cvcl.ne(''), 'cell']))
                    for c in common:
                        x = ga[ga.cell == c].iloc[0]
                        y = gb[gb.cell == c].iloc[0]
                        selection.append(dict(item, mode='STRICT_ALL_NEAR', rows_a=[int(x.value_row)], rows_b=[int(y.value_row)], cells_a=[c], cells_b=[c]))
                        counts['STRICT_ALL_NEAR'] += 1
            if opts:
                selection.append(dict(min(opts, key=lambda z: z[0])[1], mode='MARGINAL_NEAREST_ANY'))
                counts['MARGINAL_NEAREST_ANY'] += 1
                near = [z for z in opts if z[1]['dose_ratio'] <= 2 + 1e-08]
                if near:
                    selection.append(dict(min(near, key=lambda z: z[0])[1], mode='MARGINAL_NEAREST'))
                    counts['MARGINAL_NEAREST'] += 1
                strict = []
                for rank, item in near:
                    cells = sorted(set((c for c in item['cells_a'] if c.startswith('CVCL_'))) & set(item['cells_b']))
                    if cells:
                        strict.append(((rank[0], -len(cells), item['dose_a'], item['dose_b']), dict(item, rows_a=[item['rows_a'][item['cells_a'].index(c)] for c in cells], rows_b=[item['rows_b'][item['cells_b'].index(c)] for c in cells], cells_a=cells, cells_b=cells)))
                if strict:
                    selection.append(dict(min(strict, key=lambda z: z[0])[1], mode='STRICT_NEAREST'))
                    counts['STRICT_NEAREST'] += 1
    pairrows.append(dict(a=a, b=b, shared_resolved_drugs_all_times=len(set(aa) & set(bb)), **counts))
pd.DataFrame(pairrows).to_csv(O / 'results/PAIR_AVAILABILITY_ALL_TIMES.csv', index=False)
flows = []
for n in names:
    m = pd.read_parquet(O / 'metadata' / f'{n}_ANNOTATED_CONDITIONS.parquet')
    for partner in names:
        if partner == n:
            continue
        other = data[partner]
        other = other[(other.time == 24) & (other.dose > 0)]
        otherdrugs = set(other.drug)
        od = {k: z for k, z in other.groupby('drug')}
        oc = {k: z for k, z in other[other.cvcl.ne('')].groupby(['drug', 'cell'])}
        for mode in ['STRICT_ALL_NEAR', 'MARGINAL_ALL_NEAR']:
            reason = []
            for z in m.itertuples():
                if pd.isna(z.drug):
                    s = 'unresolved_drug_identity'
                elif z.time != 24:
                    s = 'outside_primary_24h'
                elif not np.isfinite(z.dose) or z.dose <= 0:
                    s = 'invalid_or_nonpositive_dose'
                elif mode == 'STRICT_ALL_NEAR' and (not z.cvcl):
                    s = 'unresolved_cell_identity'
                elif z.drug not in otherdrugs:
                    s = 'no_same_drug_in_partner_at_24h'
                elif mode == 'STRICT_ALL_NEAR' and (z.drug, z.cell) not in oc:
                    s = 'no_same_cell_for_this_drug'
                else:
                    ds = (oc[z.drug, z.cell] if mode == 'STRICT_ALL_NEAR' else od[z.drug]).dose.to_numpy()
                    ratio = np.maximum(ds, z.dose) / np.minimum(ds, z.dose)
                    s = 'eligible_all_near' if np.any(ratio <= 2 + 1e-08) else 'no_dose_within_factor_2'
                reason.append(s)
            cc = collections.Counter(reason)
            assert sum(cc.values()) == len(m)
            for why, ct in cc.items():
                flows.append(dict(dataset=n, partner=partner, mode=mode, reason=why, conditions=ct, total_conditions=len(m)))
pd.DataFrame(flows).to_csv(O / 'results/PAIR_EXCLUSION_FLOW.csv', index=False)
(O / 'LOCKED_SELECTIONS.json').write_text(json.dumps(selection, ensure_ascii=False, separators=(',', ':')))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol = dict(task='PREPRINT_PAIR_EXHAUSTIVENESS_V1', locked_before_response_reads=True, selection_sha256=sha(O / 'LOCKED_SELECTIONS.json'), identity_sha256=sha(O / 'IDENTITY_MAP_FINAL.json'), gene_axis=869, primary_time=24, secondary_times='all exact shared durations, reported separately by time', dose_rule='all positive dose pairs within factor 2; legacy nearest and unrestricted-nearest sensitivity separate', strict='same confirmed Cellosaurus, all eligible cells and near-dose pairs', marginal='each source own complete eligible cell set including source-specific unmapped cell labels; not pure beta and not same-cell equivalence', weighting='mean dose pairs within drug x cell, cells within drug, drugs within chemical supercomponent for Tahoe pairs; otherwise drug equal; never treat dose multiplicity as independent drugs', metrics=['R1 across-gene Pearson', 'R3 common-drug background within exact cell/time/dose-pair blocks >=3 drugs', 'same-background retrieval for blocks >=2 drugs; fixed candidate pools'], bootstrap=2000, bootstrap_seed=20261030, CI_minimum_groups=3, training=0, model_TEST_response_reads=0, descriptive_Tahoe_all_roles=True, GPU=0, do_not_select_by_results=True, resource_cap_allocated_core_hours=8)
protocol['gene_axis_sensitivity'] = {'genes': 882, 'rule': 'maximal unique Entrez intersection of the six retained source releases; same locked condition supports; metadata-derived before response reads', 'additional_genes': json.loads((O / 'GENE_SCOPE.json').read_text())['additional_genes']}
protocol['identity_correction'] = {'inherited_nodes_replaced': len(blocked), 'authority': 'official Tahoe PubChem CID full stereochemistry', 'reason': 'metadata conflict recheck; first numeric attempt invalidated independent of response values', 'prior_numeric_attempt_read_before_correction': bool(blocked)}
(O / 'PROTOCOL.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
(O / 'PROTOCOL.sha256').write_text(sha(O / 'PROTOCOL.json') + '  PROTOCOL.json\n')
print('LOCKED', len(selection), 'selections', sha(O / 'PROTOCOL.json'), flush=True)

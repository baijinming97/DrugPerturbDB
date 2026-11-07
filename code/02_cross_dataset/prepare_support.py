"""Reconstruct comparison support from frozen identities and fresh condition annotations."""
from pathlib import Path
import json,itertools,collections
import numpy as np
import pandas as pd
R=Path('@WORKSPACE@/4MAN/TAHOE_DESCRIPTIVE_FULL_V2')
O=R/'pairs';F=Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
identity=json.loads((O/'IDENTITY_MAP_FINAL.json').read_text())
mapping=identity['drug'];cm=identity['cell'];names=list(mapping)
md={n:pd.read_parquet((R if n=='Tahoe' else F)/'data'/f'{n}_CONDITION_META.parquet') for n in names}
md['Tahoe']['normalization']='author_linear_normalized'
annotated={}
data = {}
flow = []
for n, m in md.items():
    m = m[m.normalization != 'logCPM100'].copy()
    m['source_drug'] = m.drug.astype(str)
    m['source_cell'] = m.cell.astype(str)
    m['drug'] = m.source_drug.map(mapping[n])
    m['cvcl'] = m.source_cell.map(cm[n]).fillna('')
    m['cell'] = np.where(m.cvcl.ne(''), m.cvcl, n + '::' + m.source_cell)
    annotated[n]=m
    good = m[m.drug.notna() & (m.dose > 0) & np.isfinite(m.time)].copy()
    groups = good.groupby(['drug', 'cell', 'dose', 'time'], sort=True)
    a = groups.agg(cvcl=('cvcl', 'first'), source_rows=('row', lambda s: ','.join((str(int(i)) for i in s))), n_source_conditions=('row', 'size')).reset_index()
    a['value_row'] = np.arange(len(a))
    data[n] = a
    pd.testing.assert_frame_equal(a,pd.read_parquet(O/'metadata'/f'{n}_VECTOR_KEYS.parquet'),check_dtype=False)
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
    m = annotated[n]
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

assert selection==json.loads((O/'LOCKED_SELECTIONS.json').read_text()),'Reconstructed condition selection differs from the locked protocol'
(O/'SUPPORT_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',selection_rows=len(selection),condition_tables=len(names),identity_maps_modified=False,response_values_read=False),indent=2))
print('SUPPORT_VERIFIED',len(selection),flush=True)

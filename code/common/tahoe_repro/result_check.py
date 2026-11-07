"""Compare newly computed manuscript endpoints with the supplied numerical tables."""
from pathlib import Path
import io
import json
import zipfile
import numpy as np
import pandas as pd


def check_results(repo, workspace, baseline=None, tolerance=1e-8):
    repo, workspace = Path(repo), Path(workspace)
    full = workspace / 'FULL_DATASET_FIGURES_V1'
    tahoe = workspace / '4MAN/TAHOE_DESCRIPTIVE_FULL_V2'
    figure = repo / 'result/figure_inputs/updated_inputs/figures'
    rows, missing = [], []

    def exists(path):
        if path.is_file(): return True
        missing.append(str(path)); return False

    def record(name, errors, **labels):
        error = max(errors.values(), default=0.)
        rows.append(dict(table=name, **labels, max_absolute_error=float(error),
                         status='PASS' if error < tolerance else 'FAIL'))

    def compare_tables(name, old, new):
        if list(old.columns) != list(new.columns) or old.shape != new.shape:
            rows.append(dict(table=name, status='FAIL_SCHEMA')); return
        numeric = list(old.select_dtypes(include='number').columns)
        errors = {}
        for col in old.columns:
            if col in numeric:
                a, b = old[col].to_numpy(float), new[col].to_numpy(float)
                if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                    rows.append(dict(table=name, column=col, status='FAIL_MISSINGNESS')); return
                errors[col] = float(np.max(abs(a[np.isfinite(a)]-b[np.isfinite(b)]), initial=0))
            elif not old[col].fillna('').astype(str).equals(new[col].fillna('').astype(str)):
                rows.append(dict(table=name, column=col, status='FAIL_IDENTITIES')); return
        record(name, errors)

    for name in ['fig1a_expression.csv', 'fig1b_response.csv', 'fig3a_full.csv']:
        reference = pd.read_csv(figure / name)
        # Archived inputs also retain L5 and NCI; neither appears in the six-source manuscript.
        reference = reference[reference.dataset.isin(['Tahoe','L1000_L3','DRUGseq','CIGS','sciPlex3','MIXseq'])]
        for dataset, subset in reference.groupby('dataset', sort=False):
            folder = tahoe if dataset == 'Tahoe' else full
            composition = name == 'fig3a_full.csv'
            path = folder / 'results' / (dataset + ('_FULL_COMPOSITION.csv' if composition else '_REPEAT_SUMMARY.csv'))
            if not exists(path): continue
            actual = pd.read_csv(path)
            if dataset == 'Tahoe':
                actual = actual[actual.scope.eq('ALL_ELIGIBLE')] if composition else actual[actual.dose_scope.eq('all_doses')]
            for row in subset.to_dict('records'):
                selected = actual
                for key in (['axis','normalization'] if composition else ['mode','axis','normalization']):
                    selected = selected[selected[key].eq(row[key])]
                if len(selected) != 1: raise ValueError((name,dataset,row,len(selected)))
                keys = ([k for k in row if k in actual and (k.endswith('_fraction') or k in ['genes','drugs','cells','conditions','strata','interaction_identifiable_weight'])]
                        if composition else ['pcc','low','high','drugs','conditions','pairs','genes'])
                errors = {key: float(abs(selected.iloc[0][key]-row[key])) for key in keys if pd.notna(row[key])}
                record(name, errors, dataset=dataset, axis=row['axis'], normalization=row['normalization'])

    with zipfile.ZipFile(repo/'result/source_tables/Data_S1_measurements.zip') as archive:
        prefix = 'evidence_inputs/TAHOE_DESCRIPTIVE_FULL_V2/pairs/'
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith(('.csv','.csv.gz')): continue
            leaf = name.removeprefix(prefix)
            path = tahoe/'pairs/results'/leaf
            if not exists(path): continue
            old = pd.read_csv(io.BytesIO(archive.read(name)), compression='gzip' if name.endswith('.gz') else None)
            compare_tables(leaf,old,pd.read_csv(path))
        prefix='evidence_inputs/PAPER_TWO_CHECKS_V1/results/exp1/'
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith('.csv'):continue
            leaf=name.removeprefix(prefix);path=tahoe/'matched_repeats/exp1'/leaf
            if exists(path):
                compare_tables('Matched_repeats/'+leaf,pd.read_csv(io.BytesIO(archive.read(name))),pd.read_csv(path))

    with zipfile.ZipFile(repo/'result/source_tables/Data_S2_predictions.zip') as archive:
        name = 'evidence_inputs/PAPER_TWO_CHECKS_V1/results/exp2/WITHIN_CROSS_PLATE.csv'
        path = workspace/'4MAN/PAPER_TWO_CHECKS_V1/exp2/reweighted_common_support/WITHIN_CROSS_PLATE.csv'
        if exists(path):
            compare_tables('WITHIN_CROSS_PLATE.csv',pd.read_csv(io.BytesIO(archive.read(name))),pd.read_csv(path))
        for row in pd.read_csv(io.BytesIO(archive.read('tables/S4_exploration.csv'))).to_dict('records'):
            path = workspace/'4MAN/EXPLORATORY/results'/row['id']/'SUMMARY.json'
            if not exists(path): continue
            actual = json.loads(path.read_text())
            mapping = dict(value='value',lo='value_lo',hi='value_hi',delta='paired_delta',dlo='paired_delta_lo',dhi='paired_delta_hi',n='drugs')
            record('Figure_S2', {key: abs(float(row[key])-float(actual[value])) for key,value in mapping.items()}, configuration=row['id'])

    baseline = Path(baseline) if baseline else repo/'result/04_prediction/computed/baseline'
    for name in ['BASELINE_REPRODUCTION.json','G5_REPRODUCTION.json']:
        path = baseline/name
        if exists(path):
            data = json.loads(path.read_text())
            # The rounded baseline reference has six decimal places.
            rows.append(dict(table=name,status=data['status'],max_absolute_error=max(data['absolute_error'].values())))
    path = baseline/'BASELINE_COMPONENTS_BY_FOLD.csv'
    if exists(path):
        old = pd.read_csv(repo/'result/figure_inputs/evidence_inputs/PREPRINT_FULL_DATA_V1/evidence_model/FOUR_COMPONENT_COMPARISON.csv')
        current = pd.read_csv(path)
        for component, table in current.groupby('component'):
            ref = old[(old.model == 'B') & (old.component == component)].iloc[0]
            errors = {key: abs(float(np.average(table[key],weights=table.mass))-float(ref[key])) for key in ['PCC','MSE']}
            record('Figure_4_components', errors, component=component)
    path=baseline/'BASELINE_ERROR_ALLOCATION.csv'
    if exists(path):
        with zipfile.ZipFile(repo/'result/source_tables/Data_S2_predictions.zip') as archive:
            ref=pd.read_csv(io.BytesIO(archive.read('tables/S3_component_energy_error.csv')))
        ref=ref[ref.panel.eq('error')].set_index('component')
        actual=pd.read_csv(path).set_index('component')
        for component,row in actual.iterrows():
            record('Baseline_error_allocation',{'MSE':abs(row.MSE-ref.loc[component,'B_error']),
                   'fraction':abs(row.error_fraction-ref.loc[component,'error_fraction'])},component=component)
    status = 'FAIL' if any(row['status'] != 'PASS' for row in rows) else 'INCOMPLETE' if missing else 'PASS'
    return dict(status=status,checked_endpoints=len(rows),absolute_tolerance=tolerance,
                missing_files=sorted(set(missing)),comparisons=rows)

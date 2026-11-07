"""Validate immutable companion code, metadata and numerical result files."""
from pathlib import Path
import json
import io
import zipfile
import numpy as np
from .sources import sha256


def verify_release(root):
    root=Path(root).resolve()
    checked=0
    for line in (root/'data/SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1)
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256(p)!=expected:
            raise ValueError(f'Release file missing or modified: {name}')
        checked+=1
    archives=0
    for p in (root/'result/source_tables').glob('*.zip'):
        with zipfile.ZipFile(p) as archive:
            bad=archive.testzip()
            if bad:raise ValueError(f'Corrupt result archive member: {p.name}/{bad}')
        archives+=1
    for row in json.loads((root/'data/config/production_sources.json').read_text()):
        p=(root/row['file']).resolve()
        if not p.is_relative_to(root) or sha256(p)!=row['sha256']:
            raise ValueError('Processing-source registry mismatch')
    return dict(status='PASS',immutable_files=checked,result_archives=archives,
                raw_data_recomputed=False)


def verify_measured_examples(root, tolerance=1e-10):
    """Independently recalculate retained real-measurement Pearson examples.

    These examples check the metric and common-gene support, not complete
    dataset aggregation, bootstrapping, or reconstruction of original files.
    """
    def pearson(x,y):
        valid=np.isfinite(x)&np.isfinite(y)
        if valid.sum()<3:
            raise ValueError('Insufficient jointly observed genes')
        return float(np.corrcoef(x[valid],y[valid])[0,1])

    with zipfile.ZipFile(Path(root)/'result/source_tables/Data_S1_measurements.zip') as z:
        def member(suffix):
            found=[name for name in z.namelist() if name.endswith(suffix)]
            if len(found)!=1:raise ValueError(f'Ambiguous example source: {suffix}')
            return z.read(found[0])
        reference=json.loads(member('/REPLAY_EXAMPLES.json'))
        with np.load(io.BytesIO(member('/REPLAY_EXAMPLES.npz')),allow_pickle=False) as a:
            if len(reference)!=len(a['x']) or a['x'].shape!=a['y'].shape:
                raise ValueError('Example metadata and arrays disagree')
            errors=[abs(pearson(x,y)-row['pcc']) for x,y,row in zip(a['x'],a['y'],reference)]
            genes=a['x'].shape[1]
        with np.load(io.BytesIO(member('/verification_pair.npz')),allow_pickle=False) as a:
            within_a=np.mean([pearson(a['A'][i],a['A'][j]) for i,j in a['AA_pairs']])
            within_b=np.mean([pearson(a['B'][i],a['B'][j]) for i,j in a['BB_pairs']])
            across=np.mean([pearson(x,y) for x in a['A'] for y in a['B']])
            paired_error=max(abs(within_a-float(a['claimed_rAA'])),
                             abs(within_b-float(a['claimed_rBB'])),
                             abs(across-float(a['claimed_rAB'])))
    maximum=max(max(errors),paired_error)
    if not np.isfinite(maximum) or maximum>=tolerance:
        raise ValueError(f'Measured example mismatch: {maximum}')
    return dict(status='PASS',cross_source_examples=len(reference),genes=genes,
                matched_repeatability_cases=1,largest_absolute_error=maximum,
                original_downloads_recomputed=False,
                full_dataset_statistics_recomputed=False)

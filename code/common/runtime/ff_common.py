from pathlib import Path
import os, sys, json, time, hashlib, subprocess
sys.dont_write_bytecode = True
R = Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
O = R.parent
A = O / '4MAN/CROSS_DATASET_ATLAS_V1'
C = O / 'baicai/canonical_perturbation_data_v1'
import numpy as np, pandas as pd, h5py
from scipy import sparse
GENE = pd.read_csv(O / '4MAN/minimum_restore_v1/retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data/GENE_AXIS.tsv', sep='\t', dtype=str)
COMMON = json.loads((O / 'REPEATABILITY_HARMONIZED_V1/outputs/GLOBAL_GENE_MASK.json').read_text())['axis']

def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def quota():
    import shutil
    free=shutil.disk_usage(R).free/2**30
    if free<float(os.environ.get('TAHOE_REPRO_MIN_FREE_GIB','15')):
        raise RuntimeError('Insufficient workspace disk space')
    return free

def guard(event, args):
    if event != 'open' or not args or (not isinstance(args[0], (str, bytes, os.PathLike))):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    fl = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    if str(p).startswith(str(O)) and fl & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        assert p.is_relative_to(R), str(p)
sys.addaudithook(guard)

def gene_ids(native, kind):
    native = np.asarray(native, str)
    out = np.full(len(native), '', object)
    lookup = {}
    if kind == 'drugseq':
        m = pd.read_csv(O / '4MAN/DRUGSEQ_BETAW_PILOT_V1/features/OBSERVED_GENES.tsv', sep='\t', dtype=str)
        lookup = dict(zip(m.source_gene_id, m.entrez))
        return np.asarray([lookup.get(v, '') for v in native], object)
    fields = {'symbol': ['l1000_gene_symbol', 'hgnc_approved_symbol', 'tahoe_gene_symbol'], 'ensembl': ['tahoe_ensembl_id'], 'entrez': ['l1000_entrez_gene_id']}[kind]
    for row in GENE.to_dict('records'):
        for f in fields:
            s = str(row.get(f, ''))
            s = s.split('.')[0] if kind == 'ensembl' else s.upper()
            if s and s != 'NAN':
                lookup[s] = str(row['l1000_entrez_gene_id'])
    for i, s in enumerate(native):
        out[i] = lookup.get(s.split('.')[0] if kind == 'ensembl' else s.upper(), '')
    return out

def newh5(name, meta, genes, kind, post=True, delta=True):
    quota()
    meta = meta.reset_index(drop=True).copy()
    meta['row'] = np.arange(len(meta))
    meta.to_parquet(R / 'data' / f'{name}_META.parquet', index=False)
    p = R / 'data' / f'{name}.h5'
    h = h5py.File(p, 'w')
    dt = h5py.string_dtype('utf-8')
    h.create_dataset('gene', data=np.asarray(genes, object), dtype=dt)
    h.create_dataset('entrez', data=gene_ids(genes, kind).astype(object), dtype=dt)
    for k, use in [('post', post), ('delta', delta)]:
        if use:
            h.create_dataset(k, shape=(len(meta), len(genes)), dtype='f4', chunks=(min(64, len(meta)), min(4096, len(genes))), compression='lzf')
    h.attrs['source_gene_kind'] = kind
    h.attrs['protocol_sha256'] = sha(R / 'PROTOCOL.json')
    return h

def finish(name, start, details):
    p = R / 'data' / f'{name}.h5'
    v = dict(status='PREPARED', seconds=time.time() - start, job=os.getenv('TAHOE_REPRO_RUN_ID'), bytes=p.stat().st_size, sha256=sha(p), **details)
    dump(R / 'receipts' / f'{name}_PREPARED.json', v)
    print(name, v, flush=True)

def read_cat(h, key):
    x = h[key]
    if isinstance(x, h5py.Group):
        cat = x['categories'].asstr()[:] if x['categories'].dtype.kind in 'OS' else x['categories'][:]
        code = x['codes'][:]
        if (code < 0).any():
            out = np.full(len(code), None, object)
            ok = code >= 0
            out[ok] = cat[code[ok]]
            return out
        return cat[code]
    if x.dtype.kind in 'OS':
        return x.asstr()[:]
    return x[:]

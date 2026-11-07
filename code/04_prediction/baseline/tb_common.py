from pathlib import Path
import os, json, hashlib, datetime, subprocess, fcntl
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
TASK_NAME = R.parent.name
B = Path('@WORKSPACE@/baicai')
V = B/'tahoe_x1only_h2048_delta978_20260903'
S = B/'tahoe_x1only_h2048_delta978_5fold_v2_20260903/03_splits'
C = B/'l3_responseaware_singlex1_transfer_20260908/04_TARGET_SINGLE_X1'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.part')
    t.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def allocated():
    assert not os.environ.get('CUDA_VISIBLE_DEVICES',''), 'CPU only'
def protocol(): return read(R/'PROTOCOL.json')
def meta(): return pd.read_csv(R/'data/CONDITIONS.tsv',sep='\t',keep_default_na=False)
def splits(b,f): return dict(np.load(R/f'data/{b}_fold{f}.npz',allow_pickle=False))
def metric(p,y):
    p=np.asarray(p,np.float64);y=np.asarray(y,np.float64)
    def cosine(a,b):
        d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        return np.divide((a*b).sum(1),d,out=np.full(len(a),np.nan),where=d>1e-15)
    mse=((p-y)**2).mean(1)
    return dict(PCC=cosine(p-p.mean(1,keepdims=True),y-y.mean(1,keepdims=True)),MSE=mse,RMSE=np.sqrt(mse),cosine=cosine(p,y))
def verify_lock():
    lock=read(R/'LOCK.json')
    for p,h in lock['files'].items(): assert sha(p)==h, 'Frozen hash drift: '+p
    return lock
def storage(release=False):
    import shutil
    free=shutil.disk_usage(R).free
    minimum=int(float(os.environ.get('TAHOE_REPRO_MIN_FREE_GIB','15'))*2**30)
    if not release and free<minimum:raise RuntimeError('Insufficient workspace storage')
    return dict(free_bytes=free,minimum_free_bytes=minimum)

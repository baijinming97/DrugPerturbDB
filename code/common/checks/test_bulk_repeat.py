"""Full-panel aggregation must equal explicit independent-plate pair enumeration."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def test_block_normalization_preserves_cross_plate_pairs():
    source=Path(__file__).resolve().parents[2]/'01_repeatability/bulk_repeats.py'
    tree=ast.parse(source.read_text())
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    scope=dict(np=np,pd=pd,sparse=sparse)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),scope)
    x=np.random.default_rng(41).normal(size=(31,143)).astype('float32');x[2]=1
    codes=np.repeat([0,1],[17,14]);plates=np.arange(len(x))%4
    meta=pd.DataFrame({'plate':plates})
    cm=pd.DataFrame({'drug':['a','b'],'cell':['c','c'],'dose':[1.,1.],
                     'time':[24.,24.],'panel':['all','all'],'normalization':['native','native']})
    common=np.array([17,2,80,34,81,90,1])
    got=scope['bulk_repeats'](x,meta,cm,codes,np.arange(len(x)),common,'delta','fixture')
    for row in got:
        pos=np.arange(x.shape[1]) if row['axis']=='native' else common
        ids=np.flatnonzero(codes==(['a','b'].index(row['drug'])));pairs=[]
        for k,i in enumerate(ids):
            for j in ids[k+1:]:
                if plates[i]==plates[j]:continue
                a=x[i,pos].astype(float);b=x[j,pos].astype(float)
                if np.linalg.norm(a-a.mean())<=1e-12 or np.linalg.norm(b-b.mean())<=1e-12:continue
                pairs.append(np.corrcoef(a,b)[0,1])
        assert row['pairs']==len(pairs)
        np.testing.assert_allclose(row['pcc'],np.mean(pairs),atol=1e-12,rtol=0)

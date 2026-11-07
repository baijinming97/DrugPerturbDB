"""Independent linear-algebra checks of the production kernels, no biology data."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'code/common'))
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from tahoe_repro.projection import Projector
from tahoe_repro.statistics import decompose,weighted_design,all_cross_plate_pcc

root=Path(__file__).resolve().parents[3]
rng=np.random.default_rng(20261105)
# Sparse connected design: compare the fitted additive signal with direct WLS.
meta=pd.DataFrame([(c,d) for c in range(4) for d in range(7) if (c+2*d)%5],columns=['cell','drug'])
y=rng.normal(size=(len(meta),11));energy,comp,info=decompose(y,meta)
d,c,w,*_=weighted_design(meta)
design=np.column_stack([np.ones(len(meta)),np.eye(c.max()+1)[c],np.eye(d.max()+1)[d]])
fit=design@np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=1e-12)[0]
wls_error=float(np.max(abs(fit-(comp['mu']+comp['cell']+comp['drug']))))
closure=float(abs(energy.sum()-np.sum(w[:,None]*y*y)))
assert wls_error<1e-10 and closure<1e-10
# An independently constructed null-space projection checks weights and masks.
C,D,G=3,9,7
X=np.zeros((C,D,2));X[:,:6,0]=1;X[:,6:,0]=2;X[:,:,1]=np.arange(C)[:,None]
w=np.arange(1,D+1,dtype=float);w/=w.sum()
mask=np.ones((C,D),bool);mask[0,0]=False;mask[2,7:]=False
projector=Projector(X,w,minimum=3,mask=mask)
z=rng.normal(size=(C,D,G));actual=projector(z)
ix=np.flatnonzero(mask);wt=np.tile(w/C,C)[ix];rows=[]
for cc in range(C):
    for ctx in [1,2]:rows.append(((np.repeat(np.arange(C),D)==cc)&(X[:,:,0].ravel()==ctx))[ix]*wt)
for dd in range(D):rows.append((np.tile(np.arange(D),C)==dd)[ix]*wt)
constraint=np.asarray(rows)/np.sqrt(wt)[None,:]
N=null_space(constraint,rcond=1e-12)
q=z.reshape(C*D,G)[ix]*np.sqrt(wt[:,None]);expected=np.zeros((C*D,G))
expected[ix]=(N@(N.T@q))/np.sqrt(wt[:,None]);expected=expected.reshape(C,D,G)
projection_error=float(np.max(abs(expected-actual)))
residuals=projector.residuals(actual)
assert projection_error<1e-10 and max(residuals.values())<1e-10
# Cross-plate shortcut must equal explicit condition-pair Pearson calculation.
z=rng.normal(size=(7,11));plates=np.array([0,0,1,1,1,2,2])
fast,n=all_cross_plate_pcc(z.copy(),plates)
explicit=[np.corrcoef(z[i],z[j])[0,1] for i in range(len(z)) for j in range(i) if plates[i]!=plates[j]]
pcc_error=float(abs(fast-np.mean(explicit)))
assert n==len(explicit) and pcc_error<1e-12
out=dict(status='PASS_SYNTHETIC_INDEPENDENT_LINEAR_ALGEBRA',wls_max_error=wls_error,
         energy_closure=closure,projection_max_error=projection_error,
         crossplate_correlation_error=pcc_error,constraint_residuals=residuals,
         scientific_result_recomputed=False)
print(json.dumps(out,indent=2))

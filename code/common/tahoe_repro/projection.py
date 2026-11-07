"""Weighted KKT orthogonal projection; inactive rows fixed at zero."""
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
class Projector:
 def __init__(self,X,w,minimum=0,mask=None):
  self.C,self.D=X.shape[:2];self.w=np.asarray(w);labels=[];counts=[];offset=0
  for c in range(self.C):
   mapping={};ids=[]
   for x in X[c]:
    key=x.tobytes()
    if key not in mapping:mapping[key]=len(mapping)
    ids.append(mapping[key])
   ids=np.array(ids);labels.extend(ids+offset);counts.extend(np.bincount(ids));offset+=len(mapping)
  self.labels=np.array(labels);self.counts=np.array(counts);self.eligible=self.counts[self.labels].reshape(self.C,self.D)>=3
  active=np.ones((self.C,self.D),bool) if minimum==0 else self.counts[self.labels].reshape(self.C,self.D)>=minimum
  if mask is not None:active&=mask
  self.active=active;self.ix=np.flatnonzero(active);drug=np.tile(np.arange(self.D),self.C)
  self.wt=np.tile(w/self.C,self.C)[self.ix]
  G=sparse.csr_matrix((np.ones(2*len(self.ix)),(np.r_[self.labels[self.ix],offset+drug[self.ix]],np.tile(np.arange(len(self.ix)),2))),shape=(offset+self.D,len(self.ix)))
  self.G=G;L=(G.multiply(self.wt)@G.T).toarray()
  # LAPACK syevr can fail on exactly repeated eigenvalues in small inner panels.
  # Divide-and-conquer computes the same symmetric eigensystem robustly.
  ev,u=eigh(L,check_finite=False,driver='evd');keep=ev>max(float(ev.max(initial=0))*1e-12,1e-18)
  self.pinv=(u[:,keep]/ev[keep])@u[:,keep].T
  self.info=dict(rank=int(keep.sum()),variables=len(self.ix),dimension=int(len(self.ix)-keep.sum()),eligible_fraction=float(self.eligible.mean()),active_weight=float((active*w[None]/self.C).sum()))
 def __call__(self,x):
  flat=x.reshape(self.C*self.D,-1);out=np.zeros_like(flat,dtype=float);z=flat[self.ix]
  if len(self.ix):out[self.ix]=z-self.G.T@(self.pinv@(self.G@(self.wt[:,None]*z)))
  return out.reshape(x.shape)
 def residuals(self,x):
  z=x.reshape(self.C*self.D,-1)
  cons=self.G@(self.wt[:,None]*z[self.ix]);den=np.asarray(self.G@self.wt)[:,None]
  mx=float(np.max(abs(np.divide(cons,den,out=np.zeros_like(cons),where=den>0)),initial=0))
  return dict(constraint_mean_maxabs=mx,inactive_maxabs=float(np.max(abs(z[~self.active.ravel()]),initial=0)))
def independent_ap(x,X,w,active):
 C,D=x.shape[:2];out=x.copy()*active[...,None];groups=[]
 for c in range(C):
  buckets={}
  for d,v in enumerate(X[c]):buckets.setdefault(v.tobytes(),[]).append(d)
  groups.extend((c,np.array([d for d in ds if active[c,d]],int)) for ds in buckets.values())
 for it in range(30000):
  old=out.copy()
  for c,ds in groups:
   if len(ds):out[c,ds]-=np.average(out[c,ds],axis=0,weights=w[ds])
  counts=active.sum(0);mu=np.divide(out.sum(0),counts[:,None],out=np.zeros_like(out[0]),where=counts[:,None]>0)
  out-=mu[None]*active[...,None]
  if np.max(abs(out-old))<1e-12:break
 else:raise RuntimeError('AP did not converge')
 return out,it+1

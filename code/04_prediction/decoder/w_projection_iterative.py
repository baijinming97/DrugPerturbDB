"""Weighted alternating orthogonal projections; no responses used to define W."""
import numpy as np
from scipy.sparse import csr_matrix
class WProjection:
 def __init__(self,X,w,minimum=0):
  self.C,self.D=X.shape[:2];self.w=np.asarray(w,float)/np.sum(w)
  labels=[];offset=0;counts=[]
  for c in range(self.C):
   rows=np.ascontiguousarray(X[c]);_,inv,n=np.unique(rows.view(np.dtype((np.void,rows.dtype.itemsize*rows.shape[1]))).ravel(),return_inverse=True,return_counts=True)
   labels.extend(inv+offset);counts.extend(n);offset+=len(n)
  self.ctx=np.array(labels).reshape(self.C,self.D);self.counts=np.array(counts)
  self.eligible=self.counts[self.ctx]>=3
  self.active=np.ones((self.C,self.D),bool) if minimum==0 else self.counts[self.ctx]>=minimum
  n=self.C*self.D;idx=np.arange(n);wg=np.tile(self.w,self.C)*self.active.ravel();cl=self.ctx.ravel()
  den=np.bincount(cl,weights=wg,minlength=offset)
  val=np.divide(wg,den[cl],out=np.zeros(n),where=den[cl]>0)
  self.mean=csr_matrix((val,(cl,idx)),shape=(offset,n))
  self.drugden=self.active.sum(0)
  drug=np.tile(np.arange(self.D),self.C)
  self.context_drug=csr_matrix((val,(cl,drug)),shape=(offset,self.D))
  dval=np.divide(self.active.ravel().astype(float),self.drugden[drug],out=np.zeros(n),where=self.drugden[drug]>0)
  self.drug_context=csr_matrix((dval,(drug,cl)),shape=(self.D,offset))
 def residuals(self,x):
  z=x.reshape(self.C*self.D,-1)
  return dict(context_mean=float(np.max(abs(self.mean@z))),drug_mean=float(np.max(abs(x.sum(0)/self.C))),inactive=float(np.max(abs(x[~self.active]))) if (~self.active).any() else 0.)
 def __call__(self,value,return_info=False):
  x=np.asarray(value,dtype=np.float64).copy();shape=x.shape;x=x.reshape(self.C,self.D,-1);x*=self.active[...,None]
  # Exact alternating projections represented by context/drug mean corrections.
  # Avoid repeatedly copying the large condition x gene array.
  cy=self.mean@x.reshape(self.C*self.D,-1)
  dy=np.divide(x.sum(0),self.drugden[:,None],out=np.zeros_like(x[0]),where=self.drugden[:,None]>0)
  u=np.zeros_like(cy);v=np.zeros_like(dy)
  for it in range(200000):
   un=cy-self.context_drug@v;vn=dy-self.drug_context@un
   du=un-u;dv=vn-v;bound=float(np.max(abs(du))+np.max(abs(dv)))
   u,v=un,vn
   # Gauge shifts can cancel on edges; a coefficient bound alone need not decay.
   # Test the requested actual array change regularly, without relaxing tolerance.
   if bound<1e-10 or (it+1)%16==0:
    change=float(np.max(abs((du[self.ctx]+dv[None])*self.active[...,None])))
    if change<1e-10:break
  else:raise RuntimeError(f'Alternating projection did not converge: change={change}, bound={bound}')
  x-=(u[self.ctx]+v[None])*self.active[...,None]
  res=self.residuals(x);assert max(res.values())<3e-9,(it,res)
  info=dict(iterations=it+1,change=change,**res)
  x=x.reshape(shape)
  return (x,info) if return_info else x
 def summary(self):
  return dict(contexts=len(self.counts),context_sizes=self.counts.tolist(),eligible_fraction=float(self.eligible.mean()),eligible_weight=float(np.mean(self.eligible@self.w)),active_fraction=float(self.active.mean()))
def pcc0(p,y):
 p=p-p.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True)
 den=np.sqrt(np.sum(p*p,-1)*np.sum(y*y,-1));return np.divide(np.sum(p*y,-1),den,out=np.zeros(den.shape),where=den>1e-25)
def wstats(p,y):return np.stack([pcc0(p,y),np.mean((p-y)**2,-1)],-1)

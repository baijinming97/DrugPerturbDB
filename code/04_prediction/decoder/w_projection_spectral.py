"""Exact batches of alternating-projection iterates, using the drug transition spectrum."""
import numpy as np
from scipy.linalg import eigh
from w_projection_iterative import WProjection as IterativeProjection,pcc0,wstats
class WProjection(IterativeProjection):
 def __init__(self,X,w,minimum=0):
  super().__init__(X,w,minimum);self.ids=np.flatnonzero(self.drugden>0)
  pi=self.w[self.ids]*self.drugden[self.ids];self.sqrtpi=np.sqrt(pi)
  T=(self.drug_context@self.context_drug).toarray()[np.ix_(self.ids,self.ids)]
  sym=T*self.sqrtpi[:,None]/self.sqrtpi[None];assert np.max(abs(sym-sym.T))<1e-12
  ev,U=eigh((sym+sym.T)/2,check_finite=False);assert ev.min()>-1e-10 and ev.max()<1+1e-10
  self.ev=np.clip(ev,0,1);self.gauge=abs(1-self.ev)<1e-12;self.ev[self.gauge]=1.;self.U=U
 def _basis(self,value):
  x=np.array(value,dtype=float,copy=True);shape=x.shape;x=x.reshape(self.C,self.D,-1);x*=self.active[...,None]
  cy=self.mean@x.reshape(self.C*self.D,-1);dy=np.divide(x.sum(0),self.drugden[:,None],out=np.zeros_like(x[0]),where=self.drugden[:,None]>0)
  b=dy-self.drug_context@cy;coef=self.U.T@(b[self.ids]*self.sqrtpi[:,None]);gauge=float(np.max(abs(coef[self.gauge]))) if self.gauge.any() else 0.;assert gauge<1e-10,gauge;coef[self.gauge]=0
  return x,shape,cy,coef,gauge
 def _drug(self,coef,factor):
  out=np.zeros((self.D,coef.shape[1]));out[self.ids]=(self.U@(coef*factor[:,None]))/self.sqrtpi[:,None];return out
 def at_iteration(self,value,t):
  assert t>=1;x,shape,cy,coef,_=self._basis(value);gap=1-self.ev
  def v(n):return self._drug(coef,np.divide(1-self.ev**n,gap,out=np.zeros_like(gap),where=~self.gauge))
  u=cy-self.context_drug@v(t-1);return (x-(u[self.ctx]+v(t)[None])*self.active[...,None]).reshape(shape)
 def __call__(self,value,return_info=False):
  x,shape,cy,coef,gauge=self._basis(value);gap=1-self.ev
  infinity=self._drug(coef,np.divide(1.,gap,out=np.zeros_like(gap),where=~self.gauge));limit=x-((cy-self.context_drug@infinity)[self.ctx]+infinity[None])*self.active[...,None]
  for power in range(1,29):
   t=2**power
   vm1=self._drug(coef,np.divide(1-self.ev**(t-1),gap,out=np.zeros_like(gap),where=~self.gauge));vt=self._drug(coef,np.divide(1-self.ev**t,gap,out=np.zeros_like(gap),where=~self.gauge))
   u=cy-self.context_drug@vm1;out=x-(u[self.ctx]+vt[None])*self.active[...,None]
   prev=self._drug(coef,self.ev**(t-2));curr=self._drug(coef,self.ev**(t-1));change=float(np.max(abs(((self.context_drug@prev)[self.ctx]-curr[None])*self.active[...,None])))
   limit_error=float(np.max(abs(out-limit)))
   if change<1e-10 and limit_error<1e-10:break
  else:raise RuntimeError('Spectral evaluation of AP failed convergence')
  res=self.residuals(out);assert max(res.values())<3e-9,res
  info=dict(iterations=t,actual_evaluated_powers=power,change=change,limit_error=limit_error,gauge_roundoff=gauge,algorithm='Exact closed-form batch of alternating mean-subtraction iterates',**res)
  return (out.reshape(shape),info) if return_info else out.reshape(shape)

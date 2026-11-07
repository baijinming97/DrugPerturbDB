"""Independent-drug inference. No query labels outside measured cells enter infer()."""
import numpy as np
from scipy.linalg import solve
from tp_projection import Projector

def decoder(h,refX,refw,queryX):
 C=refX.shape[0];D=refX.shape[1];X=np.concatenate([refX,queryX[:,None]],1)
 weights=np.r_[refw/refw.sum()*D,1.];weights/=weights.sum()
 P=Projector(X,weights,3);raw=np.zeros((C,D+1,C));raw[:,-1]=np.eye(C)
 projected=P(raw);Q=projected[:,-1]
 rawG=(h['coef']/h['scale'][:,None]@h['U']).reshape(len(h['v']),C,978).transpose(1,0,2)
 G=np.einsum('ab,brg->arg',Q,rawG)
 assert np.max(abs(G.mean(0)))<1e-8
 return dict(V=h['v'],G=G,T=G+h['v'][None],Q=Q,anchor_residual=P.residuals(projected),eligible=P.active[:,-1],reference_compensation_max=float(abs(projected[:,:-1]).max(initial=0)))

def infer(dec,measured_residual,observed,alpha,prior=None,prior_strength=0.):
 """Input is m x genes only. Return beta, gamma for this drug, no batch state."""
 observed=np.asarray(observed,int);r=np.asarray(measured_residual,float)
 assert r.shape==(len(observed),978) and len(set(observed))==len(observed)
 T=dec['T'][observed];gram=np.einsum('crg,csg->rs',T,T)/len(observed)
 rhs=np.einsum('crg,cg->r',T,r)/len(observed)
 prior_coef=np.zeros(len(rhs)) if prior is None else prior@dec['V'].T
 coef=solve(gram+(alpha+prior_strength)*np.eye(len(rhs)),rhs+prior_strength*prior_coef,assume_a='pos',check_finite=False)
 beta=coef@dec['V'];gamma=np.einsum('r,crg->cg',coef,dec['G'])
 return beta,gamma,coef

def cache_stats(dec,B,Y):
 T=dec['T'];Tc=T-T.mean(-1,keepdims=True);bc=B-B.mean(-1,keepdims=True);yc=Y-Y.mean(-1,keepdims=True)
 return dict(gram=np.einsum('crg,csg->crs',T,T),rhs=np.einsum('crg,cg->cr',T,Y-B),
  ynorm=np.sum(yc*yc,1),bnorm=np.sum(bc*bc,1),by=np.sum(bc*yc,1),
  ty=np.einsum('crg,cg->cr',Tc,yc),tb=np.einsum('crg,cg->cr',Tc,bc),
  tt=np.einsum('crg,csg->crs',Tc,Tc),baseerr=np.mean((Y-B)**2,1))

def score_cache(st,observed,alpha):
 gram=st['gram'][observed].mean(0);rhs=st['rhs'][observed].mean(0)
 coef=solve(gram+alpha*np.eye(len(rhs)),rhs,assume_a='pos',check_finite=False)
 dot=st['by']+st['ty']@coef
 norm=st['bnorm']+2*st['tb']@coef+np.einsum('r,crs,s->c',coef,st['tt'],coef)
 den=np.sqrt(np.maximum(norm,0)*st['ynorm']);pcc=np.divide(dot,den,out=np.zeros_like(dot),where=den>1e-12)
 mse=st['baseerr']+(-2*st['rhs']@coef+np.einsum('r,crs,s->c',coef,st['gram'],coef))/978
 keep=np.ones(len(pcc),bool);keep[observed]=False
 return pcc[keep].mean(),mse[keep].mean()

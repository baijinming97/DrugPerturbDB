"""Fit and apply the drug-average-to-interaction decoder using training data."""
from eval_lib import *
from scipy.linalg import eigh,solve
def pca(x,k=32):
 mu=x.mean(0);a=x-mu;ev,u=eigh(a@a.T,check_finite=False)
 ix=np.argsort(ev)[::-1][:min(k,len(x)-1,x.shape[1])];ev=ev[ix];u=u[:,ix];ok=ev>max(ev.max(initial=0)*1e-12,1e-20)
 v=u[:,ok].T@a/np.sqrt(ev[ok])[:,None] if ok.any() else np.zeros((0,x.shape[1]))
 z=a@v.T;scale=np.maximum(z.std(0),1e-8)
 return z/scale,dict(mu=mu,v=v,scale=scale)
def weighted_basis(y,eligible,w,K=20):
 X=y.transpose(1,0,2).reshape(y.shape[1],-1);obs=np.repeat(eligible.T,y.shape[-1],axis=1);filled=np.where(obs,X,0);sw=np.sqrt(w/w.sum());previous=None
 for it in range(250 if not obs.all() else 1):
  S=filled*sw[:,None];ev,u=eigh(S@S.T,check_finite=False);ix=np.argsort(ev)[::-1][:min(K,len(X)-1)];ev=ev[ix];u=u[:,ix];ok=ev>max(ev.max(initial=0)*1e-12,1e-20);U=u[:,ok].T@S/np.sqrt(ev[ok])[:,None]
  if obs.all():break
  recon=filled@U.T@U;obj=np.sum((recon-X)**2*obs*w[:,None]);filled[~obs]=recon[~obs]
  if previous is not None and abs(obj-previous)/max(abs(previous),1e-20)<1e-5:break
  previous=obj;budget()
 z=np.zeros((len(X),len(U)))
 for d in range(len(X)):
  v=U[:,obs[d]];z[d]=np.linalg.lstsq(v.T,X[d,obs[d]],rcond=1e-10)[0]
 return U,z,dict(iterations=it+1,rank=len(U),missing_fraction=float((~obs).mean()))
def fit_linear(p,K=20,alpha=100,seed=None,beta_kind='full'):
 w=weights(p['group']);P=Projector(p['X'],w);y=P(p['Y']);U,z,info=weighted_basis(y,P.eligible,w,K)
 b=dec(p['Y'],w)['beta']
 if beta_kind=='within':b=P_betaw(b,p['X'],w)
 if seed is not None:b=b[np.random.default_rng(seed).permutation(len(b))]
 x,prep=pca(b,32);ww=w/w.sum()*len(w);xm=np.average(x,axis=0,weights=ww);ym=np.average(z,axis=0,weights=ww);a=x-xm
 coef=solve(a.T@(a*ww[:,None])+alpha*np.eye(x.shape[1]),a.T@((z-ym)*ww[:,None]),assume_a='pos',check_finite=False)
 return dict(**prep,xm=xm,ym=ym,coef=coef,U=U,linear=True,info=info,train_drug=p['drug'],beta_kind=beta_kind)
def raw_gamma(h,b):
 x=(b-h['mu'])@h['v'].T/h['scale'];z=(x-h['xm'])@h['coef']+h['ym']
 return (z@h['U']).reshape(len(b),47,978).transpose(1,0,2)
def linear_decoder(h,refX,refw,queryX,reference='R'):
 if reference=='R':return single_decoder(h,refX,refw,queryX)
 V=h['v'];raw=(h['coef']/h['scale'][:,None]@h['U']).reshape(len(V),47,978).transpose(1,0,2);G=raw-raw.mean(0)
 return dict(V=V,G=G,T=V[None]+G,Q=np.eye(47)-np.ones((47,47))/47,eligible=np.ones(47,bool),reference_compensation_max=0.)

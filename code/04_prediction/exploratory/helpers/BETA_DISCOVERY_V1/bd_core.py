"""TRAIN-only nested chemical screening. Never imports/reads outer VALID responses."""
from pathlib import Path
import os,sys,json,time,collections
import numpy as np,pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parent;O=R.parent;START=time.monotonic();CPUS=int(os.environ.get('TAHOE_REPRO_CPUS',4))
def dump(path,x):
 p=R/path;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)));q.replace(p)
def nz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def weights(g):
 c=collections.Counter(g);return np.array([1/(len(c)*c[x]) for x in g])
def splits(g,n=4,seed=20261022):
 u=sorted(set(g));rank=dict(zip(u,np.random.default_rng(seed).permutation(len(u))))
 return list(GroupKFold(min(n,len(u))).split(np.arange(len(g)),groups=[rank[x] for x in g]))
def corr(x,y):
 x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True);de=np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
 return np.divide((x*y).sum(-1),de,out=np.zeros(de.shape),where=de>1e-12)
def proj(v,context,w,available=None):
 out=np.zeros_like(v,dtype=float);av=np.ones(len(v),bool) if available is None else available
 for g in np.unique(context):
  ix=np.flatnonzero((context==g)&av)
  if len(ix)>1:out[ix]=v[ix]-np.average(v[ix],axis=0,weights=w[ix])
 return out
def basis(y,w,k):
 S=y*np.sqrt(w/w.sum())[:,None];val,u=eigh(S@S.T,driver='evd',check_finite=False);ix=np.argsort(val)[::-1][:min(k,len(y)-1)];val=val[ix];u=u[:,ix];ok=val>max(val.max(initial=0)*1e-12,1e-20)
 return u[:,ok].T@S/np.sqrt(val[ok])[:,None]
def group_perm(g,seed):
 rng=np.random.default_rng(seed);out=np.arange(len(g));buckets={}
 for n in sorted(set(g)):buckets.setdefault(int(np.sum(g==n)),[]).append(np.flatnonzero(g==n))
 for size,blocks in buckets.items():
  for i,j in enumerate(rng.permutation(len(blocks))):out[blocks[i]]=blocks[j]
 return out
def preprocess(x,q,fingerprint=False):
 ok=np.isfinite(x).mean(0)>=.5
 if not ok.any():return np.zeros((len(x),1)),np.zeros((len(q),1))
 x=x[:,ok];q=q[:,ok];mi=~np.isfinite(x);mq=~np.isfinite(q);mu=np.nanmean(x,0);x=np.where(mi,mu,x);q=np.where(mq,mu,q)
 if not fingerprint:
  sd=x.std(0);keep=sd>1e-8;x=(x[:,keep]-mu[keep])/sd[keep];q=(q[:,keep]-mu[keep])/sd[keep]
  if mi.any():x=np.c_[x,mi[:,keep]];q=np.c_[q,mq[:,keep]]
 else:
  keep=np.ptp(x,0)>0;x=x[:,keep];q=q[:,keep]
 if not x.shape[1]:x=np.zeros((len(x),1));q=np.zeros((len(q),1))
 return x,q
def gram(x,q,kind,scale=1):
 if kind=='linear':return x@x.T/max(x.shape[1],1),q@x.T/max(x.shape[1],1)
 if kind=='tanimoto':
  aa=x@x.T;qa=q@x.T;d=(x*x).sum(1)
  return aa/np.maximum(d[:,None]+d[None]-aa,1e-12),qa/np.maximum((q*q).sum(1)[:,None]+d[None]-qa,1e-12)
 if kind=='cosine':
  x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
  return x@x.T,q@x.T
 d=(x*x).sum(1);dist=np.maximum(d[:,None]+d[None]-2*x@x.T,0);med=np.median(dist[np.triu_indices(len(x),1)]);gd=max(med,1e-12)*scale
 return np.exp(-dist/gd),np.exp(-np.maximum((q*q).sum(1)[:,None]+d[None]-2*q@x.T,0)/gd)
def regress(K,Q,z,w,alpha):
 wn=w/w.sum();km=wn@K;glob=km@wn;Kc=K-km[None]-km[:,None]+glob;Qc=Q-(Q@wn)[:,None]-km[None]+glob;sw=np.sqrt(wn*len(w))
 val,u=eigh(Kc*sw[:,None]*sw[None],check_finite=False,driver='evd');ym=wn@z
 co=sw[:,None]*(u@((u.T@((z-ym)*sw[:,None]))/np.maximum(val[:,None]+alpha,1e-10)))
 return Qc@co+ym
def knn(Q,z,w,k):
 ix=np.argsort(Q,axis=1)[:,-min(k,len(z)):];s=np.maximum(np.take_along_axis(Q,ix,1),0)*w[ix];s=np.where(s.sum(1,keepdims=True)>1e-12,s,np.ones_like(s));s/=s.sum(1,keepdims=True)
 return np.einsum('ij,ijk->ik',s,z[ix])
_BOOTSTRAP_DRAWS={}
def summarize(rows,cols):
 t=pd.DataFrame(rows);ix,g=pd.factorize(t.group,sort=True);w=t.weight.to_numpy(float);de=np.bincount(ix,weights=w,minlength=len(g));values=t[cols].to_numpy(float)
 nu=np.column_stack([np.bincount(ix,weights=w*values[:,j],minlength=len(g)) for j in range(len(cols))])
 if len(g) not in _BOOTSTRAP_DRAWS:_BOOTSTRAP_DRAWS[len(g)]=np.random.default_rng(20261022).multinomial(len(g),np.ones(len(g))/len(g),2000)
 dr=_BOOTSTRAP_DRAWS[len(g)];bs=dr@nu/np.maximum((dr@de)[:,None],1e-20)
 return dict(zip(cols,nu.sum(0)/de.sum()))|dict(zip([c+'_lo' for c in cols],np.quantile(bs,.025,axis=0)))|dict(zip([c+'_hi' for c in cols],np.quantile(bs,.975,axis=0)))
class Screen:
 def __init__(self,f,feature):
  self.f=f;self.p=nz(R/f'TRAIN_f{f}.npz');self.n=len(self.p['drug']);self.feature=feature
  if feature.startswith('Gaussian_'):
   seed=int(feature.split('_')[1]);dim=[64,256,2048][(seed-8100)%3];ids=pd.read_csv(R/'DRUG_IDENTITIES.csv').drug.to_numpy();v=dict(drug=ids,value=np.random.default_rng(seed).normal(size=(len(ids),dim)),available=np.ones(len(ids),bool))
  else:v=nz(R/'features'/(f'Positive_0p5_f{f}.npz' if feature=='Positive_0p5' else f'{feature}.npz'))
  ids=v['drug'];val=v['value'];lu={d:i for i,d in enumerate(ids)};self.X=np.full((self.n,val.shape[1]),np.nan);self.av=np.zeros(self.n,bool)
  for j,d in enumerate(self.p['drug']):
   if d in lu:self.X[j]=val[lu[d]];self.av[j]=v['available'][lu[d]]
  self.fp=feature.startswith(('ECFP','FCFP','MACCS','AtomPair','TopologicalTorsion','ErG_Tanimoto'));self.cache={};self.targets={};self.plans=splits(self.p['group'],4)
 def target(self,ix):
  key=tuple(ix)
  if key not in self.targets:
   w=weights(self.p['group'][ix]);self.targets[key]=(proj(self.p['mean'][ix],self.p['context'][ix],w),w)
  return self.targets[key]
 def block(self,fit,query,model,param):
  key=(tuple(fit),tuple(query),model,str(param))
  if key not in self.cache:
   # Refit labels and target basis on the actual covered fit drugs; no held response participates.
   fit=fit[self.av[fit]];qy=query[self.av[query]];y,w=self.target(fit);U=basis(y,w,20);z=y@U.T
   x,q=preprocess(self.X[fit],self.X[qy],self.fp and model=='KRR')
   kind='linear' if model=='Ridge' else 'cosine' if model=='kNN' else 'tanimoto' if self.fp else 'rbf'
   K,Q=gram(x,q,kind,float(param[1]) if model=='KRR' else 1);self.cache[key]=dict(fit=fit,query=qy,U=U,z=z,w=w,K=K,Q=Q)
  return self.cache[key]
 def predict(self,fit,query,model,param,k,seed=None):
  out=np.zeros((len(query),978))
  if self.av[fit].sum()<6 or self.av[query].sum()==0:return out
  v=self.block(fit,query,model,param);z=v['z'][:,:k]
  if seed is not None:z=z[group_perm(self.p['group'][v['fit']],seed)]
  pp=knn(v['Q'],z,v['w'],int(param)) if model=='kNN' else regress(v['K'],v['Q'],z,v['w'],float(param[0] if model=='KRR' else param))
  lu={j:i for i,j in enumerate(query)};out[[lu[j] for j in v['query']]]=pp@v['U'][:z.shape[1]]
  return proj(out,self.p['context'][query],weights(self.p['group'][query]),self.av[query])
 def params(self,model):
  return [.1,1.,10.,100.] if model=='Ridge' else [(a,s) for a in [.1,1.,10.] for s in ([1.] if self.fp else [.25,1.,4.])] if model=='KRR' else [3,5,10]
 def choose(self,fit,model,k,seed):
  plans=splits(self.p['group'][fit],3,20261023);scores=[]
  for par in self.params(model):
   loss=0.;den=0.
   for it,iv in plans:
    va=fit[iv];pp=self.predict(fit[it],va,model,par,k,seed);y,w=self.target(va);ok=self.av[va]
    if ok.any():loss+=np.sum(np.mean((pp-y)**2,1)*w*ok);den+=np.sum(w*ok)
   scores.append(loss/max(den,1e-20))
  return self.params(model)[int(np.argmin(scores))]
 def run(self,model,k,seed=None,plans=None):
  pred=np.zeros((self.n,978));sel=[];block=np.zeros(self.n,int)
  for bi,(fit,va) in enumerate(self.plans if plans is None else plans):
   par=self.choose(fit,model,k,seed);pred[va]=self.predict(fit,va,model,par,k,seed);block[va]=bi
   sel.append(dict(block=bi,param=par,fit=self.p['drug'][fit].tolist(),holdout=self.p['drug'][va].tolist()))
  return proj(pred,self.p['context'],weights(self.p['group']),self.av),sel,block
 def probes(self):
  ix=np.flatnonzero(self.av)
  if len(ix)<4:return dict(CKA=0.,Mantel=0.,covered=len(ix))
  y,w=self.target(ix);x,q=preprocess(self.X[ix],self.X[ix]);K=x@x.T;L=y@y.T;H=np.eye(len(ix))-np.ones((len(ix),len(ix)))/len(ix);K=H@K@H;L=H@L@H
  cka=np.sum(K*L)/max(np.linalg.norm(K)*np.linalg.norm(L),1e-20);_,Q=gram(x,x,'cosine');_,S=gram(y,y,'cosine');tri=np.triu_indices(len(ix),1);cross=self.p['group'][ix[tri[0]]]!=self.p['group'][ix[tri[1]]]
  return dict(CKA=float(cka),Mantel=float(spearmanr(Q[tri][cross],S[tri][cross]).statistic),covered=len(ix))


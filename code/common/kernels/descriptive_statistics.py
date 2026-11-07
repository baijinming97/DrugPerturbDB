import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def pearson_rows(x,y):
 x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True);den=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
 return np.divide((x*y).sum(-1),den,out=np.full(den.shape,np.nan),where=den>1e-12)

def boot_mean(values,clusters=None,seed=20261029):
 values=np.asarray(values,float);good=np.isfinite(values);values=values[good]
 if clusters is not None:
  clusters=np.asarray(clusters)[good];values=pd.DataFrame({'v':values,'g':clusters}).groupby('g').v.mean().to_numpy()
 if len(values)<2:return float(np.mean(values)) if len(values) else np.nan,np.nan,np.nan
 rng=np.random.default_rng(seed);b=[]
 for i in range(0,2000,20):b.extend(values[rng.integers(0,len(values),size=(min(20,2000-i),len(values)))].mean(1))
 return float(values.mean()),*np.quantile(b,[.025,.975]).tolist()

def weighted_design(meta):
 # Drug equal; observed cell/dose/time conditions equal within each drug.
 d,drugs=pd.factorize(meta.drug,sort=True);c,cells=pd.factorize(meta.cell,sort=True)
 w=meta['_weight'].to_numpy(float).copy() if '_weight' in meta else 1/np.bincount(d)[d];w/=w.sum();nd=len(drugs);nc=len(cells)
 wd=np.bincount(d,weights=w);wc=np.bincount(c,weights=w)
 D=sparse.csr_matrix((w/wd[d],(d,np.arange(len(d)))),shape=(nd,len(d)))
 Cc=sparse.csr_matrix((w/wc[c],(c,np.arange(len(c)))),shape=(nc,len(c)))
 adj=sparse.csr_matrix((np.ones(len(d)),(c,d+nc)),shape=(nc+nd,nc+nd));nconn=connected_components(adj+adj.T,directed=False,return_labels=False)
 return d,c,w,D,Cc,dict(drugs=nd,cells=nc,observed=len(meta),rectangle_fraction=len(meta)/(nd*nc),connected_components=int(nconn),interaction_identifiable=bool(nc>1 and nd>1 and len(meta)>nd+nc-int(nconn)))

def decompose(x,meta,tol=1e-10):
 x=np.asarray(x,np.float64);d,c,w,D,Cc,info=weighted_design(meta)
 mu=w@x;z=x-mu;ad=Cc@z;bd=D@z
 s0=(w[:,None]*z*z).sum(0);sa=(w[:,None]*(z-ad[c])**2).sum(0);sb=(w[:,None]*(z-bd[d])**2).sum(0)
 # Exact descriptive weighted least-squares, eliminating drug effects.
 A=sparse.csr_matrix((w,(c,d)),shape=(Cc.shape[0],D.shape[0]));wc=np.bincount(c,weights=w);wd=np.bincount(d,weights=w)
 M=np.diag(wc)-(A.multiply(1/wd)@A.T).toarray();ev,U=np.linalg.eigh((M+M.T)/2);good=ev>max(float(ev.max()),1e-15)*1e-12
 rhs=wc[:,None]*ad-A@bd
 a=U[:,good]@((U[:,good].T@rhs)/ev[good,None]) if good.any() else np.zeros_like(ad)
 a-=wc@a;b=D@(z-a[c]);it=0
 err=max(float(np.max(abs(Cc@(z-a[c]-b[d])))),float(np.max(abs(D@(z-a[c]-b[d])))))
 residual=z-a[c]-b[d];sab=(w[:,None]*residual**2).sum(0)
 energy=np.array([np.sum(mu*mu),np.sum(.5*((s0-sa)+(sb-sab))),np.sum(.5*((s0-sb)+(sa-sab))),np.sum(sab)])
 closure=abs(energy.sum()-np.sum(w[:,None]*x*x));assert closure<1e-7*max(1,energy.sum()),closure
 if err>1e-7:raise RuntimeError(('additive_normal_equation_failed',err,info))
 info.update(iterations=it+1,max_update=err,closure=closure)
 # Components are unique on balanced connected support. On sparse support use predictions, not individual coefficient claims.
 return energy,dict(mu=mu,cell=a[c],drug=b[d],residual=residual,weighted_mean=w@x),info

def all_cross_plate_pcc(x,plate):
 x=np.array(x,dtype=float,copy=True);x-=x.mean(1,keepdims=True);norm=np.linalg.norm(x,axis=1);ok=norm>1e-12;x=x[ok]/norm[ok,None];plate=np.asarray(plate)[ok]
 n=len(x)
 if n<2:return np.nan,0
 ids,_=pd.factorize(plate);npl=np.bincount(ids);den=n*n-np.sum(npl*npl)
 if den==0:return np.nan,0
 sm=np.zeros((len(npl),x.shape[1]));np.add.at(sm,ids,x)
 return float(((x.sum(0)**2).sum()-(sm*sm).sum())/den),int(den//2)

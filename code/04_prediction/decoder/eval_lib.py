"""Frozen evaluation contract; support and denominators remain explicit."""
from tp_assets import *
from tp_projection import Projector
from tp_single import decoder as single_decoder,infer as single_infer
from tp_model_io import load_model
METRICS=['gamma_PCC','gamma_MSE','full_PCC','full_MSE','beta_w_PCC','gamma_w_PCC']
SEEDS=[1337,2026,3407]
def correlation(x,y):
 z=pcc(x,y);norm=np.linalg.norm(x-x.mean(-1,keepdims=True),axis=-1)
 return np.where(norm<=1e-6,0.,np.nan_to_num(z,nan=0.))
def P_w(value,X,w,minimum=0,mask=None):return Projector(X,w,minimum,mask)(value)
def P_betaw(beta,X,w,mask=None):
 ids=full_context(X);out=np.zeros_like(beta,dtype=float)
 if mask is None:mask=np.ones(len(ids),bool)
 for g in np.unique(ids):
  ix=np.flatnonzero((ids==g)&mask)
  if len(ix)>1:out[ix]=beta[ix]-np.average(beta[ix],axis=0,weights=w[ix])
 return out
def metric_rows(pred,Y,X,w,observed=()):
 keep=np.ones(len(Y),bool);keep[list(observed)]=False;p=pred[keep];y=Y[keep];xx=X[keep];dy=dec(y,w);dp=dec(p,w)
 gb=correlation(dp['gamma'],dy['gamma']);gm=np.mean((dp['gamma']-dy['gamma'])**2,-1);fp=correlation(p,y);fm=np.mean((p-y)**2,-1)
 bp=P_betaw(dp['beta'],xx,w);by=P_betaw(dy['beta'],xx,w);bc=correlation(bp,by)
 P=Projector(xx,w);yp=P(y);pp=P(p);wp=correlation(pp,yp);eligible=P.eligible
 gc=np.divide(np.sum(wp*eligible,axis=0),eligible.sum(0),out=np.zeros(len(w)),where=eligible.sum(0)>0)
 return np.stack([gb.mean(0),gm.mean(0),fp.mean(0),fm.mean(0),bc,gc],axis=1),dict(eligible=eligible,unmeasured_cells=np.flatnonzero(keep),gamma_w_metric_support='eligible contexts only; do not subtract conditional metric from all-support MSE')
def add_records(records,f,p,pred,q,model,setting,observed=(),direction='',extra=None):
 w=q/q.sum();s,info=metric_rows(pred,p['Y'],p['X'],w,observed);b,_=metric_rows(p['B'],p['Y'],p['X'],w,observed)
 for j,d in enumerate(p['drug']):
  row=dict(fold=f,drug=d,group=p['group'][j],weight=q[j],gamma_w_weight=q[j]*info['eligible'][:,j].mean(),model=model,setting=setting,direction=direction,m=len(observed),**dict(zip(METRICS,s[j])),**{'B_'+n:v for n,v in zip(METRICS,b[j])})
  if extra:row.update(extra)
  records.append(row)
 return info
def paired_summary(t,value_cols=METRICS,reference_prefix='B_'):
 z=t.copy()
 for n in value_cols:z['delta_'+n]=z[n]-z[reference_prefix+n]
 cols=value_cols+['delta_'+n for n in value_cols];v,ci,bs=bootstrap_linear(z,cols)
 if 'gamma_w_weight' in z and z.gamma_w_weight.sum()>0 and 'gamma_w_PCC' in value_cols:
  zz=z.copy();zz['weight']=zz.gamma_w_weight
  vv,cc,_=bootstrap_linear(zz,['gamma_w_PCC','delta_gamma_w_PCC']);v.update(vv);ci.update(cc)
 return dict(**v,**{k+'_lo':a[0] for k,a in ci.items()},**{k+'_hi':a[1] for k,a in ci.items()},drugs=z.drug.nunique(),groups=z.group.nunique(),weight=z.weight.sum())
def chemistry_splits(g,n=4,seed=20261021):
 from sklearn.model_selection import GroupKFold
 # Deterministic shuffled labels break lexicographic ties, without label access.
 u=sorted(set(g));order=np.random.default_rng(seed).permutation(len(u));mp=dict(zip(u,order));gg=np.array([mp[x] for x in g]);k=min(n,len(u))
 if k<2:raise ValueError('Insufficient chemistry groups for selection')
 return list(GroupKFold(k).split(np.arange(len(g)),groups=gg))
def subset(p,ids):
 ids=np.asarray(ids);out={}
 for key,val in p.items():
  if key in ['X','Y','B']:out[key]=val[:,ids]
  elif key in ['drug','group']:out[key]=val[ids]
  else:out[key]=val
 return out
def shuffled_map(drugs,coverage,seed):
 result=np.arange(len(drugs));rng=np.random.default_rng(seed)
 for pattern in sorted(set(coverage)):
  ix=np.flatnonzero(np.array(coverage)==pattern)
  if len(ix)>1:result[ix]=rng.permutation(ix)
 return result
def check_pure_gamma(delta,X,w):
 d=dec(delta,w);return dict(mu=float(abs(d['mu']).max()),alpha=float(abs(d['alpha']).max()),beta=float(abs(d['beta']).max()),P_w_residual=float(abs(P_w(delta,X,w)-delta).max()))

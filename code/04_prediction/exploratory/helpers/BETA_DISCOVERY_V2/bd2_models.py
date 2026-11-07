"""Nested chemical-OOD models. Fit-dependent sources/targets never cross a held block."""
from bd2_common import *
from scipy.linalg import eigh,solve,svd,orthogonal_procrustes
from scipy.special import softmax
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA,NMF,FastICA
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
def simplex(x):
 u=np.sort(x,axis=-1)[:,::-1];s=np.cumsum(u,1)-1;ind=np.arange(1,x.shape[1]+1);rho=np.sum(u-s/ind>0,1)-1;theta=s[np.arange(len(x)),rho]/(rho+1);return np.maximum(x-theta[:,None],0)
def target_basis(y,w,k,kind='PCA'):
 k=min(k,len(y)-2,y.shape[1]);k=max(k,1)
 if kind=='PCA':u=basis(y,w,k);return y@u.T,u,np.zeros(y.shape[1])
 if kind=='NMF':
  lift=np.c_[np.maximum(y,0),np.maximum(-y,0)];sw=np.sqrt(w/w.sum()*len(w));m=NMF(n_components=k,init='nndsvda',random_state=20261025,max_iter=160,tol=.001);z=m.fit_transform(lift*sw[:,None])/sw[:,None];u=m.components_[:,:y.shape[1]]-m.components_[:,y.shape[1]:];return z,u,np.zeros(y.shape[1])
 if kind=='ICA':
  v=basis(y,w,min(32,len(y)-2));low=y@v.T;m=FastICA(n_components=min(k,low.shape[1]),whiten='unit-variance',random_state=20261025,max_iter=180,tol=.002);z=m.fit_transform(low);return z,m.mixing_.T@v,m.mean_@v
 if kind=='Archetype':
  # Alternating convex archetypal analysis in a TRAIN PCA working space.
  v=basis(y,w,min(32,len(y)-2));z=y@v.T;chosen=[int(np.argmax(np.sum(z*z,1)))];dist=np.sum((z-z[chosen[0]])**2,1)
  while len(chosen)<k:
   j=int(np.argmax(dist));chosen.append(j);dist=np.minimum(dist,np.sum((z-z[j])**2,1));dist[chosen]=-1
  C=np.eye(len(z))[chosen];B=np.ones((len(z),k))/k;ww=w/w.sum()*len(w);z2=max(np.linalg.norm(z,2)**2,1e-12)
  for it in range(40):
   A=C@z;lb=max(np.linalg.norm(A,2)**2*ww.max(),1e-12);B=simplex(B-((B@A-z)*ww[:,None])@A.T/lb)
   lc=max(np.linalg.norm(B*np.sqrt(ww[:,None]),2)**2*z2,1e-12);C=simplex(C-B.T@((B@(C@z)-z)*ww[:,None])@z.T/lc)
  return B,C@y,np.zeros(y.shape[1])
 raise ValueError(kind)
def pca_input(x,q,n=32,whiten=False):
 if n=='full':return x,q
 n=min(int(n),len(x)-2,x.shape[1]);n=max(n,1)
 if len(x)>512:
  unique,count=np.unique(x,axis=0,return_counts=True);u=basis(unique,count/count.sum(),n)
 else:u=basis(x,np.ones(len(x))/len(x),n)
 xx=x@u.T;qq=q@u.T
 if whiten:
  sd=np.maximum(xx.std(0),1e-6);xx/=sd;qq/=sd
 return xx,qq
def ridge_features(x,q,z,w,alpha):
 k,qk=gram(x,q,'linear');return regress(k,qk,z,w,alpha)
def cca_embed(x,q,z,w,reg):
 x,q=pca_input(x,q,32);xm=np.average(x,0,weights=w);zm=np.average(z,0,weights=w);x0=x-xm;z0=z-zm
 def invsqrt(A):
  v,u=eigh(A,check_finite=False);return (u/np.sqrt(np.maximum(v,1e-8)))@u.T
 wx=invsqrt(x0.T@(x0*w[:,None])+reg*np.eye(x.shape[1]));wz=invsqrt(z0.T@(z0*w[:,None])+reg*np.eye(z.shape[1]));u,s,v=svd(wx@(x0.T@(z0*w[:,None]))@wz,full_matrices=False,check_finite=False);L=wx@u[:,:min(10,len(s))];return x0@L,(q-xm)@L
def soft_retrieve(Q,z,w,k,temp,amplitude=None):
 ix=np.argsort(Q,axis=1)[:,-min(int(k),len(z)):];v=np.take_along_axis(Q,ix,1);ss=softmax(v/max(float(temp),1e-6)+np.log(np.maximum(w[ix],1e-20)),axis=1);return np.einsum('ij,ijk->ik',ss,z[ix])
class Trial:
 def __init__(self,f,cfg):
  self.f=f;self.cfg=cfg;self.p=nz(V1/f'TRAIN_f{f}.npz');self.n=len(self.p['drug']);self.name=cfg['feature'];self.fitcache={};self.predcache={};self.targetcache={};self.exclusions=[];self.av=np.ones(self.n,bool);self.X=None;self.low=[];self.pairs=nz(R/'PAIRED_TRAIN_BETA.npz');self.plans=splits(self.p['group'],4)
  if self.name in ['S1','S3']:
   self.ref=nz(R/'features/S1_REFERENCE.npz');lu={d:i for i,d in enumerate(self.ref['drug'])};self.ri=np.array([lu.get(d,-1) for d in self.p['drug']]);self.av=self.ri>=0
   if self.name=='S3':
    with np.load(reg(R/'features/LINCS_TARGET_CONSENSUS.npz'),allow_pickle=False) as zz:gen={'target':zz['target'],'value':zz['value']}
    tg=json.loads((V1/'CHEMBL_TARGET_GENES.json').read_text());gl={g:i for i,g in enumerate(gen['target'])};self.tgene=np.zeros((len(self.ref['target']),978));self.tmask=np.zeros(len(self.ref['target']),bool)
    for j,t in enumerate(self.ref['target']):
     ix=[gl[g] for g in tg.get(t,[]) if g in gl]
     if ix:self.tgene[j]=np.nanmean(gen['value'][ix],0);self.tmask[j]=True
  else:
   if self.name.startswith('Gaussian_'):
    se=int(self.name.split('_')[1]);dim=[64,256,2048][(se-8100)%3];ids=pd.read_csv(V1/'DRUG_IDENTITIES.csv').drug.to_numpy(str);v=dict(drug=ids,value=np.random.default_rng(se).normal(size=(len(ids),dim)),available=np.ones(len(ids),bool))
   else:
    path=R/'features'/(self.name+'.npz')
    if not path.exists():path=V1/'features'/(f'Positive_0p5_f{f}.npz' if self.name=='Positive_0p5' else self.name+'.npz')
    v=nz(path)
   if self.name=='ChEMBL_targetAffinity':v['value']=np.c_[np.nan_to_num(v['value']),np.isfinite(v['value']).astype(float)]
   lu={d:i for i,d in enumerate(v['drug'])};self.X=np.full((self.n,v['value'].shape[1]),np.nan);self.av=np.zeros(self.n,bool)
   for j,d in enumerate(self.p['drug']):
    if d in lu:self.X[j]=v['value'][lu[d]];self.av[j]=v['available'][lu[d]]
  if cfg.get('matched_support'):
   ref=Trial(f,dict(feature=cfg['matched_support']));self.av&=ref.av
  if cfg.get('target','T0') in ['T3','T3_slope'] or self.name=='Positive_0p5':
   for dose in [.05,.5]:
    q=nz(V1/f'LOWDOSE_f{f}_{dose}.npz');self.low.append((dose,dict(drug=q['drug'],group=q['group'],context=contexts(q['X']),mean=q['Y'].mean(0))))
  if cfg.get('model')=='CellRidge':self.Y=nz(V1/f'TRAIN_Y_f{f}.npz')['Y']
  self.moa=None;self.searchlog=[]
 def target(self,ix):
  key=tuple(ix)
  if key not in self.targetcache:
   w=weights(self.p['group'][ix]);self.targetcache[key]=(proj(self.p['mean'][ix],self.p['context'][ix],w),w)
  return self.targetcache[key]
 def cell_embedding(self,fit,query):
  key=hashlib.sha256((str(self.f)+'|'+','.join(self.p['drug'][fit])+'|'+','.join(self.p['drug'][query])).encode()).hexdigest();folder=R/'cell_embedding_cache';folder.mkdir(exist_ok=True);file=folder/(key+'.npz')
  if file.exists():
   with np.load(file,allow_pickle=False) as z:return z['x'],z['q']
  x=self.p['X'][:,fit].reshape(-1,978);q=self.p['X'][:,query].reshape(-1,978);x,q=preprocess(x,q);x,q=pca_input(x,q,8);tmp=file.with_suffix('.tmp'+str(os.getpid()))
  with tmp.open('wb') as h:np.savez_compressed(h,x=x,q=q)
  tmp.replace(file);return x,q
 def reliability(self,fit):
  ids=self.pairs[f'f{self.f}_drug'];lu={d:i for i,d in enumerate(ids)};js=[j for j,d in enumerate(self.p['drug'][fit]) if d in lu];ii=[lu[self.p['drug'][fit[j]]] for j in js]
  rd=np.ones(len(fit));rg=np.ones(978)
  if len(ii)>=4:
   a=self.pairs[f'f{self.f}_plate6'][ii].copy();b=self.pairs[f'f{self.f}_plate14'][ii].copy();w=weights(self.p['group'][fit[js]]);a-=np.average(a,0,weights=w);b-=np.average(b,0,weights=w);r=corr(a,b);rd[:]=np.clip(np.average(r,weights=w),0,1);rd[js]=np.clip(r,0,1);rg=np.clip(np.average(a*b,0,weights=w)/np.maximum(np.sqrt(np.average(a*a,0,weights=w)*np.average(b*b,0,weights=w)),1e-20),.05,1)
  return rd,rg
 def train_target(self,fit,shrink=.5):
  y,w=self.target(fit);y=y.copy();typ=self.cfg.get('target','T0');gw=np.ones(978);tasks=None
  if typ=='T1':
   rd,rg=self.reliability(fit);U=basis(y,w,min(20,len(fit)-2));sub=y@U.T@U;y=sub+(1-shrink*(1-rd[:,None]))*(y-sub)
  elif typ=='T2':rd,gw=self.reliability(fit)
  elif typ in ['T3','T3_slope']:
   vals=[]
   for dose,lp in self.low:
    lut={d:i for i,d in enumerate(lp['drug'])};jj=np.array([j for j,d in enumerate(self.p['drug'][fit]) if d in lut]);ii=np.array([lut[self.p['drug'][fit[j]]] for j in jj]);v=np.full_like(y,np.nan)
    if len(ii):v[jj]=proj(lp['mean'][ii],lp['context'][ii],weights(lp['group'][ii]))
    vals.append(v)
   cube=np.stack(vals+[y],1);tasks=cube;ld=np.log10([.05,.5,5]);slope=np.zeros_like(y)
   for j in range(len(y)):
    ok=np.isfinite(cube[j]).all(1)
    if ok.sum()>1:xx=ld[ok]-ld[ok].mean();slope[j]=xx@cube[j,ok]/(xx@xx)
   if typ=='T3_slope':y=slope
  return y,w,gw,tasks
 def source(self,fit,query):
  if self.name not in ['S1','S3']:
   if self.name=='Positive_0p5':
    lp=self.low[1][1];lu={d:i for i,d in enumerate(lp['drug'])}
    def val(ix):
     ii=np.array([lu[self.p['drug'][j]] for j in ix]);return proj(lp['mean'][ii],lp['context'][ii],weights(lp['group'][ii]))
    return val(fit),val(query)
   return self.X[fit],self.X[query]
  rf=self.ref;fit_hash=hashlib.sha256('|'.join(sorted(self.p['drug'][fit])).encode()).hexdigest();cache=R/'feature_fit_cache';cache.mkdir(exist_ok=True);file=cache/(fit_hash+'.npz');allidx=np.r_[self.ri[fit],self.ri[query]]
  if file.exists():
   with np.load(file,allow_pickle=False) as zz:z=zz['z'][allidx];stats=zz['stats']
  else:
   foreign=~np.isin(rf['drug'],self.p['drug'][fit]);keep=~rf['exclude'][foreign].any(0);active=csr_matrix(rf['active'][keep].astype(float));counts=np.asarray(active.sum(0)).ravel();den=np.maximum(counts,1);sim=np.maximum(rf['similarity'][:,keep].astype(float)-.3,0);raw=np.asarray(active.T@sim.T).T/den
   bgmask=keep[rf['background_index']];bsim=np.maximum(rf['background_similarity'][bgmask][:,keep].astype(float)-.3,0);bg=np.asarray(active.T@bsim.T).T/den;mu=bg.mean(0) if len(bg) else np.zeros(len(counts));sd=bg.std(0) if len(bg) else np.ones(len(counts));allz=(raw-mu)/np.maximum(sd,.001);allz[:,counts==0]=0;stats=np.array([keep.sum(),(~keep).sum(),(counts>0).sum(),bgmask.sum()]);tmp=file.with_suffix('.tmp'+str(os.getpid()))
   with tmp.open('wb') as hh:np.savez_compressed(hh,z=allz,stats=stats,fit_drug=self.p['drug'][fit])
   tmp.replace(file);z=allz[allidx]
  self.exclusions.append(dict(fit_hash=fit_hash,fit_drugs=len(fit),references_retained=int(stats[0]),references_excluded=int(stats[1]),target_columns_nonempty=int(stats[2]),background_retained=int(stats[3])))
  if self.name=='S3':
   zz=np.maximum(z,0)*self.tmask;z=(zz@np.nan_to_num(self.tgene))/np.maximum(zz.sum(1,keepdims=True),1e-12)
  return z[:len(fit)],z[len(fit):]
 def base(self,fit,query,seed=None,shrink=.5):
  key=(tuple(fit),tuple(query),seed,shrink)
  if key in self.fitcache:return self.fitcache[key]
  y,w,gw,tasks=self.train_target(fit,shrink)
  if seed is not None:
   pi=group_perm(self.p['group'][fit],seed);y=y[pi]
   if tasks is not None:tasks=tasks[pi]
  sg=np.sqrt(gw);z,U,offset=target_basis(y*sg,w,self.cfg.get('K',10),self.cfg.get('basis','PCA'));U=U/sg;offset=offset/sg
  x,q=self.source(fit,query);x,q=preprocess(x,q);x,q=pca_input(x,q,self.cfg.get('input_dim',32),self.cfg.get('whiten',False));K,Q=gram(x,q,'cosine');KL,QL=gram(x,q,'linear');rawY,_=self.target(fit)
  amplitude_target=rawY if seed is None else rawY[group_perm(self.p['group'][fit],seed)]
  amp=np.linalg.norm(amplitude_target,axis=1);logamp=ridge_features(x,q,np.log(np.maximum(amp,1e-8))[:,None],w,10).ravel();amp_pred=np.exp(np.clip(logamp,-30,np.log(max(np.max(amp),1e-8)*2)))
  # Three label-free-at-query uncertainty summaries. Neighbour disagreement uses fit labels.
  var=np.maximum(1-np.sum(Q*solve(K+.1*np.eye(len(K)),Q.T,assume_a='pos').T,axis=1),0)
  ix=np.argsort(Q,axis=1)[:,-min(10,len(fit)):];zz=z[ix];agreement=np.linalg.norm(zz.mean(1),axis=1)/np.maximum(np.linalg.norm(zz,axis=2).mean(1),1e-12)
  ens=np.stack([soft_retrieve(Q,z,w,k,.3) for k in [3,10,20]],1);disagreement=np.mean(np.var(ens,1),1)
  b=dict(x=x,q=q,z=z,U=U,offset=offset,w=w,K=K,Q=Q,KL=KL,QL=QL,y=y,rawY=rawY,tasks=tasks,amp=amp_pred,conf=np.column_stack([-var,agreement,-disagreement]),fit=fit,query=query)
  self.fitcache[key]=b;return b
 def params(self):
  m=self.cfg['model']
  if m in ['Ridge','CellRidge','Stack']:a=[.1,1,10,100]
  elif m in ['CCA','KCCA_cosine','KCCA_rbf']:a=[.01,.1,1]
  elif m in ['Procrustes','DirectPrior']:a=[1]
  elif m=='Sinkhorn':a=[.03,.1,.3,1]
  elif m in ['ContrastiveRetrieve','ContrastiveLinear']:a=[.05,.2,.5]
  elif m=='Soft':a=[(k,t) for k in range(1,41) for t in [.03,.1,.3,1.,3.]]
  elif m=='DistanceKNN':a=[k for k in range(1,41)]
  elif m=='MetricSoft':a=[(k,t) for k in [1,3,5,10,20,40] for t in [.03,.1,.3,1.,3.]]
  elif m=='ProgramKNN':a=[1,2,3,5,10,20,40]
  else:raise ValueError(m)
  if self.cfg.get('target')=='T1':return [(p,sh) for p in a for sh in [0,.5,1]]
  return a
 def predict(self,fit,query,param,seed=None,confidence=False):
  result=np.zeros((len(query),978));conf=np.full((len(query),3),-1e12);fit=np.array(fit)[self.av[fit]];qy=np.array(query)[self.av[query]]
  if self.cfg.get('strong_only') and len(fit):
   yy,_=self.target(fit);norm=np.linalg.norm(yy,axis=1);fit=fit[norm>=np.quantile(norm,1/3)]
  if len(fit)<6 or not len(qy):return (result,conf) if confidence else result
  par,shrink=param if self.cfg.get('target')=='T1' else (param,.5)
  b=self.base(fit,qy,seed,shrink);x,q,z,U,w=b['x'],b['q'],b['z'],b['U'],b['w'];m=self.cfg['model'];K,Q=b['K'],b['Q']
  if m=='Ridge':
   pp=regress(b['KL'],b['QL'],z,w,float(par))
   if self.cfg.get('target')=='T3' and b['tasks'] is not None:
    # Shared low-rank output regression over three dose tasks, masked observations.
    zz=[]
    for di in range(3):
     tar=b['tasks'][:,di];ok=np.isfinite(tar).all(1)
     if ok.sum()>=6:
      uz=basis(tar[ok],w[ok],min(self.cfg.get('K',10),ok.sum()-2));zp=regress(b['KL'][np.ix_(ok,ok)],b['QL'][:,ok],tar[ok]@uz.T,w[ok],float(par))@uz;zz.append(zp)
    # Shared direction, 5uM amplitude; dose tasks cannot introduce held labels.
    if zz:
     yy=np.mean([v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12) for v in zz],0);val=yy/np.maximum(np.linalg.norm(yy,axis=1,keepdims=True),1e-12)*b['amp'][:,None]
    else:val=pp@U+b['offset']
   else:val=pp@U+b['offset']
  elif m in ['Soft','DistanceKNN','MetricSoft','ProgramKNN']:
   if m=='MetricSoft':
    key=('metric',id(b));
    if key not in self.predcache:self.predcache[key]=(ridge_features(x,x,z,w,10),ridge_features(x,q,z,w,10))
    ex,eq=self.predcache[key];_,Q=gram(ex,eq,'cosine')
   if m=='DistanceKNN':pp=c.knn(Q,z,w,int(par))
   elif m=='ProgramKNN':
    if isinstance(par,(list,tuple,np.ndarray)):pp=np.column_stack([c.knn(Q,z[:,j:j+1],w,int(par[min(j,len(par)-1)]))[:,0] for j in range(z.shape[1])])
    else:pp=c.knn(Q,z,w,int(par))
   else:pp=soft_retrieve(Q,z,w,par[0],par[1])
   val=pp@U+b['offset']
   if self.cfg.get('direction_amp',m=='Soft'):val=val/np.maximum(np.linalg.norm(val,axis=1,keepdims=True),1e-12)*b['amp'][:,None]
  elif m in ['CCA','KCCA_cosine','KCCA_rbf']:
   xx,qq=x,q
   if m.startswith('KCCA'):
    kk,qqk=gram(x,q,'cosine' if m.endswith('cosine') else 'rbf');mu=kk.mean(0);kc=kk-mu[None]-mu[:,None]+mu.mean();qc=qqk-mu[None]-qqk.mean(1,keepdims=True)+mu.mean();ev,vec=eigh(kc,check_finite=False);ix=np.argsort(ev)[::-1][:min(32,len(x)-2)];u=vec[:,ix]/np.sqrt(np.maximum(ev[ix],1e-8));xx=kc@u;qq=qc@u
   ex,eq=cca_embed(xx,qq,z,w,float(par));pp=ridge_features(ex,eq,z,w,1);val=pp@U+b['offset']
  elif m in ['Procrustes','Sinkhorn']:
   xx,qq=pca_input(x,q,min(z.shape[1],x.shape[1]));n=min(xx.shape[1],z.shape[1]);xm=np.average(xx[:,:n],0,weights=w);zm=np.average(z[:,:n],0,weights=w);xa=xx[:,:n]-xm;za=z[:,:n]-zm;rot,s=orthogonal_procrustes(xa*np.sqrt(w[:,None]),za*np.sqrt(w[:,None]));scale=s/max(np.sum(xa*xa*w[:,None]),1e-12);ex=xa@rot*scale+zm;eq=(qq[:,:n]-xm)@rot*scale+zm
   if m=='Procrustes':pp=np.zeros((len(q),z.shape[1]));pp[:,:n]=eq
   else:
    dist=np.maximum(np.sum(ex*ex,1)[:,None]+np.sum(z[:,:n]**2,1)[None]-2*ex@z[:,:n].T,0);eps=float(par)*max(np.median(dist),1e-8);ker=np.exp(-np.minimum(dist/eps,100));u=np.ones(len(x));v=np.ones(len(x))
    for _ in range(60):u=w/np.maximum(ker@v,1e-20);v=w/np.maximum(ker.T@u,1e-20)
    plan=u[:,None]*ker*v[None];transport=plan@z/np.maximum(plan.sum(1,keepdims=True),1e-20);pp=soft_retrieve(Q,transport,w,10,.3)
   val=pp@U+b['offset']
  elif m in ['ContrastiveRetrieve','ContrastiveLinear']:
   key=('contrast',id(b),float(par))
   if key not in self.predcache:
    import torch
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(20261025);nx=nn.Sequential(nn.Linear(x.shape[1],64),nn.Tanh(),nn.Linear(64,32));ny=nn.Sequential(nn.Linear(z.shape[1],32));assert sum(t.numel() for net in [nx,ny] for t in net.parameters())<2000000
    tx=torch.tensor(x,dtype=torch.float32);tz=torch.tensor(z/np.maximum(z.std(0),1e-8),dtype=torch.float32);ww=torch.tensor(w,dtype=torch.float32);opt=torch.optim.Adam(list(nx.parameters())+list(ny.parameters()),lr=.003,weight_decay=.01);t0=time.monotonic()
    for ep in range(100):
     opt.zero_grad();ax=nn.functional.normalize(nx(tx),dim=1);az=nn.functional.normalize(ny(tz),dim=1);logits=ax@az.T/float(par);labels=torch.arange(len(x));loss=(nn.functional.cross_entropy(logits,labels,reduction='none')*ww).sum()+(nn.functional.cross_entropy(logits.T,labels,reduction='none')*ww).sum();loss.backward();opt.step()
     if time.monotonic()-t0>1200:raise TimeoutError('contrastive fit>20min')
    with torch.inference_mode():ex=nn.functional.normalize(nx(tx),dim=1).numpy();ey=nn.functional.normalize(ny(tz),dim=1).numpy();eq=nn.functional.normalize(nx(torch.tensor(q,dtype=torch.float32)),dim=1).numpy()
    self.predcache[key]=(ex,ey,eq)
   ex,ey,eq=self.predcache[key];pp=soft_retrieve(eq@ey.T,z,w,20,.1) if m=='ContrastiveRetrieve' else ridge_features(ex,eq,z,w,10);val=pp@U+b['offset']
  elif m=='CellRidge':
   # Cell residual target uses same full-X1 contexts. It is fitted only on fit drugs.
   cellkey=('cell_design',id(b))
   if cellkey not in self.predcache:
    yf=self.Y[:,fit].astype(float);ctx=self.p['context'][fit]
    for cc in np.unique(ctx):
     jj=ctx==cc;yf[:,jj]-=np.average(yf[:,jj],axis=1,weights=w[jj])[:,None]
    if seed is not None:yf=yf[:,group_perm(self.p['group'][fit],seed)]
    cx,cq=self.cell_embedding(fit,qy);dx,dq=pca_input(x,q,16);dx=np.tile(dx,(47,1));dq=np.tile(dq,(47,1));design=np.c_[dx,cx,(dx[:,:,None]*cx[:,None,:]).reshape(len(dx),-1)];qd=np.c_[dq,cq,(dq[:,:,None]*cq[:,None,:]).reshape(len(dq),-1)];tar=yf.reshape(-1,978)@np.linalg.pinv(U);ww=np.tile(w,47)/47;dm=np.average(design,0,weights=ww);tm=np.average(tar,0,weights=ww);dd=design-dm;self.predcache[cellkey]=(dd.T@(dd*ww[:,None]),dd.T@((tar-tm)*ww[:,None]),qd-dm,tm,len(ww))
   A,C,qd,tm,nn=self.predcache[cellkey];coef=solve(A+float(par)/nn*np.eye(len(A)),C,assume_a='pos');pp=(qd@coef+tm).reshape(47,len(qy),-1).mean(0);val=pp@U+b['offset']
  elif m=='DirectPrior':
   xf,qf=self.source(fit,qy);xf=np.nan_to_num(xf);qf=np.nan_to_num(qf);target,_=self.target(fit)
   if seed is not None:target=target[group_perm(self.p['group'][fit],seed)]
   amp=float(np.clip(np.sum(xf*target*w[:,None])/max(np.sum(xf*xf*w[:,None]),1e-20),0,100));val=qf*amp
  elif m=='Stack':
   cachekey=('stack',tuple(fit),tuple(qy),seed)
   if cachekey not in self.predcache:
    trainmeta=[];querymeta=[]
    for source in self.cfg.get('stack_sources',['KPGT','S1','S3','CIGS_cells','MoA','L1000_mean']):
     sub=Trial(self.f,dict(feature=source,model='Ridge',K=10,basis='PCA',target='T0'));oo=np.zeros((len(fit),978));oc=np.zeros((len(fit),3));par2=sub.choose(fit,seed)
     for it,iv in splits(self.p['group'][fit],3,20261023):oo[iv],oc[iv]=sub.predict(fit[it],fit[iv],par2,seed,True)
     pp,pc=sub.predict(fit,qy,par2,seed,True);uu=np.linalg.pinv(U);trainmeta.append(np.c_[oo@uu,sub.av[fit],np.clip(oc,-10,10)]);querymeta.append(np.c_[pp@uu,sub.av[qy],np.clip(pc,-10,10)]);del sub
    sx,sq=preprocess(np.concatenate(trainmeta,1),np.concatenate(querymeta,1));self.predcache[cachekey]=(sx,sq)
   sx,sq=self.predcache[cachekey];val=ridge_features(sx,sq,z,w,float(par))@U+b['offset']
  else:raise ValueError(m)
  lookup={j:i for i,j in enumerate(query)};ii=[lookup[j] for j in qy];result[ii]=val;conf[ii]=b['conf'];result=proj(result,self.p['context'][query],weights(self.p['group'][query]),self.av[query])
  return (result,conf) if confidence else result
 def choose(self,fit,seed=None):
  pars=self.params();scores=np.zeros(len(pars));den=0.;plans=splits(self.p['group'][fit],3,20261023)
  code_scores=np.zeros((len(pars),self.cfg.get('K',10)))
  for it,iv in plans:
   va=fit[iv];y,w=self.target(va);ok=self.av[va];den+=sum(w*ok)
   for pi,par in enumerate(pars):
    check_budget();pp=self.predict(fit[it],va,par,seed);scores[pi]+=np.sum(np.mean((pp-y)**2,1)*w*ok)
    if self.cfg['model']=='ProgramKNN' and self.av[fit[it]].sum()>=6 and ok.any():
     b=self.base(fit[it][self.av[fit[it]]],va[ok],seed);uu=np.linalg.pinv(b['U']);err=((pp[ok]-y[ok])@uu)**2;code_scores[pi,:err.shape[1]]+=np.sum(err*w[ok,None],axis=0)
   # Independent transform cache per fit/query inner block avoids quadratic memory growth.
   self.fitcache.clear();self.predcache.clear()
  losses=scores/max(den,1e-20)
  for par,loss in zip(pars,losses):self.searchlog.append(dict(fold=self.f,fit_drugs=len(fit),fit_hash=hashlib.sha256('|'.join(self.p['drug'][fit]).encode()).hexdigest(),permutation=seed,param=par,inner_raw_MSE=float(loss)))
  if self.cfg['model']=='ProgramKNN':return [pars[i] for i in np.argmin(code_scores,axis=0)]
  return pars[int(np.argmin(losses))]
 def run(self,seed=None,plans=None):
  plans=self.plans if plans is None else plans;pred=np.zeros((self.n,978));raw=pred.copy();truth=pred.copy();conf=np.full((self.n,3),-1e12);block=np.zeros(self.n,int);sel=[];selective={};programscores=np.zeros((self.n,2))
  for bi,(fit,va) in enumerate(plans):
   fraction=float(self.cfg.get('train_fraction',1))
   if fraction<1:
    ug=np.unique(self.p['group'][fit]);pick=np.random.default_rng(20261025+bi).permutation(ug)[:max(3,int(np.ceil(len(ug)*fraction)))];fit=fit[np.isin(self.p['group'][fit],pick)]
   check_budget();par=self.choose(fit,seed);oo=np.zeros((len(fit),978));yt=oo.copy();oc=np.full((len(fit),3),-1e12)
   for it,iv in splits(self.p['group'][fit],3,20261023):
    oo[iv],oc[iv]=self.predict(fit[it],fit[iv],par,seed,True);yt[iv]=self.target(fit[iv])[0]
   wf=weights(self.p['group'][fit]);lam=float(np.clip(np.sum(yt*oo*wf[:,None])/max(np.sum(oo*oo*wf[:,None]),1e-20),0,1));rv,cv=self.predict(fit,va,par,seed,True);raw[va]=rv;pred[va]=lam*rv;truth[va]=self.target(va)[0];conf[va]=cv;block[va]=bi
   for ci,cn in enumerate(['GP_variance','neighbour_agreement','ensemble_disagreement']):
    good=self.av[fit]&(oc[:,ci]>-1e11)
    for portion in [.2,.4,.6,.8,1.]:
     threshold=float(np.quantile(oc[good,ci],1-portion)) if good.any() else np.inf;mask=self.av[va]&(cv[:,ci]>=threshold);key=f'{cn}_{portion}'
     if key not in selective:selective[key]=dict(pred=np.zeros_like(pred),mask=np.zeros(self.n,bool),threshold=[])
     vv=proj(pred[va],self.p['context'][va],weights(self.p['group'][va]),mask);selective[key]['pred'][va]=vv;selective[key]['mask'][va]=mask;selective[key]['threshold'].append(threshold)
   sel.append(dict(block=bi,param=par,lambda_beta=lam,fit=self.p['drug'][fit].tolist(),holdout=self.p['drug'][va].tolist(),fold=self.f));self.fitcache.clear();self.predcache.clear()
  return dict(pred=pred,raw=raw,truth=truth,conf=conf,block=block,selected=sel,selective=selective,available=self.av,exclusions=self.exclusions)

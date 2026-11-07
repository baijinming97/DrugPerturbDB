from ff_common import *
def bulk_repeats(ds,meta,cm,codes,source_rows,cp,mode,name):
 ng=len(cm);n=len(meta);xs=np.empty((n,ds.shape[1]),np.float32)
 for lo in range(0,n,4096):
  rr=source_rows[lo:lo+4096];xs[lo:lo+4096]=ds[int(rr[0]):int(rr[-1])+1][rr-rr[0]]
 pg,_=pd.factorize(pd.MultiIndex.from_arrays([codes,meta.plate.to_numpy()]),sort=False);npg=int(pg.max())+1;pc=np.zeros(npg,int);pc[pg]=codes
 G=sparse.csr_matrix((np.ones(n),(codes,np.arange(n))),shape=(ng,n));H=sparse.csr_matrix((np.ones(n),(pg,np.arange(n))),shape=(npg,n));out=[]
 for axis,pos in [('native',np.arange(xs.shape[1])),('common',cp)]:
  # Normalize bounded row blocks without allocating a second full float64 matrix.
  z=np.empty((n,len(pos)),np.float64);ok=np.zeros(n,bool)
  for lo in range(0,n,4096):
   hi=min(n,lo+4096);block=xs[lo:hi][:,pos].astype(np.float64)
   block-=block.mean(1,keepdims=True);norm=np.linalg.norm(block,axis=1)
   ok[lo:hi]=norm>1e-12;block/=np.where(ok[lo:hi],norm,1)[:,None];z[lo:hi]=block
  del block
  nc=np.bincount(codes,weights=ok,minlength=ng);np_=np.bincount(pg,weights=ok,minlength=npg);den=nc*nc-np.bincount(pc,weights=np_*np_,minlength=ng);q=np.zeros(ng)
  for lo in range(0,len(pos),128):
   a=G@z[:,lo:lo+128];b=H@z[:,lo:lo+128];q+=(a*a).sum(1)-np.bincount(pc,weights=(b*b).sum(1),minlength=ng)
  v=np.divide(q,den,out=np.full(ng,np.nan),where=den>0);take=np.flatnonzero(den>0);t=cm.iloc[take][['drug','cell','dose','time','panel','normalization']].copy();t['dataset']=name;t['mode']=mode;t['axis']=axis;t['genes']=len(pos);t['pcc']=v[take];t['pairs']=(den[take]/2).astype(np.int64);t['physical_units']=nc[take].astype(int);t['plates']=np.bincount(pc,minlength=ng)[take];out.extend(t.to_dict('records'));del z
  print(name,mode,axis,'all repeat pairs evaluated',len(t),flush=True)
 return out

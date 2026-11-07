"""Record input/output checksums and independently verify scalar statistics."""
from eval_lib import *
import traceback,resource,subprocess
def lock_protocol():
 p=R/'PROTOCOL.json';h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert (R/'PROTOCOL.sha256').read_text().strip()==h
 shared=Path(__file__).parent
 for line in (shared/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1)
  assert hashlib.sha256((shared/name).read_bytes()).hexdigest()==digest,name
 dump('START.json',dict(job=os.environ['TAHOE_REPRO_RUN_ID'],protocol_sha256=h,TEST='CLOSED',GPU=0,start=time.time()))
def summary_records(rows,path='RESULTS.csv'):
 t=pd.DataFrame(rows);t.to_csv(R/'PER_DRUG_RESULTS.csv.gz',index=False)
 out=[]
 if len(t):
  keys=[x for x in ['setting','model','m','direction','reference','variant'] if x in t]
  for key,s in t.groupby(keys,dropna=False):
   key=key if isinstance(key,tuple) else (key,)
   labels=dict(zip(keys,key));z=paired_summary(s)
   ff=s.groupby('fold').apply(lambda a:np.average(a.full_PCC-a.B_full_PCC,weights=a.weight),include_groups=False)
   z.update(folds_improved=int((ff>1e-8).sum()),folds_tied=int((abs(ff)<=1e-8).sum()),folds_total=len(ff))
   out.append(dict(**labels,scope='ALL',**z))
   for f,ss in s.groupby('fold'):out.append(dict(**labels,scope='fold'+str(f),**paired_summary(ss)))
 pd.DataFrame(out).to_csv(R/path,index=False)
 return pd.DataFrame(out)
def verify_slice(pred,Y,X,w,observed=()):
 from scipy.stats import pearsonr
 keep=np.ones(len(Y),bool);keep[list(observed)]=False
 a=pred[keep];b=Y[keep];calc=np.array([[pearsonr(x,y).statistic if np.std(x)>1e-8 else 0 for x,y in zip(ac,bc)] for ac,bc in zip(a,b)])
 direct=float(w@calc.mean(0));fast,_=metric_rows(pred,Y,X,w,observed)
 mse=float(np.einsum('cdg,d->',(a-b)**2,w)/(len(a)*a.shape[-1]))
 err=max(abs(direct-w@fast[:,2]),abs(mse-w@fast[:,3]))
 assert err<1e-8,err
 return dict(status='PASS',independent_scipy_PCC=direct,independent_MSE=mse,max_difference=err)
def receipt(status,extra=None):
 sec=time.monotonic()-START
 dump('RESOURCE_RECEIPT.json',dict(status=status,job=os.environ['TAHOE_REPRO_RUN_ID'],allocated_CPUs=CPUS,elapsed_seconds=sec,allocated_core_hours=sec*CPUS/3600,process_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU=0,TEST='CLOSED',**(extra or {})))
def complete(status='COMPLETE',extra=None):
 finish_reads('WORKFLOW');receipt(status,extra);dump('COMPLETE.json',dict(status=status,job=os.environ['TAHOE_REPRO_RUN_ID'],TEST='CLOSED',**(extra or {})))
def fail(error):
 dump('FAILURE_'+os.environ['TAHOE_REPRO_RUN_ID']+'.json',dict(error=repr(error),traceback=traceback.format_exc()))
 receipt('ERROR')
def tablemd(t):
 if not len(t):return '无符合条件的条目。'
 a=t.copy()
 for c in a.select_dtypes('number'):a[c]=a[c].map(lambda x:f'{x:.6g}')
 return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in a.itertuples(index=False,name=None))
def weighted_ridge(x,y,xt,w,alpha):
 from scipy.linalg import solve
 w=w/w.sum()*len(w);mx=np.average(x,axis=0,weights=w);my=np.average(y,axis=0,weights=w)
 a=x-mx;coef=solve(a.T@(a*w[:,None])+alpha*np.eye(x.shape[1]),a.T@((y-my)*w[:,None]),assume_a='pos')
 return (xt-mx)@coef+my,dict(coef=coef,xmean=mx,ymean=my)
def shrink(y,p,w):
 v=float(np.einsum('cdg,d->',y*p,w));den=float(np.einsum('cdg,d->',p*p,w))
 return float(np.clip(v/den,0,1)) if den>1e-20 else 0.

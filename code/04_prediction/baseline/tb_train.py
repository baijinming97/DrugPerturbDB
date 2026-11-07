from tb_common import *
import argparse, ast, copy, signal, time, traceback
import torch
from torch import nn
import torch.nn.functional as F

DEVICE='cpu'
STOP=False
def request_stop(signum,frame):
    global STOP
    STOP=True

class BasalMLP(nn.Module):
    def __init__(self,depth=3,width=2048):
        super().__init__();layers=[];n=978
        for _ in range(depth): layers.extend([nn.Linear(n,width),nn.ReLU(),nn.Dropout(.1)]);n=width
        layers.append(nn.Linear(n,978));self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)

def sample_pcc(p,y):
    p=p-p.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
    return (p*y).sum(1)/torch.sqrt((p.square().sum(1)*y.square().sum(1)).clamp_min(1e-12))
def loss_on_delta(p,x1,x2):
    mse=F.mse_loss(x1+p,x2);pcc=sample_pcc(p,x2-x1).mean()
    return .5*mse+.5*(1-pcc)
def seed(s):
    np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def state_cpu(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def save_torch(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.part');torch.save(x,t);t.replace(p)
def tensor_pair(ids):
    x=np.load(V/'02_data/X1.npy',mmap_mode='r');z=np.load(V/'02_data/X2.npy',mmap_mode='r')
    a=np.asarray(x[ids],np.float32);b=np.asarray(z[ids],np.float32)
    d=np.load(V/'02_data/delta.npy',mmap_mode='r')
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(b-a,d[ids])
    return torch.tensor(a,device=DEVICE),torch.tensor(b,device=DEVICE)
@torch.no_grad()
def forward(net,x):
    net.eval();p=torch.cat([net(x[k:k+512]) for k in range(0,len(x),512)])
    assert torch.isfinite(p).all();return p
def valid_scores(net,xy):
    x,z=xy;p=forward(net,x);m=metric(p.cpu().numpy(),(z-x).cpu().numpy())
    assert np.isfinite(m['PCC']).all()
    return dict(PCC=float(m['PCC'].mean()),MSE=float(m['MSE'].mean()))
def train_epoch(net,opt,xy,b,f,epoch,seed_value):
    x,z=xy;net.train()
    s=int(hashlib.sha256(f'batch|{seed_value}|{b}|{f}|{epoch}'.encode()).hexdigest()[:16],16)
    order=np.random.default_rng(s).permutation(len(x))
    for batch,k in enumerate(range(0,len(x),512)):
        ix=torch.as_tensor(order[k:k+512],device=DEVICE)
        ds=int(hashlib.sha256(f'dropout|{seed_value}|{b}|{f}|{epoch}|{batch}'.encode()).hexdigest()[:8],16)
        torch.manual_seed(ds);torch.cuda.manual_seed_all(ds)
        opt.zero_grad(set_to_none=True);loss=loss_on_delta(net(x[ix]),x[ix],z[ix]);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(norm);opt.step()

def fit(b,f,cfg,xy,va,smoke=False):
    p=protocol();key=f'{b}_f{f}_{cfg["id"]}';out=R/('smoke' if smoke else 'runs')/key;out.mkdir(parents=True,exist_ok=True)
    done=out/'COMPLETE.json'
    if done.exists():
        d=read(done)
        for path,h in d['artifacts'].items():assert sha(path)==h
        return d
    seed(p['seed']);net=BasalMLP(cfg['depth'],cfg['width']).to(DEVICE);opt=torch.optim.Adam(net.parameters(),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'],weight_decay=0)
    nparams=sum(v.numel() for v in net.parameters());initial=state_cpu(net)
    last=R/('smoke' if smoke else 'runs')/f'active_{b}_{f}.pt';epoch0=0;stale=0;best=-np.inf;history=[];previous_seconds=0.
    if last.exists():
        chk=torch.load(last,map_location=DEVICE,weights_only=True)
        if chk['key']==key:
            assert chk['lock_sha256']==sha(R/'LOCK.json')
            net.load_state_dict(chk['model']);opt.load_state_dict(chk['optimizer']);epoch0=chk['epoch'];stale=chk['stale'];best=chk['best'];history=chk['history'];previous_seconds=chk['seconds']
            bestcp=chk['selected_checkpoint']
            assert bestcp['epoch']<=epoch0
            save_torch(out/'selected.pt',bestcp)
        elif not smoke:
            assert (R/'runs'/chk['key']/'COMPLETE.json').exists(),'Unfinished different active fit'
    started=time.monotonic();job=os.environ['TAHOE_REPRO_RUN_ID'];attempt=out/f'ATTEMPT_{job}_{epoch0}.json'
    write(attempt,dict(job=job,start_utc=utc(),resume_epoch=epoch0,key=key,host=os.uname().nodename))
    maximum=2 if smoke else p['max_epochs'];executed=epoch0
    for epoch in range(epoch0+1,maximum+1):
        if stale>=p['patience']:break
        train_epoch(net,opt,xy,b,f,epoch,p['seed']);scores=valid_scores(net,va);executed=epoch
        history.append(dict(epoch=epoch,valid_PCC=scores['PCC'],valid_MSE=scores['MSE']))
        if scores['PCC']>best:
            best=scores['PCC'];stale=0
            save_torch(out/'selected.pt',dict(model=state_cpu(net),architecture=cfg,epoch=epoch,valid=scores,seed=p['seed'],lock_sha256=sha(R/'LOCK.json')))
        else:stale+=1
        if epoch%10==0 or STOP or epoch==maximum or stale>=p['patience']:
            save_torch(last,dict(key=key,model=state_cpu(net),optimizer=opt.state_dict(),epoch=epoch,stale=stale,best=best,history=history,seconds=previous_seconds+time.monotonic()-started,lock_sha256=sha(R/'LOCK.json'),selected_checkpoint=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)))
            write(out/'PROGRESS.json',dict(key=key,epoch=epoch,stale=stale,best_VALID_PCC=best,job=job,utc=utc()))
        if STOP:raise SystemExit(75)
    selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=True);net.load_state_dict(selected['model'])
    assert any(not torch.equal(initial[k],selected['model'][k]) for k in initial),'No learning'
    pred=forward(net,va[0]).cpu().numpy();np.save(out/'valid_prediction.npy',pred)
    check=valid_scores(net,va)
    assert abs(check['PCC']-selected['valid']['PCC'])<1e-10
    pd.DataFrame(history).to_csv(out/'HISTORY.csv',index=False)
    duration=previous_seconds+time.monotonic()-started
    d=dict(status='PASS',key=key,benchmark=b,fold=f,architecture=cfg,parameters=nparams,seed=p['seed'],selected_epoch=selected['epoch'],epochs=executed,valid=check,seconds=duration,job=job,utc=utc(),TEST='CLOSED',lock_sha256=sha(R/'LOCK.json'),artifacts={str(q):sha(q) for q in [out/'selected.pt',out/'valid_prediction.npy',out/'HISTORY.csv']})
    write(done,d);write(attempt,dict(**read(attempt),end_utc=utc(),status='COMPLETE',seconds=time.monotonic()-started))
    print(json.dumps(dict(key=key,epochs=executed,seconds=duration,status='PASS')),flush=True)
    del net,opt,initial;torch.cuda.empty_cache();return d

def select_export(b,f,results):
    ordered=sorted(results,key=lambda z:(-z['valid']['PCC'],z['valid']['MSE'],z['parameters'],z['architecture']['id']))
    fixed=next(z for z in results if z['architecture']['id']=='3H2048');selected=ordered[0]
    out=R/f'predictions/{b}_fold{f}';out.mkdir(parents=True,exist_ok=True)
    lock=dict(status='LOCKED_USING_VALID_ONLY',benchmark=b,fold=f,utc=utc(),fixed=fixed['key'],selected=selected['key'],candidates=[dict(key=z['key'],valid=z['valid']) for z in ordered],TEST='CLOSED')
    if (out/'SELECTION.json').exists():
        prev=read(out/'SELECTION.json');assert (prev['fixed'],prev['selected'])==(lock['fixed'],lock['selected'])
    else:write(out/'SELECTION.json',lock)
    ix=splits(b,f);x=np.load(V/'02_data/X1.npy',mmap_mode='r')
    for name,rr in [('FIXED_3H2048',fixed),('VALID_SELECTED',selected)]:
        for path,h in rr['artifacts'].items():assert sha(path)==h
        cfg=rr['architecture'];net=BasalMLP(cfg['depth'],cfg['width']).to(DEVICE)
        net.load_state_dict(torch.load(R/'runs'/rr['key']/'selected.pt',map_location='cpu',weights_only=True)['model'])
        for role in ['valid','test']:
            # TEST export accesses X1 only. No X2, delta, beta, alpha, or TEST metric access.
            tx=torch.tensor(np.asarray(x[ix[role]],np.float32),device=DEVICE);pred=forward(net,tx).cpu().numpy()
            if role=='valid':assert np.array_equal(pred,np.load(R/'runs'/rr['key']/'valid_prediction.npy'))
            np.savez_compressed(out/f'{name}_{role}.npz',sample_idx=ix[role],prediction=pred)
        del net,tx;torch.cuda.empty_cache()
    write(out/'EXPORT_COMPLETE.json',dict(status='PASS',TEST='PREDICTIONS_ONLY_LABELS_CLOSED',files={str(p):sha(p) for p in out.glob('*.npz')},selection_sha256=sha(out/'SELECTION.json')))

def engineering_checks():
    tree=ast.parse((R/'references/historical_train_exact_h2048.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['ExactH2048_978','sample_pcc']]
    ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<historical definitions only>','exec'),ns)
    seed(1337);old=ns['ExactH2048_978']();seed(1337);new=BasalMLP()
    assert sum(p.numel() for p in new.parameters())==12401618
    assert list(old.state_dict())==list(new.state_dict())
    for k in old.state_dict():assert torch.equal(old.state_dict()[k],new.state_dict()[k])
    x=torch.randn(8,978);old.eval();new.eval();assert torch.equal(old(x),new(x))
    y=torch.randn(8,978);assert torch.equal(ns['sample_pcc'](x,y),sample_pcc(x,y))
    # Save/reload an optimizer state, then take the same next stochastic CPU step.
    seed(1337);m=BasalMLP(2,1024).to(DEVICE);o=torch.optim.Adam(m.parameters(),lr=1e-4)
    xx=torch.randn(16,978,device=DEVICE);zz=xx+torch.randn_like(xx)*.1
    train_epoch(m,o,(xx,zz),'smoke',0,1,1337)
    save_torch(R/'smoke/resume_check.pt',dict(model=state_cpu(m),optimizer=o.state_dict()))
    checkpoint=torch.load(R/'smoke/resume_check.pt',weights_only=True)
    m2=BasalMLP(2,1024).to(DEVICE);m2.load_state_dict(checkpoint['model']);o2=torch.optim.Adam(m2.parameters(),lr=1e-4);o2.load_state_dict(checkpoint['optimizer'])
    train_epoch(m,o,(xx,zz),'smoke',0,2,1337);train_epoch(m2,o2,(xx,zz),'smoke',0,2,1337)
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in m.state_dict().items()),'Resume numerical parity'
    from scipy.stats import pearsonr
    a=np.random.default_rng(2).normal(size=(9,978));b=np.random.default_rng(3).normal(size=(9,978))
    assert np.max(abs(metric(a,b)['PCC']-np.array([pearsonr(p,q).statistic for p,q in zip(a,b)])))<1e-12
    assert np.isnan(metric(np.zeros((1,978)),a[:1])['PCC'][0])
    write(R/'smoke/ENGINEERING_CHECKS.json',dict(status='PASS',historical_initialization_and_forward_exact=True,parameters=12401618,loss_pcc_parity=True,resume_next_step_exact=True,independent_scipy_metric_parity=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--benchmark',choices=['A','B']);parser.add_argument('--fold',type=int,choices=range(5));a=parser.parse_args()
    allocated();verify_lock();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    assert int(os.environ['TAHOE_REPRO_CPUS'])>=8
    job=os.environ.get('TAHOE_REPRO_RUN_ID','local-cpu')
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if a.smoke:
        engineering_checks();ix=splits('A',0);xy=tensor_pair(ix['train']);va=tensor_pair(ix['valid'])
        result=fit('A',0,protocol()['architectures'][0],xy,va,smoke=True)
        write(R/'SMOKE.json',dict(status='PASS',result=result,seconds_per_epoch=result['seconds']/2,TEST='CLOSED',device='CPU_FP32_8_THREADS'))
        return
    assert a.benchmark and a.fold is not None and read(R/'SMOKE.json')['status']=='PASS'
    with (R/f'worker_{a.benchmark}_{a.fold}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage()
        for f in [a.fold]:
            ix=splits(a.benchmark,f);xy=tensor_pair(ix['train']);va=tensor_pair(ix['valid'])
            results=[fit(a.benchmark,f,c,xy,va) for c in protocol()['architectures']]
            select_export(a.benchmark,f,results);del xy,va;torch.cuda.empty_cache()
        write(R/f'WORKER_{a.benchmark}_{a.fold}_COMPLETE.json',dict(status='PASS',fits=6,job=job,utc=utc(),TEST='CLOSED'))
if __name__=='__main__':
    try:main()
    except Exception as e:
        write(R/f'logs/FAILURE_{os.environ.get("TAHOE_REPRO_RUN_ID","local")}.json',dict(error=str(e),traceback=traceback.format_exc(),utc=utc()));raise

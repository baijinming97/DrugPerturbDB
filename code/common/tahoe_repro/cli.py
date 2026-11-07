from pathlib import Path
import argparse,json,os,subprocess,sys,importlib.metadata
from .sources import select_sources,download_one,sha256

def repo_root(value):
    root=Path(value).expanduser().resolve()
    if not (root/'data/config/paper_scope.json').is_file():raise ValueError('Use --repo to point at the checked-out repository root')
    return root

def main():
    p=argparse.ArgumentParser(prog='drugperturbdb')
    p.add_argument('--repo',default='.',help='Path to the repository root')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor');sub.add_parser('figures');sub.add_parser('kernel-check');sub.add_parser('models-check');sub.add_parser('verify')
    sub.add_parser('example-check')
    c=sub.add_parser('results-check');c.add_argument('--workspace');c.add_argument('--baseline',help='Non-default baseline computation directory')
    b=sub.add_parser('baseline');b.add_argument('--workspace',help='Prepared workspace; defaults to data/prepared/_runtime');b.add_argument('--canonical');b.add_argument('--multiplate-manifest');b.add_argument('--multiplate-values');b.add_argument('--output');b.add_argument('--threads',type=int,default=2)
    b.add_argument('--include-decoder',action='store_true',help='Also recompute the measured-input G5 diagnostic')
    s=sub.add_parser('sources');s.add_argument('--dataset',action='append')
    d=sub.add_parser('download');d.add_argument('--dataset',action='append',required=True);d.add_argument('--destination',required=True);d.add_argument('--max-gib',type=float,required=True)
    w=sub.add_parser('workspace-init');w.add_argument('--workspace',required=True)
    w=sub.add_parser('workflow-run');w.add_argument('--workspace',required=True);w.add_argument('--script',required=True);w.add_argument('--cpus',type=int,default=4);w.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();root=repo_root(a.repo)
    if a.command=='verify':
        from .verification import verify_release
        print(json.dumps(verify_release(root),indent=2));return
    if a.command=='example-check':
        from .verification import verify_measured_examples
        print(json.dumps(verify_measured_examples(root),indent=2));return
    if a.command=='results-check':
        from .result_check import check_results
        workspace=Path(a.workspace).resolve() if a.workspace else root/'data/prepared/_runtime'
        result=check_results(root,workspace,a.baseline)
        print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASS' else 2)
    if a.command=='baseline':
        from .evaluate import evaluate
        workspace=Path(a.workspace).resolve() if a.workspace else root/'data/prepared/_runtime'
        physical=workspace/'4MAN/minimum_restore_v1/retained/orion/baicai/new_audit/tahoe_l3_x1_noise_audit_v1/results/tahoe/plate978'
        canonical=a.canonical or workspace/'baicai/canonical_perturbation_data_v1/tahoe/data'
        manifest=a.multiplate_manifest or physical/'tahoe_multiplate_plate_profile_manifest_978_v1.parquet'
        values=a.multiplate_values or physical/'tahoe_multiplate_plate_profiles_978_v1.h5'
        output=a.output or root/'result/04_prediction/computed/baseline'
        print(json.dumps(evaluate(root,canonical,manifest,values,output,a.threads,a.include_decoder),indent=2));return
    if a.command=='workspace-init':
        from .workspace import materialize
        print(json.dumps(materialize(root,a.workspace),indent=2));return
    if a.command=='workflow-run':
        from .workspace import run_script
        args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
        run_script(root,a.workspace,a.script,args,a.cpus);return
    if a.command=='sources':
        rows=select_sources(root,a.dataset);print(json.dumps(dict(files=rows,total_GiB=sum(x['bytes'] for x in rows)/1024**3),indent=2));return
    if a.command=='download':
        rows=select_sources(root,a.dataset);required=sum(x['bytes'] for x in rows)
        if required>a.max_gib*1024**3:p.error(f'Frozen selection is {required/1024**3:.3f} GiB, exceeding --max-gib')
        records=[]
        try:
            for row in rows:records.append(download_one(row,Path(a.destination)/row['dataset']))
        finally:
            out=Path(a.destination)/'DOWNLOAD_RUN.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps(records,indent=2));return
    if a.command=='doctor':
        result=dict(python=sys.version,packages={})
        for name in ['numpy','pandas','scipy','h5py','pyarrow','torch','scikit-learn','matplotlib','PyMuPDF']:
            try:result['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:result['packages'][name]='MISSING'
        print(json.dumps(result,indent=2));return
    if a.command=='models-check':
        reg=json.loads((root/'data/config/models.json').read_text());out=[]
        for m in reg['models']:
            path=(root/m['target']).resolve();assert path.is_relative_to(root)
            status='PASS' if path.is_file() and path.stat().st_size==m['bytes'] and sha256(path)==m['sha256'] else 'MISSING_OR_MISMATCH'
            out.append(dict(model=m['name'],fold=m['fold'],status=status,path=m['target']))
        print(json.dumps(out,indent=2));raise SystemExit(0 if all(x['status']=='PASS' for x in out) else 2)
    script={'figures':'code/common/scripts/reproduce_figures.py','kernel-check':'code/common/checks/test_original_contract.py'}[a.command]
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,check=True)

if __name__=='__main__':main()

"""Prepare shared dataset inputs and run the four analysis workflows.

Check mode reports dependencies without running analyses. Computation uses
registered programs in a generated workspace."""
from pathlib import Path
import argparse
import json
import os
import platform
import shutil
import sys
import time
from .sources import select_sources, download_one, sha256
from .workspace import materialize, run_script


def under(root, relative):
    root=Path(root).resolve()
    result=(root/relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes its data directory: '+str(relative))
    return result


def runtime_path(root, workspace=None):
    return Path(workspace).resolve() if workspace else Path(root)/'data/prepared/_runtime'


def inspect_stage(root, runtime, stage):
    missing=[]
    for pattern in stage.get('requires',[]):
        under(runtime,pattern)
        if not any(p.is_file() for p in Path(runtime).glob(pattern)):
            missing.append(pattern)
    if stage.get('linux',True) and platform.system()!='Linux':
        missing.append('Linux CPU environment (WSL or a Linux workstation/server)')
    if stage.get('r',False) and not shutil.which('Rscript'):
        missing.append('Rscript: install R 4.3 or newer')
    for rel in stage.get('repo_requires',[]):
        if not under(root,rel).is_file():missing.append('repository: '+rel)
    conflicts=[]
    bindings_path=Path(root)/'data/config/metadata_bindings.json'
    supplied={x['destination']:Path(root)/x['file'] for x in json.loads(bindings_path.read_text())}
    for pattern in stage.get('outputs',[]):
        under(runtime,pattern)
        for p in Path(runtime).glob(pattern):
            if not p.is_file():continue
            relative=p.relative_to(runtime).as_posix()
            original=supplied.get(relative)
            # Supplied condition annotations may be regenerated from source;
            # previously computed or modified outputs still block overwriting.
            if original and p.stat().st_size==original.stat().st_size and p.read_bytes()==original.read_bytes():continue
            conflicts.append(relative)
    return dict(stage=stage['id'],status='MISSING_INPUTS' if missing else 'OUTPUTS_EXIST' if conflicts else 'INPUTS_PRESENT',
                missing=missing,existing_outputs=conflicts,
                note=stage.get('note',''),execution_performed=False)


def execute_stage(root, runtime, stage, cpus, result_folder=None):
    if not 1<=cpus<=8:raise ValueError('Use 1 to 8 CPU threads')
    destination=under(root,f'result/{result_folder}/computed/{stage["id"]}') if result_folder else None
    if destination and destination.exists():raise RuntimeError('Computed export already exists: '+str(destination))
    if not (runtime/'REPLAY_WORKSPACE.json').is_file():materialize(root,runtime)
    status=inspect_stage(root,runtime,stage)
    if status['status']!='INPUTS_PRESENT':
        raise RuntimeError(json.dumps(status,indent=2))
    # Stage outputs may introduce a new result subtree in a clean workspace.
    for pattern in stage.get('outputs',[]):
        parent=under(runtime,pattern).parent
        if not any(char in str(parent) for char in '*?['):parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for call in stage['calls']:
        arguments=[arg.replace('{workspace}',runtime.as_posix()) for arg in call.get('args',[])]
        run_script(root,runtime,call['file'],arguments,cpus)
    files=[]
    for pattern in stage.get('outputs',[]):
        matches=[p for p in runtime.glob(pattern) if p.is_file()]
        if not matches:raise RuntimeError('Expected output not produced: '+pattern)
        files.extend(matches)
    record=dict(stage=stage['id'],status='COMPUTED',threads=cpus,
                elapsed_seconds=time.monotonic()-start,stage_executed=True,full_pipeline_recomputed=False,
                outputs=[dict(file=p.relative_to(runtime).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in files])
    if result_folder:
        destination.mkdir(parents=True)
        for p in files:
            if p.suffix.lower() not in {'.csv','.json','.tsv','.gz'}:continue
            out=destination/p.relative_to(runtime)
            out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,out)
        (destination/'COMPUTATION.json').write_text(json.dumps(record,indent=2))
    return record


def experiment_main(config_path):
    config_path=Path(config_path).resolve();root=config_path.parents[2]
    config=json.loads(config_path.read_text())
    parser=argparse.ArgumentParser(description=config['description'])
    parser.add_argument('--stage',choices=[s['id'] for s in config['stages']])
    parser.add_argument('--workspace',help='Defaults to data/prepared/_runtime')
    parser.add_argument('--cpus',type=int,default=4)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='Execute one analysis stage')
    mode.add_argument('--check',action='store_true',help='Check dependencies without reading responses')
    args=parser.parse_args()
    runtime=runtime_path(root,args.workspace)
    stages=[s for s in config['stages'] if args.stage in (None,s['id'])]
    if args.execute and not args.stage:parser.error('--execute requires one --stage')
    try:
        result=execute_stage(root,runtime,stages[0],args.cpus,config_path.parent.name) if args.execute else [inspect_stage(root,runtime,s) for s in stages]
        print(json.dumps(result,indent=2))
        if not args.execute and any(s['status']=='MISSING_INPUTS' for s in result):return 2
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr);return 2
    return 0


def bind_raw(root,runtime,dataset):
    """Verify raw-file checksums and bind inputs with filesystem hard links."""
    rows=select_sources(root,[dataset]);bindings=[]
    for row in rows:
        source=under(root,'data/raw/'+dataset+'/'+row['filename'])
        if not source.is_file() or source.stat().st_size!=row['bytes'] or sha256(source)!=row['sha256']:
            raise ValueError('Raw file missing or hash mismatch: '+row['filename'])
    for row in rows:
        source=under(root,'data/raw/'+dataset+'/'+row['filename'])
        folder='Noise/LINCS2020/raw/level3' if dataset=='L1000_L3' else 'FULL_DATASET_FIGURES_V1/raw'
        target=under(runtime,folder+'/'+row['filename']);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if not os.path.samefile(source,target):raise ValueError('Raw binding already occupied: '+str(target))
        else:
            try:os.link(source,target)
            except OSError as exc:raise RuntimeError('Raw and prepared directories must share a filesystem for zero-copy binding') from exc
        bindings.append(dict(file=row['filename'],sha256=row['sha256'],storage='hardlink'))
    return bindings


def data_main(root):
    root=Path(root).resolve()
    configs=[json.loads(p.read_text()) for p in sorted((root/'data/datasets').glob('*/config.json'))]
    names={c['dataset']:c for c in configs}
    parser=argparse.ArgumentParser(description='Download, validate and prepare source datasets and prediction features')
    parser.add_argument('action',choices=['plan','download','init','bind','check','build'])
    parser.add_argument('--dataset',choices=sorted(names))
    parser.add_argument('--stage')
    parser.add_argument('--max-gib',type=float,help='Required for a download; exact byte count is checked first')
    parser.add_argument('--workspace')
    parser.add_argument('--cpus',type=int,default=4)
    args=parser.parse_args();runtime=runtime_path(root,args.workspace)
    if args.action=='init':
        print(json.dumps(materialize(root,runtime),indent=2));return 0
    if args.action=='plan':
        selected=[names[args.dataset]] if args.dataset else configs
        for c in selected:
            rows=select_sources(root,[c['dataset']]);c['download_GiB']=sum(x['bytes'] for x in rows)/2**30
            c['unavailable_sources']=[x['id'] for x in rows if x.get('availability')=='ACCESS_FAILED']
        print(json.dumps(selected,indent=2));return 0
    if not args.dataset:parser.error('Specify --dataset')
    if args.action=='download':
        rows=select_sources(root,[args.dataset]);size=sum(x['bytes'] for x in rows)
        if args.max_gib is None or args.max_gib<0 or size>args.max_gib*2**30:
            parser.error(f'Download is {size/2**30:.3f} GiB; set an adequate --max-gib')
        destination=root/'data/raw'/args.dataset;destination.mkdir(parents=True,exist_ok=True)
        records=[]
        try:
            for row in rows:records.append(download_one(row,destination))
        finally:
            (destination/'DOWNLOAD.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records,indent=2));return 0
    if args.action=='bind':
        if not (runtime/'REPLAY_WORKSPACE.json').exists():materialize(root,runtime)
        print(json.dumps(bind_raw(root,runtime,args.dataset),indent=2));return 0
    config=names[args.dataset]
    if args.stage and args.stage not in {s['id'] for s in config['stages']}:parser.error('Unknown stage for dataset')
    stages=[s for s in config['stages'] if args.stage in (None,s['id'])]
    if args.action=='build':
        if not args.stage:parser.error('Build requires one --stage, in the order shown by plan')
        print(json.dumps(execute_stage(root,runtime,stages[0],args.cpus),indent=2));return 0
    checks=[inspect_stage(root,runtime,s) for s in stages]
    print(json.dumps(checks,indent=2))
    return 2 if any(c['status']=='MISSING_INPUTS' for c in checks) else 0

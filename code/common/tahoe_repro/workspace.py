"""Configure registered processing programs in a CPU analysis workspace.

Workspace tokens resolve input paths, execution guards and disk checks.
Input data are prepared separately; program hashes are recorded for validation."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

ORIGIN = '@WORKSPACE@'


class RuntimeAdapter(ast.NodeTransformer):
    def visit_Call(self, node):
        code=ast.unparse(node)
        if code in ["os.getenv('TAHOE_REPRO_RUN_ID')", "os.environ.get('TAHOE_REPRO_RUN_ID')"]:
            return ast.copy_location(ast.Constant(True),node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == 'quota':
            return ast.copy_location(ast.parse('''
def quota():
    import shutil
    free = shutil.disk_usage(R).free / 2**30
    minimum = float(os.environ.get('TAHOE_REPRO_MIN_FREE_GIB', '15'))
    if free < minimum:
        raise RuntimeError(f'Insufficient workspace disk: {free:.2f} GiB free')
    return free
''').body[0], node)
        return self.generic_visit(node)


def materialize(repo, destination):
    repo, dest = Path(repo).resolve(), Path(destination).resolve()
    internal = repo/'data/prepared/_runtime'
    if dest != internal and (dest == repo or repo.is_relative_to(dest) or dest.is_relative_to(repo)):
        raise ValueError('Choose data/prepared/_runtime or a separate directory outside the repository')
    marker = dest/'REPLAY_WORKSPACE.json'
    if dest.exists() and any(dest.iterdir()) and not marker.exists():
        raise ValueError('Refuse to populate a nonempty unrelated directory')
    dest.mkdir(parents=True, exist_ok=True)
    bindings=json.loads((repo/'data/config/metadata_bindings.json').read_text())
    for binding in bindings:
        src=repo/binding['file']
        out=dest/binding['destination']
        if not src.resolve().is_relative_to(repo) or not out.resolve().is_relative_to(dest):
            raise ValueError('Unsafe metadata binding')
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.read_bytes() != src.read_bytes():
            raise ValueError(f'Changed metadata in replay workspace: {out}')
        if not out.exists():
            shutil.copyfile(src, out)
    rows = []
    for item in json.loads((repo/'data/config/production_sources.json').read_text()):
        src = repo/item['file']
        original = src.read_text(encoding='utf-8-sig')
        if hashlib.sha256(src.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Changed production source: {src}')
        relative = item['destination']
        out = dest/relative
        if not out.resolve().is_relative_to(dest):
            raise ValueError(relative)
        # Resolve both canonical-data path aliases to the same workspace location.
        text = original.replace('4MAN/minimum_restore_v1/retained/orion/baicai/canonical_perturbation_data_v1', 'baicai/canonical_perturbation_data_v1')
        text = text.replace(ORIGIN, dest.as_posix())
        text = text.replace("'/baicai/", "'"+dest.as_posix()+'/baicai/')
        text = text.replace('"/baicai/', '"'+dest.as_posix()+'/baicai/')
        text = text.replace('"/orion/', '"'+dest.as_posix()+'/')
        text = text.replace("'/orion/", "'"+dest.as_posix()+'/')
        text = text.replace("'/usr/bin/Rscript'", "'Rscript'")
        if src.suffix == '.py':
            tree = RuntimeAdapter().visit(ast.parse(text))
            ast.fix_missing_locations(tree)
            text = ast.unparse(tree)+'\n'
            compile(text, str(out), 'exec')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        rows.append(dict(source=item['file'],destination=relative,
                         original_sha256=item['sha256'],adapted_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    # Directory creation does not materialize raw/processed data.
    roots = [dest/'FULL_DATASET_FIGURES_V1', dest/'4MAN/TAHOE_DESCRIPTIVE_FULL_V2',
             dest/'baicai/canonical_perturbation_data_v1/tahoe',
             dest/'baicai/canonical_perturbation_data_v1/l1000']
    for root in roots:
        for name in ['data','metadata','audit','results','receipts','logs','tmp','raw']:
            (root/name).mkdir(parents=True, exist_ok=True)
    # Reuse the shipped frozen weights without duplicating their disk storage.
    model_manifest=repo/'data/config/models.json'
    model_rows=json.loads(model_manifest.read_text())['models'] if model_manifest.exists() else []
    g5=json.loads((repo/'data/metadata/prediction/G5_LINEAGE_AUDIT.json').read_text()) if any(m['name']=='G5' for m in model_rows) else {}
    for model in model_rows:
        src=repo/model['target'];fold=model['fold']
        if model['name']=='G5':
            relative=next(x['path'] for x in g5['folds'] if x['fold']==fold).removeprefix('@WORKSPACE@/')
        else:
            relative=f'4MAN/minimum_restore_v1/retained/orion/tahoe_x1only_3h2048_baseline_20261007/cpu_v2/runs/A_f{fold}_3H2048/selected.pt'
        out=dest/relative
        if not out.resolve().is_relative_to(dest):raise ValueError('Unsafe model binding')
        with src.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=model['sha256']:
                raise ValueError('Frozen model checksum mismatch: '+model['target'])
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if not os.path.samefile(src,out):
                with out.open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=model['sha256']:
                        raise ValueError('Occupied model binding: '+str(out))
        else:
            try:os.link(src,out)
            except OSError:
                shutil.copyfile(src,out)
    marker.write_text(json.dumps(dict(status='INITIALIZED_NO_RESPONSE_DATA',release=str(repo),
                                      production_files=rows,raw_files_copied=0),indent=2))
    return dict(workspace=str(dest),production_files=len(rows),raw_files_copied=0)


def run_script(repo, workspace, source_file, arguments, cpus=4):
    repo, workspace = Path(repo).resolve(), Path(workspace).resolve()
    state = json.loads((workspace/'REPLAY_WORKSPACE.json').read_text())
    match = [x for x in state['production_files'] if x['source']==source_file]
    if len(match)!=1:
        raise ValueError('Choose a registered processing script from data/config/production_sources.json')
    if not 1 <= cpus <= 8:
        raise ValueError('CPU count must be in [1, 8]')
    script=workspace/match[0]['destination']
    if hashlib.sha256(script.read_bytes()).hexdigest()!=match[0]['adapted_sha256']:
        raise ValueError('Adapted script changed after workspace initialization')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
               TAHOE_REPRO_RUN_ID='local-cpu',TAHOE_REPRO_CPUS=str(cpus))
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]=str(cpus)
    interpreter='Rscript' if script.suffix=='.R' else sys.executable
    subprocess.run([interpreter,str(script),*arguments],cwd=script.parent,env=env,check=True)

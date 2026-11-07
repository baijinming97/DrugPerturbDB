"""Regenerate the six manuscript figures from checksum-protected source tables."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / 'result' / 'figure_inputs'
CODE = ROOT / 'code' / 'common' / 'plotting'
OUTPUT = ROOT / 'result'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def temporary_workspace():
    # An ordinary directory inherits Windows workspace ACLs correctly.
    path = ROOT / 'result' / ('.figure_work_' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if not resolved.is_relative_to((ROOT / 'result').resolve()) or not path.name.startswith('.figure_work_'):
            raise RuntimeError('Unexpected temporary workspace path')
        shutil.rmtree(resolved)


def main(numbers=None):
    manifest = json.loads((ENGINE / 'SOURCE_DATA_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest:
        if digest(ENGINE / row['file']) != row['sha256']:
            raise ValueError(f"Figure source checksum mismatch: {row['file']}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with temporary_workspace() as temporary:
        work = Path(temporary)
        shutil.copytree(ENGINE, work / 'base')
        shutil.copytree(CODE, work / 'base/code')
        env = os.environ.copy()
        env.update(TAHOE_FIGURE_WORKDIR=str(work), PYTHONUTF8='1',
                   PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(work / 'mpl'), CUDA_VISIBLE_DEVICES='')
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            env[key] = '2'
        scripts = ['render_full_scope_fig12.py', 'render_full_scope_fig3.py',
                   'render_main_numbering.py', 'render_concise_s1.py', 'compose_panels.py',
                   'layout_main.py', 'layout_schematic.py', 'layout_decoder.py',
                   'layout_s1.py', 'render_s2.py']
        for name in scripts:
            result = subprocess.run([sys.executable, str(work / 'base/code' / name)],
                                    cwd=work / 'base', env=env, capture_output=True, text=True,
                                    encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'Figure stage {name} failed:\n{result.stdout}\n{result.stderr}')
        for number in (numbers or ['1','2','3','4','S1','S2']):
            stage = 'final_main' if number.isdecimal() else 'supplement'
            for ext in ['pdf','svg','png']:
                folder = 'output/pdf' if ext == 'pdf' and number != 'S1' else 'figures'
                src = work / stage / folder / f'Figure_{number}.{ext}'
                experiment = {'1':'01_repeatability','S1':'01_repeatability','2':'02_cross_dataset','3':'03_components','4':'04_prediction','S2':'04_prediction'}[number]
                destination=OUTPUT/experiment/'figures'/src.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src, destination)
    for row in manifest:
        if digest(ENGINE / row['file']) != row['sha256']:
            raise ValueError(f"Figure source modified during rendering: {row['file']}")
    record = {'status':'PASS', 'figures':len(numbers or ['1','2','3','4','S1','S2']), 'formats':['pdf','svg','png'],
              'source_tables':len(manifest), 'raw_data_reanalysis':False,
              'outputs':{p.relative_to(OUTPUT).as_posix():digest(p) for p in sorted(OUTPUT.glob('*/figures/*')) if p.is_file()}}
    (ROOT/'result/FIGURE_REPRODUCTION.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Exported requested figures from verified numerical tables: result/<experiment>/figures/')


if __name__ == '__main__':
    main()

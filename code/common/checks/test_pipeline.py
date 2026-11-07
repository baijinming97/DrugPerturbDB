"""Checks for the public data/experiment interfaces, without large downloads."""
from pathlib import Path
import hashlib
import json
import pytest
from tahoe_repro.pipeline import under,inspect_stage,execute_stage,bind_raw

ROOT=Path(__file__).resolve().parents[3]


def test_missing_stage_does_not_create_outputs(tmp_path):
    stage=dict(id='missing',linux=False,requires=['data/measurement.h5'],outputs=['results/answer.csv'])
    result=inspect_stage(ROOT,tmp_path,stage)
    assert result['status']=='MISSING_INPUTS'
    assert result['missing']==['data/measurement.h5']
    assert not list(tmp_path.iterdir())


def test_input_paths_cannot_escape_data_root(tmp_path):
    with pytest.raises(ValueError):under(tmp_path,'../unrelated')


def test_supplied_annotations_are_not_computed_outputs(tmp_path):
    root=tmp_path/'repo';runtime=tmp_path/'runtime'
    (root/'data/config').mkdir(parents=True);runtime.mkdir()
    source=root/'data/annotations.csv';source.write_text('id\na\n')
    target=runtime/'annotations.csv';target.write_bytes(source.read_bytes())
    (root/'data/config/metadata_bindings.json').write_text(json.dumps([
        dict(file='data/annotations.csv',destination='annotations.csv')]))
    stage=dict(id='profiles',linux=False,requires=[],outputs=['annotations.csv','profiles.h5'])
    assert inspect_stage(root,runtime,stage)['status']=='INPUTS_PRESENT'
    (runtime/'profiles.h5').write_bytes(b'computed')
    assert inspect_stage(root,runtime,stage)['existing_outputs']==['profiles.h5']
    target.write_text('id\nchanged\n')
    assert 'annotations.csv' in inspect_stage(root,runtime,stage)['existing_outputs']


def test_model_copy_fallback_is_repeatable_and_checks_integrity(tmp_path,monkeypatch):
    from tahoe_repro.workspace import materialize
    import tahoe_repro.workspace as workspace
    root=tmp_path/'repo';(root/'data/config').mkdir(parents=True)
    source=root/'data/model.pt';source.write_bytes(b'frozen model')
    model=dict(name='control_only_3H2048',fold=0,target='data/model.pt',
               sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for name,value in [('models.json',dict(models=[model])),
                       ('metadata_bindings.json',[]),('production_sources.json',[])]:
        (root/'data/config'/name).write_text(json.dumps(value))
    def unavailable(*args):raise OSError('separate filesystems')
    monkeypatch.setattr(workspace.os,'link',unavailable)
    runtime=tmp_path/'runtime'
    materialize(root,runtime);materialize(root,runtime)
    target=next(runtime.rglob('selected.pt'))
    assert target.read_bytes()==source.read_bytes()
    target.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='Occupied model binding'):
        materialize(root,runtime)


def test_all_entry_points_resolve_registered_sources():
    sources={s['file'] for s in json.loads((ROOT/'data/config/production_sources.json').read_text())}
    for path in [*ROOT.glob('code/0*/config.json'),*ROOT.glob('data/datasets/*/config.json')]:
        config=json.loads(path.read_text())
        assert len({s['id'] for s in config['stages']})==len(config['stages'])
        for stage in config['stages']:
            assert stage['requires'] and stage['outputs']
            for call in stage['calls']:
                assert call['file'] in sources
                assert (ROOT/call['file']).is_file()


def test_raw_binding_does_not_duplicate_or_change_file(tmp_path):
    root=tmp_path/'repo';source=root/'data/raw/Tiny/measurement.bin'
    source.parent.mkdir(parents=True);source.write_bytes(b'measured example')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    row=dict(dataset='Tiny',filename=source.name,bytes=source.stat().st_size,sha256=digest)
    (root/'data/sources.json').write_text(json.dumps([row]))
    runtime=root/'data/prepared/_runtime'
    records=bind_raw(root,runtime,'Tiny')
    target=runtime/'FULL_DATASET_FIGURES_V1/raw'/source.name
    assert source.samefile(target) and records[0]['sha256']==digest
    assert bind_raw(root,runtime,'Tiny')==records


def test_one_stage_computes_and_preserves_reference(tmp_path):
    root=tmp_path/'repo';(root/'data/config').mkdir(parents=True)
    (root/'code').mkdir()
    script=root/'code/sum.py'
    script.write_text("from pathlib import Path\np=Path('@WORKSPACE@')\nx=[float(v) for v in (p/'input.txt').read_text().split()]\n(p/'new/results/sum.csv').write_text('sum\\n'+str(sum(x))+'\\n')\n")
    source=dict(file='code/sum.py',destination='sum.py',sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    (root/'data/config/production_sources.json').write_text(json.dumps([source]))
    (root/'data/config/metadata_bindings.json').write_text('[]')
    runtime=tmp_path/'runtime'
    from tahoe_repro.workspace import materialize
    materialize(root,runtime)
    (runtime/'input.txt').write_text('1 2 3')
    reference=root/'result/01_repeatability/reference.csv'
    reference.parent.mkdir(parents=True);reference.write_text('do not replace')
    stage=dict(id='tiny',linux=False,requires=['input.txt'],outputs=['new/results/sum.csv'],calls=[dict(file='code/sum.py')])
    receipt=execute_stage(root,runtime,stage,1,'01_repeatability')
    assert receipt['status']=='COMPUTED'
    assert (root/'result/01_repeatability/computed/tiny/new/results/sum.csv').read_text()=='sum\n6.0\n'
    assert reference.read_text()=='do not replace'
    with pytest.raises(RuntimeError):execute_stage(root,runtime,stage,1,'01_repeatability')

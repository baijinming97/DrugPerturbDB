"""Behavioral tests for portable, immutable CPU analysis interfaces."""
from pathlib import Path
import ast
import hashlib
import io
import json
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/common'))
from tahoe_repro.statistics import all_cross_plate_pcc, pearson_rows, decompose
from tahoe_repro.projection import Projector
from tahoe_repro.workspace import RuntimeAdapter, materialize
from tahoe_repro.sources import download_one, select_sources
from tahoe_repro.verification import verify_release, verify_measured_examples


def test_correlation_does_not_mutate_input():
    x=np.arange(32,dtype=float).reshape(4,8)
    original=x.copy()
    score,pairs=all_cross_plate_pcc(x,[0,0,1,1])
    assert np.array_equal(x,original)
    assert pairs==4 and score==pytest.approx(1.)


def test_undefined_correlation_is_missing():
    assert np.isnan(pearson_rows(np.zeros((2,8)),np.ones((2,8)))).all()
    score,pairs=all_cross_plate_pcc(np.ones((2,8)),[0,1])
    assert np.isnan(score) and pairs==0


def test_single_cell_has_no_identifiable_interaction():
    meta=pd.DataFrame({'cell':['c']*4,'drug':['a','b','c','d']})
    x=np.random.default_rng(7).normal(size=(4,7))
    e,components,info=decompose(x,meta)
    assert not info['interaction_identifiable']
    assert e[-1]<1e-20
    assert e.sum()==pytest.approx(np.mean(x*x,axis=0).sum())


def test_masked_projection_preserves_inactive_conditions():
    rng=np.random.default_rng(11)
    x=np.zeros((3,7,2));x[:,5:,0]=1
    mask=np.ones((3,7),bool);mask[0,1]=False
    projector=Projector(x,np.ones(7)/7,minimum=3,mask=mask)
    y=projector(rng.normal(size=(3,7,9)))
    assert np.all(y[:,5:]==0) and np.all(y[0,1]==0)
    assert max(projector.residuals(y).values())<1e-11


def test_runtime_adapter_preserves_non_scheduler_conditions():
    source="assert os.getenv('TAHOE_REPRO_RUN_ID') and not output_exists and not gpu"
    node=ast.fix_missing_locations(RuntimeAdapter().visit(ast.parse(source)))
    compiled=compile(node,'<test>','exec')
    exec(compiled,{'output_exists':False,'gpu':False})
    for values in [{'output_exists':True,'gpu':False},{'output_exists':False,'gpu':True}]:
        with pytest.raises(AssertionError):exec(compiled,values)


def test_workspace_cannot_target_repository():
    with pytest.raises(ValueError):materialize(ROOT,ROOT)
    with pytest.raises(ValueError):materialize(ROOT,ROOT/'work')


def test_source_names_are_validated():
    with pytest.raises(ValueError):select_sources(ROOT,['not-a-dataset'])


def test_download_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError):download_one({'filename':'../escape.bin'},tmp_path)


def test_download_keeps_mismatched_data_for_diagnosis(tmp_path,monkeypatch):
    import urllib.request
    class Response(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response(b'wrong'))
    row={'filename':'a.bin','id':'test','url':'https://example.invalid/a','bytes':5,
         'sha256':hashlib.sha256(b'right').hexdigest()}
    with pytest.raises(RuntimeError,match='checksum'):download_one(row,tmp_path,reserve_bytes=0)
    assert not (tmp_path/'a.bin').exists()
    assert (tmp_path/'a.bin.partial').read_bytes()==b'wrong'


def test_current_s2_has_ten_complete_paired_controls():
    data=pd.read_csv(ROOT/'result/figure_inputs/inputs/EXPLORATORY_CONFIGURATIONS.csv')
    assert len(data)==10
    assert np.isfinite(data[['value','lo','hi','delta','dlo','dhi']].to_numpy()).all()
    assert ((data.lo<=data.value)&(data.value<=data.hi)).all()
    assert ((data.dlo<=data.delta)&(data.delta<=data.dhi)).all()


def test_release_integrity():
    assert verify_release(ROOT)['status']=='PASS'


def test_retained_measurement_examples():
    checked=verify_measured_examples(ROOT)
    assert checked['cross_source_examples']==74
    assert checked['largest_absolute_error']<1e-10


def test_retained_projection_helpers_match_constraint_solution(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'code/04_prediction/decoder'))
    from w_projection_iterative import WProjection as Iterative
    from w_projection_spectral import WProjection as Spectral
    rng=np.random.default_rng(23)
    labels=np.array([[0,0,0,0,0,1,1,1],
                     [0,0,1,1,1,1,1,1],
                     [0,0,0,1,1,1,1,1],
                     [0,0,0,0,1,1,1,1]])
    x=np.stack([labels,labels*2],axis=-1).astype(float)
    weights=np.arange(1,9,dtype=float);weights/=weights.sum()
    value=rng.normal(size=(4,8,11))
    expected=Projector(x,weights,minimum=3)(value)
    for implementation in [Iterative,Spectral]:
        actual=implementation(x,weights,minimum=3)(value)
        assert np.allclose(actual,expected,rtol=0,atol=3e-9)

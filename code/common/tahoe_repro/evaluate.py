"""Evaluate the frozen baseline from canonical and physical-plate profiles.

Predictions are recomputed for the registered VALID conditions."""
from pathlib import Path
import collections,json
import numpy as np
import pandas as pd
from .panels import build_panel
from .inference import predict_control_only, predict_g5
from .metrics import panel_metrics, decompose, row_pcc
from .projection import Projector


def summarize_drugs(table, columns, weight='weight'):
    """Chemical-group bootstrap with the manuscript's fixed random seed."""
    # A component-specific bootstrap samples only its eligible support.
    table=table.loc[table[weight]>0]
    if table.empty:raise ValueError('No positive-weight observations')
    groups=sorted(set(table.group));num=[];den=[]
    for group in groups:
        t=table[table.group.eq(group)]
        den.append(t[weight].sum())
        num.append([(t[weight]*t[c]).sum() for c in columns])
    num=np.asarray(num);den=np.asarray(den)
    draws=np.random.default_rng(20261021).multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
    point=num.sum(0)/den.sum();boot=draws@num/(draws@den)[:,None]
    intervals=np.quantile(boot,[.025,.975],axis=0)
    return {key:float(value) for i,c in enumerate(columns)
            for key,value in [(c,point[i]),(c+'_lo',intervals[0,i]),(c+'_hi',intervals[1,i])]}

def evaluate(repo, canonical, multiplate_manifest, multiplate_values, output, threads=2, include_decoder=False):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'BASELINE_REPRODUCTION.json').exists():raise FileExistsError(output)
    models=json.loads((repo/'data/config/models.json').read_text())['models']
    roles=pd.read_csv(repo/'data/metadata/MODEL_VALID_ROLES.csv',dtype={'drug':str,'group':str})
    counts=collections.Counter(roles.drug);groups=dict(zip(roles.drug,roles.group))
    sizes=collections.Counter(groups.values());ng=len(sizes)
    axes=[repo/'data/metadata/genes/GENE_AXIS.tsv']
    if len(axes)!=1:raise ValueError('Ambiguous frozen gene axis')
    rows=[];total=collections.Counter();decoder_rows=[];pure=[];components=[];allocation=[]
    for f in range(5):
        z=build_panel(canonical,repo/f'data/metadata/fold{f}_VALID_identities.csv',
                      axes[0],'VALID',multiplate_manifest,multiplate_values)
        drugs=np.array(sorted(set(z['drug_id'])));cells=np.array(sorted(set(z['cell_id'])))
        lookup={(c,d):j for j,(c,d) in enumerate(zip(z['cell_id'],z['drug_id']))}
        ix=np.array([[lookup[c,d] for d in drugs] for c in cells])
        weights=np.array([1/(ng*sizes[groups[d]]*counts[d]) for d in drugs]);mass=weights.sum()
        ck=next(m for m in models if m['fold']==f and m['name'].startswith('Control'))
        pred=predict_control_only(repo/ck['target'],z['x'][ix],expected_sha256=ck['sha256'],threads=threads)
        result=panel_metrics(pred,z['y'][ix],weights)
        rows.append(dict(fold=f,drugs=len(drugs),mass=float(mass),**result))
        for k,v in result.items():total[k]+=mass*v
        w=weights/mass;truth=z['y'][ix].astype(float);x=z['x'][ix]
        db,dy=decompose(pred,w),decompose(truth,w)
        for name in ['mu','alpha','beta','gamma']:
            pc=row_pcc(db[name],dy[name]);err=np.mean((db[name]-dy[name])**2,axis=-1)
            if name in ['mu','alpha']:pv=float(np.mean(pc));mv=float(np.mean(err))
            else:pv=float(np.mean(pc,axis=0)@w) if name=='gamma' else float(pc@w);mv=float(np.mean(err,axis=0)@w) if name=='gamma' else float(err@w)
            components.append(dict(fold=f,mass=float(mass),component=name,PCC=pv,MSE=mv))
        # Error allocation uses the full evaluation support, including small contexts.
        flat=np.ascontiguousarray(x.transpose(1,0,2).reshape(len(drugs),-1))
        _,contexts=np.unique(flat.view(np.dtype((np.void,flat.shape[1]*flat.dtype.itemsize))).ravel(),return_inverse=True)
        beta_error=db['beta']-dy['beta'];beta_uncovered=beta_error.copy()
        for context in np.unique(contexts):
            mask=contexts==context
            beta_uncovered[mask]-=np.average(beta_error[mask],axis=0,weights=w[mask])
        projector=Projector(x,w)
        gamma_error=db['gamma']-dy['gamma'];gamma_uncovered=projector(gamma_error)
        error_parts={'mu':db['mu']-dy['mu'],'alpha':db['alpha']-dy['alpha'],
                     'beta shared':beta_error-beta_uncovered,'beta uncovered':beta_uncovered,
                     'gamma shared':gamma_error-gamma_uncovered,'gamma uncovered':gamma_uncovered}
        fold_error=0.
        for name,array in error_parts.items():
            per=np.mean(array**2,axis=-1)
            value=float(np.mean(per)) if name in ['mu','alpha'] else float((per.mean(0) if name.startswith('gamma') else per)@w)
            allocation.append(dict(fold=f,mass=float(mass),component=name,MSE=value))
            fold_error+=value
        assert abs(fold_error-result['full_MSE'])<1e-8
        if include_decoder:
            g5=next(m for m in models if m['fold']==f and m['name']=='G5')
            lineage=json.loads((repo/'data/metadata/prediction/G5_LINEAGE_AUDIT.json').read_text())['folds']
            assert list(cells)==next(t['cells'] for t in lineage if t['fold']==f)
            flat=np.ascontiguousarray(x.transpose(1,0,2).reshape(len(drugs),-1))
            _,contexts=np.unique(flat.view(np.dtype((np.void,flat.shape[1]*flat.dtype.itemsize))).ravel(),return_inverse=True)
            beta=dy['beta'].copy()
            for context in np.unique(contexts):
                mask=contexts==context;beta[mask]-=np.average(beta[mask],axis=0,weights=w[mask])
            raw,shrinkage=predict_g5(repo/g5['target'],beta,expected_sha256=g5['sha256'])
            projector=Projector(x,w);delta=shrinkage*Projector(x,w,minimum=3)(raw);new=pred+delta
            dg=decompose(delta,w);dn=decompose(new,w)
            residual=max(float(np.max(abs(dg[k]))) for k in ['mu','alpha','beta'])
            closure=abs(panel_metrics(new,truth,w)['full_MSE']-result['full_MSE']-(panel_metrics(new,truth,w)['gamma_MSE']-result['gamma_MSE']))
            assert residual<1e-8 and closure<1e-8
            pure.append(dict(fold=f,lambda_train=shrinkage,pure_marginal_maxabs=residual,MSE_closure=closure))
            eligible=projector.eligible
            projected=projector(new)
            gw=row_pcc(projected,projector(truth))
            norm=np.linalg.norm(projected-projected.mean(-1,keepdims=True),axis=-1)
            gw=np.where(norm<=1e-6,0.,np.nan_to_num(gw,nan=0.))
            for j,d in enumerate(drugs):
                record=dict(fold=f,drug=d,group=groups[d],weight=weights[j],gamma_weight=weights[j]*eligible[:,j].mean())
                for prefix,p,dec in [('baseline_',pred,db),('',new,dn)]:
                    record.update({prefix+'full_PCC':float(row_pcc(p[:,j],truth[:,j]).mean()),
                                   prefix+'full_MSE':float(np.mean((p[:,j]-truth[:,j])**2)),
                                   prefix+'gamma_PCC':float(row_pcc(dec['gamma'][:,j],dy['gamma'][:,j]).mean()),
                                   prefix+'gamma_MSE':float(np.mean((dec['gamma'][:,j]-dy['gamma'][:,j])**2))})
                record['gamma_increment_PCC']=float(gw[:,j][eligible[:,j]].mean()) if eligible[:,j].any() else 0.
                decoder_rows.append(record)
    expected=dict(full_PCC=.582112,full_MSE=.007005123,gamma_PCC=.488200,gamma_MSE=.003482728)
    differences={k:float(abs(total[k]-v)) for k,v in expected.items()}
    receipt=dict(status='PASS' if max(differences.values())<1e-6 else 'FAIL',
                 source='Rebuilt canonical + equal physical-plate profiles; no prediction caches',
                 aggregate=dict(total),reference=expected,absolute_error=differences,
                 unique_VALID_drugs=len(groups),folds=rows,TEST_response_rows=0,training=0)
    (output/'BASELINE_REPRODUCTION.json').write_text(json.dumps(receipt,indent=2))
    pd.DataFrame(rows).to_csv(output/'BASELINE_BY_FOLD.csv',index=False)
    pd.DataFrame(components).to_csv(output/'BASELINE_COMPONENTS_BY_FOLD.csv',index=False)
    a=pd.DataFrame(allocation)
    a.to_csv(output/'BASELINE_ERROR_ALLOCATION_BY_FOLD.csv',index=False)
    aggregate=[dict(component=name,MSE=float((part.mass*part.MSE).sum())) for name,part in a.groupby('component')]
    at=pd.DataFrame(aggregate);at['error_fraction']=at.MSE/total['full_MSE']
    assert abs(at.error_fraction.sum()-1)<1e-8
    at.to_csv(output/'BASELINE_ERROR_ALLOCATION.csv',index=False)
    if receipt['status']!='PASS':raise AssertionError(receipt)
    if include_decoder:
        t=pd.DataFrame(decoder_rows);t.to_csv(output/'G5_BY_DRUG.csv',index=False)
        metrics=['full_PCC','full_MSE','gamma_PCC','gamma_MSE']
        for key in metrics:t['delta_'+key]=t[key]-t['baseline_'+key]
        scores=summarize_drugs(t,metrics+['delta_'+c for c in metrics])
        scores.update(summarize_drugs(t,['gamma_increment_PCC'],'gamma_weight'))
        reference=pd.read_csv(repo/'result/figure_inputs/evidence_inputs/PREPRINT_FULL_DATA_V1/evidence_model/PRIMARY_ORACLE_DIAGNOSTIC.csv').query("workflow=='WS1'").iloc[0]
        errors={key+suffix:abs(scores[key+suffix]-reference[key+suffix]) for key in metrics for suffix in ['', '_lo', '_hi']}
        for suffix in ['', '_lo', '_hi']:errors['gamma_increment_PCC'+suffix]=abs(scores['gamma_increment_PCC'+suffix]-reference['gamma_w_PCC'+suffix])
        status='PASS' if max(errors.values())<1e-6 else 'FAIL'
        decoder=dict(status=status,metrics=scores,absolute_error=errors,reference_tolerance=1e-6,
                     input='Measured drug-average effect; diagnostic, not zero-shot deployment',
                     predictions_recomputed=True,TEST_response_rows=0,training=0)
        pd.DataFrame([scores]).to_csv(output/'G5_RESULTS.csv',index=False)
        (output/'G5_REPRODUCTION.json').write_text(json.dumps(decoder,indent=2))
        (output/'PURE_GAMMA_CHECK.json').write_text(json.dumps(pure,indent=2))
        if status!='PASS':raise AssertionError(decoder)
        receipt['G5']=decoder
    return receipt


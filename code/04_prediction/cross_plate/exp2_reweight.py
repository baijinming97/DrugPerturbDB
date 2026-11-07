"""Recalculate common-support estimates from saved sufficient statistics.

Primary and alternative-weight estimates use identical bootstrap draws and
condition support. Outputs are written to separate directories."""
from pathlib import Path
import os, sys, json, collections, time, resource
sys.dont_write_bytecode=True
import exp2_worker as core
import numpy as np
import pandas as pd

SOURCE=core.R
START=time.monotonic()

def fold_rows(configs,panels):
    rows=[]
    for f,p in enumerate(panels):
        w=p['q']/p['q'].sum()
        for c in configs:
            s={k.split('__',1)[1]:v for k,v in p.items() if k.startswith(c['key']+'__')}
            z={k:float(v[0]) for k,v in core.stats(s,w[None]).items()}
            rows.append(dict(fold=f,model=c['model'],direction=c['direction'],setting='within' if c['source']==c['target'] else 'cross',
                         drugs=len(w),groups=len(set(p['group'])),weight=float(p['q'].sum()),lambda_used=1.,**z))
    return pd.DataFrame(rows)

def summarize(configs,panels,mode,dest,counts,groups):
    core.R=dest;dest.mkdir(parents=True,exist_ok=True)
    ix={g:i for i,g in enumerate(groups)};full=[];bsall={}
    per=fold_rows(configs,panels);per.to_csv(dest/'PER_FOLD.csv',index=False)
    for f,t in per.groupby('fold'):t.to_csv(dest/f'POINT_f{f}.csv',index=False)
    for c in configs:
        core.budget();sums={};dens={};foldg=[]
        for f,p in enumerate(panels):
            mult=np.r_[np.ones((1,len(p['group']))),counts[:,[ix[g] for g in p['group']]]]
            qw=mult*p['q'][None];mass=qw.sum(1);W=np.divide(qw,mass[:,None],out=np.zeros_like(qw),where=mass[:,None]>0)
            s={k.split('__',1)[1]:v for k,v in p.items() if k.startswith(c['key']+'__')}
            st=core.stats(s,W)
            for key,val in st.items():
                if key not in sums:sums[key]=np.zeros(2001);dens[key]=np.zeros(2001)
                m=mass*st[key+'_defined_weight'] if key+'_defined_weight' in st else mass
                sums[key]+=np.nan_to_num(val,nan=0)*m;dens[key]+=np.isfinite(val)*m
            foldg.append(float(st['full_PCC_gain'][0]))
        values={k:np.divide(v,dens[k],out=np.full(2001,np.nan),where=dens[k]>0) for k,v in sums.items()}
        values['B_gamma_PCC'][:]=np.nan;bsall[c['model'],c['direction']]=values
        row=dict(weighting=mode,model=c['model'],direction=c['direction'],setting='within' if c['source']==c['target'] else 'cross',
                 drugs=len({d for p in panels for d in p['drug']}),chemical_groups=len(groups),lambda_used=1.,
                 folds_improved=int(np.sum(np.array(foldg)>1e-8)),folds_tied=int(np.sum(abs(np.array(foldg))<=1e-8)))
        for key,v in values.items():
            row[key]=v[0];row[key+'_lo']=np.quantile(v[1:],.025) if np.isfinite(v[1:]).all() else np.nan
            row[key+'_hi']=np.quantile(v[1:],.975) if np.isfinite(v[1:]).all() else np.nan
        full.append(row)
    for name in ['G5_true']+[f'shuffle_{s}' for s in core.SEEDS]:
        for label,directions in [('cross_mean',['6_to_14','14_to_6']),('within_mean',['6_to_6','14_to_14'])]:
            values={k:np.mean([bsall[name,d][k] for d in directions],axis=0) for k in bsall[name,directions[0]]}
            bsall[name,label]=values
            row=dict(weighting=mode,model=name,direction=label,setting=label,drugs=len({d for p in panels for d in p['drug']}),
                     chemical_groups=len(groups),lambda_used=1.)
            foldgain=per[(per.model==name)&per.direction.isin(directions)].groupby('fold').full_PCC_gain.mean().to_numpy()
            row.update(folds_improved=int(np.sum(foldgain>1e-8)),folds_tied=int(np.sum(abs(foldgain)<=1e-8)))
            for key,v in values.items():
                row[key]=v[0];row[key+'_lo']=np.quantile(v[1:],.025) if np.isfinite(v[1:]).all() else np.nan
                row[key+'_hi']=np.quantile(v[1:],.975) if np.isfinite(v[1:]).all() else np.nan
            full.append(row)
    table=pd.DataFrame(full);table.to_csv(dest/'WITHIN_CROSS_PLATE.csv',index=False);table.to_csv(dest/'FULL_RESPONSE_RESULTS.csv',index=False)
    contrasts=[]
    for direction in ['6_to_6','6_to_14','14_to_6','14_to_14','cross_mean','within_mean']:
        true=bsall['G5_true',direction]
        for compare in ['mean_three_shuffles']+[f'shuffle_{s}' for s in core.SEEDS]:
            nulls=[bsall[f'shuffle_{s}',direction] for s in core.SEEDS] if compare=='mean_three_shuffles' else [bsall[compare,direction]]
            for metric in ['full_PCC','full_MSE','gamma_PCC','gamma_MSE']:
                diff=true[metric]-np.mean([v[metric] for v in nulls],axis=0)
                contrasts.append(dict(weighting=mode,direction=direction,comparison='G5_true_minus_'+compare,metric=metric,
                     delta=diff[0],lo=np.quantile(diff[1:],.025),hi=np.quantile(diff[1:],.975),positive_is_better=metric.endswith('PCC')))
    pd.DataFrame(contrasts).to_csv(dest/'SHUFFLE_CONTROLS.csv',index=False)
    t=bsall['G5_true','cross_mean'];ns=[bsall[f'shuffle_{s}','cross_mean'] for s in core.SEEDS]
    adv=t['gamma_PCC']-np.mean([n['gamma_PCC'] for n in ns],axis=0)
    relation=bool(t['gamma_PCC'][0]>0 and all(bsall['G5_true',d]['gamma_PCC'][0]>0 for d in ['6_to_14','14_to_6']) and np.quantile(adv[1:],.025)>0)
    complete=bool(relation and t['gamma_MSE_gain'][0]>0 and t['full_PCC_gain'][0]>0 and t['full_MSE_gain'][0]>0
                  and np.quantile(t['full_PCC_gain'][1:],.025)>0 and np.quantile(t['full_MSE_gain'][1:],.025)>0)
    core.dump('PRIMARY_ENDPOINTS.json',dict(weighting=mode,cross_plate_relationship_evidence=relation,
       complete_prediction_improvement_evidence=complete,gamma_PCC=float(t['gamma_PCC'][0]),
       true_minus_mean_shuffle_gamma_PCC=float(adv[0]),true_minus_mean_shuffle_gamma_PCC_CI=list(map(float,np.quantile(adv[1:],[.025,.975]))),
       complete_rule_scope='Pre-specified cross-direction mean; direction-specific results must also be reported.',
       mathematical_clarification='Single-plate X1-only gamma is zero; its PCC is undefined. Predicted-gamma PCC positivity and superiority to shuffle replace a meaningless comparison to zero-PCC.',
       gain_sign='PCC new minus baseline; MSE baseline minus new.',scope='Conditional on one measured plate pair; historical development VALID.'))
    pure=per[['fold','model','direction']].copy();pure['MSE_gain_closure']=abs(per.full_MSE_gain-per.gamma_MSE_gain)
    assert pure.MSE_gain_closure.max()<1e-8
    core.dump('PURE_GAMMA_CHECK.json',dict(status='PASS_GRAM_AND_REWEIGHT_ALGEBRA',MSE_checks=pure.to_dict('records'),
       proof='Original per-fold direct checks establish zero cell marginal. Changing drug weights subtracts a shared cell vector whose cell mean is zero. Re-centering with the new weights therefore preserves both pure-gamma constraints.',
       independent_full_array_replay='Main fold0 verification_inputs.npz below uses the corrected weights and re-centered delta.'))
    return per

def main():
    configs=json.loads((SOURCE/'CONFIG_f0.json').read_text());panels=[]
    for f in range(5):
        with np.load(SOURCE/f'SUFFICIENT_f{f}.npz',allow_pickle=False) as z:panels.append({k:z[k] for k in z.files})
    groupmap={};roles=collections.Counter()
    for p in panels:
        assert len(set(p['drug']))==len(p['drug'])
        for d,g in zip(p['drug'],p['group']):
            if d in groupmap:assert groupmap[d]==g
            groupmap[d]=g;roles[d]+=1
    sizes=collections.Counter(groupmap.values());groups=sorted(sizes)
    corrected=[];ledger=[]
    for f,p in enumerate(panels):
        a=p.copy();a['q']=np.array([1/(len(groups)*sizes[groupmap[d]]*roles[d]) for d in p['drug']]);corrected.append(a)
        for d,g,qold,qnew in zip(p['drug'],p['group'],p['q'],a['q']):ledger.append(dict(fold=f,drug=d,group=g,inherited_q=qold,common_support_q=qnew,role_count=roles[d]))
    t=pd.DataFrame(ledger);m=t.groupby('group').common_support_q.sum();assert np.max(abs(m-1/len(groups)))<1e-12
    assert abs(t.common_support_q.sum()-1)<1e-12
    counts=np.random.default_rng(20261103).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    mainroot=SOURCE/'reweighted_common_support';sensroot=SOURCE/'inherited_weight_sensitivity'
    mainper=summarize(configs,corrected,'common_support_group_equal',mainroot,counts,groups)
    core.R=mainroot;t.to_csv(mainroot/'WEIGHT_LEDGER.csv',index=False)
    core.dump('WEIGHT_CORRECTION_AUDIT.json',dict(reason='Original 153-drug role weights restricted to the 47-drug plate intersection do not give equal total mass to the 36 groups on the actual common support.',
       decision_basis='Identity/support metadata, not observed effect sizes; no model/threshold/condition changes.',
       n_drugs=len(groupmap),n_groups=len(groups),rule='q(d,fold)=1/[number common groups * common drugs in group * actual shared-fold appearances of d]',
       group_mass_min=float(m.min()),group_mass_max=float(m.max()),total_mass=float(t.common_support_q.sum()),
       refit=False,new_response_reads=False,new_inference=False,
       exact_reuse='Source-beta centering changes only a drug-shared vector. A fixed linear G5 maps this to a drug-shared vector, removed by the single-context gamma projection. Cached uncentered Gram statistics therefore exactly support reweighting.'))
    # Write a separate verification input for the recalculated weights.
    with np.load(SOURCE/'verification_inputs.npz',allow_pickle=False) as z:v={k:z[k] for k in z.files}
    p=corrected[0];assert np.array_equal(v['drug'],p['drug']);w=p['q']/p['q'].sum()
    delta=v['delta']-np.einsum('cdg,d->cg',v['delta'],w)[:,None]
    direct=core.direct_metrics(v['Y'],v['B'],delta,w)
    row=mainper[(mainper.fold==0)&(mainper.model=='G5_true')&(mainper.direction=='6_to_14')].iloc[0]
    err=max(abs(direct[k]-row[k]) for k in direct if np.isfinite(direct[k]));assert err<1e-8
    dd=core.dec(delta,w);pureerr=max(np.max(abs(dd[k])) for k in ['mu','alpha','beta']);assert pureerr<1e-8
    v.update(w=w,delta=delta,claimed_full_pcc_gain=row.full_PCC_gain,claimed_full_mse_gain=row.full_MSE_gain,claimed_gamma_mse_gain=row.gamma_MSE_gain)
    np.savez_compressed(mainroot/'verification_inputs.npz',**v)
    core.dump('REWEIGHT_DIRECT_VERIFICATION.json',dict(status='PASS',fold=0,direction='6_to_14',full_array_vs_Gram_maxabs=float(err),pure_marginal_maxabs=float(pureerr)))
    summarize(configs,panels,'original_153_drug_inherited',sensroot,counts,groups)
    core.R=mainroot
    core.dump('RESOURCE_REWEIGHT.json',dict(job_id=os.environ['TAHOE_REPRO_RUN_ID'],CPU=core.CPUS,GPU=0,
       elapsed_seconds=time.monotonic()-START,allocated_core_hours=(time.monotonic()-START)*core.CPUS/3600,
       process_cpu_seconds=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       new_fits=0,new_frozen_response_reads=0,TEST_response_reads=0))
    print('EXP2_REWEIGHT_COMPLETE',len(groupmap),len(groups),'seconds',time.monotonic()-START,flush=True)

if __name__=='__main__':main()

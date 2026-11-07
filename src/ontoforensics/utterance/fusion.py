"""Ontology-compiled recording features with held-out calibration."""
import itertools
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from .data import write, ROLES, validate, identity, fingerprint
from ..metrics import binary_metrics, choose_threshold, risk_coverage

def table(rows, observations, experts):
    if not experts or len(set(experts)) != len(experts):
        raise ValueError('Expert names must be nonempty and unique')
    ids = {r['recording_id'] for r in rows}
    lookup={}; sources={}
    for o in observations:
        if o['expert'] not in experts or o['recording_id'] not in ids: continue
        key=(o['recording_id'],o['expert'])
        if key in lookup:
            if lookup[key]!=o: raise ValueError('Conflicting duplicate evidence')
            continue
        if o.get('higher_is_fake') is not True or o.get('scope')!='utterance':
            raise ValueError('Evidence direction/scope not explicit')
        if o.get('status') not in ('valid', 'missing', 'failed'):
            raise ValueError('Unknown evidence status')
        source=(o['checkpoint_sha256'],o['config_sha256'])
        if not all(isinstance(v, str) and v for v in source):
            raise ValueError('Evidence provenance is missing')
        if o['expert'] in sources and sources[o['expert']]!=source:
            raise ValueError('Mixed model versions')
        sources[o['expert']]=source
        lookup[key]=o
    if len({v[0] for v in sources.values()})!=len(sources):
        raise ValueError('Same model registered as multiple experts')
    scores=np.full((len(rows),len(experts)),np.nan); q=np.zeros((len(rows),2))
    for i,r in enumerate(rows):
        qualities=[]
        for j,e in enumerate(experts):
            o=lookup.get((r['recording_id'],e))
            if o and o['status']=='valid':
                if isinstance(o['score'],bool) or not isinstance(o['score'],(int,float)) or not np.isfinite(o['score']):
                    raise ValueError('Score must be a finite number')
                quality=np.asarray(o['quality'],dtype=float)
                if quality.shape!=(2,) or not np.isfinite(quality).all():
                    raise ValueError('Quality requires finite RMS and clipping fraction')
                if not 0<=quality[1]<=1: raise ValueError('Invalid clipping fraction')
                scores[i,j]=o['score']; qualities.append(quality)
        if qualities:
            if not np.allclose(qualities,qualities[0]): raise ValueError('Inconsistent audio quality')
            q[i]=qualities[0]
    if not np.isfinite(q).all(): raise ValueError('Nonfinite quality')
    return scores,q,sources

def features(scores,q,experts,ontology,stats,mode):
    valid=np.isfinite(scores); z=np.where(valid,(scores-stats[0])/stats[1],0)
    values=[z,valid.astype(float)]; names=['score/'+e for e in experts]+['valid/'+e for e in experts]
    if ontology.get('quality', True):
        values.append(q)
        names += ['quality/rms_db', 'quality/clipping_fraction']
    if mode in ('relation_lr','pairwise_lr'):
        pairs=list(itertools.combinations(range(len(experts)),2))
        groups={}
        for a,b in pairs:
            relation='same_family' if ontology['families'][experts[a]]==ontology['families'][experts[b]] else 'different_family'
            key=f'{experts[a]}:{experts[b]}' if mode=='pairwise_lr' else relation
            groups.setdefault(key,[]).append((a,b))
        if ontology.get('relations',True):
            for kind,pairs in sorted(groups.items()):
                mask=np.stack([valid[:,a]&valid[:,b] for a,b in pairs],axis=1)
                count=mask.sum(1); denom=np.maximum(count,1)
                product=np.stack([z[:,a]*z[:,b] for a,b in pairs],axis=1)
                diff=np.stack([abs(z[:,a]-z[:,b]) for a,b in pairs],axis=1)
                values.append(np.column_stack([(product*mask).sum(1)/denom,(diff*mask).sum(1)/denom,count/len(pairs)]))
                names += [kind+'/'+x for x in ('product','disagreement','availability')]
        if ontology.get('quality',True):
            for k,kind in enumerate(('rms_db','clipping_fraction')):
                values.append(z*q[:,k,None]); names += [kind+'/'+e for e in experts]
    return np.concatenate(values,axis=1),names

def raw(model,x,z):
    if model['mode']=='mean': return z.mean(1)
    classifier=model['classifier']
    if hasattr(classifier,'decision_function'): return classifier.decision_function(x)
    p=np.clip(classifier.predict_proba(x)[:,1],1e-8,1-1e-8)
    return np.log(p/(1-p))

def fit(rows,observations,ontology,mode,output,target_fpr=.05,seed=17):
    validate(rows)
    if mode not in ('mean','flat_lr','mlp','relation_lr','pairwise_lr'):
        raise ValueError('Unknown fusion mode: '+mode)
    experts=ontology['experts']
    if set(experts) - set(ontology.get('families', {})):
        raise ValueError('Every expert needs an ontology family')
    all_fitting=[r for r in rows if r['role'] not in ('test', 'unspecified')]
    rows=[r for r in rows if r['role'] in ROLES]
    # Test evidence is never inspected when fitting, selecting, or calibrating.
    s,q,sources=table(rows,observations,experts)
    masks={role:np.array([r['role']==role for r in rows]) for role in ROLES}
    # Never silently train on incomplete evidence; missingness is evaluated separately.
    for role,mask in masks.items():
        if not np.isfinite(s[mask]).all(): raise ValueError('Incomplete evidence in '+role)
    train=masks['fusion_train']; selection=masks['fusion_select']
    stats=(np.mean(s[train],0),np.maximum(np.std(s[train],0),1e-8))
    x,names=features(s,q,experts,ontology,stats,mode)
    scaler=StandardScaler().fit(x[train]); x=scaler.transform(x)
    y=np.array([r['label'] for r in rows]); z=(s-stats[0])/stats[1]
    model=dict(format_version=2,mode=mode,ontology=ontology,stats=stats,scaler=scaler,names=names,sources=sources)
    best=float('inf')
    trials=[]
    for c in ([.01,.1,1,10] if mode!='mean' else [1]):
        messages=[]
        if mode!='mean':
            classifier=(MLPClassifier(hidden_layer_sizes=(32,),alpha=1/c,max_iter=500,random_state=seed)
                        if mode=='mlp' else LogisticRegression(C=c,max_iter=2000))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                classifier.fit(x[train],y[train])
            messages=[str(w.message) for w in caught]
            model['classifier']=classifier
        metric=binary_metrics(y[selection],expit(raw(model,x[selection],z[selection])))['nll']
        trials.append(dict(C=c,selection_nll=metric,warnings=messages))
        if metric<best:
            import copy
            winner=copy.deepcopy(model); best=metric; best_c=c; best_warnings=messages
    model=winner
    cal=masks['calibration']; threshold=masks['threshold']
    calibrator=LogisticRegression(C=1e6).fit(raw(model,x[cal],z[cal])[:,None],y[cal])
    model['calibrator']=calibrator
    p=calibrator.predict_proba(raw(model,x[threshold],z[threshold])[:,None])[:,1]
    model['threshold']=choose_threshold(y[threshold],p,target_fpr)
    model['fit_ids']=[r['recording_id'] for r in all_fitting]
    model['fit_groups']=sorted({r['group_id'] for r in all_fitting})
    model['fit_locations']=[identity(r) for r in all_fitting]
    model['fit_audio_hashes']=sorted({r['audio_sha256'] for r in all_fitting if r.get('audio_sha256')})
    model['manifest_fingerprint']=fingerprint(all_fitting)
    report=dict(mode=mode,features=names,selection_nll=best,selected_C=best_c,
                convergence_warnings=best_warnings,trials=trials,seed=seed,
                target_fpr=target_fpr,threshold=model['threshold'],sources=sources,
                manifest_fingerprint=model['manifest_fingerprint'])
    model['training_report']=report
    Path(output).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,output)
    Path(str(output)+'.report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report

def evaluate(rows,observations,model_path,output,predict=False):
    model=joblib.load(model_path); ontology=model['ontology']; experts=ontology['experts']
    if model.get('format_version') != 2:
        raise ValueError('Unsupported fusion artifact; refit with the current code')
    rows=[r for r in rows if r['role'] in (('test','unspecified') if predict else ('test',))]
    if not rows: raise ValueError('No evaluation recordings selected')
    validate(rows,require_roles=False,allow_unlabelled=predict)
    if set(model['fit_ids']) & {r['recording_id'] for r in rows}: raise ValueError('Evaluation leakage')
    if set(model['fit_groups']) & {r['group_id'] for r in rows}:
        raise ValueError('Evaluation speaker/group leakage')
    if set(map(tuple,model['fit_locations'])) & {identity(r) for r in rows}:
        raise ValueError('Evaluation audio location leakage')
    if set(model['fit_audio_hashes']) & {r.get('audio_sha256') for r in rows if r.get('audio_sha256')}:
        raise ValueError('Evaluation audio content leakage')
    s,q,sources=table(rows,observations,experts)
    if any(model['sources'].get(e)!=source for e,source in sources.items()):
        raise ValueError('Evidence model version changed')
    x,names=features(s,q,experts,ontology,model['stats'],model['mode']); x=model['scaler'].transform(x)
    z=np.where(np.isfinite(s),(s-model['stats'][0])/model['stats'][1],0)
    margins=raw(model,x,z); p=model['calibrator'].predict_proba(margins[:,None])[:,1]
    predictions=[]; metrics={}
    for i,r in enumerate(rows):
        complete=bool(np.isfinite(s[i]).all())
        row=dict(recording_id=r['recording_id'],dataset=r['dataset'],label=r.get('label'),
                 probability=float(p[i]) if complete else None,status='valid' if complete else 'missing_expert',
                 prediction=int(p[i]>=model['threshold']) if complete else None,
                 missing_experts=[e for e,v in zip(experts,np.isfinite(s[i])) if not v],
                 precalibration_logit=float(margins[i]) if complete else None,
                 evidence=[dict(expert=e,score=float(s[i,j]),checkpoint_sha256=model['sources'][e][0],
                                config_sha256=model['sources'][e][1],family=ontology['families'][e])
                           for j,e in enumerate(experts) if np.isfinite(s[i,j])])
        classifier=model.get('classifier')
        if complete and classifier is not None and hasattr(classifier,'coef_'):
            row['contributions']=dict(zip(names,(x[i]*classifier.coef_[0]).tolist()))
            row['intercept']=float(classifier.intercept_[0])
            slope=float(model['calibrator'].coef_[0,0])
            row['calibrated_contributions']={k:v*slope for k,v in row['contributions'].items()}
            row['calibrated_intercept']=row['intercept']*slope+float(model['calibrator'].intercept_[0])
        predictions.append(row)
    for dataset in sorted({r['dataset'] for r in rows}):
        selected=[r for r in predictions if r['dataset']==dataset and r['status']=='valid' and r['label'] in (0,1)]
        m=binary_metrics([r['label'] for r in selected],[r['probability'] for r in selected],model['threshold'])
        m['total']=sum(r['dataset']==dataset for r in rows)
        m['usable']=sum(r['dataset']==dataset and r['status']=='valid' for r in predictions)
        m['coverage']=m['usable']/m['total']
        m['risk_coverage']=risk_coverage([r['label'] for r in selected],[r['probability'] for r in selected],
                                        model['threshold'],n_total=m['total'])
        if selected:
            m['fnr']=1-m['recall'] if m['recall'] is not None else None; m['balanced_accuracy']=(m['recall']+1-m['fpr'])/2 if m['fpr'] is not None and m['recall'] is not None else None
        metrics[dataset]=m
    write(Path(output)/'predictions.jsonl',predictions)
    Path(output,'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    return metrics

"""Reproducible experiment matrix; external test scores never select models."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .data import read, write, digest
from .experts import train, export
from .fusion import fit, evaluate
from ..metrics import binary_metrics

def paired_bootstrap(a,b,rows,iterations=1000,seed=17):
    a={r['recording_id']:r for r in a if r['status']=='valid'}
    b={r['recording_id']:r for r in b if r['status']=='valid'}
    selected=[r for r in rows if r['recording_id'] in a and r['recording_id'] in b]
    groups={}
    for r in selected: groups.setdefault(r['group_id'],[]).append(r)
    keys=list(groups); rng=np.random.default_rng(seed); delta=[]
    if iterations < 1: return {'n':len(selected),'groups':len(keys),'valid_replicates':0,'delta_eer_ci95':None}
    if not keys: return {'n':0}
    for _ in range(iterations):
        sample=[r for k in rng.choice(keys,len(keys),replace=True) for r in groups[k]]
        y=[r['label'] for r in sample]
        if len(set(y))<2: continue
        scores=[[table[r['recording_id']]['probability'] for r in sample] for table in (a,b)]
        metrics=[binary_metrics(y,p)['eer_interpolated'] for p in scores]
        delta.append(metrics[0]-metrics[1])
    return dict(n=len(selected),groups=len(keys),valid_replicates=len(delta),
                delta_eer_ci95=np.quantile(delta,[.025,.975]).tolist() if delta else None,
                grouping='speaker where known, otherwise recording',difference='first_minus_second')

def run(manifest,config_dir,output,device='cuda:0',seeds=(17,42,2026),bootstrap=1000,target_fpr=.05):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; full experiments have not been run')
    rows=read(manifest); output=Path(output); config_dir=Path(config_dir)
    if not seeds: raise ValueError('At least one seed required')
    from .data import validate
    validate(rows)
    ontology=json.loads((config_dir/'ontology.json').read_text())
    summary=[]
    for seed in seeds:
        root=output/str(seed); observations=[]
        for expert in ontology['experts']:
            config=json.loads((config_dir/(expert+'.json')).read_text()); config['seed']=seed
            target=root/expert
            code_files=[Path(__file__),Path(__file__).with_name('experts.py'),Path(__file__).with_name('data.py')]
            if expert in ('aasist','ssl'): code_files.append(Path(__file__).parent/'vendor'/(expert+'.py'))
            signature=dict(config=config,manifest_sha256=digest(manifest),
                           code_sha256={p.name:digest(p) for p in code_files},torch_version=torch.__version__)
            stamp=target/'completed.json'; scorefile=target/'scores.jsonl'
            if stamp.exists():
                saved=json.loads(stamp.read_text())
                if (saved['signature']!=signature or saved['scores_sha256']!=digest(scorefile)
                    or saved.get('checkpoint_sha256')!=digest(target/'best.pt')):
                    raise ValueError('Stale experiment cache: '+str(target))
            else:
                started=target/'started.json'
                if started.exists() and json.loads(started.read_text())!=signature:
                    raise ValueError('Incomplete experiment has changed configuration or code: '+str(target))
                target.mkdir(parents=True,exist_ok=True)
                started.write_text(json.dumps(signature,indent=2))
                last=target/'last.pt'
                train(rows,config,target,device,resume=last if last.exists() else None)
                export(rows,target/'best.pt',scorefile,device)
                stamp.write_text(json.dumps(dict(signature=signature,scores_sha256=digest(scorefile),
                                                checkpoint_sha256=digest(target/'best.pt')),indent=2))
            observations.extend(read(scorefile))
        variants=[(m,m,copy.deepcopy(ontology)) for m in ['mean','flat_lr','mlp','relation_lr','pairwise_lr']]
        for flag in ['relations','quality']:
            o=copy.deepcopy(ontology); o[flag]=False
            variants.append(('no_'+flag,'relation_lr',o))
        for expert in ontology['experts']:
            o=copy.deepcopy(ontology); o['experts'].remove(expert)
            if o['experts']: variants.append(('without_'+expert,'relation_lr',o))
            single=copy.deepcopy(ontology); single['experts']=[expert]
            variants.append(('single_'+expert,'mean',single))
        for name,mode,o in variants:
            folder=root/name; model=folder/'model.joblib'
            fit(rows,observations,o,mode,model,target_fpr=target_fpr,seed=seed)
            metrics=evaluate(rows,observations,model,folder)
            summary.append(dict(seed=seed,variant=name,metrics=metrics))
        a=read(root/'relation_lr/predictions.jsonl'); b=read(root/'flat_lr/predictions.jsonl')
        comparisons={d:paired_bootstrap(a,b,[r for r in rows if r['dataset']==d and r['role']=='test'],iterations=bootstrap,seed=seed)
                     for d in {r['dataset'] for r in rows if r['role']=='test'}}
        (root/'paired_bootstrap.json').write_text(json.dumps(comparisons,indent=2))
        write(output/'results.jsonl',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--configs',default='configs/utterance'); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',nargs='+',type=int,default=[17,42,2026]); p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--target-fpr',type=float,default=.05)
    a=p.parse_args(); run(a.manifest,a.configs,a.output,a.device,a.seeds,a.bootstrap,a.target_fpr)

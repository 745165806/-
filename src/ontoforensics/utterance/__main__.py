import argparse
import json
from pathlib import Path
from .data import read, prepare, write, validate

def select(rows, per_class=2, seed=17):
    """A bounded engineering pilot within fixed roles; never a benchmark result."""
    import numpy as np
    if per_class < 1: raise ValueError('per-class must be positive')
    rng=np.random.default_rng(seed); selected=[]
    for role,dataset in sorted({(r['role'],r['dataset']) for r in rows}):
        for label in (0,1):
            candidates=[r for r in rows if r['role']==role and r['dataset']==dataset and r['label']==label]
            order=rng.permutation(len(candidates))[:per_class]
            selected.extend(candidates[int(i)] for i in order)
    validate(selected)
    return selected

def main():
    p=argparse.ArgumentParser(description='Whole-recording speech forgery research')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--indexes',nargs='+',required=True); a.add_argument('--output',required=True); a.add_argument('--check-audio',action='store_true'); a.add_argument('--seed',type=int,default=17)
    a=sub.add_parser('select'); a.add_argument('--manifest',required=True); a.add_argument('--output',required=True); a.add_argument('--per-class',type=int,default=2); a.add_argument('--seed',type=int,default=17)
    a=sub.add_parser('train-expert'); a.add_argument('--manifest',required=True); a.add_argument('--config',required=True); a.add_argument('--output',required=True); a.add_argument('--device',default='cpu'); a.add_argument('--resume')
    a=sub.add_parser('export'); a.add_argument('--manifest',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--output',required=True); a.add_argument('--device',default='cpu')
    a=sub.add_parser('fit-fusion'); a.add_argument('--manifest',required=True); a.add_argument('--scores',nargs='+',required=True); a.add_argument('--ontology',required=True); a.add_argument('--mode',choices=['mean','flat_lr','mlp','relation_lr','pairwise_lr'],default='relation_lr'); a.add_argument('--output',required=True)
    a.add_argument('--target-fpr',type=float,default=.05); a.add_argument('--seed',type=int,default=17)
    for name in ('evaluate','predict'):
        a=sub.add_parser(name); a.add_argument('--manifest',required=True); a.add_argument('--scores',nargs='+',required=True); a.add_argument('--model',required=True); a.add_argument('--output',required=True)
    a=sub.add_parser('export-graph'); a.add_argument('--predictions',required=True); a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='prepare': result=prepare(args.indexes,args.output,seed=args.seed,check_audio=args.check_audio)
    elif args.command=='select':
        rows=select(read(args.manifest),args.per_class,args.seed); write(args.output,rows)
        result={'recordings':len(rows),'engineering_pilot_only':True}
    elif args.command=='train-expert':
        from .experts import train
        result=train(read(args.manifest),json.loads(Path(args.config).read_text()),args.output,args.device,args.resume)
    elif args.command=='export':
        from .experts import export
        result=export(read(args.manifest),args.checkpoint,args.output,args.device)
    elif args.command=='export-graph':
        from .graph import export_graph
        result=export_graph(read(args.predictions),args.output)
    else:
        from .fusion import fit,evaluate
        rows=read(args.manifest); observations=[o for file in args.scores for o in read(file)]
        if args.command=='fit-fusion': result=fit(rows,observations,json.loads(Path(args.ontology).read_text()),args.mode,args.output,args.target_fpr,args.seed)
        else: result=evaluate(rows,observations,args.model,args.output,predict=args.command=='predict')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()

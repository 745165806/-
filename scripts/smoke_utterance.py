"""Bounded real-audio two-expert verification, never a benchmark experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from ontoforensics.utterance.__main__ import select
from ontoforensics.utterance.data import read,write,audio
from ontoforensics.utterance.experiment import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args(); root=Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new output directory to avoid mixing verification runs')
    root.mkdir(parents=True,exist_ok=True)
    rows=select(read(args.manifest),per_class=2)
    for row in rows:
        row['audio_sha256']=hashlib.sha256(audio(row).tobytes()).hexdigest()
    write(root/'manifest.jsonl',rows)
    configs=root/'configs'; configs.mkdir()
    project=Path(__file__).resolve().parents[1]
    for name in ('lfcc_cnn','aasist'):
        config=json.loads((project/'configs/utterance'/(name+'.json')).read_text())
        config.update(epochs=1,patience=1,cpu_threads=2)
        if name=='lfcc_cnn': config['samples']=16000
        (configs/(name+'.json')).write_text(json.dumps(config,indent=2))
    ontology=dict(version=1,experts=['aasist','lfcc_cnn'],families=dict(aasist='raw_waveform',lfcc_cnn='lfcc'),relations=True,quality=True)
    (configs/'ontology.json').write_text(json.dumps(ontology,indent=2))
    results=run(root/'manifest.jsonl',configs,root/'matrix',args.device,seeds=[17],bootstrap=50)
    report=dict(engineering_verification_only=True,recordings=len(rows),experts=ontology['experts'],
                fusion_variants=len(results),datasets=sorted({r['dataset'] for r in rows}),
                test_recordings=sum(r['role']=='test' for r in rows),epochs_per_expert=1,device=args.device)
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()

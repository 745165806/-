"""Real waveform experts, source-only training, full-recording inference."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import expit

from .data import audio, quality, digest, write, validate, fingerprint
from ..metrics import binary_metrics

class LFCCCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear-spaced triangular filterbank followed by orthonormal DCT.
        frequencies = torch.linspace(0, 8000, 257)
        edges = torch.linspace(0, 8000, 62)
        filters = torch.minimum((frequencies[None]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],
                                (edges[2:,None]-frequencies[None])/(edges[2:]-edges[1:-1])[:,None]).clamp(min=0)
        k = torch.arange(20)[:,None]
        dct = torch.cos(torch.pi/60*(torch.arange(60)[None]+.5)*k)*(2/60)**.5
        dct[0] /= 2**.5
        self.register_buffer('filters', filters)
        self.register_buffer('dct', dct)
        self.register_buffer('window', torch.hann_window(400))
        self.net = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),
                                 nn.Flatten(),nn.Linear(32,2))
    def forward(self, x):
        power = torch.stft(x,n_fft=512,hop_length=160,win_length=400,window=self.window,return_complex=True).abs().square()
        cepstra = self.dct @ (self.filters @ power).clamp(min=1e-10).log()
        return self.net(cepstra[:,None])

def build(config, device):
    name = config['expert']
    if name == 'lfcc_cnn':
        model = LFCCCNN()
    elif name == 'aasist':
        from .vendor.aasist import Model
        model = Model(config['model_config'])
    elif name == 'ssl':
        checkpoint = os.environ.get('ONTO_XLSR_CHECKPOINT') or config.get('xlsr_checkpoint')
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError('Set ONTO_XLSR_CHECKPOINT to an existing XLS-R checkpoint')
        os.environ['ONTO_XLSR_CHECKPOINT'] = str(Path(checkpoint).resolve())
        try:
            from .vendor.ssl import Model
        except (ImportError, ValueError) as exc:
            raise RuntimeError('SSL requires a compatible upstream fairseq environment; see docs/UTTERANCE.md') from exc
        model = Model(None, device)
    else:
        raise ValueError('Unknown expert: '+name)
    return model.to(device)

def logits(model, x):
    out = model(x)
    return out[-1] if isinstance(out, tuple) else out

def chunks(x, size):
    for start in range(0,len(x),size):
        part = x[start:start+size]
        yield np.pad(part,(0,size-len(part)),mode='wrap') if len(part)<size else part

@torch.no_grad()
def score(model, row, size, device):
    model.eval()
    x = audio(row)
    values = [logits(model,torch.from_numpy(v[None]).to(device)).cpu().numpy()[0]
              for v in chunks(x,size)]
    lengths = [min(size,len(x)-start) for start in range(0,len(x),size)]
    out = np.average(values,axis=0,weights=lengths)
    if not np.isfinite(out).all() or np.shape(out)!=(2,):
        raise ValueError('Expert must output two finite logits')
    return float(out[1]-out[0]), quality(x)

def train(rows, config, output, device='cpu', resume=None):
    validate(rows)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this environment')
    seed = config.get('seed',17)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.get('cpu_threads',4))
    model = build(config,device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.get('lr',1e-4),weight_decay=1e-4)
    training = [r for r in rows if r['role']=='expert_train']
    selection = [r for r in rows if r['role']=='expert_select']
    if {r['label'] for r in training} != {0,1} or {r['label'] for r in selection} != {0,1}:
        raise ValueError('Training and selection require both classes')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if resume is None and any((output/name).exists() for name in ('best.pt','last.pt')):
        raise ValueError('Existing checkpoint found; use --resume or a new output directory')
    size=config.get('samples',64600); micro=config.get('micro_batch',4)
    effective=config.get('effective_batch',24)
    if size < 512 or micro < 1 or effective < 1:
        raise ValueError('samples must be at least 512 and batch sizes must be positive')
    if effective % micro: raise ValueError('Effective batch must be divisible by micro batch')
    signature=fingerprint([r for r in rows if r['role'] in ('expert_train','expert_select')])
    best=float('inf'); stale=0; history=[]; start=time.monotonic(); first_epoch=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['config']!=config or state.get('manifest_fingerprint')!=signature:
            raise ValueError('Resume configuration or data signature changed')
        if (Path(resume).resolve().parent != output.resolve() or not (output/'best.pt').is_file()):
            raise ValueError('Resume from the existing run directory, retaining best.pt')
        model.load_state_dict(state['state_dict']); optimizer.load_state_dict(state['optimizer'])
        best=state['best']; stale=state['stale']; history=state['history']; first_epoch=state['epoch']+1
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        if stale>=config.get('patience',10): return history
    for epoch in range(first_epoch,config.get('epochs',100)):
        training.sort(key=lambda r:r['recording_id'])
        random.shuffle(training); model.train(); optimizer.zero_grad(); losses=[]
        # Normalize each accumulation group by its actual number of examples.
        for group_start in range(0,len(training),effective):
            group=training[group_start:group_start+effective]
            for offset in range(0,len(group),micro):
                batch=group[offset:offset+micro]; waves=[]
                for r in batch:
                    x=audio(r)
                    if len(x)>size:
                        begin=random.randint(0,len(x)-size); x=x[begin:begin+size]
                    waves.append(next(chunks(x,size)))
                out=logits(model,torch.from_numpy(np.stack(waves)).to(device))
                y=torch.tensor([r['label'] for r in batch],device=device)
                loss=nn.functional.cross_entropy(out,y,reduction='sum')/len(group)
                loss.backward(); losses.append(float(loss.detach())*len(group))
            optimizer.step(); optimizer.zero_grad()
        margins=[score(model,r,size,device)[0] for r in selection]
        eer=binary_metrics([r['label'] for r in selection],expit(margins))['eer_interpolated']
        history.append(dict(epoch=epoch,eer=eer,loss=sum(losses)/len(training),elapsed_sec=time.monotonic()-start))
        improved=eer<best
        if eer<best:
            best=eer; stale=0
        else: stale+=1
        state=dict(state_dict=model.state_dict(),config=config,epoch=epoch,optimizer=optimizer.state_dict(),
                   training_ids=[r['recording_id'] for r in training],manifest_fingerprint=signature,
                   best=best,stale=stale,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state,output/'last.pt.tmp'); (output/'last.pt.tmp').replace(output/'last.pt')
        if improved:
            torch.save(state,output/'best.pt.tmp'); (output/'best.pt.tmp').replace(output/'best.pt')
        write(output/'history.jsonl',history)
        if stale>=config.get('patience',10): break
    return history

def export(rows, checkpoint, output, device='cpu'):
    if Path(checkpoint).resolve()==Path(output).resolve():
        raise ValueError('Score output cannot overwrite checkpoint')
    validate(rows,require_roles=False,allow_unlabelled=True)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    config=state['config']; model=build(config,device); model.load_state_dict(state['state_dict'])
    torch.set_num_threads(config.get('cpu_threads',4))
    checkpoint_hash=digest(checkpoint)
    config_hash=__import__('hashlib').sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    observations=[]
    for r in rows:
        value,q=score(model,r,config.get('samples',64600),device)
        observations.append(dict(recording_id=r['recording_id'],expert=config['expert'],score=value,
                                 status='valid',quality=q,checkpoint_sha256=checkpoint_hash,
                                 config_sha256=config_hash,higher_is_fake=True,scope='utterance'))
    write(output,observations)
    report={'count':len(observations),'checkpoint_sha256':checkpoint_hash,'config_sha256':config_hash,
            'scope':'utterance','aggregation':'duration_weighted_chunk_logits',
            'manifest_fingerprint':fingerprint(rows),'training_manifest_fingerprint':state.get('manifest_fingerprint')}
    Path(str(output)+'.report.json').write_text(json.dumps(report,indent=2))
    return report

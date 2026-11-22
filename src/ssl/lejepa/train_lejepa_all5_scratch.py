#!/usr/bin/env python3
"""Radar LeJEPA adaptation: official DINOv2 ViT-S/14 + new MLP projector.
Official SIGReg library; per-view Epps-Pulley random projections.
Two global crops define a differentiable center; all views regress toward it.
No EMA teacher, stop-gradient, masked-patch objective or unit normalization of z.
Uses identical inventory/sampling/augmentations to train_dinov2_all_radar.py.
Held-out frames are excluded, not evaluated by this script. HAXR boundary
filter is conservative and does not certify interior frame completeness.
This is a radar-specific experiment, not reproduction of authors' ImageNet run.
References: https://github.com/galilai-group/lejepa
https://github.com/facebookresearch/dinov2
"""
import os
os.environ['XFORMERS_DISABLED']='1'
import argparse, csv, hashlib, json, math, random, importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from all5_inventory import all5_inventory
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path('/home/remoteuser/projects/myproject/external-datasets')
    p.add_argument('--raw-roots', nargs='*', type=Path, default=[root/'CANOE',root/'Boreas'])
    p.add_argument('--prepared-roots', nargs='*', type=Path, default=[root/'SSL-prepared/HAXR-polar'])
    p.add_argument('--cartesian-roots', nargs='*', type=Path, default=[root/'SSL-prepared/Pohang-Cartesian'])
    p.add_argument('--reference-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init', choices=['dinov2','scratch'], default='dinov2')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--frame-stride', type=int, default=16)
    p.add_argument('--prepared-stride', type=int, default=1)
    p.add_argument('--local-crops', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=2e-5)
    p.add_argument('--projector-lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--proj-dim', type=int, default=128)
    p.add_argument('--num-slices', type=int, default=256)
    p.add_argument('--sigreg-weight', type=float, default=.02)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=0, help='Smoke test in separate output only; cannot resume.')
    p.add_argument('--resume', type=Path)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--professor-root', type=Path, default=Path(
        '/home/remoteuser/projects/myproject/Prof-dataset'))
    p.add_argument('--mae-manifest', type=Path)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.frame_stride,a.prepared_stride,a.proj_dim,a.num_slices)<1 or a.batch_size<2:
        p.error('Positive settings required; batch-size >= 2.')
    if not 0<a.sigreg_weight<1 or min(a.encoder_lr,a.projector_lr)<=0:
        p.error('sigreg-weight must be between 0 and 1; learning rates > 0.')
    if min(a.workers,a.local_crops,a.warmup_epochs,a.max_steps)<0:
        p.error('workers/crops/warmup/max-steps cannot be negative.')
    if a.resume and a.max_steps:p.error('Smoke tests cannot be resumed.')
    return a


class Encoder(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone=backbone
        self.projector=nn.Sequential(
            nn.Linear(backbone.embed_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,dim))
    def forward(self,x):
        emb=self.backbone.forward_features(x)['x_norm_clstoken']
        with torch.autocast('cuda', enabled=False):
            return self.projector(emb.float())

    def forward_views(self, views):
        embeddings = [
            self.backbone.forward_features(v)['x_norm_clstoken']
            for v in views
        ]
        # Normalize the projector over all images and views together.
        with torch.autocast('cuda', enabled=False):
            projected = self.projector(torch.cat(embeddings, dim=0).float())
        return projected.reshape(len(views), views[0].shape[0], -1)


def objective(z, regularizer, weight):
    # z: [views, batch, projection_dim]. Center stays in the autograd graph.
    z=z.float()
    center=z.mean(dim=0,keepdim=True)
    invariance=(z-center).square().mean()
    gaussian=torch.stack([regularizer(view) for view in z]).mean()
    return (1-weight)*invariance+weight*gaussian, invariance, gaussian

def date_group(sequence):
    # CANOE/Boreas recording dates; prepared sources should use recording/session names.
    import re
    match = re.search(r'\d{4}-\d{2}-\d{2}', sequence)
    return match.group(0) if match else sequence

def inventory(a):
    return all5_inventory(a)


def load_intensity(record):
    with Image.open(record['path']) as im:
        if record['strip11']:
            if im.mode != 'L' or im.width <= 11:
                raise ValueError(f'Expected grayscale raw radar PNG: {record["path"]}')
            image = im.crop((11, 0, im.width, im.height)).copy()
        else:
            image = im.convert('L').copy()
    return image

def crop_view(image, size, low, high, representation="polar"):
    # Independent azimuth/range spans for a polar baseline. No camera colour jitter/flips.
    w, h = image.size
    if representation == 'cartesian':
        side = max(1, int(min(w, h) * random.uniform(low, high)))
        cw = ch = side  # preserve x/y scale within Cartesian crops
    else:
        cw = max(1, int(w * random.uniform(low, high)))
        ch = max(1, int(h * random.uniform(low, high)))
    left, top = random.randint(0, w-cw), random.randint(0, h-ch)
    image = image.crop((left, top, left+cw, top+ch)).resize((size, size), Image.Resampling.BILINEAR)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.9, 1.1))
    array = np.asarray(image, dtype=np.float32).copy() / 255.
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = tensor.new_tensor([.229, .224, .225])[:, None, None]
    return (tensor-mean)/std

class RadarDataset(Dataset):
    def __init__(self, records, local_crops):
        self.records, self.local_crops = records, local_crops
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        image = load_intensity(self.records[index])
        representation = self.records[index]['representation']
        return [crop_view(image, 224, .5, 1., representation) for _ in range(2)] + [
            crop_view(image, 98, .2, .5, representation) for _ in range(self.local_crops)]

def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def main():
    a=arguments()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    all_records,records=inventory(a)
    signature=hashlib.sha256(json.dumps(all_records,sort_keys=True).encode()).hexdigest()
    if a.reference_manifest:
        raise ValueError(
            'Use --training-manifest for this new experiment; '
            'do not pass --reference-manifest.'
        )
    if a.inventory_only:
        print('Inventory finished; no training started.'); return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; activate your GPU environment.')
    if (a.output/'latest.pt').exists() and not a.resume:
        raise RuntimeError('Output has checkpoint. Use --resume or another --output.')
    if a.resume and a.resume.parent.resolve()!=a.output.resolve():
        raise ValueError('Resume into the same checkpoint output folder.')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['sha256']!=signature:
        raise ValueError('Dataset selection changed; use another output.')
    manifest.write_text(json.dumps({'sha256':signature,'records':all_records},indent=2))
    config={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(a).items()}
    config.update(
        sampling='uniform_without_replacement',
        cudnn_enabled=False,
        recipe_version='all5_scratch_v1',
    )
    for record in records[::max(1,len(records)//8)][:8]:load_intensity(record)
    try:
        import lejepa
        regularizer=lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(n_points=17),
            num_slices=a.num_slices).cuda()
    except (ImportError,AttributeError,TypeError) as e:
        raise RuntimeError('Install the official lejepa package in this environment; see provided commands.') from e
    config['lejepa_version']=importlib.metadata.version('lejepa')
    config['torch_version']=torch.__version__
    print('GPU:',torch.cuda.get_device_name(0),'LeJEPA:',config['lejepa_version'],flush=True)
    counts=Counter(r['source'] for r in records)
    rng=torch.Generator().manual_seed(a.seed)
    weights=torch.tensor([1./counts[r['source']] for r in records],dtype=torch.double)
    sampler=WeightedRandomSampler(torch.ones_like(weights),len(records),replacement=False,generator=rng)
    loader=DataLoader(RadarDataset(records,a.local_crops),batch_size=a.batch_size,
        sampler=sampler,num_workers=a.workers,drop_last=False,pin_memory=True,
        worker_init_fn=worker_seed)
    backbone=torch.hub.load('/home/remoteuser/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local',
        pretrained=a.init=='dinov2' and not a.resume,trust_repo=True)
    model=Encoder(backbone,a.proj_dim).cuda()
    opt=torch.optim.AdamW([
        {'params':model.backbone.parameters(),'lr':a.encoder_lr},
        {'params':model.projector.parameters(),'lr':a.projector_lr}],weight_decay=a.weight_decay)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    epoch0=step=0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)  # own trusted checkpoint only
        if ck['smoke_test']:raise ValueError('Smoke checkpoint cannot be resumed as a full run.')
        if ck['manifest_sha256']!=signature:raise ValueError('Resume manifest differs.')
        for key in ('recipe_version', 'sampling', 'cudnn_enabled'):
            if ck['config'].get(key) != config[key]:
                raise ValueError('Resume configuration differs: ' + key)
        for k in ['init','epochs','batch_size','local_crops','encoder_lr','projector_lr','weight_decay',
                  'warmup_epochs','proj_dim','num_slices','sigreg_weight','seed','lejepa_version','torch_version']:
            if ck['config'][k]!=config[k]:raise ValueError('Resume configuration changed: '+k)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        regularizer.load_state_dict(ck['regularizer']);scaler.load_state_dict(ck['scaler'])
        epoch0,step=ck['next_epoch'],ck['step']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        rng.set_state(ck['sampler_rng'])
    else:(a.output/'config.json').write_text(json.dumps(config,indent=2))
    total=a.epochs*len(loader);warm=min(a.warmup_epochs*len(loader),max(1,total//2))
    log=a.output/'losses.csv'
    if not log.exists():log.write_text('epoch,step,loss,invariance,sigreg,projection_std,encoder_lr,projector_lr\n')
    for epoch in range(epoch0,a.epochs):
        model.train()
        bar=tqdm(loader,desc=f'LeJEPA {epoch+1}/{a.epochs}')
        for views in bar:
            factor=(step+1)/warm if step<warm else .001+.999*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))/2
            for group,base in zip(opt.param_groups,[a.encoder_lr,a.projector_lr]):group['lr']=factor*base
            views=[v.cuda(non_blocking=True) for v in views]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype):
                z=model.forward_views(views)
            with torch.autocast('cuda',enabled=False):
                loss,inv,reg=objective(z,regularizer,a.sigreg_weight)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at step {step}.')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f'Nonfinite gradients at step {step}.')
            scaler.step(opt);scaler.update();step+=1
            std=z.float().std(dim=1,unbiased=False).mean().item()
            with log.open('a',newline='') as f:csv.writer(f).writerow([epoch+1,step,loss.item(),inv.item(),reg.item(),std,*[g['lr'] for g in opt.param_groups]])
            bar.set_postfix(loss=f'{loss.item():.4f}',inv=f'{inv.item():.4f}',sigreg=f'{reg.item():.4f}')
            if a.max_steps and step>=a.max_steps:break
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'regularizer':regularizer.state_dict(),
            'scaler':scaler.state_dict(),'next_epoch':epoch+1,'step':step,'config':config,
            'manifest_sha256':signature,'smoke_test':bool(a.max_steps),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':rng.get_state()}
        atomic_save(ck,a.output/'latest.pt')
        atomic_save({'architecture':'dinov2_vits14','state_dict':model.backbone.state_dict(),
            'objective':'LeJEPA radar adaptation','init':a.init,'epoch':epoch+1,'step':step,
            'smoke_test':bool(a.max_steps)},a.output/'encoder.pt')
        print('Saved:',a.output,flush=True)
        if a.max_steps and step>=a.max_steps:break
    print('Finished. Smoke test only.' if a.max_steps else 'Finished. Evaluate encoder.pt with your segmentation decoder.')

if __name__=='__main__':main()

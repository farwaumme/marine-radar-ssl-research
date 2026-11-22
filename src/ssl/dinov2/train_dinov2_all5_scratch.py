#!/usr/bin/env python3
"""Single-GPU DINOv2-style radar SSL (not Meta's exact large-scale recipe).

Official ViT-S/14 backbone; image DINO, masked-patch iBOT, KoLeo, EMA teacher.
Smaller heads, random patch masks, two local crops and centered teacher targets.
Raw inputs: CANOE/Boreas */radar/*.png ONLY (first 11 columns removed).
Prepared inputs: ROOT/recording_name/*.png, with metadata already removed.
Mixed polar/Cartesian inputs use separate crop geometry.
HAXR first/last frames per chunk excluded; interior frames still require QC.
Default: continue pretrained backbone; --init scratch: random backbone.
Dataset/date groups held out; balanced sampling across training sources.
No supervised masks, segmentation metrics or automatic best-encoder selection.
"""
import os
os.environ['XFORMERS_DISABLED'] = '1'
import argparse
import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from all5_inventory import all5_inventory
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-roots', nargs='*', type=Path, default=[])
    p.add_argument('--prepared-roots', nargs='*', type=Path, default=[])
    p.add_argument('--cartesian-roots', nargs='*', type=Path, default=[])
    p.add_argument('--prepared-stride', type=int, default=1)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init', choices=['pretrained', 'scratch'], default='pretrained')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--frame-stride', type=int, default=16)
    p.add_argument('--local-crops', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--head-dim', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-steps', type=int, default=0, help='Smoke test only; use a separate output folder.')
    p.add_argument('--cpu', action='store_true', help='Testing only; training should use CUDA.')
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--professor-root', type=Path, default=Path(
        '/home/remoteuser/projects/myproject/Prof-dataset'))
    p.add_argument('--mae-manifest', type=Path)
    a = p.parse_args()
    if not a.training_manifest:
        p.error('Specify --raw-roots CANOE/Boreas or --prepared-roots converted datasets.')
    if min(a.batch_size, a.epochs, a.frame_stride, a.prepared_stride, a.head_dim) < 1 or a.batch_size < 2:
        p.error('Positive settings required; batch-size must be at least 2 for KoLeo.')
    if a.workers < 0 or a.local_crops < 0:
        p.error('workers/local-crops must be non-negative.')
    return a


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


class Head(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(),
                                 nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 256))
        self.last = nn.Linear(256, out, bias=False)
    def forward(self, x):
        x = F.normalize(self.mlp(x).float(), dim=-1, eps=1e-6)
        return F.linear(x, F.normalize(self.last.weight.float(), dim=1, eps=1e-6))


class SSLModel(nn.Module):
    def __init__(self, backbone, out):
        super().__init__()
        self.backbone = backbone
        self.cls_head = Head(backbone.embed_dim, out)
        self.patch_head = Head(backbone.embed_dim, out)
    def forward(self, x, mask=None):
        result = self.backbone(x, masks=mask, is_training=True)
        return result['x_norm_clstoken'], result['x_norm_patchtokens']


def cross_entropy(student, teacher):
    return -(teacher * F.log_softmax(student.float()/.1, dim=-1)).sum(-1).mean()


def koleo(features):
    x = F.normalize(features.float(), dim=-1, eps=1e-6)
    distances = torch.cdist(x, x, p=2)
    distances = distances.masked_fill(torch.eye(len(x), device=x.device, dtype=torch.bool), float('inf'))
    return -distances.min(dim=1).values.clamp_min(1e-6).log().mean()


def random_mask(batch, device):
    # ViT-S/14: 224/14=16 -> 256 patch tokens; same crop for teacher/student.
    scores = torch.rand(batch, 256, device=device)
    ratios = torch.empty(batch, 1, device=device).uniform_(.1, .5)
    return scores < ratios


@torch.no_grad()
def ema_update(student, teacher, momentum):
    for s, t in zip(student.parameters(), teacher.parameters()):
        t.mul_(momentum).add_(s, alpha=1-momentum)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    a = arguments()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.inventory_only:
        a.output.mkdir(parents=True, exist_ok=True)
        all_records, _ = inventory(a)
        (a.output / 'inventory_preview.json').write_text(json.dumps(all_records, indent=2))
        print('Inventory only: training has not started.')
        return
    if not a.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate your working GPU PyTorch environment.')
    device = torch.device('cpu' if a.cpu else 'cuda')
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__, flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'latest.pt').exists() and not a.resume:
        raise RuntimeError('Output already has a checkpoint. Use --resume or a new --output.')
    all_records, records = inventory(a)
    signature = hashlib.sha256(json.dumps(all_records, sort_keys=True).encode()).hexdigest()
    if (a.output / 'manifest.json').exists():
        old = json.loads((a.output / 'manifest.json').read_text())
        if old['sha256'] != signature:
            raise RuntimeError('Dataset/split changed. Use a new output folder.')
    else:
        (a.output / 'manifest.json').write_text(json.dumps({'sha256': signature, 'records': all_records}, indent=2))
    config = {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
              for k, v in vars(a).items()}
    config.update(
        sampling='uniform_without_replacement',
        cudnn_enabled=False,
        recipe_version='all5_scratch_v1',
    )
    if not a.resume:
        (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    # Inspect a few files before the first training batch.
    for record in records[::max(1, len(records)//8)][:8]:
        load_intensity(record)
    counts = Counter(r['source'] for r in records)
    weights = torch.tensor([1./counts[r['source']] for r in records], dtype=torch.double)
    sampler_rng = torch.Generator().manual_seed(a.seed)
    sampler = WeightedRandomSampler(torch.ones_like(weights), len(records), replacement=False, generator=sampler_rng)
    dataset = RadarDataset(records, a.local_crops)
    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler,
                        num_workers=a.workers, pin_memory=device.type == 'cuda', drop_last=False,
                        worker_init_fn=worker_seed, persistent_workers=False)
    print('Loading official DINOv2 ViT-S/14 backbone...', flush=True)
    backbone = torch.hub.load(str(Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'), 'dinov2_vits14', source='local',
                              pretrained=(a.init == 'pretrained' and not a.resume), trust_repo=True)
    student = SSLModel(backbone, a.head_dim).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=.04)
    center_cls = torch.zeros(1, a.head_dim, device=device)
    center_patch = torch.zeros(1, a.head_dim, device=device)
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype == torch.float16)
    start_epoch, step = 0, 0
    if a.resume:
        # Only load your own checkpoints: torch.load uses pickle here.
        ckpt = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ckpt.get('smoke_test', False) or ckpt['config'].get('max_steps', 0):
            raise ValueError('Smoke checkpoints cannot resume a full run')
        for key in ('recipe_version', 'sampling', 'cudnn_enabled', 'seed'):
            if ckpt['config'].get(key) != config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        for key in ('epochs', 'batch_size', 'frame_stride', 'prepared_stride', 'local_crops', 'head_dim', 'lr', 'warmup_epochs', 'init'):
            if ckpt['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if ckpt['manifest_sha256'] != signature:
            raise ValueError('Checkpoint dataset manifest differs.')
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        center_cls.copy_(ckpt['center_cls'].to(device))
        center_patch.copy_(ckpt['center_patch'].to(device))
        start_epoch, step = ckpt['next_epoch'], ckpt['step']
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        sampler_rng.set_state(ckpt['sampler_rng'])
        if device.type == 'cuda' and ckpt['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    total_steps = a.epochs * len(loader)
    warmup_steps = min(a.warmup_epochs * len(loader), max(1, total_steps//2))
    log_path = a.output / 'losses.csv'
    if not log_path.exists():
        log_path.write_text('epoch,step,loss,dino,ibot,koleo,lr,teacher_momentum\n')
    for epoch in range(start_epoch, a.epochs):
        student.train()
        bar = tqdm(loader, desc=f'epoch {epoch+1}/{a.epochs}')
        for views in bar:
            progress = min(1., step/max(1, total_steps-1))
            momentum = 1-(1-.996)*(math.cos(math.pi*progress)+1)/2
            if step < warmup_steps:
                lr = a.lr * (step+1)/warmup_steps
            else:
                ratio = (step-warmup_steps)/max(1, total_steps-warmup_steps)
                lr = a.lr*(.01+.99*(1+math.cos(math.pi*ratio))/2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            views = [x.to(device, non_blocking=True) for x in views]
            temp = .04 + .03 * min(1., epoch/10.)
            masks = [random_mask(len(views[0]), device) for _ in range(2)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                with torch.no_grad():
                    tc, tp = [], []
                    for x in views[:2]:
                        cls, patches = teacher(x)
                        tc.append(teacher.cls_head(cls))
                        tp.append(teacher.patch_head(patches))
                    qc = [F.softmax((x-center_cls)/temp, dim=-1).detach() for x in tc]
                    qp = [F.softmax((x-center_patch)/temp, dim=-1).detach() for x in tp]
                sc, sp, sf = [], [], []
                for i, x in enumerate(views):
                    cls, patches = student(x, masks[i] if i < 2 else None)
                    sf.append(cls)
                    sc.append(student.cls_head(cls))
                    if i < 2:
                        sp.append(student.patch_head(patches))
                dino_terms = [cross_entropy(sc[j], qc[i]) for i in range(2)
                              for j in range(len(sc)) if i != j]
                dino = torch.stack(dino_terms).mean()
                ibot_terms = []
                for i in range(2):
                    per_patch = -(qp[i]*F.log_softmax(sp[i].float()/.1, dim=-1)).sum(-1)
                    # Average within each image, then across images.
                    ibot_terms.append(((per_patch*masks[i]).sum(1)/masks[i].sum(1).clamp_min(1)).mean())
                ibot = torch.stack(ibot_terms).mean()
                kl = torch.stack([koleo(x) for x in sf[:2]]).mean()
                loss = dino + ibot + .1*kl
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at step {step}; no checkpoint overwritten.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Freeze prototype last layers for the first epoch.
            if epoch == 0:
                student.cls_head.last.weight.grad = None
                student.patch_head.last.weight.grad = None
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 3.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Non-finite gradient at step {step}.')
            scaler.step(optimizer)
            scaler.update()
            ema_update(student, teacher, momentum)
            with torch.no_grad():
                center_cls.mul_(.9).add_(torch.cat(tc).mean(0, keepdim=True), alpha=.1)
                center_patch.mul_(.9).add_(torch.cat(tp).mean((0,1), keepdim=False).unsqueeze(0), alpha=.1)
            step += 1
            bar.set_postfix(loss=f'{loss.item():.4f}', dino=f'{dino.item():.3f}', ibot=f'{ibot.item():.3f}')
            with log_path.open('a', newline='') as handle:
                csv.writer(handle).writerow([epoch+1, step, loss.item(), dino.item(), ibot.item(), kl.item(), lr, momentum])
            if a.max_steps and step >= a.max_steps:
                break
        checkpoint = {'student': student.state_dict(), 'teacher': teacher.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'center_cls': center_cls.cpu(), 'center_patch': center_patch.cpu(),
                      'next_epoch': epoch+1, 'step': step, 'config': config, 'smoke_test': bool(a.max_steps),
                      'manifest_sha256': signature, 'python_rng': random.getstate(),
                      'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                      'sampler_rng': sampler_rng.get_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        atomic_save(checkpoint, a.output/'latest.pt')
        atomic_save({'architecture': 'dinov2_vits14', 'state_dict': teacher.backbone.state_dict(),
                     'init': a.init, 'epoch': epoch+1, 'step': step,
                     'recipe': 'single-GPU DINOv2-style radar SSL'}, a.output/'teacher_encoder.pt')
        print(f'Saved epoch {epoch+1}: {a.output}', flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    print('Finished. Teacher encoder is for downstream evaluation; SSL loss is not segmentation accuracy.')


if __name__ == '__main__':
    main()

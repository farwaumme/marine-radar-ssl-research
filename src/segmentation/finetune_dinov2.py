#!/usr/bin/env python3
"""Radar-adapted DINOv2-S/14 segmentation fine-tuning. Pretrained transfer fine-tuning, not radar SSL pretraining.
Install: pip install numpy pillow pandas tqdm matplotlib
Install CUDA-enabled torch/torchvision using pytorch.org if absent.
Only load your own trusted .pt checkpoints. The official DINOv2 torch.hub repository supplies architecture code.
Folder splits are preserved; recording-level leakage cannot be established from filenames alone.
"""
from pathlib import Path
import os
os.environ["XFORMERS_DISABLED"]="1"
import csv, json, random, time, gc, hashlib, shutil, sys, zipfile, importlib, platform
from collections import Counter, defaultdict
from contextlib import nullcontext
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import argparse
# Defaults are overwritten by main; importing this module does not start training.
DEVICE='cpu'; USE_AMP=False; AMP_DTYPE=torch.float16
SEED=42; NUM_CLASSES=4
PALETTE_RGB=[[0,0,0],[0,255,0],[0,0,255],[255,0,0]]
GRAY_VALUES=[34,51,68,85,102,119,136,153,170,187,204,221,238]
PALETTE=[(r<<16)|(g<<8)|b for r,g,b in PALETTE_RGB]
NEUTRAL_ARTIFACTS={(v<<16)|(v<<8)|v for v in GRAY_VALUES}
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
SPLITS=('train','val','test')
LEJEPA_REPO='OK-AI/lejepa-vits16-pretrain-in1k'
LEJEPA_REV='cc7022877d51494709ef398d437fb8619349e0f9'
TIMM_NAMES={'dinov2_s14':'vit_small_patch14_dinov2.lvd142m','mae_b16':'vit_base_patch16_224.mae'}
def seed_everything(seed=None):
    seed=SEED if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def autocast_context():
    return torch.autocast('cuda',dtype=AMP_DTYPE) if USE_AMP else nullcontext()

def discover_pairs(root):
    exts={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    result=[]
    for folder in sorted(root.rglob('*')):
        if not folder.is_dir() or folder.name.lower() not in SPLITS: continue
        split=folder.name.lower()
        annotation=folder.parent/(split+'annot')
        if not annotation.is_dir(): raise FileNotFoundError(f'Missing mask folder: {annotation}')
        branch=str(folder.parent.relative_to(root))
        parts=[p.upper() for p in folder.relative_to(root).parts]
        if 'POLAR' in parts: rep='POLAR'
        elif 'CARTESIAN' in parts: rep='CARTESIAN'
        else: raise ValueError(f'Cannot determine representation: {folder}')
        masks={}
        for p in sorted(annotation.iterdir()):
            if p.suffix.lower() not in exts: continue
            key=p.stem[3:] if p.stem.startswith('gt_') else p.stem
            if key in masks: raise ValueError(f'Duplicate mask key {key} in {annotation}')
            masks[key]=p
        seen=set()
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in exts: continue
            key=p.stem
            if key in seen: raise ValueError(f'Duplicate image stem {key} in {folder}')
            seen.add(key)
            if key not in masks: raise ValueError(f'No matching mask for {p}; inspect filename conventions.')
            result.append(dict(branch=branch,split=split,rep=rep,key=key,
                               image=str(p),mask=str(masks[key])))
        if seen != set(masks): raise ValueError(f'Unpaired annotations in {annotation}')
    return result

def pack(rgb):
    a=np.asarray(rgb,dtype=np.uint32)
    return (a[...,0]<<16)|(a[...,1]<<8)|a[...,2]

def mask_to_ids(rgb):
    packed=pack(rgb); ids=np.full(packed.shape,255,dtype=np.uint8)
    for i,color in enumerate(PALETTE): ids[packed==color]=i
    for color in NEUTRAL_ARTIFACTS: ids[packed==color]=0
    if np.any(ids==255):
        unknown=np.unique(packed[ids==255])[:10]
        colors=[[int(k>>16),int((k>>8)&255),int(k&255)] for k in unknown]
        raise ValueError('Unexpected mask colors: '+str(colors))
    return ids

def cache_filename(r):
    return hashlib.sha256((r['image']+'|'+r['mask']).encode()).hexdigest()+'.npz'

def cache_valid():
    return (LOCAL_CACHE/'complete.json').is_file() and all((LOCAL_CACHE/cache_filename(r)).is_file() for r in selected_rows)

class RadarDataset(Dataset):
    def __init__(self,items,train=False):self.items=items;self.train=train
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        r=self.items[i]
        with np.load(LOCAL_CACHE/cache_filename(r),allow_pickle=False) as d:
            x=d['x'].copy();y=d['y'].copy()
        x=torch.from_numpy(x).permute(2,0,1).float()/255
        y=torch.from_numpy(y).long()
        if self.train and random.random()<0.5:x=(x*random.uniform(.85,1.15)).clamp(0,1)
        x=(x-torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None]
        return x,y,r['rep'],r['branch']

def make_loader(split,train=False,items=None):
    items=items if items is not None else [r for r in selected_rows if r['split']==split]
    if not items:raise ValueError(f'No examples for {split}')
    return DataLoader(RadarDataset(items,train),batch_size=BATCH_SIZE,shuffle=train,
                      num_workers=NUM_WORKERS,pin_memory=DEVICE=='cuda',drop_last=False)

class SSLEncoder(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__()
        if name!='dinov2_s14':raise ValueError('This script fine-tunes DINOv2-S/14 only')
        self.backbone=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=False)
        self.channels=self.backbone.embed_dim
        if pretrained:
            ckpt=torch.load(ENCODER_CHECKPOINT,map_location='cpu',weights_only=True)
            if ckpt.get('architecture')!='dinov2_vits14':raise ValueError('Expected dinov2_vits14 encoder checkpoint')
            self.backbone.load_state_dict(ckpt['state_dict'],strict=True)
            print('Loaded radar SSL encoder:',ENCODER_CHECKPOINT,'epoch:',ckpt.get('epoch'),flush=True)
    def forward(self,x):
        return list(self.backbone.get_intermediate_layers(x,n=[2,5,8,11],reshape=True,norm=True))

def conv_block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU())

class ViTUNetDecoder(nn.Module):
    def __init__(self,channels,k=NUM_CLASSES):
        super().__init__()
        self.projections=nn.ModuleList([nn.Conv2d(channels,64,1) for _ in range(4)])
        self.bottom=conv_block(64,128)
        self.fuse=nn.ModuleList([conv_block(128+64,128) for _ in range(3)])
        self.refine=conv_block(128,64)
        self.head=nn.Conv2d(64,k,1)
    def forward(self,features,size):
        h,w=size
        pyramid=[F.interpolate(p(f),size=(max(1,h//s),max(1,w//s)),
                    mode='bilinear',align_corners=False)
                 for p,f,s in zip(self.projections,features,(4,8,16,32))]
        z=self.bottom(pyramid[-1])
        for block,skip in zip(self.fuse,reversed(pyramid[:-1])):
            z=F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            z=block(torch.cat([z,skip],dim=1))
        z=F.interpolate(z,size=(h//2,w//2),mode='bilinear',align_corners=False)
        z=self.refine(z)
        return F.interpolate(self.head(z),size=(h,w),mode='bilinear',align_corners=False)

class RadarSSLUNet(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.encoder=SSLEncoder(name,pretrained)
        self.decoder=ViTUNetDecoder(self.encoder.channels);self.encoder_frozen=False
    def set_encoder_frozen(self,frozen):
        self.encoder_frozen=frozen
        for p in self.encoder.parameters():p.requires_grad_(not frozen)
        if frozen:self.encoder.eval()
    def train(self,mode=True):
        super().train(mode)
        if self.encoder_frozen:self.encoder.eval()
        return self
    def forward(self,x):
        if self.encoder_frozen:
            with torch.no_grad():features=self.encoder(x)
        else:features=self.encoder(x)
        return self.decoder(features,x.shape[-2:])

def seg_loss(logits,y):
    logits=logits.float()
    ce=F.cross_entropy(logits,y)
    probs=logits.softmax(1)
    target=F.one_hot(y,NUM_CLASSES).permute(0,3,1,2).float()
    axes=(0,2,3)
    dice=(2*(probs*target).sum(axes)+1)/((probs+target).sum(axes)+1)
    return ce+1-dice[1:].mean()

def add_cm(cm,pred,y):
    # Bound the histogram on CPU before allocating bins.
    target=y.detach().to('cpu',dtype=torch.int64).reshape(-1).clone()
    prediction=pred.detach().to('cpu',dtype=torch.int64).reshape(-1).clone()
    if target.numel()!=prediction.numel():raise ValueError('Prediction/target shape mismatch')
    if ((target<0)|(target>=NUM_CLASSES)|(prediction<0)|(prediction>=NUM_CLASSES)).any():
        raise ValueError('Mask or prediction contains invalid class IDs; expected 0..3')
    encoded=target*NUM_CLASSES+prediction
    cm+=torch.bincount(encoded,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)

def segmentation_metrics(cm):
    tp=cm.diag().double();support=cm.sum(1).double();predicted=cm.sum(0).double()
    union=support+predicted-tp
    dice=2*tp/(support+predicted).clamp_min(1)
    iou=tp/union.clamp_min(1)
    precision=tp/predicted.clamp_min(1)
    recall=tp/support.clamp_min(1)
    active=support[1:]>0
    def macro(v):return float(v[1:][active].mean()) if active.any() else None
    per_class={str(j):dict(dice=float(dice[j]) if union[j]>0 else None,
               iou=float(iou[j]) if union[j]>0 else None,
               precision=float(precision[j]) if predicted[j]>0 else None,
               recall=float(recall[j]) if support[j]>0 else None,
               pixels=int(support[j]),predicted_pixels=int(predicted[j])) for j in range(NUM_CLASSES)}
    return dict(dice=macro(dice),iou=macro(iou),precision=macro(precision),recall=macro(recall),per_class=per_class)

@torch.inference_mode()
def evaluate(model,loader,groups=False):
    model.eval();cm=torch.zeros(NUM_CLASSES,NUM_CLASSES,dtype=torch.int64)
    group_cm={};loss_sum=0.;n=0
    for x,y,reps,branches in loader:
        x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE,non_blocking=True)
        with autocast_context():logits=model(x)
        loss=seg_loss(logits,y)
        pred=logits.argmax(1);add_cm(cm,pred,y)
        if groups:
            for i,(rep,branch) in enumerate(zip(reps,branches)):
                for key in ('representation:'+rep,'branch:'+branch):
                    if key not in group_cm:group_cm[key]=torch.zeros_like(cm)
                    add_cm(group_cm[key],pred[i],y[i])
        loss_sum+=float(loss)*len(x);n+=len(x)
    if not n:raise ValueError('Empty evaluation loader.')
    return dict(loss=loss_sum/n,**segmentation_metrics(cm)),cm,{k:segmentation_metrics(v) for k,v in group_cm.items()}

@torch.inference_mode()
def forward_latency(model,loader,repeats=20):
    model.eval();x=next(iter(loader))[0].to(DEVICE)
    for _ in range(5):
        with autocast_context():model(x)
    if DEVICE=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(repeats):
        with autocast_context():model(x)
    if DEVICE=='cuda':torch.cuda.synchronize()
    return dict(ms_per_image=1000*(time.perf_counter()-start)/(repeats*len(x)),
                timing_batch_size=len(x),timing_repeats=repeats)

def atomic_torch_save(obj,path):
    temp=Path(str(path)+'.tmp');torch.save(obj,temp);os.replace(temp,path)

def save_json(obj,path):
    temp=Path(str(path)+'.tmp')
    with open(temp,'w') as f:json.dump(obj,f,indent=2,allow_nan=False)
    os.replace(temp,path)

def experiment_config(name):
    return dict(model=name,image_size=list(IMAGE_SIZE),num_classes=NUM_CLASSES,palette=PALETTE_RGB,
       gray=GRAY_VALUES,manifest_hash=MANIFEST_HASH,cache_id=CACHE_ID,seed=SEED,
       batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,epochs=EPOCHS,warmup=WARMUP_EPOCHS,
       schedule='cosine_v1',patience=PATIENCE,encoder_lr=ENCODER_LR,decoder_lr=DECODER_LR,weight_decay=WEIGHT_DECAY,
       mean=MEAN,std=STD,quick=QUICK_CHECK,decoder='vit_unet_adapter_v1',torch_version=torch.__version__,encoder_sha256=ENCODER_SHA256,
       source=str(ENCODER_CHECKPOINT),input_mode='grayscale_repeated_rgb')

def train_one(name):
    seed_everything()
    run=RUN_ROOT/name;run.mkdir(parents=True,exist_ok=True)
    config=experiment_config(name)
    last_path=run/'last.pt';best_path=run/'best.pt';done_path=run/'training_complete.json'
    if done_path.is_file():
        with open(done_path) as f:done=json.load(f)
        if done['config']!=json.loads(json.dumps(config)):
            raise ValueError('Completed run uses different settings. Change EXPERIMENT_TAG.')
        if not best_path.is_file():raise FileNotFoundError(best_path)
        print('Skipping completed:',name);return done
    resume=torch.load(last_path,map_location='cpu',weights_only=False) if last_path.is_file() else None
    if resume and resume['config']!=config:
        raise ValueError('Resume settings differ. Restore original settings or change EXPERIMENT_TAG.')
    model=RadarSSLUNet(name,pretrained=resume is None).to(DEVICE)
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':ENCODER_LR},
                                 {'params':model.decoder.parameters(),'lr':DECODER_LR}],weight_decay=WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=USE_AMP and AMP_DTYPE==torch.float16)
    history=[];best=-1.;wait=0;start=1
    if resume:
        model.load_state_dict(resume['state'],strict=True);optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler']);history=resume['history'];best=resume['best'];wait=resume['wait']
        start=resume['epoch']+1
        random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng'])
        torch.set_rng_state(resume['torch_rng'])
        if DEVICE=='cuda':torch.cuda.set_rng_state_all(resume['cuda_rng'])
        print('Resuming',name,'at epoch',start)
        del resume
    train_loader=make_loader('train',train=True);val_loader=make_loader('val')
    for epoch in range(start,EPOCHS+1):
        if wait>=PATIENCE:break
        frozen=epoch<=WARMUP_EPOCHS
        phase=max(0,epoch-WARMUP_EPOCHS-1)/max(1,EPOCHS-WARMUP_EPOCHS-1)
        factor=1.0 if frozen else .1+.9*(1+np.cos(np.pi*phase))/2
        for group,base in zip(optimizer.param_groups,(ENCODER_LR,DECODER_LR)):group['lr']=base*factor
        model.set_encoder_frozen(frozen);model.train();total=0.;count=0
        progress=tqdm(train_loader,desc=f'{name} {epoch}/{EPOCHS} '+('decoder' if frozen else 'joint'))
        for x,y,_,_ in progress:
            x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():logits=model(x)
            loss=seg_loss(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError(f'{name}: nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(x);count+=len(x)
            progress.set_postfix(loss=total/count)
        val,_,_=evaluate(model,val_loader)
        if val['dice'] is None:raise ValueError('Validation set has no foreground pixels.')
        record=dict(epoch=epoch,encoder_lr=optimizer.param_groups[0]['lr'],decoder_lr=optimizer.param_groups[1]['lr'],encoder_frozen=frozen,train_loss=total/count,
                    **{'val_'+k:v for k,v in val.items() if k!='per_class'})
        history.append(record);print(name,record,flush=True)
        if val['dice']>best+1e-5:
            best=val['dice'];wait=0
            atomic_torch_save(dict(state=model.state_dict(),config=config,epoch=epoch,val=val),best_path)
        else:wait+=1
        # Ensure a decoder-only warm-up never triggers stopping before joint fine-tuning begins.
        if epoch<=WARMUP_EPOCHS:wait=0
        atomic_torch_save(dict(state=model.state_dict(),config=config,epoch=epoch,
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),history=history,best=best,wait=wait,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else []),last_path)
        save_json(history,run/'history.json')
        if wait>=PATIENCE:print('Early stopping:',name)
    if not best_path.is_file():raise RuntimeError('No best checkpoint was saved.')
    done=dict(model=name,best_val_dice=best,checkpoint=str(best_path),config=config)
    save_json(done,done_path)
    del model,optimizer,scaler,train_loader,val_loader;gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return done

def evaluate_saved_models():
    split='val' if QUICK_CHECK else 'test'
    loader=make_loader(split)
    summary=[];detail={}
    for name in MODELS:
        path=RUN_ROOT/name/'best.pt'
        if not path.is_file():raise FileNotFoundError(f'Train {name} first: {path}')
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        if ckpt['config']!=experiment_config(name):raise ValueError('Checkpoint configuration mismatch.')
        model=RadarSSLUNet(name,pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt['state'],strict=True)
        result,cm,groups=evaluate(model,loader,groups=True)
        timing=forward_latency(model,loader)
        record=dict(model=name,split=split,best_epoch=ckpt['epoch'],best_val_dice=ckpt['val']['dice'],
             **{k:v for k,v in result.items() if k!='per_class'},**timing,
             parameters=sum(p.numel() for p in model.parameters()),height=IMAGE_SIZE[0],width=IMAGE_SIZE[1])
        summary.append(record)
        detail[name]=dict(overall=result,groups=groups,confusion_matrix=cm.tolist(),timing=timing)
        save_json(detail[name],RUN_ROOT/name/f'{split}_metrics.json')
        del ckpt,model;gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    df=pd.DataFrame(summary)
    df.to_csv(RUN_ROOT/f'{split}_comparison.csv',index=False)
    save_json(detail,RUN_ROOT/f'{split}_details.json')
    print(df.to_string(index=False));return df,detail

def prepare_data(root,cache_root):
    global rows,selected_rows,MANIFEST_HASH,CACHE_ID,LOCAL_CACHE
    rows=sorted(discover_pairs(root),key=lambda r:(r['split'],r['branch'],r['image']))
    if not rows:raise ValueError('No paired images found. Check --dataset-root.')
    counts=Counter(r['split'] for r in rows)
    if any(counts[s]==0 for s in SPLITS):raise ValueError('Each split must contain paired images.')
    print('Discovered split counts:',dict(counts),flush=True)
    print(pd.DataFrame(rows).groupby(['branch','rep','split']).size().unstack(fill_value=0))
    if counts!=Counter(train=2520,val=420,test=420):
        print('NOTE: counts differ from the original 2520/420/420 inventory; verify the manifest.')
    with open(RUN_ROOT/'paired_manifest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    MANIFEST_HASH=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    selected_rows=rows
    if QUICK_CHECK:
        selected_rows=[]
        for split in ('train','val'):
            items=[r for r in rows if r['split']==split]
            random.Random(SEED).shuffle(items);selected_rows.extend(items[:8])
    stats=[[(Path(r[k]).stat().st_size,Path(r[k]).stat().st_mtime_ns) for k in ('image','mask')] for r in selected_rows]
    config=dict(rows=selected_rows,stats=stats,size=IMAGE_SIZE,palette=PALETTE_RGB,gray=GRAY_VALUES,input_mode='grayscale_repeated_rgb')
    CACHE_ID=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    LOCAL_CACHE=cache_root/CACHE_ID;LOCAL_CACHE.mkdir(parents=True,exist_ok=True)
    for r in tqdm(selected_rows,desc='Preparing persistent resized cache'):
        p=LOCAL_CACHE/cache_filename(r)
        if p.is_file():continue
        with Image.open(r['image']) as im,Image.open(r['mask']) as mask:
            if im.size!=mask.size:raise ValueError('Dimension mismatch: '+r['image'])
            x=np.asarray(im.convert('L').convert('RGB').resize(IMAGE_SIZE[::-1],Image.Resampling.BILINEAR))
            y=mask_to_ids(np.asarray(mask.convert('RGB').resize(IMAGE_SIZE[::-1],Image.Resampling.NEAREST)))
        temp=p.with_suffix('.tmp')
        with open(temp,'wb') as f:np.savez_compressed(f,x=x,y=y)
        os.replace(temp,p)
    save_json(dict(classes=PALETTE_RGB,gray_to_background=GRAY_VALUES,
                   note='Gray mapping is provisional; semantic class names are unknown.'),RUN_ROOT/'palette.json')

def choose_batch(maximum):
    # Probe joint fine-tuning, including gradients and AdamW state, rather than inference alone.
    selected=[]
    for name in MODELS:
        model=None;optimizer=None;x=y=logits=loss=None
        model=RadarSSLUNet(name,pretrained=True).to(DEVICE)
        batch=maximum
        while True:
            try:
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
                x=torch.randn(batch,3,*IMAGE_SIZE,device=DEVICE)
                y=torch.randint(NUM_CLASSES,(batch,*IMAGE_SIZE),device=DEVICE)
                torch.cuda.reset_peak_memory_stats()
                with autocast_context():logits=model(x)
                loss=seg_loss(logits,y);loss.backward();optimizer.step()
                torch.cuda.synchronize()
                print(name,'fits batch',batch,'peak GB',round(torch.cuda.max_memory_allocated()/2**30,2),flush=True)
                selected.append(batch);break
            except torch.cuda.OutOfMemoryError:
                if batch==1:raise RuntimeError('Batch 1 does not fit. Use --image-size 224.')
                batch=max(1,batch//2);print(name,'retrying batch',batch,flush=True)
            finally:
                model.zero_grad(set_to_none=True)
                optimizer=None;x=y=logits=loss=None;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    # Leave a margin for real data, validation and other GPU processes.
    return max(1,int(min(selected)*.75))

def plot_history():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name in MODELS:
        p=RUN_ROOT/name/'history.json'
        if p.exists():
            h=json.loads(p.read_text());ax.plot([r['epoch'] for r in h],[r['val_dice'] for r in h],label=name)
    ax.set(xlabel='Epoch',ylabel='Validation foreground macro Dice');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RUN_ROOT/'validation_dice.png',dpi=160);plt.close(fig)

def main(argv=None):
    global DEVICE,USE_AMP,AMP_DTYPE,IMAGE_SIZE,BATCH_SIZE,NUM_WORKERS,EPOCHS,WARMUP_EPOCHS
    global PATIENCE,ENCODER_LR,DECODER_LR,WEIGHT_DECAY,SEED,MODELS,RUN_ROOT,QUICK_CHECK
    global ENCODER_CHECKPOINT,ENCODER_SHA256
    p=argparse.ArgumentParser(description='Pretrained SSL encoders + shared U-Net-style radar segmentation decoder.')
    base=Path('/home/remoteuser/projects/myproject')
    p.add_argument('--encoder-checkpoint',type=Path,default=Path('/home/remoteuser/projects/dinov2sslpretraining/runs/dinov2_all4_ssl_v1/teacher_encoder.pt'))
    p.add_argument('--freeze-epochs',type=int,default=5)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--dataset-root',type=Path,default=base/'Prof-dataset')
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent/'results_ssl')
    p.add_argument('--cache-root',type=Path,default=Path(__file__).resolve().parent/'cache_ssl')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--image-size',type=int,default=448)
    p.add_argument('--batch-size',default='auto',help='auto or a positive integer; same batch for all models')
    p.add_argument('--max-batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tag',default='radar_dinov2_finetune_v1')
    p.add_argument('--patience',type=int,default=0,help='0 disables early stopping')
    p.add_argument('--models',nargs='+',choices=['dinov2_s14'],default=['dinov2_s14'])
    p.add_argument('--quick-check',action='store_true',help='Two epochs on 8 train/8 validation examples; separate output directory.')
    p.add_argument('--evaluate',action='store_true',help='After training, evaluate best checkpoints on test (validation in quick check).')
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args(argv)
    if a.epochs<1 or a.image_size<112 or a.image_size%112 or a.workers<0 or a.max_batch_size<1 or a.patience<0:
        p.error('Positive epochs/batch limit, nonnegative workers/patience; image size must be a multiple of 112.')
    ENCODER_CHECKPOINT=a.encoder_checkpoint.resolve()
    if not ENCODER_CHECKPOINT.is_file():raise FileNotFoundError(ENCODER_CHECKPOINT)
    ENCODER_SHA256=hashlib.sha256(ENCODER_CHECKPOINT.read_bytes()).hexdigest()
    if a.inventory_only:
        items=discover_pairs(a.dataset_root)
        print('Paired split counts:',dict(Counter(r['split'] for r in items)))
        if not items:raise ValueError('No paired images found')
        print(pd.DataFrame(items).groupby(['branch','rep','split']).size().unstack(fill_value=0))
        return
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE!='cuda':raise RuntimeError('CUDA is unavailable. Check nvidia-smi and install CUDA-enabled PyTorch in this virtual environment.')
    USE_AMP=True;AMP_DTYPE=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    IMAGE_SIZE=(a.image_size,a.image_size);NUM_WORKERS=a.workers;SEED=a.seed;MODELS=a.models
    QUICK_CHECK=a.quick_check;EPOCHS=2 if QUICK_CHECK else a.epochs;WARMUP_EPOCHS=min(a.freeze_epochs,max(0,EPOCHS-1))
    PATIENCE=a.patience or EPOCHS+1;ENCODER_LR=1e-5;DECODER_LR=2e-4;WEIGHT_DECAY=1e-4
    if not a.dataset_root.is_dir():raise FileNotFoundError(a.dataset_root)
    RUN_ROOT=a.output_root/(a.tag+f'_{a.image_size}px_{EPOCHS}ep'+('_quick' if QUICK_CHECK else ''))
    RUN_ROOT.mkdir(parents=True,exist_ok=True);seed_everything()
    print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'encoder:',ENCODER_CHECKPOINT,flush=True)
    batch_path=RUN_ROOT/'batch_selection.json'
    request=dict(size=a.image_size,models=MODELS,requested=a.batch_size,maximum=a.max_batch_size)
    if batch_path.is_file():
        saved=json.loads(batch_path.read_text())
        if saved['request']!=request:raise ValueError('Batch settings changed; use a new --tag for a new experiment.')
        BATCH_SIZE=saved['batch_size']
    else:
        if a.batch_size=='auto':BATCH_SIZE=choose_batch(a.max_batch_size)
        else:
            BATCH_SIZE=int(a.batch_size)
            if BATCH_SIZE<1:raise ValueError('Batch size must be positive.')
        save_json(dict(request=request,batch_size=BATCH_SIZE),batch_path)
    print('Shared batch size:',BATCH_SIZE,'epochs:',EPOCHS,'output:',RUN_ROOT,flush=True)
    prepare_data(a.dataset_root,a.cache_root)
    if not a.evaluate_only:
        for name in MODELS:train_one(name)
        plot_history()
    if a.evaluate or a.evaluate_only:evaluate_saved_models()
    print('Finished. Results:',RUN_ROOT,flush=True)

if __name__=='__main__':
    main()


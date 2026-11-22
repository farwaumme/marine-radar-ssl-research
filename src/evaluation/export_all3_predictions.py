#!/usr/bin/env python3
"""Export held-out test predictions from trusted, locally trained segmentation checkpoints.
Requires finetune_radar_ssl.py alongside this file. Never trains or alters checkpoints.
All exported masks use the saved checkpoint's class mapping, at its evaluation resolution.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import re
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE = Path('/home/remoteuser/projects/dinov2sslpretraining')
RUNS = {
    'mae_b16': 'mae_scratch_all5_finetune_v1_448px_100ep',
    'lejepa_s14': 'lejepa_scratch_all5_finetune_v1_448px_100ep',
    'dinov2_s14': 'dinov2_all5_nocudnn_finetune_v1_448px_100ep',
}

def identifier(row):
    digest = hashlib.sha256((row['image']+'|'+row['mask']).encode()).hexdigest()[:12]
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(row['image']).stem)
    return name + '_' + digest

def read_manifest(run, config):
    path = run / 'paired_manifest.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if digest != config['manifest_hash']:
        raise ValueError(f'Paired manifest differs from saved training checkpoint: {path}')
    tests = [r for r in rows if r['split']=='test']
    if not tests:
        raise ValueError('No test images in saved manifest')
    return tests

def read_pair(row, config, module):
    size = tuple(config['image_size'][::-1])
    with Image.open(row['image']) as im, Image.open(row['mask']) as gt:
        if im.size != gt.size:
            raise ValueError('Image/mask dimension mismatch: '+row['image'])
        image = np.array(im.convert('L').convert('RGB').resize(size, Image.Resampling.BILINEAR))
        rgb = np.array(gt.convert('RGB').resize(size, Image.Resampling.NEAREST))
    return image, module.mask_to_ids(rgb)

class TestPairs(Dataset):
    def __init__(self, rows, config, module):
        self.rows, self.config, self.module = rows, config, module
        self.mean = torch.tensor(config['mean'])[:,None,None]
        self.std = torch.tensor(config['std'])[:,None,None]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        image, mask = read_pair(self.rows[index], self.config, self.module)
        x = torch.from_numpy(image.copy()).permute(2,0,1).float()/255
        return (x-self.mean)/self.std, torch.from_numpy(mask.copy()).long(), index

def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)

def overlay(image, ids, palette):
    result = image.copy()
    active = ids != 0
    result[active] = (.55*image[active] + .45*palette[ids[active]]).astype(np.uint8)
    return result

def panel(images, titles, output):
    w, h = images[0].size
    canvas = Image.new('RGB', (w*len(images), h+36), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(zip(images, titles)):
        canvas.paste(im, (i*w,36))
        draw.text((i*w+10,10), title, fill='black')
    canvas.save(output)

def configure(module, config):
    if config.get('input_mode') != 'grayscale_repeated_rgb':
        raise ValueError('Unsupported saved preprocessing')
    if config['num_classes'] != 4:
        raise ValueError('Expected four-class checkpoint')
    module.IMAGE_SIZE = tuple(config['image_size'])
    module.NUM_CLASSES = config['num_classes']
    module.PALETTE_RGB = config['palette']
    module.PALETTE = [(r<<16)|(g<<8)|b for r,g,b in config['palette']]
    module.NEUTRAL_ARTIFACTS = {(v<<16)|(v<<8)|v for v in config['gray']}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-script', type=Path, default=BASE/'finetune_radar_ssl.py')
    p.add_argument('--results-root', type=Path, default=BASE/'results_ssl')
    p.add_argument('--output', type=Path, default=BASE/'test_predictions_mae_lejepa_dinov2')
    p.add_argument('--panels', type=int, default=20, help='Number of comparison panels; 0 means all test images')
    p.add_argument('--batch-size', type=int, default=2)
    a = p.parse_args()
    if a.batch_size<1 or a.panels<0:
        p.error('Positive batch size and nonnegative panel count required')
    if not a.training_script.is_file():
        raise FileNotFoundError(a.training_script)
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new --output folder; existing outputs will not be overwritten')
    if not torch.cuda.is_available():
        raise RuntimeError('Activate the CUDA environment used for fine-tuning')
    # Same workaround that successfully completed the user\'s fine-tuning runs.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    spec = importlib.util.spec_from_file_location('radar_finetune', a.training_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    a.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(a.output).free < 2*1024**3:
        raise RuntimeError('Free at least 2 GiB for exported images before running')
    reference_rows = reference_config = None
    summary, manifest = {}, []
    for name, run_name in RUNS.items():
        training_script = (
            BASE/'finetune_dinov2_radar.py'
            if name == 'dinov2_s14'
            else a.training_script
        )
        if not training_script.is_file():
            raise FileNotFoundError(training_script)

        spec = importlib.util.spec_from_file_location(
            'radar_export_' + name, training_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Keep the cuDNN workaround active after importing.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

        if not hasattr(module, 'RadarSSLUNet'):
            raise RuntimeError(
                f'{training_script} does not expose RadarSSLUNet. '
                'Its model constructor must be checked before export.'
            )

        run = a.results_root/run_name
        checkpoint_path = run/name/'best.pt'
        # These are the user\'s own trusted checkpoints, which contain metadata.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        if config['model'] != name or config.get('quick'):
            raise ValueError('Expected the full fine-tuning checkpoint: '+str(checkpoint_path))
        rows = read_manifest(run, config)
        if reference_rows is None:
            reference_rows, reference_config = rows, config
        else:
            if rows != reference_rows:
                raise ValueError('Models have different test manifests')
            for key in ('image_size','palette','gray','mean','std','input_mode'):
                if config[key] != reference_config[key]:
                    raise ValueError('Models have different preprocessing: '+key)
        configure(module, config)
        palette = np.array(config['palette'], dtype=np.uint8)
        model = module.RadarSSLUNet(name, pretrained=False)
        model.load_state_dict(checkpoint['state'], strict=True)
        model = model.cuda().eval()
        print(f'{name}: best epoch {checkpoint["epoch"]}; {len(rows)} test images', flush=True)
        loader = DataLoader(TestPairs(rows, config, module), batch_size=a.batch_size,
                            shuffle=False, num_workers=0)
        cm = torch.zeros(4,4,dtype=torch.int64)
        per_image = []
        with torch.inference_mode():
            for x, targets, indices in tqdm(loader, desc='Export '+name):
                with torch.autocast('cuda', dtype=precision):
                    logits = model(x.cuda())
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite model output')
                predictions = logits.argmax(1).cpu()
                module.add_cm(cm, predictions, targets)
                for pred_t, gt_t, index_t in zip(predictions, targets, indices):
                    index = int(index_t)
                    row = rows[index]
                    key = identifier(row)
                    image, gt = read_pair(row, config, module)
                    pred = pred_t.numpy().astype(np.uint8)
                    if not np.array_equal(gt, gt_t.numpy()):
                        raise ValueError('Ground truth changed during export')
                    if name == 'mae_b16':
                        save_png(image, a.output/'inputs'/f'{key}.png')
                        save_png(palette[gt], a.output/'ground_truth_color'/f'{key}.png')
                        save_png(gt, a.output/'ground_truth_ids'/f'{key}.png')
                        manifest.append(dict(id=key, **row))
                    save_png(pred, a.output/name/'prediction_ids'/f'{key}.png')
                    save_png(palette[pred], a.output/name/'prediction_color'/f'{key}.png')
                    save_png(overlay(image,pred,palette), a.output/name/'overlays'/f'{key}.png')
                    save_png((pred!=gt).astype(np.uint8)*255, a.output/name/'errors'/f'{key}.png')
                    one_cm = torch.zeros_like(cm)
                    module.add_cm(one_cm, pred_t, gt_t)
                    values = module.segmentation_metrics(one_cm)
                    per_image.append(dict(id=key,branch=row['branch'],representation=row['rep'],
                        **{k:values[k] for k in ('dice','iou','precision','recall')}))
        summary[name] = dict(checkpoint=str(checkpoint_path), epoch=checkpoint['epoch'],
                             count=len(rows), confusion_matrix=cm.tolist(),
                             **module.segmentation_metrics(cm))
        with (a.output/name/'per_image_metrics.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
            writer.writeheader(); writer.writerows(per_image)
        del model, checkpoint, loader
        gc.collect(); torch.cuda.empty_cache()
    panel_root = a.output/'comparison_panels'
    panel_root.mkdir(exist_ok=True)
    # Evenly spaced selection in saved manifest order, not selection by test accuracy.
    count = len(reference_rows) if a.panels==0 else min(a.panels,len(reference_rows))
    chosen = np.linspace(0,len(reference_rows)-1,count,dtype=int)
    for index in chosen:
        key = identifier(reference_rows[int(index)])
        paths = [a.output/'inputs'/f'{key}.png',
                 a.output/'ground_truth_color'/f'{key}.png',
                 a.output/'mae_b16'/'prediction_color'/f'{key}.png',
                 a.output/'lejepa_s14'/'prediction_color'/f'{key}.png',
                 a.output/'dinov2_s14'/'prediction_color'/f'{key}.png']
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert('RGB').copy())
        panel(images,['Radar input','Ground truth','MAE prediction','LeJEPA prediction','DINOv2 prediction'],panel_root/f'{key}.png')
    with (a.output/'test_images.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (a.output/'README.txt').write_text(
        'Held-out test predictions from best validation-Dice checkpoints. No retraining.\n'
        'Inputs and masks are resized to the saved evaluation resolution, not original resolution.\n'
        'Inputs: grayscale repeated to RGB. Ground truth uses the training class mapping;\n'
        'neutral gray annotation pixels are mapped to class 0, as during fine-tuning.\n'
        'Class IDs: 0 black, 1 green, 2 blue, 3 red. Semantic names were not provided.\n'
        'prediction_ids contains class IDs 0..3 and can look black in an image viewer;\n'
        'use prediction_color for viewing. Error images: white=incorrect, black=correct.\n'
        'test_images.csv links each export to its original image and annotation.\n'
        'Per-image foreground macro scores are for inspection; averaging them does not\n'
        'equal dataset scores, which are computed from the accumulated confusion matrix.\n'
        'Panels were chosen evenly in manifest order, without looking at scores.\n')
    print('Finished. Saved predictions:',a.output)
    print('Comparison panels:',panel_root)
    for name, values in summary.items():
        print(name,'Dice:',values['dice'],'IoU:',values['iou'])

if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Scratch MAE-B/16 radar pretraining using the official MAE model.

External training selections come directly from the existing manifest.
Only Professor dataset /train/ images are added; annotation folders are excluded.
Recording independence must be verified separately before interpreting results.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

OFFICIAL = Path(__file__).resolve().parent / "official_mae"
sys.path.insert(0, str(OFFICIAL))
try:
    import models_mae
except ImportError as exc:
    raise RuntimeError("Download official_mae and install timm first.") from exc


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-manifest", type=Path, required=True)
    p.add_argument("--professor-root", type=Path, default=Path(
        "/home/remoteuser/projects/myproject/Prof-dataset"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Maximum microbatches for a separate smoke test.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--inventory-only", action="store_true")
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.accum_steps, a.image_size) < 1:
        p.error("epochs, batch-size, accum-steps and image-size must be positive")
    if a.image_size % 16:
        p.error("image-size must be divisible by 16")
    if not 0 < a.mask_ratio < 1:
        p.error("mask-ratio must be between 0 and 1")
    if min(a.workers, a.warmup_epochs, a.max_steps) < 0:
        p.error("workers, warmup-epochs and max-steps cannot be negative")
    if not 0 < a.min_lr <= a.lr:
        p.error("Require 0 < min-lr <= lr")
    if a.resume and a.max_steps:
        p.error("Do not resume a smoke test")
    return a


def inventory(a):
    ref = json.loads(a.reference_manifest.read_text())
    if not isinstance(ref.get("records"), list):
        raise ValueError("Reference manifest must contain a records list")

    records = []
    for r in ref["records"]:
        if r["split"] != "train":
            continue
        records.append({
            "path": str(Path(r["path"]).resolve()),
            "source": r["source"],
            "group": r["group"],
            "strip11": bool(r["strip11"]),
            "representation": r["representation"].lower(),
        })

    external_sources = {r["source"] for r in records}
    if len(external_sources) != 4:
        raise ValueError(
            f"Expected four external sources; found {external_sources}")

    root = a.professor_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    professor = []
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir() or folder.name.lower() != "train":
            continue
        parts = [s.upper() for s in folder.relative_to(root).parts]
        rep = "polar" if "POLAR" in parts else "cartesian"
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                professor.append({
                    "path": str(path.resolve()),
                    "source": "Professor",
                    "group": str(folder.parent.relative_to(root)),
                    "strip11": False,
                    "representation": rep,
                })
    if not professor:
        raise ValueError("No professor images found directly in train folders")

    records += professor
    paths = [r["path"] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths in inventory")

    for r in records:
        if not Path(r["path"]).is_file():
            raise FileNotFoundError(r["path"])

    print("AVAILABLE TRAINING IMAGES", flush=True)
    for source, count in sorted(Counter(r["source"] for r in records).items()):
        print(f"  {source}: {count:,}", flush=True)
    print(f"Total: {len(records):,}", flush=True)
    print("Professor validation/test images and masks are excluded.", flush=True)
    return records


class RadarImages(Dataset):
    def __init__(self, records, size):
        self.records, self.size = records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(r["path"]) as im:
            if r["strip11"]:
                if im.mode != "L" or im.width <= 11:
                    raise ValueError(f"Unexpected raw radar format: {r['path']}")
                im = im.crop((11, 0, im.width, im.height))
            im = im.convert("L").copy()

        w, h = im.size
        if r["representation"] == "cartesian":
            side = max(1, int(min(w, h) * random.uniform(0.5, 1.0)))
            cw = ch = side
        else:
            cw = max(1, int(w * random.uniform(0.5, 1.0)))
            ch = max(1, int(h * random.uniform(0.5, 1.0)))

        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        im = im.crop((left, top, left + cw, top + ch))
        im = im.resize((self.size, self.size), Image.Resampling.BILINEAR)
        im = ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.1))

        array = np.array(im, dtype=np.float32, copy=True) / 255.0
        x = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    a = arguments()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    records = inventory(a)
    signature = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()).hexdigest()

    if a.inventory_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if len(records) < a.batch_size:
        raise ValueError("Too few images for one batch")

    a.output.mkdir(parents=True, exist_ok=True)
    latest = a.output / "latest.pt"
    if latest.exists() and not a.resume:
        raise ValueError("Output already has latest.pt; use --resume or a new output")
    if a.resume and not latest.exists():
        raise FileNotFoundError(latest)

    revision = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(a).items()
        if k not in {"resume", "inventory_only"}
    }
    config.update(
        torch_version=str(torch.__version__),
        official_mae_commit=revision,
        architecture="mae_vit_base_patch16",
        initialization="random",
        norm_pix_loss=True,
        sampling="equal expected probability per source; replacement",
    )

    # Equal expected contribution from each of the five dataset sources.
    counts = Counter(r["source"] for r in records)
    generator = torch.Generator().manual_seed(a.seed)
    weights = torch.tensor(
        [1.0 / counts[r["source"]] for r in records], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, len(records), replacement=True, generator=generator)
    loader = DataLoader(
        RadarImages(records, a.image_size),
        batch_size=a.batch_size,
        sampler=sampler,
        num_workers=a.workers,
        drop_last=True,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Official factory initializes randomly; no pretrained checkpoint is loaded.
    model = models_mae.mae_vit_base_patch16(
        img_size=a.image_size, norm_pix_loss=True).cuda()

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith(".bias")
         else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))

    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
             else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    start_epoch = microsteps = updates = 0
    if a.resume:
        # Load only your own trusted training checkpoint.
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["smoke_test"]:
            raise ValueError("Smoke checkpoint cannot be resumed")
        if ck["manifest_sha256"] != signature or ck["config"] != config:
            raise ValueError("Resume inventory or configuration differs")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["next_epoch"]
        microsteps, updates = ck["microsteps"], ck["updates"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        generator.set_state(ck["loader_rng"])
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)
    else:
        (a.output / "config.json").write_text(json.dumps(config, indent=2))
        (a.output / "manifest.json").write_text(json.dumps({
            "sha256": signature, "records": records}, indent=2))

    log = a.output / "losses.csv"
    if not log.exists():
        log.write_text("epoch,microstep,optimizer_updates,loss,lr\n")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Random initialization; MAE-B/16; mask ratio:", a.mask_ratio)
    print("Microbatch:", a.batch_size,
          "nominal effective batch:", a.batch_size * a.accum_steps)
    print("Batches per epoch:", len(loader), flush=True)

    stop = False
    with log.open("a", newline="", buffering=1) as handle:
        writer = csv.writer(handle)
        for epoch in range(start_epoch, a.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            seen = 0
            bar = tqdm(loader, desc=f"MAE {epoch + 1}/{a.epochs}")

            for index, x in enumerate(bar):
                group_start = (index // a.accum_steps) * a.accum_steps
                group_length = min(a.accum_steps, len(loader) - group_start)
                if a.max_steps:
                    remaining = a.max_steps - microsteps
                    group_length = min(
                        group_length,
                        remaining + index - group_start)

                if index % a.accum_steps == 0:
                    progress = epoch + index / len(loader)
                    warm = min(a.warmup_epochs, a.epochs / 2)
                    if warm > 0 and progress < warm:
                        lr = a.lr * max(
                            progress / warm, 1.0 / (warm * len(loader)))
                    else:
                        ratio = (progress - warm) / max(a.epochs - warm, 1)
                        lr = a.min_lr + (a.lr - a.min_lr) * (
                            1 + math.cos(math.pi * ratio)) / 2
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                x = x.cuda()
                with torch.autocast("cuda", dtype=dtype):
                    loss, _, _ = model(x, mask_ratio=a.mask_ratio)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Nonfinite loss at microstep {microsteps}")

                scaler.scale(loss / group_length).backward()
                microsteps += 1
                last = (index + 1 == len(loader))
                stop = bool(a.max_steps and microsteps >= a.max_steps)
                boundary = (index + 1) % a.accum_steps == 0 or last or stop

                if boundary:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm).item():
                        raise RuntimeError("Nonfinite gradient norm")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1

                value = loss.detach().item()
                total_loss += value * len(x)
                seen += len(x)
                writer.writerow([epoch + 1, microsteps, updates, value, lr])
                bar.set_postfix(loss=f"{value:.4f}", lr=f"{lr:.2e}")
                if stop:
                    break

            ck = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "next_epoch": epoch + 1,
                "microsteps": microsteps,
                "updates": updates,
                "config": config,
                "manifest_sha256": signature,
                "smoke_test": bool(a.max_steps),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(),
            }
            save_atomic(ck, latest)

            encoder = {
                k: v.detach().cpu() for k, v in model.state_dict().items()
                if not k.startswith("decoder_") and k != "mask_token"
            }
            save_atomic({
                "architecture": "mae_vit_base_patch16",
                "state_dict": encoder,
                "image_size": a.image_size,
                "patch_size": 16,
                "initialization": "random",
                "objective": "MAE masked reconstruction",
                "epoch": epoch + 1,
                "smoke_test": bool(a.max_steps),
                "manifest_sha256": signature,
            }, a.output / "encoder.pt")

            print(f"Epoch {epoch + 1}: mean loss={total_loss / seen:.6f}",
                  flush=True)
            print("Saved:", a.output, flush=True)
            if stop:
                break

    print("Finished smoke test." if a.max_steps else
          "Finished MAE pretraining. Encoder saved as encoder.pt.", flush=True)


if __name__ == "__main__":
    main()

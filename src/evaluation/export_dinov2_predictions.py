from pathlib import Path
from datetime import datetime
import csv
import importlib.util
import json
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE = Path("/home/remoteuser/projects/dinov2sslpretraining")

RUN = BASE / (
    "results_ssl/"
    "dinov2_all5_nocudnn_finetune_v1_448px_100ep"
)
CHECKPOINT = RUN / "dinov2_s14/best.pt"

SSD = Path("/media/remoteuser/Radar Data")
OUTPUT_BASE = SSD / (
    "MOANA_POHNAG_PROF DATASET_RESULTS/"
    "DINOv2/predictions"
)

BATCH_SIZE = 2


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if not SSD.is_mount():
        raise RuntimeError(f"SSD is not mounted: {SSD}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate your training environment.")

    # Reuse the verified export preprocessing and original DINO architecture.
    helper = load_module(
        "dino_export_helpers",
        BASE / "export_radar_predictions.py",
    )
    training = load_module(
        "dino_original_training",
        BASE / "finetune_dinov2_radar.py",
    )

    # Your own trusted segmentation checkpoint.
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]

    if config["model"] != "dinov2_s14" or config.get("quick"):
        raise ValueError("Expected the full DINOv2 segmentation checkpoint.")
    if config.get("decoder") != "vit_unet_adapter_v1":
        raise ValueError("Unexpected segmentation decoder.")

    # Verifies the complete manifest hash before selecting only test rows.
    rows = helper.read_manifest(RUN, config)

    for row in rows:
        for field in ("image", "mask"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(
                    f"Original test {field} is missing: {row[field]}"
                )

    helper.configure(training, config)

    # Same cuDNN workaround as the earlier successful export.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    model = training.RadarSSLUNet("dinov2_s14", pretrained=False)
    model.load_state_dict(checkpoint["state"], strict=True)
    model = model.cuda().eval()

    epoch = checkpoint["epoch"]
    del checkpoint

    output = OUTPUT_BASE / (
        "old_all5_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    folders = (
        "comparison_panels",
        "prediction_color",
        "prediction_ids",
        "inputs",
        "ground_truth",
        "errors",
    )
    for folder in folders:
        (output / folder).mkdir(parents=True, exist_ok=True)

    palette = np.asarray(config["palette"], dtype=np.uint8)
    loader = DataLoader(
        helper.TestPairs(rows, config, training),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    precision = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    exported = []

    print("Checkpoint:", CHECKPOINT, flush=True)
    print("Saved epoch:", epoch, flush=True)
    print("Test images:", len(rows), flush=True)
    print("Resolution:", config["image_size"], flush=True)
    print("Saving to:", output, flush=True)

    with torch.inference_mode():
        for x, targets, indices in tqdm(loader, desc="DINOv2 test predictions"):
            with torch.autocast("cuda", dtype=precision):
                logits = model(x.cuda())

            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite DINO prediction.")

            predictions = logits.argmax(1).cpu()
            training.add_cm(confusion, predictions, targets)

            for prediction, target, index in zip(
                predictions, targets, indices
            ):
                row = rows[int(index)]
                key = "DINOv2_" + helper.identifier(row)
                filename = key + ".png"

                image, ground_truth = helper.read_pair(
                    row, config, training
                )
                if not np.array_equal(ground_truth, target.numpy()):
                    raise RuntimeError("Ground truth changed during export.")

                ids = prediction.numpy().astype(np.uint8)
                colored = palette[ids]
                gt_colored = palette[ground_truth]

                helper.save_png(image, output / "inputs" / filename)
                helper.save_png(
                    gt_colored, output / "ground_truth" / filename
                )
                helper.save_png(
                    colored, output / "prediction_color" / filename
                )
                helper.save_png(
                    ids, output / "prediction_ids" / filename
                )
                helper.save_png(
                    (ids != ground_truth).astype(np.uint8) * 255,
                    output / "errors" / filename,
                )

                helper.panel(
                    [
                        Image.fromarray(image),
                        Image.fromarray(gt_colored),
                        Image.fromarray(colored),
                    ],
                    ["Radar input", "Ground truth", "DINOv2 prediction"],
                    output / "comparison_panels" / filename,
                )

                exported.append({"export_id": key, **row})

    with (output / "test_images.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    metrics = training.segmentation_metrics(confusion)
    summary = {
        "checkpoint": str(CHECKPOINT),
        "epoch": epoch,
        "test_images": len(rows),
        "config": config,
        "confusion_matrix": confusion.tolist(),
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\nFINISHED", flush=True)
    print("Comparison images:", output / "comparison_panels")
    print("Colored predictions:", output / "prediction_color")
    print("Metrics:", metrics)
    print("No training was performed. Saved weights were not changed.")


if __name__ == "__main__":
    main()

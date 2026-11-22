# Marine Radar Self-Supervised Learning and Segmentation

Research-code organization for radar SSL pretraining, supervised encoder–decoder fine-tuning, and held-out segmentation prediction export. **Research snapshot: static checks passed; end-to-end training not verified.**

## Research workflow

1. Prepare radar inputs and document their provenance. The three-source experiment uses Pohang, professor-provided clean Cartesian radar images, and pseudo-intensity grayscale MOANA images. Other included scripts target a **different five-source inventory**.
2. Pretrain MAE, LeJEPA, or a **DINOv2-style** encoder on unlabeled radar images.
3. Fine-tune an encoder + ViT/U-Net-like segmentation decoder with labeled radar masks.
4. Export held-out test predictions and compare metrics using identical evaluation settings.

**Not included:** raw datasets, model weights, original upstream MAE source, a verified MOANA preprocessing script, or an I-JEPA implementation. This repository is not an official MAE/DINOv2/LeJEPA implementation.

## Repository layout

```text
marine-radar-ssl-research/
├── README.md
├── requirements.txt
├── src/
│   ├── ssl/
│   │   ├── mae/            # MAE training scripts
│   │   ├── lejepa/         # LeJEPA + inventory helper
│   │   └── dinov2/         # DINOv2-style + inventory helper
│   ├── segmentation/      # encoder, decoder and supervised fine-tuning
│   └── evaluation/        # prediction export
├── notebooks/             # original supervised Colab experiments
├── docs/                  # audit, preprocessing, architecture, source hashes
├── tests/                 # static source checks
└── .gitignore
```

## Installation (Ubuntu / Linux)

Requires Python 3.11+ (Python 3.12 was used in some earlier local experiments), an appropriate NVIDIA driver and a PyTorch build matching your CUDA environment. **These dependency versions have not been tested together in this packaged layout.**

```bash
git clone https://github.com/<YOUR-USERNAME>/marine-radar-ssl-research.git
cd marine-radar-ssl-research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install the correct torch and torchvision build for your GPU/CUDA:
# https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m compileall -q src
python src/ssl/dinov2/train_dinov2_all5_scratch.py --help
```

### External upstream dependencies

- MAE scripts require an `official_mae/models_mae.py` module **adjacent to each MAE script**. Obtain the original MAE implementation under its own license and place or link it in `src/ssl/mae/official_mae/`. Check its `timm` compatibility.
- LeJEPA scripts require the upstream SIGReg implementation and compatible DINOv2 backbone code; follow the respective upstream repositories.
- DINOv2-style scripts may download backbone code or weights and require network access or pre-cached resources.
- `requirements.txt` lists common Python packages only; it is **not** a verified lockfile.

## Dataset preparation

See [`docs/PREPROCESSING_PROTOCOL.md`](docs/PREPROCESSING_PROTOCOL.md) for the professor's exact MOANA instructions. In particular, trail/no-trail pixelwise equivalence is a **test to perform**, not an established result.

For the reported three-source pretraining selection: Pohang **11,047**, professor clean Cartesian **1,800**, MOANA grayscale Method C **38,292**, total **51,139**. These are historical experiment counts, **not** the default inventory for every uploaded script. An additional 8,500 converted MOANA frames were not in that selected subset; exclusion reasons need manifest verification.

Store datasets outside Git, e.g. `data/` (ignored), and create a machine-local JSON manifest. Do not commit absolute personal file paths, dataset images, proprietary data, or checkpoints.

## Training commands

The following are **CLI examples, not verified end-to-end reproductions**. Replace placeholders with existing local paths. Use `--help` for every script's full options. Start with inventory checks and a small smoke test in a separate output directory.

### 1. MAE (manifest-based, scratch)

```bash
# Obtain official_mae first (see above).
python src/ssl/mae/train_mae_manifest.py \
  --reference-manifest /PATH/TO/reference_manifest.json \
  --professor-root /PATH/TO/professor_dataset \
  --output runs/mae_scratch --epochs 100 --batch-size 8
```

### 2. LeJEPA (five-source manifest experiment)

```bash
python src/ssl/lejepa/train_lejepa_all5_scratch.py \
  --training-manifest /PATH/TO/training_manifest.json \
  --professor-root /PATH/TO/professor_dataset \
  --output runs/lejepa_scratch --init scratch --epochs 100 --batch-size 8
```

### 3. DINOv2-style (five-source manifest experiment)

```bash
python src/ssl/dinov2/train_dinov2_all5_scratch.py \
  --training-manifest /PATH/TO/training_manifest.json \
  --professor-root /PATH/TO/professor_dataset \
  --output runs/dinov2_scratch --init scratch --epochs 100 --batch-size 8
```

### 4. DINOv2-style (alternate manifest-based implementation)

```bash
python src/ssl/dinov2/pretrain_dinov2_manifest.py \
  --manifest /PATH/TO/radar_allframes_v1.json \
  --output runs/dinov2_manifest --init scratch --epochs 100 --batch-size 8
```

The five-source scripts' `all5_inventory.py` helper requires a specific manifest schema and, in the supplied code, validates an expected **415,244**-path inventory and **2,520** professor training images. Do not use a 51,139-image three-source manifest with them without intentionally changing and revalidating the experiment.

## Supervised fine-tuning (encoder + decoder)

These are supervised transfer-learning experiments, **not** SSL pretraining. They include `SSLEncoder`, `ViTUNetDecoder`, and `RadarSSLUNet` classes. Images and masks must use the expected train/val/test directory structure and class palette.

```bash
python src/segmentation/finetune_dinov2.py \
  --dataset-root /PATH/TO/segmentation_dataset \
  --output-root runs/segmentation_dinov2 --models dinov2_s14

python src/segmentation/finetune_mae_lejepa.py \
  --dataset-root /PATH/TO/segmentation_dataset \
  --output-root runs/segmentation_mae --model mae_b16

python src/segmentation/finetune_mae_lejepa.py \
  --dataset-root /PATH/TO/segmentation_dataset \
  --output-root runs/segmentation_lejepa --model lejepa_s14
```

**Caution:** Inspect encoder initialization and checkpoint loading before claiming these scripts evaluate *your radar-pretrained weights*. Some scripts initialize published pretrained encoders by default; different training modes must not be conflated. Recording-level split independence remains unverified.

## Evaluation / prediction export

```bash
python src/evaluation/export_all3_predictions.py --help
python src/evaluation/export_mae_lejepa_predictions.py --help
```

Some export scripts assume original absolute paths and/or `finetune_radar_ssl.py`; they are preserved for traceability and **require path/configuration adaptation** before use in the new layout. Do not treat their inclusion as proof that export runs successfully.

## Tests and reproducibility

```bash
python -m compileall -q src
python tests/test_source_layout.py
```

These are static checks only. Full tests must verify model forward passes, checkpoint key compatibility, data splitting, mask conversion, deterministic seeds, and prediction export on a small permitted sample. See [`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md).

## Attribution, rights and publication

See [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md). Confirm permission to publish institutional research code and comply with upstream project and dataset licenses before making the GitHub repository public. No datasets or pretrained weights are included.

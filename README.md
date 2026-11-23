# Marine Radar Self-Supervised Learning and Segmentation

Research-code organization for radar SSL pretraining, supervised encoder–decoder fine-tuning, and held-out segmentation prediction export. **Research snapshot: static checks passed; end-to-end training not verified.**

## Research workflow

1. Prepare radar inputs and document their provenance. The three-source experiment uses Pohang, an own private Cartesian radar dataset, and pseudo-intensity grayscale MOANA images. Other included scripts target a **different five-source inventory**.
2. Pretrain MAE, LeJEPA, or a **DINOv2-style** encoder on unlabeled radar images.
3. Fine-tune an encoder + ViT/U-Net-like segmentation decoder with labeled radar masks.
4. Export held-out test predictions and compare metrics using identical evaluation settings.

**Not included:** raw datasets, model weights, original upstream MAE source, a verified MOANA preprocessing script, the newer four-method fine-tuning script, or an I-JEPA implementation. This repository is not an official MAE/DINOv2/LeJEPA implementation.

## Supervised semantic segmentation benchmarks

Six encoder configurations were evaluated within a U-Net segmentation framework on **open marine radar datasets** (prototype testing). These results are distinct from the private-dataset SSL pretraining inventory and from the earlier SSL downstream comparison below.

| Configuration | Main feature-extraction approach |
|---|---|
| Custom CNN + U-Net | Convolution blocks trained from scratch |
| ResNet-18 + U-Net | Lightweight residual encoder |
| ResNet-34 + U-Net | Deeper residual encoder |
| EfficientNet-B0 + U-Net | Efficient convolutional feature extraction |
| ConvNeXt-Tiny + U-Net | Modern convolutional architecture |
| Swin-Tiny + U-Net | Hierarchical shifted-window attention |

### Prototype results — supervised U-Net encoders

| Model / encoder | Val Dice | Test Dice | Test IoU | Precision | Recall | Time (ms/image) |
|---|---:|---:|---:|---:|---:|---:|
| Custom U-Net | 0.8879 | 0.8830 | 0.7967 | 0.8667 | 0.9003 | 19.24 |
| ResNet-34 | 0.9197 | 0.9091 | 0.8372 | 0.9202 | 0.8990 | 9.73 |
| **EfficientNet-B0** | **0.9252** | **0.9234** | **0.8612** | 0.9200 | **0.9270** | 10.54 |
| ResNet-18 | 0.9219 | 0.9142 | 0.8457 | 0.9211 | 0.9075 | **7.24** |
| ConvNeXt-Tiny | 0.9231 | 0.9224 | 0.8598 | 0.9200 | 0.9254 | 17.35 |
| Swin-Tiny | 0.9198 | 0.9165 | 0.8501 | **0.9222** | 0.9109 | 17.66 |

**Observation:** EfficientNet-B0 has the highest reported **test** Dice (0.9234) and test IoU (0.8612); ConvNeXt-Tiny is close (test Dice 0.9224). ResNet-18 has the lowest reported inference time (7.24 ms/image). These are supplied experimental results, not independently reproduced in this packaged repository. Statistical significance, recording-level independence, and cross-dataset generalization remain to be assessed. The evaluation protocol and hardware should be documented before making strong speed comparisons.

### Earlier three-method SSL downstream segmentation results

The following are reported results from an **earlier five-source experiment**; they must not be attributed to a current four-method run or directly treated as an apples-to-apples comparison against the supervised U-Net baselines.

| SSL method + segmentation decoder | Best val Dice (%) | Test Dice (%) | Test IoU (%) | Precision (%) | Recall (%) |
|---|---:|---:|---:|---:|---:|
| **MAE-B/16** | **90.04** | **88.30** | **79.72** | **88.29** | **88.32** |
| DINOv2-style-S/14 | 89.56 | 87.50 | 78.52 | 88.10 | 86.96 |
| LeJEPA-S/14 | 88.14 | 86.33 | 76.84 | 86.64 | 86.03 |

MAE has the highest reported test Dice in this earlier SSL comparison: 0.80 percentage points above DINOv2-style and 1.97 points above LeJEPA. Differences in architectures and evaluation settings mean these numbers do **not** establish SSL superiority over the six supervised baselines.

### New four-method SSL experiment — three-source radar pretraining (51,139 images)

This **separate, newer experiment** pretrained four compact SSL encoders from random initialization on the same selected three-source radar dataset, then transferred each newly trained encoder to a U-Net-style segmentation decoder. **No encoder checkpoints from the earlier five-source experiment were loaded.** The supervised segmentation stage used labeled image–mask pairs from the **Own Private Dataset** only.

#### SSL pretraining dataset

| Source | Training images |
|---|---:|
| Own Private Dataset — clean Cartesian radar images | 1,800 |
| Pohang Canal radar images | 11,047 |
| MOANA converted pseudo-intensity images | 38,292 |
| **Total** | **51,139** |

The MOANA representation was generated using the `C_continuous_HSV` conversion after trail removal. It is **heuristic pseudo-intensity**, not recovered physical radar intensity. The selected dataset used no-trace MOANA images for preprocessing verification rather than counting them as a fourth pretraining source.

#### Supervised segmentation evaluation

The downstream evaluation used **2,400** clean Cartesian image–mask pairs from the Own Private Dataset: **1,800 training**, **300 validation**, and **300 test**. Pohang and MOANA images contributed to SSL pretraining but **not** to supervised fine-tuning. All four methods used the same segmentation dataset, a newly initialized custom U-Net-style decoder, and a consistent fine-tuning protocol. Best epochs were selected using validation foreground macro Dice; test metrics are foreground-class macro averages over classes 1–3, excluding background.

| SSL method | Best epoch | Test Dice (%) | Test IoU (%) |
|---|---:|---:|---:|
| **MAE** | 99 | **77.67** | **65.10** |
| **I-JEPA** | 100 | **76.77** | **64.05** |
| **DINOv2-style** | 100 | **73.79** | **60.62** |
| **LeJEPA** | 100 | **72.10** | **58.59** |

**Observation:** MAE had the highest test Dice and IoU in this four-method run. I-JEPA was second, **0.90 percentage points** behind MAE in test Dice. The figures are reported results for this particular experiment, not independent replications or evidence that one method is universally superior.

**Experiment details:** All four SSL encoders were trained for 100 epochs (batch size 64), with 512 × 512 single-channel radar inputs. Each transferred encoder was fine-tuned for up to 100 epochs with a newly initialized decoder. The segmentation checkpoint was selected using validation Dice and evaluated on the 300-image test split. The image–mask audit found no identical decoded images across splits, but **recording-level independence has not been established**. Multi-seed variability and uncertainty estimates are not yet reported.

**Code availability:** This README reports the newer **I-JEPA** experiment, but the packaged repository **does not currently include the corresponding I-JEPA implementation or the newer four-method fine-tuning script**. The training commands below document the older scripts already present; they should not be interpreted as instructions that reproduce these newer results without additional code and configuration.

**Comparison caveat:** The earlier **five-source, three-method** SSL results above and this **three-source, four-method** experiment are distinct. Different training inventories, encoders, checkpoints, and evaluation protocols prevent a controlled cross-experiment performance claim.

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
git clone https://github.com/farwaumme/marine-radar-ssl-research.git
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

See [`docs/PREPROCESSING_PROTOCOL.md`](docs/PREPROCESSING_PROTOCOL.md) for the MOANA preprocessing protocol. In particular, trail/no-trail pixelwise equivalence is a **test to perform**, not an established result.

For the reported three-source pretraining selection: Pohang **11,047**, own private Cartesian **1,800**, MOANA grayscale Method C **38,292**, total **51,139**. These are historical experiment counts, **not** the default inventory for every uploaded script. An additional 8,500 converted MOANA frames were not in that selected subset; exclusion reasons need manifest verification.

Store datasets outside Git, e.g. `data/` (ignored), and create a machine-local JSON manifest. Do not commit absolute personal file paths, dataset images, proprietary data, or checkpoints.

**CLI compatibility note:** The existing Python scripts use the command-line option `--professor-root`. This option name is retained in the examples because changing it only in the README would break the commands. It points to the **own private dataset** path; renaming the actual option to `--private-root` requires coordinated changes to the Python code and inventory helpers.

## Training commands

The following are **CLI examples, not verified end-to-end reproductions**. Replace placeholders with existing local paths. Use `--help` for every script's full options. Start with inventory checks and a small smoke test in a separate output directory.

### 1. MAE (manifest-based, scratch)

```bash
# Obtain official_mae first (see above).
python src/ssl/mae/train_mae_manifest.py \
  --reference-manifest /PATH/TO/reference_manifest.json \
  --professor-root /PATH/TO/own_private_dataset \
  --output runs/mae_scratch --epochs 100 --batch-size 8
```

### 2. LeJEPA (five-source manifest experiment)

```bash
python src/ssl/lejepa/train_lejepa_all5_scratch.py \
  --training-manifest /PATH/TO/training_manifest.json \
  --professor-root /PATH/TO/own_private_dataset \
  --output runs/lejepa_scratch --init scratch --epochs 100 --batch-size 8
```

### 3. DINOv2-style (five-source manifest experiment)

```bash
python src/ssl/dinov2/train_dinov2_all5_scratch.py \
  --training-manifest /PATH/TO/training_manifest.json \
  --professor-root /PATH/TO/own_private_dataset \
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

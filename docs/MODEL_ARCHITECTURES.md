# Model components and naming

| Family | Pretraining | Segmentation fine-tuning | Export |
|---|---|---|---|
| MAE | `src/ssl/mae/train_mae_manifest.py`, `train_mae_allframes.py` | `src/segmentation/finetune_mae_lejepa.py` | `src/evaluation/export_mae_lejepa_predictions.py` |
| LeJEPA | `src/ssl/lejepa/train_lejepa.py`, `train_lejepa_v2.py`, `train_lejepa_all5_scratch.py` | `src/segmentation/finetune_mae_lejepa.py` | `src/evaluation/export_mae_lejepa_predictions.py` |
| DINOv2-style | `src/ssl/dinov2/train_dinov2_all5_scratch.py`, `pretrain_dinov2_manifest.py` | `src/segmentation/finetune_dinov2.py` | `src/evaluation/export_dinov2_predictions.py` |
| I-JEPA | **not supplied** | **not supplied** | **not supplied** |

The fine-tuning code includes encoder wrapper (`SSLEncoder`), convolutional blocks (`conv_block`), U-Net-like decoder (`ViTUNetDecoder`), and integrated model (`RadarSSLUNet`). These definitions have not been extracted into a shared library because changing module names or checkpoint keys can break reproducibility.

`finetune_multi_encoder.py` is an additional supervised transfer-learning comparison script, **not** an SSL pretraining script.

# Initial source audit (static only)

## What was inspected
- 15 uploaded Python scripts and two Colab notebooks were preserved in named folders.
- Python source was syntax-parsed; this does **not** prove training or inference correctness.
- The supervised scripts define `SSLEncoder`, `ViTUNetDecoder`, and `RadarSSLUNet` in the same files rather than separate encoder/decoder modules.
- MAE scripts import `models_mae` from an adjacent `official_mae/` folder, which is **not included**.
- All-five-source scripts require `all5_inventory.py` in the import path; copies are placed alongside each dependent training script.
- Some evaluation scripts use hard-coded absolute paths and dynamically load the original fine-tuning file; use explicit CLI overrides where available or adapt paths carefully before running.
- Some scripts use distinct training inventories (three-source radar selection versus five-source external datasets). Do not claim identical data or comparable results across them without checking manifests.
- No uploaded I-JEPA training script was supplied; I-JEPA is **not implemented in this archive**.
- No MOANA trail-removal / RGB-to-pseudo-intensity conversion source was supplied here; professor's requirements are documented, but not presented as implemented.
- Supervised evaluation preserves existing train/val/test folders; source comments explicitly warn that recording-level leakage is not certified.

## Release blockers
1. Obtain permission for public release of research scripts and any upstream code.
2. Validate real CLI runs on the intended CUDA machine and pin tested package versions.
3. Supply data layout, reproducible split manifests, and SHA256 checksums with private paths removed.
4. Remove hard-coded machine paths, validate checkpoint loading and export scripts.
5. Confirm segmentation class IDs, masks, encoder initialization, and decoder interface in controlled tests.
6. Add preprocessing implementation, trail/no-trail pixelwise verification, and regression tests.
7. Add I-JEPA only after an actual implementation and tests are provided.

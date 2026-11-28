# Unused starter scaffold

These files were created at project start and were never used by the reported experiments:

- `train.py` – raises `NotImplementedError`; training lives in the notebooks and `src/controlled_field_mixed.py`.
- `models.py` – torchvision factory for ResNet18 / **ViT-B/16** (the study uses timm ViT-Small).
- `config.yaml` – folder-based settings not read by any code.
- `app/` – upload-only Streamlit page without model inference.

They are kept only so that earlier commits referring to them remain understandable.

# Workflow and artifact provenance

## Current workflow

Run commands from the repository root so `src` imports resolve. Preparation scripts are retained for provenance and deliberate future dataset construction. They are **not** prerequisites when the committed manifests already exist. In particular, `src/data_prep.py` writes original splits and label mappings, and the verified preparation script can replace derived splits with backups when explicitly requested. Neither was run during cleanup.

The current verified pair is `src/prepare_field_dev_verified.py` and `src/evaluate_field_dev_verified.py`. The latter validates manifests and loads frozen ResNet18 checkpoints through `load_model`. Baseline model loading also lives in `src/evaluate.py` (ResNet18) and `src/evaluate_vit.py` (timm `vit_small_patch16_224.augreg_in1k`). Field-adapted and mixed-domain loading is implemented in notebooks 06–08. No model factory was substituted or consolidated.

Experiment parameters remain in notebook configuration cells. `configs/config.yaml`, `src/train.py`, and `src/models.py` are starter scaffolds, not an alternative reproduction pipeline. Moving notebook parameters into YAML would change the working code and is outside this cleanup.

## Manifests and audit evidence

All paths below are relative to `data/splits/`:

| Files | Purpose |
| --- | --- |
| `lab_train.csv`, `lab_val.csv`, `lab_test.csv` | Fixed lab partitions, including leaf grouping metadata |
| `field_train.csv`, `field_test.csv` | Original 476/47 PlantDoc partitions; retain unchanged |
| `field_adapt_train.csv`, `field_dev.csv` | Current 370/92 verified development workflow |
| `field_dev_protocol.json` | pHash threshold, seed, candidate count, quarantine counts, overlap assertions |
| `field_near_duplicate_candidates.csv` | 20 candidate pairs across original field train/test |
| `field_label_conflicts.csv` | 10 training rows in conflicting-label groups |
| `field_quarantined_train.csv` | 14 excluded training rows and quarantine reasons |
| `label_map.json` | Shared five-class numeric order |
| `dataset_sources.json` | Source names, requested revisions, seed, lab splitting description |
| `split_backups/` | Earlier random and exact-duplicate-grouped manifests/protocols |

Four quarantined training rows are near original Field Test images and ten have label conflicts, according to the saved current protocol. The audit uses Field Test image content to detect contamination; model development does not use Field Test inference or labels for checkpoint selection. Source revision metadata is incomplete: do not invent resolved SHAs or refresh row-index manifests against a changed upstream dataset.

## Saved result families

Historical output folders stay in place because notebooks and report work may depend on their paths. Folder names distinguish the protocols; they are not redundant copies.

| Output folder under `outputs/` | Provenance |
| --- | --- |
| `resnet18_domain_gap/` | Frozen ResNet18 predictions/reports on Lab Test and original Field Test |
| `vit_small_domain_gap/` | Frozen ViT-Small predictions/reports on those same tests |
| `comparison/` | Notebook 04 aggregate metrics, paired field predictions, plots |
| `field_dev_comparison/` | Historical random split, 96 Dev images |
| `field_dev_comparison_grouped/` | Historical exact-duplicate-grouped split, 95 Dev images |
| `field_dev_comparison_verified/` | Current pHash-grouped split, 92 Dev images |

Each development result family contains its own `evaluation_protocol.json`, including the SHA-256 of the manifest evaluated. The historical `evaluation_set` strings name the then-current `data/splits/field_dev.csv`; they are provenance, not instructions to use today's manifest.

| Result family | Matching Field Dev manifest | SHA-256 |
| --- | --- | --- |
| Random | `split_backups/20261103T072114096478Z/field_dev.csv` | `10f9cb1b8bb48b6062e763de2a7956420edcf6cff85f9a27dcc61a0d4222b435` |
| Exact grouped | `split_backups/20261103T074311335002Z/field_dev.csv` | `3064e691e65879a4251ce728e65a7dbef8ca06e1f413931657a3dd4ffdadf663` |
| Verified | `field_dev.csv` | `2c61790f90214b3a4bbeb5e7f3e6228e6f2c013cab3cfe0b320a28a6aae960e9` |

## Checkpoints and persistent Colab results

Google Drive paths below are relative to `/content/drive/MyDrive/plant-disease-diagnosis/`. They are configured in notebook source; external files were not inspected during cleanup.

| Notebook | Drive folder / checkpoint |
| --- | --- |
| 02 | `resnet18_baseline/resnet18_lab_best.pt` |
| 03 | `vit_small_baseline/vit_small_lab_best.pt` |
| 05 | `resnet18_aug_{photometric,geometric}/resnet18_best.pt` |
| 06 verified | `field_adaptation_verified/<experiment>/resnet18_field_best.pt` |
| 07 | `vit_small_field_adaptation_verified/<experiment>/vit_small_field_best.pt` |
| 08 | `mixed_domain_verified/<experiment>/mixed_domain_best.pt` |

Histories, predictions, summaries, plots, and experiment metadata are written alongside the Drive experiments. Preserve them there. This checkout contains only four local checkpoints in ignored `models/`:

| Local filename | Corresponding Drive checkpoint |
| --- | --- |
| `resnet18_lab_best.pt` | `resnet18_baseline/resnet18_lab_best.pt` |
| `vit_small_lab_best.pt` | `vit_small_baseline/vit_small_lab_best.pt` |
| `resnet18_aug_photometric_best.pt` | `resnet18_aug_photometric/resnet18_best.pt` |
| `resnet18_aug_geometric_best.pt` | `resnet18_aug_geometric/resnet18_best.pt` |

The augmentation filenames differ locally from Drive; do not rename Drive source paths inside notebooks. Adaptation/mixed-domain histories and checkpoints are not present locally, so the checkout alone cannot regenerate all extension analyses.

## Re-evaluation without training

Prefer reading saved CSVs and notebook outputs. When deliberately rerunning inference, retain the existing evidence and select a **new** output folder, for example:

```bash
python -m src.evaluate_field_dev_verified --output-dir outputs/recheck_field_dev_verified
```

This evaluates only current Field Dev, using the three local CNN checkpoints. It downloads source images if absent from the Hugging Face cache; it performs no training. To regenerate the already reported baseline test evaluation (not for model selection), use:

```bash
python -m src.evaluate --checkpoint models/resnet18_lab_best.pt --output outputs/recheck_resnet18_domain_gap
python -m src.evaluate_vit --checkpoint models/vit_small_lab_best.pt --output outputs/recheck_vit_small_domain_gap
```

These two commands access original Field Test; do not use their results to select extensions. None of these inference commands was run during cleanup. The saved analysis in notebook 04 can be recomputed from existing predictions without loading models, but its default output directory is the existing `outputs/comparison/`.

## Historical implementation paths

| Previous location | Archived location |
| --- | --- |
| `notebooks/06_field_adaptation.ipynb` | `notebooks/archive/06_field_adaptation.ipynb` |
| `notebooks/06_field_adaptation_grouped.ipynb` | `notebooks/archive/06_field_adaptation_grouped.ipynb` |
| `src/prepare_field_dev.py` | `src/archive/prepare_field_dev.py` |
| `src/prepare_field_dev_grouped.py` | `src/archive/prepare_field_dev_grouped.py` |
| `src/evaluate_field_dev.py` | `src/archive/evaluate_field_dev.py` |
| `src/evaluate_field_dev_grouped.py` | `src/archive/evaluate_field_dev_grouped.py` |

Archived evaluators are importable as `src.archive.evaluate_field_dev` and `src.archive.evaluate_field_dev_grouped`. Their repository-root discovery was adjusted for the extra directory. No main code imports them. The archived grouped notebook's preparation command points at its archived script; its scientific code and outputs are preserved.

**Do not run historical workflows against current manifests or output directories.** Their defaults still refer to the shared splits and historical output folders to preserve their original semantics. The archived notebooks also share the old `field_adaptation/` Drive folder. For historical reconstruction, use an isolated checkout or copy of the appropriate backup manifests and separate result storage; never replace the current verified files. Older notebooks can contain superseded methodology claims and preparation instructions.

## Dependencies and reproducibility limits

`requirements.txt` now declares the directly imported research packages, including previously missing `torch`, `torchvision`, `scipy`, and `ImageHash`; `ipykernel` supports local VS Code execution. `app/requirements.txt` adds optional Streamlit. Google Colab supplies `google.colab`; it should not be installed into the macOS environment. Notebook setup cells retain their existing dependency installation commands and rely on the Colab runtime for PyTorch.

Versions remain unpinned because historical training versions were not recovered. Installing the requirements is not an exact historical environment reconstruction. Record resolved dataset SHAs, package versions, runtime/hardware, source commit, split hashes, and checkpoint metadata for future experiments. This cleanup does not retroactively change existing experiment provenance.

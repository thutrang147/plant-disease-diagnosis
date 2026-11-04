# Fine-grained Plant Disease Diagnosis

Compare **ResNet18** and **ViT-Small** on five tomato classes, then investigate the generalization gap from controlled PlantVillage images to PlantDoc field images. The assignment covers baseline fine-tuning, lab-to-field evaluation, and an augmentation extension. Completed extensions include duplicate-aware field adaptation and mixed-domain fine-tuning. Mobile quantization remains an optional future extension; the demo application is a placeholder.

## Data and experimental protocol

Images are loaded through Hugging Face: [PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full) and [PlantDoc](https://huggingface.co/datasets/geraldmc/plantdoc-full) (`v0.1.0` for PlantDoc). CSV manifests contain source row indices, not pixels. The label order in `data/splits/label_map.json` is fixed:

| ID | Class |
| --- | --- |
| 0 | Healthy |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Bacterial Spot |
| 4 | Septoria Leaf Spot |

PlantVillage has 5,365 Lab Train, 1,341 Lab Validation, and 1,692 Lab Test images. Validation uses a leaf-group-aware split of original training data. The original PlantDoc training partition has 476 images: the verified protocol allocates 370 to Field Adapt Train, 92 to Field Dev, and quarantines 14. The original **47-image Field Test remains separate**.

The current field split groups pHash connected components at Hamming distance ≤ 4 and quarantines training groups near Field Test or with conflicting labels. Keep all manifests, audit tables, protocol JSON, label mappings, and timestamped split backups in `data/splits/`. **Reuse these files; do not regenerate splits to reproduce the saved analysis.**

## Repository layout

```text
plant-disease-diagnosis/
├── README.md
├── requirements.txt              # Research and local notebook dependencies
├── notebooks/                    # Eight main research notebooks
│   └── archive/                  # Two superseded adaptation notebooks
├── src/                          # Current dataset/preparation/evaluation code
│   └── archive/                  # Historical random/exact-duplicate workflows
├── data/splits/                   # Versioned manifests, audits, and protocols
│   └── split_backups/            # Historical split snapshots
├── models/                       # Local checkpoints, ignored by Git
├── outputs/                      # Saved evaluation evidence; new files ignored
├── configs/config.yaml           # Starter configuration (not notebook settings)
├── app/                          # Future Streamlit demo and optional requirements
└── docs/                         # Workflow, storage, provenance, cleanup record
```

Existing tracked results remain version-controlled. New outputs, checkpoints, raw/processed data, caches, and local environments are ignored. Add new small research evidence intentionally after review; never force-add checkpoints or dataset caches.

## Main notebook workflow

Use this order to understand the study. Training notebooks are records of completed experiments; running them again is optional and performs training.

| Order | Notebook in `notebooks/` | Purpose / environment |
| --- | --- | --- |
| 01 | `01_data_exploration.ipynb` | Dataset exploration; downloads images |
| 02 | `02_train_cnn.ipynb` | Lab-only ResNet18 baseline; Colab GPU |
| 03 | `03_train_vit.ipynb` | Lab-only timm ViT-Small baseline; Colab GPU |
| 04 | `04_domain_gap.ipynb` | Local comparison of saved baseline predictions |
| 05 | `05_augmentation_experiments.ipynb` | Photometric/geometric ResNet18 experiments; Colab GPU |
| 06 | `06_field_adaptation_verified.ipynb` | Verified ResNet18 field adaptation; Colab GPU |
| 07 | `07_vit_field_adaptation.ipynb` | Verified ViT-Small field adaptation; Colab GPU |
| 08 | `08_mixed_domain_finetuning.ipynb` | Mixed lab/field fine-tuning; Colab GPU |

The two earlier variants of notebook 06 live in `notebooks/archive/`. They document different split protocols and are not part of the current workflow.

## Setup and analysis without retraining

From the repository root on macOS, use Python 3.10 or newer with a compatible PyTorch/torchvision pair:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select `.venv` as the VS Code notebook kernel. Open notebook 04 from the root or `notebooks/` and run it against the committed prediction CSVs. It needs no checkpoints, image downloads, GPU, or training, but writes comparison artifacts under `outputs/comparison/`. To preserve saved files exactly, simply view the notebook's existing outputs and the committed CSVs/reports instead.

For a read-only view of saved verified development metrics:

```bash
python -c "import pandas as pd; print(pd.read_csv('outputs/field_dev_comparison_verified/summary.csv').to_string(index=False))"
```

For Colab training, use the notebook's GPU setup, repository synchronization, and Google Drive mount cells. The notebooks use `/content/drive/MyDrive/plant-disease-diagnosis/` for persistent checkpoints and histories. Ensure the cloned revision includes your intended source and manifests. Colab execution has **not** been verified during repository cleanup.

## Current scripts and stored results

| File | Role |
| --- | --- |
| `src/dataset.py` | Manifest-backed Hugging Face image loader; validates labels against source rows |
| `src/data_prep.py` | Original manifest construction; writes splits and mappings, so do not run for saved-analysis reproduction |
| `src/prepare_field_dev_verified.py` | Current pHash grouping, quarantine, audit, and split preparation; reuse existing outputs |
| `src/evaluate_field_dev_verified.py` | Frozen ResNet18 baseline/augmentation evaluation on the verified Field Dev |
| `src/evaluate.py` | Frozen ResNet18 Lab Test / original Field Test evaluation |
| `src/evaluate_vit.py` | Frozen timm ViT-Small Lab Test / original Field Test evaluation |

Use repository-root module commands for evaluation, such as `python -m src.evaluate_field_dev_verified --help`. Model loading is implemented in the evaluators and training notebooks. `src/models.py` is a separate starter factory supporting ResNet18 and torchvision ViT-B/16; it is not the research ViT-Small loader. `src/train.py` deliberately raises `NotImplementedError`; training lives in the notebooks. The YAML config is starter scaffolding and does not control the notebook experiments.

See [workflow and artifact provenance](docs/workflow.md) for checkpoint names, Drive locations, evaluation commands, and historical split/result pairings. See [cleanup record](docs/cleanup.md) for moves, verification, and unresolved gaps.

## Interpretation and limitations

- PlantDoc labels can be ambiguous or inconsistent; watermarks, backgrounds, and non-field images complicate the domain comparison.
- Exact and near-duplicates can inflate metrics. The verified grouping is conservative and its candidate pairs and label conflicts still need visual review; pHash is not proof that every duplicate was detected.
- Field Dev (92) and original Field Test (47) are small and imbalanced. Report accuracy, macro-F1, per-class metrics, and sample counts together.
- Field Dev supports model selection. It is not an independent final test. The original Field Test was inspected in earlier analysis, so confirmatory claims require a new independent field test. Do not tune using it.
- Results from the historical 96/95-image Dev splits cannot be treated as matched comparisons with the current 92-image split.
- Dataset row IDs depend on upstream ordering. PlantVillage is unpinned and PlantDoc uses a tag rather than a resolved commit SHA. Dependencies are not locked; exact historical environments and external Drive artifacts are not fully recoverable from this checkout alone.

The optional app only uploads images; it does not predict diseases. Install `app/requirements.txt` and run `streamlit run app/streamlit_app.py` to inspect the scaffold.

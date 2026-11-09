# Tomato Disease Classification: CNN and Vision Transformer under Domain Shift

## Abstract

This assignment compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato disease classification, with PlantVillage as the laboratory source domain and PlantDoc as the field target domain. Completed experiments cover laboratory baselines, lab-to-field evaluation, photometric and geometric augmentation, supervised field adaptation with duplicate-aware splits, mixed-domain fine-tuning, and exploratory error analysis. Both laboratory baselines lose substantial macro-F1 on PlantDoc. The geometric-initialized mixed-domain ResNet18 achieves the highest Field Dev macro-F1 among the four models in the final comparison (0.4459), while mixed-domain ViT-Small retains higher Lab Test macro-F1 (0.9886 versus 0.9140). Field Dev was used for checkpoint selection; these field results are developmental rather than independent confirmation.

## Research Questions

1. How do the two pretrained model configurations compare on laboratory images and under PlantVillage-to-PlantDoc domain shift?
2. How do photometric and geometric training augmentation affect ResNet18 performance?
3. How do field-only and mixed-domain fine-tuning change Field Dev performance and source-domain retention?
4. Which classes and images remain difficult, and what can Grad-CAM and ViT attention reveal for qualitative review?

## Datasets and Evaluation Protocol

Images are loaded from the Hugging Face mirrors [PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full) and [PlantDoc](https://huggingface.co/datasets/geraldmc/plantdoc-full). PlantDoc uses revision `v0.1.0`; PlantVillage is unpinned. The label order is fixed in [`label_map.json`](data/splits/label_map.json): Healthy (0), Early Blight (1), Late Blight (2), Bacterial Spot (3), and Septoria Leaf Spot (4). CSV manifests store source row indices rather than image pixels.

| Partition | Images | Role |
| --- | ---: | --- |
| Lab Train | 5,365 | PlantVillage training; also sampled during mixed-domain fine-tuning |
| Lab Validation | 1,341 | Laboratory checkpoint selection; source-retention monitoring during adaptation |
| Lab Test | 1,692 | Fixed laboratory evaluation partition |
| Field Adapt Train | 370 | Labeled PlantDoc training for adaptation |
| Field Dev | 92 | Target-domain checkpoint selection and exploratory analysis |
| Quarantined field training rows | 14 | Excluded from adaptation and Field Dev |
| Original Field Test | 47 | Previously inspected baseline evaluation; not a fresh independent final test |

Lab Validation is formed from original PlantVillage training data with leaf-group-aware splitting; images without reliable leaf IDs are treated as singleton groups. The verified PlantDoc protocol groups pHash connected components at Hamming distance ≤ 4 across the original training/test partitions. Of 476 original training rows, four near Field Test and ten in conflicting-label groups are quarantined, leaving 370/92 adaptation/development images. See the saved [protocol](data/splits/field_dev_protocol.json) and [artifact provenance](docs/workflow.md).

Field Dev class counts are 11 Healthy, 16 Early Blight, 20 Late Blight, 18 Bacterial Spot, and 27 Septoria Leaf Spot. Accuracy, macro-F1 over all five classes, and per-class metrics are reported. Field Test image fingerprints are used in contamination audits, but notebooks 06–10 do not perform Field Test model inference. Historical 96- and 95-image development splits are separate protocols and must not be pooled with the current 92-image results.

## Models and Experimental Design

ResNet18 uses torchvision `IMAGENET1K_V1` weights; ViT-Small uses timm `vit_small_patch16_224.augreg_in1k`. Both use 224 × 224 inputs and seed 42. Baselines train the head for three epochs and the full model for five, selecting by Lab Validation macro-F1 without stochastic augmentation. Model-specific transforms, batch sizes, learning rates, pretraining recipes, and parameter counts differ (11,179,077 versus 21,667,589); the comparison does not isolate architecture alone.

Augmentation experiments modify ResNet18 training transforms while retaining its optimizer and epoch schedule. Field-only adaptation starts from the laboratory checkpoints, trains the head for two epochs and the final block for up to six, and selects by Field Dev macro-F1. Mixed-domain experiments start from the same laboratory checkpoints and sample 185 field plus 185 laboratory images per epoch, with balanced batches. This approximates the field-only optimizer-step budget but halves field-image exposure per epoch; it is not a measured compute match. Lab Validation monitors source retention and does not select adaptation checkpoints.

| Notebook | Experiment record |
| --- | --- |
| [01](notebooks/01_data_exploration.ipynb) | Dataset composition, label harmonization, image examples, leaf-group coverage |
| [02](notebooks/02_train_cnn.ipynb) | ResNet18 laboratory baseline |
| [03](notebooks/03_train_vit.ipynb) | ViT-Small laboratory baseline |
| [04](notebooks/04_domain_gap.ipynb) | Saved baseline Lab Test / original Field Test comparison |
| [05](notebooks/05_augmentation_experiments.ipynb) | ResNet18 photometric/geometric augmentation; saved training run is geometric |
| [06](notebooks/06_field_adaptation_verified.ipynb) | Verified ResNet18 field-only adaptation |
| [07](notebooks/07_vit_field_adaptation.ipynb) | Verified ViT-Small field-only adaptation |
| [08](notebooks/08_mixed_domain_finetuning.ipynb) | Three mixed-domain runs and field-only controls |
| [09](notebooks/09_final_model_comparison.ipynb) | Four fixed checkpoints on Lab Test and Field Dev |
| [10](notebooks/10_error_analysis.ipynb) | Paired errors, Grad-CAM, and last-block ViT attention |

Two superseded adaptation notebooks and four historical scripts remain in `notebooks/archive/` and `src/archive/`.

## Main Results

Values below are rounded to four decimals from saved outputs; no training or inference was performed for this documentation revision.

Laboratory checkpoint selection ([02](notebooks/02_train_cnn.ipynb), [03](notebooks/03_train_vit.ipynb)):

| Baseline | Lab Validation accuracy | Lab Validation macro-F1 |
| --- | ---: | ---: |
| ResNet18 | 0.9896 | 0.9879 |
| ViT-Small | 0.9970 | 0.9966 |

Final fixed-model comparison, from [notebook 09](notebooks/09_final_model_comparison.ipynb):

| Model | Lab Test accuracy | Lab Test macro-F1 | Field Dev correct / 92 | Field Dev accuracy | Field Dev macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 laboratory baseline | 0.9793 | 0.9744 | 30 | 0.3261 | 0.2355 |
| ViT-Small laboratory baseline | 0.9947 | 0.9938 | 37 | 0.4022 | 0.2677 |
| ResNet18 geometric + mixed | 0.9297 | 0.9140 | 43 | 0.4674 | 0.4459 |
| ViT-Small mixed | 0.9900 | 0.9886 | 37 | 0.4022 | 0.3872 |

Mixed-domain ViT improves macro-F1 while retaining the same number of correct Field Dev predictions as its baseline, indicating a change in class-level errors. The ResNet18 comparison combines geometric initialization and mixed-domain fine-tuning; it cannot attribute the full change to mixing alone.

On the current verified Field Dev, frozen ResNet18 macro-F1 is 0.2355 without augmentation, 0.2444 with photometric augmentation, and 0.2497 with geometric augmentation ([saved evaluation](outputs/field_dev_comparison_verified/summary.csv)). Their Lab Validation macro-F1 values are 0.9879, 0.9841, and 0.9753, respectively.

Field-only versus mixed-domain adaptation, from [06](notebooks/06_field_adaptation_verified.ipynb)–[08](notebooks/08_mixed_domain_finetuning.ipynb):

| Laboratory initialization | Field-only Field Dev macro-F1 | Mixed Field Dev macro-F1 | Field-only Lab Validation macro-F1 | Mixed Lab Validation macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 baseline | 0.4119 | 0.3523 | 0.8185 | 0.9556 |
| ResNet18 geometric | 0.4347 | 0.4459 | 0.7668 | 0.9223 |
| ViT-Small baseline | 0.3838 | 0.3872 | 0.9892 | 0.9901 |

Mixing retains more laboratory performance than field-only adaptation in all three saved comparisons, but does not uniformly increase Field Dev macro-F1. The small ViT difference is descriptive and has no repeated-run uncertainty estimate.

Historical baseline Field Test evaluation ([04](notebooks/04_domain_gap.ipynb), 47 images):

| Baseline | Field Test accuracy | Field Test macro-F1 | Lab Test minus Field Test macro-F1 |
| --- | ---: | ---: | ---: |
| ResNet18 | 0.3617 | 0.2802 | 0.6942 |
| ViT-Small | 0.4468 | 0.3617 | 0.6321 |

These historical results establish the observed baseline domain gap. They are not post-adaptation test results or an independent confirmation of later model choices.

## Error Analysis

On Field Dev, the two mixed-domain models are both correct on 24 images and both wrong on 36; ResNet18 alone is correct on 19 and ViT alone on 13. Each recognizes only 3 of 18 Bacterial Spot images. ResNet18 makes 49 errors, including three with softmax confidence ≥ 0.90; ViT makes 55 errors, including nine above that threshold. Softmax scores are not calibrated probabilities of correctness.

[Notebook 10](notebooks/10_error_analysis.ipynb) visualizes eight deterministically selected cases using predicted-class ResNet18 Grad-CAM and mean-head CLS-to-patch attention from the final ViT block. These are different visualization mechanisms and do not establish causal reliance on lesions. The saved manual review template has no completed interpretations; biological attribution remains unverified.

## Reproducibility

The ten main notebooks are executable records of completed runs. Viewing their saved outputs requires no retraining. Re-executing training notebooks updates models and artifacts; preparation scripts construct splits and are not required to inspect saved evidence. Reuse the existing manifests, label map, protocol, audits, and split backups.

For a local Python environment, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements are not a lock of the historical environment. A read-only view of current saved augmentation metrics is available with:

```bash
python -c "import pandas as pd; print(pd.read_csv('outputs/field_dev_comparison_verified/summary.csv').to_string(index=False))"
```

Notebook 04 can recompute comparisons from committed predictions without a GPU, image downloads, or checkpoints, but execution writes to `outputs/comparison/`. Notebooks 02–03 and 05–08 contain Colab training workflows. Notebooks 09–10 use fixed checkpoints for evaluation/visualization; 09's recorded run used CPU. Drive checkpoints and extension CSVs are not all available in this checkout. See [workflow and artifact provenance](docs/workflow.md) for exact paths, prerequisites, and separate output destinations.

The active pipeline is implemented in the notebooks, `src/dataset.py`, the evaluation scripts, and the verified split preparation script. `src/models.py` is a starter factory whose ViT-B/16 is not the research ViT-Small; `src/train.py` raises `NotImplementedError`, and `configs/config.yaml` does not configure the reported runs. The optional Streamlit app uploads images without prediction. These scaffolds are retained; their disposition is recorded in the [documentation audit](docs/documentation_audit.md).

## Limitations

- Field Dev is small, imbalanced, and repeatedly used for selection. The original Field Test was inspected earlier; no fresh independent field evaluation is recorded.
- pHash grouping screens near-duplicates at a fixed threshold and does not guarantee complete duplicate detection. Candidate pairs and conflicting-label groups require visual adjudication.
- PlantDoc includes heterogeneous image sources and potentially ambiguous labels. Dataset composition differs beyond acquisition conditions, so metric gaps cannot be assigned solely to background or lighting.
- Results use one seed and different architecture-specific training configurations. No general CNN-versus-transformer superiority, statistical improvement across runs, or deployment readiness is established.
- Upstream dataset commit SHAs are unresolved, dependency versions are not fully locked, and external Drive artifacts are not completely recoverable from this checkout. Notebook 06 also lacks an A2 epoch-6 log line despite a saved epoch-6 selection summary.
- Quantization and mobile deployment remain optional, unperformed extensions. A verified paper bibliography and completed expert visual review remain documentation gaps.

## References

The following source identifiers are those used in the implementation; they are not a substitute for a verified scholarly bibliography:

- PlantVillage mirror: `geraldmc/plantvillage-full`, linked in the dataset section; default revision.
- PlantDoc mirror: `geraldmc/plantdoc-full`, linked in the dataset section; requested revision `v0.1.0`.
- ResNet18 implementation and weights: `torchvision.models.resnet18`, `ResNet18_Weights.IMAGENET1K_V1`.
- ViT-Small implementation and weights: timm `vit_small_patch16_224.augreg_in1k`; saved ViT runs record timm `1.0.26`.
- Local evidence: [verified evaluation](outputs/field_dev_comparison_verified/summary.csv), [baseline comparison](outputs/comparison/cnn_vs_vit_metrics.csv), and saved outputs in notebooks 01–10.

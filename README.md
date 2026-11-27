# Tomato Disease Classification: CNN and Vision Transformer under Domain Shift

[![CI](https://github.com/thutrang147/plant-disease-diagnosis/actions/workflows/ci.yml/badge.svg)](https://github.com/thutrang147/plant-disease-diagnosis/actions/workflows/ci.yml)

## Abstract

This study investigates the generalization of deep learning models for five-class tomato leaf disease classification under laboratory-to-field domain shift. We compare ImageNet-pretrained **ResNet18** and **ViT-Small** using PlantVillage as the laboratory source domain and PlantDoc as the target domain.

The research examines data augmentation, supervised field adaptation, mixed-domain fine-tuning, external evaluation, class-specific errors, and image framing sensitivity. A final controlled experiment investigates whether laboratory rehearsal improves field performance when Field exposure and optimizer updates are matched.

Our findings reveal a substantial laboratory-to-field performance gap. Adaptation improves several evaluation results, but its benefits vary across classes and experimental protocols.

## Research Questions

The study addresses four connected research questions:

1. **RQ1 — Domain Shift:** How does classification performance change when laboratory-trained models are evaluated on field images?
2. **RQ2 — Domain Adaptation:** How do augmentation and fine-tuning affect field performance, source-domain retention, and class-specific errors?
3. **RQ3 — Image Framing:** How do leaf cropping and alternative inference views affect model predictions?
4. **RQ4 — Controlled Fine-tuning:** Does mixed-domain training consistently improve field performance under matched Field exposure and optimizer updates?

## Datasets and Evaluation Protocol

### Datasets

The primary datasets are:

- **[PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full):** Controlled laboratory images used for baseline training and evaluation.
- **[PlantDoc](https://huggingface.co/datasets/geraldmc/plantdoc-full):** Heterogeneous photographs used for field adaptation and evaluation.

Models classify five categories: **Healthy, Early Blight, Late Blight, Bacterial Spot, and Septoria Leaf Spot**.

| Partition                     |                Images | Purpose                                                    |
| ----------------------------- | --------------------: | ---------------------------------------------------------- |
| Lab Train / Validation / Test | 5,365 / 1,341 / 1,692 | Baseline training, selection, and evaluation               |
| Field Adapt Train / Dev       |              370 / 92 | Supervised adaptation and checkpoint selection             |
| Field Test                    |                    47 | Historical field evaluation                                |
| External Evaluation v1        |                    62 | Additional cross-source evaluation                         |
| External Evaluation v2        |                   103 | Extended evaluation collection with reviewed Healthy crops |

Dataset splits and class mappings are recorded in [`data/splits/`](data/splits/). The verified PlantDoc preparation protocol excludes 14 training images due to near-test overlap or conflicting labels.

### External Evaluation v1 and v2

**External v1** contains 62 images: 16 self-collected Healthy images and 46 online-sourced disease images. It is used for fixed-model evaluation and class-specific error analysis in Notebooks 11–12.

**External v2** extends the collection to 103 images, preserving the 46 online disease images and including 57 self-collected Healthy evaluation images:

| Healthy subset                 | Images |
| ------------------------------ | -----: |
| Original photographs           |      8 |
| Previously approved leaf crops |      7 |
| Newly accepted leaf crops      |     42 |
| **Total Healthy**              | **57** |

The v2 dataset preserves human review decisions, image provenance, parent–crop relationships, and crop coordinates. Its construction and validation are documented in the [External v2 Dataset README](data/external_evaluation/tomato_field_eval_v2/README.md).

**Notebook 13 evaluates only the 57 Healthy images**, not all 103 v2 images. A separate exploratory analysis compares 23 parent photographs with their 49 associated leaf crops.

Because Healthy and disease images have different acquisition sources, v2 is not treated as an independent five-class field benchmark.

## Models and Experimental Design

Two pretrained architectures are evaluated:

- **ResNet18:** A convolutional neural network using torchvision ImageNet-1K pretrained weights.
- **ViT-Small:** A Vision Transformer using timm `vit_small_patch16_224.augreg_in1k` pretrained weights.

Both use 224 × 224 model inputs with architecture-specific preprocessing.

The research follows four experimental stages:

| Stage                                 | Experiments                                                                                    | Notebooks |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- | --------- |
| Laboratory baselines and domain shift | Data exploration, ResNet18/ViT training, baseline evaluation                                   | 01–04     |
| Adaptation and error analysis         | Augmentation, Field-only/Mixed fine-tuning, model comparison, Grad-CAM and external evaluation | 05–12     |
| Image framing sensitivity             | Healthy original–crop analysis and multi-crop inference                                        | 13–14     |
| Controlled comparison                 | Three-seed Field-only versus Mixed ResNet18 fine-tuning                                        | 15        |

The complete experimental implementations and saved outputs are available in [`notebooks/`](notebooks/).

## Main Results

### 1. Laboratory-to-Field Generalization

The following table compares four historical model checkpoints using five-class **Macro-F1**.

| Model                      | Lab Test |  Field Dev | External v1 |
| -------------------------- | -------: | ---------: | ----------: |
| ResNet18 Baseline          |   0.9744 |     0.2355 |      0.2406 |
| ViT-Small Baseline         |   0.9938 |     0.2677 |      0.4157 |
| ResNet18 Geometric + Mixed |   0.9140 | **0.4459** |  **0.4882** |
| ViT-Small Mixed            |   0.9886 |     0.3872 |      0.4808 |

Both laboratory-trained models experience substantial performance degradation on field-domain data. Adaptation improves several results, although improvements are not uniform across models or classes.

### 2. External Evaluation and Class-Specific Errors

On External v1, ResNet18 Geometric + Mixed achieves **53.23% accuracy** and **0.4882 Macro-F1**, compared with 30.65% accuracy and 0.2406 Macro-F1 for the laboratory baseline.

However, aggregate improvements conceal important class-specific trade-offs:

- **Healthy:** Correct predictions increase from 0/16 to 12/16.
- **Early Blight:** Correct predictions decrease from 11/13 to 1/13.
- **Bacterial Spot:** Correct predictions increase from 1/9 to 5/9.

These results motivate further error analysis rather than relying on aggregate metrics alone. Notebook 10 additionally examines selected predictions using Grad-CAM and ViT attention visualizations.

### 3. External v2 — Healthy Leaf Recognition

Notebook 13 evaluates the same four frozen models on 57 self-collected Healthy images.

| Model                      | Correct / 57 | Healthy Recall |
| -------------------------- | -----------: | -------------: |
| ResNet18 Baseline          |         0/57 |           0.0% |
| ViT-Small Baseline         |        15/57 |          26.3% |
| ResNet18 Geometric + Mixed |    **44/57** |      **77.2%** |
| ViT-Small Mixed            |        43/57 |          75.4% |

The adapted models recognize substantially more Healthy images in this collection.

A separate paired analysis of 23 parent photographs and 49 leaf crops finds improved recognition after cropping for several models. These results describe selected, correlated images rather than independent plant-level trials.

### 4. Image Framing and Cropping Sensitivity

Notebook 14 compares standard inference, center crops, and five-crop inference using frozen model checkpoints.

For ViT-Small Mixed on External v1:

| Inference Strategy | Correct / 62 |   Macro-F1 |
| ------------------ | -----------: | ---------: |
| Standard           |        32/62 |     0.4808 |
| Five-crop          |    **38/62** | **0.5840** |

Five-crop inference averages predictions from five 224 × 224 views. It changes image coverage, object scale, and aggregation simultaneously; the observed improvement cannot be attributed to resolution alone.

### 5. Controlled Field-only vs. Mixed Fine-tuning

Notebook 15 addresses limitations in the historical training comparisons through a controlled ResNet18 experiment.

Both conditions share the same laboratory initialization, Field-image exposure, optimizer-update budget, and paired checkpoint-selection protocol. Mixed training additionally includes laboratory samples.

Across seeds 42, 43, and 44, the historical 47-image Field Test produces:

| Strategy       | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) |
| -------------- | -------------------: | -------------------: |
| **Field-only** |  **0.4610 ± 0.0123** |  **0.4300 ± 0.0075** |
| Mixed          |      0.4468 ± 0.0213 |      0.4072 ± 0.0296 |

Field-only achieves higher mean Macro-F1 in this controlled experiment, leading in two of the three seeds.

The historical advantages of Mixed-domain adaptation are therefore not consistently reproduced under this ResNet18 protocol. The result does not establish that Mixed training is universally inferior.

## Discussion and Research Contributions

The study provides three main contributions:

1. **Laboratory-to-field evaluation:** Quantifies the generalization gap between controlled and heterogeneous images while distinguishing development, historical test, and external evaluation results.
2. **Adaptation and error analysis:** Examines source/target performance trade-offs, class-specific failures, and sensitivity to image framing, including a reviewed Healthy crop dataset.
3. **Controlled experimental evidence:** Tests laboratory rehearsal under matched ResNet18 training conditions and clarifies the interpretation of earlier Mixed-domain comparisons.

Overall, the findings show that high laboratory accuracy, adaptation gains, and cropping improvements depend on the evaluation population and experimental protocol.

Further interpretation and supporting evidence are provided in the [Research Results Summary](docs/research_results_summary.md).

## Reproducibility

Clone the repository and install the dependencies:

```bash
git clone https://github.com/thutrang147/plant-disease-diagnosis.git
cd plant-disease-diagnosis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the available automated tests and repository audit:

```bash
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

These checks use saved research evidence and do not require model retraining. The same checks run in GitHub Actions on every pull request and push to `main` ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

The repository contains 15 canonical notebooks, evaluation modules, dataset manifests, selected prediction exports, and supporting documentation. Full experiment reproduction additionally requires the original images, model checkpoints, and recorded execution environments, some of which are not publicly distributed.

See [Workflow and Artifact Provenance](docs/workflow.md) for setup and reproduction details.

## Demo App and Single-Image Inference

The Streamlit demo has two tabs. **Diagnose a leaf** runs every available checkpoint on one photo (example, upload or webcam) and shows each model's determination side by side, so the lab-only and field-adapted checkpoints can be compared live, optionally with their explanation maps. **Research findings** charts the domain gap, the class trade-offs and the controlled comparison from committed evidence and works without checkpoints.

Checkpoints are not tracked in Git: copy the `.pt` files from the project's Google Drive results (see [Checkpoints and persistent Colab results](docs/workflow.md#checkpoints-and-persistent-colab-results)) into `models/` using these local filenames:

| Local file                         | Model                                                            |
| ---------------------------------- | ---------------------------------------------------------------- |
| `resnet18_geometric_mixed_best.pt` | ResNet18 Geometric + Mixed (highest reported Field Dev macro-F1) |
| `vit_small_mixed_best.pt`          | ViT-Small Mixed                                                  |
| `resnet18_lab_best.pt`             | ResNet18 Baseline                                                |
| `vit_small_lab_best.pt`            | ViT-Small Baseline                                               |

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Run it from the repository root so the theme in `.streamlit/config.toml` applies (its fonts load from Google Fonts and fall back to system fonts offline). For a presentation, put a few photos in `app/examples/`; they appear as one-click examples. That folder is ignored by Git so private or third-party photos are not committed.

```bash
python -m src.inference --checkpoint models/resnet18_geometric_mixed_best.pt path/to/leaf.jpg
python -m src.inference --checkpoint models/vit_small_mixed_best.pt --strategy five_crop --json path/to/folder/
```

[`src/inference.py`](src/inference.py) validates class order and architecture, loads weights strictly, applies EXIF orientation as in Notebook 11 and uses the evaluation preprocessing of each architecture. `five_crop` reuses the Notebook 14 views and is exploratory. Reported probabilities are uncalibrated softmax outputs. On field images these models reach a reported macro-F1 of 0.24–0.49 with standard inference, so predictions are research outputs, not crop-management advice. Set `PLANT_MODELS_DIR` or `PLANT_EXAMPLES_DIR` to use other folders. The app can also overlay Notebook 10's ResNet18 Grad-CAM or ViT-Small last-block attention on the standard view; like the notebook figures, these maps are qualitative and do not show reliance on a lesion.

## Limitations

The evaluation is constrained by small field datasets, previously inspected test images, acquisition-source differences, and correlated image crops. Grad-CAM and attention analyses are qualitative, and the cropping experiments do not isolate lesion resolution.

The controlled comparison covers one ResNet18 configuration, with all three seeds evaluated on the same historical test images. Independent field evaluation would be needed to support stronger generalization claims.

Some evaluation images and checkpoints remain outside the public repository. Historical third-party image redistribution rights require separate review.

## References and Documentation

**Research documentation**

- [Research Results Summary](docs/research_results_summary.md)
- [Experiment Reconciliation](docs/experiment_reconciliation.md)
- [Workflow and Artifact Provenance](docs/workflow.md)
- [Submission Audit](docs/submission_audit.md)

**Key references**

- Hughes and Salathé (2015). _An Open Access Repository of Images on Plant Health._
- Singh et al. (2020). _PlantDoc: A Dataset for Visual Plant Disease Detection._
- He et al. (2016). _Deep Residual Learning for Image Recognition._
- Dosovitskiy et al. (2021). _An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale._
- Selvaraju et al. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization._

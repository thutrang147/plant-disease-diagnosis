# Tomato Disease Classification: CNN and Vision Transformer under Domain Shift

## Abstract

This study investigates the generalization of deep learning models for five-class tomato leaf disease classification under laboratory-to-field domain shift. We compare ImageNet-pretrained **ResNet18** and **ViT-Small** using PlantVillage as the laboratory source domain and PlantDoc as the target domain.

The research examines data augmentation, supervised field adaptation, mixed-domain fine-tuning, external evaluation, class-specific errors, and image framing sensitivity. A final controlled experiment investigates whether laboratory rehearsal improves field performance when Field exposure and optimizer updates are matched.

Laboratory-trained models lose most of their accuracy on field photographs. Supervised field adaptation mainly teaches the models to recognise healthy field leaves; on independent online disease images, accuracy shows no detectable gain (ResNet 19 → 21, ViT 22 → 19 of 46). Under matched training exposure, adding laboratory images (Mixed fine-tuning) produced no detectable difference in field performance, although a difference of a few points cannot be excluded; its demonstrated benefit is retention of laboratory accuracy.

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

All counts are correct images / images. Field partitions are small, and one image changes a class recall by 9–12.5 percentage points. Paired tests use exact McNemar with Holm adjustment across the reported family. Supporting tables are in [`data/results/supplementary_statistics/`](data/results/supplementary_statistics/).

### 1. Laboratory-to-Field Generalization (RQ1)

Five-class macro-F1 of the four fixed checkpoints. Field Dev selected the two adapted checkpoints (development scores); the 47-image Field Test was inspected from Notebook 04 onward.

| Model | Lab Test (1,692) | Field Dev (92) | Field Test (47) | External v1 (62) |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 Lab baseline | 0.9744 | 0.2355 | 0.2802 (17/47) | 0.2406 (19/62) |
| ViT-Small Lab baseline | 0.9938 | 0.2677 | 0.3617 (21/47) | 0.4157 (27/62) |
| ResNet18 Geometric + Mixed | 0.9140 | 0.4459 | 0.3587 (19/47) | 0.4882 (33/62) |
| ViT-Small Mixed | 0.9886 | 0.3872 | 0.5103 (25/47) | 0.4808 (32/62) |

Laboratory accuracy does not transfer: on PlantDoc the lab-trained baselines fall from 0.97–0.99 to 0.24–0.36 macro-F1, and the adapted models reach at most 0.51. No model is best on every field partition. ResNet18 Geometric + Mixed leads on Field Dev, which selected it, but is below the ViT-Small Lab baseline on Field Test. None of the eight paired comparisons between these models on Field Test and External v1 is significant after Holm adjustment over the ten-test family, which also includes the two disease-stratum tests (smallest adjusted p = 0.24).

### 2. Adaptation and Class-Specific Errors (RQ2)

External v1 combines two sources that coincide with class: **16 self-collected Healthy** photographs and **46 online disease** images. Pooled accuracy therefore mixes "recognises healthy field leaves" with "discriminates diseases", so the two strata are reported separately.

| Model | Healthy, self-collected (16) | Disease, online (46) | Disease 4-class macro-F1 |
| --- | ---: | ---: | ---: |
| ResNet18 Lab baseline | 0/16 | 19/46 | 0.360 |
| ViT-Small Lab baseline | 5/16 | 22/46 | 0.445 |
| ResNet18 Geometric + Mixed | 12/16 | 21/46 | 0.437 |
| ViT-Small Mixed | 13/16 | 19/46 | 0.399 |

- Adaptation mainly fixes the baselines' failure to predict **Healthy** for field leaves. On online disease images, it gives no detectable gain: ResNet 19 → 21 (McNemar p = 0.83), ViT 22 → 19 (p = 0.58). The pooled ResNet improvement (19 → 33 of 62, unadjusted p = 0.024) is driven by the Healthy stratum and is not significant after Holm adjustment.
- Adaptation also shifts the ResNet decision threshold towards Healthy in general. On Lab Test, the adapted ResNet labels 63 diseased images as Healthy, against 2 for the baseline.
- **Early Blight is a failure for every model, in different ways.** The ResNet baseline's 11/13 Early Blight recall comes from predicting Early Blight for 39 of 62 external images in the canonical Notebook 11 run (precision 0.28; 43 of 62, precision 0.26, in the tracked upstream-loader predictions summarised in `external_v1_prediction_distribution.csv`) and for 55 of 92 Field Dev images. The adapted ResNet almost never predicts it (1/13).
- Bacterial Spot is confused with Septoria Leaf Spot across all models and partitions.

Notebook 10 adds Grad-CAM (ResNet) and last-block attention (ViT) views of Field Dev errors. These are qualitative only: there are no lesion annotations, several PlantDoc images carry stock-photo watermarks, and the ViT maps concentrate on scattered background tokens.

### 3. Image Framing and Cropping Sensitivity (RQ3)

**Healthy recognition (Notebook 13, 57 self-collected Healthy images, 49 of them crops from 23 parent photographs in 21 scenes).** Correct: ResNet Lab 0/57, ViT Lab 15/57, ResNet Mixed 44/57, ViT Mixed 43/57. Comparing 23 parent photographs with their own crops, parent-balanced crop recall exceeds the original-photo recall for the adapted models (ResNet Mixed 10/23 originals vs 0.756 crop recall; ViT Mixed 14/23 vs 0.828). This is a Healthy-only population: it measures sensitivity, not specificity. The effective sample size is closer to 21–23 scenes than to 57 images.

**Automatic inference views (Notebook 14).** Center crops (0.70, 0.50) and five-crop averaging change predictions in both directions. Across 4 models × 3 alternative views × 2 datasets, the largest gain is ViT Mixed five-crop on External v1 (32 → 38/62, macro-F1 0.4808 → 0.5840). The largest loss is ResNet Mixed center-0.50 on Field Dev (macro-F1 0.4459 → 0.3603). These policies were compared after the outcomes were known. Every model input stays 224 × 224, so the study shows **sensitivity to framing**, not a resolution effect or a deployable policy.

### 4. Controlled Field-only vs. Mixed Fine-tuning (RQ4)

Notebook 15 holds initialization (ResNet18 Lab baseline), Field exposure (all 370 images per epoch, identical augmented tensors), optimizer updates (192) and a shared checkpoint-selection epoch fixed. Mixed adds an equal number of Lab images per step. Seeds 42, 43 and 44 were declared before training, and checkpoints were hash-sealed before the Field Test was opened.

| Seed | Selected epoch | Field Dev macro-F1, FO / Mixed | Field Test macro-F1, FO / Mixed | Test images only FO / only Mixed correct |
| --- | ---: | ---: | ---: | ---: |
| 42 | 8 | 0.4388 / 0.4390 | 0.4338 / 0.3970 | 3 / 2 |
| 43 | 7 | 0.4361 / 0.4478 | 0.4214 / 0.3840 | 1 / 0 |
| 44 | 8 | 0.4147 / 0.4327 | 0.4349 / 0.4405 | 1 / 1 |

Mean Field Test macro-F1 is 0.4300 ± 0.0075 for Field-only and 0.4072 ± 0.0296 for Mixed. The seed-mean difference is −0.023, with a **paired bootstrap 95% CI of [−0.086, +0.024]**; for accuracy it is −0.014, CI [−0.064, +0.028]. On Field Dev at the selected epoch, Mixed is equal or higher in all three seeds. The two evaluation sets point in opposite directions, and the Test difference rests on a net two images.

**Lab retention.** Notebook 15 selected checkpoints on Field Dev only, so neither Lab partition was used by the controlled run. The six sealed checkpoints and their shared Lab baseline were evaluated on Lab Validation (1,341) and Lab Test (1,692) with [`scripts/evaluate_notebook15_lab_retention.py`](scripts/evaluate_notebook15_lab_retention.py) (inference only; every checkpoint SHA-256 checked against the seal first). The baseline reproduces its previously reported Lab Test result exactly (1,657/1,692, macro-F1 0.9744).

| Seed | Lab Val macro-F1, FO / Mixed | Lab Test macro-F1, FO / Mixed | Lab Test correct, FO / Mixed | Lab Test Δ macro-F1 (Mixed − FO) |
| --- | ---: | ---: | ---: | ---: |
| Lab baseline (shared initialization) | 0.9879 | 0.9744 | 1,657 | — |
| 42 | 0.8441 / 0.9733 | 0.8116 / 0.9553 | 1,490 / 1,634 | +0.144 |
| 43 | 0.8341 / 0.9706 | 0.8092 / 0.9449 | 1,482 / 1,622 | +0.136 |
| 44 | 0.8533 / 0.9780 | 0.8199 / 0.9603 | 1,499 / 1,641 | +0.140 |

Mean Lab Test macro-F1 is 0.8136 ± 0.0056 for Field-only and 0.9535 ± 0.0079 for Mixed (seed-mean difference +0.140). On Lab Validation it is 0.8438 ± 0.0096 vs 0.9740 ± 0.0038 (+0.130). Relative to the shared initialization, Field-only loses 0.161 Lab Test macro-F1 and Mixed loses 0.021. The difference has the same sign and similar size in every seed, and is more than ten times the seed-to-seed SD. The three seeds are evaluated on the same images, so they show training-run stability, not independent samples of laboratory data. Lab Validation also selected the baseline checkpoint, so its baseline value is optimistic, and Lab Test is the cleaner comparison. Images come from `geraldmc/plantvillage-full` at revision `f808f27` (the default branch at evaluation time). Per-checkpoint results, summaries and the protocol record are in [`data/results/notebook15_lab_retention/`](data/results/notebook15_lab_retention/).

**Conclusion:** under matched exposure, laboratory rehearsal produces **no detectable difference in field performance** for this ResNet18 recipe (Field Test macro-F1 −0.023, 95% CI [−0.086, +0.024]; this does not establish equivalence, and a difference of a few points in either direction cannot be excluded), but it **largely prevents the loss of laboratory performance** (Lab Test macro-F1 −0.021 with Mixed vs −0.161 with Field-only). Both findings agree with the matched-initialization historical pair (Notebook 06 A1 vs Notebook 08 R1). There, Mixed did not improve Field Dev but retained Lab Validation much better (0.8185 → 0.9556). Notebook 15 now reproduces that retention effect with initialization, Field exposure and optimizer updates controlled. The earlier apparent Mixed advantage on field data came from comparing a geometric-augmentation initialization and selecting the best of six configurations on the same 92 Dev images.

## Discussion and Research Contributions

1. **Laboratory accuracy does not transfer.** The lab-trained baselines score 0.97–0.99 macro-F1 in the lab and 0.24–0.36 on PlantDoc. PlantVillage has documented capture bias (Noyan, 2022), which is a plausible contributor, though this study does not isolate it.
2. **Supervised field adaptation fixes the coarsest error, not disease discrimination.** Adapted models learn to call field leaves Healthy. On online disease images they are no better than the lab baselines, and Early Blight / Bacterial Spot remain unsolved. Aggregate gains must be read by stratum and class.
3. **Predictions are sensitive to framing**, both under human-reviewed crops and under automatic inference views. The direction depends on model and dataset.
4. **Lab rehearsal's demonstrated benefit is retention.** With exposure matched, no field-performance difference between Mixed and Field-only fine-tuning was detected; equivalence was not tested.
5. **Evaluation hygiene.** Duplicate-aware splits, hash-sealed checkpoints before test access, byte-level provenance of every reported prediction, and an explicit separation of development, historical-test and external evidence.

Further interpretation: [Results Summary](docs/research_results_summary.md) · settings for every run: [Methods](docs/methods.md).

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
python -m src.supplementary_analysis --output data/results/supplementary_statistics_rerun
```

These checks use saved research evidence and do not require model retraining.

The repository contains 15 canonical notebooks, evaluation modules, dataset manifests, selected prediction exports, and supporting documentation. Full experiment reproduction additionally requires the original images, model checkpoints, and recorded execution environments, some of which are not publicly distributed.

See [Workflow and Artifact Provenance](docs/workflow.md) for setup and reproduction details.

## Limitations

- **No locked, independent field test.** The 47-image Field Test was inspected early and reused. Field Dev selected checkpoints. External v1 confounds source with class. No experiment has the power to detect differences below about 10 percentage points.
- **Label noise.** PlantDoc contains conflicting duplicates. One Field Test image (row 2524, Bacterial Spot) duplicates a training image labelled Septoria and is predicted Septoria in every controlled run.
- **Confounded architecture comparison.** Pretraining, preprocessing, batch size, precision and the share of trainable parameters during adaptation (ResNet 75%, ViT 8%) all differ.
- **Low training budget.** Adaptation runs selected their final or near-final epoch while Field Dev was still rising.
- **Web-sourced "field" images.** PlantDoc photographs were collected from the web; several carry stock-photo watermarks. Field results describe web photographs, not images captured in the field by the project.
- **Metric choice and non-significance.** Accuracy depends on class priors: a model biased towards the largest class can match another model's accuracy with worse class balance, so accuracy and macro-F1 are reported together. No equivalence margin was pre-specified, so a non-significant difference is not evidence of equivalence.
- **Single seed** for Notebooks 02–14. Notebook 15 has three seeds on one ResNet18 recipe, all evaluated on the same 47 images.
- **Qualitative explanations only.** There are no lesion annotations, and the cropping experiments do not isolate resolution.
- **Distribution.** Checkpoints, private evaluation images and some local outputs are not in the repository; see [Data Notice](DATA_NOTICE.md).

## References and Documentation

**Documentation**

- [Methods at a glance](docs/methods.md): data, every training recipe, evaluation and statistics
- [Research Results Summary](docs/research_results_summary.md)
- [Workflow and Artifact Provenance](docs/workflow.md)
- [Data Notice and licences](DATA_NOTICE.md)
- Process records (audits, reconciliation logs): [`docs/archive/`](docs/archive/)

**Key references**

- Hughes and Salathé (2015). _An Open Access Repository of Images on Plant Health to Enable the Development of Mobile Disease Diagnostics._ arXiv:1511.08060.
- Singh et al. (2020). _PlantDoc: A Dataset for Visual Plant Disease Detection._ CoDS-COMAD.
- Noyan (2022). _Uncovering Bias in the PlantVillage Dataset._ arXiv:2206.04374.
- He et al. (2016). _Deep Residual Learning for Image Recognition._ CVPR.
- Dosovitskiy et al. (2021). _An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale._ ICLR.
- Steiner et al. (2022). _How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers._ TMLR.
- Selvaraju et al. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization._ ICCV.
- McNemar (1947). _Note on the Sampling Error of the Difference between Correlated Proportions or Percentages._ Psychometrika.
- Holm (1979). _A Simple Sequentially Rejective Multiple Test Procedure._ Scandinavian Journal of Statistics.

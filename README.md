# Tomato Disease Classification: CNN and Vision Transformer under Domain Shift

## Abstract

This study compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato disease classification, with PlantVillage as the laboratory source domain and PlantDoc as the field target domain. Completed experiments cover laboratory baselines, lab-to-field evaluation, photometric and geometric augmentation, supervised field adaptation with duplicate-aware splits, mixed-domain fine-tuning, and exploratory error analysis. Both laboratory baselines lose substantial macro-F1 on PlantDoc. Mixed-domain ResNet18 initialized from the geometric-augmentation checkpoint achieves the highest Field Dev macro-F1 among the four models in the final comparison (0.4459), while mixed-domain ViT-Small retains higher Lab Test macro-F1 (0.9886 versus 0.9140). Field Dev was used for checkpoint selection; these field results are developmental rather than independent confirmation.

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

The main comparison evaluates four fixed checkpoints on Lab Test (1,692 images) and verified Field Dev (92 images), as recorded in [notebook 09](notebooks/09_final_model_comparison.ipynb) and [`outputs/final_comparison_development/`](outputs/final_comparison_development/). Values are rounded to four decimals, with 95% Wilson accuracy confidence intervals from [`outputs/statistical_analysis/accuracy_confidence_intervals.csv`](outputs/statistical_analysis/accuracy_confidence_intervals.csv). Field Dev selected the mixed checkpoints, so its scores describe development performance.

| Model | Lab Test accuracy (95% CI) | Lab Test macro-F1 | Field Dev correct / 92 | Field Dev accuracy (95% CI) | Field Dev macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 laboratory baseline | 0.9793 [0.9714, 0.9851] | 0.9744 | 30 | 0.3261 [0.2389, 0.4272] | 0.2355 |
| ViT-Small laboratory baseline | 0.9947 [0.9899, 0.9972] | 0.9938 | 37 | 0.4022 [0.3078, 0.5043] | 0.2677 |
| ResNet18 geometric + mixed | 0.9297 [0.9165, 0.9409] | 0.9140 | 43 | 0.4674 [0.3688, 0.5686] | 0.4459 |
| ViT-Small mixed | 0.9900 [0.9840, 0.9937] | 0.9886 | 37 | 0.4022 [0.3078, 0.5043] | 0.3872 |

On Lab Test, ViT-Small's accuracy lead over ResNet18 is statistically significant (exact two-sided McNemar $p = 2.4 \times 10^{-5}$; 32 versus 6 discordant predictions; [`paired_mcnemar_tests.csv`](outputs/statistical_analysis/paired_mcnemar_tests.csv)). Mixed-domain ResNet18 has the highest observed Field Dev macro-F1 of these four models, while mixed-domain ViT retains higher Lab Test performance. Mixed-domain ViT increases macro-F1 without changing its number of correct Field Dev predictions (raising Healthy recall from 0.0000 to 0.5455 and Bacterial Spot recall from 0.0000 to 0.1667 in both adapted models), reflecting a redistribution of class-level errors. The ResNet18 contrast combines geometric initialization and mixed-domain fine-tuning; the full change cannot be attributed to mixing alone.

Supporting laboratory results in [02](notebooks/02_train_cnn.ipynb) and [03](notebooks/03_train_vit.ipynb) report Lab Validation accuracy/macro-F1 of 0.9896/0.9879 for ResNet18 and 0.9970/0.9966 for ViT-Small on 1,341 images. These are checkpoint-selection scores, separate from Lab Test.

On the current verified Field Dev, frozen ResNet18 macro-F1 is 0.2355 (30/92 correct) without augmentation, 0.2444 (31/92) with photometric augmentation, and 0.2497 (32/92, exact McNemar $p = 0.7539$ versus baseline) with geometric augmentation ([saved evaluation](outputs/field_dev_comparison_verified/summary.csv)). Their Lab Validation macro-F1 values are 0.9879, 0.9841, and 0.9753, respectively.

Field-only versus mixed-domain adaptation, from [06](notebooks/06_field_adaptation_verified.ipynb)–[08](notebooks/08_mixed_domain_finetuning.ipynb):

| Laboratory initialization | Field-only Field Dev macro-F1 | Mixed Field Dev macro-F1 | Field-only Lab Validation macro-F1 | Mixed Lab Validation macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 baseline | 0.4119 | 0.3523 | 0.8185 | 0.9556 |
| ResNet18 geometric | 0.4347 | 0.4459 | 0.7668 | 0.9223 |
| ViT-Small baseline | 0.3838 | 0.3872 | 0.9892 | 0.9901 |

Mixing retains more laboratory performance than field-only adaptation in all three saved comparisons, but does not uniformly increase Field Dev macro-F1. The small ViT difference is descriptive and has no repeated-run uncertainty estimate.

The historical [baseline Field Test evaluation](notebooks/04_domain_gap.ipynb) used 47 images: ResNet18 accuracy/macro-F1 was 0.3617/0.2802 (17/47 correct), and ViT-Small was 0.4468/0.3617 (21/47 correct). The corresponding Lab Test minus Field Test macro-F1 gaps were 0.6942 and 0.6321. These observations document the baseline domain shift; they do not test the adapted models or independently confirm later choices.

## Error Analysis and Overconfidence Metrics

On Field Dev, the two mixed-domain models are both correct on 24 images (26.1%) and both wrong on 36 (39.1%, including 13 of 18 Bacterial Spot images); ResNet18 alone is correct on 19 and ViT alone on 13 (32 discordant images, 34.8%; exact two-sided McNemar $p = 0.3771$). Each recognizes only 3 of 18 Bacterial Spot images.

Statistical calibration and confidence metrics ([`calibration_and_confidence_metrics.csv`](outputs/statistical_analysis/calibration_and_confidence_metrics.csv)) show that while the ResNet18 laboratory baseline is well-calibrated on Lab Test (10-bin ECE = 0.0098), its zero-shot Field Dev predictions are severely overconfident (ECE = 0.5660; mean confidence on incorrect predictions = 0.8816), with 39 of 62 mistakes (62.9% of errors) made at softmax confidence ≥ 0.90. After mixed-domain fine-tuning, high-confidence errors (softmax confidence ≥ 0.90) drop to 3 of 49 mistakes (6.1%) for ResNet18 geometric + mixed and 9 of 55 mistakes (16.4%) for ViT-Small mixed. Softmax scores for the mixed models record top-1 confidence rather than full class probability vectors, so high-confidence error share does not by itself measure overall calibration.

[Notebook 10](notebooks/10_error_analysis.ipynb) visualizes eight deterministically selected cases ([`cnn_gradcam_vs_vit_attention.png`](outputs/error_analysis_development/cnn_gradcam_vs_vit_attention.png) and the 4-case grid [`gradcam_vit_attention_4cases.png`](outputs/error_analysis_development/gradcam_vit_attention_4cases.png)) using predicted-class ResNet18 Grad-CAM and mean-head CLS-to-patch attention from the final ViT block. These are different visualization mechanisms and do not establish causal reliance on lesions. The notebook exports a manual review template ([`manual_visual_review_template.csv`](outputs/error_analysis_development/manual_visual_review_template.csv)); human adjudication is currently ongoing.

## Reproducibility and Ongoing Work

For the recorded findings, start with [09: final comparison](notebooks/09_final_model_comparison.ipynb) and [10: error analysis](notebooks/10_error_analysis.ipynb); the notebook index above links the preceding experiments. Committed CSV and figure artifacts are available in [`outputs/comparison/`](outputs/comparison/), [`outputs/field_dev_comparison_verified/`](outputs/field_dev_comparison_verified/), [`outputs/final_comparison_development/`](outputs/final_comparison_development/), [`outputs/error_analysis_development/`](outputs/error_analysis_development/), and [`outputs/statistical_analysis/`](outputs/statistical_analysis/).

- [`src/statistical_analysis.py`](src/statistical_analysis.py) reproduces all 95% Wilson/bootstrap confidence intervals, exact McNemar tests, ECE / high-confidence error summaries, and the 4-case qualitative figure directly from saved outputs.
- [`src/evaluate_confirmatory_field.py`](src/evaluate_confirmatory_field.py) provides the locked evaluation and pHash duplicate-screening pipeline for the ongoing phase of collecting real-world tomato leaf photographs, testing leaf crops at 224 × 224 resolution, and running a one-shot confirmatory evaluation on held-out field data.

Model weight files (`.pth`) are not distributed in Git due to size. This local checkout contains four ignored baseline/augmentation checkpoints; field-adapted and mixed-domain `.pth` checkpoints and raw training histories remain on external Google Drive storage, while their evaluation summaries, predictions, and figures are committed in `outputs/`. Exact filenames, artifact locations, local setup, and execution behavior are documented in [workflow and artifact provenance](docs/workflow.md#local-setup-and-saved-evidence).

Existing manifests, label mappings, protocols, and audits define the reported samples and should be reused. Notebook 04 can recompute baseline comparisons from committed predictions without checkpoints or image downloads, but writes comparison artifacts when executed. Dependencies are not locked to a complete historical environment.

The active pipeline is implemented in the notebooks, `src/dataset.py`, the evaluation and statistical scripts, and the verified split preparation script. `src/models.py` is a starter factory whose ViT-B/16 is not the research ViT-Small; `src/train.py` raises `NotImplementedError`, and `configs/config.yaml` does not configure the reported runs. The optional Streamlit app uploads images without prediction. These scaffolds are retained; their disposition is recorded in the [documentation audit](docs/documentation_audit.md).

## Limitations and Ongoing Tasks

- Field Dev is small ($n=92$), imbalanced, and used for checkpoint selection. Collecting an independent real-world tomato leaf dataset (including close-up leaf crops to test the 224 × 224 resolution bottleneck on *Bacterial Spot*) and running a one-shot confirmatory evaluation via [`src/evaluate_confirmatory_field.py`](src/evaluate_confirmatory_field.py) are currently **[Ongoing]**.
- pHash grouping screens near-duplicates at a fixed threshold ($d_{\text{pHash}} \le 4$) and does not guarantee complete duplicate detection. Candidate pairs and conflicting-label groups require visual adjudication.
- PlantDoc includes heterogeneous image sources and potentially ambiguous labels. Dataset composition differs beyond acquisition conditions, so metric gaps cannot be assigned solely to background or lighting.
- Results use one seed and different architecture-specific training configurations. No general CNN-versus-transformer superiority, statistical improvement across runs, or deployment readiness is established.
- Upstream dataset commit SHAs are unresolved, dependency versions are not fully locked, and external Drive `.pth` weights are not tracked in Git. Notebook 06 also lacks an A2 epoch-6 log line despite a saved epoch-6 selection summary.
- Completing the human visual review adjudication ([`manual_visual_review_template.csv`](outputs/error_analysis_development/manual_visual_review_template.csv)) is **[Ongoing]**; quantization and mobile deployment remain optional future extensions.

## References

### Dataset and Software Resources
- PlantVillage mirror: `geraldmc/plantvillage-full`, linked in the dataset section; default revision.
- PlantDoc mirror: `geraldmc/plantdoc-full`, linked in the dataset section; requested revision `v0.1.0`.
- ResNet18 implementation and weights: `torchvision.models.resnet18`, `ResNet18_Weights.IMAGENET1K_V1`.
- ViT-Small implementation and weights: timm `vit_small_patch16_224.augreg_in1k`; saved ViT runs record timm `1.0.26`.
- Local evidence: [`outputs/field_dev_comparison_verified/`](outputs/field_dev_comparison_verified/), [`outputs/comparison/`](outputs/comparison/), [`outputs/final_comparison_development/`](outputs/final_comparison_development/), [`outputs/error_analysis_development/`](outputs/error_analysis_development/), [`outputs/statistical_analysis/`](outputs/statistical_analysis/), and saved outputs in notebooks 01–10.

### Scholarly Bibliography
1. D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015.
2. S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016.
3. K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018.
4. J. G. A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," *Biosystems Engineering*, vol. 172, pp. 84–91, 2018.
5. D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease detection," in *Proc. ACM India Joint Int. Conf. Data Science and Management of Data (CODS-COMAD)*, 2020, pp. 249–253.
6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
7. A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.
8. A. Steiner, A. Kolesnikov, X.~Zhai, R. Wightman, J. Uszkoreit, and L. Beyer, "How to train your ViT? Data, augmentation, and regularization in Vision Transformers," *Transactions on Machine Learning Research (TMLR)*, 2022.
9. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.

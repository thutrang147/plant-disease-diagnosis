# Tomato Disease Classification: CNN and Vision Transformer under Domain Shift

## Abstract

This study compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato disease classification across PlantVillage (laboratory source domain), PlantDoc (field target domain), and an independent 62-image external field test set (`tomato_field_eval_v1`, combining 16 self-collected smartphone leaf photographs and 46 verified university extension field images). Completed experiments (Notebooks 01–11) cover duplicate-audited data splitting, laboratory baselines, photometric and geometric augmentation, partial field-only versus 50:50 mixed-domain fine-tuning, confirmatory evaluation on held-out field data, paired error and overconfidence analysis, and a multi-scale cropping test of the 224 × 224 spatial resolution bottleneck on *Bacterial Spot*.

Both laboratory baselines suffer large macro-F1 drops on all field partitions (losing 56.01 to 73.90 percentage points) and make overconfident field errors (ResNet18 10-bin ECE = 0.5660 on Field Dev and 0.5623 on External Test). Combining geometric pre-augmentation with 50:50 mixed-domain fine-tuning achieves the highest Field Dev macro-F1 (0.4459) while retaining 0.9140 Lab Test macro-F1, and generalizes to the independent 62-image External Test set (raising ResNet18 macro-F1 from 0.2456 to 0.4882, accuracy from 30.65% to 53.23%, exact McNemar $p = 0.0243$, lifting self-collected *Healthy* recall from 0.00% to 75.00%, and eliminating errors with confidence ≥ 0.90 from 60.47% down to 0.00%). Meanwhile, ViT-Small Mixed reaches 0.4710 macro-F1 on External Test, leads on PlantDoc Field Test ($n=47$, macro-F1 = 0.5103), and reaches 0.5840 macro-F1 (61.29% accuracy, 66.67% *Bacterial Spot* recall) under 448 × 448 five-crop high-resolution tiling.

## Research Questions

1. How do ResNet18 and ViT-Small compare on laboratory images and under lab-to-field domain shift across PlantDoc and independent external field images?
2. How do source-only photometric and geometric augmentations affect zero-shot field performance?
3. How do field-only and 50:50 mixed-domain fine-tuning trade off target-domain adaptation, field-image exposure budget, source-domain retention, and high-confidence errors?
4. How do local CNN filters and global ViT attention differ on paired field cases, and does downsampling full field scenes to 224 × 224 obscure small *Bacterial Spot* lesions?

## Datasets and Evaluation Protocol

Images are loaded from the Hugging Face mirrors [PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full) and [PlantDoc](https://huggingface.co/datasets/geraldmc/plantdoc-full) (`v0.1.0`), plus the local independent external test dataset [`data/external_evaluation/tomato_field_eval_v1/`](data/external_evaluation/tomato_field_eval_v1/tomato_field_eval_v1/manifest.csv). The shared label order is fixed in [`label_map.json`](data/splits/label_map.json): Healthy (0), Early Blight (1), Late Blight (2), Bacterial Spot (3), and Septoria Leaf Spot (4).

| Partition | Images | Role |
| --- | ---: | --- |
| Lab Train | 5,365 | PlantVillage training; also sampled during 50:50 mixed-domain fine-tuning |
| Lab Validation | 1,341 | Laboratory checkpoint selection; source-retention monitoring during adaptation |
| Lab Test | 1,692 | Fixed held-out laboratory evaluation partition |
| Field Adapt Train | 370 | Cleaned PlantDoc training pool for field adaptation |
| Verified Field Dev | 92 | Target-domain early stopping, checkpoint selection, and exploratory error analysis |
| Quarantined field training rows | 14 | Excluded from adaptation and Field Dev (4 near Field Test + 10 label conflicts) |
| PlantDoc Field Test | 47 | Held-out original PlantDoc test split |
| External Field Test (`tomato_field_eval_v1`) | 62 | Independent confirmatory test set (16 self-collected + 46 verified extension images) |

Lab Validation is formed from original PlantVillage training data with leaf-group-aware 5-fold splitting (`StratifiedGroupKFold`, seed 42). The verified PlantDoc protocol groups 64-bit pHash connected components at Hamming distance ≤ 4 across the original 476 training and 47 test images (`20` candidate pairs: `11` benign same-label pairs grouped within splits, `4` cross-split leaks near Field Test removed, and `10` images in `5` conflicting-label pairs removed). See [`field_dev_protocol.json`](data/splits/field_dev_protocol.json) and [`docs/workflow.md`](docs/workflow.md).

The 62-image External Field Test ([`manifest.csv`](data/external_evaluation/tomato_field_eval_v1/tomato_field_eval_v1/manifest.csv)) contains 16 Healthy (self-collected smartphone photos: 8 multi-leaf and 8 cropped target leaflets), 13 Early Blight, 11 Late Blight, 9 Bacterial Spot, and 13 Septoria Leaf Spot images. Audited in [`external_audit_protocol.json`](outputs/external_confirmatory_eval/external_audit_protocol.json), all 62 images have zero SHA-256 or pHash matches ($d_{\text{pHash}} \le 4$) against PlantDoc (minimum Hamming distance = 16) and zero internal duplicates.

## Models and Experimental Design

ResNet18 uses torchvision `IMAGENET1K_V1` weights (11.18M parameters); ViT-Small uses timm `vit_small_patch16_224.augreg_in1k` (21.67M parameters). Both take 224 × 224 inputs with seed 42. Baselines train the head for 3 epochs and the full network for 5 epochs, selecting by Lab Validation macro-F1.

In Stage 3 adaptation, after a 2-epoch head warmup, only the final backbone block is unfrozen for up to 6 epochs (early stopping patience = 3 on Field Dev macro-F1): `layer4 + fc` in ResNet18 (8.40M trainable parameters, 75.1% of the model) and `blocks.11 + norm + head` in ViT-Small (1.77M trainable parameters, 8.2% of the model). Because trainable parameter shares and pretraining recipes differ substantially, comparing adaptation gains does not isolate backbone architecture alone.

Mixed-domain fine-tuning samples 185 field plus 185 laboratory images per epoch in 50:50 balanced batches (16 field + 16 lab). This matches the 370-image optimizer-step budget of field-only fine-tuning per epoch, but **halves field-image exposure per epoch** (185 versus 370).

| Notebook | Experiment record |
| --- | --- |
| [01](notebooks/01_data_exploration.ipynb) | Dataset composition, label harmonization, image examples, leaf-group coverage |
| [02](notebooks/02_train_cnn.ipynb) | ResNet18 laboratory baseline |
| [03](notebooks/03_train_vit.ipynb) | ViT-Small laboratory baseline |
| [04](notebooks/04_domain_gap.ipynb) | Baseline Lab Test / original PlantDoc Field Test comparison |
| [05](notebooks/05_augmentation_experiments.ipynb) | ResNet18 photometric and geometric source-only augmentation |
| [06](notebooks/06_field_adaptation_verified.ipynb) | Verified ResNet18 field-only adaptation |
| [07](notebooks/07_vit_field_adaptation.ipynb) | Verified ViT-Small field-only adaptation |
| [08](notebooks/08_mixed_domain_finetuning.ipynb) | Three 50:50 mixed-domain runs and field-only controls |
| [09](notebooks/09_final_model_comparison.ipynb) | Four locked checkpoints on Lab Test ($n=1,692$) and Field Dev ($n=92$) |
| [10](notebooks/10_error_analysis.ipynb) | Paired Field Dev errors, Grad-CAM, and last-block ViT attention |
| [11](notebooks/11_external_field_and_resolution_eval.ipynb) | Confirmatory evaluation on External Test ($n=62$) & Field Test ($n=47$) + 224 × 224 resolution/cropping test |

## Main Results

### 1. Four-Model Benchmark Across Laboratory, Development, and Confirmatory Field Partitions

From [Notebook 09](notebooks/09_final_model_comparison.ipynb) ([`outputs/final_comparison_development/`](outputs/final_comparison_development/)) and [Notebook 11](notebooks/11_external_field_and_resolution_eval.ipynb) ([`outputs/external_confirmatory_eval/`](outputs/external_confirmatory_eval/)):

| Model | Lab Test Acc ($n=1,692$) | Lab Test F1 | Field Dev Acc ($n=92$) | Field Dev F1 | Field Test Acc ($n=47$) | Field Test F1 | External Test Acc ($n=62$, 95% CI) | External Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 laboratory baseline | 0.9793 | 0.9744 | 0.3261 (30/92) | 0.2355 | 0.3617 (17/47) | 0.2802 | 0.3065 (19/62) [0.2058, 0.4297] | 0.2456 |
| ViT-Small laboratory baseline | **0.9947** | **0.9938** | 0.4022 (37/92) | 0.2677 | 0.4468 (21/47) | 0.3617 | 0.4516 (28/62) [0.3342, 0.5747] | 0.4346 |
| ResNet18 geometric + mixed | 0.9297 | 0.9140 | **0.4674** (43/92) | **0.4459** | 0.4043 (19/47) | 0.3587 | **0.5323** (33/62) [0.4098, 0.6509] | **0.4882** |
| ViT-Small mixed | 0.9900 | 0.9886 | 0.4022 (37/92) | 0.3872 | **0.5319** (25/47) | **0.5103** | 0.5000 (31/62) [0.3792, 0.6208] | 0.4710 |

- **Development Ablation (Notebooks 05–08):** Source-only photometric and geometric augmentations yield minor zero-shot Field Dev changes (`0.2355 -> 0.2444` and `0.2497`, exact McNemar $p = 0.7539$). Field-only adaptation raises ResNet18 Field Dev macro-F1 (`0.4119` from standard baseline, `0.4347` from geometric baseline) but causes severe source forgetting on Lab Validation (`0.9879 -> 0.8185` and `0.9753 -> 0.7668`). Mixing 50:50 Lab + Field images preserves Lab Validation F1 (`0.9556`, `0.9223`, `0.9901`); because mixing halves field-image exposure per epoch, standard ResNet18 mixed tuning stops early at epoch 2 (`0.3523`), whereas geometric initialization + 50:50 mixed tuning trains for 6 epochs and reaches `0.4459`.
- **Independent External Confirmation ($n=62$):** `ResNet18 geometric + mixed` significantly outperforms its laboratory baseline on External Test (`33/62` vs. `19/62`, exact two-sided McNemar $p = 0.0243$, `24` vs. `10` discordant images), nearly doubling macro-F1 (`0.2456 -> 0.4882`). On the 16 self-collected smartphone photos of healthy leaves, both `ResNet18 geometric + mixed` and `ViT-Small mixed` achieve **75.00% accuracy (12/16)**, compared with **0.00% (0/16)** for ResNet18 baseline and **37.50% (6/16)** for ViT-Small baseline. On external *Bacterial Spot* ($n=9$), both mixed models reach **55.56% recall (5/9)** (up from `11.11%` and `22.22%`).
- **Original PlantDoc Field Test ($n=47$):** `ViT-Small mixed` achieves the highest score (`0.5103` macro-F1, `53.19%` accuracy, `25/47`), while `ResNet18 geometric + mixed` improves from `0.2802` to `0.3587`. Across all three field splits (`Field Dev`, `Field Test`, and `External Test`), exact McNemar tests between the two adapted models show no statistically significant difference ($p = 0.3771$, $p = 0.1460$, and $p = 0.8036$).

### 2. Paired Error Analysis and Overconfidence Reduction

- On Field Dev ($n=92$), the two mixed models are both correct on 24 images (26.1%), both wrong on 36 (39.1%), and disagree on 32 (34.8%: 19 ResNet-only, 13 ViT-only). On External Test ($n=62$), both are correct on 24 (38.7%), both are wrong on 22 (35.5%), and they disagree on 16 (25.8%: 9 ResNet-only, 7 ViT-only).
- **Overconfident Error Suppression:** For ResNet18, the share of mistakes made at softmax confidence ≥ 0.90 drops from **62.9% (39/62) to 6.1% (3/49)** on Field Dev, from **70.0% (21/30) to 7.1% (2/28)** on PlantDoc Field Test (10-bin ECE `0.5459 -> 0.2408`), and from **60.5% (26/43) to 0.0% (0/29)** on External Test (10-bin ECE `0.5623 -> 0.1273`). For ViT-Small Mixed on External Test, 10-bin ECE drops from `0.3631` to `0.2551` and high-confidence errors drop from `29.4% (10/34)` to `22.6% (7/31)`.

### 3. Testing the 224 × 224 Resolution Hypothesis on *Bacterial Spot*

In External Test, *Bacterial Spot* images have a median original resolution of 1080 × 1440 pixels (a **41.33× area compression** or **6.43× linear reduction** when resized to 224 × 224). Evaluating four inference-time spatial regimes ([`bacterial_spot_resolution_experiment.csv`](outputs/external_confirmatory_eval/bacterial_spot_resolution_experiment.csv)) reveals:

| Partition | Model | `1.00x` Full-Scene (All F1 / Bact. Recall) | `0.70x` Center Crop (All F1 / Bact. Recall) | `0.50x` Close-Up (All F1 / Bact. Recall) | `448x448` 5-Crop (All F1 / Bact. Recall) |
| --- | --- | ---: | ---: | ---: | ---: |
| Field Dev ($n=92$, $18$ Bact.) | ViT-Small Mixed | 0.3872 / 0.1667 | **0.4201** / 0.2222 | 0.4181 / **0.2778** | 0.3761 / 0.1667 |
| External Test ($n=62$, $9$ Bact.) | ViT-Small Mixed | 0.4710 / 0.5556 | 0.5084 / 0.4444 | 0.4897 / 0.2222 | **0.5840** / **0.6667** |
| External Test ($n=62$, $9$ Bact.) | ResNet18 Geom+Mixed | **0.4882** / **0.5556** | 0.4540 / 0.4444 | 0.4284 / **0.5556** | 0.4441 / **0.5556** |

- **Higher spatial resolution helps ViT-Small resolve fine lesions:** On Field Dev, tighter center crops (`0.70x`, `0.50x`) raise ViT-Small Mixed's *Bacterial Spot* recall from `16.67% (3/18)` to `22.22% (4/18)` and `27.78% (5/18)` while doubling mean true-class probability $P(\text{Bacterial Spot})$ (`0.1419 -> 0.2832`). On External Test, **`448x448` 5-Crop high-resolution tiling** lifts ViT-Small Mixed's *Bacterial Spot* recall to **66.67% (6/9)**, halves misclassifications into *Septoria* (`4/9 -> 2/9`), and boosts overall 5-class macro-F1 to **0.5840 (61.29% accuracy, 38/62)**. Consistently, on External Test images that already isolate a single/cropped leaf (`visible_leaf_context == single_or_partial_leaf`, $n=15$), ViT-Small Mixed reaches **60.00% accuracy (9/15)** versus **46.81% (22/47)** on multi-leaf scenes.
- **Why blind center-cropping is not a universal fix:** When symptomatic leaflets lie off-center near the image borders (e.g., Field Dev Case ID 1629), a fixed `0.50x` center crop clips out the lesion, lowering ResNet18 Geometric+Mixed's Field Dev macro-F1 (`0.4459 -> 0.3603`). Thus, overcoming the 224 × 224 resolution bottleneck requires **multi-crop high-resolution tiling (`448x448` 5-Crop) or target-leaf detection/cropping** rather than blind center cropping.

## Reproducibility

All evaluation summaries, per-image predictions, statistical tests, and figures are committed in:
- [`outputs/comparison/`](outputs/comparison/) (Notebook 04 baseline domain gap)
- [`outputs/field_dev_comparison_verified/`](outputs/field_dev_comparison_verified/) (Notebook 05 verified augmentation comparison)
- [`outputs/final_comparison_development/`](outputs/final_comparison_development/) (Notebook 09 four-model development comparison)
- [`outputs/error_analysis_development/`](outputs/error_analysis_development/) (Notebook 10 paired error analysis and Grad-CAM / ViT attention figures)
- [`outputs/statistical_analysis/`](outputs/statistical_analysis/) (Wilson/bootstrap 95% CIs, exact McNemar tests, ECE, produced by [`src/statistical_analysis.py`](src/statistical_analysis.py))
- [`outputs/external_confirmatory_eval/`](outputs/external_confirmatory_eval/) (Notebook 11 confirmatory results on `tomato_field_eval_v1` and `PlantDoc Field Test`, subgroup breakdown, pHash audit, and 224 × 224 resolution experiment, supported by [`src/evaluate_confirmatory_field.py`](src/evaluate_confirmatory_field.py))

Large `.pt` checkpoint weight files remain on Google Drive (`/content/drive/MyDrive/plant-disease-diagnosis/`) and are excluded from Git via `.gitignore`. See [`docs/workflow.md`](docs/workflow.md) for full artifact provenance.

## Limitations

- Experiments use a single random seed (42) and standard architecture-specific training setups.
- 50:50 mixed-domain fine-tuning matches the per-epoch optimizer step count of field-only fine-tuning (370 images/epoch) but halves field-image exposure per epoch (185 vs. 370).
- Partial adaptation unfreezes 75.1% of ResNet18 parameters (`layer4 + fc`) versus 8.2% of ViT-Small parameters (`blocks.11 + norm + head`), so differences in adaptation gains reflect trainable parameter capacity as well as backbone architecture.
- Grad-CAM and ViT attention visualizations use different mechanisms (class-gradient weighting vs. unguided token attention) and provide exploratory qualitative context rather than causal proof.

## References

1. D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015.
2. S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016.
3. K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018.
4. J. G. A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," *Biosystems Engineering*, vol. 172, pp. 84–91, 2018.
5. D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease detection," in *Proc. ACM India Joint Int. Conf. Data Science and Management of Data (CODS-COMAD)*, 2020, pp. 249–253.
6. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
7. A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.
8. A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, and L. Beyer, "How to train your ViT? Data, augmentation, and regularization in Vision Transformers," *Transactions on Machine Learning Research (TMLR)*, 2022.
9. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.

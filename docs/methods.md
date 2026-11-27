# Methods at a glance

This page puts every training and evaluation setting in one place, so the comparisons can be judged
without reading the notebooks. Values come from the saved notebook code and outputs; cell references
point to where they can be checked.

## Data

| Partition | Images (H / EB / LB / BS / SLS) | Role |
| --- | --- | --- |
| Lab Train (PlantVillage) | 5,365 (1012 / 640 / 1221 / 1363 / 1129) | Lab training; Lab rehearsal in Mixed runs |
| Lab Validation | 1,341 (253 / 160 / 306 / 340 / 282) | Lab checkpoint selection; retention monitoring |
| Lab Test | 1,692 (326 / 200 / 382 / 424 / 360) | Lab evaluation (original PlantVillage test split) |
| Field Adapt Train (PlantDoc) | 370 (44 / 60 / 79 / 80 / 107) | Supervised adaptation |
| Field Dev | 92 (11 / 16 / 20 / 18 / 27) | Adaptation checkpoint selection (development, optimistic) |
| Field Test | 47 (8 / 9 / 10 / 9 / 11) | Original PlantDoc test split; **inspected since Notebook 04** |
| External v1 | 62 = 16 self-collected Healthy + 46 online disease | Source and class fully confounded; report by stratum |
| Healthy v2 | 57 Healthy (8 originals + 49 crops from 23 parents / 21 scenes) | Healthy recognition only |

Lab splits use `StratifiedGroupKFold` on PlantVillage `leaf_id`. Some images (1,510 / 378 / 465 in
train / val / test) have no reliable leaf group and are their own group, so same-leaf duplicates across
Lab partitions cannot be ruled out. PlantDoc Adapt/Dev are separated by pHash (Hamming ≤ 4)
connected components. 14 training rows were quarantined: 4 near Field Test images and 10 with conflicting labels.
One retained Field Test image (row 2524, labelled Bacterial Spot) has a pHash-0 duplicate in training
labelled Septoria (`data/splits/field_near_duplicate_candidates.csv`), so its label is uncertain.

## Training recipes

| Run | Notebook | Init | Trainable in adaptation stage | Batch | LR (head / backbone) | Precision | Epochs | Selection | Selected epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 Lab baseline | 02 | ImageNet-1K V1 | head 3 ep, then all 11.18 M | 32 | 1e-3 → 5e-4 / 1e-4 | fp32 | 3 + 5 | Lab Val macro-F1 | FT 4/5 |
| ViT-Small Lab baseline | 03 | AugReg in1k | head 3 ep, then all 21.67 M | 16 | 1e-3 → 5e-5 / 1e-5 | AMP fp16 | 3 + 5 | Lab Val macro-F1 | FT 3/5 |
| ResNet18 geometric aug | 05 | ImageNet-1K V1 | as baseline + geometric aug | 32 | as baseline | fp32 | 3 + 5 | Lab Val macro-F1 | FT 5/5 |
| A1 / A2 Field-only (ResNet) | 06 | Lab baseline / geometric | `layer4`+fc = 8.40 M (**75%**) | 32 | 1e-4 / 1e-5 | fp32, BN frozen | 2 + ≤6 | Field Dev macro-F1 | 6/6, 6/6 |
| V1 Field-only (ViT) | 07 | ViT Lab baseline | last block+norm+head = 1.78 M (**8%**) | 16 | 1e-4 / 1e-5 | AMP | 2 + ≤6 | Field Dev macro-F1 | 5/6 |
| R1 / R2 / V1 Mixed | 08 | as A1 / A2 / V1 | as above | 16+16 (ViT 8+8) | as above | as above | 2 + ≤6, 185 Field + 185 Lab per epoch | Field Dev macro-F1 | 6/6 for all |
| Controlled Field-only / Mixed | 15 | ResNet Lab baseline | `layer4`+fc | 16 Field (+ Lab batch in Mixed) | 1e-4 / 1e-5 | fp32, BN frozen | 2 + 6, all 370 Field per epoch | shared epoch, mean pair Dev macro-F1 | 8, 7, 8 of 8 |

Things to keep in mind when comparing these runs:

- **ResNet vs ViT** differ in pretraining recipe, preprocessing, batch size, precision and, during
  adaptation, the share of parameters updated (75% vs 8%). Differences in Lab retention can
  follow from the trainable share alone.
- **Budget.** Almost every adaptation run selected its last or second-to-last epoch while Field Dev
  was still rising. The results describe a low-budget fine-tuning regime (Notebook 15: 192 updates)
  rather than converged models.
- **Historical Field-only vs Mixed (06 vs 08)** differ in Field exposure (370 vs 185 per epoch) and stop
  independently. With the same initialization (A1 vs R1), Mixed was already lower on Field Dev
  (0.4119 vs 0.3523) but retained Lab much better (Lab Val 0.8185 vs 0.9556).

## Evaluation and statistics

- Metrics: accuracy, five-class macro-F1 (absent predictions score F1 = 0), per-class precision and
  recall with **correct/total counts**. Per-class denominators on Field Test are 8–11, so one image
  changes a class recall by 9–12.5 percentage points.
- External v1 is reported **by stratum**: Healthy recognition on 16 self-collected images, and
  4-class accuracy / macro-F1 on 46 online disease images
  (`data/results/supplementary_statistics/external_v1_by_stratum.csv`).
- Paired comparisons use exact McNemar tests with **Holm adjustment** across the reported family
  (`mcnemar_holm.csv`). Notebook 15 uses an image-level paired bootstrap of the seed-mean difference
  (`notebook15_paired_bootstrap.csv`).
- Softmax confidence is reported as-is. It is not calibrated, and ECE on 47–62 images is unstable.
- Grad-CAM (ResNet) and last-block CLS attention (ViT) are qualitative only. No lesion annotations
  exist, and the ViT attention maps concentrate on scattered background tokens.

Regenerate the supplementary statistics (no models required):

```bash
python -m src.supplementary_analysis --output data/results/supplementary_statistics_rerun
```

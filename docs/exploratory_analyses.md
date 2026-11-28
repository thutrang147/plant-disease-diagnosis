# Exploratory analyses (details behind the README summaries)

None of these analyses was pre-registered except where a declaration commit is named, and every one that touches
External v3 reuses a test set that had already been evaluated (see the [evaluation ledger](evaluation_ledger.md)).
The README keeps a one-paragraph summary of each and links here.

## 1. Frozen foundation-model reference (RQ1)

**Frozen foundation-model reference (exploratory, not pre-registered).** [`scripts/evaluate_frozen_foundation_probe.py`](../scripts/evaluate_frozen_foundation_probe.py) trains only a logistic-regression head on a frozen DINOv2 backbone, using the same manifests.

| DINOv2 ViT-B/14, frozen (macro-F1) | Lab Test | Field Dev | Field Test | External v3 accuracy / disease macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Lab-only probe (C chosen on Lab Val) | 0.976 | 0.399 | 0.336 (20/47) | 0.473 / 0.366 |
| Lab + Field probe (C chosen on Field Dev) | 0.961 | 0.622 | 0.665 (31/47) | 0.469 / 0.456 |
| DINOv2 ViT-S/14 (same size as ViT-Small), Lab + Field probe | 0.935 | 0.570 | 0.538 (25/47) | 0.436 / 0.432 |
| Best fixed project checkpoint in each column (Notebooks 08–09) | 0.989 | 0.446 | 0.510 (25/47) | 0.444 / 0.441 |
| Full fine-tuning (LP-FT), exploratory (below): ViT-Small at 8 epochs and ResNet18 at 24 epochs, the two LP-FT runs with the highest External v3 accuracy | 0.863–0.984 | 0.516–0.555 | 0.562–0.564 | 0.484–0.498 / 0.469–0.487 |

The frozen DINOv2-B probe exceeds the fixed project checkpoints on both PlantDoc partitions. That advantage does not carry over cleanly:
- On External v3 the margin is 2–3 pp of accuracy, and fully fine-tuned ResNet18 and ViT-Small reach the same or higher (0.484–0.498).
- The same-size DINOv2-S/14 probe does not beat them.
- The comparison mixes representation, model capacity (86M vs 22M parameters), pretraining data (web-scale LVD-142M, which may contain similar images) and input size (224 instead of DINOv2's native 518).

What it does show is that gains measured inside PlantDoc overstate gains on other field data, which is why a separate external set is needed. Results: [`data/results/frozen_foundation_probe/`](../data/results/frozen_foundation_probe/).

## 2. Architecture comparison under one recipe and the training budget (RQ1)

**Architecture comparison under one recipe (exploratory, 3 seeds, CPU).** The historical adaptation updated 75.1% of ResNet18 parameters (`layer4` + head) but only 8.2% of ViT-Small (last block + head). [`scripts/explore_lpft_architecture.py`](../scripts/explore_lpft_architecture.py) re-adapts both Lab baselines with the same Field-only recipe and batch size. It compares that last-block setting with a two-stage schedule: the head for 2 epochs, then all parameters. The Lab heads were already trained, so the first stage changes little, and this is mainly a test of fine-tuning depth rather than of LP-FT's feature-distortion argument; the label "LP-FT" is kept for brevity. [`scripts/explore_training_budget.py`](../scripts/explore_training_budget.py) repeats it with 24 instead of 8 epochs (seed 42).

| Seed mean (macro-F1) | Lab Test | Field Dev | Field Test | External v3 accuracy / F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet18, last block (75%) | 0.815 | 0.432 | 0.422 | 0.418 / 0.408 |
| ResNet18, LP-FT (100%) | 0.786 | 0.460 | 0.461 | 0.403 / 0.389 |
| ViT-Small, last block (8%) | 0.989 | 0.399 | 0.521 | 0.441 / 0.433 |
| ViT-Small, LP-FT (100%) | 0.984 | 0.516 | 0.564 | 0.484 / 0.469 |
| *24 epochs, seed 42:* ResNet18, last block | 0.864 | 0.531 | 0.507 | 0.471 / 0.457 |
| *24 epochs, seed 42:* ResNet18, LP-FT | 0.863 | 0.555 | 0.562 | 0.498 / 0.487 |
| *24 epochs, seed 42:* ViT-Small, last block | 0.990 | 0.470 | 0.579 | 0.461 / 0.454 |
| *24 epochs, seed 42:* ViT-Small, LP-FT | 0.972 | 0.517 | 0.584 | 0.475 / 0.464 |

At the historical 8-epoch budget, full fine-tuning helps ViT-Small (+4 pp on External v3) more than ResNet18. At seed 42, ResNet18 Field Dev is still rising through epoch 23 of 24 without a clear plateau, while ViT-Small plateaus around epochs 12–18, so the ResNet-versus-ViT ranking depends on the training budget as well as the fine-tuning depth. Longer Field-only training also costs Lab retention: ResNet18 Lab Test macro-F1 is about 0.86 at 24 epochs, against 0.97–0.99 for ViT-Small. The historical 8-epoch runs (including Notebook 15) compared models before they converged. External v3 numbers in this table are exploratory: they are a few of more than 20 evaluations made on External v3 after the confirmatory run, so the largest of them should not be read as a best model. With one seed for the 24-epoch runs and about ±4.4 pp single-model intervals, none of these architecture differences is established. The ResNet18 last-block 8-epoch row reproduces the sealed Notebook 15 Field-only results (Lab Test 0.815 vs 0.814), which supports the CPU pipeline.

## 3. Boundary sensitivity: Bacterial Spot and Septoria merged (RQ2)

**Boundary sensitivity** (post-hoc, exploratory; [`external_v3_label_merge/`](../data/results/external_v3_label_merge/)). Merging Bacterial Spot and Septoria into one class raises External v3 accuracy by 8–15 pp for every model. Under the merge, P1 becomes +4.5 pp (95% CI −0.8 to +9.9; McNemar p = 0.12, Holm 0.28), while P2 and P3 barely change (+1.2 and +1.5 pp); none is significant. A merge credits every BS↔Septoria confusion, whether it comes from label noise or from the model, so this shows how much of the error lies on that boundary, not that the labels are wrong. Label noise on this pair is documented separately: 6 of the 13 label conflicts among 64 images shared with PlantDoc are BS↔Septoria. An expert audit of those images would estimate the noise rate.

| Model | External v3 errors | of which BS↔Septoria swaps |
| --- | ---: | ---: |
| ResNet18 Lab baseline | 279 | 39 (14%) |
| ViT-Small Lab baseline | 277 | 59 (21%) |
| ResNet18 Geometric + Mixed | 280 | 62 (22%) |
| ViT-Small Mixed | 270 | 58 (21%) |
| Notebook 15 Field-only, 3 seeds | 282–286 | 73–74 (26%) |
| Notebook 15 Mixed, 3 seeds | 272–279 | 70–74 (26–27%) |

## 4. Normalization statistics: AdaBN (RQ2)

**Normalization statistics (AdaBN, exploratory; declared in `828bbfe`).** [`scripts/evaluate_adabn.py`](../scripts/evaluate_adabn.py) re-estimates the Lab baseline's BatchNorm statistics on the 370 unlabelled Field Adapt Train images, with no weight changed. On Field Dev, macro-F1 rises from 0.235 to 0.391 (+0.155, paired bootstrap 95% CI 0.049 to 0.257), while accuracy rises by 8.7 pp (95% CI −2.2 to +19.6; McNemar p = 0.18, Holm 0.55). This indicates that part of the gap is a normalization-statistics mismatch that can be reduced without labels. The clearest evidence is within PlantDoc, the same source as the statistics (Field Dev). On cross-source External v3 the accuracy gain is +3.3 pp (−1.9 to +8.6; p = 0.25, Holm 0.55), not distinguishable from zero, and disease macro-F1 goes from 0.333 to 0.431 (+0.098, 0.050 to 0.146), similar to what supervised Mixed adaptation reached (0.419). On the 47-image Field Test, macro-F1 changes by +0.115 (−0.038 to +0.262; accuracy p = 0.61). Accuracy changes are not significant, and their intervals are wide (up to +19.6 pp on Field Dev), so a change in the number of errors is not excluded either. Applying AdaBN after supervised adaptation lowers every adapted checkpoint (Field Dev macro-F1 −0.09 to −0.11; External v3 accuracy −1.6 to −7.0 pp), consistent with those weights having been fine-tuned with BN frozen at laboratory statistics (not tested directly). AdaBN applies to ResNet18 only (ViT-Small uses LayerNorm). It also lowers Lab Test macro-F1 from 0.974 to 0.673, so a deployed model would need domain-specific statistics. Whether adapting BN first and then fine-tuning combines the gains was not tested ([`adabn/`](../data/results/adabn/), [`adabn_paired/`](../data/results/adabn_paired/)).

## 5. Lesion reliance and Grad-CAM localization with PlantSeg masks (RQ2)

**Lesion reliance, measured with PlantSeg masks** ([`scripts/evaluate_lesion_reliance.py`](../scripts/evaluate_lesion_reliance.py), exploratory). Notebook 10's Grad-CAM and attention views are qualitative. The expert lesion masks allow two measurements on the same 486 images:

| Measurement | Lab baselines | Adapted / Notebook 15 models |
| --- | ---: | ---: |
| Accuracy: original → lesions inpainted | 0.43 → 0.30 | 0.41–0.44 → 0.25–0.28 |
| Lesion-free images predicted Healthy | 0–3% | 5–8% (ViT Mixed 18%) |
| Grad-CAM pointing accuracy, ResNet18 (chance 0.46) | 0.65 | 0.58–0.66 |
| Grad-CAM mass inside lesions / lesion area share (median) | 1.23 | 1.20–1.27 |
| Accuracy with only the lesions kept (rest set to mean colour) | 0.38–0.39 | 0.26–0.38 |
| *Same 378-image subset (lesions small enough for a control region):* original / lesion removed / random same-area region removed / lesion only | 0.43–0.47 / 0.30–0.34 / 0.44–0.49 / 0.38–0.40 | 0.40–0.43 / 0.26–0.29 / 0.40–0.43 / 0.23–0.35 |
| Extra accuracy drop from lesion vs control inpainting (95% CI) | 0.14–0.15 (0.10 to 0.20) | 0.12–0.15 (0.07 to 0.20) |
| Centre-point baseline for the pointing game (ResNet18 rows) | 0.74 | 0.74 |

- **The models do rely on lesion pixels.** On all 486 images, inpainting the lesions turns 38–52% of the originally correct predictions into errors (net accuracy loss 29–43%). On the 378 images whose lesions leave room for a non-overlapping region, inpainting a same-area random region instead changes accuracy by only −1.6 to +1.9 pp. The lesion-specific extra drop is 0.12–0.15 for every model, with intervals excluding zero ([`occlusion_control/`](../data/results/occlusion_control/), [`occlusion_subset/`](../data/results/occlusion_subset/)). The 108 excluded images have the largest lesions, so the control is biased towards small lesions. Keeping *only* the lesions also lowers accuracy, possibly because removing all context is a much larger edit.
- **Grad-CAM localizes lesions above chance but no better than a centre-point prior.** Its peak falls on a lesion for 0.58–0.66 of images, against 0.46 by area chance but 0.74 for a trivial baseline that always points at the image centre. A likely explanation, not tested here, is that web photos centre the lesion. CAM mass on lesions is only about 1.25 times their area share (a mass ratio, not a frequency), and adaptation does not improve it. So the models use lesion pixels, but these ResNet18 Grad-CAM maps are not reliable evidence of where they look; other attribution methods were not tested. Grad-CAM was computed for ResNet18 only.
- **The Healthy class is not "no lesion".** Images with their lesions removed are rarely called Healthy. This is consistent with the stratum result above, but the 5-px dilation can leave chlorotic halos, so it is suggestive only.

## 6. Field-only vs Mixed at a 24-epoch budget (RQ4)

**At a 24-epoch budget** (exploratory; declared in `0d47ea8`, extended in `7c1a280`, sanity check corrected before the runs in `64d749b`). Both arms were re-run on CPU for 24 epochs with the same recipe, seeds and Field-Dev selection rule ([`scripts/explore_mixed_budget.py`](../scripts/explore_mixed_budget.py); these runs are unsealed). At the Notebook 15 selected epoch, every run reproduces its sealed Lab Test macro-F1 within 0.001, bit-identically in four of six runs. Field Dev at that epoch matches the sealed values in 5 of 6 runs. For Mixed s44 the CPU re-run gives 0.418 against the sealed 0.433, although Lab Test is bit-identical. Re-scoring the sealed checkpoints reproduces 0.433 ([`nb15_sealed_field_dev/`](../data/results/nb15_sealed_field_dev/)), so the re-run, not the record, diverges; GPU-vs-CPU numerics is a likely but untested cause.

| Seed | Selected epoch, FO / Mixed | Lab Test macro-F1, FO / Mixed | Field Dev macro-F1 at selected epoch | Field Dev, mean of epochs 20–24 | Field Test macro-F1 | External v3 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 23 / 20 | 0.864 / 0.977 | 0.531 / 0.554 | 0.517 / 0.531 | 0.507 / 0.535 | 0.471 / 0.484 |
| 43 | 22 / 22 | 0.855 / 0.980 | 0.546 / 0.550 | 0.522 / 0.526 | 0.548 / 0.531 | 0.461 / 0.488 |
| 44 | 22 / 23 | 0.865 / 0.980 | 0.544 / 0.552 | 0.523 / 0.531 | 0.490 / 0.490 | 0.457 / 0.490 |

Retention, the primary endpoint, holds: Mixed is higher on Lab Test macro-F1 by +0.113 to +0.125 in every seed. On Field Dev, Mixed is ahead in all three seeds (+0.004 to +0.023 at the selected epoch, seed-level p = 0.19; +0.004 to +0.015 for the epoch 20–24 mean, p = 0.09), a margin of 1–2 correct images out of 92 per seed. Selected values are maxima over 25 epochs and therefore optimistic for both arms. Field Test differences are +0.028, −0.017 and 0.000 macro-F1 (n = 47; inconclusive). On External v3, Mixed is higher by +1.2, +2.7 and +3.3 pp. With the P3 analysis this is +2.4 pp (paired bootstrap 95% CI +0.7 to +4.1; sign-flip p = 0.008), and disease macro-F1 is +0.018 (0.003 to 0.035; p = 0.026, an arm-swap permutation, since macro-F1 is not a per-image statistic; accuracy uses the declared test). Training longer helps both arms on External v3 (seed-mean accuracy: Field-only 0.417 → 0.463, Mixed 0.434 → 0.487), so for field accuracy the budget matters more than rehearsal; this is a description, not a test.

At 24 epochs, rehearsal keeps its retention benefit (+0.11 to +0.13 Lab Test macro-F1 in every seed, seed-level p = 0.001), and External v3 shows a small exploratory advantage of +2.4 pp (95% CI +0.7 to +4.1). That CI conditions on three training runs; with seeds as the unit, p = 0.06. External v3 had already been reused before this analysis ([ledger](evaluation_ledger.md)). The 8-epoch interval (−0.1 to +3.6 pp) was compatible with an effect of this size. Per-run results, seed-level differences and consistency checks: [`mixed_budget_summary/`](../data/results/mixed_budget_summary/).

## 7. Weight interpolation as an alternative to rehearsal (RQ4)

**Can weight interpolation replace rehearsal?** [`scripts/evaluate_weight_interpolation.py`](../scripts/evaluate_weight_interpolation.py) interpolates linearly between the Lab baseline and each Field-only checkpoint, θ(α) = (1 − α)·θ_Lab + α·θ_FO (WiSE-FT/PAINT style; no Lab data used during fine-tuning). The curve is computed on development data, and the operating points are chosen by pre-declared rules before the test partitions are touched. The endpoints reproduce the sealed results exactly.

| Seed mean (macro-F1 unless noted) | Lab Val | Field Dev | Lab Test | Field Test (acc / F1) | External v3 (acc / disease F1) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interpolation, retention-matched (α = 0.4) | 0.976 | 0.266 | 0.951 | 0.426 / 0.339 | 0.447 / 0.375 |
| Interpolation, Field-Dev-selected (α = 1.0 = Field-only) | 0.844 | 0.430 | 0.814 | 0.461 / 0.430 | 0.417 / 0.405 |
| Mixed (rehearsal) | 0.974 | 0.440 | 0.954 | 0.447 / 0.407 | 0.434 / 0.420 |

Along the curve, Field Dev rises steadily with α while Lab Validation falls. Matching Mixed's Field Dev (about 0.43) requires α ≥ 0.9, where Lab Validation is 0.88 or lower. At equal Lab retention, interpolation gives up 0.17 Field Dev macro-F1 relative to Mixed. On macro-F1, **rehearsal reaches an operating point that post-hoc interpolation does not**. On External v3 accuracy, however, the retention-matched interpolation is slightly higher (0.447 vs 0.434), because it stays closer to the Lab baseline's bias towards the largest class. The comparison is therefore metric-dependent, and α = 1 (Field-only) adds no new operating point.


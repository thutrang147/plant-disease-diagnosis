# Research results summary

## Objective and design

This project compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato classification under laboratory-to-field domain shift. The experiments test augmentation, supervised field adaptation, mixed-domain fine-tuning, external transfer and sensitivity to crop framing. Different pretraining recipes and model-specific transforms mean the comparison does not isolate architecture alone. Notebooks 01–14 retain the historical experimental record. Notebook 15 adds the completed three-seed controlled ResNet18 comparison without revising those earlier observations.

The study follows four connected questions. **RQ1 — Laboratory-to-Field Generalization** establishes the transfer gap (Notebooks 01–04 and 09). **RQ2 — Domain Adaptation and Class-Specific Trade-offs** examines augmentation, supervised adaptation and mixing, then investigates their errors (05–12). **RQ3 — Image Framing and Cropping Sensitivity** examines Healthy parent/crop pairs and Bacterial Spot inference views (13–14). **RQ4 — Controlled Mixed-Domain Comparison** tests laboratory rehearsal under matched ResNet18 conditions (15). RQ3 investigates a plausible sensitivity without identifying the cause of RQ2's errors; RQ4 tests a training comparison that the historical runs could not isolate.

## Evaluation protocols and units of analysis

Training uses 5,365 PlantVillage Lab Train and 370 PlantDoc Field Adapt Train images. The 14 quarantined PlantDoc training rows are excluded from adaptation and development. The evaluation roles below apply throughout the study; fixed membership alone does not imply a new independent test.

| Partition / analysis | Size | Role and unit | Interpretation limit |
| --- | ---: | --- | --- |
| Lab Validation | 1,341 images | Laboratory checkpoint selection; source-retention monitoring during adaptation | Selection scores, not held-out laboratory test performance |
| Lab Test | 1,692 images | Fixed laboratory evaluation after checkpoint selection | Source-domain performance does not estimate field performance |
| Field Dev | 92 images | Target-domain checkpoint selection and exploratory diagnosis | Development scores are subject to selection optimism |
| Historical Field Test | 47 images | Previously inspected PlantDoc evaluation; reused for the controlled comparison | Not fresh confirmation; repeated seeds reuse the same images |
| External v1 | 62 images | 16 self-collected Healthy + 46 online disease images; canonical evaluation in Notebook 11 | Small, source/class-confounded collection; reused exploratorily in Notebook 14 |
| Healthy v2 primary analysis | 57 images | 8 originals + 7 historical crops + 42 newly approved crops; Healthy recognition | Correlated selected observations; recall does not measure five-class discrimination |
| Healthy paired extension | 23 parents / 49 crops | Equal-parent original–crop comparison across 21 recorded scene groups | Shared photograph/scene dependence; not independent plant trials |
| External v3 (PlantSeg) | 486 images | Locked, pre-registered, one-shot disease-only evaluation; duplicate-screened against all PlantDoc and PlantVillage tomato rows | Web-collected; no Healthy images; PlantSeg labels not audited here |

The complete v2 manifest contains 103 images, including 46 unchanged online disease images. Notebook 13's Healthy-only analysis uses 57 of those; the paired extension uses all 49 crop records and their 23 corresponding parent photographs. Counts of crops, photographs and recorded scene groups do not establish numbers of independent physical plants.

Labels are Healthy=0, Early Blight=1, Late Blight=2, Bacterial Spot=3 and Septoria Leaf Spot=4. Macro-F1 averages the five labels. Healthy-only results are recognition/recall, not a five-class discrimination estimate. Source attributions and human annotation decisions are preserved in the manifests.

## RQ1–RQ2: laboratory generalization, adaptation and external transfer

| Model | Lab Test Acc / F1 | Field Dev Acc / F1 | Field Test Acc / F1 | External Acc / F1 |
| --- | --- | --- | --- | --- |
| ResNet18 · Lab baseline | 0.9793 / 0.9744 | 0.3261 / 0.2355 | 0.3617 / 0.2802 | 0.3065 / 0.2406 |
| ViT-Small · Lab baseline | 0.9947 / 0.9938 | 0.4022 / 0.2677 | 0.4468 / 0.3617 | 0.4355 / 0.4157 |
| ResNet18 · Geometric + Mixed | 0.9297 / 0.9140 | 0.4674 / 0.4459 | 0.4043 / 0.3587 | 0.5323 / 0.4882 |
| ViT-Small · Mixed | 0.9900 / 0.9886 | 0.4022 / 0.3872 | 0.5319 / 0.5103 | 0.5161 / 0.4808 |

Lab Test has 1,692 images, Field Dev 92, historical Field Test 47 and external v1 62. Results use the fixed checkpoints and verified local outputs: Notebook 09 for Lab Test and Field Dev, the saved Field Test evaluation, and Notebook 11 for external v1. External inference applies EXIF orientation before RGB conversion; PlantDoc follows the established RGB loader. Architecture-specific normalization and standard transforms remain fixed across the baseline evaluations.

### RQ1: the laboratory-to-field gap

Laboratory baselines perform strongly on Lab Test but degrade sharply on PlantDoc. The fixed-model table separates this baseline transfer gap from the later adapted configurations. Changes in acquisition conditions, class composition and annotation may contribute, but this comparison does not isolate their effects.

### RQ2: adaptation and class-specific trade-offs

ResNet geometric + mixed has the highest Field Dev macro-F1 of the four fixed checkpoints (0.4459), while ViT Mixed retains higher Lab Test macro-F1 (0.9886 versus 0.9140). Field Dev selected adaptation checkpoints, so this describes development performance. Historical Field Test favors ViT Mixed among these four configurations (25/47 correct, macro-F1 0.5103), while external v1 has ResNet Mixed at 33/62, macro-F1 0.4882 and ViT Mixed at 32/62, macro-F1 0.4808. Such small, heterogeneous samples do not support general CNN-versus-ViT superiority.

Notebooks 06–08 establish the adaptation path. ResNet standard field-only / geometric field-only / geometric mixed Field Dev macro-F1 values are 0.4119 / 0.4347 / 0.4459. ViT field-only / mixed values are 0.3838 / 0.3872; the difference is descriptive. Field-only adaptation sacrifices source retention more strongly, while mixed-domain sampling improves the trade-off for these runs. Mixed-domain training samples 185 field and 185 laboratory images per epoch, compared with 370 field images in field-only adaptation. This approximates the optimizer-step budget but halves field-image exposure; it is not a measured compute-matched comparison.

**External v1 must be read by stratum.** All 16 Healthy images are self-collected and all 46 disease images are online, so the pooled gain is driven by Healthy recognition. On the 46 disease images, accuracy is ResNet Lab 19, ResNet Mixed 21, ViT Lab 22, ViT Mixed 19 (paired McNemar p = 0.83 and 0.58; `data/results/supplementary_statistics/`). External class trade-offs remain material. ResNet Lab recognizes 11/13 Early Blight images but ResNet Mixed only 1/13, while Healthy changes from 0/16 to 12/16 and Bacterial Spot from 1/9 to 5/9. ViT Lab versus Mixed recognizes 3/13 versus 2/13 Early Blight, 5/16 versus 13/16 Healthy, and 2/9 versus 5/9 Bacterial Spot. Gains in one class should not hide degradation in another. The ResNet Lab Early Blight recall is not a strength: the model predicts Early Blight for 39 of 62 external images (precision 0.28) and 55 of 92 Field Dev images, while ResNet Mixed almost never predicts it.

For external ResNet Mixed versus ResNet Lab, exact paired McNemar gives p=0.0243 (24 mixed-only correct, 10 baseline-only, 9 both correct, 19 both wrong). This is an unadjusted comparison on the selected sample. After Holm adjustment across the eight reported paired tests it is not significant (adjusted p = 0.24), and it disappears on the disease stratum. The share of errors made at maximum softmax confidence ≥0.90 decreases from **27/43=62.8%** for ResNet Lab to **0/29=0%** for ResNet Mixed. This denominator is all errors, rather than all high-confidence predictions. Softmax values are not calibrated correctness probabilities.

Notebook 10 complements these quantitative comparisons with predicted-class ResNet18 Grad-CAM and final-block ViT attention. The saved cases permit qualitative inspection of model responses; the two visualization mechanisms are different, and neither measures verified lesion localization. No completed expert adjudication or lesion annotations support a causal explanation of the class trade-offs.

**Frozen foundation-model reference (exploratory).** A frozen DINOv2 ViT-B/14 with only a logistic-regression head (`scripts/evaluate_frozen_foundation_probe.py`) reaches the following:
- Trained on Lab only: Lab Test 0.976, Field Dev 0.399 and Field Test 0.336 macro-F1, External v3 accuracy 0.473.
- Trained on Lab + Field Adapt Train (370 images; C chosen on Field Dev): 0.961, 0.622 and 0.665 (31/47), External v3 accuracy 0.469 (disease macro-F1 0.456).

The best project models reach 0.446 Field Dev, 0.510 Field Test and 0.444 / 0.441 on External v3. The frozen representation therefore matters more than the adaptation recipes compared here. Its large PlantDoc gain shrinks to about 2–3 pp on External v3, inside the interval width. This was not part of the pre-registered External v3 comparisons.

**External v3 confirms the disease-stratum result.** External v3 contains 486 PlantSeg images (Early Blight 151, Late Blight 148, Bacterial Spot 97, Septoria 90). They were screened against all PlantDoc and PlantVillage tomato rows: 94 of 589 candidates were exact or near duplicates, 74 of them rows used here. The set was evaluated once under a protocol committed beforehand (`data/external_v3_plantseg/protocol.json`). Accuracy is 207, 209, 206 and 216 of 486 for ResNet Lab, ViT Lab, ResNet Geometric + Mixed and ViT Mixed (Wilson intervals about ±4.4 pp). The paired Lab-versus-Mixed comparisons give McNemar p = 1.0 (ResNet) and 0.49 (ViT). The accuracy intervals exclude gains above about 5 pp (ResNet −0.2 pp, 95% CI −5.3 to +5.1; ViT +1.4 pp, −2.1 to +4.9), and the same holds when Healthy predictions are excluded, for images of at least 224 px, and on PlantSeg's test split. The pre-registered secondary metric, disease macro-F1, does improve (ResNet +0.086, 0.039 to 0.133; ViT +0.065, 0.029 to 0.102; post-hoc intervals in `data/results/external_v3_secondary_metrics/`). Adaptation therefore reshapes errors more than it changes their number. The ResNet baseline predicts Early Blight for 300 of 486 images, the same 62% as on External v1, with Bacterial Spot recall 0.00. Disease macro-F1 rises from 0.333 to 0.419 (ResNet) and from 0.375 to 0.441 (ViT).

**Lesion reliance (exploratory).** PlantSeg's expert-reviewed lesion masks allow the Notebook 10 question to be measured. Inpainting the lesions removes 38–52% of correct predictions, so lesion pixels are used. Images without visible lesions are rarely predicted Healthy (0–8%; ViT Mixed 18%), so the adapted Healthy class is not equivalent to "no lesion". ResNet18 Grad-CAM peaks fall inside the lesion tolerance region for 0.58–0.66 of images against a chance level of 0.46, and CAM mass inside lesions is about 1.25 times their area share. Adaptation does not improve localization, and a centre-point baseline scores higher (0.74), so Grad-CAM localizes lesions above chance but no better than a positional prior. A control that inpaints a same-area random non-lesion region changes accuracy by only −1.6 to +1.9 pp, so the lesion-specific drop is 0.12–0.15 for every model, with intervals excluding zero (378 images with room for the control; `data/results/occlusion_control/`). The models therefore rely on lesion pixels even though their saliency maps do not show where.

**Calibration and lesion size (External v3, saved predictions).** The Lab baselines are confidently wrong on field images: ResNet18 has mean confidence 0.90 at accuracy 0.43 (ECE 0.475), and ViT-Small 0.81 at 0.43 (ECE 0.384). The adapted models reach similar accuracy with far less over-confidence (ECE 0.16–0.29). Adapted models are somewhat more accurate on images with larger lesion areas. The ResNet Lab baseline is the exception (0.28 on the largest-lesion tertile), because it predicts Early Blight for most images (`data/results/external_v3_error_analysis/`).

**Architecture comparison under one recipe (exploratory).** Historical adaptation trained 75.1% of ResNet18 parameters but 8.2% of ViT-Small. With both models fully trainable under the same LP-FT recipe and batch size (3 seeds, CPU, Field-only), ViT-Small rises from 0.441 to 0.484 External v3 accuracy and keeps Lab Test macro-F1 at 0.984. ResNet18 falls slightly to 0.403 and forgets more Lab performance (0.786). The CNN-versus-ViT comparison is therefore recipe-dependent (`data/results/lpft_architecture/`).

**Label agreement.** 64 images appear in both PlantSeg and this repository's PlantDoc manifests (pHash ≤ 4), and 13 carry conflicting labels. Six of these conflicts are Bacterial Spot versus Septoria. Field Test row 2524, already suspected, is labelled Septoria by PlantSeg. The conflicts are listed for expert audit in `data/external_v3_plantseg/label_audit_candidates.csv`; no label has been changed.

## RQ3: image framing and cropping sensitivity

The error analyses motivate a closer examination of how image presentation relates to predictions. Human-reviewed Healthy crops and automatic Bacterial Spot inference views address different populations and selection processes; they are reported separately.

### Self-collected Healthy originals and reviewed crops

Notebook 13's primary 57-image results are ResNet Lab **0/57**, ViT Lab **15/57**, ResNet Mixed **44/57**, and ViT Mixed **43/57** recognized Healthy. These are correlated images/crops from selected Healthy sources, not independent plant trials or a disease discrimination test.

For the exploratory paired extension, the same 23 parent photographs are compared with their 49 accepted crops:

| Model | Original correct / 23 | Crop correct / 49 | Parent-macro crop recall | Change (pp) |
| --- | --- | --- | --- | --- |
| ResNet18 · Lab baseline | 0/23 | 0/49 | 0.0000 | 0.0 |
| ViT-Small · Lab baseline | 3/23 | 14/49 | 0.3529 | 22.2 |
| ResNet18 · Geometric + Mixed | 10/23 | 37/49 | 0.7558 | 32.1 |
| ViT-Small · Mixed | 14/23 | 37/49 | 0.8275 | 21.9 |

Parent-macro recall first averages crop correctness within each parent, then weights parents equally. The change is relative to equally weighted parent-original correctness. This distinguishes crop-count-weighted recall from parent-level comparisons. Mixed ResNet has 10 improved / 2 worsened / 11 equal parents; Mixed ViT has 9 / 3 / 11. Human-approved crop selection, occlusion and framing can contribute; these observations do not isolate lesion resolution.

### Automatic inference views and Bacterial Spot sensitivity

[Notebook 14](../notebooks/14_lesion_resolution_study.ipynb) compares full-scene, source-center 0.70, source-center 0.50 and square-448 five-crop strategies. The first three apply architecture-specific standard transforms after any source crop. The fourth bicubically resizes to a square 448 canvas, takes four corner and one center 224 crops, and averages softmax probabilities. Every model input is **224×224**. It changes coverage, object scale, aspect ratio and aggregation; it is not true 448-input inference or a lesion-localization experiment.

| Model / dataset | Standard Bacterial Spot | Center 0.70 | Center 0.50 | Five-crop |
| --- | ---: | ---: | ---: | ---: |
| ResNet Mixed / Field Dev | 3/18 | 3/18 | 2/18 | 2/18 |
| ViT Mixed / Field Dev | 3/18 | 4/18 | 5/18 | 3/18 |
| ResNet Mixed / external v1 | 5/9 | 4/9 | 5/9 | 5/9 |
| ViT Mixed / external v1 | 5/9 | 4/9 | 2/9 | 6/9 |

ViT Mixed five-crop external accuracy/macro-F1 is **0.6129 / 0.5840** (38/62), compared with standard **0.5161 / 0.4808** (32/62). ResNet Mixed Field Dev macro-F1 falls from 0.4459 to 0.3603 under center 0.50; removing useful peripheral content is plausible but unproven. The full four-model results, class trade-offs and linked local model-input gallery are in Notebook 14.

These policies were compared after observing external outcomes. Results are exploratory and do not select a deployment policy or provide a new independent test. Human-reviewed Healthy crops in Notebook 13 and automatic spatial interventions here have different selection mechanisms and should not be pooled.

## Controlled Field-only versus Mixed-domain Fine-tuning

**RQ4** asks whether laboratory rehearsal consistently improves ResNet18 field performance when initialization, Field exposure, optimizer updates and checkpoint selection are controlled ([Notebook 15](../notebooks/15_controlled_field_mixed_finetuning.ipynb)). Historical Notebooks 06–08 used unequal Field exposure, independent selection/stopping and, in the final geometric ResNet18 comparison, different initialization. Their Mixed advantage did not isolate laboratory rehearsal.

Both controlled conditions reload the same Lab baseline, use all 370 Field Adapt Train images per epoch in identical order with identical augmented tensors, and complete two head epochs plus six final-block epochs. Each condition receives **2,960 Field presentations and 192 updates per seed**; Mixed adds **2,960 Lab presentations**. AdamW, fixed phase learning rates, frozen BatchNorm statistics, transforms and the Field loss coefficient match. Mixed adds mean Lab cross-entropy with a fixed weight of 1.0, so information and compute remain unequal. Seeds 42, 43 and 44 were declared before training.

Checkpoint selection maximizes the pair's mean five-class Field Dev macro-F1, with the earliest tie and epoch zero eligible. Both conditions share the selected epoch within each pair. Saved validation output reports epochs **8, 7 and 8**: evaluated seed-43 checkpoints have **2,590 Field presentations and 168 updates per condition**, while the other evaluated pairs have 2,960 and 192. All three full runs completed the matched eight-epoch budget. The implementation validates exposure and selection before sealing checkpoint hashes and accessing the historical 47-image Field Test; test data do not select the checkpoints.

| Seed | Field-only correct / 47 | Mixed correct / 47 | Field-only macro-F1 | Mixed macro-F1 | Mixed − Field-only macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 22/47 | 21/47 | 0.4338 | 0.3970 | −0.0368 |
| 43 | 21/47 | 20/47 | 0.4214 | 0.3840 | −0.0373 |
| 44 | 22/47 | 22/47 | 0.4349 | 0.4405 | +0.0056 |

| Condition | Accuracy mean ± sample SD | Macro-F1 mean ± sample SD |
| --- | ---: | ---: |
| Field-only | 0.4610 ± 0.0123 | 0.4300 ± 0.0075 |
| Mixed | 0.4468 ± 0.0213 | 0.4072 ± 0.0296 |
| Paired Mixed − Field-only | −0.0142 ± 0.0123 | −0.0228 ± 0.0246 |

Differences and sample SD (`ddof=1`) are calculated before rounding. Accuracy favors Field-only in two seeds and ties in one; macro-F1 favors Field-only in two and Mixed in one. The mean accuracy change is −1.42 percentage points. Three seeds reuse the same test images, so these SDs describe seed variation rather than independent dataset uncertainty. No statistical significance is claimed.

| Class (images per run) | Field-only correct, seeds 42 / 43 / 44 | Mixed correct, seeds 42 / 43 / 44 | Field-only mean recall | Mixed mean recall |
| --- | --- | --- | ---: | ---: |
| Healthy (8) | 3 / 3 / 3 | 2 / 2 / 3 | 0.3750 | 0.2917 |
| Early Blight (9) | 1 / 2 / 2 | 2 / 2 / 3 | 0.1852 | 0.2593 |
| Late Blight (10) | 8 / 7 / 8 | 8 / 7 / 7 | 0.7667 | 0.7333 |
| Bacterial Spot (9) | 2 / 1 / 1 | 1 / 1 / 1 | 0.1481 | 0.1111 |
| Septoria Leaf Spot (11) | 8 / 8 / 8 | 8 / 8 / 8 | 0.7273 | 0.7273 |

Mixed modestly improves Early Blight recall in two seeds and ties in one, but reduces Healthy recall in two and Bacterial Spot recall in one, tying the others. Bacterial Spot is predominantly assigned Septoria Leaf Spot: 7/9 images in Field-only seed 42 and 8/9 in every other run. The external Healthy gains and Early Blight losses from earlier notebooks concern different comparisons and samples; the controlled Field Test does not reproduce that same trade-off.

**Laboratory retention.** The original run did not evaluate either Lab partition, and neither partition took part in Notebook 15 training or checkpoint selection (Mixed samples Lab Train only). After the run, `scripts/evaluate_notebook15_lab_retention.py` evaluated the six sealed checkpoints and their shared Lab baseline on Lab Validation (1,341) and Lab Test (1,692). The evaluation was inference only, on CPU in float32, and every checkpoint SHA-256 was matched against the seal before inference. The baseline reproduces its earlier Lab Test result exactly (1,657/1,692, macro-F1 0.9744). Images come from `geraldmc/plantvillage-full` revision `f808f27`, and outputs are in [`data/results/notebook15_lab_retention/`](../data/results/notebook15_lab_retention/).

| Seed | Lab Val macro-F1, Field-only / Mixed | Lab Test correct, Field-only / Mixed | Lab Test macro-F1, Field-only / Mixed | Mixed − Field-only Lab Test macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Lab baseline (shared initialization) | 0.9879 | 1,657/1,692 | 0.9744 | — |
| 42 | 0.8441 / 0.9733 | 1,490 / 1,634 | 0.8116 / 0.9553 | +0.1436 |
| 43 | 0.8341 / 0.9706 | 1,482 / 1,622 | 0.8092 / 0.9449 | +0.1357 |
| 44 | 0.8533 / 0.9780 | 1,499 / 1,641 | 0.8199 / 0.9603 | +0.1404 |

| Condition | Lab Val macro-F1 mean ± sample SD | Lab Test macro-F1 mean ± sample SD | Lab Test change from baseline |
| --- | ---: | ---: | ---: |
| Field-only | 0.8438 ± 0.0096 | 0.8136 ± 0.0056 | −0.1608 |
| Mixed | 0.9740 ± 0.0038 | 0.9535 ± 0.0079 | −0.0209 |

Mixed retains more laboratory performance in every seed, by +0.130 (Lab Validation) and +0.140 (Lab Test) seed-mean macro-F1, more than ten times the seed-to-seed SD. As with the Field Test, the seeds reuse the same images, so the SDs describe training-run variation rather than dataset uncertainty. Lab Validation also selected the baseline checkpoint, so its baseline value is optimistic, and Lab Test is the cleaner comparison. The retention gain is the expected effect of rehearsal on source data; the controlled design shows it is obtained without a detectable field-performance cost.

**External v3 and weight interpolation.** On the 486 locked External v3 images (pre-declared comparison P3), seed-mean accuracy is 0.434 for Mixed and 0.417 for Field-only. The difference is +1.7 pp (paired bootstrap 95% CI [−0.1, +3.6]; sign-flip p = 0.075, Holm 0.23), and disease macro-F1 is 0.420 vs 0.405. Mixed is higher in accuracy in all three seeds (+2.9, +0.6, +1.6 pp; seed-level t-test p = 0.12); disease macro-F1 differs by +0.014 (95% CI −0.003 to +0.032). Both arms were compared at an 8-epoch budget where ResNet18 Field Dev had not plateaued. The direction is opposite to the historical Field Test, and both intervals include zero, so a small benefit for Mixed is neither shown nor excluded.

A post-hoc alternative to rehearsal was also tested: linear weight interpolation between the Lab baseline and each Field-only checkpoint (`scripts/evaluate_weight_interpolation.py`). The curve was computed on Lab Validation and Field Dev only, and the operating points were chosen by pre-declared rules. The endpoints reproduce the sealed values. Interpolation traces a monotone trade-off. At Lab retention equal to Mixed (α = 0.4, Lab Test 0.951 vs 0.954), Field Dev macro-F1 is 0.266 against 0.440 for Mixed, Field Test 0.339 against 0.407, and External v3 disease macro-F1 0.375 against 0.420. The Field-Dev-selected point is α = 1 (Field-only). On macro-F1, rehearsal therefore reaches an operating point that interpolation does not; on External v3 accuracy the retention-matched interpolation is slightly higher (0.447 vs 0.434), so the comparison is metric-dependent.

**24-epoch re-run (exploratory).** Both arms were re-run on CPU for 24 epochs with the same recipe, seeds and Field-Dev selection (`scripts/explore_mixed_budget.py`; unsealed). At the Notebook 15 selected epoch every run reproduces its sealed Lab Test macro-F1 within 0.001; Field Dev matches in 5 of 6 runs (Mixed s44 diverges, 0.418 vs the sealed 0.433, which re-scoring the sealed checkpoint reproduces). Mixed keeps a Lab Test macro-F1 advantage of +0.113 to +0.125 in every seed (seed-level p = 0.001). Field Dev favours Mixed in all seeds by 1–2 correct images out of 92 (selected-epoch p = 0.19; epoch 20–24 mean p = 0.09); Field Test differences are +0.028, −0.017 and 0.000 macro-F1 (inconclusive). On External v3, a pre-declared secondary endpoint on an already reused set, Mixed is +2.4 pp higher (paired bootstrap 95% CI +0.7 to +4.1; sign-flip p = 0.008; seed-level p = 0.06), and training longer raises both arms (Field-only 0.417 → 0.463, Mixed 0.434 → 0.487). Results: `data/results/mixed_budget_summary/`.

**Conclusion:** under this fixed ResNet18 recipe at the 8-epoch budget, no field-performance difference between Mixed and Field-only fine-tuning was **detected** (equivalence was not tested), while Mixed **largely prevents laboratory forgetting**. The seed-mean Field Test difference (Mixed − Field-only) is −0.023 macro-F1 with a paired image-level bootstrap 95% CI of [−0.086, +0.024]. It rests on a net two images, and at the shared selected epoch Field Dev favours Mixed in all three seeds (+0.0002, +0.0117, +0.0180). On Lab Test, Mixed loses 0.021 macro-F1 from the shared initialization, against 0.161 for Field-only. Both results agree with the matched-initialization historical pair: A1 Field-only 0.4119 vs R1 Mixed 0.3523 Field Dev macro-F1, where the only historical Mixed benefit was Lab retention (Lab Validation 0.8185 vs 0.9556). Notebook 15 reproduces that retention benefit with initialization, Field exposure and updates controlled. The earlier apparent Mixed advantage on field data combined geometric initialization with selection of the best of six configurations on the same Field Dev. This does not establish universal Mixed inferiority on field data or explain which historical difference caused the earlier result.

The small and previously inspected Field Test, shared images across three seeds, correlated observations, coupled selection, one fixed Lab-loss weight, additional Mixed compute/data and a single architecture/recipe constrain interpretation. Previously explored External v1/v2 and Healthy v2 were not used to retrospectively optimize the controlled run. No sealed ViT-Small extension was run; the exploratory CPU re-adaptations of both architectures (8 and 24 epochs) are reported separately and are not part of the sealed comparison.

## Overall Discussion

The laboratory results answer RQ1 by showing that strong performance on controlled PlantVillage imagery does not guarantee transfer to the sampled PlantDoc photographs. This is a descriptive generalization gap between differently acquired and annotated populations. Background, illumination, symptom presentation and label uncertainty are plausible contributors, but their individual effects were not experimentally separated. The difference between ResNet18 and ViT-Small likewise reflects their full pretrained configurations, rather than architecture alone.

RQ2 shows why adaptation must be assessed beyond aggregate accuracy. Selected adapted configurations improve target-domain macro-F1 or external accuracy while redistributing errors across classes. The historical ResNet18 external comparison is particularly clear: improved Healthy recognition coexists with severe Early Blight degradation. These counts establish a class-specific trade-off on the evaluated images. They do not establish that the models learned a particular symptom or background cue. Grad-CAM and ViT attention provide qualitative diagnostic views, with no verified lesion-localization or causal mechanism claim.

RQ3 shows sensitivity to image framing under both human-reviewed crops and automatic inference views. Parent-balanced Healthy summaries prevent photographs with more crops from dominating the mean, but cannot remove crop-selection bias or scene dependence. Automatic center crops can enlarge a target while removing context; five-crop inference changes coverage, scale, aspect ratio and probability aggregation together. All five-crop model inputs remain 224×224. Consequently, an observed gain is evidence about the tested view policy, not an isolated recovery of lesion detail or a higher model-input-resolution effect.

The historical and controlled Mixed results answer different comparisons. Historical runs describe source-retention and target-performance trade-offs with unequal Field exposure, independent stopping/selection and, for the final ResNet18 baseline-to-Mixed contrast, different initialization. RQ4 holds initialization, Field exposure, updates and the paired selection rule fixed. It finds no consistent Mixed advantage over Field-only on field data, and a consistent laboratory-retention advantage (Lab Test macro-F1 0.9535 vs 0.8136, same sign in all three seeds). The controlled result therefore limits the interpretation of the earlier apparent field advantage without contradicting its measured scores, and it reproduces the historical retention benefit under matched conditions. It cannot identify which historical difference produced the earlier field scores or establish universal Mixed inferiority on field data.

The absence of a new independent Field Test limits population-level confirmation, not the descriptive comparisons themselves. The saved evidence supports a laboratory-to-field transfer gap, uneven class trade-offs, sensitivity to crop/view policies, and the recipe-specific controlled result. It does not support a deployment-performance estimate, a general CNN-versus-transformer ranking, or causal claims about visual failure mechanisms. Repeated seeds characterize variation under the same test images rather than supplying independent test datasets.

## Scientific limitations

- Field Dev participates in checkpoint selection; the 47-image test was inspected historically and external v1 was reused for sensitivity analysis.
- Class denominators are small; source, class and acquisition conditions are entangled. Crops and parent photographs are correlated, and scene identifiers do not count independent plants.
- Exact/pHash screening of the self-collected and online external images found no matches with the 523 PlantDoc rows used here. The PlantSeg screen for External v3 did find overlap: 94 of 589 candidates (74 matching rows used in this repository) were excluded. Whole-image hashing cannot detect crops, flips or re-posted variants, and cannot rule out overlap with pretraining data.
- Historical model recipes are single-seed and differ in pretraining. Notebook 15 repeats one controlled ResNet18 recipe across three seeds sharing one test set. Class trade-offs must accompany aggregate comparisons.
- Historical Field-only and Mixed training have unequal Field exposure; Notebook 15 matches exposure and updates but retains additional Lab information/compute and coupled checkpoint selection.
- PlantDoc has no lesion annotations. On External v3, PlantSeg masks quantify ResNet18 Grad-CAM localization and the effect of image edits, but both lesion removal and lesion-only views lower accuracy, so edits are not clean causal manipulations. No controlled object sizes establish loss or recovery of lesion information. Prospective confirmation needs a locked protocol and newly collected grouped photographs.
- External v3 covers four disease classes only (no Healthy specificity), relies on unaudited PlantSeg labels (13 of 64 overlapping images disagree with PlantDoc), and only comparisons P1–P3 were pre-registered; later analyses on it are exploratory.
- The historical adaptation runs and Notebook 15 stopped at 8 epochs; a 24-epoch ablation shows ResNet18 still improving, so those comparisons describe under-trained models.
- Third-party photographs retain their source attributions and require appropriate usage rights; image files are excluded from the public source distribution.

## Evidence and notebook organization

Notebooks 01–10 establish the training, domain-shift, adaptation and error-analysis progression. [11](../notebooks/11_external_evaluation.ipynb) records canonical external evaluation; [12](../notebooks/12_external_error_analysis.ipynb) analyzes those errors; [13](../notebooks/13_healthy_field_evaluation.ipynb) retains Healthy-only and parent/crop evaluation; [14](../notebooks/14_lesion_resolution_study.ipynb) provides the focused sensitivity investigation. [15](../notebooks/15_controlled_field_mixed_finetuning.ipynb) reports the completed controlled ResNet18 experiment; its original Colab execution is archived with a byte-level provenance hash.

Evidence includes the [unaltered executed Colab notebook](../notebooks/archive/15_controlled_field_mixed_colab_executed.ipynb), [original prediction exports and provenance](../data/results/notebook15/), and the unchanged [training implementation](../src/controlled_field_mixed.py). The full Drive run has been imported locally. A [read-only artifact audit](../data/results/notebook15/imported_artifact_audit.json) confirms all six aggregate results and 30 class rows directly from original prediction CSVs, canonical source/image identities and labels, probabilities/correctness, all batch Field IDs and tensor hashes, per-row exposure, update budgets, full-precision common-epoch selection, and six selected checkpoint hashes against completion records and the evaluation seal. The recorded identical initial state also matches the local Lab baseline. All 53 imported file hashes remain unchanged. These checks establish consistency of the saved evidence, not proof against undocumented human inspection; the scientific sampling limitations remain. No training or inference was repeated.

Primary evidence (links other than the Notebook 09 summary point to local outputs that are not distributed in Git): [Notebook 09 summary](../outputs/final_comparison_development/final_comparison_summary.csv), external evaluation outputs (`outputs/external_evaluation/tomato_field_eval_v1/20261125_163347_4ca8183f/`, local output not distributed in Git), Healthy primary run (`outputs/healthy_field_evaluation/tomato_field_eval_v2/20261126_103724_5c4da7f2/`, local output not distributed in Git), Healthy paired run (`outputs/healthy_field_evaluation/tomato_field_eval_v2/paired_20261126_105328_eddcf985/`, local output not distributed in Git), and crop-sensitivity outputs (`outputs/research_integration/20261126_reconciliation/inference/`, local output not distributed in Git). Local data, weights and ignored outputs are needed to rerun the notebooks; they are not a self-contained public distribution.

## Research contributions and conclusion

The project contributes three connected results:

1. A traceable laboratory-to-field evaluation that separates validation, development, historical test and external evidence, and retains unfavorable class-level outcomes.
2. An analysis of adaptation and framing that distinguishes aggregate gains, source retention and correlated parent/crop observations from unverified visual mechanisms.
3. A controlled, three-seed ResNet18 comparison showing that laboratory rehearsal does not consistently improve field performance under the tested matched protocol, but consistently preserves laboratory performance (Lab Test macro-F1 0.9535 vs 0.8136).

Together, these findings make the study's central conclusion precise: laboratory accuracy, adaptation gains and cropping gains each depend on the evaluation population and protocol. The completed evidence supports those bounded conclusions while leaving prospective field confirmation and causal lesion analysis unresolved.

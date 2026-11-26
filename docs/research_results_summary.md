# Research results summary

## Objective and design

This project compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato classification under laboratory-to-field domain shift. The experiments test augmentation, supervised field adaptation, mixed-domain fine-tuning, external transfer and sensitivity to crop framing. Different pretraining recipes and model-specific transforms mean the comparison does not isolate architecture alone. Notebooks 01–14 retain the historical experimental record. Notebook 15 adds the completed three-seed controlled ResNet18 comparison without revising those earlier observations.

## Datasets and evaluation units

| Partition | Images | Interpretation |
| --- | ---: | --- |
| PlantVillage Lab Train / Validation / Test | 5,365 / 1,341 / 1,692 | Laboratory training, checkpoint selection and fixed evaluation |
| PlantDoc adaptation train / Field Dev | 370 / 92 | Supervised adaptation and target-domain checkpoint selection |
| Quarantined PlantDoc training rows | 14 | Excluded from adaptation and development |
| Historical PlantDoc Field Test | 47 | Previously inspected; fixed additional evaluation, not fresh confirmation |
| External v1 | 62 | 16 self-collected Healthy + 46 online disease; source/class confounded |
| Healthy v2 primary analysis | 57 | 8 original photographs + 7 historical crops + 42 newly approved crops |
| Healthy paired exploratory analysis | 23 parents / 49 crops | 21 recorded scene groups; includes additional parent originals |

The complete v2 manifest contains 103 images, including 46 unchanged online disease images. Notebook 13's Healthy-only analysis uses 57 of those; the paired extension uses all 49 crop records and their 23 corresponding parent photographs. Counts of crops, photographs and recorded scene groups do not establish numbers of independent physical plants.

Labels are Healthy=0, Early Blight=1, Late Blight=2, Bacterial Spot=3 and Septoria Leaf Spot=4. Macro-F1 averages the five labels. Healthy-only results are recognition/recall, not a five-class discrimination estimate. Source attributions and human annotation decisions are preserved in the manifests.

## Experiments 1–3: laboratory performance, adaptation and external transfer

| Model | Lab Test Acc / F1 | Field Dev Acc / F1 | Field Test Acc / F1 | External Acc / F1 |
| --- | --- | --- | --- | --- |
| ResNet18 · Lab baseline | 0.9793 / 0.9744 | 0.3261 / 0.2355 | 0.3617 / 0.2802 | 0.3065 / 0.2406 |
| ViT-Small · Lab baseline | 0.9947 / 0.9938 | 0.4022 / 0.2677 | 0.4468 / 0.3617 | 0.4355 / 0.4157 |
| ResNet18 · Geometric + Mixed | 0.9297 / 0.9140 | 0.4674 / 0.4459 | 0.4043 / 0.3587 | 0.5323 / 0.4882 |
| ViT-Small · Mixed | 0.9900 / 0.9886 | 0.4022 / 0.3872 | 0.5319 / 0.5103 | 0.5161 / 0.4808 |

Lab Test has 1,692 images, Field Dev 92, historical Field Test 47 and external v1 62. Results use the fixed checkpoints and verified local outputs: Notebook 09 for Lab Test and Field Dev, the saved Field Test evaluation, and Notebook 11 for external v1. External inference applies EXIF orientation before RGB conversion; PlantDoc follows the established RGB loader. Architecture-specific normalization and standard transforms remain fixed across the baseline evaluations.

Laboratory baselines perform strongly on Lab Test but degrade sharply on PlantDoc. ResNet geometric + mixed has the highest Field Dev macro-F1 of the four fixed checkpoints (0.4459), while ViT Mixed retains higher Lab Test macro-F1 (0.9886 versus 0.9140). Field Dev selected adaptation checkpoints, so this describes development performance. Historical Field Test favors ViT Mixed among these four configurations (25/47 correct, macro-F1 0.5103), while external v1 has ResNet Mixed at 33/62, macro-F1 0.4882 and ViT Mixed at 32/62, macro-F1 0.4808. Such small, heterogeneous samples do not support general CNN-versus-ViT superiority.

Notebooks 06–08 establish the adaptation path. ResNet standard field-only / geometric field-only / geometric mixed Field Dev macro-F1 values are 0.4119 / 0.4347 / 0.4459. ViT field-only / mixed values are 0.3838 / 0.3872; the difference is descriptive. Field-only adaptation sacrifices source retention more strongly, while mixed-domain sampling improves the trade-off for these runs. Mixed-domain training samples 185 field and 185 laboratory images per epoch, compared with 370 field images in field-only adaptation. This approximates the optimizer-step budget but halves field-image exposure; it is not a measured compute-matched comparison.

External class trade-offs remain material. ResNet Lab recognizes 11/13 Early Blight images but ResNet Mixed only 1/13, while Healthy changes from 0/16 to 12/16 and Bacterial Spot from 1/9 to 5/9. ViT Lab versus Mixed recognizes 3/13 versus 2/13 Early Blight, 5/16 versus 13/16 Healthy, and 2/9 versus 5/9 Bacterial Spot. Gains in one class should not hide degradation in another.

For external ResNet Mixed versus ResNet Lab, exact paired McNemar gives p=0.0243 (24 mixed-only correct, 10 baseline-only, 9 both correct, 19 both wrong). This is an unadjusted comparison on the selected sample. The share of errors made at maximum softmax confidence ≥0.90 decreases from **27/43=62.8%** for ResNet Lab to **0/29=0%** for ResNet Mixed. This denominator is all errors, rather than all high-confidence predictions. Softmax values are not calibrated correctness probabilities.

## Experiment 4: self-collected Healthy originals and reviewed crops

Notebook 13's primary 57-image results are ResNet Lab **0/57**, ViT Lab **15/57**, ResNet Mixed **44/57**, and ViT Mixed **43/57** recognized Healthy. These are correlated images/crops from selected Healthy sources, not independent plant trials or a disease discrimination test.

For the exploratory paired extension, the same 23 parent photographs are compared with their 49 accepted crops:

| Model | Original correct / 23 | Crop correct / 49 | Parent-macro crop recall | Change (pp) |
| --- | --- | --- | --- | --- |
| ResNet18 · Lab baseline | 0/23 | 0/49 | 0.0000 | 0.0 |
| ViT-Small · Lab baseline | 3/23 | 14/49 | 0.3529 | 22.2 |
| ResNet18 · Geometric + Mixed | 10/23 | 37/49 | 0.7558 | 32.1 |
| ViT-Small · Mixed | 14/23 | 37/49 | 0.8275 | 21.9 |

Parent-macro recall first averages crop correctness within each parent, then weights parents equally. The change is relative to equally weighted parent-original correctness. This distinguishes crop-count-weighted recall from parent-level comparisons. Mixed ResNet has 10 improved / 2 worsened / 11 equal parents; Mixed ViT has 9 / 3 / 11. Human-approved crop selection, occlusion and framing can contribute; these observations do not isolate lesion resolution.

## Experiment 5: resolution and cropping sensitivity

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

[Notebook 15](../notebooks/15_controlled_field_mixed_finetuning.ipynb) tests whether Lab rehearsal helps ResNet18 when Field exposure and optimizer updates match. Historical Notebooks 06–08 used unequal Field exposure, independent selection/stopping and, in the final geometric ResNet18 comparison, different initialization. Their Mixed advantage did not isolate laboratory rehearsal.

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

**Conclusion:** Field-only has higher mean accuracy and macro-F1 under this fixed ResNet18 recipe. The historical Mixed advantage is not consistently reproduced after matching initialization, Field exposure and update counts. This does not establish universal Mixed inferiority, explain which historical difference caused the earlier result, or establish a laboratory-retention benefit for the new checkpoints; Lab retention was not evaluated here.

The small and previously inspected Field Test, shared images across three seeds, correlated observations, coupled selection, one fixed Lab-loss weight, additional Mixed compute/data and a single architecture/recipe constrain interpretation. Previously explored External v1/v2 and Healthy v2 were not used to retrospectively optimize the controlled run. No ViT-Small extension or new training was performed during finalization.

Evidence includes the [unaltered executed Colab notebook](../notebooks/archive/15_controlled_field_mixed_colab_executed.ipynb), [original prediction exports and provenance](../data/results/notebook15/), and the unchanged [training implementation](../src/controlled_field_mixed.py). The full Drive run has been imported locally. A [read-only artifact audit](../data/results/notebook15/imported_artifact_audit.json) confirms all six aggregate results and 30 class rows directly from original prediction CSVs, canonical source/image identities and labels, probabilities/correctness, all batch Field IDs and tensor hashes, per-row exposure, update budgets, full-precision common-epoch selection, and six selected checkpoint hashes against completion records and the evaluation seal. The recorded identical initial state also matches the local Lab baseline. All 53 imported file hashes remain unchanged. These checks establish consistency of the saved evidence, not proof against undocumented human inspection; the scientific sampling limitations remain. No training or inference was repeated.

## Scientific limitations

- Field Dev participates in checkpoint selection; the 47-image test was inspected historically and external v1 was reused for sensitivity analysis.
- Class denominators are small; source, class and acquisition conditions are entangled. Crops and parent photographs are correlated, and scene identifiers do not count independent plants.
- Exact/pHash screening across all 523 PlantDoc rows found no external matches within the tested rule, but cannot prove independence or eliminate overlap with untested pretraining sources.
- Historical model recipes are single-seed and differ in pretraining. Notebook 15 repeats one controlled ResNet18 recipe across three seeds sharing one test set. Class trade-offs must accompany aggregate comparisons.
- Historical Field-only and Mixed training have unequal Field exposure; Notebook 15 matches exposure and updates but retains additional Lab information/compute and coupled checkpoint selection.
- No lesion annotations, controlled object sizes or localization judgments establish loss/recovery of lesion information. Prospective confirmation needs a locked protocol and newly collected grouped photographs.
- Third-party photographs retain their source attributions and require appropriate usage rights; image files are excluded from the public source distribution.

## Evidence and notebook organization

Notebooks 01–10 establish the training, domain-shift, adaptation and error-analysis progression. [11](../notebooks/11_external_evaluation.ipynb) records canonical external evaluation; [12](../notebooks/12_external_error_analysis.ipynb) analyzes those errors; [13](../notebooks/13_healthy_field_evaluation.ipynb) retains Healthy-only and parent/crop evaluation; [14](../notebooks/14_lesion_resolution_study.ipynb) provides the focused sensitivity investigation. [15](../notebooks/15_controlled_field_mixed_finetuning.ipynb) reports the completed controlled ResNet18 experiment; its original Colab execution is archived with a byte-level provenance hash.

Primary evidence: [Notebook 09 summary](../outputs/final_comparison_development/final_comparison_summary.csv), [external evaluation outputs](../outputs/external_evaluation/tomato_field_eval_v1/20261125_163347_4ca8183f/), [Healthy primary run](../outputs/healthy_field_evaluation/tomato_field_eval_v2/20261126_103724_5c4da7f2/), [Healthy paired run](../outputs/healthy_field_evaluation/tomato_field_eval_v2/paired_20261126_105328_eddcf985/), and [crop-sensitivity outputs](../outputs/research_integration/20261126_reconciliation/inference/). Local data, weights and ignored outputs are needed to rerun the notebooks; they are not a self-contained public distribution.

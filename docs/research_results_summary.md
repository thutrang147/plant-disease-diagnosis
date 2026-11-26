# Research results summary

## Objective and design

This project compares ImageNet-pretrained ResNet18 and ViT-Small for five-class tomato classification under laboratory-to-field domain shift. The experiments test augmentation, supervised field adaptation, mixed-domain fine-tuning, external transfer and sensitivity to crop framing. Different pretraining recipes and model-specific transforms mean the comparison does not isolate architecture alone. The executed local Notebooks 01–14 and their verified outputs form the canonical experimental record.

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

## Scientific limitations

- Field Dev participates in checkpoint selection; the 47-image test was inspected historically and external v1 was reused for sensitivity analysis.
- Class denominators are small; source, class and acquisition conditions are entangled. Crops and parent photographs are correlated, and scene identifiers do not count independent plants.
- Exact/pHash screening across all 523 PlantDoc rows found no external matches within the tested rule, but cannot prove independence or eliminate overlap with untested pretraining sources.
- Single-seed model recipes and pretraining differ. Class trade-offs, especially Early Blight degradation, must accompany aggregate improvements.
- Field-only and mixed-domain training have unequal field-image exposure, limiting causal attribution of their differences.
- No lesion annotations, controlled object sizes or localization judgments establish loss/recovery of lesion information. Prospective confirmation needs a locked protocol and newly collected grouped photographs.
- Third-party photographs retain their source attributions and require appropriate usage rights; image files are excluded from the public source distribution.

## Evidence and notebook organization

Notebooks 01–10 establish the training, domain-shift, adaptation and error-analysis progression. [11](../notebooks/11_external_evaluation.ipynb) records canonical external evaluation; [12](../notebooks/12_external_error_analysis.ipynb) analyzes those errors; [13](../notebooks/13_healthy_field_evaluation.ipynb) retains Healthy-only and parent/crop evaluation; [14](../notebooks/14_lesion_resolution_study.ipynb) provides the focused sensitivity investigation.

Primary evidence: [Notebook 09 summary](../outputs/final_comparison_development/final_comparison_summary.csv), [external evaluation outputs](../outputs/external_evaluation/tomato_field_eval_v1/20261125_163347_4ca8183f/), [Healthy primary run](../outputs/healthy_field_evaluation/tomato_field_eval_v2/20261126_103724_5c4da7f2/), [Healthy paired run](../outputs/healthy_field_evaluation/tomato_field_eval_v2/paired_20261126_105328_eddcf985/), and [crop-sensitivity outputs](../outputs/research_integration/20261126_reconciliation/inference/). Local data, weights and ignored outputs are needed to rerun the notebooks; they are not a self-contained public distribution.

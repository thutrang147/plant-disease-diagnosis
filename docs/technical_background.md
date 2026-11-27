# Technical background: the computer-vision ideas behind each design decision

This page explains *why* each technique in the project was chosen, what it assumes, and where its effect is
measured. Every section ends with the files that implement or test it.

## 1. Transfer learning and what fine-tuning changes

Both backbones start from ImageNet-1K weights. Low-level filters (edges, colour blobs, textures) transfer
across domains; higher layers encode increasingly task- and dataset-specific features. Fine-tuning only the
classification head keeps the pretrained features fixed. Unfreezing deeper blocks lets the representation adapt,
but it can also **distort** features that were useful under distribution shift: on 10 shift benchmarks, full
fine-tuning beat a linear probe by about 2% in-distribution but lost about 7% out-of-distribution (Kumar et
al., 2022).

LP-FT mitigates this. A linear probe is trained first, so the head is already sensible when the backbone is
unfrozen, and early gradients do not destroy the features.

*In this project:*
- Adaptation runs train the head first, then the last block (`src/controlled_field_mixed.py`).
- The historical recipe updated **75%** of ResNet18 parameters but only **8%** of ViT-Small parameters. That
  confounds the architecture comparison.
- `scripts/explore_lpft_architecture.py` puts both architectures on one LP-FT recipe (100% trainable). ViT-Small
  then gains about 4 pp on External v3 and keeps Lab macro-F1 at 0.984, while ResNet18 forgets more Lab
  performance (0.786).

## 2. CNN versus Vision Transformer inductive biases

A CNN builds in **locality** (small receptive fields) and **translation equivariance** (shared kernels). This is
data-efficient and suits small lesion textures. A ViT splits the image into 16×16 patches and relates them with
global self-attention. Its weaker inductive bias usually needs more pretraining data. Steiner et al. (2022) show that
augmentation and regularization ("AugReg", the recipe behind the `augreg_in1k` checkpoint used here) partly
compensate at ImageNet scale. In return, a ViT can use long-range context.

Two consequences matter here. First, ViT attention can latch onto background tokens; Notebook 10 shows this
qualitatively. Second, the two models use different preprocessing (bilinear 256→224 with ImageNet statistics
for ResNet; bicubic with crop 0.9 and mean/std 0.5 for ViT). Each model must be evaluated with *its own*
transform, which `src/inference.py` and every evaluation script enforce.

## 3. Domain shift: what differs between PlantVillage and field photos

PlantVillage images are single detached leaves on a uniform background under controlled light. Field photos
have clutter, several leaves, occlusion, varied scale and lighting, and watermarks. This is **covariate shift**:
p(x) changes while the disease labels mean the same thing. It also involves **capture bias**. Noyan (2022)
reached 49% accuracy on 38 PlantVillage classes using only 8 background pixels, so imaging conditions correlate
with the class.

A model can therefore score 0.97–0.99 macro-F1 in the lab by partly exploiting acquisition cues, then collapse
in the field. Mixing Lab and Field images during fine-tuning ("rehearsal") is a classic defence against
**catastrophic forgetting** of the source task. RQ4 tests whether it also helps the target task.

*Evidence:* README RQ1 and RQ4; `data/results/notebook15_lab_retention/`; `data/results/weight_interpolation/`.

## 4. Batch normalization under small, shifted batches

BatchNorm normalizes with running statistics estimated on the training distribution. Fine-tuning on 16-image
batches of a different domain gives noisy batch statistics and silently rewrites the running means. Notebook 15
therefore **freezes BN statistics** in both conditions, so Field-only and Mixed differ only in their gradients.
ViT uses LayerNorm, which normalizes per token and has no running statistics.

*Implementation:* `set_phase` in `src/controlled_field_mixed.py` and `scripts/explore_lpft_architecture.py`.

## 5. Near-duplicate detection with perceptual hashing

Exact hashes (SHA-256 of decoded RGB pixels) catch byte-identical copies only. A **pHash** downsamples the
image, takes the 2-D DCT, keeps the 8×8 low-frequency block and thresholds it at the median, giving 64 bits that
survive resizing, recompression and mild colour changes. The Hamming distance between hashes approximates visual
similarity. The project uses distance ≤ 4 within PlantDoc (connected components define Adapt/Dev groups) and
≤ 10 for the stricter External v3 screen.

This screen found:
- 94 of 589 PlantSeg candidates duplicating public training data;
- 13 duplicated images with conflicting labels;
- only 5 near-duplicates across the Lab splits.

*Implementation:* `src/prepare_field_dev_verified.py`, `scripts/build_external_v3_plantseg.py`,
`scripts/audit_lab_split_duplicates.py`.

## 6. Grad-CAM and how to test an explanation

Grad-CAM (Selvaraju et al., 2017) weights each channel of the last convolutional block (`layer4`, 7×7) by its
spatially averaged gradient for a class, sums the channels, applies ReLU and upsamples. It shows *where evidence
for that class lies* at coarse resolution. It is not a segmentation, and a heat map alone cannot show the model
"looks at the lesion".

Two quantitative tests use expert lesion masks:
- the **pointing game** (Zhang et al., 2018): is the CAM maximum inside the mask, with a 15 px tolerance, against
  the chance rate given by the mask area?
- the **energy ratio**: the CAM mass inside the mask divided by the mask's area share.

Occlusion complements them. Lesions are removed by inpainting and the prediction change is measured; the edited
images are out-of-distribution, so this measures reliance rather than isolated causation.

*Evidence:* `scripts/evaluate_lesion_reliance.py`. Pointing accuracy is 0.58–0.66 against a chance level of 0.46,
and inpainting the lesions removes 38–52% of correct predictions.

## 7. Weight-space interpolation

Fine-tuning from a shared initialization often stays in one low-loss basin, so linear interpolation between the
initial and the fine-tuned weights gives models that trade the two tasks smoothly (WiSE-FT, Wortsman et al.,
2022; PAINT, Ilharco et al., 2022). This makes interpolation a zero-cost alternative to rehearsal for keeping
source accuracy.

The experiment checks whether it can reach Mixed's operating point. It cannot: at equal Lab retention it loses
0.17 Field Dev macro-F1.

*Implementation:* `scripts/evaluate_weight_interpolation.py`.

## 8. Self-supervised foundation features

DINOv2 (Oquab et al., 2024) learns features by self-distillation on 142M curated images, with no labels. Frozen
features plus a linear head are a strong reference for how much the backbone, rather than the adaptation recipe,
determines field performance. With a lab+field linear head, DINOv2-B exceeds every adapted project model on
PlantDoc, but by only 2–3 pp on External v3. This is a reminder that within-PlantDoc gains overstate external
gains.

*Implementation:* `scripts/evaluate_frozen_foundation_probe.py`.

## 9. Statistics for small, paired evaluations

| Need | Method | Why this one |
| --- | --- | --- |
| Interval for one accuracy | Wilson score interval | Correct coverage near 0 or 1 and for small n, unlike the normal approximation |
| Two models on the same images | Exact McNemar test | Uses only discordant pairs; the images are shared, so the samples are not independent |
| Difference with an interval | Paired image-level bootstrap | Resamples images while keeping both models' outcomes together |
| Three seeds on the same images | Sign-flip permutation on per-image seed-mean correctness | Seeds are not independent samples of the population |
| Several tests | Holm adjustment | Controls family-wise error and is uniformly more powerful than Bonferroni |
| Over-confidence | Expected calibration error (ECE, 10 bins) | Softmax confidence is not a probability of being correct; Lab models are 0.90 confident at 0.43 accuracy on External v3 |
| Fair confirmation | Pre-registration: protocol committed before the run | Prevents choosing metrics or subsets after seeing results (commit `8762743` precedes `cfc4752`) |

With n = 47, a 95% interval on accuracy spans about ±14 pp; with n = 486 it is about ±4.4 pp. This is why the
round-2 external set was built (Varoquaux, 2018).

*Implementation:* `src/statistical_analysis.py`, `src/supplementary_analysis.py`, `scripts/evaluate_external_v3_plantseg.py`.

## References

- Kumar et al. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. ICLR.
- Steiner et al. (2022). How to Train Your ViT? TMLR.
- Noyan (2022). Uncovering Bias in the PlantVillage Dataset. arXiv:2206.04374.
- Selvaraju et al. (2017). Grad-CAM. ICCV.
- Zhang et al. (2018). Top-Down Neural Attention by Excitation Backprop. IJCV.
- Wortsman et al. (2022). Robust Fine-Tuning of Zero-Shot Models. CVPR.
- Ilharco et al. (2022). Patching Open-Vocabulary Models by Interpolating Weights. NeurIPS.
- Oquab et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
- Varoquaux (2018). Cross-validation failure: small sample sizes lead to large error bars. NeuroImage.

# Snapshot A — grading report

Checkout: `/home/user/grade_A` (single commit, bb794d7). I ran the 50-test suite (all pass) and recomputed README numbers from the committed CSV/JSON files. I did not download checkpoints and did not run inference.

## Scores
| Criterion | Weight | Score | One-line justification |
|---|---|---|---|
| C1 CV knowledge and difficulty | 20% | 7.5 | Domain shift, AdaBN, frozen DINOv2 probes, weight interpolation, lesion-mask occlusion with a same-area control, and pointing-game Grad-CAM. The two models are off-the-shelf ResNet18 and ViT-S, and nothing new is proposed. |
| C2 Implementation depth and completion | 25% | 8.0 | 15 notebooks, about 8k lines in `src/`, `scripts/` and `tests/`, hash-sealed checkpoints, a one-shot evaluator, and a demo app. The app was not run. |
| C3 Experimental design and controls | 25% | 8.0 | Leaf-grouped Lab split, pHash dedup, matched-exposure Notebook 15, a protocol file for External v3, an evaluation ledger, and paired CIs with Holm. Weaknesses: Field Test has 47 images, there is no proof that the protocol came before the run, and External v3 was reused 100+ times. |
| C4 Results analysis and limitations | 15% | 8.5 | Strongly honest: nulls are not written as equivalence, strata and confounds are reported, and the "Healthy fix" is explained. A few table-framing and ordering problems remain. |
| C5 Report, demo, reproducibility | 15% | 7.0 | Checkpoint registry with SHA-256, lockfile, repro script, and a claims test. The README is 388 dense lines. The commit-history provenance cannot be checked. |

Weighted total: 7.83 / 10 (1.50 + 2.00 + 2.00 + 1.275 + 1.05)

## Findings

### C1
- **C1-1 (medium)**, `README.md:153-166`. The architecture comparison is confounded (ResNet 75% vs ViT 8% trainable), and the "LP-FT" label is admitted to be mostly a depth test. The README discloses this. Fix: keep it, but do not frame it as a CNN-vs-ViT finding.
- **C1-2 (low)**, `README.md:214-229`. Grad-CAM is run on ResNet18 only, and ViT attention is not quantified. Fix: add attention rollout, or drop the CNN-vs-ViT "explanation" framing.
- **C1-3 (low)**, `README.md:149`. DINOv2 is run at 224 px instead of its native 518, so the foundation-model reference is handicapped. This is disclosed.

### C2
- **C2-1 (medium)**, `scripts/reproduce_round2.sh:2-3`. Full reproduction needs about 1.5 h plus 4 GB of downloads. I did not run it, so end-to-end reproduction is unverified.
- **C2-2 (low)**, `notebooks/11_external_evaluation.ipynb`. Only 12 of 23 code cells have saved outputs. Notebooks 02, 03, 05 and others also have cells without outputs.
- **C2-3 (low)**, `README.md:337`. Retraining requires Colab and the original Drive layout, so the training itself is not reproducible from the repo. The README states this.

### C3
- **C3-1 (high)**, `docs/evaluation_ledger.md:8-9`, `README.md:95,262`. The pre-registration order depends on commit hashes (`8762743`, `cfc4752`, `828bbfe`, `0d47ea8`). `git log` in this snapshot shows 1 commit, so "committed before the run" cannot be verified. Only `data/results/external_v3_plantseg/run_record.json` (protocol and manifest SHA-256) is checkable. Fix: ship the git history or an RFC-3161 timestamp.
- **C3-2 (high)**, `docs/evaluation_ledger.md:35-37`. The ledger counts 104 model evaluations on External v3 after the one confirmatory run, all without multiplicity control. The README admits this, but the abstract's 24-epoch result (+2.4 pp, "p=0.06") rests on a reused set.
- **C3-3 (medium)**, `README.md:144`. The row "Fully fine-tuned project models, exploratory" gives ranges that match only a subset of the rows in the table above it.
  - Lab Test is stated as 0.863–0.984, but the LP-FT ResNet 8-epoch value is 0.786 (`data/results/lpft_architecture/summary.csv`).
  - Field Test is stated as 0.562–0.564, but ViT 24-epoch LP-FT is 0.584 (`data/results/training_budget/summary.csv`).
  - External v3 accuracy is stated as 0.484–0.498, but ViT 24-epoch LP-FT is 0.475.
  - Fix: list all LP-FT rows, or state the selection rule.
- **C3-4 (medium)**, `README.md:55-62`, `data/splits/field_test.csv`. Field Test has 47 images, with 8–11 per class, and was "inspected from Notebook 04 onward". It carries 2 disputed labels (rows 2524 and 2532). Checkpoint selection used a 92-image Field Dev. This is disclosed. Fix: stop using Field Test for headline claims.
- **C3-5 (medium)**, `docs/methods.md` (Data section). Lab splits are leaf-grouped, but 1,510 / 378 / 465 train / val / test images have no leaf ID and are treated as singletons. The pHash audit cannot detect same-leaf photos, so Lab scores of 0.97–0.99 may be inflated. This is disclosed.
- **C3-6 (medium)**, `README.md:166,252`. Even where the conclusion is a null, it rests on 3 seeds. Seed-level t-tests with n=3 (p=0.12, p=0.06) have almost no power. Image-level CIs condition on 3 training runs, and the README says so.
- **C3-7 (low)**, `README.md:262`. One of the six 24-epoch Notebook 15 re-runs (Mixed seed 44) does not reproduce the sealed Field Dev score (0.418 vs 0.433). It is disclosed, with the cause given as likely but untested.

### C4
- **C4-1 (low)**, `README.md:260`. This sentence is correct but confusing: "the External v3 interval excludes losses from rehearsal larger than 0.1 pp". It reads as a strong claim.
- **C4-2 (low)**, `README.md:10` (Abstract). The abstract is about 270 words of dense statistics, with no single plain headline.
- **C4-3 (low)**, `README.md:146-151`. The DINOv2 probe "exceeds" the fixed checkpoints on PlantDoc, but the comparison is on 47 and 92 images with confounds. The README hedges this well.
- **C4-4 (low)**, `README.md:235`. Notebook 14 framing policies were compared after the outcomes were known. The README says so, and the best and worst cells are both reported.

### C5
- **C5-1 (medium)**, `README.md` (388 lines, about 58 KB). The README is too long and dense for a reader. The key results are buried among about 15 exploratory sub-analyses. Fix: a 1-page summary with the exploratory material moved to `docs/`.
- **C5-2 (medium)**, `README.md:33,304`, `REPO_REVIEW.md`. The README points to a Vietnamese-language internal review file (`REPO_REVIEW.md`) as "what changed in review round 2". Process documents are mixed into the deliverable. Fix: move them to `docs/archive/`.
- **C5-3 (medium)**, `README.md:262`, `docs/evaluation_ledger.md`. Many provenance claims cite commit hashes that do not exist in this snapshot (see C3-1). A reader cannot follow them.
- **C5-4 (low)**, `data/checkpoints.json`. Checkpoints are hosted on Google Drive, so the link's long-term availability and the SHA-256 match were not verified.
- **C5-5 (low)**, `app/streamlit_app.py`. I did not launch the demo, so the README's claims about it are unverified.

## Number checks
| Claim (file:line) | Source file | Stated | Actual | Match? |
|---|---|---|---|---|
| External v3 accuracy of all checkpoints (README:10) | `data/results/external_v3_plantseg/metrics.csv` | 41–44% | 0.4115–0.4444 | Yes |
| P1 and P2 differences (README:193) | `external_v3_plantseg/comparisons.csv` | −0.2 pp (−5.3, +5.1; p=1.0); +1.4 pp (−2.1, +4.9; p=0.49) | −0.0021 (−0.0535, 0.0514), p=1.0; +0.0144 (−0.0206, 0.0494), p=0.494 | Yes |
| Disease macro-F1 differences (README:194) | `external_v3_secondary_metrics/comparison_intervals.csv` | ResNet +0.086 (0.039–0.133); ViT +0.065 (0.029–0.102) | 0.08626 (0.03932, 0.13329); 0.06548 (0.02884, 0.10153) | Yes |
| P3 at 8 epochs (README:260) | `comparisons.csv`, `p3_seed_level.csv` | +1.7 pp (−0.1, +3.6), sign-flip p=0.075, Holm 0.23, seed-level p=0.12 | 0.01715 (−0.0007, 0.0357), 0.0754, 0.226, 0.120 | Yes |
| Mixed 211.0/486 vs Field-only 202.7/486 (README:260) | `comparisons.csv` | 0.434 vs 0.417 | 0.43416 and 0.41701, which give 211.0 and 202.7 | Yes |
| Predicted Early Blight, ResNet18 Lab (README:188) | `external_v3_plantseg/predictions.csv` | 300 | 300 | Yes |
| Predicted Healthy per model (README:188-191) | `predictions.csv` | 0, 1, 23, 22 | 0, 1, 23, 22 | Yes |
| Errors and BS↔Septoria swaps (README:204-210) | `predictions.csv`, `external_v3_label_merge/merged_accuracy.csv` | 279/39, 277/59, 280/62, 270/58; FO 282–286 / 73–74; Mixed 272–279 / 70–74 | Same | Yes |
| Calibration (README:200) | `external_v3_error_analysis/calibration.csv` | 0.90 at 0.43, ECE 0.475; ViT 0.81 at 0.43, ECE 0.384; adapted ECE 0.16–0.29 | 0.9012 / 0.4259 / 0.4753; 0.8139 / 0.43 / 0.3838; 0.1618–0.2948 | Yes |
| Boundary merge for P1 (README:201) | `external_v3_label_merge/merged_comparisons.csv` | +4.5 pp (−0.8, +9.9), p=0.12, Holm 0.28 | 0.0453 (−0.0082, 0.0988), 0.121, 0.276 | Yes |
| Notebook 15 Field Test (README:243-247) | `supplementary_statistics/notebook15_per_seed.csv`, `notebook15_paired_bootstrap.csv` | Per-seed values; Δ −0.023 [−0.086, +0.024]; accuracy −0.014 [−0.064, +0.028] | −0.0228 [−0.0859, 0.0237]; −0.0142 [−0.0638, 0.0284] | Yes |
| Lab retention (README:258) | `notebook15_lab_retention/lab_retention_summary.csv` | 0.8136±0.0056 vs 0.9535±0.0079; Val 0.8438±0.0096 vs 0.9740±0.0038 | Same | Yes |
| 24-epoch Notebook 15 table (README:266-268) | `mixed_budget_summary/per_run.csv` | All cells | All checked cells match | Yes |
| 24-epoch seed-level differences (README:270-272) | `mixed_budget_summary/seed_level_differences.csv`, `external_v3_secondary.csv` | +0.113 to +0.125 Lab Test, p=0.001; Field Dev p=0.19 and 0.09; +2.4 pp (0.7, 4.1), sign-flip p=0.008; macro-F1 p=0.026; seed-level p=0.06 | 0.113–0.125, 0.00105; 0.186, 0.092; 0.0240 (0.0069, 0.0412), 0.0083; 0.0259; 0.0589 | Yes |
| Frozen DINOv2 table (README:140-142) | `frozen_foundation_probe/probe_results.csv` | B/14 lab-only 0.976 / 0.399 / 0.336 (20/47) / 0.473 / 0.366; B/14 lab+field 0.961 / 0.622 / 0.665 (31/47) / 0.469 / 0.456; S/14 lab+field 0.935 / 0.570 / 0.538 (25/47) / 0.436 / 0.432 | Same | Yes |
| "Fully fine-tuned project models" row (README:144) | `lpft_architecture/summary.csv`, `training_budget/summary.csv` | Lab 0.863–0.984, Dev 0.516–0.555, Field Test 0.562–0.564, External v3 accuracy 0.484–0.498 / F1 0.469–0.487 | Rows include 0.786 Lab, 0.584 Field Test, and 0.475 External v3 accuracy | **No** (C3-3) |
| LP-FT table (README:157-160) | `lpft_architecture/summary.csv` | e.g. ResNet last-block 0.815 / 0.432 / 0.422 / 0.418 / 0.408 | 0.8153 / 0.4323 / 0.4218 / 0.4184 / 0.4079 | Yes |
| 24-epoch seed 42 rows (README:161-164) | `training_budget/summary.csv` | e.g. 0.864 / 0.531 / 0.507 / 0.471 / 0.457 | 0.8637 / 0.5312 / 0.5073 / 0.4712 / 0.457 | Yes |
| Weight interpolation table (README:278-280) | `weight_interpolation/operating_points_summary.csv` | Retention-matched 0.951 Lab Test, 0.266 Field Dev, 0.426/0.339 Field Test, 0.447/0.375 External v3; gap 0.17 | Same; 0.4399 − 0.2656 = 0.174 | Yes |
| External v1 strata and Holm (README:134,172-179) | `supplementary_statistics/external_v1_by_stratum.csv`, `mcnemar_holm.csv` | 19/46, 22/46, 21/46, 19/46; unadjusted p=0.024, smallest Holm p=0.24; 12/16, 13/16; p=0.83 and 0.58 | 19, 22, 21, 19; 0.0243, 0.2431; 12, 13; 0.8318, 0.5811 | Yes |
| Lab duplicate audit (README:134) | `lab_split_duplicates/duplicate_summary.csv`, `lab_test_clean_vs_all.csv` | 3/1,341 and 2/1,692 flagged; no Lab Test macro-F1 change over 0.0001 | 3 and 2; largest change about 0.0001 | Yes |
| AdaBN (README:212) | `adabn_paired/adabn_paired.csv` | 0.235 → 0.391, CI 0.049–0.257, accuracy p=0.18, Holm 0.55 | 0.23546 → 0.39074, CI 0.0492–0.25714, 0.18493, 0.5548 | Yes |
| PlantSeg screening (README:89,93,97) | `external_v3_plantseg/manifest.csv`, `shared_image_label_check.csv` | 94 excluded (77 + 17), 486 included, 13 of 64 disagree | 94 (77 ≤ 4, 17 above), 486, 13 of 64 | Yes |
| Test count (README:28) | `tests/` | 50 | Ran 50, OK | Yes |
| Field Test and External v1 baselines (README:129-132) | `outputs/comparison/cnn_vs_vit_metrics.csv` | ResNet Lab 0.2802 (17/47); ViT Lab 0.3617 | 0.3617 accuracy (17/47), macro-F1 0.2802, 0.3617 | Yes |

I did not check the Notebook 13/14 numbers, the lesion and Grad-CAM table beyond the ranges below, or the Field Dev figures for External v1, only from the ranges in `occlusion_control/control_summary.csv` and `lesion_reliance/gradcam_summary.csv`. Extra drop 0.12–0.15 and pointing accuracy 0.58–0.66 match.

## Cross-check notes
- I dropped one candidate finding about a "10 core vs 16 checkpoints" inconsistency. Line 16 says "core" and line 329 says "all 16", and `download_checkpoints.py` has core and all groups, so there is no contradiction.
- C3-1 and C5-3 are the same root cause, a squashed history, and I scored it once under C3 and once under C5 for the reproducibility angle. Roughly the same 1.0 point of impact is not double-counted: C3 stays at 8.0 because the sealed `run_record.json` hashes and in-file sequencing still support the one-shot design.
- C3 and C4 are consistent: every limit I found in C3 is also disclosed in the README Limitations (C4 credit). The only places where the text pushes beyond the evidence are the "Fully fine-tuned" row (C3-3) and the 24-epoch "+2.4 pp" emphasis in the abstract, both on a reused set.
- No contradictions between the README abstract and result files. All headline numbers I checked match.

## What I could not verify
- Pre-registration ordering: the commits (`8762743`, `cfc4752`, `828bbfe`, `0d47ea8`, `7c1a280`, `64d749b`, `33672f4` and so on) cited in the README and ledger are absent; the repo has 1 commit.
- Checkpoint downloads and SHA-256 match (Drive), the end-to-end `reproduce_round2.sh`, and the Streamlit demo.
- That model checkpoints in `data/checkpoints.json` produce the stored predictions (I did not run inference).
- Notebook 13 and 14 numbers and the notebook-embedded images.
- PlantSeg and PlantDoc image content, the duplicate-screen correctness (it needs the datasets), and the "13 of 64 label conflicts" beyond the CSV count.
- The REPO_REVIEW.md claims (a review file, treated as an unverified document).

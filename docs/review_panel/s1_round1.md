# Snapshot C — grading report

Commit 61c3cd1. I ran the 21 unit tests (all pass) and `scripts/audit_repository.py` (runs, output ends cleanly). I did not train or run inference. The prompt asked me to send the report with send_message, but I have no such tool, so this is the report as my final answer.

## Scores
| Criterion | Weight | Score | One-line justification |
|---|---|---|---|
| C1 CV knowledge and difficulty | 20% | 6.5 | The study covers lab-to-field domain shift with two architectures (ResNet18, ViT-Small), Grad-CAM and attention views, and crop/multi-view sensitivity. Everything uses standard pretrained backbones and small fine-tuning recipes, so the method depth is modest. |
| C2 Implementation depth and completion | 25% | 7.0 | There are about 4.7k lines of src, scripts and tests, 15 notebooks, and hash-sealed checkpoints. It also has a lab-retention evaluator, a McNemar/Holm/bootstrap module, and dataset audit tools. Raw images and checkpoints are absent, so I could not re-execute it. |
| C3 Experimental design and correctness | 25% | 7.5 | Notebook 15 matches initialization, Field exposure, optimizer updates and checkpoint epoch, with 3 pre-declared seeds and sealed checkpoints. Duplicate-aware splits, Holm correction and a stratified External v1 analysis are present. Weaknesses: the 47-image Field Test is reused, Field Dev is both the selection and reporting set, and Notebooks 02–14 are single-seed. |
| C4 Results analysis and limitations | 15% | 8.0 | Nulls are reported as "no detectable change", not as equivalence. It reports the class/source confound, label noise (row 2524), the architecture confound, and the net-two-image Field Test difference. It admits post-hoc view comparison and that Lab Validation is optimistic. |
| C5 Report, demo and reproducibility | 15% | 7.5 | The README is clear and the tests plus audit run offline. Almost every README number traces to a committed CSV. Gaps: no downloader or checkpoints, canonical External v1 predictions are not tracked, and there is clutter (`REPO_REVIEW.md`, archives). |

Weighted total: 7.25 / 10  
(6.5×0.20 + 7.0×0.25 + 7.5×0.25 + 8.0×0.15 + 7.5×0.15 = 1.30 + 1.75 + 1.875 + 1.20 + 1.125)

## Findings

### C1
- **C1-1 (medium)**, README.md "Models and Experimental Design". Evidence: only ResNet18 and ViT-Small with torchvision/timm weights, and "Architecture comparison confounded" is itself listed as a limitation. Fix: add a matched-capacity comparison or drop architecture-superiority language.
- **C1-2 (low)**, `notebooks/10_error_analysis.ipynb`. Evidence: the README says Grad-CAM and ViT attention are "qualitative only", with no lesion annotations or localization metric. Fix: add a quantitative faithfulness check, such as deletion/insertion curves.
- **C1-3 (low)**, README §3. Evidence: crop views change the field of view but all inputs stay at 224×224, so the study cannot separate resolution from framing. Fix: add a scale-controlled variant.

### C2
- **C2-1 (medium)**, README "Reproducibility". Evidence: "model checkpoints… not publicly distributed", `.gitignore` excludes `*.pth`/`models/`, and there is no downloader. Fix: publish checkpoints or add a fetch script.
- **C2-2 (medium)**, `outputs/external_confirmatory_eval/`. Evidence: the tracked External v1 predictions are from a superseded loader without EXIF orientation. The README's canonical per-model numbers come from Notebook 11 outputs, with "canonical_counts_from_notebook11" only in the stratum CSV. Fix: commit the canonical EXIF-corrected prediction CSVs.
- **C2-3 (low)**, `REPO_REVIEW.md` (45 KB, Vietnamese), `src/archive/`, `notebooks/archive/`. Evidence: several superseded pipelines remain alongside canonical ones. Fix: move to a branch or tag.
- **C2-4 (low)**, `tests/`. Evidence: 21 tests cover the dataset, the controlled-run code and the statistics exports. I saw no test for the training or evaluation loops (e.g. `src/evaluate.py`). Fix: add a smoke test on synthetic tensors.

### C3
- **C3-1 (high)**, `data/splits/field_test.csv` (47 rows), README Limitations. Evidence: "The 47-image Field Test was inspected early and reused… Field Dev selected checkpoints." No locked test set exists, and the repo says so. Fix: lock a larger independent field set before any further selection.
- **C3-2 (medium)**, README §1. Evidence: the adapted checkpoints were selected on Field Dev (92 images, 0.4459 macro-F1) and also reported there. Fix: label Field Dev columns "development", as the README partly does, in every table.
- **C3-3 (medium)**, README §4 / `notebook15_per_seed.csv`. Evidence: n = 47, per-seed differences of 0–2 images, and a bootstrap CI of [−0.086, +0.024]. The conclusion is underpowered, but it is worded as "no detectable change". Fix: run a power analysis and add more seeds or a bigger test set.
- **C3-4 (medium)**, README "Limitations". Evidence: "Single seed for Notebooks 02–14", so the RQ1/RQ2 model rankings rest on single runs. Fix: add seeds for the headline models.
- **C3-5 (low)**, README §3 (Notebook 14). Evidence: "These policies were compared after the outcomes were known." Fix: pre-register the policy or report it as exploratory only.
- **C3-6 (low)**, `data/results/notebook15_lab_retention/protocol.json`. Evidence: Lab images come from a HuggingFace default branch (revision f808f27). Fix: pin and checksum the images.

### C4
- **C4-1 (low)**, README §2 and Discussion item 1. Evidence: capture bias is offered as a "plausible contributor" to the domain gap, and the README says "this study does not isolate it". Fix: no change, or add a diagnostic such as a background-only probe.
- **C4-2 (low)**, README §4 Conclusion. Evidence: "The earlier apparent Mixed advantage… came from comparing a geometric-augmentation initialization and selecting the best of six configurations on the same 92 Dev images." I could not check the six-configuration claim against the code. Fix: link the exact table that shows it.
- **C4-3 (low)**, README §1. Evidence: "Laboratory accuracy does not transfer: … 0.97–0.99 to 0.24–0.36" compares different class mixes and source sizes. The field macro-F1 over 5 classes on 47 images is noisy. Fix: report per-class support next to the headline.

### C5
- **C5-1 (medium)**, README "Reproducibility". Evidence: "These checks use saved research evidence", so the commands verify committed artifacts, not the experiments. Fix: state this explicitly next to the commands.
- **C5-2 (low)**, README §1 vs `outputs/external_confirmatory_eval/confirmatory_summary_all_partitions.csv`. Evidence: the committed tracked file differs from the README (see Number checks). `SUPERSEDED.md` discloses this. Fix: move the superseded folder out of `outputs/` or rename it.
- **C5-3 (low)**, repo root. Evidence: `REPO_REVIEW.md`, `CITATION.cff` and `DATA_NOTICE.md` are present, but the review file is not a primary result document. Fix: remove it or mark it as an internal note.

## Number checks
| Claim (file:line) | Source file | Stated | Actual | Match? |
|---|---|---|---|---|
| ResNet18 Lab Test macro-F1 (README:90) | `outputs/comparison/cnn_vs_vit_metrics.csv` | 0.9744 | 0.9744 | Yes |
| ViT-Small Lab Test macro-F1 (README:91) | same | 0.9938 | 0.9938 | Yes |
| ResNet Lab Field Test, 17/47, 0.2802 (README:90) | same | 17/47, 0.2802 | acc 0.3617 (=17/47), F1 0.2802 | Yes |
| ViT Lab Field Test, 21/47, 0.3617 (README:91) | same | 21/47, 0.3617 | acc 0.4468 (=21/47), F1 0.3617 | Yes |
| ResNet Geom+Mixed Field Test, 19/47, 0.3587 (README:92) | `confirmatory_summary_all_partitions.csv` | 19/47, 0.3587 | 19/47, 0.3587 | Yes |
| ViT Mixed Field Test, 25/47, 0.5103 (README:93) | same | 25/47, 0.5103 | 25/47, 0.5103 | Yes |
| ResNet Geom+Mixed External v1, 33/62, 0.4882 (README:92) | same | 33/62, 0.4882 | 33/62, 0.4882 | Yes |
| ResNet Lab External v1, 19/62, 0.2406 (README:90) | same | 0.2406 | 0.2456 (19/62) | No, F1 differs. `SUPERSEDED.md` says 0.2406 is the canonical Notebook 11 value, but its source CSV is not committed. |
| ViT Lab External v1, 27/62, 0.4157 (README:91) | same | 27/62, 0.4157 | 28/62, 0.4346 | No, same explanation. Not verifiable from committed predictions. |
| ViT Mixed External v1, 32/62, 0.4808 (README:93) | same, and the prediction CSV (`correct` sum = 31) | 32/62, 0.4808 | 31/62, 0.4710 | No, same explanation. Not verifiable from committed predictions. |
| McNemar: ResNet pooled 19→33, unadjusted p = 0.024, Holm 0.24 (README §2) | `mcnemar_holm.csv` | p 0.024, adj 0.24 | 0.024307, 0.24307 | Yes |
| McNemar on online disease: ResNet p = 0.83, ViT p = 0.58 | `mcnemar_holm.csv` | 0.83 / 0.58 | 0.8318 / 0.5811 | Yes |
| Notebook 15 per-seed Field Test macro-F1 (README §4) | `notebook15_per_seed.csv` | 0.4338/0.3970; 0.4214/0.3840; 0.4349/0.4405 | same | Yes |
| Paired bootstrap F1: −0.023, CI [−0.086, +0.024]; accuracy −0.014, [−0.064, +0.028] | `notebook15_paired_bootstrap.csv` | as stated | −0.0228 [−0.0859, 0.0237]; −0.0142 [−0.0638, 0.0284] | Yes |
| Lab Test macro-F1 means 0.8136 / 0.9535 | `lab_retention_summary.csv` | 0.8136 ± 0.0056; 0.9535 ± 0.0079 | 0.81360 ± 0.00563; 0.95352 ± 0.00786 | Yes |
| Lab Val means 0.8438 / 0.9740 | same | as stated | 0.84385; 0.97396 | Yes |
| Per-seed Lab Test correct counts (README §4) | `lab_retention_by_checkpoint.csv` | 1,490/1,634; 1,482/1,622; 1,499/1,641 | same | Yes |
| Baseline Lab Test 1,657/1,692, 0.9744 | same | 1,657, 0.9744 | 1,657, 0.97445 | Yes |
| External v1 strata (README §2): ResNet Lab 0/16 and 19/46; ViT Lab 5/16 and 22/46; ResNet Mixed 12/16 and 21/46; ViT Mixed 13/16 and 19/46 | `external_v1_by_stratum.csv` | as stated | same | Yes |
| Split sizes (README Datasets table) | `data/splits/*.csv` line counts | 5,365 / 1,341 / 1,692; 370 / 92 / 47 | 5365; 1341; 1692; 370; 92; 47 (rows minus header) | Yes |
| Field Dev macro-F1 columns (0.2355, 0.2677, 0.4459, 0.3872) | not checked individually | n/a | Not checked individually. 0.4459 and 0.3872 are stated elsewhere in the repo's results summary. | Unverified |
| Notebook 13 counts (0/57, 15/57, 44/57, 43/57) and Notebook 14 values | notebooks only | as stated | I did not open the notebook outputs | Unverified |

## Cross-check notes
- C3-1 and C3-2 are consistent with C4's credit. The repo states these limitations itself, so I penalised the design, not the honesty.
- C2-2 and C5-2 are the same root issue (the canonical External v1 predictions are not committed). I scored it once under C2 and once under C5 as a reproducibility gap.
- I found no contradiction between C3 and C4: the main nulls are not written up as equivalence.
- I dropped one possible finding about the "no tuning on test" claim for the adapted checkpoints. The repo discloses that Field Dev did the selecting, and the Field Test was not used for selection in Notebook 15, as far as I could check from the seal files.
- I did not independently verify the six-configuration selection claim (C4-2).

## What I could not verify
- Any claim that depends on raw images or checkpoints, which are not in the repo.
- The canonical External v1 per-model F1 and counts for the two non-Mixed ResNet/ViT lab baselines and ViT Mixed (README rows 90, 91 and 93). These rely on Notebook 11 outputs, and `SUPERSEDED.md` documents the difference.
- Hash-seal integrity (SHA-256 of checkpoints), since the checkpoints are absent.
- Notebook 13, 14 and 15 internal consistency beyond the committed CSVs.
- The "14 excluded training images" claim against the code. I saw only the quarantine and conflict CSVs, with 14 data rows in `field_quarantined_train.csv`.
- That no model training was done on Field Test images in Notebooks 02–14, because the saved prediction exports do not record the selection history.

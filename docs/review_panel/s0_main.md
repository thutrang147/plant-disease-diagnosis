# Snapshot B — grading report

Checked: README, committed CSVs and JSON, notebook outputs for 12, 13 and 14, split files and `src/data_prep.py`. I ran the unit tests (15 pass) and `scripts/audit_repository.py` (runs). No inference was run and no checkpoints were downloaded. The v2 images and the model checkpoints are not in the repo.

## Scores
| Criterion | Weight | Score | One-line justification |
|---|---|---|---|
| C1 CV knowledge and difficulty | 20% | 7.0 | ResNet18 vs ViT-Small under lab-to-field shift, Grad-CAM and attention, multi-crop inference, calibration (ECE), McNemar tests and a custom reviewed dataset. No novel method. |
| C2 Implementation depth and completion | 25% | 7.5 | 15 notebooks, a `src/` pipeline with group-aware splits, pHash near-duplicate quarantine, dataset build and validate scripts, a Streamlit app, and 15 passing tests. Several key artifacts are missing from the repo. |
| C3 Experimental design and correctness | 25% | 6.0 | Group-aware splits, a quarantine step, and a matched-budget, 3-seed controlled experiment (Notebook 15). Weaknesses: the 47-image Field Test was already inspected, the Field Dev set was used for both checkpoint selection and the headline table, and the n values are tiny. |
| C4 Results analysis and limitations | 15% | 7.5 | Candid limitations section, a hedged read of the Notebook 15 null (Field-only vs Mixed), a per-class trade-off analysis, and an explicit warning that cropping confounds several factors. |
| C5 Report, demo and reproducibility | 15% | 6.0 | Extensive docs, hashes and provenance records, and a Streamlit demo. Headline external-v1 numbers cannot be traced to a committed CSV and conflict with the committed one. Data, checkpoints and the notebook 12/13/14 output folders are absent. |

Weighted total: 0.20×7.0 + 0.25×7.5 + 0.25×6.0 + 0.15×7.5 + 0.15×6.0 = 1.40 + 1.875 + 1.50 + 1.125 + 0.90 = **6.80 / 10**

## Findings

### C1
- C1-1 (low) `README.md` and `notebooks/14_lesion_resolution_study.ipynb`: the "lesion resolution" framing is not backed by lesion masks or lesion-size measurements. The README itself says the experiment cannot isolate resolution. Fix: rename Notebook 14 to "inference-view sensitivity".
- C1-2 (low) Only ResNet18 and ViT-Small are used, with no stronger domain-adaptation baseline. The adaptation is plain fine-tuning (Field-only or Mixed), with no DANN, CORAL or similar. Fix: add one standard UDA or domain-generalization baseline.

### C2
- C2-1 (medium) `outputs/` has no folders for the External v1 canonical run (`outputs/external_error_analysis/...`), Notebook 13 or Notebook 14 (`ls outputs/external_error_analysis` fails). Their results exist only as saved cell outputs inside the notebooks. Fix: commit the prediction CSVs.
- C2-2 (medium) The Notebook 15 per-image Field Dev predictions are not committed. Only the six Field Test CSVs are in `data/results/notebook15/prediction_exports/`. Fix: commit the Dev CSVs, or state that they are absent.
- C2-3 (low) `src/archive/`, `notebooks/archive/` and three field-dev output folders (`field_dev_comparison`, `_grouped`, `_verified`) sit side by side, so it is unclear which is canonical. Fix: label the canonical folder in the README.

### C3
- C3-1 (high) The 47-image Field Test was previously inspected. `data/splits/field_dev_protocol.json` says "Field Test was previously inspected, so obtain a truly independent test". Notebook 15 then evaluates on the same 47 images for all three seeds. Fix: collect a fresh field test.
- C3-2 (high) The README headline table uses Field Dev (92 images, 18 Bacterial Spot, 8 or fewer per class on some classes) for ResNet18 Geometric + Mixed (0.4459, bold). Field Dev was used for checkpoint selection, so this number is optimistic. The README labels the column but does not say so. Its External v1 column reuses the same fixed checkpoints on small external data. Fix: add a footnote that Field Dev is a selection set, and report the Field Test column.
- C3-3 (medium) The Field Test macro-F1 for ResNet18 Mixed is 0.3587 (`outputs/external_confirmatory_eval/confirmatory_summary_all_partitions.csv`), below the Field Dev 0.4459. The README does not show this drop. Fix: show the Field Test column in the headline table.
- C3-4 (medium) External v1 and v2 mix sources: Healthy images are self-collected and disease images come from the web. Healthy recall is therefore confounded with source. The README admits this for v2 but still presents the Healthy recall gains as a headline. Fix: caveat the gains where they are presented.
- C3-5 (medium) Notebook 15 uses 3 seeds on 47 images. The difference (Field-only 0.4300 vs Mixed 0.4072 macro-F1) is smaller than the Mixed SD of 0.0296 and has no significance test. The README words this cautiously, but it still says "Field-only achieves higher mean Macro-F1". Fix: report paired seed deltas with a bootstrap or exact test.
- C3-6 (low) Gains in the five-crop comparison (38/62 vs 32/62) come from a single model on 62 images, with no confidence interval in the README. Fix: add a paired McNemar test or a bootstrap CI.
- C3-7 (low, positive) Leakage control is real: StratifiedGroupKFold on the lab data and pHash grouping on the field data, with 0 train/dev/test group overlap recorded. 14 images were quarantined.

### C4
- C4-1 (low) The README's "Healthy 0/16 to 12/16" claim is verified, but the paired Notebook 13 analysis of 23 parents and 49 crops says its images are correlated. The README does say this.
- C4-2 (low) The README mentions Grad-CAM and attention only qualitatively, and no quantitative faithfulness check backs them. The README states this limit.
- C4-3 (low, positive) The README correctly shows the Early Blight collapse (11/13 to 1/13) and the Bacterial Spot precision drop, not only the aggregate gain.

### C5
- C5-1 (high) The README's External v1 ViT-Small Baseline Macro-F1 is 0.4157, and ViT-Small Mixed is 0.4808. The committed `outputs/external_confirmatory_eval/confirmatory_summary_all_partitions.csv` gives 0.4346 and 0.4710. The README values do come from the notebook 12 output (run `20261125_163347_4ca8183f`, which uses EXIF-before-RGB loading). The committed CSV comes from an archived raw-RGB run. `docs/experiment_reconciliation.md` explains the discrepancy, but the README does not. A reader checking the README against the committed CSVs sees a mismatch. Fix: commit the canonical run's CSVs, or add a footnote in the README.
- C5-2 (medium) The README says tests and the audit "use saved research evidence", but full reproduction needs checkpoints and images that are not public. Notebook cells contain absolute paths such as `/Volumes/Lexar E6/...`. Fix: ship the checkpoints (the task allowed a downloader, but none exists) or document how to retrain.
- C5-3 (low) `docs/` has seven overlapping audit documents (about 1000 lines combined), including self-assessment files. They are unverified claims and add noise. Fix: consolidate them.

## Number checks
| Claim (file:line) | Source file | Stated | Actual | Match? |
|---|---|---|---|---|
| Split sizes 5,365 / 1,341 / 1,692 (README) | `data/splits/lab_*.csv` | 5365/1341/1692 | row counts minus header: 5365/1341/1692 | Yes |
| Field 370/92/47, 14 excluded (README) | `data/splits/field_adapt_train.csv`, `field_dev.csv`, `field_test.csv`, `field_quarantined_train.csv` | 370/92/47/14 | 370/92/47/14 | Yes |
| Lab Test Macro-F1, 4 models (README:87-90) | `outputs/final_comparison_development/final_comparison_summary.csv` | 0.9744/0.9938/0.9140/0.9886 | same | Yes |
| Field Dev Macro-F1, 4 models (README:87-90) | `outputs/final_comparison_development/final_comparison_summary.csv` | 0.2355/0.2677/0.4459/0.3872 | same | Yes |
| External v1 Macro-F1 ResNet18 Baseline 0.2406 | `confirmatory_summary_all_partitions.csv` | 0.2406 | 0.2456; notebook 12 output shows 0.2406 | Partial (the committed CSV differs by 0.005) |
| External v1 ViT-Small Baseline 0.4157 (README:89) | `confirmatory_summary_all_partitions.csv` | 0.4157 | 0.4346 in CSV; 0.4157 in notebook 12 output | **No** against the CSV; matches notebook 12 |
| External v1 ResNet18 Geometric + Mixed 0.4882, 53.23% | `confirmatory_summary_all_partitions.csv` | 0.4882 / 53.23% | 0.4882 / 0.5323 | Yes |
| External v1 ViT-Small Mixed 0.4808 (README:91) | `confirmatory_summary_all_partitions.csv` | 0.4808 | 0.4710 in CSV; 0.4808 in notebook 12 output | **No** against the CSV; matches notebook 12 |
| Baseline accuracy 30.65% (README:99) | `confirmatory_summary_all_partitions.csv` | 30.65% | 0.3065 | Yes |
| Healthy 0/16 → 12/16 (README:104) | Notebook 12 output | 0/16 → 12/16 | 0/16 → 12/16 | Yes |
| Early Blight 11/13 → 1/13 | Notebook 12 output | 11/13 → 1/13 | 11/13 → 1/13 | Yes |
| Bacterial Spot 1/9 → 5/9 | Notebook 12 output | 1/9 → 5/9 | 1/9 → 5/9 | Yes |
| Notebook 13 Healthy correct, 4 models (README) | Notebook 13 output | 0 / 15 / 44 / 43 of 57 | ResNet 44, ViT recovered 28 (15+28=43) | Yes |
| Notebook 13 Healthy recall 77.2%, 26.3%, 75.4% | arithmetic from the counts above | 77.2/26.3/75.4 | 44/57, 15/57, 43/57 | Yes |
| Five-crop 38/62, Macro-F1 0.5840; standard 32/62, 0.4808 | Notebook 14 output; `docs/research_results_summary.md` | 38/62; 0.5840 | per-class counts give 6+8+4+5+15=38 | Yes (notebook only) |
| Notebook 15 mean ± SD, accuracy and Macro-F1 (README:140-141) | `data/results/notebook15/original_field_test_summary.csv` | Field-only 0.4610 ± 0.0123 / 0.4300 ± 0.0075; Mixed 0.4468 ± 0.0213 / 0.4072 ± 0.0296 | recomputed from per-seed rows with ddof=1: same | Yes |
| "Field-only leads in two of three seeds" | same CSV | 2 of 3 | seeds 42 and 43 favour Field-only; seed 44 favours Mixed (0.4405 vs 0.4349) | Yes |

## Cross-check notes
- C5-1 and C3-2 are consistent. The README's External v1 values rest on notebook-only evidence that conflicts with the committed CSVs. They are credited as verified only through notebook outputs.
- I did not raise findings for the "ViT Mixed beats ResNet on Field Test" claim, because the README does not make it.
- C2-1 and C5-1 partly overlap (missing output folders). I kept both: one concerns implementation completeness, the other concerns number traceability.
- C3-1 and the README's own limitations section agree. The README discloses the test-set reuse, so it is not an honesty violation, but the controlled experiment's conclusions still rest on that reused set.
- I dropped a suspected concern about leakage between lab train and test, because `src/data_prep.py` preserves the original PlantVillage test split and group-splits the training data.

## What I could not verify
- Anything that needs the external v2 images, model checkpoints or the original PlantVillage and PlantDoc images (retraining, Grad-CAM figures, the Streamlit app with real weights).
- Whether Notebooks 12, 13 and 14 reproduce, since their cell outputs are saved evidence only and the notebook 12/13/14 output folders are not committed.
- The Notebook 15 per-image Dev predictions and training logs (the archived Colab notebook exists, but I did not re-execute it).
- The 52 summaries that `docs/submission_audit.md` says it recalculated for Notebook 14, and the claim of "no test-based model selection" (the `field_dev_protocol.json` flag `test_inference_or_test_based_model_selection: false`) beyond what the code and notebooks show.
- The self-collection and licensing provenance of the images.

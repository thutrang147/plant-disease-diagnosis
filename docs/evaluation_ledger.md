# Evaluation ledger: every look at a test partition

Test-set reuse weakens every later claim. This ledger lists each evaluation that touched the historical
**Field Test** (47 images) or **External v3** (486 PlantSeg images) after round 1, with what selected the models
and whether the analysis was confirmatory. Field Dev and Lab Validation are development data and are not listed.

Only **P1–P3 on External v3** were confirmatory: they were pre-registered in
`data/external_v3_plantseg/protocol.json` (commit `8762743`) before the single run in `cfc4752`. Everything after
that is exploratory, made with knowledge of the External v3 results, and without multiplicity control. In
particular, the largest External v3 score in any exploratory table is a maximum over many evaluations and should
not be read as the best model.

| # | Commit | Analysis (script) | Test partitions touched | Models evaluated | Selection data | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `33672f4` | Lab retention (`evaluate_notebook15_lab_retention.py`) | Lab Test only | 7 sealed checkpoints | none (fixed checkpoints) | Descriptive |
| 2 | `cfc4752` | External v3 one-shot run (`evaluate_external_v3_plantseg.py`) | External v3 | 10 sealed/fixed checkpoints | none (fixed checkpoints) | **Confirmatory for P1–P3**; other columns descriptive |
| 3 | `f3e1a30` | Weight interpolation (`evaluate_weight_interpolation.py`) | Lab Test, Field Test, External v3 | 3 operating points × 3 seeds | Lab Val, Field Dev (pre-declared rules) | Exploratory |
| 4 | `bd52233` | Lesion reliance (`evaluate_lesion_reliance.py`) | External v3 (edited images) | 10 checkpoints × 3 conditions | none | Exploratory |
| 5 | `fe7fdb7` | Frozen DINOv2 probe (`evaluate_frozen_foundation_probe.py`) | Lab Test, Field Test, External v3 | 2 backbones × 2 probes | Lab Val / Field Dev (C grid) | Exploratory |
| 6 | `ef20700` | LP-FT architecture check (`explore_lpft_architecture.py`) | Lab Test, Field Test, External v3 | 2 architectures × 2 depths × 3 seeds | Field Dev (epoch) | Exploratory |
| 7 | `a18dbf6` | Error analysis (`analyze_external_v3_errors.py`) | External v3 (saved predictions) | no new inference | none | Descriptive |
| 8 | `82ceb3a` | Lab split duplicate audit (`audit_lab_split_duplicates.py`) | Lab Test | 10 checkpoints | none | Descriptive |
| 9 | `ab02c7e` | Training budget, 24 epochs (`explore_training_budget.py`) | Lab Test, Field Test, External v3 | 2 architectures × 2 depths, seed 42 | Field Dev (epoch) | Exploratory |
| 10 | `dbeaf09` → results commit | Occlusion control (`evaluate_occlusion_control.py`) | External v3 (edited images) | 10 checkpoints | none | Exploratory |
| 11 | `248a4b0` | Secondary-metric intervals (`analyze_external_v3_secondary.py`) | External v3 (saved predictions) | no new inference | none | Post-hoc summary of run 2 |
| 12 | `034ad91` | Same-subset occlusion table (`analyze_occlusion_subset.py`) | External v3 (saved predictions) | no new inference | none | Descriptive |
| 13 | `eede9dd` | Boundary sensitivity, Bacterial Spot + Septoria merged (`analyze_external_v3_label_merge.py`) | External v3 (saved predictions) | no new inference | none | Post-hoc, exploratory |
| 14 | `4a4c8ff` (+ paired re-run) | AdaBN (`evaluate_adabn.py`, `analyze_adabn.py`) | Lab Test, Field Test, External v3 | 8 ResNet18 checkpoints × 2 BN variants | none (protocol declared in `828bbfe`) | Exploratory |
| 15 | `034ad91`, `6a6faca`–`53d3d20` | Field-only vs Mixed at 24 epochs (`explore_mixed_budget.py`, `analyze_mixed_budget.py`) | Lab Test, Field Test, External v3 | 2 arms × 3 seeds (+ Mixed s42 re-run, identical) | Field Dev (epoch) | Exploratory; External v3 a secondary endpoint pre-declared on a reused set (`0d47ea8`, `7c1a280`, `64d749b`) |
| 16 | `0766fb8` | Sealed Notebook 15 checkpoints re-scored (`check_nb15_sealed_field_dev.py`) | none (Field Dev only) | 6 sealed checkpoints | none | Consistency check |

Counting distinct model evaluations on External v3 after the confirmatory run: 9 (interpolation) + 30 (lesion
reliance, three image conditions) + 4 (DINOv2) + 12 (LP-FT) + 4 (budget) + 30 (occlusion control) + 8 (AdaBN; the
8 unchanged checkpoints were re-scored too and match run 2 exactly) + 7 (24-epoch Field-only vs Mixed) = 104.
The historical Field Test was additionally inspected from Notebook 04 onward (see README, Limitations).

## Rules applied when reporting

- Confirmatory claims cite only P1–P3 and their pre-registered metrics. Accuracy is primary; disease macro-F1 is
  secondary. Intervals for the secondary metric were computed afterwards (`data/results/external_v3_secondary_metrics/`).
- Non-significant differences are reported as interval bounds, never as equivalence.
- Exploratory External v3 numbers appear only in exploratory tables, with their selection data stated.

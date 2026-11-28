# Superseded loader – read before using these numbers

The files in this folder were produced by the upstream (teammate) External v1 evaluation, which
converts images to RGB **without applying EXIF orientation**. Eleven predictions on self-collected
Healthy photographs (EXIF orientation 6) differ from the canonical Notebook 11 run, so some
aggregate values here differ from the README:

| Model | This folder (acc / macro-F1) | Canonical Notebook 11 |
| --- | --- | --- |
| ResNet18 · Lab baseline | 0.3065 / 0.2456 | 0.3065 / 0.2406 |
| ViT-Small · Lab baseline | 0.4516 / 0.4346 | 0.4355 / 0.4157 |
| ResNet18 · Geometric + Mixed | 0.5323 / 0.4882 | 0.5323 / 0.4882 |
| ViT-Small · Mixed | 0.5000 / 0.4710 | 0.5161 / 0.4808 |

The 46 online disease predictions are identical under both loaders, so disease-only analyses
(`data/results/supplementary_statistics/`) can use these files. Report the canonical values.
Details: `docs/archive/experiment_reconciliation.md`.

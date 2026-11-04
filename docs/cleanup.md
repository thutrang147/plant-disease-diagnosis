# Repository cleanup record

Cleanup date: 2026-11-04. Scope: organization and documentation only; no model training, split generation, label changes, metric changes, commit, or push.

## Updated directory tree

```text
plant-disease-diagnosis/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── requirements.txt
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/splits/
│   ├── dataset_sources.json
│   ├── label_map.json
│   ├── lab_train.csv
│   ├── lab_val.csv
│   ├── lab_test.csv
│   ├── field_train.csv
│   ├── field_test.csv
│   ├── field_adapt_train.csv
│   ├── field_dev.csv
│   ├── field_dev_protocol.json
│   ├── field_near_duplicate_candidates.csv
│   ├── field_label_conflicts.csv
│   ├── field_quarantined_train.csv
│   └── split_backups/
│       ├── 20261103T072114096478Z/   # Earlier random split/protocol
│       └── 20261103T074311335002Z/   # Earlier exact-grouped split/protocol/quarantine
├── docs/
│   ├── workflow.md
│   └── cleanup.md
├── models/                        # Four local checkpoints, ignored and preserved
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_cnn.ipynb
│   ├── 03_train_vit.ipynb
│   ├── 04_domain_gap.ipynb
│   ├── 05_augmentation_experiments.ipynb
│   ├── 06_field_adaptation_verified.ipynb
│   ├── 07_vit_field_adaptation.ipynb
│   ├── 08_mixed_domain_finetuning.ipynb
│   └── archive/
│       ├── 06_field_adaptation.ipynb
│       └── 06_field_adaptation_grouped.ipynb
├── outputs/                       # Saved files preserved, including historical results
│   ├── comparison/
│   ├── resnet18_domain_gap/
│   ├── vit_small_domain_gap/
│   ├── field_dev_comparison/
│   ├── field_dev_comparison_grouped/
│   └── field_dev_comparison_verified/
└── src/
    ├── __init__.py
    ├── data_prep.py
    ├── dataset.py
    ├── evaluate.py
    ├── evaluate_vit.py
    ├── prepare_field_dev_verified.py
    ├── evaluate_field_dev_verified.py
    ├── models.py
    ├── train.py
    └── archive/
        ├── __init__.py
        ├── prepare_field_dev.py
        ├── prepare_field_dev_grouped.py
        ├── evaluate_field_dev.py
        └── evaluate_field_dev_grouped.py
```

Local `.venv/`, operating-system metadata, and Python caches are omitted from the research tree; none was deleted.

## Inventory and disposition

- Archived two notebook 06 predecessors and their four random/exact-grouped preparation/evaluation scripts with `git mv`. Their split methods are superseded by the saved pHash protocol, not judged unused from their names. No main code imports these scripts. Exact move mappings are in [workflow.md](workflow.md).
- Retained all eight main notebooks, the current verified scripts, original baseline preparation/evaluation code, and the manifest-backed dataset loader at their existing paths.
- Retained all manifests, mappings, duplicate candidate pairs, conflicting labels, quarantine records, protocol metadata, and both timestamped backup sets without modification. Backup files are intentional historical snapshots, not redundant files to remove.
- Retained all six saved result families, including predictions, metrics, reports, plots, confusion matrices, and evaluation protocols. Historical Dev results match their backup manifests by SHA-256. They must not be compared as if evaluated on today's split.
- Retained all four ignored local checkpoints (about 211 MB total). External Drive histories, logs, and checkpoints were not available in this checkout and were not touched.
- Retained `src/models.py`, `src/train.py`, `configs/config.yaml`, and `app/streamlit_app.py` unchanged. They are starter scaffolds with no active research callers found; future use is uncertain, so they were documented rather than removed or rewritten.
- Preserved the pre-existing uncommitted changes in `notebooks/08_mixed_domain_finetuning.ipynb` byte-for-byte. Its large diff belongs to the user and was not staged by cleanup.

## Updated references and dependencies

- Archived scripts discover the repository with `Path(__file__).resolve().parents[2]` to account for the added directory. Usage examples now use `src/archive/` or `src.archive.*`; the grouped evaluator's error path and an archived notebook reference were adjusted.
- The archived grouped notebook's preparation command points to `src/archive/prepare_field_dev_grouped.py`. Its code cells and saved outputs are unchanged.
- Notebook 06 verified had stale explanatory references to grouped preparation/evaluation. Markdown now names the verified scripts and describes the pHash protocol and reuse of existing manifests. No code, outputs, or notebook metadata changed.
- No main research imports, checkpoint filenames, dataset paths, output paths, experiment settings, or label orders changed.
- Added missing direct research dependencies `torch`, `torchvision`, `scipy`, and `ImageHash`; added `ipykernel` for VS Code. Optional `app/requirements.txt` includes the research requirements and Streamlit. Versions were not fabricated or retroactively pinned.
- Extended `.gitignore` for `checkpoints/`, bytecode, notebook checkpoints, caches, and newly generated `outputs/` files. Existing tracked outputs remain tracked. New small result files require intentional `git add -f <specific-file>` after review; avoid force-adding whole output trees or binary checkpoints.

## Validation

- All 14 Python files under `src/`, including archived modules and package initializers, parsed and imported successfully in the existing local virtual environment. Imports ran with Hugging Face offline mode, so no datasets were downloaded. The app source parsed; Streamlit was not installed or launched.
- All 10 notebooks parsed as JSON. Every code cell passed Python syntax parsing after IPython magic transformation. This is a static check, not Colab execution.
- Compared notebook code cells, saved outputs, execution counts, and metadata against pre-cleanup content: unchanged. Notebook 08's complete file hash matches the pre-cleanup snapshot.
- SHA-256 comparisons confirmed every pre-existing file under `data/`, `models/`, and `outputs/` remained present and byte-identical.
- The existing verified evaluator's `validate_manifests` passed for the 370/92/47 partitions, label mapping, row separation, Adapt/Dev hash/group separation, and protocol counts. Current Field Dev SHA-256 matches the verified saved evaluation protocol. Both historical result-protocol hashes match their corresponding split backups.
- Notebook 04 code was executed in a temporary copy using saved predictions. The generated aggregate metrics and paired prediction CSVs match the committed evidence; figures were generated only in that temporary directory. No repository notebook outputs or result files were rewritten.
- Archived executable ASTs match their original implementations after accounting for the repository-root depth adjustment and clarified error text. No scientific logic was rewritten.
- Main notebooks have no remaining references to the superseded grouped script filenames. Current manifest and model paths were retained; external Drive and Colab paths were inspected statically only.
- `git diff --check` and `git diff --cached --check` passed. Ignore checks confirmed new checkpoints/results/caches are ignored, while existing verified summary results remain tracked.

Matplotlib initially emitted font-cache permission warnings in the restricted environment; imports completed successfully. Validation used temporary cache locations. No dependency installation was needed.

## Unresolved issues and intentional limits

Colab training, remote Drive artifacts, dataset downloads, full model inference, and the optional app were not executed or verified. The local analysis check validates notebook 04's code against saved predictions; it is not an end-to-end training reproduction.

The saved source metadata lacks resolved dataset commit SHAs and the original training environment is not locked. Preserve row order and use recorded experiment metadata when recovering historical environments. Adding current package versions would not prove the historical training setup.

Visual adjudication of near-duplicate candidates and label conflicts remains research work. Small Field Dev/Test sets and previous inspection of original Field Test limit confirmatory claims. A new independent field test is still needed for confirmation.

Historical scripts and notebooks retain their old default data/output destinations. Run them only in isolation with the matching historical manifests; their presence in an archive does not make them safe to run against the current verified split.

## Manual review and commit

`git mv` staged the six moves. Subsequent edits and new documentation are unstaged; review both views. The user's notebook 08 remains unstaged and separate from the cleanup.

```bash
git status --short
git diff --stat
git diff --cached --stat
git diff -- README.md .gitignore requirements.txt notebooks/06_field_adaptation_verified.ipynb
git diff -- src/archive notebooks/archive
git diff --cached --find-renames
```

After reviewing new files under `docs/` and `app/requirements.txt`, stage only the cleanup paths:

```bash
git add -- .gitignore README.md requirements.txt app/requirements.txt docs src/archive notebooks/archive notebooks/06_field_adaptation_verified.ipynb
git diff --cached --check
git diff --cached --stat
git commit -m "Organize verified research workflow and preserve historical experiments"
```

Review notebook 08 separately with `git diff -- notebooks/08_mixed_domain_finetuning.ipynb`. No commit or push was performed by cleanup.

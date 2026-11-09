# Documentation audit

Audit date: 2026-11-09. This inventory was written before revising the README or notebook Markdown. Evidence: local source, committed manifests and result tables, and saved notebook outputs; no training or inference was run. The working tree was clean at the start.

| File | Original issue and editorial action |
| --- | --- |
| `README.md` | Lists eight main notebooks and omits final comparison/error analysis. No main-results table or references section. Replace with the requested research overview, separating Lab Validation, Lab Test, verified Field Dev, and previously inspected Field Test. |
| `docs/workflow.md` | Useful provenance, but omits notebooks 09–10 and their external artifacts. Clarify analysis dependencies and distinguish historical runtime records from an environment lock. |
| `docs/cleanup.md` | Dated historical record lists eight main notebooks and an earlier user's uncommitted work. Preserve the record, explicitly identify it as a historical snapshot, and link current documentation. |
| `notebooks/01_data_exploration.ipynb` | Repeated dataset descriptions and generic objectives; claims an RGB configuration although none is specified. Distinguish Hugging Face's single loaded split from original train/test metadata. Replace prevalence language with sample composition; report the limits of incomplete leaf IDs. |
| `notebooks/02_train_cnn.ipynb` | Setup dominates headings; textbook initialization prose and unanswered final reporting prompts. Record the observed validation result, selected epoch, and class recall. |
| `notebooks/03_train_vit.ipynb` | Repeated selection warnings and fragmented interpretation. Consolidate measured validation findings and architecture/preprocessing differences. |
| `notebooks/04_domain_gap.ipynb` | Generic metric explanations and future-work checklist. Present saved test metrics and paired errors as retrospective observations; clarify that execution writes comparison artifacts. |
| `notebooks/05_augmentation_experiments.ipynb` | Procedural preset instructions and unanswered interpretation prompts. Saved training run is geometric; final prose cites a historical 96-image Field Dev result. Label that result historical and separately report current 92-image evidence from the verified evaluation CSV. |
| `notebooks/06_field_adaptation_verified.ipynb` | Stale tentative counts, repeated warnings to rerun evaluations, and an unanswered study-note section. Record 370/92/14 counts and measured adaptation/forgetting. The image audit accesses Field Test fingerprints, so avoid claiming that it never accesses test image content. |
| `notebooks/07_vit_field_adaptation.ipynb` | Repeats protocol rationale and ends with fill-in prompts. Report macro-F1 gain alongside the decrease in accuracy and class-specific trade-offs. |
| `notebooks/08_mixed_domain_finetuning.ipynb` | Incorrect pending-results status, procedural instructions, and proposed additional experiments. Report three completed runs and field-only controls; preserve the unequal field-exposure caveat without implying measured compute equivalence. |
| `notebooks/09_final_model_comparison.ipynb` | Repeated “Purpose” labels and unanswered conclusion questions. Report the fixed four-model comparison; distinguish Field Dev selection from Lab Test evaluation and avoid architecture-wide causal claims. |
| `notebooks/10_error_analysis.ipynb` | Conversational questions and report checklist despite saved error counts. Report paired/class-level failures and high-confidence errors. Visual review template is unfilled; lesion localization and biological explanations remain unverified. |
| `src/dataset.py` | Long introductory example and incomplete argument/return contract. Document manifest assumptions and model-specific transform requirement concisely. |
| `src/data_prep.py` | Existing documentation describes original manifest construction adequately; retain. |
| `src/prepare_field_dev_verified.py` | Introductory command encourages replacing saved manifests; “protection” can sound absolute. Clarify deliberate reconstruction, pHash grouping, and manual-review limits. |
| `src/evaluate.py`, `src/evaluate_vit.py` | Clarify one evaluation pass per invocation and preserved output destinations; test results already exist. |
| `src/evaluate_field_dev_verified.py` | “Near-duplicate-safe” overstates completeness; return contract omits manifest hash. Describe checks against recorded hashes/groups and distinguish them from a new image-content audit. |
| `src/models.py` | Generic factory docstring obscures starter status and ViT-B/16 versus research ViT-Small. Retain in place with explicit scope; no active research import found. |
| `src/train.py` | Docstrings promise configuration loading/training although function only raises `NotImplementedError`. Retain in place and document actual behavior. |
| `configs/config.yaml` | Starter folder-based settings differ from manifest-based notebook configuration. Retain values and add scope comments. |
| `app/streamlit_app.py` | Upload-only placeholder, no model inference. Clarify module responsibility; retain UI and behavior. |
| `src/archive/`, `notebooks/archive/` | Historical random/exact-grouped workflows already identified as superseded. Preserve unchanged; their results cannot be pooled with the verified split. |

## Scaffolding disposition

Retain `src/models.py`, `src/train.py`, and `configs/config.yaml` in this documentation pass. A separate implementation task should decide whether to archive them together or replace them with an entry point extracted from the active notebooks. The current factory is callable but does not reconstruct the research ViT-Small. The training entry point is unimplemented, and YAML is not consumed by it. There is no evidence that these files reproduce the reported experiments. The app likewise remains a demonstration scaffold.

## Scientific documentation gaps

- Resolved dataset commit SHAs and a complete historical dependency lock are absent. Row-index manifests depend on upstream ordering.
- Adaptation, mixed-domain, final-comparison and error-analysis artifacts are primarily referenced on Google Drive. Saved notebook outputs are available, but external files and their complete histories were not inspected in this pass.
- Notebook 06's A2 epoch-6 summary is saved, but its epoch-6 training log line is absent. Report the saved summary without reconstructing missing logs.
- Notebook 10 checks metadata and selected-case predictions, but does not compare both mixed checkpoint hashes against notebook 09's exported record. Exact cross-notebook checkpoint identity remains dependent on preservation of the external files; this documentation pass does not add missing checks.
- pHash grouping is a screening method; candidate duplicates, conflicting labels, and the error-analysis visual review require manual adjudication. No expert lesion-localization findings are recorded.
- Field Dev supports selection, the original Field Test was previously inspected, and no fresh independent field test is recorded. One-seed results lack repeated-run uncertainty estimates; architectures and fine-tuning protocols are not matched for capacity or compute.
- No paper bibliography is present. References in the revised overview identify the actual dataset mirrors and model identifiers already used; a verified scholarly bibliography remains a documentation gap.

## Completed editorial changes and verification

The README now follows the nine requested research sections and reports saved
validation, test, development, augmentation, and adaptation evidence separately.
All 110 Markdown cells across notebooks 01–10 were revised. Report prompts were
replaced with supported observations, historical split results were labeled,
and unverified visual interpretations were left explicit. No experiment was
added. `docs/workflow.md` now covers notebooks 09–10 and their required external
artifacts; `docs/cleanup.md` is marked as a historical snapshot.

Documentation-only changes cover `src/dataset.py`, the three active evaluation
modules, `src/prepare_field_dev_verified.py`, `src/models.py`, `src/train.py`,
`configs/config.yaml`, and `app/streamlit_app.py`. Starter files remain in place
with their original behavior and configuration values.

Verification used the clean starting revision and pre-edit snapshots:

- All ten main notebooks parse as JSON. After masking only Markdown source
  arrays, each notebook is byte-identical to its starting version. Code cells,
  outputs (including embedded figures), execution counts, cell order, IDs,
  attachments, and metadata are unchanged.
- SHA-256 comparisons confirm all 82 pre-existing files under `data/`, `models/`,
  and `outputs/`, plus the two archived notebooks (84 files total), are unchanged. This includes
  manifests, protocols, labels, checkpoints, predictions, metrics, and figures.
- Executable Python ASTs, with docstrings excluded, match for all 15 files in
  `src/` and `app/`. YAML differs only by added comments; values are unchanged.
- Local links in the README and `docs/` resolve, and `git diff --check` passes.
- README and notebook 01–03 diffs were checked before continuing with 04–10;
  source-documentation diffs and saved historical/current CSV values were also
  inspected. No notebook execution, training, inference, data download, split
  generation, commit, or push was performed.

The changed files are the README, ten main notebooks, the eight Python modules
listed above, the YAML comments, and three documents under `docs/` (including
this new audit). All changes remain unstaged for review.

# Final research repository audit

Review date: 27 November 2026. Scope: all 15 canonical notebooks, four archived notebooks, all Python modules/scripts/tests, documentation, committed manifests and available local evidence. No model training, new inference, checkpoint modification, commit, merge, push or history rewrite was performed.

## Decision and distribution boundary

**Technically ready for manual integration, with distribution limitations.** The
cleanup follow-up separates three decisions. The branch can safely fast-forward
into the fetched `main`; the proposed commit introduces no unintended photographs
or private evaluation packages; public redistribution is **not fully cleared**.
Existing reachable history contains 62 external photographs, including 16
self-collected Healthy images and 46 third-party images. Rights/access review is a
separate repository-owner action and does not by itself invalidate the technical
merge relationship. No history was rewritten or image bytes deleted.

No standalone raw/external photograph files are currently tracked. Existing
notebook galleries and derived PlantDoc image figures remain intentional research
evidence, with their underlying rights obligations. A source snapshot without Git
history avoids distributing old commits, but does not automatically clear those
embedded galleries. Do not distribute the entire local directory: it contains
ignored private evaluation packages, source photographs, weights and environments.
The [cleanup handoff](#cleanup-and-integration-handoff) below records the final
scope and commands; the technical review findings are retained.

## Improvements made

- `src/dataset.py`: reject negative, fractional, non-finite and non-numeric row/label IDs before opening source datasets; reject class-name/ID mismatches. Valid historical rows retain the same RGB decoding and transforms.
- `src/evaluate_confirmatory_field.py`: repair an import of two nonexistent helpers; compare exact RGB pixel hashes with stored pixel hashes rather than comparing encoded-file hashes with them; require fingerprint metadata; validate checkpoint architecture/class order before loading a model; correct the historical Mixed filename; report unavailable weights explicitly; refuse an existing CLI output destination. Folder inference applies EXIF orientation before RGB. The audit's raw-RGB Adapt/Dev-only scope is explicit; it does not certify test/pretraining independence or replace Notebook 11.
- `src/statistical_analysis.py`: move generated transcriptions/figures/statistics into a fresh output parent and refuse existing destinations. Rounded notebook transcriptions are labelled as such; original CSVs are preserved. The previous default could overwrite committed evidence despite being described as non-destructive.
- Four canonical notebook edits: add a data-exploration conclusion (01); distinguish committed summary artifacts from external full prediction exports (09); reconcile the external-error conclusion with later exploratory reuse (12); align the title convention (14).
- Two archive edits: identify the retained historical duplicate-validation error (old 06); distinguish the combined predecessor from canonical Notebook 11, correct its archive import, translate the remaining Vietnamese instruction and remove unsupported confirmatory/resolution wording (old 11).
- README: consolidate the research overview, all 15 notebook links, canonical findings, controlled versus historical comparisons, practical verification commands and distribution limits. Workflow documentation now identifies 15 notebooks, six local historical weights and the imported controlled run.
- Add `scripts/audit_repository.py`, declare `nbformat`, and add six regression tests covering loader validation, external-audit correctness and statistical export safety.

No executed metric value, training setting, original notebook output or execution count was changed. All tracked data and result artifacts remain byte-identical to HEAD. The controlled training implementation and original executed Colab notebook remain unchanged. Historical split backups, quarantine records and original prediction exports were retained.

## Individual notebook review

“Static pass” means schema, transformed Python syntax and local `src` imports passed; it does not mean training/inference was rerun. Saved tables, code, conclusions and quantitative figures were inspected individually. Historical angled class labels remain readable and were retained with their saved figures rather than cosmetically rerunning inference.

| Notebook | Purpose | Code status | Presentation | Changes | Remaining execution/evidence limits |
| --- | --- | --- | --- | --- | --- |
| 01 | Dataset composition and grouping | Static pass | Complete, with conclusion added | Markdown conclusion | Hugging Face images required; incomplete leaf grouping and third-party galleries |
| 02 | ResNet18 Lab baseline | Static pass | Complete | Reviewed; retained | Colab CUDA, Drive and dataset access; unpinned Lab source and incomplete environment lock |
| 03 | ViT-Small Lab baseline | Static pass | Complete | Reviewed; retained | Colab CUDA, Drive and dataset access; architecture/pretraining comparison is confounded |
| 04 | Historical baseline domain gap | All code cells executed in temporary checkout | Complete | Reviewed; retained | Writes comparison figures normally; historical 47-image test was already inspected |
| 05 | Augmentation ablation | Static pass | Complete | Reviewed; retained | Saved training execution is geometric; photometric values have separate saved provenance |
| 06 | Verified ResNet field adaptation | Static pass | Complete | Reviewed; retained | Original Drive adaptation histories/weights; development-selected, single-seed observations |
| 07 | ViT field adaptation | Static pass | Complete | Reviewed; retained | Colab/Drive dependencies; selected epoch and target/source trade-off retained |
| 08 | Historical mixed fine-tuning | Static pass | Complete | Reviewed; retained | Unequal Field exposure and independent selection; not a controlled mixing effect |
| 09 | Four fixed model comparison | Static pass | Complete | Artifact-availability correction | Full original four-model prediction exports/protocol are not local; saved matrices/tables remain available |
| 10 | Paired errors and visualizations | Static pass | Complete with qualitative limitation | Reviewed; retained | Requires 09's full prediction files for execution; manual lesion adjudication is unfilled, not a completed result |
| 11 | Canonical external evaluation | Static pass | Complete | Reviewed; retained | Private v1 manifest/images and weights needed for inference; EXIF convention retained |
| 12 | External error trade-offs | All code cells executed with temporary output destination | Complete | Later exploratory reuse clarified | Private manifest and canonical run required; saved historical conclusion contextualized |
| 13 | Healthy and parent/crop analysis | Static pass; saved primary/paired arithmetic recalculated | Complete | Reviewed; retained | Private v2 annotations/images and saved runs; notebook includes inference on 22 additional originals, not rerun |
| 14 | Spatial/crop sensitivity | Cached analysis and gallery cells checked with temporary figure destination | Complete | Title convention | Local inference cache, weights and photos needed; no isolated resolution effect |
| 15 | Controlled three-seed comparison | All report code cells executed from portable committed evidence | Complete | Reviewed; retained | Original training requires Colab/Drive; shared historical test and coupled selection limit interpretation |

The four archives also pass schema/syntax checks. The old unverified 06 retains its saved `ValueError` for exact duplicate overlap, with an explicit supersession note. It is not a failed canonical experiment. The original Notebook 15 Colab execution remains byte-preserved; archives are not recommended for execution against current manifests.

## Scientific and arithmetic checks

Canonical class order is Healthy, Early Blight, Late Blight, Bacterial Spot, Septoria Leaf Spot. All checked manifest labels and stored predictions use IDs 0–4. Laboratory row/group separation passes across 5,365/1,341/1,692 partitions. Adapt/Dev/quarantine account for all 476 original training rows as 370/92/14, with no row overlap against the 47-image Field Test. Stored Adapt/Dev exact hashes and pHash groups are disjoint. These are metadata/fingerprint checks, not proof of physical-plant or pretraining independence.

The canonical external run `20261125_163347_4ca8183f` recalculates to accuracy/macro-F1 0.3065/0.2406, 0.4355/0.4157, 0.5323/0.4882 and 0.5161/0.4808 in the four-model order. ResNet Lab recognizes 11/13 Early Blight images and ResNet Mixed 1/13, while Healthy changes from 0/16 to 12/16. Notebook 11's EXIF-before-RGB protocol remains distinct from archived raw-RGB results.

Notebook 13's primary predictions reproduce 0/57, 15/57, 44/57 and 43/57 Healthy counts. The paired records reproduce 23 originals, 49 crops and 21 scenes per model; original/crop correct counts are 0/0, 3/14, 10/37 and 14/37. Equal-parent crop recall is 0.0000, 0.3529, 0.7558 and 0.8275. These summaries do not turn dependent crops into independent plants.

Notebook 14's 52 protocol/partition/model/regime summaries were recalculated separately from stored predictions. Its canonical external ViT Mixed five-crop result is 38/62 and macro-F1 0.5840, with 6/9 Bacterial Spot recognized. Standard inference gives 32/62, macro-F1 0.4808 and 5/9. Each view remains 224×224. Neither framing nor multi-view aggregation isolates lesion detail or resolution.

Notebook 15's fresh read-only original-artifact audit passes: all six test exports, six Dev exports, 30 per-class rows, full-precision shared-epoch selection, matched per-batch Field identities/tensor hashes, per-row exposure and six selected checkpoint hashes reconcile. All 53 imported file hashes remain unchanged. Seeds 42/43/44 select epochs 8/7/8; full runs receive 2,960 Field presentations and 192 updates per condition. Seed 43's evaluated pair has 2,590 presentations and 168 updates. Field-only macro-F1 is 0.4300 ± 0.0075; Mixed is 0.4072 ± 0.0296, using sample SD before rounding. Mixed leads slightly in one seed and loses in two. No universal inferiority or statistical-significance claim is supported.

## Validation record

- Existing automated tests: 9 passed, 0 failed. After six targeted regression tests: 15 passed, 0 failed.
- All 15 canonical and four archive notebook schemas and 235 transformed code cells checked; no canonical saved error outputs. One retained archived error is explained above.
- All 30 repository Python sources parse and all 21 `src` modules import with Hugging Face offline flags. Optional Streamlit is not installed or launched; Colab/CUDA training environments are not certified.
- Notebook 04 and 15 code cells executed in temporary source checkouts. Notebook 04's exported metric and paired-prediction tables exactly match committed CSV values. Notebook 15 executes from committed evidence without original weights or the full Drive run.
- Notebook 12 analysis and Notebook 14 cached analysis use temporary output destinations; no new model inference is required. Notebook 13 was validated from its saved prediction records rather than rerunning its inference cells.
- The statistical CLI completed in a new `/tmp` destination. The overwrite refusal is covered by a regression test. Its historical hardcoded rounded transcriptions remain distinct from full-precision exports.
- 98 available prediction CSVs (189 separate metric groups) were checked for canonical IDs, correctness flags, finite normalized probabilities, probability argmax and confidence consistency where present. Aggregate metrics were recalculated within separate protocols/partitions/regimes/models/seeds. Canonical external, Healthy, paired and controlled results received additional explicit reconciliation.
- Saved quantitative figures were visually inspected using contact sheets and the full six-panel controlled confusion matrix. Important tables, curves, matrices, galleries and conclusions were preserved. No clipping or misleading metric inconsistency requiring inference was identified.
- Relative Markdown/notebook links were checked. Links to ignored local artifacts are access dependencies, not promises of fresh-clone availability. Historical prose records are labelled as historical rather than rewritten as current facts.
- `git diff --check`, staged whitespace checks, tracked-file inspection and common credential-pattern scanning were performed. No credential match, tracked checkpoint, environment or current external photo file was found. This is a pattern scan, not certification of all possible secrets.

Useful read-only checks from the repository root:

```bash
MPLCONFIGDIR=/tmp/plant-review-mpl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/plant-review-mpl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/audit_repository.py
MPLCONFIGDIR=/tmp/plant-review-mpl HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/audit_repository.py --include-local-results
.venv/bin/python scripts/audit_notebook15_results.py
```

The last command requires the full imported local run and asset bundle; it writes derived audit records outside that run. Missing original artifacts must be reported, not counted as passed. Saved outputs support supervisor reading but do not make a fresh clone an end-to-end reproduction bundle.

## Git and safety record

- Branch: `experiment/controlled-field-mixed`.
- HEAD: `c7c2875435f9521d94d074b8e2fe168c81a89896`.
- Freshly fetched `origin/main`: `9c32a3a27af4d0a8548683657c46c9e2ae6e6c52`.
- Relative to `origin/main`: ahead 2, behind 0; `origin/main` is an ancestor. Local `main` points to the same commit as `origin/main`. No unmerged index entries or conflict markers were found.
- The two research commits add Notebook 15, its preserved execution, portable evidence, implementation/tests and report updates. Notebook 01–14 research remains the historical baseline.
- Initial tree: clean, no staged or untracked work. Audit changes remain unstaged; new audit/test/documentation files are untracked. Nothing was committed or staged.
- Current tracked files include selected third-party dataset galleries in Notebooks 01/05/10 and derived PlantDoc visualizations. Largest tracked notebook is about 7.8 MB; large evidence was retained intentionally rather than stripped for size.
- Commit `59e3bc4` (25 November 2026) introduced the double-nested external package: 62 images, including 16 self-collected Healthy images and 46 online disease images. They were removed from the current tree but remain reachable through existing `main` history. Ignore rules were checked against tracked files as well as local exclusions.

Changed paths for manual review: `README.md`, `requirements.txt`, `docs/workflow.md`, this audit, Notebooks 01/09/12/14, archived 06 and combined 11, `src/dataset.py`, `src/evaluate_confirmatory_field.py`, `src/statistical_analysis.py`, `scripts/audit_repository.py`, and three new regression-test files. No broad `git add .` is recommended because local artifacts are intentionally excluded.

## Final quality gate

| Dimension | Result | Evidence or limitation |
| --- | --- | --- |
| Scientific correctness | PASS WITH LIMITATIONS | Claims match inspected evidence; sampling, historical reuse and confounding are explicit |
| Code quality | PASS | Broken import/hash contract repaired, loader hardened, evidence overwrite prevented; regression tests pass |
| Notebook presentation | PASS WITH LIMITATIONS | All canonical chapters complete; original outputs and labelled archive failures retained |
| Results | PASS | Available original predictions/artifacts reconcile, including the complete controlled run |
| Reproducibility | PASS WITH LIMITATIONS | Lightweight reports work; private data, original Drive exports and historical environment metadata are incomplete for fresh clones |
| Documentation | PASS | Concise overview, current notebook index, artifact access and audit boundaries |
| Git integrity | PASS | Clean starting state; fetched main is an ancestor; no conflicts or history changes |
| Submission readiness | PASS WITH LIMITATIONS | Technical supervisor handoff is prepared; public redistribution of historical photos and existing galleries is not cleared |

The repository owner should resolve whether historical photographs and existing galleries may be shared with the intended recipient and under what access conditions. No destructive history cleanup is proposed or authorized. Technical integration is assessed separately below; a safe Git graph alone does not establish public distribution clearance.


## Cleanup and integration handoff

Follow-up date: 27 November 2026. This cleanup began with the previous audit's
12 modified and five untracked paths; that work was preserved. No notebooks,
model calculations, dataset source files or saved results were changed in this
follow-up. The earlier combined distribution decision is superseded by the three
separate readiness assessments below.

### Changes and retained structure

| File | Cleanup action |
| --- | --- |
| `.gitignore` | Keep evaluation packages private by default; allowlist only four reviewed v2 code/documentation/mapping files; exclude local credential files while allowing `.env.example` |
| `README.md` | Document local validation, deliberately private metadata and the separate technical/distribution decisions |
| `docs/workflow.md` | Explain why builder, validator and build-time report remain adjacent to the dataset; record access and execution requirements |
| `docs/submission_audit.md` | Record this cleanup, verified counts, Git status, rights limitations and manual integration commands |

Four byte-unchanged v2 files are newly eligible for intentional staging:
`README.md`, `build_dataset.py`, `validate_dataset.py`, `class_mapping.json`.
There is no script relocation or duplicated script copy. The builder imports the
adjacent validator and copies both into new packages; Notebook 13 invokes that
validator's current path. The original `validation.json` is included in the
package's hash inventory and records build-time checks that cannot be recreated
by an ordinary validation pass. It remains local with the package.

The v2 manifest contains original annotation/reviewer metadata, so its exclusion
is deliberate. All full manifests, source workbooks, review decisions, crop/parent
provenance, images and build records remain unchanged and local. Public split
manifests and the portable controlled-result records remain tracked. Broad image
extension ignores were avoided so legitimate report figures remain reviewable.

35 ignored disposable files (470,433 bytes: Python bytecode and `.DS_Store` files)
and five empty `__pycache__` directories were removed. The virtual environment and
Git internals were left alone. Source photographs were not deleted. Python caches
can regenerate normally and remain ignored; the cleanup validation used
`PYTHONDONTWRITEBYTECODE=1`.

```text
plant-disease-diagnosis/
├── configs/config.yaml
├── data/
│   ├── splits/                         # versioned canonical splits and history
│   ├── results/notebook15/             # portable prediction evidence
│   └── external_evaluation/
│       ├── tomato_field_eval_v1/       # local canonical package + historical nested copy
│       └── tomato_field_eval_v2/
│           ├── README.md               # eligible for version control
│           ├── build_dataset.py        # eligible; adjacent validator dependency
│           ├── validate_dataset.py     # eligible; no training/inference dependencies
│           ├── class_mapping.json      # eligible; canonical five-class IDs
│           ├── manifest.csv            # private canonical annotations, ignored
│           ├── validation.json         # original build-time report, ignored
│           ├── images/                 # 103 evaluation images, ignored
│           └── provenance/             # inputs, decisions and 32 parent photos, ignored
├── docs/
├── notebooks/                          # canonical 01–15
│   └── archive/                        # historical protocols and original Colab execution
├── scripts/
├── src/
├── models/                             # local, ignored
├── outputs/                            # selected tracked evidence; other runs ignored
├── tests/
├── README.md
└── .gitignore
```

### Current tracking versus history

- Current tracked external evaluation images: **0**. No `git rm --cached` was
  needed or run, and no index changes were staged. Raw dataset images/checkpoints
  are absent from the tracked tree. Public metadata CSVs are intentional.
- Local v1: **124 ignored image files**, the canonical 62 and 62 in the historical
  nested package. The nested package is retained because archived source points
  to it; it was not treated as disposable clutter.
- Local v2: **135 ignored image files**: 103 evaluation images and 32 parent audit
  photographs. All 259 local v1/v2 image paths pass actual `git check-ignore -q`.
  No additional photos needed new ignore coverage; the new rules expose only the
  reviewed non-image files and strengthen credential exclusion.
- Historical addition: commit
  `59e3bc4903462d121ba7e1675de427ecf48bb51a` (25 November 2026), under
  `data/external_evaluation/tomato_field_eval_v1/tomato_field_eval_v1/images/`.
  It contains 16 Healthy/self-collected and 46 online disease images. All 62
  historical paths still exist locally as ignored files; none is available only
  in history on this machine. They are absent from the current tracked tree.
- `origin/main` already contains that introduction commit as an ancestor. Merge
  `9c32a3a27af4d0a8548683657c46c9e2ae6e6c52` removes these paths relative to its
  upstream image-bearing parent; it preserves their ancestor history.
- Current image-bearing research evidence includes galleries in canonical
  Notebooks 01 (cells 11/17/24), 05 (cell 8), 10 (cells 14/22), archived grouped
  06 (cell 22), and three committed gallery/Grad-CAM PNGs under
  `outputs/error_analysis_development/`. Notebook 15's image outputs are numeric
  confusion matrices, not photographs. Notebooks 12–14 link local external
  galleries rather than embedding external photo bytes. No Markdown document
  embeds raw external photographs. These distinctions prevent mistaking all PNG
  result figures for private image datasets.

Third-party manifests explicitly record `redistribution_rights=not_established`;
attribution or a university/extension website does not establish permission.
The owner can document adequate permission/licenses and appropriate access for
existing material, or distribute a separately reviewed source snapshot/new
repository without old history, also reviewing embedded galleries. If removal
from existing shared history is ultimately necessary, it requires a separately
authorized, coordinated remediation with collaborators and copies/forks considered.
No history-rewrite or force-push command is part of this handoff.

### Cleanup validation

- Existing v2 validator: **PASS**, 1,800 assertions, 103 images, 59 native
  crop-coordinate/pixel checks, 61 retained-v1 byte checks. Composition remains
  8 Healthy originals + 7 historical Healthy crops + 42 new approved Healthy
  crops + 46 online disease images. Four accepted-but-held crops remain excluded;
  no decision, label, crop coordinate or split was changed.
- The 153-file v2 package inventory has **zero hash mismatches**. All **1,037**
  protected research files across data/models/notebooks/outputs are byte-identical
  to the start of this follow-up, including all 15 notebooks and Notebook 15's
  predictions and original Colab execution.
- Automated tests: **15 passed, 0 failed**. Repository audit: **PASS**, all 19
  notebook schemas, 235 code-cell syntax checks, 30 existing project Python files,
  21 `src` imports and 98 saved prediction CSVs. Both newly eligible dataset
  scripts additionally pass syntax checks; the builder's `--help` works without
  building anything. The validator's actual CLI was executed; the builder itself
  was not run.
- No relocated paths or broken local Markdown/notebook links. Notebooks 11–15,
  utilities and docs continue to reference their original data/run paths. Full
  private-package access remains required for those evaluation workflows.
- Ignore checks pass for 259 actual photo paths, 12 private/cache/checkpoint
  examples and 10 intended public source/metadata examples. Canonical manifests
  under `data/splits/` and required configuration files are not accidentally
  excluded. Credential-pattern scan of all candidate files finds no matches;
  newly eligible files have no private user/mount absolute paths. Historical
  experiment outputs retain their recorded paths rather than being rewritten.
- No new photograph, checkpoint binary, environment, raw Colab archive or temporary
  notebook is eligible for this commit. The original executed Colab notebook in
  the archive is intentional evidence, not a temporary download.

### Final readiness and actual Git state

| Scope | Decision |
| --- | --- |
| Technical merge readiness | **Ready**: fetched `origin/main` is an ancestor; no results/code will be lost by the documented fast-forward path |
| Proposed commit contents | **Ready with documented limitations**: no unintended photo/private data additions; existing intended dataset galleries retain rights obligations |
| Historical distribution | **Unresolved**: 62 historical external photographs remain reachable; existing image-bearing galleries also need appropriate redistribution rights |

Branch: `experiment/controlled-field-mixed`; HEAD:
`c7c2875435f9521d94d074b8e2fe168c81a89896`. Fetched `origin/main` and local `main`:
`9c32a3a27af4d0a8548683657c46c9e2ae6e6c52`. Ahead **2**, behind **0**. No unmerged
index entries. There are **13 modified** tracked paths and **9 untracked** intended
source/documentation/test paths (22 paths total), all unstaged. Twelve modified
paths and five untracked paths came from the preceding audit; this cleanup adds
`.gitignore` and makes the four existing v2 support files eligible, while updating
the existing documentation. No commit, push, merge, PR, author configuration or
shared history change was made.

### Manual Git commands

The explicit stage list includes the preceding audit fixes and this cleanup. It
excludes photos, full manifests, provenance, local validation snapshots, weights
and generated runs. Review the staged names/diff before executing the commit.
These are manual instructions; they were not executed by the agent.

```bash
git switch experiment/controlled-field-mixed
git add -- \
  .gitignore \
  README.md \
  data/external_evaluation/tomato_field_eval_v2/README.md \
  data/external_evaluation/tomato_field_eval_v2/build_dataset.py \
  data/external_evaluation/tomato_field_eval_v2/class_mapping.json \
  data/external_evaluation/tomato_field_eval_v2/validate_dataset.py \
  docs/submission_audit.md \
  docs/workflow.md \
  notebooks/01_data_exploration.ipynb \
  notebooks/09_final_model_comparison.ipynb \
  notebooks/12_external_error_analysis.ipynb \
  notebooks/14_lesion_resolution_study.ipynb \
  notebooks/archive/06_field_adaptation.ipynb \
  notebooks/archive/11_external_field_and_resolution_eval.ipynb \
  requirements.txt \
  scripts/audit_repository.py \
  src/dataset.py \
  src/evaluate_confirmatory_field.py \
  src/statistical_analysis.py \
  tests/test_dataset.py \
  tests/test_external_audit.py \
  tests/test_statistical_export.py
git diff --cached --check
git diff --cached --name-only
git diff --cached --stat
git diff --cached
```

After reviewing those exact staged changes, commit and integrate:

```bash
(
  set -e
  test "$(git branch --show-current)" = "experiment/controlled-field-mixed"
  git diff --cached --check
  git commit -m "Finalize research audit and safe evaluation package tracking"
  git fetch origin
  git switch main
  git merge --ff-only origin/main
  git merge --ff-only experiment/controlled-field-mixed
  git status --short --branch
)
```

If either fast-forward check fails, stop and inspect the newly fetched history;
do not force a merge or overwrite another contributor's work. Both the research
commit and any newly fetched work remain preserved on their respective branches.
Publishing is a separate repository-owner action; no push or history-rewrite
command is included in this handoff.

### Final Git verification

Remote references were fetched again for the final handoff. Branch, HEAD,
`origin/main`, the 2-ahead/0-behind relationship and the 22-path stage list above
remain unchanged. The real index is empty and has no unmerged entries. All 259
local evaluation/parent image files pass ignore checks; no evaluation file is
currently tracked. The four allowlisted v2 support files are the only evaluation
package paths in the proposed stage list. No standalone photograph, private
manifest, provenance workbook, model weight or cache is included.

The changed notebooks retain every pre-existing saved output byte-for-byte at
the JSON output-object level; no external photograph output was introduced.
Credential-pattern checks find no matches in the 22 candidates, and the new
untracked files introduce no private user/mount absolute paths. Existing intended
PlantVillage/PlantDoc galleries and the 62 photographs in ancestor history are
unchanged; this verification does not grant redistribution rights for them.

The final read-only repository audit and whitespace checks passed. Dataset and
automated-test results above remain the completed validation record; no training,
inference or additional dataset generation was run for this final Git check.

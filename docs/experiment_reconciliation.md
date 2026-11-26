# Experiment reconciliation — teammate integration, 26 November 2026

## Decision and integration scope

Selective integration is applied on `integration/research-reconciliation-20261126`, based on local HEAD `5d6b2ee`. No remote merge, commit, stash, reset or push was performed. Existing local changes were retained. `origin/main` was fetched to `3c87a91` and is five commits ahead of the original local HEAD. Those commits are `59e3bc4` (external package/audit/notebook), `7141025` (dataset wrapper fix), `caafa06` (executed results/README), `2724d31` (PR #3 merge), and `3c87a91` (merge).

The remote work adds `11_external_field_and_resolution_eval.ipynb`, replaces `src/evaluate_confirmatory_field.py`, changes README, and adds 13 result artifacts plus a nested 62-image dataset and manifest. It does not alter notebooks 01–10, the established split CSVs or model weight files. The combined remote notebook overlaps local Notebook 11 and includes checkout/pull commands, fixed Colab paths and a fixed output directory. Its narrative and class/failure analysis are incomplete. **Case B, with justified Case C corrections:** retain its methods in a focused Notebook 14, correct orientation/provenance handling and scientific interpretation, and preserve its original outputs.

New active code is `src/evaluate_resolution_sensitivity.py` and `src/reconcile_research_results.py`. The former implements the same four methods with strict frozen-weight checks and view-level provenance. The latter aligns IDs, recomputes metrics/statistics and extends the duplicate screen. The existing confirmatory script and all notebooks 01–13 are retained byte-for-byte; adopting the upstream replacement would remove existing APIs and introduce a nested default path. No textual conflicts were merged. README receives targeted updates rather than the upstream rewrite.

Local evidence root: [`outputs/research_integration/20261126_reconciliation/`](../outputs/research_integration/20261126_reconciliation/). `upstream/` preserves the 16 non-dataset remote files, including the original notebook, results and script; `upstream_manifest.csv` preserves the remote CSV. Images were compared via Git blobs without checking them out or duplicating them. These ignored evidence files are required for a fresh rerun of the reconciliation, alongside the existing local reference outputs.

## Dataset identity

The local manifest SHA-256 is `1e775095a904f2b9c2c2f4000b89b1f1b1212cf8f8b26d504dd08b407ed2d329`; the upstream manifest is `bf59791f760124d9aa1ea0a64989efc511447c6f39265a303a8a68d8ad8943db`. After CRLF-to-LF normalization their bytes are identical. All 62 image IDs, row order, labels, provenance fields and image file hashes match; all actual local bytes also match the remote image blobs. See [`image_identity.csv`](../outputs/research_integration/20261126_reconciliation/image_identity.csv).

Both protocols use Healthy=0, Early Blight=1, Late Blight=2, Bacterial Spot=3 and Septoria Leaf Spot=4. External v1 comprises 16 self-collected Healthy and 46 online disease images: 13 Early Blight, 11 Late Blight, 9 Bacterial Spot and 13 Septoria Leaf Spot. It is not a 62-image self-collected field set. Its selection history and source/class confounding prevent a broad claim of independent field generalization.

The upstream duplicate report actually checks **476** PlantDoc images (370 adaptation train + 92 development + 14 quarantine), omitting the 47-image test despite claiming all partitions. We reran the same 64-bit DCT pHash rule (distance ≤4) and exact RGB pixel screen against all **523** unique PlantDoc rows, under both external orientation conventions. There were no exact or threshold matches; minimum pHash distance was 16. Within-external screening also found no threshold matches. See [`extended_duplicate_screen.csv`](../outputs/research_integration/20261126_reconciliation/extended_duplicate_screen.csv). This screen is not proof of physical independence, image licensing or absence of transformed duplicates. It does not cover PlantVillage or ImageNet pretraining sources.

## External result reconciliation

The root cause of label disagreements is **EXIF orientation**. Notebook 11 applies `ImageOps.exif_transpose` before RGB conversion; upstream applies RGB conversion without transposition. Running both loaders with identical local frozen weights reproduces both sets of labels. Standard-inference probability vectors differ from the corresponding saved runs by less than 0.000004 (upstream rounds to six decimals). Accuracy and macro-F1 use the same five-label definition; averaging differences or changing the metric definition is unnecessary.

| Model | Metric | Notebook 11 | Upstream | Recomputed EXIF | Upstream − ours | Canonical |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 · Lab baseline | accuracy | 0.3065 | 0.3065 | 0.3065 | 0.0000 | 0.3065 |
| ResNet18 · Lab baseline | macro_f1 | 0.2406 | 0.2456 | 0.2406 | 0.0050 | 0.2406 |
| ViT-Small · Lab baseline | accuracy | 0.4355 | 0.4516 | 0.4355 | 0.0161 | 0.4355 |
| ViT-Small · Lab baseline | macro_f1 | 0.4157 | 0.4346 | 0.4157 | 0.0189 | 0.4157 |
| ResNet18 · Geometric + Mixed | accuracy | 0.5323 | 0.5323 | 0.5323 | 0.0000 | 0.5323 |
| ResNet18 · Geometric + Mixed | macro_f1 | 0.4882 | 0.4882 | 0.4882 | 0.0000 | 0.4882 |
| ViT-Small · Mixed | accuracy | 0.5161 | 0.5000 | 0.5161 | -0.0161 | 0.5161 |
| ViT-Small · Mixed | macro_f1 | 0.4808 | 0.4710 | 0.4808 | -0.0098 | 0.4808 |

The recommended canonical values are Notebook 11's EXIF-corrected results, consistent with display orientation and Notebook 13's overlapping retained samples. The upstream results remain valid records of a different loader. The mixed ResNet labels match across loaders, but its probabilities are not necessarily identical. Even unchanged accuracy can hide changed labels or confidence.

Exact label disagreements are below (label IDs follow the mapping above); all have EXIF orientation 6. No other image labels differ.

| model_key | image_id | original_filename | our_label | upstream_label | exif_orientation |
| --- | --- | --- | --- | --- | --- |
| resnet_lab | SC_0001 | IMG_3735.JPG | 2 | 1 | 6 |
| resnet_lab | SC_0005 | IMG_3762.JPG | 2 | 1 | 6 |
| resnet_lab | SC_0007 | IMG_3778.JPG | 2 | 1 | 6 |
| resnet_lab | SC_0009 | IMG_3790.JPG | 2 | 1 | 6 |
| vit_lab | SC_0004 | IMG_3749.JPG | 2 | 4 | 6 |
| vit_lab | SC_0005 | IMG_3762.JPG | 4 | 0 | 6 |
| vit_lab | SC_0009 | IMG_3790.JPG | 2 | 4 | 6 |
| vit_lab | SC_0016 | IMG_3805.JPG | 4 | 1 | 6 |
| vit_mixed | SC_0004 | IMG_3749.JPG | 0 | 3 | 6 |
| vit_mixed | SC_0007 | IMG_3778.JPG | 0 | 3 | 6 |
| vit_mixed | SC_0009 | IMG_3790.JPG | 2 | 0 | 6 |

Canonical Healthy recognition is ResNet Mixed **12/16** and ViT Mixed **13/16**; upstream reports **12/16 for both**. Both mixed models recognize **5/9** Bacterial Spot images under standard inference. Detailed rows, metrics, differences and evidence paths are in [`metric_reconciliation.csv`](../outputs/research_integration/20261126_reconciliation/metric_reconciliation.csv), [`prediction_disagreements.csv`](../outputs/research_integration/20261126_reconciliation/prediction_disagreements.csv) and [`probability_reproduction.csv`](../outputs/research_integration/20261126_reconciliation/probability_reproduction.csv).

## Checkpoints, preprocessing and reproducibility

The local frozen weights, verified against Notebook 11, are:

| Model | Local checkpoint | SHA-256 |
| --- | --- | --- |
| resnet_lab | models/resnet18_lab_best.pt | 65d53766258f26cb505cce4c319c3f783cb0f3c6ef4608680906919731f2782d |
| vit_lab | models/vit_small_lab_best.pt | 42c26a8aa70311d9504d55dda422f20e2ea5f0a5878f5922b9f855c052a18c2e |
| resnet_mixed | models/resnet18_geometric_mixed_best.pt | 20bf844b4bc931f4b891b1ffc6c1c59cf178d1b7cf946667ad96e865b03868c7 |
| vit_mixed | models/vit_small_mixed_best.pt | 4a064e328502737dee736bc88b76952f21d0d4163ac6ad251ba41e7703f4f986 |

The source geometric ResNet checkpoint has SHA-256 `51266f98875ba67402c547a43d78247f2e2d0fd08f161256607d4a7612e0690c`; ViT Mixed uses the ViT laboratory source above. Architectures are torchvision ResNet18 with a five-class head and timm `vit_small_patch16_224.augreg_in1k`. Strict state-dict loading, evaluation mode, label metadata, mixed experiment identity, source linkage and development/validation split hashes are checked. **Upstream did not record checkpoint SHA-256**. Matching filenames and reproduced outputs support consistency with these weights; they do not prove byte identity of the teammate's original files.

Standard ResNet inference uses ImageNet1K V1 resize-short-side 256, bilinear interpolation, center crop 224 and ImageNet normalization. ViT uses the saved AugReg preprocessing: bicubic resize-short-side 248, center crop 224, crop fraction 0.90 and mean/std 0.5. The new run uses CPU float32, four threads, seed 42, Torch 2.15.0, torchvision 0.29.2, timm 1.0.26, NumPy 2.5.4 and Pillow 13.0.0. All per-image probabilities, confidence values, pixel hashes, split and Arrow cache hashes, transforms, model hashes and per-view crop boxes are saved under [`inference/`](../outputs/research_integration/20261126_reconciliation/inference/). Upstream environment provenance is incomplete.

PlantDoc is read from the existing cached `geraldmc/plantdoc-full` Arrow data corresponding to the established wrapper's `v0.1.0` request, cache key `66df4b49a58d9c676aa32d9ab0c0d3fff56fb8ac`. Row indices and original class strings are checked. No network refresh or split change occurs. Cache and image pixel hashes are recorded; a newly resolved immutable Hub commit is still unavailable.

## PlantDoc Field Test, n=47

Rerun accuracy / macro-F1 matches upstream: ResNet Lab **17/47, 0.3617 / 0.2802**; ViT Lab **21/47, 0.4468 / 0.3617**; ResNet Mixed **19/47, 0.4043 / 0.3587**; ViT Mixed **25/47, 0.5319 / 0.5103**. The baseline values also match Notebook 04. Upstream saved only aggregate test results; the new run supplies ID-aligned per-image evidence. Field Test was previously inspected, so this is additional evaluation of frozen checkpoints on a fixed historical split, not a fresh independent final test.

## Statistical claims

For ResNet Mixed versus ResNet Lab on the upstream external protocol, the paired table is:

| | Lab correct | Lab wrong |
| --- | ---: | ---: |
| Mixed correct | 9 | 24 |
| Mixed wrong | 10 | 19 |

The exact two-sided binomial McNemar test uses 34 discordant cases and gives **p=0.0243065**, reported as 0.0243. ID sets and truth labels are identical. The same correctness table and p-value hold for the canonical orientation protocol. This is an unadjusted paired comparison on this selected sample; it does not establish repeat-run or population-wide superiority. The eight upstream external/Field Test paired tables and rounded p-values reproduce. All paired counts are in [`paired_statistics.csv`](../outputs/research_integration/20261126_reconciliation/paired_statistics.csv).

The upstream confidence claim means **the share of errors** with maximum softmax ≥0.90: ResNet Lab **26/43=60.5%**, Mixed **0/29=0%**. It is not the error rate among confident predictions; those denominators give **26/37=70.3%** and **0/7=0%**. Canonical orientation changes the baseline to **27/43=62.8%**, with **27/38=71.1%** error among confident predictions; Mixed remains **0/29**, with **0/5** among confident predictions. See [`confidence_denominators.csv`](../outputs/research_integration/20261126_reconciliation/confidence_denominators.csv). Maximum softmax is not calibrated correctness probability, and fewer confident errors alone does not establish calibration.

## Resolution study: verified method and limits

The upstream four strategies are retained, without invented lesion localization:

- Full scene: architecture-specific standard transform, not a direct 224-square warp.
- Center 0.70 and 0.50: source width/height multiplied by the fraction and rounded, minimum 16 pixels; integer-centered box; then the standard transform.
- Five-crop: bicubic **square** 448×448 resize, then 224×224 boxes `(0,0,224,224)`, `(224,0,448,224)`, `(0,224,224,448)`, `(224,224,448,448)`, `(112,112,336,336)`. Coordinates are right/bottom exclusive. The square canvas changes non-square aspect ratios. Average the five float32 softmax vectors, then argmax. Every forward pass is 224×224.

All 32 upstream accuracy, macro-F1, Bacterial Spot counts and confusion-count rows reproduce to saved precision. The new run also includes 16 EXIF-corrected external counterparts and four Field Test standard evaluations. Some five-crop mean class probabilities differ by up to **0.000423852**; the original and reproduced values remain side-by-side in [`resolution_probability_residuals.csv`](../outputs/research_integration/20261126_reconciliation/resolution_probability_residuals.csv). The cause is **unresolved** because upstream view/image probabilities, exact weights and full environment metadata were not saved. The matching class counts and metrics do not justify claiming bitwise reproduction.

The corrected external standard ViT Mixed result is 32/62, macro-F1 0.4808; five-crop remains 38/62, macro-F1 0.5840, Bacterial Spot 6/9. Field Dev ViT Mixed changes from 3/18 to 5/18 Bacterial Spot with center 0.50; ResNet Mixed changes from 3/18 to 2/18 and overall macro-F1 0.4459 to 0.3603. Removing off-center leaves is a hypothesis, not a verified causal explanation. Five-crop also changes coverage, aggregation, aspect ratio and object scale. No lesion masks or lesion-size measurements support a claim that higher model input resolution recovers lesions.

The external outcomes were already examined when these policies were compared. All sensitivity results are exploratory and no policy is chosen for a new final test. Notebook 13's human-reviewed Healthy crops are a different intervention and population; their results remain unchanged.

## Preservation, organization and remaining evidence

The pre-edit inventory hashes 833 existing files across datasets, weights, notebooks, outputs and dirty source files. Final checks compare these bytes, allowing only the requested additive `.gitignore` edit. No training, model selection, dataset rebuild or old-results overwrite occurred. Notebook 14 executes all nine code cells, with native quantitative figures and a linked local-only qualitative gallery. Photos are not embedded in the saved notebook. The two Markdown reports and README provide the research entry points.

The remote commits already contain third-party photos under a double-nested v1 folder. This branch does not merge or check out those files. `.gitignore` now also covers recursively nested evaluation `images/` folders; the current index contains no evaluation image files. Ignoring files cannot repair the existing remote history. Repository owners still need to review rights and any upstream remediation separately; no history rewrite was attempted.

Keep the combined upstream notebook and replaced script as archived evidence under the ignored run, rather than introducing a second active Notebook 11 or deleting history. Remaining gaps are upstream checkpoint/environment provenance and multi-crop probability residuals, source licensing and pretraining overlap, immutable dataset revisions, human review of lesion/coverage mechanisms, and a genuinely new prospective evaluation if generalization claims are needed.

Reproduction commands (project root; local data/weights/evidence required):

```bash
.venv/bin/python -m src.evaluate_resolution_sensitivity --output outputs/resolution_sensitivity/NEW_RUN
.venv/bin/python -m src.reconcile_research_results
```

The first refuses existing output directories. The second revalidates the preserved integration run. Both avoid training and mutation of source data. Notebook 14 reads the recorded inference cache by default; its methodology and figures can be rerun without repeating inference.


## Main publication preparation — 26 November 2026

After research commit `f05af78`, local `main` was fast-forwarded to `origin/main` at `3c87a91`. A merge of the research branch was prepared without creating a commit. The README conflict was resolved using the verified unified research narrative. The original combined evaluation notebook is archived under `notebooks/archive/`; the incoming audit script is preserved as `src/archive/evaluate_confirmatory_field_upstream.py`, while the active script retains its verified local API. Notebooks 01–14, their saved results, datasets and weights are preserved.

The nested evaluation package introduced upstream is removed from the index while its files remain on disk. The complete private evaluation directory is ignored, including annotation workbooks and provenance records. The upstream commits remain ancestors of `main`; removing files from the current tree does not erase photographs already present in remote history. The final merge commit and push are left to the repository owner.

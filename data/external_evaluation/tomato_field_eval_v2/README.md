# Tomato Field Evaluation v2

Finalized from the saved human crop decisions on 26 November 2026. No models were run. Version 1 and its historical results are unchanged.

## Composition

| Evaluation subset | Images | Source |
|---|---:|---|
| `healthy_field_originals` | 8 | Previously selected v1 self-collected original photographs |
| `healthy_leaf_crops` — historical | 7 | v1 crops retained under the user's explicit authorization to preserve historical `keep` approvals |
| `healthy_leaf_crops` — newly accepted | 42 | Explicit `accept` decisions with no unresolved recorded holds |
| `online_disease_images` | 46 | Existing v1 online disease images, copied without modification |
| **Total** | **103** | Manifest-listed evaluation samples only |

The final review workbook contains **46 accept, 19 reject, 0 pending and 0 fix_crop** decisions. Read from the saved literal cells of `New candidates`, not formula caches. Four accepts remain excluded because the preparation status still says `technical discussion hold` and the human review notes do not resolve it. Their accept decisions are preserved verbatim in `provenance/review_dispositions.csv`:

- HLV3_0018 (IMG_3748.JPG): unresolved technical discussion hold.
- HLV3_0043 (IMG_3776.JPG): unresolved technical discussion hold.
- HLV3_0084 (IMG_3796.JPG): unresolved technical discussion hold.
- HLV3_0146 (IMG_3805.JPG): unresolved technical discussion hold.

The review workbook does not provide a human reviewer name or review date on the decision rows; those fields remain blank. Original source annotators are retained separately and are not presented as crop reviewers. No human decision was rewritten or automatically promoted.

Historical v1 `keep` is recorded verbatim as `human_review_decision=keep`, `approval_basis=historical_v1_keep`. It is an explicitly authorized historical approval basis, not a fabricated new `accept` entry. The explicit literal `accept` test applies to all newly included Healthy crops. Online historical crops are disease images and are outside that Healthy approval test.

- Historical exclusion: SC_0013 (IMG_3798.JPG), originally `keep`: current exact-match v3 target requires technical review: HLV3_0106 | current unresolved duplicate group: possible_scene_P031_P032. The file remains untouched in v1.

IMG_3798/P031 versus IMG_3799/P032 is still an unresolved scene relationship. No related leaf crop is included. IMG_3782/IMG_3783 (P020) annotation uncertainty remains held and absent. Healthy labels derive exclusively from `class_name=Healthy` and `source=self_collected` in the original annotation workbook; no visual disease suspicion was used to relabel or exclude a source.

## Independence and intended evaluation

There are 76 distinct source photograph identifiers overall, including 30 self-collected source photographs. Self-collected samples span 28 distinct recorded plant IDs and 28 recorded scene groups. These identifiers are not a verified count of independent physical plants. Different leaf crops and original/crop pairs share source and scene context. Use `scene_group_id` and `parent_image_id` for clustered analysis; never treat each crop as an independent plant.

The Healthy field-original and Healthy leaf-crop subsets describe different image conditions. They support a field-context Healthy check and a human-annotated Healthy leaf-level check. Online disease images provide a separate source stratum; they are not self-collected field photographs. A combined five-class table would mix sources and image conditions and must not be described as an independent five-class field test or laboratory-confirmed health.

The newly accepted crops are visually distinct targets according to the review records. Exact file hashes and recorded leaf-group IDs are unique. Neighboring target rectangles may share background or leaf context; geometric intersection alone is not a repeated-leaf verdict. The source eligibility, historical overlap flags, possible duplicate groups and current preparation holds are applied before selection. Biological independence beyond the recorded groups has not been established.

## Metadata and portability

`manifest.csv` uses canonical class IDs: Healthy 0, Early Blight 1, Late Blight 2, Bacterial Spot 3, Septoria Leaf Spot 4. All file paths are relative to this directory. V1 image IDs and all existing attribution/verification fields are retained for reused records. New image IDs preserve the HLV3 crop IDs.

`is_crop` describes actual image preparation for both self-collected and online images. The 10 previously cropped online images remain in `online_disease_images`; they do not become Healthy leaf samples. Every crop has an original parent, zero-based `[left,top,right,bottom)` coordinates in EXIF-oriented RGB pixels, image/parent SHA-256 hashes and a portable parent reference. Saved evaluation images retain their exact bytes; no resizing or enhancement was performed.

`provenance/parents/` contains raw parent photographs required to reproduce crop pixels when a parent is not already in `healthy_field_originals`. They are audit inputs, **not additional evaluation samples**, and are not counted in the manifest total. The two original workbooks and relevant CSV snapshots are copied unchanged under `provenance/inputs/` so provenance and decisions can be checked after relocation. See their SHA-256 values in `provenance/build_config.json`.

## Validation and reproduction

Run with the project environment (Python 3 and Pillow only):

```sh
python validate_dataset.py .
```

The validator reads the actual decision cells and original annotations, recomputes the accepted/held selection, verifies class/subset membership and parent/group links, loads every image, checks every SHA-256 hash and crop's native source pixels, tests unique evaluation IDs/files/leaf groups, verifies the complete online image set and checks v1 attribution against its manifest snapshot. `validation.json` additionally records the build-time checks that original v1, review-package files, notebooks and checkpoints were not changed.

To reproduce into a **new, nonexistent** directory, using the archived v1 mapping CSV:

```sh
python build_dataset.py --raw-root /path/to/tomato_field_raw \
  --v1-root /path/to/tomato_field_eval_v1 \
  --mapping provenance/inputs/v1_image_mapping.csv \
  --historical-policy retain-keep --repo-root /path/to/plant-disease-diagnosis \
  --output /path/to/a/new/tomato_field_eval_v2
```

The builder refuses to overwrite an existing dataset. Inputs must match the completed review and original annotations; both workbook snapshots and review dispositions make the selection inspectable. No training or inference dependencies are imported.

## Attribution, copyright and Git safety

All 46 online evaluation files are byte-identical to v1. `source`, `source_url`, `original_image_url`, `source_group_id`, verification status and historical review metadata are retained. The unmodified v1 manifest and original annotation workbook preserve all supplied attribution metadata. There was no explicit redistribution license in the v1 manifest; `copyright_license` remains blank and `redistribution_rights=not_established` for online images. Source attribution does not grant redistribution rights, and no new license is asserted.

The repository ignores both `tomato_field_eval_v2/images/` and `tomato_field_eval_v2/provenance/parents/`, including all third-party image bytes. Do not force-add or publicly redistribute third-party photos. Local validation and build files may be reviewed independently. No commit or push was performed.

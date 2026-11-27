# Tomato Disease Classification under Laboratory-to-Field Domain Shift

This research compares ImageNet-pretrained ResNet18 and ViT-Small on five tomato leaf categories. High laboratory accuracy can conceal poor transfer to heterogeneous photographs. The project examines this domain shift, supervised adaptation, laboratory rehearsal during fine-tuning, external errors, and sensitivity to image framing.

The completed study contains 15 canonical notebooks. Start with [09: fixed-model comparison](notebooks/09_final_model_comparison.ipynb), [11: canonical external evaluation](notebooks/11_external_evaluation.ipynb), and [15: controlled fine-tuning](notebooks/15_controlled_field_mixed_finetuning.ipynb). The [research summary](docs/research_results_summary.md) provides detailed results and scientific limitations; the [final audit](docs/submission_audit.md) records validation and submission constraints.

## Objectives and methodology

The study asks how laboratory models transfer to PlantDoc; whether augmentation and supervised adaptation improve target-domain performance; how mixed-domain training affects source retention; and how errors change across classes, acquisition sources and crop policies. A final controlled comparison tests laboratory rehearsal with matched Field exposure and optimizer updates.

ResNet18 uses torchvision `IMAGENET1K_V1`; ViT-Small uses timm `vit_small_patch16_224.augreg_in1k`. Both receive 224×224 inputs with their own recorded transforms. The laboratory baselines use seed 42, three head epochs and five full-model epochs, selecting by Lab Validation macro-F1. Different pretraining, transforms, capacities and optimization settings prevent attribution to architecture alone.

Historical field-only adaptation trains the head and final block, selecting on Field Dev. Historical mixed runs sample 185 Field and 185 Lab images per epoch, versus 370 Field images for field-only adaptation. These runs have unequal Field exposure and independent selection/stopping; their differences are descriptive. Notebook 15 separately controls initialization, Field exposure, updates and shared-epoch selection across seeds 42, 43 and 44. Mixed still receives additional Lab data and compute.

## Datasets and partitions

Source images come from the Hugging Face mirrors [PlantVillage](https://huggingface.co/datasets/geraldmc/plantvillage-full) and [PlantDoc](https://huggingface.co/datasets/geraldmc/plantdoc-full). PlantDoc requests `v0.1.0`; PlantVillage is unpinned. [CSV manifests](data/splits/) retain original source row indices, with the canonical order **Healthy=0, Early Blight=1, Late Blight=2, Bacterial Spot=3, Septoria Leaf Spot=4**.

| Partition | Images | Role |
| --- | ---: | --- |
| Lab Train / Validation / Test | 5,365 / 1,341 / 1,692 | Training / checkpoint selection / fixed laboratory evaluation |
| Field Adapt Train / Field Dev | 370 / 92 | Supervised adaptation / target checkpoint selection |
| Quarantined PlantDoc training rows | 14 | Excluded for near-test overlap or conflicting labels |
| Original Field Test | 47 | Previously inspected historical evaluation, not fresh confirmation |
| External v1 | 62 | 16 self-collected Healthy and 46 online disease images |
| Healthy v2 primary analysis | 57 | 8 originals, 7 historical crops and 42 newly approved crops |
| Healthy paired extension | 23 parents / 49 crops | 21 recorded scene groups; exploratory matched-parent analysis |

Original test membership is preserved. Lab Validation uses leaf-group-aware splitting of original laboratory training images; unknown leaf IDs are singleton groups. The [verified PlantDoc protocol](data/splits/field_dev_protocol.json) groups pHash connected components at distance ≤4 across the 476/47 original partitions. Four training rows near Field Test and ten in conflicting-label groups are quarantined. Field Dev class supports are 11, 16, 20, 18 and 27; Field Test supports are 8, 9, 10, 9 and 11.

Accuracy, five-class macro-F1 and class-level metrics are reported separately. Field Dev participates in selection. Healthy-only recognition is recall, not five-class discrimination. Crops and their parents are correlated observations; scene IDs do not establish independent physical plants.

## Notebook organization

| Notebook | Purpose |
| --- | --- |
| [01](notebooks/01_data_exploration.ipynb) | Dataset composition, harmonized labels, examples and leaf-group coverage |
| [02](notebooks/02_train_cnn.ipynb) | ResNet18 laboratory baseline |
| [03](notebooks/03_train_vit.ipynb) | ViT-Small laboratory baseline |
| [04](notebooks/04_domain_gap.ipynb) | Baseline Lab Test and historical Field Test from saved predictions |
| [05](notebooks/05_augmentation_experiments.ipynb) | Photometric/geometric ResNet18 augmentation; saved training run is geometric |
| [06](notebooks/06_field_adaptation_verified.ipynb) | Verified ResNet18 field-only adaptation |
| [07](notebooks/07_vit_field_adaptation.ipynb) | Verified ViT-Small field-only adaptation |
| [08](notebooks/08_mixed_domain_finetuning.ipynb) | Historical mixed-domain runs and field-only controls |
| [09](notebooks/09_final_model_comparison.ipynb) | Four fixed models on Lab Test and Field Dev |
| [10](notebooks/10_error_analysis.ipynb) | Paired Field Dev errors, Grad-CAM and ViT attention |
| [11](notebooks/11_external_evaluation.ipynb) | Canonical external v1 evaluation with EXIF correction |
| [12](notebooks/12_external_error_analysis.ipynb) | External class, source and error trade-offs |
| [13](notebooks/13_healthy_field_evaluation.ipynb) | Healthy recognition and exploratory original–crop pairs |
| [14](notebooks/14_lesion_resolution_study.ipynb) | Exploratory center-crop and five-crop sensitivity |
| [15](notebooks/15_controlled_field_mixed_finetuning.ipynb) | Completed three-seed controlled ResNet18 comparison |

`notebooks/archive/` contains superseded protocols and the unchanged original Notebook 15 Colab execution. Archives are provenance records, not an alternative canonical sequence. Historical failures and decoding differences are identified explicitly.

## Key findings

The historical four-model results use Notebook 09 for Lab Test/Field Dev and Notebook 11 for external v1. Values below are accuracy / macro-F1.

| Fixed checkpoint | Lab Test (1,692) | Field Dev (92) | External v1 (62) |
| --- | ---: | ---: | ---: |
| ResNet18 Lab baseline | 0.9793 / 0.9744 | 0.3261 / 0.2355 | 0.3065 / 0.2406 |
| ViT-Small Lab baseline | 0.9947 / 0.9938 | 0.4022 / 0.2677 | 0.4355 / 0.4157 |
| ResNet18 Geometric + Mixed | 0.9297 / 0.9140 | 0.4674 / 0.4459 | 0.5323 / 0.4882 |
| ViT-Small Mixed | 0.9900 / 0.9886 | 0.4022 / 0.3872 | 0.5161 / 0.4808 |

Laboratory performance transfers poorly to PlantDoc. Historical mixing retains more Lab Validation performance than corresponding field-only runs, but does not uniformly improve Field Dev macro-F1. The ResNet18 baseline-to-final-Mixed contrast also includes geometric initialization.

Aggregate external gains hide class losses: ResNet18 Healthy recognition rises from 0/16 to 12/16, while Early Blight falls from 11/13 to 1/13. External v1 is source/class confounded and too small for broad architectural claims. EXIF orientation must precede RGB conversion; raw-RGB historical results are kept separate in the [reconciliation record](docs/experiment_reconciliation.md).

Notebook 13 recognizes 0/57, 15/57, 44/57 and 43/57 Healthy images for ResNet Lab, ViT Lab, ResNet Mixed and ViT Mixed. The paired extension uses 23 source photographs and 49 associated crops. Both Mixed models recognize 37/49 crops, but equal-parent crop recall differs: 0.7558 for ResNet and 0.8275 for ViT. Crop counts are not independent plant trials.

Notebook 14 finds model- and sample-specific sensitivity to framing. ViT Mixed five-crop external inference reaches 38/62 correct and macro-F1 0.5840. It averages five **224×224** views from a square 448 canvas, changing scale, coverage, aspect ratio and aggregation. This exploratory reuse of external outcomes does not isolate resolution or select a deployment policy.

### Controlled Field-only versus Mixed fine-tuning

Both conditions complete 2,960 Field presentations and 192 optimizer updates per seed; Mixed adds 2,960 Lab presentations. Shared Field Dev selection chooses epochs 8, 7 and 8. Seed 43's evaluated pair therefore has 2,590 Field presentations and 168 updates, despite completing the full budget.

| Condition | Field Test accuracy, mean ± sample SD | Macro-F1, mean ± sample SD |
| --- | ---: | ---: |
| Field-only | 0.4610 ± 0.0123 | 0.4300 ± 0.0075 |
| Mixed | 0.4468 ± 0.0213 | 0.4072 ± 0.0296 |

Mixed does not consistently outperform Field-only under this controlled ResNet18 recipe: Field-only leads macro-F1 in two seeds and Mixed slightly in one. This does not establish universal Mixed inferiority or statistical superiority. All seeds reuse the same historically inspected 47 test images; sample SD describes seed variation. Original predictions, selected checkpoint hashes and exposure records were checked against the imported Colab artifacts. [Portable exports and provenance](data/results/notebook15/) support the report without distributing weights; see the [Colab/report guide](docs/notebook15_colab_guide.md).

## Repository and reproducibility

```text
notebooks/          Canonical 01–15 notebooks and historical archive
src/                Manifest loader, evaluation, split preparation and analysis
scripts/            Lightweight repository/artifact audits and Colab packaging
tests/             Automated protocol, evidence and correctness checks
data/splits/       Versioned manifests, mappings, quarantine and split history
data/results/      Portable Notebook 15 predictions and provenance
outputs/            Selected tracked historical evidence; other runs ignored
models/             Local checkpoints, excluded from Git
docs/              Research summary, artifact access, reconciliation and final audit
```

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

These checks require no training, inference or dataset download. Notebook 04 recalculates baseline comparisons from committed predictions; Notebook 15 recalculates the controlled report from committed exports. Their normal execution writes derived figures, so use a temporary checkout to preserve existing outputs. `python -m src.statistical_analysis --output /tmp/plant-statistical-review` exports historical transcriptions and statistics into a **new** directory and refuses existing destinations.

Viewing saved outputs requires no GPU. Training Notebooks 02–03 and 05–08 require their documented Colab CUDA/Drive setup. Notebook 09 performs inference; Notebook 10 requires its original four Field Dev prediction exports and checkpoints, which are not distributed. Notebooks 11–14 require private external manifests, saved local runs and, for image processing/inference cells, images and weights. Notebook 13 includes inference on 22 additional parents; do not execute it merely to restyle figures. Notebook 14 uses cached predictions but its gallery also loads a checkpoint and source images. Full Notebook 15 artifact verification additionally needs the imported Drive run and asset bundle; `python scripts/audit_notebook15_results.py` performs that read-only verification without training or inference.

Evaluation v2 remains a self-contained local package. Its [builder](data/external_evaluation/tomato_field_eval_v2/build_dataset.py), [validator](data/external_evaluation/tomato_field_eval_v2/validate_dataset.py), [class mapping](data/external_evaluation/tomato_field_eval_v2/class_mapping.json) and [dataset README](data/external_evaluation/tomato_field_eval_v2/README.md) may be versioned. The full annotated manifest, image/parent trees, provenance workbooks and original `validation.json` stay local and ignored. With the authorized package restored, run `PYTHONDONTWRITEBYTECODE=1 python data/external_evaluation/tomato_field_eval_v2/validate_dataset.py data/external_evaluation/tomato_field_eval_v2`; it validates without rebuilding or changing the dataset.

[Workflow and artifact provenance](docs/workflow.md) documents exact filenames and access dependencies. Requirements are not a complete historical environment lock. Preserve existing manifests rather than regenerating row-index splits against refreshed datasets. `src/train.py`, `src/models.py`, `configs/config.yaml` and `app/` are explicitly labelled starter scaffolds; the recorded research runs use notebook implementations, not that training entry point or its ViT-B/16 factory.

## Limitations and distribution

The small, imbalanced and previously inspected evaluation samples, incomplete grouping, unpinned source metadata, annotation uncertainty and potential pretraining overlap constrain generalization. Historical recipes are single-seed; the three controlled seeds share one test set. Grad-CAM and attention are qualitative diagnostics, with no completed blinded lesion adjudication or causal localization claim. No deployment study was performed.

The branch is technically ready for manual fast-forward integration; this does not establish public redistribution rights. Private photographs, external annotation packages, checkpoints and environments are excluded from the proposed commit. No standalone raw/external photograph files are currently tracked. Tracked laboratory/PlantDoc galleries still contain third-party dataset imagery and retain the underlying rights obligations. An earlier upstream commit also contains the 62 external photographs, including self-collected images, in Git history. Current ignore rules do not erase those copies: repository-owner privacy/licensing review remains necessary before wider redistribution. Share source and saved evidence with the supervisor using the documented access restrictions; do not attach the entire local working directory.

## Research references

The study builds on Hughes and Salathé (2015), *An open access repository of images on plant health*; Singh et al. (2020), *PlantDoc: A dataset for visual plant disease detection*; He et al. (2016), *Deep residual learning for image recognition*; Dosovitskiy et al. (2021), *An image is worth 16x16 words*; Steiner et al. (2022), *How to train your ViT?*; and Selvaraju et al. (2017), *Grad-CAM*. Model and dataset identifiers above specify the implementations actually used.

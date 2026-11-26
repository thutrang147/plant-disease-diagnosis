# Notebook 15: Google Colab execution guide

Training has not run. This workflow prepares the controlled experiment; only an executed Colab notebook can supply results. All code, notebook text and exported evidence are in English.

## Ready-to-upload files

The current private export is about 390 MiB and contains 509 PlantDoc images (370 adaptation train, 92 Dev, 47 historical Test), 5,365 PlantVillage Lab Train images, both Lab baseline checkpoints, and `asset_manifest.json`. Only existing splits are used. No Lab validation/test, External v1/v2 or Healthy v2 images are required.

Upload these local items to Google Drive:

1. Folder `outputs/controlled_field_mixed/colab_assets_v1/`, including every nested file.

Place the folder inside `MyDrive/plant-disease-diagnosis/controlled_field_mixed_assets/`. The final tree must be:

```text
MyDrive/plant-disease-diagnosis/
  controlled_field_mixed_assets/
    colab_assets_v1/
      asset_manifest.json
      datasets/
        plantdoc/          # Arrow, dataset_info.json, state.json
        plantvillage/     # Arrow, dataset_info.json, state.json
      checkpoints/
        resnet18_lab_best.pt
        vit_small_lab_best.pt
  outputs/controlled_field_mixed/    # created by the notebook
```

Upload `notebooks/15_controlled_field_mixed_finetuning.ipynb` directly in Colab via **File → Upload notebook**. Keep the private assets on your own Drive. Code cell 4 clones/fetches the helper module and exact four CSV splits plus label map from `https://github.com/thutrang147/plant-disease-diagnosis.git`. No code ZIP is required. Data and baseline checkpoints are excluded from Git and must remain on Drive. A private repository requires Git authentication in Colab.

Before using Colab, commit and push `src/controlled_field_mixed.py`, this notebook and guide, and the canonical split files to the intended GitHub branch. In code cell 3, set `GIT_REF` to that branch (default `main`) or a commit SHA. Code cell 4 reports missing files if the chosen revision does not contain the helper. It fetches the requested ref and checks it out without overwriting local changes. Save the printed commit SHA and use it as `GIT_REF` when resuming. Branch names fetch their latest revision whenever cell 4 runs.

## Execute in Colab

1. Select **Runtime → Change runtime type → GPU**. A T4 is a reasonable starting runtime; resource feasibility is determined by the CUDA smoke test.
2. Execute **code cells 1–4**, identified by comments at their top: CUDA gate, dependency installation, Drive mount/paths, and GitHub clone/fetch. Check `GIT_REF` before running cell 4. Grant Drive access when Colab asks. Keep Colab's existing matched CUDA PyTorch/torchvision installation.
3. In **code cell 5**, choose `ARCHITECTURE='resnet18'`, a unique `EXPERIMENT_ID`, and the seed budget **before opening the run**. The default declares `(42, 43, 44)`. For a shorter pilot choose `(42,)` and a matching ID such as `controlled_resnet18_s42_pilot_v1`. Execute cell 5 to verify every asset, baseline hash and canonical split.
4. Execute **code cell 6** to assert the matched schedule. Execute **code cell 7** for the real CUDA smoke test: baseline/data loading, forward, backward, AdamW update, persistent checkpoint write/read. The smoke model is discarded; baseline files remain unchanged.
5. Execute **code cell 8** to train the seed-42 pair. Execute **code cell 9** to finish all remaining declared seeds and validate actual exposure, update counts and shared selection. For a declared one-seed pilot cell 9 only validates the completed pair.
6. Execute **code cell 10** for Field Dev learning curves. Execute **code cell 11** only when all declared seeds have finished: checkpoint hashes are sealed before historical Field Test inference. Execute **code cells 12–15** for per-class recall, confusion matrices, Early Blight errors and saved findings.
7. To extend to ViT-Small, first complete ResNet and obtain `training_validation.json` from cell 9. In a fresh GPU session, run setup cells, then set `ARCHITECTURE='vit_small'`, a new ID, and `RESNET_RUN_FOR_VIT` to the completed ResNet run in cell 5. Repeat cells 5–15. Run the architectures independently.

Do not use **Run all** before checking the paths and experiment budget. Each numbered code cell also has surrounding research explanations; notebook cell positions include Markdown cells and differ from the code-cell numbers above.

## Controls and historical differences

Notebook 06 used all 370 Field images with ResNet Field batches of 32; Notebook 07 used ViT Field batches of 16. Notebook 08 sampled 185 Field and 185 Lab images per epoch with half-batches, independently selected/stopped each model, and optimized mean cross-entropy over the concatenated domains. That scaled the Field component by one half. Earlier phase transitions rewound to the selected checkpoint. The final historical geometric ResNet Field-only/Mixed comparison also used different initializations. Historical results stay separate from this experiment.

Notebook 15 reloads the same recorded Lab baseline for both conditions. Each fixed epoch visits every Field image exactly once, in the same order, with the same augmented tensor. Field-only uses mean `L_field`; Mixed accumulates gradients from `L_field + 1.0 * L_lab` before one AdamW step. The weight 1.0 is fixed for equal domain-mean contributions, without examining held-out outcomes; it is not claimed optimal. It preserves equal domain emphasis while keeping the Field gradient coefficient the same as Field-only.

Both use the existing mild crop/flip/jitter transforms and model-specific normalization; ResNet BatchNorm running statistics stay frozen. Both train for two head-only epochs and six final-block epochs, reset AdamW once at the transition, and use fixed phase learning rates: classifier 1e-4, last block 1e-5, weight decay 1e-4. No early stopping or phase rewind is used.

| Per seed and condition | ResNet18 | ViT-Small |
| --- | ---: | ---: |
| Field batch size | 16 | 8 |
| Updates per epoch | 24 | 47 |
| Full-budget optimizer updates | 192 | 376 |
| Full-budget Field presentations | 2,960 | 2,960 |
| Additional Mixed Lab presentations | 2,960 | 2,960 |

Checkpoint selection maximizes the pair's mean five-class Field Dev macro-F1 over epochs 0–8, choosing the earliest tie. Both selected checkpoints therefore have the same epoch, Field exposure and update count. Epoch zero is eligible. This couples selection across conditions and must accompany interpretation. Mixed consumes extra information and computation; training seconds per condition and image forward counts are recorded.

The deterministic schedule assertions run before training. Actual per-row counters, batch Field IDs, shared augmented-tensor hashes and update counts are checked after training. `field_exposure.csv`, `batch_audit.csv`, `complete.json` and `training_validation.json` provide inspectable evidence. Summary metrics are computed from saved image-level predictions.

Field Test is not used for selection or tuning. Row/group separation, exact RGB and pHash distance ≤4 screen the preserved assets; the current screen found no matches between either training source and Field Dev/Test. These checks do not prove physical-plant independence or absence of pretraining overlap. The 47-image Field Test was already inspected historically and is not a fresh prospective test. Do not choose settings or seeds from its results.

## Disconnects and persistent outputs

After disconnect, keep the same Git commit, assets, ID, seed list and configuration. Set `GIT_REF` to the commit SHA printed in the original session. Rerun code cells 1–7, then cells 8–9. Completed seeds are verified and skipped, including after evaluation sealing. Each epoch commits a single atomic `resume.pt` containing both model/optimizer states, counts, history and shared best states. An interrupted epoch is replayed from the last committed epoch. Logical committed exposure excludes abandoned computation; recorded time also excludes abandoned epochs. Resume requires matching library versions and code/assets hashes. Restore the recorded versions if Colab's installed environment changes. Device/CUDA changes can affect bitwise reproducibility; smoke-test metadata captures the current environment.

An experiment cannot resume with a different configuration. Use a new ID for any new recipe; do not retune from held-out results. Missing or incompatible assets stop execution. Existing canonical baseline checkpoints are never replaced. Allow Drive space for epoch resume files and two selected checkpoints per seed (several GiB for both architectures and three seeds), in addition to the 390 MiB assets.

Outputs are written directly under `MyDrive/plant-disease-diagnosis/outputs/controlled_field_mixed/<experiment ID>/`: configuration, asset/code/initialization hashes, environment, smoke evidence, paired resume states, selected checkpoints, histories, exposure/audit CSVs, training validation, evaluation seal, image-level predictions, summary/per-class metrics, confusion CSVs, figures and findings.

## Rebuild assets locally (no training)

The existing asset directory can be used directly. To regenerate from the original local HF Arrow caches, use a **new** output directory; the exporter refuses overwrite:

```bash
.venv/bin/python scripts/prepare_controlled_colab_assets.py \
  --output outputs/controlled_field_mixed/colab_assets_v2
```

Change `ASSET_ROOT` in Colab accordingly. If caches are elsewhere, supply `--cache-root /path/to/huggingface/datasets`. Missing or ambiguous caches and noncanonical checkpoint hashes stop the export. `--resnet-only` omits the ViT checkpoint when exporting assets for ResNet alone. No local command above trains a model.

The exporter's optional bundle flags remain available for legacy offline workflows; Notebook 15 now loads code from GitHub and does not require them.

## Import completed results

Download the entire experiment folder and executed notebook from Colab/Drive. From the repository root, copy the folder to a new ignored output location, preserving all files:

```bash
.venv/bin/python - /path/to/downloaded/controlled_resnet18_s42_43_44_v1 <<'PY'
import shutil
import sys
from pathlib import Path
source = Path(sys.argv[1])
target = Path('outputs/controlled_field_mixed') / source.name
if target.exists():
    raise FileExistsError(f'Refusing overwrite: {target}')
shutil.copytree(source, target)
print(target)
PY
```

The workflow does not commit or push. Conclusions remain pending until Colab executes training and evaluation.

# Notebook 15 saved-result evidence

The original executed Colab notebook is preserved without modification at
`notebooks/archive/15_controlled_field_mixed_colab_executed.ipynb`.
`provenance.json` records its SHA-256, training revision, displayed runtime,
baseline identity and source cell indices (zero-based).

`confusion_counts_from_notebook.csv` transcribes all 150 integer annotations in
the original saved six-panel confusion-matrix figure. These are **derived
notebook counts**, not original image-level prediction CSVs. The read-only
`src/controlled_field_report.py` checks their hashes, class supports, six
aggregate metrics, 30 per-class rows, displayed means/SDs and paired differences
against the original saved outputs. Metrics are calculated before rounding;
SD across seeds uses `ddof=1`. Row = true class, column = predicted class.

The canonical class order is Healthy, Early Blight, Late Blight, Bacterial Spot,
Septoria Leaf Spot. Each matrix describes the same 47-image historical Field
Test, with class supports 8, 9, 10, 9 and 11. Repeated seeds do not add independent
evaluation images.

The executed version differed from the prior Git version only in the Drive/
Git configuration cell (`GIT_REF` selected the experiment branch) and added
execution outputs. It was archived before any report changes. Its complete
configuration cells, original training logs, selection summary, figures and
original results remain intact. The training module is unchanged from source
commit `0066ddb3a93996252a8c80591a4ac49ac6324034`.

The full Drive run has now been imported at
`outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1/`.
`prediction_exports/` contains byte-preserved copies of the six original Field
Test CSVs, with source paths and SHA-256 in `provenance.json`. These CSVs now
supply the report metrics. All 282 prediction rows retain their original image
identities and probability records; they represent six evaluations of the same
47 images, not 282 independent samples. Original configuration, environment,
training validation, evaluation seal, summary and class exports are also copied
without modification, with hashes in the provenance record.

`imported_artifact_audit.json` records a successful read-only audit of the full
local run: six test and six Dev prediction files, exact common-epoch selection
from full-precision histories, matching per-batch Field identities and tensor
hashes, eight exposures per Field row, 192 updates per condition, additional Lab
exposure, identical recorded initialization, and six selected checkpoint hashes
and metadata matching the completion records and evaluation seal. All 53
imported files are hashed before and after; none is changed. Resume files are
preserved and hashed, without replaying their optimizer states. The script is
`scripts/audit_notebook15_results.py`; it needs the full local run and original
assets but never trains, performs inference, or writes into the imported run.

The main notebook runs only lightweight analysis and writes figures to a separate
ignored `outputs/controlled_field_mixed/notebook15_report/` directory. Set
`PLANT_CONTROLLED_RUN` to optionally check another original run location. The
portable prediction copies support report execution without model weights.
See `docs/notebook15_colab_guide.md` for report execution and audit instructions.

#!/usr/bin/env bash
# Re-run the round-2 analyses from a fresh clone and compare them with the committed results.
#
#   bash scripts/reproduce_round2.sh          # inference analyses (~1.5 h on 4 CPU cores)
#   bash scripts/reproduce_round2.sh --all    # also the DINOv2 probe and the exploratory training runs (~4 h more)
#
# Requires: pip install -r requirements.txt (or requirements-lock.txt on Linux x86_64), internet access to
# Google Drive and Hugging Face (about 4 GB of datasets and checkpoints on first use). Outputs go to
# reproduction/ (git-ignored); committed results are not touched. Inference re-runs are expected to match
# within 1e-4; training re-runs (--all) can differ slightly across hardware, so their tolerance is looser.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1
OUT=reproduction
mkdir -p "$OUT"
status=0

python scripts/download_checkpoints.py
python -m unittest discover -s tests

run() {  # name, atol, script, extra args...
  local name=$1 atol=$2 script=$3; shift 3
  if [[ -e "$OUT/$name" ]]; then echo "skip $name (exists)"; else python "scripts/$script" --output "$OUT/$name" "$@"; fi
  python scripts/compare_reproduction.py "data/results/$name" "$OUT/$name" --atol "$atol" || status=1
}

run notebook15_lab_retention 1e-4 evaluate_notebook15_lab_retention.py \
    --run outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1 --baseline models/resnet18_lab_best.pt
run external_v3_plantseg 1e-4 evaluate_external_v3_plantseg.py
run external_v3_error_analysis 1e-4 analyze_external_v3_errors.py
run weight_interpolation 1e-4 evaluate_weight_interpolation.py
run lesion_reliance 1e-4 evaluate_lesion_reliance.py
run occlusion_control 1e-4 evaluate_occlusion_control.py
run occlusion_subset 1e-4 analyze_occlusion_subset.py
run external_v3_label_merge 1e-4 analyze_external_v3_label_merge.py
run lab_split_duplicates 1e-4 audit_lab_split_duplicates.py
if [[ $ALL == 1 ]]; then
  run frozen_foundation_probe 1e-3 evaluate_frozen_foundation_probe.py --backbones dinov2_vits14 dinov2_vitb14
  run lpft_architecture 2e-2 explore_lpft_architecture.py
  run training_budget 2e-2 explore_training_budget.py
fi

if [[ $status == 0 ]]; then echo "All re-runs match the committed results."; else echo "Some re-runs differ; see above."; fi
exit $status

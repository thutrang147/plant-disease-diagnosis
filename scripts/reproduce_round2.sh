#!/usr/bin/env bash
# Re-run the round-2 evaluations from a fresh clone and compare them with the committed results.
#
#   bash scripts/reproduce_round2.sh          # inference analyses (~1 h on 4 CPU cores)
#   bash scripts/reproduce_round2.sh --all    # also the DINOv2 probe and the exploratory LP-FT runs (~2.5 h more)
#
# Requires: pip install -r requirements-lock.txt (or requirements.txt), internet access to
# Google Drive and Hugging Face. Outputs go to reproduction/ (git-ignored); committed results are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1
OUT=reproduction
mkdir -p "$OUT"
status=0

python scripts/download_checkpoints.py
python -m pytest -q

run() {  # name, script, extra args...
  local name=$1 script=$2; shift 2
  if [[ -e "$OUT/$name" ]]; then echo "skip $name (exists)"; else python "scripts/$script" --output "$OUT/$name" "$@"; fi
  python scripts/compare_reproduction.py "data/results/$name" "$OUT/$name" || status=1
}

run notebook15_lab_retention evaluate_notebook15_lab_retention.py \
    --run outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1 --baseline models/resnet18_lab_best.pt
run external_v3_plantseg evaluate_external_v3_plantseg.py
run weight_interpolation evaluate_weight_interpolation.py
run lesion_reliance evaluate_lesion_reliance.py
if [[ $ALL == 1 ]]; then
  run frozen_foundation_probe evaluate_frozen_foundation_probe.py --backbones dinov2_vits14 dinov2_vitb14
  run lpft_architecture explore_lpft_architecture.py
fi

if [[ $status == 0 ]]; then echo "All re-runs match the committed results."; else echo "Some re-runs differ; see above."; fi
exit $status

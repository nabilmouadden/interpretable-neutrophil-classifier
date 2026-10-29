#!/usr/bin/env bash
# Train, calibrate and evaluate five seeds, with and without stain augmentation.
#
# Usage:
#   bash scripts/train_all_seeds.sh MANIFEST DATA_ROOT OUT_DIR [NEUNN_MANIFEST NEUNN_ROOT]
#
#   MANIFEST        multi-label manifest (see README, "Data")
#   DATA_ROOT       root of the image paths listed in MANIFEST
#   OUT_DIR         where the run directories and summaries are written
#   NEUNN_MANIFEST  optional external manifest (image_path,label), e.g. built with
#                   `neutroclf make-manifest --folders <NeuNN root> --external --out ...`
#   NEUNN_ROOT      root of the external image paths
#
# Five seeds (0 7 13 42 1337) x {standard, stain augmentation}; each run draws its own
# cell-level stratified 80/10/10 split. Extra arguments for `neutroclf train` can be
# passed through the TRAIN_ARGS environment variable, e.g.
#   TRAIN_ARGS="--set data.num_workers=8" bash scripts/train_all_seeds.sh ...
set -euo pipefail

MANIFEST=${1:?manifest}
DATA_ROOT=${2:?data root}
OUT=${3:?output dir}
NEUNN_MANIFEST=${4:-}
NEUNN_ROOT=${5:-}
SEEDS=(0 7 13 42 1337)
CONFIG=${CONFIG:-configs/default.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}

for variant in standard stain; do
  runs=()
  for seed in "${SEEDS[@]}"; do
    run="${OUT}/${variant}/seed${seed}"
    if [[ ! -f "${run}/checkpoint.pt" ]]; then
      # shellcheck disable=SC2086
      neutroclf train --config "${CONFIG}" --manifest "${MANIFEST}" --data-root "${DATA_ROOT}" \
        --out "${run}" --seed "${seed}" --set "data.augmentation=${variant}" ${TRAIN_ARGS}
    fi
    neutroclf calibrate --run "${run}"
    neutroclf evaluate --run "${run}" --figures
    if [[ -n "${NEUNN_MANIFEST}" ]]; then
      neutroclf external-eval --run "${run}" --manifest "${NEUNN_MANIFEST}" \
        --data-root "${NEUNN_ROOT}" --label-map configs/external_neunn.yaml
    fi
    runs+=("${run}")
  done
  neutroclf summarize "${runs[@]}" --out "${OUT}/${variant}/summary"
done
echo "Tables: ${OUT}/standard/summary/tables.md (and stain/summary for the augmentation variant)"

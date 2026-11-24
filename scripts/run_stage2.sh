#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VISION_MODEL:?Set VISION_MODEL to a local DINOv2-L/14 model directory}"
: "${LANGUAGE_MODEL:?Set LANGUAGE_MODEL to a local Vicuna-7B-v1.5 model directory}"
: "${CC3M_MANIFEST:?Set CC3M_MANIFEST to a local CC3M image-caption manifest}"
: "${STAGE1_CKPT:?Set STAGE1_CKPT to a trained SPARE--BEAM checkpoint}"
: "${STAGE2_OUTPUT:?Set STAGE2_OUTPUT to an output directory}"
cd "$ROOT"
python -m savera.stage2.train \
  --spare_beam_ckpt "$STAGE1_CKPT" \
  --vision_model "$VISION_MODEL" --text_model "$LANGUAGE_MODEL" \
  --manifest "$CC3M_MANIFEST" --image_root "${CC3M_IMAGE_ROOT:-}" \
  --output_dir "$STAGE2_OUTPUT" --batch_size 2 --grad_accum 4 "$@"

#!/usr/bin/env bash
# Source this file from any directory. Override paths before sourcing it.
export VLFM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLFM_RUNTIME="${VLFM_RUNTIME:-$VLFM_ROOT/runtime}"
export VLFM_PYTHON="${VLFM_PYTHON:-$VLFM_RUNTIME/envs/vlfm/bin/python}"
export LANGMAP_PYTHON="${LANGMAP_PYTHON:-$VLFM_RUNTIME/envs/sim/bin/python}"
export SAM2_PYTHON="${SAM2_PYTHON:-$VLFM_RUNTIME/envs/sam2/bin/python}"
export VLFM_MODELS="${VLFM_MODELS:-$VLFM_RUNTIME/models}"
export VLFM_DINO_SOURCE="${VLFM_DINO_SOURCE:-$VLFM_ROOT/external/GroundingDINO}"
export VLFM_YOLO_SOURCE="${VLFM_YOLO_SOURCE:-$VLFM_ROOT/external/yolov7}"
export VLFM_SAM2_SOURCE="${VLFM_SAM2_SOURCE:-$VLFM_ROOT/external/sam2}"
export EMBODIEDSCORE_DATA_ROOT="${EMBODIEDSCORE_DATA_ROOT:-$VLFM_RUNTIME/datasets}"
export EMBODIEDSCORE_SCENE_ROOT="${EMBODIEDSCORE_SCENE_ROOT:-$VLFM_RUNTIME/scenes}"
export PYTHONPATH="$VLFM_ROOT:$VLFM_ROOT/third_party/EmbodiedScore-envs${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$VLFM_RUNTIME/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$VLFM_RUNTIME/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$VLFM_RUNTIME/cache}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet TOKENIZERS_PARALLELISM=false
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

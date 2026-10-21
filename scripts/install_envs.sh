#!/usr/bin/env bash
# Linux x86_64; requires conda and a CUDA toolkit compatible with the policy build.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$VLFM_ROOT"
conda_command="${CONDA_EXE:-conda}"
"$conda_command" --version
python3 -m scripts.fetch_dependencies
for spec in 'vlfm:3.9' 'sim:3.11' 'sam2:3.11'; do
  environment_name="${spec%%:*}"
  python_version="${spec##*:}"
  environment_path="$VLFM_RUNTIME/envs/$environment_name"
  if [[ ! -x "$environment_path/bin/python" ]]; then
    "$conda_command" create -y --prefix "$environment_path" "python=$python_version" pip
  fi
done

# Preserve legacy dependency versions instead of letting Habitat/LAVIS change Torch.
"$VLFM_PYTHON" -m pip install 'pip<24.1' 'setuptools<81' wheel
"$VLFM_PYTHON" -m pip install --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements/policy.txt
"$VLFM_PYTHON" -m pip install --no-deps \
  ./external/frontier_exploration ./external/depth_camera_filtering

# This legacy binary is needed only for imports in the Python 3.9 policy process.
# The actual simulator is the separately pinned 0.3.3 wheel below.
mkdir -p "$VLFM_RUNTIME/downloads"
legacy_archive="$VLFM_RUNTIME/downloads/habitat-sim-0.2.4-py3.9_headless_linux_f179b584bcd713c5a2a998132211e2cae881d6d1.tar.bz2"
python3 - "$legacy_archive" <<'PY'
import hashlib, sys, urllib.request
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    temp = p.with_name(p.name + '.part')
    urllib.request.urlretrieve('https://conda.anaconda.org/aihabitat/linux-64/' + p.name, temp)
    temp.replace(p)
actual = hashlib.sha256(p.read_bytes()).hexdigest()
if actual != 'b0609226581c04c48b6bef3df1252d2ff970c2985984bbf417e2048b52a0a3c3':
    raise ValueError('Legacy Habitat archive hash mismatch')
PY
"$conda_command" install -y --offline --no-deps --prefix "$VLFM_RUNTIME/envs/vlfm" "$legacy_archive"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$VLFM_PYTHON" -m pip install --no-deps --no-build-isolation -e ./external/GroundingDINO
"$VLFM_PYTHON" -m pip install --no-deps -e .

"$LANGMAP_PYTHON" -m pip install -r requirements/simulator.txt
"$LANGMAP_PYTHON" -m pip install --no-deps -e ./third_party/EmbodiedScore-envs -e .

"$SAM2_PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$SAM2_PYTHON" -m pip install -r requirements/sam2.txt
SAM2_BUILD_CUDA=0 "$SAM2_PYTHON" -m pip install --no-build-isolation --no-deps -e ./external/sam2
"$SAM2_PYTHON" -m pip install --no-deps -e .
echo 'Environments installed. Next: download checkpoints and configure dataset/scene paths.'

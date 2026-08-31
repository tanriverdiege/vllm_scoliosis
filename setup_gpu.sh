#!/usr/bin/env bash
# GPU box (Linux + CUDA) environment for scoliosis VLM inference.
#
#   bash setup_gpu.sh          # transformers lane only
#   bash setup_gpu.sh --vllm   # also install the vLLM serving engine
#
# THE ONE RULE THAT MATTERS: never pre-install torch, and never `pip install -U
# torch` afterwards. vLLM pins an exact torch build and silently breaks its
# compiled kernels if torch is swapped underneath it. Let vLLM pull its own
# torch, then layer everything else on top.
set -euo pipefail

ENV_NAME="${ENV_NAME:-scoliosis-vlm}"
WITH_VLLM=0
[[ "${1:-}" == "--vllm" ]] && WITH_VLLM=1

echo "==> creating conda env '$ENV_NAME' (python 3.11)"
conda create -n "$ENV_NAME" python=3.11 -y

run() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }

run python -m pip install --upgrade pip

if [[ $WITH_VLLM -eq 1 ]]; then
  # vllm FIRST so it owns the torch pin. Its wheels are manylinux cp38-abi3,
  # so this step only works on Linux -- there is no macOS vLLM wheel.
  echo "==> installing vLLM (brings its own pinned torch + CUDA runtime)"
  run python -m pip install "vllm==0.28.0"
else
  echo "==> installing torch (CUDA build)"
  run python -m pip install torch torchvision
fi

echo "==> installing transformers lane + project deps"
run python -m pip install \
  "transformers==4.57.6" \
  accelerate "huggingface_hub[hf_transfer]" safetensors sentencepiece protobuf \
  pyyaml \
  qwen-vl-utils timm einops av \
  pillow "numpy<2.3" scipy opencv-python-headless pydicom matplotlib \
  datasets pandas tqdm

# flash-attn LAST: it compiles against whatever torch is already present, so
# installing it earlier would build against a torch that vLLM then replaces.
# Skip it entirely if the GPU is older than Ampere (sm_80).
echo "==> optional: flash-attn (skip on pre-Ampere GPUs)"
run python -m pip install flash-attn --no-build-isolation || \
  echo "    flash-attn failed; continuing without it (models fall back to sdpa)"

echo
echo "==> verifying"
run python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__)
print("CUDA avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability  ", f"sm_{cap[0]}{cap[1]}")
print("transformers", transformers.__version__)
try:
    import vllm; print("vllm        ", vllm.__version__)
except ImportError:
    print("vllm         (not installed)")
PY

echo
echo "done. conda activate $ENV_NAME"

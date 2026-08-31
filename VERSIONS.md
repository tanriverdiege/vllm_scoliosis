# Environment version policy

## Why Python 3.11

Python is the one thing we pin hard, so that *everything else* stays swappable.

| Constraint | Range | 3.11 ok? |
|---|---|---|
| `vllm` 0.28.0 | `>=3.10,<3.15` | yes |
| `transformers` (4.3x → 5.16) | `>=3.10` | yes |
| `torch` (2.1 → 2.13) | `>=3.10` | yes |

3.12 and 3.13 also satisfy those, so the tiebreaker is source builds:

- **3.12 removed `distutils`** (PEP 632). The packages VLM work actually needs
  from source — `flash-attn`, `mmcv`, InternVL's deformable-attention CUDA
  extension — still reach for it. On 3.11 they build; on 3.12+ they need patching.
- Prebuilt `cp311` wheels exist across the entire transformers 4.3x → 5.x range
  and every torch from 2.1 to 2.13. `cp313` coverage thins out fast on older pins.
- Research VLM repos loaded via `trust_remote_code` are overwhelmingly written
  against 3.10/3.11.

So: Python 3.11 is the fixed constant; transformers is the variable.

## The transformers 4 → 5 split

This is the real version hazard, not Python.

- current 5.x: **5.16.1**
- last 4.x: **4.57.6**  ← what both envs install

We default to the 4.x lane because most third-party VLM code (LLaVA, InternVL,
older Qwen-VL forks, most medical-imaging VLM papers) targets 4.x, and 4.57.6
already supports Qwen2.5-VL, InternVL3, and MedGemma.

Switching lanes is a one-liner in either env — no rebuild, because 3.11 has
wheels for both:

```bash
conda activate scoliosis-vlm
pip install "transformers==5.16.1"     # newest models / newest API
pip install "transformers==4.57.6"     # back to the compatible lane
```

Keep a lane per env rather than fighting one env into serving both. If you need
4.x and 5.x simultaneously, clone: `conda create -n scoliosis-vlm-t5 --clone scoliosis-vlm`.

## Platform limits on this Mac

Two hard floors, both from macOS 13.0 — neither is fixable in the env:

1. **No MPS.** torch 2.11 requires macOS 14.0+ for the Metal backend. On 13.0
   `torch.backends.mps.is_available()` is `False` and moving a tensor to `mps`
   raises. This env is CPU-only. Upgrading to macOS 14 would enable MPS.
2. **No vLLM.** vllm 0.28.0 publishes only `manylinux` wheels
   (`cp38-abi3`, x86_64 + aarch64). There is no macOS wheel at any version, so
   vLLM exists solely on the GPU box.

Consequence: this Mac is for writing code, DICOM/data prep, and Cobb-angle
geometry. Anything that runs a model goes to the GPU box via `setup_gpu.sh`.

## Gotchas already hit and fixed

- **`av` source build fails.** Its arm64 wheels are tagged `macosx_14_0`, which
  pip rejects on macOS 13, so it falls back to compiling against ffmpeg headers
  that aren't there. `av==15.1.0` is the last release tagged `macosx_13_0_arm64`
  and installs as a real wheel. Pinned in `environment.yml`.
- **Do not `conda install` compiled libs into this env.** Pulling `av`/`ffmpeg`
  from conda-forge brings `llvm-openmp` and `libopenblas` alongside torch's own
  bundled `libomp`. Two OpenMP runtimes in one process → `OMP: Error #15` and an
  abort. Keep the env pip-only; verify with:
  ```bash
  find $CONDA_PREFIX -name "libomp*.dylib" -o -name "libiomp*.dylib"   # expect exactly 1
  ```
  `KMP_DUPLICATE_LIB_OK=TRUE` suppresses the abort but risks wrong numerics —
  don't use it.
- **Benign warning:** `av` and `cv2` each vendor `libavdevice`, so an
  `objc[...] Class AVFFrameReceiver is implemented in both...` line prints on
  import. Harmless; ignore it.

## Verified working (macOS 13.0, arm64)

```
python 3.11.16 | torch 2.11.0 (CPU) | transformers 4.57.6 | numpy 2.2.6
av 15.1.0 | opencv 5.0.0 | pydicom 3.0.2 | timm 1.0.29 | datasets 5.0.1
```

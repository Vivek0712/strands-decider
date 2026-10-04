#!/usr/bin/env bash
# Install the fused Gated DeltaNet kernels for the installed torch and CUDA:
#   - flash-linear-attention (fla): Triton kernels for the gated delta rule, pure Python;
#   - causal-conv1d: the CUDA kernel for the short depthwise convolution in front of it.
#
# Without them transformers runs the reference PyTorch path for every Gated DeltaNet layer
# (18 of 24 in Qwen3.5-2B, 24 of 32 in 4B) and logs that it is "much slower"; every H200
# run up to v21 trained that way. The fused path changes the numerics slightly, so use it
# for every arm of a comparison or for none (research/next/STAGE_A.md).
#
#   training/install_fast_kernels.sh               # the active environment's python
#   PY=~/venvs/hobson/bin/python training/install_fast_kernels.sh
#
# causal-conv1d's own setup first looks for a prebuilt wheel matching this torch, CUDA,
# C++ ABI and Python on its GitHub release, and compiles only when there is none (that
# needs nvcc of the same CUDA major as torch, and takes ~10 min). Versions can be pinned:
# CAUSAL_CONV1D_VERSION=1.5.2 FLA_VERSION=0.5.0. The script ends by checking that
# transformers sees both, and prints what it installed; measure the effect with
# training/bench_steps.py, with and without (a fresh environment), before budgeting.
set -euo pipefail

PY=${PY:-python}
CAUSAL_CONV1D_VERSION=${CAUSAL_CONV1D_VERSION:-}
FLA_VERSION=${FLA_VERSION:-}

read -r TORCH CUDA ABI PYV <<<"$("$PY" - <<'EOF'
import sys
import torch
print(torch.__version__, torch.version.cuda or "none", int(torch._C._GLIBCXX_USE_CXX11_ABI),
      f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)"
echo "torch $TORCH, CUDA $CUDA, C++11 ABI $ABI, Python $PYV"
if [ "$CUDA" = none ]; then
  echo "this torch has no CUDA build: the fused kernels are CUDA-only, nothing to install" >&2
  exit 1
fi
if command -v nvcc >/dev/null 2>&1; then
  NVCC=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  echo "nvcc $NVCC (used only if no prebuilt causal-conv1d wheel matches)"
  if [ "${NVCC%%.*}" != "${CUDA%%.*}" ]; then
    echo "warning: nvcc $NVCC and torch's CUDA $CUDA differ in major version; a source build would fail" >&2
  fi
fi

"$PY" -m pip install --upgrade pip packaging ninja wheel setuptools
# fla first: it brings nothing torch-specific, and Triton comes with torch (fla wants
# Triton >= 3.3, i.e. torch >= 2.7)
"$PY" -m pip install "flash-linear-attention${FLA_VERSION:+==$FLA_VERSION}"
# --no-build-isolation: the build must see the installed torch, or it would fetch another
"$PY" -m pip install --no-build-isolation "causal-conv1d${CAUSAL_CONV1D_VERSION:+==$CAUSAL_CONV1D_VERSION}"

"$PY" - <<'EOF'
import importlib.metadata as md

import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

import causal_conv1d  # noqa: F401  (fails here, not mid-run, if the build is broken)
import fla  # noqa: F401

ok = {"causal_conv1d": is_causal_conv1d_available(), "flash_linear_attention": is_flash_linear_attention_available()}
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "causal-conv1d": md.version("causal-conv1d"), "flash-linear-attention": md.version("flash-linear-attention"),
       "seen_by_transformers": ok})
if not all(ok.values()):
    raise SystemExit("transformers does not see both kernels (no GPU visible?); the reference path would run")
EOF
echo "fused kernels installed; now measure: python training/bench_steps.py --config CONFIG --steps 100"

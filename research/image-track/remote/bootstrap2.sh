#!/bin/bash
# Second host (2x H100): setup, fast kernels, rebuild the data deterministically, dedupe.
set -uo pipefail
exec > >(tee -a /root/bootstrap2.log) 2>&1
bash /root/sd/research/image-track/remote/setup.sh
pip install -q flash-linear-attention 2>&1 | tail -1
timeout 900 pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -1 || echo "causal-conv1d not installed"
pip list 2>/dev/null | grep -iE "^(flash-linear|fla-core|causal)"
bash /root/sd/research/image-track/remote/fetch.sh
W=28 bash /root/sd/research/image-track/remote/build.sh
cd /root/sd/research/image-track && python dedupe.py --data-root /root/data --files vqa.jsonl count.jsonl m2w.jsonl charts.jsonl docs.jsonl tabfact.jsonl --ijb-dir /root/ijb --workers 28
cd /root/data && sha256sum *.jsonl
echo BOOTSTRAP2_DONE

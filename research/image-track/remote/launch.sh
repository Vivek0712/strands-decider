#!/bin/bash
# After bootstrap2: variants A and B (one per GPU if two), v19 baseline in this env.
set -uo pipefail
cd /root/sd/research/image-track
N=$(nvidia-smi -L | wc -l)
G1=$(( N > 1 ? 1 : 0 ))
nohup bash remote/run_variant.sh 0 configs/a-full.yaml a-full-s0 --long-side 0 --max-pixels 400000 > /root/a.out 2>&1 < /dev/null &
nohup bash remote/run_variant.sh $G1 configs/b-noabl.yaml b-noabl-s0 --long-side 0 --max-pixels 400000 > /root/b.out 2>&1 < /dev/null &
nohup sh -c "bash remote/run_baseline.sh $G1 v19-px400k-host2; bash remote/run_baseline.sh $G1 v19-px800k-host2 --long-side 0 --max-pixels 800000" > /root/base.out 2>&1 < /dev/null &
echo launched on $N GPUs

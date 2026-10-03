#!/bin/bash
# v19 in this host's environment (fla kernels if installed), same protocol as run_variant.sh.
set -uo pipefail
GPU=$1; RUN=${2:-v19-px400k-host2}; shift 2; EVAL_ARGS="${@:---long-side 0 --max-pixels 400000}"
export CUDA_VISIBLE_DEVICES=$GPU
R=/root/runs/$RUN; mkdir -p $R; cd /root/sd
echo "commit $(git rev-parse HEAD) gpu $GPU args $EVAL_ARGS start $(date -u +%FT%TZ)" > $R/meta.txt
python evaluation/vision/run.py --out $R/eval --systems strands --device cuda --nb-groups 300 --pope 600 --ijb-jsonl /root/ijb/ijb_preview.jsonl $EVAL_ARGS > $R/eval.log 2>&1
python evaluation/vision/run.py --out $R/cal --systems strands --device cuda --nb-start 300 --nb-groups 300 --pope 0 --no-blind $EVAL_ARGS > $R/cal.log 2>&1
python research/image-track/temps.py fit --rows $R/cal/strands-v19.jsonl --out $R/image_temps.json > $R/temps.log 2>&1
python research/image-track/temps.py apply --run $R/eval --temps $R/image_temps.json --out $R/eval-T >> $R/temps.log 2>&1
python research/image-track/text_check.py --out $R/text_check.json > $R/text.log 2>&1
echo "done $(date -u +%FT%TZ)" >> $R/meta.txt

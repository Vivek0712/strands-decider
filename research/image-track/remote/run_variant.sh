#!/bin/bash
# run_variant.sh <gpu> <config.yaml> <run-name> [eval args...]
# Train one variant on one GPU, then evaluate it (NB 300 groups + POPE 600 + IJB preview,
# with the image-removed pass), fit image temperatures on held-out NB groups 300-599,
# and run the text check. Everything lands in /root/runs/<run-name>/.
set -uo pipefail
GPU=$1; CFG=$2; RUN=$3; shift 3
EVAL_ARGS="$@"
export CUDA_VISIBLE_DEVICES=$GPU
R=/root/runs/$RUN; mkdir -p $R
cd /root/sd
echo "commit $(git rev-parse HEAD) gpu $GPU cfg $CFG start $(date -u +%FT%TZ)" > $R/meta.txt
cp $CFG $R/config.yaml
CK=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['output_dir'])")
t0=$(date +%s)
python research/image-track/train.py $CFG > $R/train.log 2>&1 || { echo "TRAIN_FAILED" >> $R/meta.txt; exit 1; }
t1=$(date +%s); echo "train_wall_s $((t1 - t0))" >> $R/meta.txt
cp $CK/history.json $CK/data_manifest.json $R/ 2>/dev/null
python evaluation/vision/run.py --out $R/eval --systems strands --device cuda --checkpoint $CK \
  --nb-groups 300 --pope 600 --ijb-jsonl /root/ijb/ijb_preview.jsonl $EVAL_ARGS > $R/eval.log 2>&1
python evaluation/vision/run.py --out $R/cal --systems strands --device cuda --checkpoint $CK \
  --nb-start 300 --nb-groups 300 --pope 0 --no-blind $EVAL_ARGS > $R/cal.log 2>&1
t2=$(date +%s); echo "eval_wall_s $((t2 - t1))" >> $R/meta.txt
python research/image-track/temps.py fit --rows $R/cal/strands-v19.jsonl --out $R/image_temps.json > $R/temps.log 2>&1
python research/image-track/temps.py apply --run $R/eval --temps $R/image_temps.json --out $R/eval-T >> $R/temps.log 2>&1
python research/image-track/text_check.py --checkpoint $CK --out $R/text_check.json > $R/text.log 2>&1
echo "done $(date -u +%FT%TZ) total_wall_s $(( $(date +%s) - t0 ))" >> $R/meta.txt

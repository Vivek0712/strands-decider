#!/usr/bin/env bash
# One text-track arm on one GPU: train (continue v19) -> calibrate on the rebuilt held-out
# file -> JevBench v1 public set. Usage: arm.sh CONFIG GPU [SEED]
# Writes checkpoints/<name>[-sSEED], /root/runs/<name>, /root/logs/arm-<name>.log; appends to /root/runs/walls.txt.
set -euo pipefail
CFG=$1; G=$2; SEED=${3:-}
export PATH=/root/venv/bin:$PATH PY=/root/venv/bin/python HF_HUB_DISABLE_PROGRESS_BARS=1
cd /root/sd
NAME=$(basename "$CFG" .yaml); [ -n "$SEED" ] && NAME="$NAME-s$SEED"
CK=checkpoints/$NAME
if [ -n "$SEED" ]; then  # the CLI has no --seed: write a copy of the config with the seed changed
  mkdir -p /root/runs/cfg; sed "s/^seed: .*/seed: $SEED/" "$CFG" > "/root/runs/cfg/$NAME.yaml"; CFG=/root/runs/cfg/$NAME.yaml
fi
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$G strands-decider train --config "$CFG" --output-dir "$CK"
t1=$(date +%s)
CUDA_VISIBLE_DEVICES=$G strands-decider calibrate "$CK" --data data/holdout_v5_norule.jsonl
GPU=$G PORT=$((8100+G)) bash evaluation/jevbench/jevbench.sh "$CK" "/root/runs/$NAME"
t2=$(date +%s)
echo "$NAME gpu=$G train_s=$((t1-t0)) total_s=$((t2-t0)) done=$(date -u +%FT%TZ)" >> /root/runs/walls.txt

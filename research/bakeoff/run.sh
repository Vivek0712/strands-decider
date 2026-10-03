#!/usr/bin/env bash
# The base bake-off on one 4-GPU host: configs/experiments/bakeoff/{qwen35-2b,minicpm5-2b,gemma4-e2b}.yaml,
# each trained from its raw base on the same data, budget and seed.
#
#   nohup bash research/bakeoff/run.sh > /root/logs/bakeoff.log 2>&1 &
#
#   1. corpus    training/recipe.sh build ‖ fetch, then multistep ‖ generated ‖ adequacy (CPU),
#                while the 27B teacher and the three bases download at their pinned revisions
#   2. teacher   data/teacher_t4.jsonl on GPU $TEACHER_GPU: Qwen3.5-27B on every yes/no row
#                (vLLM when it imports, else the HF forward of strands_decider.data.teacher),
#                kept where it agrees with gold, over v14's replay (strands_decider.data.teacher_yn)
#   3. arms      one GPU each, in parallel: train -> calibrate on data/holdout_v5_norule.jsonl ->
#                JevBench v1 public (evaluation/jevbench/jevbench.sh) -> the v1.5 proxy
#                (evaluation/jevbench/v15_proxy.py). GPUs 0-2 take the three bases; the
#                teacher's GPU then trains a second Qwen3.5 seed (bakeoff-qwen35-2b-s1),
#                the measure of seed noise the base differences are read against.
#   4. compare   each base's proxy against the two Qwen3.5 seeds, task by task
#
# Writes checkpoints/bakeoff-<name>, $RUNS/bakeoff-<name>/ (JevBench run, proxy.json/txt),
# $LOGS/bakeoff-<name>.log, $RUNS/bakeoff-compare/, and appends one line per arm to
# $RUNS/walls.txt. Rerunning resumes: finished stages are skipped (corpus and teacher by
# marker files, an arm's training by its saved history.json, calibration by a marker in
# the checkpoint, JevBench by run_meta.json); the teacher labelling resumes row by row,
# while an interrupted training run restarts from step 0. A JevBench directory left
# unfinished is moved aside to <dir>.partial-<time>.
#
# Expects the repository at the working tree this script is in, installed editable into
# $PY's venv with the train and cuda extras (plus vllm, optionally), as research/text-track
# runs did. Env: PY (/root/venv/bin/python), RUNS (/root/runs), LOGS (/root/logs),
# TEACHER_GPU (3), ARM_GPUS ("0 1 2", one per base in the order above).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
export PY="${PY:-/root/venv/bin/python}"
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RUNS="${RUNS:-/root/runs}"
LOGS="${LOGS:-/root/logs}"
TEACHER_GPU="${TEACHER_GPU:-3}"
read -r -a ARM_GPUS <<< "${ARM_GPUS:-0 1 2}"
BASES=(qwen35-2b minicpm5-2b gemma4-e2b)
STATE="$RUNS/bakeoff-state"
mkdir -p "$RUNS" "$LOGS" "$STATE" "$RUNS/cfg"

log() { echo "[bakeoff $(date -u +%H:%M:%SZ)] $*"; }
strands-decider() { "$PY" -u -m strands_decider.cli "$@"; }

preflight() {
  [ "${#ARM_GPUS[@]}" = "${#BASES[@]}" ] || { log "ARM_GPUS needs ${#BASES[@]} GPUs"; exit 2; }
  "$PY" - <<'EOF'
import strands_decider, transformers
assert hasattr(transformers, "Gemma4TextModel") and hasattr(transformers, "Qwen3_5ForCausalLM"), \
    f"transformers {transformers.__version__} lacks Gemma 4 or Qwen3.5"
print("strands_decider", strands_decider.__file__, "transformers", transformers.__version__)
EOF
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
}

# verify FILE...: as training/recipe.sh does -- the teacher file indexes these exact bytes.
verify() {
  local f
  for f; do awk -v f="$f" '$2 == f' data/SHA256SUMS | grep -q . || { log "$f: not in data/SHA256SUMS"; return 1; }; done
  for f; do awk -v f="$f" '$2 == f' data/SHA256SUMS; done | sha256sum -c --quiet --strict -
}

# Hub downloads at the pinned revisions, so the GPUs never wait on the network.
prefetch() {
  "$PY" - "${BASES[@]}" <<'EOF'
import sys

import yaml
from huggingface_hub import snapshot_download

from strands_decider.data.teacher_yn import MODEL, REVISION

repos = [(MODEL, REVISION)]
for name in sys.argv[1:]:
    cfg = yaml.safe_load(open(f"configs/experiments/bakeoff/{name}.yaml"))
    repos.append((cfg["base_model"], cfg["base_model_revision"]))
for repo, rev in repos:
    print("downloaded", repo, rev, snapshot_download(repo, revision=rev), flush=True)
EOF
}

corpus() {
  if [ -f "$STATE/corpus.done" ]; then log "corpus: done before"; return; fi
  log "corpus: build ‖ fetch, then multistep ‖ generated ‖ adequacy"
  training/recipe.sh build > "$LOGS/bakeoff-build.log" 2>&1 & local b=$!
  training/recipe.sh fetch > "$LOGS/bakeoff-fetch.log" 2>&1 & local f=$!
  wait "$b"; wait "$f"
  local s pids=()
  for s in multistep generated adequacy; do
    training/recipe.sh "$s" > "$LOGS/bakeoff-$s.log" 2>&1 & pids+=("$!")
  done
  for s in "${pids[@]}"; do wait "$s"; done
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  sha256sum data/*.jsonl > "$RUNS/bakeoff-data_sha256.txt"
  touch "$STATE/corpus.done"
}

# data/teacher_t4.jsonl. The raw labels name their engine in a marker, so a resumed run
# never appends one engine's rows to the other's file.
teacher() {
  if [ -f "$STATE/teacher.done" ]; then log "teacher: done before"; return; fi
  verify data/train_v5.jsonl data/multistep_v14.jsonl data/synthetic/generated_v16.jsonl \
    data/synthetic/generated_v18.jsonl data/synthetic/adequacy_gen.jsonl \
    data/synthetic/replay_v14_multistep.jsonl
  cp data/synthetic/replay_v14_multistep.jsonl data/
  local raw=data/teacher_t4_raw.jsonl engine
  if [ -f "$raw.engine" ]; then engine=$(cat "$raw.engine")
  elif "$PY" -c "import vllm" 2>/dev/null; then engine=vllm
  else engine=hf; fi
  echo "$engine" > "$raw.engine"
  log "teacher: Qwen3.5-27B on GPU $TEACHER_GPU ($engine)"
  if ! CUDA_VISIBLE_DEVICES=$TEACHER_GPU "$PY" -m strands_decider.data.teacher_yn label \
      --engine "$engine" --out "$raw" >> "$LOGS/bakeoff-teacher.log" 2>&1; then
    [ "$engine" = vllm ] || { log "teacher: the HF labelling failed (see $LOGS/bakeoff-teacher.log)"; exit 1; }
    log "teacher: vLLM failed; relabelling with the HF forward"
    mv "$raw" "$raw.vllm-failed-$(date +%s)" 2>/dev/null || true
    echo hf > "$raw.engine"
    CUDA_VISIBLE_DEVICES=$TEACHER_GPU "$PY" -m strands_decider.data.teacher_yn label \
      --engine hf --out "$raw" >> "$LOGS/bakeoff-teacher.log" 2>&1
  fi
  "$PY" -m strands_decider.data.teacher_yn build --raw "$raw" \
    --base data/replay_v14_multistep.jsonl --out data/teacher_t4.jsonl | tee -a "$LOGS/bakeoff-teacher.log"
  sha256sum "$raw" data/teacher_t4.jsonl | tee -a "$RUNS/bakeoff-data_sha256.txt"
  touch "$STATE/teacher.done"
}

# arm NAME GPU [SEED]: train -> calibrate -> JevBench -> proxy, each skipped if done.
arm() {
  local name="bakeoff-$1" gpu=$2 seed=${3:-} cfg="configs/experiments/bakeoff/$1.yaml"
  if [ -n "$seed" ]; then  # the CLI has no --seed: a copy of the config with the seed changed
    name="$name-s$seed"
    sed "s/^seed: .*/seed: $seed/" "$cfg" > "$RUNS/cfg/$name.yaml"; cfg="$RUNS/cfg/$name.yaml"
  fi
  local ck="checkpoints/$name" out="$RUNS/$name" t0 t1 t2
  t0=$(date +%s)
  if [ -f "$ck/history.json" ]; then log "$name: trained before"
  else
    log "$name: training on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu strands-decider train --config "$cfg" --output-dir "$ck"
  fi
  t1=$(date +%s)
  if [ ! -f "$ck/.calibrated" ]; then
    CUDA_VISIBLE_DEVICES=$gpu strands-decider calibrate "$ck" --data data/holdout_v5_norule.jsonl
    touch "$ck/.calibrated"
  fi
  if [ ! -f "$out/run_meta.json" ]; then
    [ ! -e "$out" ] || mv "$out" "$out.partial-$(date +%s)"
    # One JevBench at a time: each run checks out and installs the shared JevBench clone.
    flock "$STATE/jevbench.lock" env GPU="$gpu" PORT=$((8100 + gpu)) \
      bash evaluation/jevbench/jevbench.sh "$ck" "$out"
  fi
  "$PY" evaluation/jevbench/v15_proxy.py "$out" --json "$out/proxy.json" > "$out/proxy.txt"
  t2=$(date +%s)
  echo "$name gpu=$gpu train_s=$((t1 - t0)) total_s=$((t2 - t0)) done=$(date -u +%FT%TZ)" >> "$RUNS/walls.txt"
  log "$name: done, $(head -1 "$out/proxy.txt")"
}

compare() {
  local base qwen=("$RUNS/bakeoff-qwen35-2b" "$RUNS/bakeoff-qwen35-2b-s1") d="$RUNS/bakeoff-compare"
  mkdir -p "$d"
  "$PY" evaluation/jevbench/v15_proxy.py "${qwen[1]}" --vs "${qwen[0]}" --json "$d/qwen-s1-vs-s0.json" \
    > "$d/qwen-s1-vs-s0.txt"
  for base in minicpm5-2b gemma4-e2b; do
    "$PY" evaluation/jevbench/v15_proxy.py "$RUNS/bakeoff-$base" --vs "${qwen[@]}" \
      --json "$d/$base-vs-qwen.json" > "$d/$base-vs-qwen.txt"
  done
  tail -n 1 "$d"/*.txt
}

preflight
prefetch > "$LOGS/bakeoff-prefetch.log" 2>&1 & PREFETCH=$!
corpus
wait "$PREFETCH" || { log "a download failed (see $LOGS/bakeoff-prefetch.log)"; exit 1; }
teacher
pids=()
for i in "${!BASES[@]}"; do
  arm "${BASES[$i]}" "${ARM_GPUS[$i]}" > "$LOGS/bakeoff-${BASES[$i]}.log" 2>&1 & pids+=("$!")
done
arm qwen35-2b "$TEACHER_GPU" 1 > "$LOGS/bakeoff-qwen35-2b-s1.log" 2>&1 & pids+=("$!")
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
[ "$bad" = 0 ] || { log "an arm failed: see $LOGS/bakeoff-*.log"; exit 1; }
compare
log "all done; walls in $RUNS/walls.txt"

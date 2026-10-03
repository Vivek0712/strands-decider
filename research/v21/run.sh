#!/usr/bin/env bash
# v21: the bake-off's winning base, trained for one full epoch at three seeds from one
# initialisation, then the three weight-averaged into a soup. On the bake-off's 4-GPU host,
# after research/bakeoff/run.sh has finished:
#
#   cd /root/sd && git pull && bash research/v21/run.sh WINNER    # qwen35-2b|minicpm5-2b|gemma4-e2b
#   (or: nohup bash research/v21/run.sh WINNER > /root/logs/v21.log 2>&1 &)
#
#   1. configs  $RUNS/cfg/v21-WINNER-s{0,1,2}.yaml from configs/experiments/bakeoff/WINNER.yaml,
#               the bake-off recipe with three changes: max_steps = one full epoch of the
#               train mixture at effective batch 32 (derived below), init_seed 0 for all
#               three, and seed 0/1/2. Same init_seed: one starting point, so the three can
#               be averaged; different seed: different validation split, data order, option
#               shuffles and dropout.
#               WINNER=qwen35-2b does not train from the raw base: v19 already is Qwen3.5-2B
#               trained on the full recipe. Its three arms continue from v19
#               (continue_from: checkpoints/v19, a link to $V19_CKPT), with the bake-off's
#               27B teacher and no frozen-KL on yes/no rows (the v19-yn27b recipe) for the
#               same full epoch -- so their shared initialisation is v19's own adapter and
#               head, and init_seed changes nothing for them (train.py).
#   2. arms     one GPU each ($ARM_GPUS), in parallel: train, then calibrate on
#               data/holdout_v5_norule.jsonl -> JevBench v1 public -> the v1.5 proxy. GPU
#               $EVAL_GPU meanwhile runs the bake-off winner's own calibrate -> JevBench ->
#               proxy if the bake-off left it unfinished, and is otherwise idle.
#   3. soup     once all three have trained: python -m strands_decider.soup of the three
#               (head averaged, LoRA as the mean of the merged updates, rank 48), then
#               calibrate -> JevBench -> proxy on GPU $EVAL_GPU.
#   4. compare  in $RUNS/v21-$WINNER-compare/: the three seeds and the soup against the bake-off's
#               Qwen3.5 arms (bakeoff-qwen35-2b, -s1), against the bake-off winner, against
#               each other, and against v19's own JevBench run(s), $V19_RUNS, when present.
#
# Max steps: one epoch of the train split is floor(floor(rows / micro_batch) / grad_accum)
# optimizer steps (the length-grouped sampler drops only the last short micro-batch). The
# rows are read from the winner's bake-off log ("train=119,623 val=..." -> 119,623 / 8 / 4 =
# 3,738 steps). STEPS overrides it; with neither, max_steps is 0 and train.py runs exactly
# one epoch itself (the same number).
#
# Writes checkpoints/v21-WINNER-{s0,s1,s2,soup}, $RUNS/v21-WINNER-{s0,s1,s2,soup}/ (JevBench run,
# proxy.json/txt), $LOGS/v21-*.log, $RUNS/v21-$WINNER-compare/, one line per arm in $RUNS/walls.txt,
# and $RUNS/v21-state/ALL_DONE at the very end. Rerunning resumes, as research/bakeoff/run.sh
# does: an arm's training is skipped once its history.json exists (an interrupted run
# restarts from step 0), calibration once its .calibrated marker exists, JevBench once its
# run_meta.json exists (an unfinished JevBench directory is moved aside), and the soup once
# its soup.json exists.
#
# Env: PY (/root/venv/bin/python), RUNS (/root/runs), LOGS (/root/logs), ARM_GPUS ("0 1 2"),
# EVAL_GPU (3), V19_CKPT (/root/ck/v19), V19_RUNS (JevBench run dirs of v19, space-separated;
# default $RUNS/v19 if it exists), STEPS (derived), JEVBENCH_LOCK (the bake-off's lock, so the
# two runners never run JevBench at once), V21_IGNORE_BAKEOFF=1 to start while a bake-off
# runner is still alive.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # the repository root
WINNER=${1:?usage: research/v21/run.sh qwen35-2b|minicpm5-2b|gemma4-e2b}
case "$WINNER" in qwen35-2b|minicpm5-2b|gemma4-e2b) ;; *) echo "unknown winner $WINNER" >&2; exit 2;; esac
export PY="${PY:-/root/venv/bin/python}"
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RUNS="${RUNS:-/root/runs}"
LOGS="${LOGS:-/root/logs}"
read -r -a ARM_GPUS <<< "${ARM_GPUS:-0 1 2}"
EVAL_GPU="${EVAL_GPU:-3}"
V19_CKPT="${V19_CKPT:-/root/ck/v19}"
V19_RUNS="${V19_RUNS:-$([ -d "$RUNS/v19" ] && echo "$RUNS/v19" || true)}"
SEEDS=(0 1 2)
STATE="$RUNS/v21-state"
JEVBENCH_LOCK="${JEVBENCH_LOCK:-$RUNS/bakeoff-state/jevbench.lock}"
SRC_CFG="configs/experiments/bakeoff/$WINNER.yaml"
BAKEOFF_CK="checkpoints/bakeoff-$WINNER"
mkdir -p "$RUNS" "$LOGS" "$STATE" "$RUNS/cfg" "$(dirname "$JEVBENCH_LOCK")"

log() { echo "[v21 $(date -u +%H:%M:%SZ)] $*"; }
strands-decider() { "$PY" -u -m strands_decider.cli "$@"; }

preflight() {
  [ "${#ARM_GPUS[@]}" = "${#SEEDS[@]}" ] || { log "ARM_GPUS needs ${#SEEDS[@]} GPUs"; exit 2; }
  if [ "${V21_IGNORE_BAKEOFF:-0}" != 1 ] && pgrep -f "research/bakeoff/run.sh" > /dev/null; then
    log "the bake-off runner is still running; wait for it (or set V21_IGNORE_BAKEOFF=1)"; exit 2
  fi
  [ -f "$RUNS/bakeoff-state/corpus.done" ] && [ -f "$RUNS/bakeoff-state/teacher.done" ] \
    || { log "the bake-off's corpus and teacher are not done: run research/bakeoff/run.sh first"; exit 2; }
  local f
  for f in data/teacher_t4.jsonl data/holdout_v5_norule.jsonl; do
    [ -s "$f" ] || { log "$f missing"; exit 2; }
  done
  if [ "$WINNER" = qwen35-2b ]; then
    if [ ! -e checkpoints/v19 ]; then
      [ -d "$V19_CKPT" ] || { log "no checkpoints/v19 and no $V19_CKPT"; exit 2; }
      mkdir -p checkpoints && ln -s "$V19_CKPT" checkpoints/v19
    fi
    "$PY" -c 'import sys; from strands_decider.modeling import StrandsDeciderConfig, config_path
c = StrandsDeciderConfig.from_json(config_path("checkpoints/v19"))
assert c.head_type == "pointer" and c.base_model.startswith("Qwen/Qwen3.5-2B"), c
print("v19:", c.base_model, c.base_model_revision, "lora r", c.lora_r)'
  fi
  "$PY" -c 'import strands_decider.soup, strands_decider.train as t; assert "init_seed" in t.TrainConfig.__dataclass_fields__'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
}

# The configs, and the step count they carry (printed).
configs() {
  "$PY" - "$WINNER" "$SRC_CFG" "$LOGS/bakeoff-$WINNER.log" "$RUNS/cfg" "${STEPS:-}" "${SEEDS[@]}" <<'EOF'
import os, re, sys

import yaml

winner, src, bakeoff_log, out_dir, steps_env, *seeds = sys.argv[1:]
cfg = yaml.safe_load(open(src))
eff = cfg["micro_batch_size"] * cfg["grad_accum"]
if steps_env:
    steps, why = int(steps_env), "STEPS"
else:
    rows = None
    if os.path.exists(bakeoff_log):
        m = re.search(r"\[strands-decider\] train=([\d,]+) val=", open(bakeoff_log, errors="replace").read())
        rows = int(m.group(1).replace(",", "")) if m else None
    if rows:
        steps = rows // cfg["micro_batch_size"] // cfg["grad_accum"]
        why = f"{rows:,} train rows / {cfg['micro_batch_size']} / {cfg['grad_accum']} ({bakeoff_log})"
    else:
        steps, why = 0, "no row count in the bake-off log: 0 = train.py's own full epoch"
cfg.update(epochs=1, max_steps=steps, init_seed=0)
if winner == "qwen35-2b":
    # v19 continued (the v19-yn27b recipe, extended to a full epoch): its adapter and head.
    cfg.update(continue_from="checkpoints/v19")
for seed in seeds:
    name = f"v21-{winner}-s{seed}"
    cfg.update(seed=int(seed), output_dir=f"checkpoints/{name}")
    with open(os.path.join(out_dir, f"{name}.yaml"), "w") as fh:
        fh.write(f"# generated by research/v21/run.sh from {src}; max_steps: {why}\n")
        yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"max_steps {steps} (effective batch {eff}; {why})")
EOF
}

# evaluate NAME CKPT GPU: calibrate -> JevBench -> proxy, each skipped if done.
evaluate() {
  local name=$1 ck=$2 gpu=$3 out="$RUNS/$1"
  if [ ! -f "$ck/.calibrated" ]; then
    CUDA_VISIBLE_DEVICES=$gpu strands-decider calibrate "$ck" --data data/holdout_v5_norule.jsonl
    touch "$ck/.calibrated"
  fi
  if [ ! -f "$out/run_meta.json" ]; then
    [ ! -e "$out" ] || mv "$out" "$out.partial-$(date +%s)"
    # One JevBench at a time: each run checks out and installs the shared JevBench clone.
    flock "$JEVBENCH_LOCK" env GPU="$gpu" PORT=$((8100 + gpu)) \
      bash evaluation/jevbench/jevbench.sh "$ck" "$out"
  fi
  "$PY" evaluation/jevbench/v15_proxy.py "$out" --json "$out/proxy.json" > "$out/proxy.txt"
  log "$name: $(head -1 "$out/proxy.txt")"
}

# train_arm SEED GPU
train_arm() {
  local name="v21-$WINNER-s$1" gpu=$2
  local ck="checkpoints/$name" t0 t1
  t0=$(date +%s)
  if [ -f "$ck/history.json" ]; then log "$name: trained before"
  else
    log "$name: training on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu strands-decider train --config "$RUNS/cfg/$name.yaml" --output-dir "$ck"
  fi
  t1=$(date +%s)
  echo "$name gpu=$gpu train_s=$((t1 - t0)) done=$(date -u +%FT%TZ)" >> "$RUNS/walls.txt"
}

# eval_arm SEED GPU
eval_arm() {
  local name="v21-$WINNER-s$1" gpu=$2 t0
  t0=$(date +%s)
  evaluate "$name" "checkpoints/$name" "$gpu"
  echo "$name gpu=$gpu eval_s=$(( $(date +%s) - t0 )) done=$(date -u +%FT%TZ)" >> "$RUNS/walls.txt"
}

# The bake-off winner's own run, if the bake-off left it unfinished.
winner_eval() {
  if [ -f "$RUNS/bakeoff-$WINNER/run_meta.json" ] && [ -f "$RUNS/bakeoff-$WINNER/proxy.json" ]; then
    log "bakeoff-$WINNER: evaluated by the bake-off; GPU $EVAL_GPU idle until the soup"; return
  fi
  [ -f "$BAKEOFF_CK/history.json" ] || { log "$BAKEOFF_CK: not trained; nothing to evaluate"; return; }
  local t0; t0=$(date +%s)
  evaluate "bakeoff-$WINNER" "$BAKEOFF_CK" "$EVAL_GPU"
  echo "bakeoff-$WINNER gpu=$EVAL_GPU eval_s=$(( $(date +%s) - t0 )) done=$(date -u +%FT%TZ) (by v21)" \
    >> "$RUNS/walls.txt"
}

soup() {
  local name="v21-$WINNER-soup" ck="checkpoints/v21-$WINNER-soup" s ins=() t0
  t0=$(date +%s)
  for s in "${SEEDS[@]}"; do ins+=("checkpoints/v21-$WINNER-s$s"); done
  if [ -f "$ck/soup.json" ]; then log "$name: souped before"
  else
    [ ! -e "$ck" ] || mv "$ck" "$ck.partial-$(date +%s)"
    log "$name: soup of ${ins[*]}"
    "$PY" -u -m strands_decider.soup --out "$ck" "${ins[@]}"
  fi
  evaluate "$name" "$ck" "$EVAL_GPU"
  echo "$name gpu=$EVAL_GPU total_s=$(( $(date +%s) - t0 )) done=$(date -u +%FT%TZ)" >> "$RUNS/walls.txt"
}

# vs TAG "RUNS..." "BASELINES...": a v15_proxy comparison, skipped when a run is missing.
vs() {
  local tag=$1 r d="$RUNS/v21-$WINNER-compare" a b
  read -r -a a <<< "$2"; read -r -a b <<< "$3"
  for r in "${a[@]}" "${b[@]}"; do
    [ -f "$r/results.jsonl" ] || { log "compare $tag: skipped, $r has no results.jsonl"; return 0; }
  done
  "$PY" evaluation/jevbench/v15_proxy.py "${a[@]}" --vs "${b[@]}" --json "$d/$tag.json" > "$d/$tag.txt"
  log "compare $tag: $(tail -1 "$d/$tag.txt")"
}

compare() {
  mkdir -p "$RUNS/v21-$WINNER-compare"
  local s arms=() soup_run="$RUNS/v21-$WINNER-soup" qwen="$RUNS/bakeoff-qwen35-2b $RUNS/bakeoff-qwen35-2b-s1"
  for s in "${SEEDS[@]}"; do arms+=("$RUNS/v21-$WINNER-s$s"); done
  vs seeds-vs-bakeoff-qwen "${arms[*]}" "$qwen"
  vs soup-vs-bakeoff-qwen "$soup_run" "$qwen"
  vs soup-vs-seeds "$soup_run" "${arms[*]}"
  if [ "$WINNER" != qwen35-2b ]; then
    vs seeds-vs-bakeoff-winner "${arms[*]}" "$RUNS/bakeoff-$WINNER"
    vs soup-vs-bakeoff-winner "$soup_run" "$RUNS/bakeoff-$WINNER"
  fi
  if [ -n "$V19_RUNS" ]; then
    vs seeds-vs-v19 "${arms[*]}" "$V19_RUNS"
    vs soup-vs-v19 "$soup_run" "$V19_RUNS"
  else
    log "compare: no v19 JevBench run (set V19_RUNS); v19 comparisons skipped"
  fi
}

preflight
STEPS_MSG=$(configs)  # an assignment, so set -e sees a failure
log "winner $WINNER: $STEPS_MSG"
pids=()
for i in "${!SEEDS[@]}"; do
  train_arm "${SEEDS[$i]}" "${ARM_GPUS[$i]}" > "$LOGS/v21-$WINNER-s${SEEDS[$i]}.log" 2>&1 & pids+=("$!")
done
winner_eval > "$LOGS/v21-bakeoff-$WINNER-eval.log" 2>&1 & WINNER_EVAL=$!
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
[ "$bad" = 0 ] || { log "a training run failed: see $LOGS/v21-$WINNER-s*.log"; exit 1; }
log "all three trained"
pids=()
for i in "${!SEEDS[@]}"; do
  eval_arm "${SEEDS[$i]}" "${ARM_GPUS[$i]}" >> "$LOGS/v21-$WINNER-s${SEEDS[$i]}.log" 2>&1 & pids+=("$!")
done
# The soup's GPU is the winner-eval GPU: wait for that first.
wait "$WINNER_EVAL" || { log "the bake-off winner's evaluation failed (see $LOGS/v21-bakeoff-$WINNER-eval.log)"; bad=1; }
soup > "$LOGS/v21-$WINNER-soup.log" 2>&1 & pids+=("$!")
for p in "${pids[@]}"; do wait "$p" || bad=1; done
[ "$bad" = 0 ] || { log "an evaluation failed: see $LOGS/v21-*.log"; exit 1; }
compare
{ echo "winner=$WINNER done=$(date -u +%FT%TZ)"; grep -h "^max_steps" "$RUNS/cfg/v21-$WINNER-s0.yaml" || true
  for s in "${SEEDS[@]/#/s}" soup; do echo "v21-$WINNER-$s $(head -1 "$RUNS/v21-$WINNER-$s/proxy.txt")"; done
} > "$STATE/ALL_DONE"
log "all done: $STATE/ALL_DONE; walls in $RUNS/walls.txt"

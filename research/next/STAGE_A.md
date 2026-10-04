# Next model, Stage A: what was built, and the commands for Stages B to E

Stage A of the next-model plan is engineering only: no GPU time. It builds the measuring
stick (unseen-v2), the three recipe changes the Stage D screen tests (A1, A2, A3), the
configs for Stages B, D and E, the speed tooling, and the Stage C rank check. Every
decision rule below is fixed here, before any stage runs. Paths are relative to the
repository root.

## What was built

| Item | Where | Notes |
| --- | --- | --- |
| unseen-v2 | `evaluation/unseen_v2/` ([README](../../evaluation/unseen_v2/README.md)) | 9,000 rows, 3,000 per type, 11 families, 40/60 dev/test per family, deterministic from a seed |
| Scoring | `evaluation/unseen_v2/score.py` | v1.5-rule competences plus NLL, yes/no Brier, band mass, score RPS, ECE; item-paired bootstrap, optionally over seeds |
| A1 training families | `src/strands_decider/data/families/` | state tracking, tool-call guardrails, lead rubrics, JSON policy; all three types; disjoint from unseen-v2 (tested) |
| A2: RPS + NLL | `TrainConfig.score_rps_weight`, `modeling.ranked_probability_score` | RPS = (1/(K−1))·Σ_k (F_k − 1[gold ≤ k])², added to the score rows' NLL; off by default |
| A3: fixed two-teacher loss | `TrainConfig.pool_teacher_files`, `pool_alpha`; `data/teacher_pool.py` | α·KL(pool ‖ student) + (1−α)·CE(gold) on every labelled row; pool = mean of per-teacher, per-kind temperature-recalibrated probabilities (NLL on a held-out split, stored as `<file>.calibration.json`); works with one teacher; off by default |
| Second teacher | `data/teacher.py` | `load` reads a multimodal Gemma 4 checkpoint's text decoder; `label` soft-caps its letter logits; `teacher_yn label --kinds` labels choice rows too |
| Configs | `configs/next/stage-{b,d,e}/` | Stage B pilots (4B-Base, 4B post-trained, 2B reference); Stage D's eight cells; Stage E's two seeds |
| Speed | `training/install_fast_kernels.sh`, `training/bench_steps.py` | fused Gated DeltaNet kernels for the installed torch/CUDA; step/s of any config ([training/README.md](../../training/README.md#fused-kernels-and-the-step-rate)) |
| Stage C | `evaluation/unseen_v2/{rank_check,adapters,spearman}.py`, `systems.json`, `official_v1.5.5.csv` | board systems on unseen-v2 through `unseen/run.py`'s adapters, evaluation only; Spearman against the official order |
| Stage D read-out | `evaluation/unseen_v2/factorial.py` | main effects with paired CIs and the decision rule |
| Stage E | `soup(..., coefs=)`, `evaluation/unseen_v2/interpolate.py`, `evaluation/unseen_v2/calibrate_mixed.py` | interpolation-barrier check before souping; 50/50 in-family + unseen-dev calibration |

### unseen-v2 rows (seed 0)

| Family | Source | Yes/no | Choice | Score |
| --- | --- | ---: | ---: | ---: |
| StrategyQA | `ChilleD/StrategyQA` @7055626 (MIT) | 400 | | |
| RuleTaker depth 5 | `strands-decider data build` held-out file | 400 | | |
| CommonsenseQA 2.0 | `tasksource/commonsense_qa_2.0` @23ff83b (CC BY 4.0) | 400 | | |
| CommonsenseQA | `tau/commonsense_qa` @94630fe (MIT) | | 400 | |
| ARC-Challenge | `allenai/ai2_arc` @210d026 (CC BY-SA 4.0) | | 400 | |
| HellaSwag | `Rowan/hellaswag` @218ec52 (MIT) | | 400 | |
| STS-B | `sentence-transformers/stsb` @ab7a5ac | | | 600 |
| chess | generated (python-chess rules) | 450 | 450 | 600 |
| arithmetic | generated | 450 | 450 | 600 |
| calendar | generated | 450 | 450 | 600 |
| seating | generated | 450 | 450 | 600 |
| **Total** | | **3,000** | **3,000** | **3,000** |

Dev: 1,201 yes/no, 1,205 choice, 1,194 score. Built twice on a laptop CPU (about 2 min
each) with the same output sha256, `7d90f28a5d24d3915b65eda3fce1330fada7a40f8ef2928996b2e0ddf709ac0e`,
with RuleTaker from a held-out file of sha256
`75358ee65d2952c3d094fe77a96e4540aee0e33832041ee8271b99c94bb3556e`. A rebuild on the GPU
host must reproduce that hash before any model is read on the set. Excluded on purpose and
tested: BoolQ and MNLI (training sources, so no BoolQ-hard or e-SNLI) and anything from
JevBench or Image JevBench. File hashes and licences are in `data/sources.md`.

### A1 rows (seed 0; evaluation seed 1)

| Family | Yes/no | Choice | Score | Asked as |
| --- | ---: | ---: | ---: | --- |
| state_tracking | 2,000 | 2,000 | 2,000 | is X in state S / which state / how many end in S |
| tool_guardrail | 2,000 | 2,000 | 2,000 | does the call violate / which rule, or none / how many rules |
| lead_rubric | 2,000 | 2,000 | 2,000 | meets X or qualified / which criterion fails / the 0–4 rating |
| json_policy | 2,000 | 2,000 | 2,000 | compliant / approve, escalate or deny / how many conditions hold |
| **Total** | **8,000** | **8,000** | **8,000** | 24,000 rows |

The in-family evaluation file has 1,797 rows (150 per family and type, less 3 that repeat
a training row and are dropped). File sha256s: training
`04f326352ee0793ab38d13c5c0badb65f7fb49ee71150ee134e1e8c328a31bbe`, evaluation
`83d058e4aea3272416731feba61fbdf3bbf714f33084c725b362b23f32fe930b`. States come in four formats
(indented JSON, compact JSON, key: value lines, prose) and every question in two or three
phrasings.

## Decision rules (fixed before the stages run)

- **Stage B**: go to 4B only if the better 4B arm scores at least +5 public v1.5-style
  Intelligence over the 2B from raw at the same 1,500 steps, **and** is not worse on unseen-v2
  dev (paired 95% CI of the Intelligence difference not entirely below 0).
- **Stage C**: proceed only if Spearman ρ(our unseen-v2 Intelligence, official sealed
  Intelligence) ≥ 0.7 over the systems run. It checks the ordering, not the numbers.
- **Stage D**: adopt arm X only if its main effect on its pre-registered metric is positive
  with a paired 95% CI excluding 0, **and** no other headline metric (unseen-v2 Intelligence,
  the three competences, public Intelligence) drops by more than 1 point. Metrics: A1 unseen-v2
  Intelligence, A2 score competence (expected-level grading), A3 yes/no competence. Read on
  `--split dev`.
- **A3's second teacher** (Gemma-4-31B-it) joins the pool only if, on the 1,000-row pilot, its
  recalibrated held-out NLL is no worse than the 27B's + 0.02 on both yes/no and choice rows;
  otherwise A3 is the 27B alone, recalibrated.
- **Stage E**: soup the two seeds only if `interpolate.py` finds no barrier (tolerance 0);
  propose for submission only if the 4B beats our best 2B on unseen-v2 **test** (paired CI
  excluding 0, read once) with pooled ECE no worse than v20-long's.

## Setup on every GPU host

```bash
pip install -e ".[train,vision,cuda,unseen]"
training/install_fast_kernels.sh                         # all arms with the fused kernels, or none
hf download StrandsAgents/strands-decider-2B-hobson-v19 --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd \
  --local-dir checkpoints/v19
training/recipe.sh build fetch multistep generated adequacy teacher_yn   # v20's data and 27B file
python evaluation/unseen_v2/build.py --out data/unseen_v2.jsonl           # check the sha256 above
```

`checkpoints/strands-decider-2B-hobson-v20` is the calibrated v20 soup (docs/v20-v21.md
builds it); copy it from wherever the results release keeps it. `calibrate CKPT` below is
the two-step calibration function in [docs/v20-v21.md](../../docs/v20-v21.md#calibration).

## Stage B: 4B pilot (2×H200, about 1.8 h)

```bash
python training/bench_steps.py --config configs/next/stage-b/pilot-4b-base.yaml --steps 20 --json reports/bench-4b.json
CUDA_VISIBLE_DEVICES=0 strands-decider train --config configs/next/stage-b/pilot-4b-base.yaml &
CUDA_VISIBLE_DEVICES=1 strands-decider train --config configs/next/stage-b/pilot-4b-instruct.yaml; wait
strands-decider train --config configs/next/stage-b/pilot-2b-base.yaml    # unless the 1,500-step 2B-from-raw run exists
for m in pilot-4b-base pilot-4b-instruct pilot-2b-base; do
  calibrate checkpoints/next-b-$m
  PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/next-b-$m reports/jevbench/next-b-$m
  python evaluation/jevbench/v15_proxy.py reports/jevbench/next-b-$m
  python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint checkpoints/next-b-$m --out reports/unseen_v2/next-b-$m
done
python evaluation/unseen_v2/score.py reports/unseen_v2/next-b-pilot-4b-base --vs reports/unseen_v2/next-b-pilot-2b-base --split dev
python evaluation/unseen_v2/score.py reports/unseen_v2/next-b-pilot-4b-instruct --vs reports/unseen_v2/next-b-pilot-2b-base --split dev
```

## Stage C: rank check (1–2 GPUs; bring-up on a cheap card)

Download each system at a pinned revision, run it, then compare orderings. Three systems are
wired (`decider-2b`, `decider-4b-v2`, `decision-2b-preview`); the others in `systems.json` need
their inference code read at the pinned revision and a factory in `adapters.py`, or a System
One server (`--url`). Never train on these answers.

```bash
python evaluation/unseen_v2/rank_check.py run --rows data/unseen_v2.jsonl --out reports/rank \
  --repo decider-2b=DIR --revision decider-2b=COMMIT \
  --repo decider-4b-v2=DIR --revision decider-4b-v2=COMMIT \
  --repo decision-2b-preview=DIR --revision decision-2b-preview=COMMIT
python evaluation/unseen_v2/spearman.py --official evaluation/unseen_v2/official_v1.5.5.csv --dir reports/rank \
  --run v20=reports/unseen_v2/v20-soup --json reports/rank/spearman.json
```

## Stage D: the 2³ factorial (4×H200, two waves of four, about 3.2 h)

Data for the arms, once:

```bash
python -m strands_decider.data.families --out data/families_a1.jsonl --eval-out data/families_a1_eval.jsonl
# A3: the 27B on every yes/no and choice row, agreeing or not. Seeding the file with the
# yes/no labels teacher_yn already wrote makes the run label only the choice rows.
cp data/teacher_yn_qwen35-27b_raw.jsonl data/teacher_pool_qwen35-27b.jsonl
python -m strands_decider.data.teacher_yn label --engine vllm --kinds noul choice --out data/teacher_pool_qwen35-27b.jsonl
python -m strands_decider.data.teacher_pool fit --teacher data/teacher_pool_qwen35-27b.jsonl
```

Optional second teacher, only through the pilot rule above:

```bash
python - <<'EOF'
import random
from strands_decider.data.format import read_jsonl, write_jsonl
from strands_decider.data.teacher_yn import SOURCES
rows = [e for f in SOURCES for e in read_jsonl(f) if e.kind in ("noul", "choice")]
write_jsonl("data/pilot_1k.jsonl", random.Random(0).sample(rows, 1000))
EOF
for t in "Qwen/Qwen3.5-27B fc05daec18b0a78c049392ed2e771dde82bdf654 27b" "google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475 gemma"; do
  set -- $t
  python -m strands_decider.data.teacher_yn label --model $1 --revision $2 --kinds noul choice \
    --train-files data/pilot_1k.jsonl --sources data/pilot_1k.jsonl --out data/pilot_1k_$3.jsonl
  python -m strands_decider.data.teacher_pool fit --teacher data/pilot_1k_$3.jsonl \
    --train-files data/pilot_1k.jsonl --heldout-fraction 1.0
done
# If Gemma passes: label the full corpus as for the 27B (data/teacher_pool_gemma4-31b.jsonl),
# fit it, and add it to pool_teacher_files in all four A3-on cells (the same edit in each).
```

The eight cells (two waves; each cell on one GPU):

```bash
for wave in "000 001 010 011" "100 101 110 111"; do
  i=0; for c in $wave; do
    CUDA_VISIBLE_DEVICES=$i strands-decider train --config configs/next/stage-d/cell-$c.yaml & i=$((i+1))
  done; wait
done
for c in 000 001 010 011 100 101 110 111; do
  calibrate checkpoints/next-d-$c
  python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint checkpoints/next-d-$c --out reports/unseen_v2/d$c
  PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/next-d-$c reports/jevbench/next-d-$c
  python evaluation/jevbench/v15_proxy.py reports/jevbench/next-d-$c --json reports/jevbench/next-d-$c.json
done
python - <<'EOF'   # public v1.5-style Intelligence per cell, for the guard
import json
cells = ["000", "001", "010", "011", "100", "101", "110", "111"]
out = {c: next(iter(json.load(open(f"reports/jevbench/next-d-{c}.json"))["runs"].values()))["intelligence_proxy"] for c in cells}
json.dump(out, open("reports/stage-d-public.json", "w"))
EOF
python evaluation/unseen_v2/factorial.py $(for c in 000 001 010 011 100 101 110 111; do echo --cell $c=reports/unseen_v2/d$c; done) \
  --split dev --public reports/stage-d-public.json --json reports/stage-d.json
```

## Stage E: the 4B build (2×H200, about 5 h per seed), only if Stage B passed

```bash
ln -s next-b-pilot-4b-base checkpoints/next-b-pilot-4b        # or -instruct: the better pilot arm
# Switch on each arm Stage D adopted in configs/next/stage-e/4b.yaml and 4b-seed1.yaml, with the
# keys of the matching Stage D cell (A1 also needs max_steps = new row count / 32).
CUDA_VISIBLE_DEVICES=0 strands-decider train --config configs/next/stage-e/4b.yaml &
CUDA_VISIBLE_DEVICES=1 strands-decider train --config configs/next/stage-e/4b-seed1.yaml; wait
python evaluation/unseen_v2/interpolate.py checkpoints/next-e-4b-s0 checkpoints/next-e-4b-s1 \
  --rows data/unseen_v2.jsonl --split dev --json reports/interp-4b.json
strands-decider soup checkpoints/next-e-4b-s0 checkpoints/next-e-4b-s1 --out checkpoints/next-e-4b   # only if "soup": true
python evaluation/unseen_v2/calibrate_mixed.py checkpoints/next-e-4b checkpoints/next-e-4b-cal \
  --unseen data/unseen_v2.jsonl --in-family data/holdout_v5_norule.jsonl data/multistep_v14_eval.jsonl \
  data/adequacy_hs2_eval.jsonl data/synthetic/adequacy_gen_eval.jsonl data/synthetic/generated_v16_eval.jsonl
python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint checkpoints/next-e-4b-cal --out reports/unseen_v2/next-e-4b
python evaluation/unseen_v2/score.py reports/unseen_v2/next-e-4b --vs reports/unseen_v2/BEST_2B --split test   # read once
```

## What Stage A did not do

- Nothing ran on a GPU: no model has been read on unseen-v2, the kernel installer has only
  been syntax-checked, and the Gemma 4 teacher path is tested on a tiny random Gemma 4, not
  on the 31B. The 4B step rate stays an estimate (0.22–0.31 step/s) until the Stage B load test.
- Stage C adapters exist for three of the ten systems. The others' inference code was not
  read, so no adapter was guessed for them; `rank_check.py` lists them as not run.
- The original plan's `family_temperature` sampler and the compact-JSON option in
  `prompting.render_content` were not built: format variety lives in the A1 generators'
  states instead, which needs no train/serve change.
- Common random numbers hold exactly within each A1 level: A2 and A3 cells see the same rows
  in the same order as their A1 counterpart, but adding the A1 file changes the shuffle, so
  the A1 contrast also carries a data-order difference (inside the seed noise it is read against).
- python-chess (GPL-3.0) is a build-time tool in the `unseen` extra for the chess family; the
  package never imports it and nothing of it is redistributed.

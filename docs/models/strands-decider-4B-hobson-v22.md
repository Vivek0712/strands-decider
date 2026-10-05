# strands-decider-4B-hobson-v22

> **Unofficial candidate.** One 4B checkpoint for text and images: Qwen3.5-4B trained in
> three short stages on the hobson-v20 recipe, with four new generated task families, then
> on questions over images with Qwen3.5's own vision tower. Estimated **#1 overall on the
> JevBench text leaderboard** (composite ~74 vs 73.70 for Cygnet, #1 today, 73.23 for
> Winnow-12B, #2, and 72.13 for Jev 1.13.0, #3; inside the tie band, so an official run
> decides) and **#1 in the 4B class** (Capability ~75 vs 72.3 for jevk5-v0.3-4b, the best 4B
> today). On the image board, Capability ~74-82, roughly #3 to #15 of 38, most likely just
> below Imajev-4B (82.1). Not submitted; every rank is an estimate from local measures.
>
> **Community release:** built on Strands Decider (v19's code and recipe), proposed to the
> Strands team; not an official Strands release. The v22 suffix continues this series'
> numbering (hobson-v20, minicpm-v21) and is not an official Strands version.

| | |
| --- | --- |
| What it is | Qwen3.5-4B (post-trained) trained from raw for 1,500 steps on the v20 recipe (the pilot), continued one full epoch on the v20 recipe plus the A1 task families (two seeds, souped after an interpolation check), calibrated, then continued one epoch on the image recipe of [hobson-v20-balanced](strands-decider-2B-hobson-v20-balanced.md) |
| Base | `Qwen/Qwen3.5-4B` (Apache-2.0; its own vision tower, frozen), trained at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Size | 4B class: Qwen3.5-4B's decoder and its vision tower (frozen), a LoRA adapter and a pointer head; the archive holds the adapter, the head and the training configs (285 MB) |
| Answers | text, and text with images (`serve --vision`) |
| Configs | [strands-decider-4B-hobson-v22-pilot.yaml](../../configs/experiments/strands-decider-4B-hobson-v22-pilot.yaml), [strands-decider-4B-hobson-v22.yaml](../../configs/experiments/strands-decider-4B-hobson-v22.yaml) and `-seed1` (text), [configs/vision/strands-decider-4B-hobson-v22-vl.yaml](../../configs/vision/strands-decider-4B-hobson-v22-vl.yaml): **the package is seed 0 of the image stage** |
| Download | https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/strands-decider-4B-hobson-v22.tar |
| sha256 | `7305d520e83e3dd0c112e22da2ded26713ed6f911e2ff071d82e484f8d55c029` ([weights_SHA256SUMS.txt](https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/weights_SHA256SUMS.txt)) |

## Results

Local measures, never board scores ([docs/evaluating.md](../evaluating.md)). Unseen-v2 is
the 9,000-row unseen-family set of [evaluation/unseen_v2/](../../evaluation/unseen_v2/README.md)
(test split, 5,400 rows); it is harder than the 1,050-row set, so its numbers are lower.
Images at the 400,000-pixel budget, without image temperatures. Every row was measured on
one host with the same code, except the published balanced model's image numbers, which are
its recorded run.

| Measure | v22 (seed 0) | image seed 1 | v22 text soup, before images | earlier 4B (1,500-step stages) | hobson-v20-balanced (2B) |
| --- | --- | --- | --- | --- | --- |
| JevBench v1 public, v1.5-style Intelligence | **64.6** | 61.1 | 62.4 | 56.1 | 39.4 |
| Tasks right, of 231 | 188 | 185 | **191** | 190 | 175 |
| Yes/no answers inside 0.2-0.8, of 74 | **15** | 17 | 19 | 24 | 36 |
| Yes/no competence, public | **+37.8** | +29.7 | +35.1 | +16.2 | -10.8 |
| ECE, public tasks | 0.060 | 0.055 | **0.029** | 0.048 | 0.041 |
| Unseen-v2 test, Intelligence | **24.7** | ~25.0 | 22.6 | 15.3 | 2.1 |
| NaturalBench accuracy / G-Acc | 0.818 / 0.400 | **0.828 / 0.447** | -- | 0.816 / 0.397 | 0.803 / 0.360 |
| POPE-adversarial accuracy | 0.875 | 0.877 | -- | **0.883** | 0.865 |
| Image JevBench preview, right of 60 | 48 | **53** | -- | 47 | 43 |
| Mean confidence with the image removed, NaturalBench / POPE (lower is better) | 0.60 / **0.55** | 0.59 / 0.53 | -- | -- | 0.61 / 0.59 |

Paired comparisons (95% bootstrap intervals):

- Unseen-v2 test, v22 against the earlier 4B: Intelligence **+9.36 (+7.92, +10.83)**, yes/no
  competence +25.2 (+21.8, +28.6), choice +1.3 (-1.2, +3.9), score +1.5 (+0.6, +2.4). The
  text soup alone: +7.28 (+5.85, +8.66).
- Public, the text soup against the earlier 4B: +6.3 (-0.1, +13.1).
- Images, both seeds against hobson-v20-balanced: NaturalBench +0.020 (+0.003, +0.037), G-Acc
  **+0.063 (+0.017, +0.113)**, POPE +0.011 (-0.006, +0.028), preview **+7.5 (+1.0, +14.0)**,
  confidence with the image removed -0.007 (NaturalBench) and **-0.052 (POPE)**. Against the
  earlier 4B (three seeds), every image measure is within noise.

**Estimated board position (unofficial).** Open half = the public score (64.6); sealed half
~61 (52.5-68.6), from the unseen-v2 score mapped to the 1,050-row set's scale plus the
open-to-sealed offset measured on decider-2b and decision-2b; Intelligence ~63 (58.6-66.6);
Calibration assumed at the ranked Qwen3.5-4B deciders' ~87 (78-90); Capability ~75 (68-78).
Text leaderboard: composite ~74 (at decider-4b-v2's measured Speed and Cost) against 73.70
for Cygnet (#1 today), 73.23 for Winnow-12B (#2) and 72.13 for Jev 1.13.0 (#3): **about #1
overall, inside the tie band** (pessimistic ~#5). 4B class: Capability ~75 against 72.3 for
jevk5-v0.3-4b, 71.6 for plumb-4b and 71.2 for decision-4b-v12: **about #1**. Image board:
**Capability ~74-82, roughly #3 to #15 of 38**, most likely just below Imajev-4B (82.1); one
anchor and 60 preview items cannot place it more precisely. The sealed half and the board's
own Calibration decide it, and only an official run measures them.

## Download and serve

```bash
NAME=strands-decider-4B-hobson-v22
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints && tar -xf "$NAME.tar" -C checkpoints
(cd checkpoints/$NAME && sha256sum -c SHA256SUMS)     # every file in the archive
pip install "strands-decider[vision]"                  # Pillow; transformers >= 5.18
strands-decider serve checkpoints/$NAME --vision --image-long-side 0 --image-max-pixels 400000 --port 8000
```

`serve` downloads Qwen3.5-4B at the pinned revision (about 9 GB). It was trained at a budget
of 400,000 pixels per image, which `--image-long-side 0 --image-max-pixels 400000`
reproduces. Text requests take the text path. It also serves on an Apple-silicon Mac
(`--device mps`); the archive's `training_configs/` holds the recorded configs of every stage,
the soup record and the interpolation curve.

## Verify it yourself

On one NVIDIA GPU, with the archive unpacked as above and v19 at its pinned revision
([docs/evaluating.md](../evaluating.md#setup)):

```bash
# Text: JevBench v1 public through the vision server, against v19
PY=$(which python) GPU=0 SERVE_ARGS=--vision evaluation/jevbench/jevbench.sh checkpoints/$NAME reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19

# Unseen-v2 (9,000 rows; read the test split), answered as `serve --vision` answers, against v19
pip install "strands-decider[unseen]"     # datasets and python-chess (build time only)
training/recipe.sh build
python evaluation/unseen_v2/build.py --out data/unseen_v2.jsonl
python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint checkpoints/$NAME --vision \
  --out reports/unseen_v2/$NAME
python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint checkpoints/v19 --out reports/unseen_v2/v19
python evaluation/unseen_v2/score.py reports/unseen_v2/$NAME --split test --vs reports/unseen_v2/v19

# Images: NaturalBench, POPE (and, with IJB_JSONL, the 60 preview items) at 400,000 pixels,
# with and without the image; then against v19 + --vision
pip install pandas pyarrow
CKPT=checkpoints/$NAME OUT=reports/$NAME training/recipe_images.sh eval
CKPT=checkpoints/v19 OUT=reports/v19-400k training/recipe_images.sh eval
python evaluation/vision/compare.py reports/$NAME/eval --vs reports/v19-400k/eval
```

`build.py` writes a manifest beside the rows; seed 0 with the pinned inputs must reproduce
its recorded sha256 ([evaluation/unseen_v2/README.md](../../evaluation/unseen_v2/README.md)).
Served from this branch, the archive reproduces the recorded per-item answers of its
JevBench run and of the preview image items (see the pull request for the check).

## Retrain

Linux with NVIDIA GPUs; the recorded runs used two H200s (one seed each), Python 3.11,
torch 2.14.1 and transformers 5.18.0, with flash-linear-attention and causal-conv1d. First
build the v20 data and its 27B teacher file
([hobson-v20](strands-decider-2B-hobson-v20.md#retrain): `training/recipe.sh build fetch
multistep generated adequacy teacher_yn`) and the image rows of
[hobson-v20-balanced](strands-decider-2B-hobson-v20-balanced.md#retrain)
(`training/recipe_images.sh fetch build dedupe`). Then:

```bash
pip install -e ".[train,vision,cuda,unseen]"
cal() {  # the usual step, then the choice/score refit, as for every candidate
  strands-decider calibrate "$1" --data data/holdout_v5_norule.jsonl
  strands-decider calibrate "$1" --kinds choice,score --objective nll --limit 1800 \
    --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
}
# 1. The pilot: 1,500 steps from the raw Qwen3.5-4B
strands-decider train --config configs/experiments/strands-decider-4B-hobson-v22-pilot.yaml
cal checkpoints/strands-decider-4B-hobson-v22-pilot

# 2. Text: one full epoch of the v20 mixture plus the A1 families, two seeds, then the soup
python -m strands_decider.data.families --out data/families_a1.jsonl --eval-out data/families_a1_eval.jsonl
for s in "" -seed1; do
  strands-decider train --config configs/experiments/strands-decider-4B-hobson-v22$s.yaml
done
python evaluation/unseen_v2/build.py --out data/unseen_v2.jsonl
python evaluation/unseen_v2/interpolate.py checkpoints/strands-decider-4B-hobson-v22-text-s0 \
  checkpoints/strands-decider-4B-hobson-v22-text-s1 --rows data/unseen_v2.jsonl --split dev \
  --json reports/interp-v22.json       # "soup": true  (barrier -0.004)
strands-decider soup checkpoints/strands-decider-4B-hobson-v22-text-s{0,1} \
  --out checkpoints/strands-decider-4B-hobson-v22-text
cal checkpoints/strands-decider-4B-hobson-v22-text

# 3. Images: the hobson-v20-vl recipe on the text soup; seed 0 is the package
python -m strands_decider.data.teacher_yn replay --teacher data/teacher_yn_qwen35-27b.jsonl \
  --n-yes-no 2000 --n-other 1000 --out data/replay_teacher_yn.jsonl
CONFIG=configs/vision/strands-decider-4B-hobson-v22-vl.yaml training/recipe_images.sh train eval
```

- The pilot is the base bake-off recipe on the post-trained Qwen3.5-4B (the -Base model
  scored 51.0 public and 12.2 unseen-v2 dev against 50.4 and 14.8; the post-trained one
  generalised better and was kept). Gate, fixed before the run: at least +5 public over the
  same 1,500 steps at 2B (34.2) and not worse on unseen-v2 dev.
- The text stage's 142,903 rows give 4,465 steps (`max_steps: 0`, one epoch). The two seeds
  share an initialisation (`init_seed`); `interpolate.py` scores weighted soups along the
  line between them and soups only if no point rises above the straight line (the recorded
  barrier: -0.004 NLL on 3,000 dev rows). The soup's LoRA rank is 32 (16 + 16), exactly the
  mean update.
- The image stage keeps the text soup's temperatures. `recipe_images.sh eval` also writes
  image temperatures fitted on held-out NaturalBench groups (recorded: yes/no 1.80, choice
  1.10); the package does not apply them.
- The calibration refit draws a sample (`--limit` caps each file's half), so a rebuild gives
  similar, not identical, temperatures; the replay draw and the 27B labels are not
  committed, as for hobson-v20-balanced.

What this adds, all opt-in:

| Piece | What it does |
| --- | --- |
| `evaluation/unseen_v2/` | 9,000 questions from 11 families no model trains on, four of them generated with exact gold; dev/test splits, proper scores and item-paired bootstrap comparisons |
| `python -m strands_decider.data.families` | The A1 training families (state tracking, tool-call guardrails, lead-qualification rubrics, JSON policy compliance), every label computed by the generator, tested disjoint from unseen-v2 |
| `soup(coefs=...)`, `evaluation/unseen_v2/interpolate.py` | Weighted soups and the interpolation-barrier check that decides whether two seeds are souped |
| The `unseen` extra | `datasets` and python-chess (GPL-3.0) for `unseen_v2/build.py` only; the package never imports python-chess |

## Lineage

1. **Pilot** ([config](../../configs/experiments/strands-decider-4B-hobson-v22-pilot.yaml)):
   Qwen3.5-4B, 1,500 steps from raw on the v20 recipe: 50.4 public, 186/231, unseen-v2 dev
   +14.2 (+12.1, +16.3) over the same recipe at 2B.
2. **Recipe screen at 2B** (eight cells continuing hobson-v20 for 1,500 steps): the A1
   families raised unseen-v2 dev Intelligence by +3.83 (+2.69, +4.96) and were adopted; an
   RPS loss for score rows had no effect, and a pooled 27B teacher on every row hurt yes/no
   competence (-13.0). Neither is in this recipe or this code.
3. **Text stage** (two seeds, souped): 62.4 public, 191/231 right, ECE 0.029.
4. **Image stage** (this package, seed 0 of 2): 64.6 public, images as above.

## Limits

- Exploratory: local runs, two seeds for the last two stages; the decision rules were written
  before each stage, the arms after earlier results.
- The v1.5 proxy is our reimplementation of the published rules; the sealed half and the
  board's Calibration are estimated, so every rank is an estimate. Unseen-v2 is our own set;
  its rank agreement with the board was not checked.
- The image numbers are public sets; the 60 preview items are rebuilt from their sources,
  not the sealed Image JevBench set.
- 4B parameters: slower and larger than the 2B candidates (JevBench p50 latency 0.11 s on an
  H200).

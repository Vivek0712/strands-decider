# Better accuracy for v19: decisive yes/no answers, and image fine-tuning

Two exploratory experiments that start from the published v19 (2B) checkpoint and change no
published weights or defaults: one on text, aimed at v19's indecisive yes/no answers, and
one on images, building on `--vision` ([vision.md](vision.md)). Everything was run locally,
on public data, with three seeds per arm. Neither is the preregistered, confirmatory run
that a new published checkpoint needs; that is proposed separately (#11). This page gives
the results, then the commands to reproduce every number.

Every per-item result, summary, training log and data hash behind this page is in the
release [better-accuracy-results-2026-10-03](https://github.com/Vivek0712/strands-decider/releases/tag/better-accuracy-results-2026-10-03)
(no checkpoints). Its `comparisons/` folder holds the outputs of the comparison commands below.

Paths are relative to the repository root unless they are links.

## Results

### Text: fewer yes/no answers in JevBench v1.5's abstention band

JevBench v1.5 counts a yes/no answer with 0.2 < P(yes) < 0.8 as wrong. v19 answers 48 of the
74 yes/no tasks of the v1 public set inside that band. The text arm continues v19 for 1,500
steps (~0.4 epoch) on its own training data with two changes
([configs/experiments/v19-yn27b.yaml](../configs/experiments/v19-yn27b.yaml)): the
Qwen3.5-27B teacher's distributions on the 54,857 yes/no rows, kept on the 45,337 where it
agrees with gold (`training/recipe.sh teacher_yn`), and no frozen-KL anchor on yes/no rows
(`kl_frozen_skip_kinds: ["noul"]`).

The JevBench numbers below come from the official harness on the 231 public v1 tasks
(`evaluation/jevbench/jevbench.sh`). The "proxy Intelligence" is
`evaluation/jevbench/v15_proxy.py`: a **local proxy** for v1.5 scoring, not the official
score (v1.5's tier weights and sealed half are not public). It is for comparing runs here,
never with a board.

| Run | Tasks right (of 231) | Proxy Intelligence | Yes/no in the band (of 74) | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| v19, published | 167 | 32.0 | 48 | 0.348 | 0.050 |
| v19, reproduced here | 167 | 32.1 | 48 | 0.348 | 0.050 |
| v19-yn27b, seeds 0 / 1 / 2 | 169 / 171 / 171 | 40.6 / 37.4 / 42.3 | 35 / 44 / 31 | 0.349 / 0.332 / 0.363 | 0.044 / 0.062 / 0.052 |
| v19-yn27b, mean of 3 | 170.3 | 40.1 | 36.7 | 0.348 | 0.053 |

The reproduced v19 matches the published run task for task (largest probability difference
0.0); the proxy differs by 0.1 between the two only because one task's result lists its
options differently, and the script reads the question type from the result.

Against v19, task by task (mean over the three seeds minus v19, paired bootstrap 95%
interval over the 231 tasks, 10,000 resamples):

- yes/no answers in the band: -11.3 (-17.7 to -5.3), fewer on every seed (-13, -4, -17);
- proxy Intelligence: +8.0 (+2.9 to +13.3), higher on every seed (+8.5, +5.2, +10.2);
- tasks right: +3.3 (-3.3 to +10.3): within noise.

So the change is in decisiveness, not accuracy: the same tasks are answered right about as
often, with fewer yes/no answers left near 0.5. About half of the yes/no answers are still
inside the band. Brier and ECE stay in v19's range.

The other arms, all continued from v19 for the same 1,500 steps, one seed unless stated
(proxy Intelligence; yes/no in the band; tasks right):

| Arm | Change | Proxy Intelligence | In the band | Tasks right |
| --- | --- | --- | --- | --- |
| control | more training, nothing else | 34.6 | 44 | 172 |
| anchor off | `kl_frozen_skip_kinds: ["noul"]` alone | 34.5 | 41 | 169 |
| 27B teacher, seeds 0 / 1 / 2 | the teacher file alone | 39.4 / 31.8 / 32.1 | 37 / 45 / 46 | 171 / 170 / 169 |
| both (v19-yn27b), seeds 0 / 1 / 2 | the two together | 40.6 / 37.4 / 42.3 | 35 / 44 / 31 | 169 / 171 / 171 |

The teacher alone is not reliably better than v19 (its 3-seed proxy Intelligence +2.3,
-1.7 to +6.6, within noise); its seed 0 was a lucky draw, which is why every arm that
mattered was run three times. Both changes together beat v19 on every seed.

### Images: the pixel budget, image temperatures, and training

All image numbers are from `evaluation/vision/run.py`: NaturalBench (first shard, groups
0-299: 1,200 questions), POPE adversarial (the first 600 by question id) and the 60 Image
JevBench preview items that can be rebuilt exactly from their sources, each also asked with
the image removed. Image temperatures are fitted only on NaturalBench groups 300-599.

**No new weights.** v19 at a 400,000-pixel budget per image (aspect kept, never upscaled,
about 390 tokens) instead of the 448 px long side, with image temperatures fitted
(noul 0.759, choice 0.790):

| v19 setting | NaturalBench acc | G-Acc | POPE acc | POPE ECE | Preview, exact |
| --- | --- | --- | --- | --- | --- |
| 448 px long side | 0.784 | 0.327 | 0.877 | 0.072 | 37/60 |
| 400,000-pixel budget | 0.788 | 0.347 | 0.858 | 0.082 | 40/60 |
| 400,000 pixels + image temperatures | 0.788 | 0.347 | 0.858 | 0.057 | 40/60 |

Paired against 448 px: preview +3 items (0 to +7), NaturalBench G-Acc +0.020 (-0.030 to
+0.067) and accuracy +0.003, all within noise; POPE -0.018 (-0.037 to 0.000): its
640 x 480 photos are not scaled down at 400,000 pixels, and v19 does slightly worse on them
at full size. Temperatures change calibration only, never an answer. The training runs use the 400,000-pixel budget.

**Training.** v19 continued for one epoch (~1,404 steps) on ~38,000 questions over images
([data/image/README.md](../data/image/README.md)) plus 3,000 of its own text rows. Variant A
([configs/vision/v19-images.yaml](../configs/vision/v19-images.yaml)) adds an image-removed
copy of 15% of the image rows, trained only toward the frozen torso's reading of the
text-only prompt; variant B is the same without those copies (`ablation_fraction: 0`).
v19 was re-measured in the same environment as the training runs: the same answer on every
one of the 1,928 items as the first measurement, probabilities within 0.003. Means over
three seeds, without image temperatures:

| Run | NaturalBench acc | G-Acc | POPE acc | POPE ECE | Preview, exact | Image removed: confidence NaturalBench / POPE | Image removed: ECE NaturalBench / POPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v19, 400,000 pixels | 0.787 | 0.347 | 0.858 | 0.082 | 40/60 | 0.652 / 0.725 | 0.152 / 0.225 |
| A, seeds 0 / 1 / 2 | 0.798 / 0.811 / 0.806 | 0.350 / 0.390 / 0.373 | 0.873 / 0.872 / 0.878 | 0.031 / 0.034 / 0.032 | 42 / 44 / 42 | 0.584 / 0.575 / 0.586 ; 0.533 / 0.524 / 0.522 | 0.084 / 0.075 / 0.086 ; 0.033 / 0.027 / 0.037 |
| A, mean of 3 | 0.805 | 0.371 | 0.874 | 0.032 | 42.7/60 | 0.582 / 0.526 | 0.082 / 0.032 |
| B, mean of 3 | 0.802 | 0.363 | 0.881 | 0.029 | 43.0/60 | 0.615 / 0.664 | 0.115 / 0.164 |

The clearest difference is with the image removed: v19 still answers at a mean confidence
of 0.65 (NaturalBench) and 0.73 (POPE) where the answer cannot be known; A falls to 0.58
and 0.53, near chance for these two-way questions (A minus v19: -0.071, -0.076 to -0.065,
and -0.198, -0.200 to -0.196). B, trained on the same images without the copies, falls far
less (0.62 and 0.66; A minus B -0.034 and -0.138, both outside noise).

On accuracy, A against v19 (mean of three seeds minus v19, paired bootstrap 95% interval):
NaturalBench +0.018 (+0.004 to +0.031) and POPE +0.016 (+0.003 to +0.031) are real but
small; G-Acc +0.024 (-0.012 to +0.063) and the preview's +2.7 items (-2.3 to +7.7) are within
noise. A and B are tied on accuracy (A minus B: NaturalBench +0.003, POPE -0.007, preview
-0.3, all within noise), so A is the variant to take forward, for the image-removed
behaviour. Fitted image temperatures lower NaturalBench ECE from 0.044-0.051 to 0.017-0.028
for every trained run.

Text is not hurt: on 899 held-out text rows of v19's own evaluation files
(`evaluation/vision/text_check.py`) v19 scores 0.817 (ECE 0.026), A 0.829 / 0.820 / 0.829
(ECE 0.029 / 0.024 / 0.011).

Against the v19-vision preregistration's bars (#11: G-Acc >= 0.383, accuracy >= 0.802,
image-removed confidence <= 0.58 and ECE <= 0.10, POPE >= 0.862, preview >= 35/60), A's mean
meets all but G-Acc (0.371; seed 1 reached 0.390) and sits at the confidence line on
NaturalBench (0.582). These runs are exploratory; they do not replace the preregistered run.

## Reproduce

Linux with NVIDIA GPUs: the text arm and the 27B labelling ran on H100 80 GB cards, one GPU
per run. Install as for training ([training/README.md](../training/README.md#setup)), plus
the extras each part names. Every model and dataset download is pinned to the revision the
results were measured on.

```bash
pip install -e ".[train,vision,cuda]"
hf download StrandsAgents/strands-decider-2B-hobson-v19 \
  --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd --local-dir checkpoints/v19
```

### v19 on JevBench, and the proxy

```bash
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/v19
```

Expected: 167 of 231 right (`summary.json`), `"intelligence_proxy": 32.1`, `"yes_no": 74`,
`"yes_no_in_band": 48`, competence by type choice 65.4, yes/no -32.4, score 63.4.

### The text arm (v19-yn27b)

```bash
training/recipe.sh build fetch multistep generated adequacy teacher_yn
for s in "" -seed1 -seed2; do
  TRAIN_CONFIG=configs/experiments/v19-yn27b$s.yaml CKPT=checkpoints/v19-yn27b$s training/recipe.sh train
  strands-decider calibrate checkpoints/v19-yn27b$s --data data/holdout_v5_norule.jsonl
  PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19-yn27b$s reports/jevbench/v19-yn27b$s
done
python evaluation/jevbench/v15_proxy.py reports/jevbench/v19-yn27b{,-seed1,-seed2} --vs reports/jevbench/v19
```

- `build` and `fetch` rebuild v19's corpus: every recorded training file matched `data/SHA256SUMS`.
  `teacher_yn` labels with Qwen3.5-27B (about 31 GPU-minutes) and prints, per file, how
  often the teacher agrees with gold: train_v5 0.831, adequacy_hs2 0.745, adequacy_gen
  0.919, generated_v16 0.865, generated_v18 0.879; `data/teacher_yn_qwen35-27b.jsonl` has
  58,246 rows. The recorded file's sha256 is in [data/README.md](../data/README.md); a
  relabel need not match it byte for byte.
- Each `train` takes about an hour on one H100 (`max_steps: 1500`).
- `calibrate` runs directly rather than through `recipe.sh calibrate`:
  `data/holdout_v5_norule.jsonl` rebuilds today with a different hash from the one in
  `data/SHA256SUMS` (an upstream held-out dataset has changed), so the recipe's check
  stops there. The recorded runs were calibrated on the rebuilt file.
- Expected: the table above, per seed and as the means, and from `--vs` the three
  intervals above (`intelligence_proxy` +7.96, `yes_no_in_band` -11.33, `n_correct` +3.33).
  Retraining reproduces the run, not every task: expect a few tasks' difference per seed.

### The image arm (v19-images, variant A)

```bash
pip install matplotlib imagehash pandas pyarrow   # the builders, dedupe and run.py
training/recipe_images.sh fetch build dedupe
for s in "" -seed1 -seed2; do CONFIG=configs/vision/v19-images$s.yaml training/recipe_images.sh train eval; done
CKPT=StrandsAgents/strands-decider-2B-hobson-v19 OUT=reports/v19-400k training/recipe_images.sh eval
python evaluation/vision/compare.py reports/v19-images{,-seed1,-seed2}/eval --vs reports/v19-400k/eval
```

- `build` prints each row file's sha256; the recorded ones are in
  [data/image/README.md](../data/image/README.md#hashes-of-record), with what may differ.
  `dedupe` reports 0 training images within Hamming distance 6 of an evaluation image.
- `train` takes about 46 minutes on one H100; `eval` about 10 more. Each run's `eval/`
  holds the per-item results and `summary.json` (with the revisions and library versions),
  `eval-T/` the same under the fitted image temperatures, and `text_check.json` the text check.
- The preview items: set `IJB_JSONL` to the file built by `ijb_preview.py` in
  [github.com/Vivek0712/vision-decider](https://github.com/Vivek0712/vision-decider/blob/phase1-baselines/phase1/ijb_preview.py)
  for `dedupe` and `eval`; without it the preview columns are left out.
- Expected: the image table above, and from `compare.py` the intervals above
  (`naturalbench_acc` +0.0175, `pope_acc` +0.0161, `blind_conf_naturalbench` -0.0706,
  `blind_conf_pope` -0.1982). The training images now decode through the server's
  decoder (EXIF rotation applied), which the recorded runs did not; a COCO photo stored
  rotated trains upright.

### The same image arm on a Gemma 4 E2B checkpoint (not run)

[configs/vision/gemma4-e2b-images.yaml](../configs/vision/gemma4-e2b-images.yaml) (and `-seed1`,
`-seed2`) is variant A unchanged on the same rows, starting from a TEXT Strands Decider on
`google/gemma-4-E2B` (by default the bake-off's `checkpoints/bakeoff-gemma4-e2b`; `INIT_FROM`
names another). Nothing about it has been measured yet.

```bash
for s in "" -seed1 -seed2; do
  INIT_FROM=/path/to/gemma-text-ckpt CONFIG=configs/vision/gemma4-e2b-images$s.yaml \
    training/recipe_images.sh train eval
done
CKPT=/path/to/gemma-text-ckpt OUT=reports/gemma-text-400k training/recipe_images.sh eval   # step 0
python evaluation/vision/compare.py reports/gemma4-e2b-images{,-seed1,-seed2}/eval --vs reports/gemma-text-400k/eval
```

Gemma's processor sizes every image to its own 280-soft-token budget, so the 400,000-pixel
budget changes the detail that reaches it, not the token count ([vision.md](vision.md#gemma-4-e2b-checkpoints)).

The no-training rows: `evaluation/vision/run.py --systems strands --long-side 0 --max-pixels
400000` (and the default 448 px) on the evaluation set, the same with `--nb-start 300
--nb-groups 300 --pope 0 --no-blind` for the held-out groups, then `evaluation/vision/temps.py
fit` and `apply` ([evaluation/vision/README.md](../evaluation/vision/README.md)). Serving a
checkpoint at the budget is `load_vision_engine(ckpt, image_long_side=0,
image_max_pixels=400_000)`; `serve --vision` keeps the 448 px long side.

## Limits

- Exploratory: three seeds, local runs on public data, chosen after looking at the
  results of earlier arms. The confirmatory run is the preregistration's to make.
- The proxy Intelligence is a local proxy, not JevBench v1.5's score, and the 74 yes/no
  tasks are few: the per-seed differences in the band count run from -4 to -17.
- Tasks right on JevBench did not move beyond noise; the text change is in decisiveness.
- Image A misses the preregistered G-Acc bar (0.371 against 0.383), and its gains in G-Acc
  and on the 60 preview items are within noise.
- The 60 preview items are rebuilt from their source datasets, not the sealed Image JevBench.

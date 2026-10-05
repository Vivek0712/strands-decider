# strands-decider-4B-hobson-v22: every stage, every result (2026-10-04/05)

Research log (exploratory), published as the results record of the `strands-decider-4B-hobson-v22`
candidate. Nothing here is an official score. **Public** = v1.5-style Intelligence on the 231 public
JevBench v1 tasks (local proxy, the rules of [RESULTS.md](RESULTS.md) §0; every arm calibrated the
same way: the usual holdout step, then the choice/score refit). **Unseen-v2** = 9,000 rows from 11
task families no model trained on (dev 40% / test 60%; harder than the 1,050-row unseen set of
RESULTS.md, so its numbers are lower). **Images** = NaturalBench (1,200 questions / 300 groups),
POPE-adversarial (600) and the 60 rebuildable Image JevBench preview items, at the 400,000-pixel
budget, run with the same code as the published 2B models' records.

Code: branch `pr/7-hobson-v22` of this fork (stacked on the 2B series). All runs on rented H200
GPUs (torch 2.14.1, transformers 5.18.0).

## 1. Lineage

| Stage | What | Result |
|---|---|---|
| Pilot | Qwen3.5-4B (post-trained, `851bf6e8`), 1,500 steps from raw on the v20 recipe | 50.4 public, gate passed |
| Recipe screen (2B) | 2³ factorial continuing hobson-v20: A1 new task families, A2 RPS loss, A3 pooled teacher | A1 adopted; A2, A3 rejected |
| Text stage | the pilot continued one full epoch (4,465 steps, 142,903 rows = v20 mixture + A1), 2 seeds, souped | 62.4 public, 191/231 |
| Image stage | the hobson-v20-balanced image recipe on the text soup, 2 seeds; **seed 0 = the package** | 64.6 public, images below |

## 2. Pilot: does a 4B torso earn its cost? (1,500 steps from raw, v20 recipe)

| Arm | Public I | Right | In band | Unseen-v2 dev I | vs 2B on unseen dev (95% CI) |
|---|---|---|---|---|---|
| Qwen3.5-4B-Base | 51.0 | 187 | 32 | 12.2 | +11.6 (+9.6, +13.7) |
| **Qwen3.5-4B (post-trained)** | 50.4 | 186 | 32 | 14.8 | +14.2 (+12.1, +16.3) |
| Qwen3.5-2B-Base (reference) | 34.2 | 164 | 42 | 0.6 | — |

Gate (fixed before): the better 4B arm >= +5 public over the 2B and not worse on unseen dev -> +16.8,
pass. The rule picks the higher public arm (Base, +0.6, within noise); the post-trained arm generalised
better on unseen dev (+2.5) and is the one continued.

## 3. Recipe screen at 2B (8 cells, continuing hobson-v20, 1,500 steps)

| Arm | Main effect on its pre-registered metric (unseen-v2 dev) | Guard | Decision |
|---|---|---|---|
| A1 new task families | Intelligence +3.83 (+2.69, +4.96) | no drops | **adopt** |
| A2 RPS + NLL for score | score competence +0.06 (−0.09, +0.20) | no drops | reject (no effect) |
| A3 pooled 27B teacher on every row | yes/no competence **−13.0** (−15.6, −10.6) | unseen −4.0, public −1.9 | reject (harmful) |

Cells (public I / unseen-v2 test I): 000 37.1/4.6 · 001 34.3/0.2 · 010 42.0/6.7 · 011 35.9/0.8 ·
**100 42.1/11.1** · 101 40.6/4.7 · 110 38.0/6.7 · 111 40.7/7.8. A3 is a clear negative result: mixing
the teacher with gold on every row made yes/no hedge more; v20's agree-with-gold teacher stays.

## 4. The earlier 4B (1,500-step stages) — the baseline v22 is compared against

The Base pilot (the arm the gate's rule picked) continued 1,500 steps on the v20 recipe without A1
(2 seeds, souped: 54.8 public, 191/231), then the image recipe (3 seeds): seed 0 56.1 public, 15.3
unseen-v2 test, NaturalBench 0.816 / G-Acc 0.397, POPE 0.883, preview 47/60. Re-run on the v22 host,
its public score reproduced (56.1).

## 5. v22: the text stage and the image stage

Interpolation check between the two text seeds (3,000 unseen-v2 dev rows, temperature 1): barrier
**−0.004 NLL** (no barrier) -> soup, then calibrated. Image stage: 2 seeds; the candidate was chosen by
the rule fixed before the run: highest public I among image seeds whose images are not worse than the
published balanced model -> **seed 0**.

| Model | Public I | Right | Yes/no in band | Yes/no comp. | ECE | Unseen-v2 test I | NB acc | NB G-acc | POPE | Preview /60 |
|---|---|---|---|---|---|---|---|---|---|---|
| **v22, image seed 0 (the package)** | **64.6** | 188 | **15** | **+37.8** | 0.060 | **24.7** | 0.818 | 0.400 | 0.875 | 48 |
| v22, image seed 1 | 61.1 | 185 | 17 | +29.7 | 0.055 | ~25.0 | **0.828** | **0.447** | 0.877 | **53** |
| v22 text soup, before images | 62.4 | **191** | 19 | +35.1 | **0.029** | 22.6 | — | — | — | — |
| earlier 4B, image seed 0 (re-run here) | 56.1 | 190 | 24 | +16.2 | 0.048 | 15.3 | 0.816 | 0.397 | 0.883 | 47 |
| published hobson-v20-balanced (2B) | 39.4 | 175 | 36 | −10.8 | — | 2.1 | 0.803 | 0.360 | 0.865 | 43 |

Paired (95% CI):

- v22 (seed 0) vs the earlier 4B on unseen-v2 test: Intelligence **+9.36 (+7.92, +10.83)**, yes/no
  **+25.2 (+21.8, +28.6)**, choice +1.3 (−1.2, +3.9), score +1.5 (+0.6, +2.4). The text soup alone:
  +7.28 (+5.85, +8.66).
- Text soup vs the earlier 4B, public: +6.3 (−0.1, +13.1).
- Images, 2 seeds vs the published balanced model: NB acc **+0.020 (+0.003, +0.037)**, NB G-acc
  **+0.063 (+0.017, +0.113)**, POPE +0.011 (−0.006, +0.028), preview **+7.5 (+1.0, +14.0)**,
  confidence with the image removed −0.007 (NB) / **−0.052 (POPE)**. Against the earlier 4B (3 seeds):
  all within noise.

Unseen-v2 test by family (v22 seed 0 / earlier 4B, competence): arithmetic 15.4 / 2.8, calendar
18.2 / 8.1, chess −0.6 / −16.6, seating 18.0 / 14.3, CommonsenseQA 2.0 −10.0 / −40.8, StrategyQA
−30.0 / −57.5, RuleTaker d5 34.2 / 31.7, CommonsenseQA 70.8 / 68.2, HellaSwag 77.2 / 77.8,
ARC-Challenge 88.3 / 89.4, STS-B 59.6 / 58.1. The gain is in yes/no decisiveness and the generated
families.

Package: image temperatures were fitted on held-out NaturalBench groups (yes/no 1.80, choice 1.10) but
are **not applied**, as in the recorded numbers above; text temperatures are the text soup's.

## 6. Estimated standing (unofficial; an official run decides)

| Input | Value |
|---|---|
| Open half (public, 231 tasks) | 64.6 |
| Unseen-v2 test Intelligence | 24.7 |
| Old-unseen equivalent | ~50 (45–54): models measured on both sets score 20–29 higher on the 1,050-row set |
| Sealed half | ~61 (52.5–68.6): old-unseen equivalent + the open-to-sealed offset measured on decider-2b and decision-2b (+7.8 to +14.9) |
| Intelligence (½ open + ½ sealed) | ~63 (58.6–66.6) |
| Calibration | assumed ~87 (78–90): the ranked Qwen3.5-4B deciders score 85.6–88.6 |
| Capability ((I + C) / 2) | ~75 (68–78) |

- **Text leaderboard, overall:** composite ~74 (at decider-4b-v2's measured Speed and Cost) against
  73.70 for Cygnet (#1 today), 73.23 for Winnow-12B (#2) and 72.13 for Jev 1.13.0 (#3): about **#1**,
  inside the tie band (gaps under ~0.5 do not separate); ~#5 pessimistic.
- **4B class:** Capability ~75 against 72.3 for jevk5-v0.3-4b (best 4B today), 71.6 plumb-4b, 71.2
  decision-4b-v12, 70.7 decider-4b-v2: about **#1** (pessimistic ~#5–#7).
- **Image board:** Capability ~74–82, roughly #3 to #15 of 38, most likely just below Imajev-4B (82.1);
  one anchor (Mapika: 38/60 preview -> 67.1) and 60 preview items cannot place it more precisely.

## 7. Limits

Exploratory local runs; 2 seeds for the last two stages; decision rules written before each stage. The
proxy scorer is our reimplementation of the v1.5 rules; the sealed half and Calibration are estimated.
Unseen-v2 is our own set; its rank agreement with the board was not checked. The pilot, the screen and
the earlier 4B ran on hosts with and without the fused linear-attention kernels; compare across those
tables only loosely.

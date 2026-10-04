# v19 → v21: every experiment, every result (2026-10-03/04)

Private research log (exploratory). Nothing here is an official score: text numbers are local runs on the
231 public JevBench v1 tasks; image numbers are NaturalBench (300 groups), POPE-adversarial (600) and the
60 rebuildable Image JevBench preview items; "unseen" is our own private set of task families no model trained
on. Upstream (strands-labs) has received only PR #10 (merged 2026-10-03 23:17Z), PR #23 and issue #24.

## 0. How to read the numbers

- **v1.5-style Intelligence** (the main text number): per type, the board's chance-corrected competence —
  noul: P(yes) in (0.2, 0.8) counts wrong, CC = 100·(acc−0.5)/0.5; choice: CC = 100·(acc−c̄)/(1−c̄);
  score: graded by the **expected level** of the returned distribution, CC = 100·(1 − nMAE/nMAE_chance);
  Intelligence = mean of the three types. Computed by `analyze_fwd.py` / the snippets in this log.
  (The earlier "proxy" in `evaluation/jevbench/v15_proxy.py` graded score by the top level and therefore
  overstated MiniCPM5 models whose score temperature was broken — see §4.)
- **Hedged** = yes/no answers with 0.2 < P(yes) < 0.8 (of 74). **ECE** = top-label ECE, 10 bins.
- **Fix A** = per-kind temperatures refit by NLL on held-out rows (v19's own eval splits + a holdout sample,
  fit half / check half): **choice and score refit, yes/no deliberately unchanged** (§4).
- 3 runs = seeds 0/1/2; "averaged" = weight soup of the 3 (`strands_decider.soup`, exact mean update).

## 1. Lineage

| Name | What | Base | Status |
|---|---|---|---|
| v19 | Strands' published checkpoint `StrandsAgents/strands-decider-2B-hobson-v19@bb282d78` | Qwen3.5-2B-Base | reproduced exactly (167/231) |
| v19 + `--vision` | image input, no weight change (PR #10, merged) | Qwen3.5-2B | upstream |
| v20 (short) | v19 + Qwen3.5-27B yes/no teacher (kept where it agrees with gold) + no frozen-KL on yes/no, 1,500 steps | Qwen3.5-2B | exp/text, PR #23 |
| v20 image A | v19 + image fine-tune with image-removed copies (variant A), 400k px | Qwen3.5-2B | exp/image, PR #23 |
| v20-long | v20 recipe, 3,738 steps (full epoch), 3 runs + soup | Qwen3.5-2B | this log |
| bake-off | same recipe from each raw base, 2,400 steps: Qwen3.5-2B ×2, MiniCPM5-2B, Gemma 4 E2B | 3 bases | this log |
| v21 | MiniCPM5-2B from raw, full epoch, 27B teacher + no yes/no KL, init_seed 0, 3 runs + soup | MiniCPM5-2B (2.5B) | this log |
| combined Qwen | v20-long soup → image stage (variant A) — one checkpoint for text + images | Qwen3.5-2B | this log |
| combined Qwen (fixed) | as above, image stage with no yes/no frozen-KL + 27B-teacher text replay | Qwen3.5-2B | running (§8) |
| MiniCPM5 + eyes | v21 soup + SigLIP2-so400m graft (projector aligned on COCO-train captions) + image stage | MiniCPM5 + SigLIP2 | this log |
| Gemma diagnostic / Gemma image | Gemma 4 E2B(-it) recipe sweep; best arm + image stage | Gemma 4 E2B | this log |

## 2. Text leaderboard (231 public tasks, all models, uniform scoring)

| # | Model | Base | v1.5-style Intelligence (mean; runs) | Tasks right | Hedged | ECE |
|---|---|---|---|---|---|---|
| 1 | MiniCPM5 + eyes **+ Fix A** | MiniCPM5 + SigLIP2 | **51.3** (49.8 / 53.6 / 50.4) | 175.0 | 16.3 | — |
| 2 | MiniCPM5 + eyes (as trained) | MiniCPM5 + SigLIP2 | 47.7 (46.1 / 50.0 / 47.0) | 175.0 | 16.3 | 0.098 |
| 3 | v21 + Fix A | MiniCPM5 2.5B | 45.8 (47.1 / 46.2 / 44.2) | **179.3** | 27.0 | 0.065 |
| 4 | v21 averaged + Fix A | MiniCPM5 2.5B | 44.5 | 179 | 31 | 0.053 |
| 5 | v20-long averaged + Fix A | Qwen 2B | 42.5 | 177 | 35 | — |
| 6 | bake-off MiniCPM5 + Fix A | MiniCPM5 2.5B | 42.4 | 179 | 30 | 0.057 |
| 7 | v20-long averaged | Qwen 2B | 39.7 | 177 | 35 | 0.051 |
| 8 | v21 | MiniCPM5 2.5B | 38.9 (39.9 / 39.7 / 37.2) | 179.3 | 27.0 | 0.088 |
| 9 | v21 averaged | MiniCPM5 2.5B | 37.8 | 179 | 31 | 0.085 |
| 10 | v20 short | Qwen 2B | 37.7 (40.1 / 34.9 / 38.3) | 170.3 | 36.7 | 0.053 |
| 11 | bake-off MiniCPM5 | MiniCPM5 2.5B | 37.3 | 179 | 30 | 0.087 |
| 12 | v20-long | Qwen 2B | 35.7 (30.1 / 37.1 / 39.8) | 173.0 | 37.7 | 0.050 |
| 13 | v19 + Fix A | Qwen 2B | 34.2 | 167 | 48 | — |
| 14 | bake-off Qwen from raw | Qwen 2B | 29.7 (24.1 / 35.3) | 171.0 | 43.5 | 0.056 |
| 15 | v19 (published) | Qwen 2B | 29.5 | 167 | 48 | 0.050 |
| 16 | combined Qwen (text + images) | Qwen 2B | 29.3 (29.7 / 29.6 / 28.8) | 171.3 | 47.0 | 0.052 |
| 17 | Gemma 4 E2B-it (best diagnostic arm) | Gemma E2B | 17.4 | 177 | 57 | 0.104 |
| 18 | Gemma text + images | Gemma E2B | 10.0 (9.1 / 12.0 / 8.9) | 161.0 | 61.7 | 0.063 |

Paired 95% intervals against v19 (proxy scorer, seeds pooled): v21 +18.3 (+10.6, +26.7) proxy, +12.3 tasks
(+3.0, +22.3); v20-long +6.5 (+1.4, +12.0); v20 short +8.0 (+2.9, +13.3); v21 vs bake-off MiniCPM5 +2.0
(−1.8, +5.9, within noise); v21 soup vs its 3 seeds −1.4 (−4.9, +1.8, within noise).

## 3. Unseen task families ("sealed-like") — the generalisation check

1,050 rows, `build_unseen.py`, seed 0, none in any training mix (`data/sources.md` checked): StrategyQA 250
(noul, question only), RuleTaker depth-5 250 (noul, held out of training by design), CommonsenseQA 150 and
ARC-Challenge 150 (choice), STS-B 250 (score, 6 levels). Same v1.5-style scoring.

| Model | Public I (Fix A) | **Unseen I** (before → Fix A) | Drop | Unseen noul / choice / score (Fix A) | Unseen ECE (before → Fix A) | Unseen yes/no over-confidence |
|---|---|---|---|---|---|---|
| **v20-long averaged** (Qwen 2B) | 42.5 | 33.2 → **34.7** | **−7.8** | −10.8 / **64.7** / 50.1 | 0.057 → **0.055** | +0.139 |
| MiniCPM5 + eyes s1 | **53.6** | 29.8 → 32.4 | −21.2 | −10.4 / 58.2 / 49.2 | 0.132 → 0.079 | +0.090 |
| MiniCPM5 + eyes s2 | 50.4 | 28.1 → 30.3 | −20.1 | −13.2 / 55.2 / 49.0 | 0.146 → 0.088 | +0.116 |
| MiniCPM5 + eyes s0 | 49.8 | 26.4 → 28.7 | −21.1 | −18.0 / 57.4 / 46.8 | 0.123 → 0.085 | +0.097 |
| v21 s0 | 47.1 | 25.0 → 28.7 | −18.4 | −14.8 / 57.0 / 44.0 | 0.126 → 0.101 | +0.151 |
| v21 averaged | 44.5 | 23.9 → 27.6 | −16.9 | −20.4 / 58.2 / 45.0 | 0.117 → 0.089 | +0.117 |
| v21 s1 | 46.2 | 23.8 → 27.5 | −18.7 | −14.8 / 52.6 / 44.7 | 0.120 → 0.085 | +0.138 |
| v21 s2 | 44.2 | 22.6 → 26.6 | −17.6 | −23.2 / 57.8 / 45.3 | 0.115 → 0.095 | +0.129 |
| v19 | 34.2 | 20.3 → 23.6 | −10.6 | −40.8 / 60.4 / 51.1 | 0.073 → **0.043** | +0.073 |
| decider-2b (Mapika; official I 42.3, 2B #1 on Intelligence) | — | 21.0 | — | −76.4 / **73.8** / **65.7** | 0.053 | +0.108 |
| decision-2b (FlyMy; official Capability 58.8, 2B #1 on Capability) | — | 20.2 | — | −57.6 / 60.4 / 57.7 | 0.105 | +0.165 |

Competitors were run locally with their own published code and calibration (`comp_run.py`: Mapika `decider.infer.Decider.system_one`, FlyMy `model.load().decide`), same 1,050 rows. Their ordering here (decider-2b 21.0 > decision-2b 20.2) matches the official board's Intelligence ordering (42.3 > 31.3). **Every trained model of ours scores above both on these unseen families** (+2.6 to +14.5): we win on yes/no (they hedge or are wrong on StrategyQA/RuleTaker-d5), they win on choice and score (decider-2b 73.8 / 65.7 vs our best 64.7 / 51.1).

Read: on families nobody trained on, the Qwen v20-long soup leads and loses least (−8 vs −17…−21 for every
MiniCPM5 model): MiniCPM5's public lead is largely familiarity with the public task formats. Everyone's unseen
yes/no is below chance and over-confident (the target of Fix B). One set of 1,050 rows: < ~3 points is noise.

## 4. Calibration

Diagnosis (held-out rows, current temperatures):

| Model | yes/no over-conf. | choice | score | T current (noul / choice / score) |
|---|---|---|---|---|
| v21 s0 / s1 / s2 | +0.094 / +0.059 / +0.057 | −0.03 / −0.02 / −0.04 | −0.11 / −0.12 / −0.11 | 0.82/0.91/**4.58**, 0.91/0.86/**3.90**, 1.01/0.91/**4.12** |
| v21 averaged | +0.051 | −0.02 | −0.12 | 0.56/0.48/2.16 |
| bake-off MiniCPM5 | +0.061 | −0.02 | −0.08 | 1.01/0.96/**3.69** |
| v19 | +0.014 | +0.01 | −0.05 | 0.91/0.73/1.33 |

Root causes: (1) yes/no over-confidence in every MiniCPM5 checkpoint — the price of the decisiveness the 27B
teacher taught; (2) **score temperature 3.7–4.6 in every MiniCPM5 checkpoint**, flattening score answers; most
likely from calibrating on the rebuilt `holdout_v5_norule.jsonl`, which no longer matches `data/SHA256SUMS`.

Three refits compared (v21 s0): current 51.5 proxy / NLL refit of all kinds (honest, but yes/no softens:
43.3, 39 hedged) / "balanced" yes/no T maximising (CC+C)/2 → T hit the grid floor 0.22 (63.2, 2 hedged) —
**rejected**: it is near-clamping out of the band, i.e. metric gaming (file kept as
`results/calib_balanced_REJECTED.json`). **Fix A** = refit choice and score, keep yes/no: see §2–3; it improves
every MiniCPM5 checkpoint on every check (public I +5…+7, public ECE −⅓…−½, held-out score ECE ~halved,
unseen ECE down) and raises score-type Intelligence (expected-level grading) from ~36 to ~56–58.
Recommended temperatures: `results/recommended_temperatures.json` (also in S3).

## 5. Images (models that can see)

| # | Model | NB acc | NB G-Acc | NB ECE | POPE acc | POPE ECE | Preview /60 | Image-removed conf. NB / POPE | Text check (899 rows) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | v20 image A (image-only Qwen, mean of 3) | **0.805** | **0.371** | 0.048 | 0.874 | 0.032 | **42.7** | 0.58 / 0.53 | 0.82–0.83 |
| 2 | combined Qwen (mean of 3) | 0.801 | 0.355 | **0.023** | **0.877** | 0.038 | 40.7 | **0.58 / 0.52** | **0.83–0.84** |
| 3 | v19 + `--vision` | 0.787 | 0.347 | 0.029 | 0.858 | 0.082 | 40 | 0.65 / 0.73 | 0.817 |
| 4 | Gemma text + images (mean) | 0.729 | 0.216 | 0.067 | 0.836 | 0.145 | 27 | 0.78 / 0.73 | 0.79–0.80 |
| 5 | MiniCPM5 + eyes (mean) | 0.679 | 0.130 | 0.104 | 0.818 | 0.100 | 16 | 0.64 / 0.62 | 0.84–0.85 |
| — | Gemma untrained | 0.512 | 0.000 | 0.125 | 0.503 | 0.123 | 24 | 0.58 / 0.55 | — |

Per run — combined Qwen: NB 0.796/0.807/0.800, POPE 0.877/0.872/0.882, preview 41/38/43. MiniCPM5 + eyes: NB
0.687/0.675/0.675, POPE 0.825/0.817/0.812, preview 20/16/12. Gemma: NB 0.723/0.739/0.726, POPE 0.835/0.837/0.837.

## 6. One checkpoint for text + images (multimodal)

| Model | Text I (v1.5-style) | NB | POPE | Preview | Image-removed | Verdict |
|---|---|---|---|---|---|---|
| **combined Qwen, fixed** (3 runs) | **39.0** (39.4 / 39.0 / 38.7; proxy 42.9 / 42.2 / 41.7; hedged 35–36) | **0.802** | 0.874 | **42.0** | mostly (0.60 / 0.58) | **best one-checkpoint model: text ≈ v20-long, images ≈ best, true 2B** |
| combined Qwen (first attempt) | 29.3 | 0.801 | 0.877 | 40.7 | fixed | images ≥ v19, text ≈ v19 (image stage undid the text gain) |
| v19 + `--vision` | 29.5 | 0.787 | 0.858 | 40 | not fixed | baseline |
| MiniCPM5 + eyes (+ Fix A) | 51.3 | 0.679 | 0.818 | 16 | partly | best text, weak images |
| Gemma text + images | 10.0 | 0.729 | 0.836 | 27 | not fixed | weakest |

Why the combined Qwen lost its text gain (v20-long soup 41.5 proxy → 31.4): the image stage re-applied frozen-KL
to every row (incl. yes/no text replay) and replayed text rows with plain labels; temperatures were unchanged.
Fix: `kl_frozen_skip_kinds: [noul]` + replay rows with 27B-teacher targets (branch exp/v21-combined-fix,
762bf14, opt-in, tested) — the "fixed combined Qwen" in §8.

## 7. Bake-off, Gemma diagnostic, graft details

- Bake-off (2,400 steps from raw; rule fixed before scores, `research/bakeoff/DECISION.md`): MiniCPM5 179 / 48.5
  proxy; Qwen 164 / 30.0 and 178 / 40.8; Gemma 150 / 14.5. MiniCPM5 vs Qwen mean +13.0 (+6.3, +20.0); Gemma
  −20.9 (−32.0, −9.6). Speeds on H200: Qwen 0.56, MiniCPM5 0.48, Gemma 0.40 step/s.
- Gemma diagnostic (proxy, tasks right, hedged): base full epoch 12.2 / 147 / 61; base lr 3e-4 22.3 / 154 / 54;
  -it lr 3e-4 23.2 / 166 / 57; **-it 27.7 / 177 / 57** → Gemma knows the answers but hedges.
- v20-long: seeds 33.4 / 40.2 / 42.4 proxy (172 / 174 / 173 right), soup 41.5 (177) — the combined-Qwen rule
  (fixed before these scores): max(seed mean 38.7, soup 41.5) ≥ v20-short 40.1 → start from the v20-long soup.
- Graft: SigLIP2-so400m-patch16-384@dd658faa (Apache-2.0), 2×2 unshuffle → 144 tokens/image, MLP projector
  4608→2048→2048; stage 1 on 80,000 COCO train2014 captions, 621 steps, 694 s, val loss 1.87; stage 2 from the
  v21 soup, variant A.

## 8. Fixed combined Qwen (finished 2026-10-04 05:04Z)

Per run — text v1.5-style 39.4 / 39.0 / 38.7 (175 / 169 / 173 right, hedged 36 / 35 / 36, ECE 0.041 / 0.051 / 0.033); NB 0.802 / 0.799 / 0.804, G-Acc 0.360 / 0.373 / 0.367, NB ECE 0.089 / 0.080 / 0.085 (worse than the first attempt's 0.023 — image temperatures not applied in this table), POPE 0.865 / 0.875 / 0.882, preview 43 / 43 / 40, image-removed confidence 0.61 / 0.60 / 0.60 (NB) and 0.59 / 0.58 / 0.56 (POPE), text check 0.829 / 0.828 / 0.825. The fix (no yes/no frozen-KL + 27B-teacher replay) restored the text gain (29.3 → 39.0) with images unchanged. Not yet run on the unseen set (machine destroyed after verified upload; checkpoints in S3 machineB/).

## 9. Data, code, compute

- S3 `s3://vision-decider-643603452951-us-east-1/v21-20261003/`: data.tar.gz (sha 1db13d3e…), image-data.tar
  (6421f38b…), ijb_preview.tgz, checkpoints.tar.gz + runs + logs + MANIFEST (machine A), v21-minicpm5-soup.tar,
  v20-short-3runs.tar (9f43f2d9…), recommended_temperatures.json; machine B's archives under `machineB/` at the end.
- Branches (Vivek0712/strands-decider): exp/v21 (multi-base, init_seed, soup, runners, this log),
  exp/v21-gemma-vision (4eb25bc), exp/v21-minicpm-vision (2f45ddc), exp/v21-combined-fix (762bf14).
- Machines: vast 54052306 4×H200 (bake-off, v21, v20-long, combined; destroyed 2026-10-04T03:08Z after verified
  upload), vast 54074831 4×H200 NVL (Gemma, graft, combined-fix, unseen eval; self-destructs after verified upload).
- Pinned bases: Qwen3.5-2B-Base b1485b2f, MiniCPM5-2B f9740005, Gemma 4 E2B d29ff6b4…, Gemma 4 E2B-it 3e22461f,
  Qwen3.5-27B fc05daec, SigLIP2 dd658faa. 27B teacher file tonight c72780dc… (afternoon 5c381fb0…).

## 10. Rules kept

No benchmark items or paraphrases in training; image dedupe 0 within Hamming 6; no probability clamping (the
"balanced" refit was rejected for that reason); decision rules fixed before scores; nothing posted upstream
beyond #10/#23/#24; no AWS credentials on rented hosts (presigned URLs only).

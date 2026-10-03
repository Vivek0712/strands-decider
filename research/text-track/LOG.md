# Text track log (private research; exp/text)

Goal: make v19's yes/no answers more accurate and more decisive (JevBench v1.5 counts
0.2 < P(yes) < 0.8 as wrong) without gaming: no JevBench items or paraphrases in training,
and no clamping or snapping of probabilities.

Code on this branch:
- `train.py`: `continue_from` (continue a checkpoint with its adapter and head trainable;
  calibration reset to 1.0 for training) and `kl_frozen_skip_kinds` (no frozen-KL on the
  listed row kinds; `collate.py` emits `kind_id`; `distributed.StepSlices` respects it).
  (`init_from`, upstream, swaps in a fresh SlotHead and freezes the torso, so it cannot continue v19.)
- `configs/experiments/text-track/t3-control.yaml`, `t3-noanchor-yn.yaml`: T3 arms (1,500 steps
  ≈ 0.4 epoch, v19 hyperparameters, v14 replay teacher on the multistep rows, seed 0).
- `research/text-track/teacher_yn.py`: T4 labelling of the yes/no-heavy rows with a stronger
  teacher, read as `data/teacher.py` does, then kept only where argmax = gold and merged over the v14 replay.
- `research/text-track/score_run.py`: v1.5 proxy plus band share, band by family, yes share, Brier and ECE.

## Run 0 — T2 yardstick: v19 on JevBench v1 (2026-10-03)

- Host: vast.ai 53990685, 1x H100 SXM 80GB (driver 560.35.03), $2.13/h, label text-track.
  Created 08:59Z, destroyed 09:12Z.
- Env: Python 3.11, torch 2.7.1+cu126, transformers 5.17.0, peft 0.21.0, flash-linear-attention.
- Checkpoint: HF StrandsAgents/strands-decider-2B-hobson-v19 @ bb282d786bc251fd4e3068de3ada9ddbb38127cd.
- Command: `PY=/root/venv/bin/python GPU=0 evaluation/jevbench/jevbench.sh /root/ck/v19 /root/runs/v19-repro`
  (JevBench 1bcc55eb6c8cffde2306b3db03ede39b61c6152a, window 4096). Server load 21 s, 231 tasks in 63 s.
- Result: **167/231, identical to Strands' published run task for task, max probability difference 0.0.**

| run | v1 acc | n_correct | proxy Intelligence | yes/no in band | band share | choice / noul / score competence | Brier | ECE | yes share |
|---|---|---|---|---|---|---|---|---|---|
| v19 published (HF eval/jevbench-w4096) | 0.723 | 167 | 32.0 | 48 / 74 (26 of them right) | 0.649 | 64.9 / −32.4 / 63.4 | 0.3477 | 0.0501 | 0.486 |
| v19 reproduced (run 0) | 0.723 | 167 | 32.1 | 48 / 74 (26 right) | 0.649 | 65.4 / −32.4 / 63.4 | 0.3477 | 0.0501 | 0.486 |

Band answers by family: judge_hard 14, adequacy 11, policy 7, long_policy 4,
temporal_numeric 4, probability 3, tradeoff 2, multi_hop 1, adversarial 1, trap 1.
(The proxy's choice competence differs by 0.5 between the two copies of identical results;
`kind()` classifies one item by the keys of its probs dict. Not investigated further.)

## Run 1 — T1 corpus build (same host, CPU, run in parallel with run 0)

`training/recipe.sh build` concurrently with `fetch` → (`multistep` ‖ `adequacy` ‖ `generated`). Wall time about 4.5 min.

- `data/train_v5.jsonl` (100,449 rows), `data/multistep_v14.jsonl` (12,909), the four raw downloads,
  every committed synthetic file: **match data/SHA256SUMS**, so the committed teacher and replay files align by position.
- `data/train_v5.holdout.jsonl` and `data/holdout_v5_norule.jsonl` (21,000 rows): **do NOT match**.
  An upstream held-out HF dataset has changed. These files are used only for calibration and the internal
  held-out eval, so recipe `calibrate`/`eval` stop at `verify`. Adaptation: run `strands-decider calibrate`/`eval` directly
  on the rebuilt file for every arm, and re-measure v19 on the same file. Absolute held-out numbers
  are then not comparable to Strands' published 0.647.
- adequacy: 4,866 train / 234 eval rows; generated: v16p/v18p paraphrases attached; multistep eval 4,084.
- Hashes of every built file: `checkpoints/text-runs/logs-h100-1/data_sha256.txt` (local, not committed).

## Run 2 — T4 teacher check: Qwen/Qwen3.5-27B @ fc05daec on the adequacy eval sets

Read as in `data/teacher.py` (HF batched forward, bf16, max_batch_tokens 32000). vLLM 0.30 was tried
for speed but its wheels need CUDA 13 (driver ≥ 580) and the host driver is 560, so it was dropped.
For a forward-only letter readout, prefill-only batched HF is close to vLLM throughput.

| eval | frozen Qwen3.5-4B (PREREG-v19) | Qwen3.5-27B | v19 |
|---|---|---|---|
| HelpSteer2 adequacy eval (234) | 0.645 | **0.744** (adequate recall 0.607, inadequate 0.880) | 0.739 |
| generated adequacy eval (302) | 0.810 bal. | **0.884** (0.884 / 0.884) | 0.788 bal. |

The 27B is a clearly stronger judge than the 4B and is ahead of v19 on the out-of-source generated set,
so T4 has a teacher worth distilling. Labelling time: 234 long rows in 84 s (including warm-up) and 302 in 31 s.

## Status: STOPPED before training

The upgraded plan needed one multi-GPU host. No 4x H100 SXM offer was available at 09:15Z; the
best was a 2x H100 SXM (offer 49362313, $4.53/h, driver 560.35.05). Creating it was **denied by the
Claude Code permission classifier ("Real-World Transactions")**, so the track stopped here pending the user's approval.
T3 (control and anchor-off), T4 labelling and training, the seeds and the 4B exploration were not run.

vast.ai spend for this track: about 13 min at $2.13/h ≈ $0.50 (plus a few cents of storage). No text-track instance is running.

### Resume attempt (2026-10-03, later)
Asked to resume after a permission rule was reportedly added. A 4x H100 SXM offer (49362311, $9.07/h, driver 560.35.05)
was available, but `vastai create instance` was **denied again by the auto-mode classifier ("Real-World Transactions")**.
No instance was created. The track is still stopped before training; spend is unchanged at about $0.50.

## Run 3 — T3/T4 arms on 4x H100 (2026-10-03)

- Host: vast.ai 53993770, 4x H100 SXM 80GB (driver 560.35.05), $9.11/h, label text-track. Rented by the user
  (the agent's own rental attempts were blocked by the permission classifier); run driven from the main session.
- Env as run 0 (torch 2.7.1+cu126, transformers 5.17.0, peft 0.21.0, fla). Host apt mirror was down; switched to
  archive.ubuntu.com to get gcc (triton needs it). `causal_conv1d` not installed (reference conv path; slower, same maths).
- Corpus rebuilt: every file's sha256 identical to run 1 (`checkpoints/text-runs/data_sha256_h100x4.txt`);
  the two held-out files differ from SHA256SUMS exactly as in run 1.
- T4 labels, Qwen3.5-27B @ fc05daec, two shards on GPU2/3 (18.4 and 12.5 min):

| source | labelled | teacher argmax = gold | mean P(gold) where kept |
|---|---|---|---|
| train_v5 (yes/no rows) | 47,449 | 0.831 | 0.952 |
| adequacy_hs2 | 4,866 | 0.745 | 0.899 |
| adequacy_gen | 1,300 | 0.919 | 0.952 |
| generated_v16 (yes/no) | 754 | 0.865 | 0.923 |
| generated_v18 (yes/no) | 488 | 0.879 | 0.907 |

  `data/teacher_t4.jsonl`: 58,246 rows (12,909 v14 replay + 45,337 kept 27B rows, none overlapping);
  sha256 5c381fb0…; raw c309aaec….
- Arms (`research/text-track/arm.sh`: train → `calibrate` on the rebuilt held-out file → JevBench v1, one GPU each):
  GPU0 t3-control, GPU1 t3-noanchor-yn, GPU2 t4-teacher27b, GPU3 t34-combined; all seed 0, 1,500 steps.

### Run 3 results (seed 0; JevBench v1 public 231 tasks, scored with score_run.py v1.5 proxy)

| arm | n_correct | proxy Intelligence | choice / noul / score competence | yes/no in band (of 74) | yes/no right | Brier | ECE | yes share | val_acc | train wall |
|---|---|---|---|---|---|---|---|---|---|---|
| v19 (run 0) | 167 | 32.1 | 65.4 / −32.4 / 63.4 | 48 | 51 | 0.3477 | 0.0501 | 0.486 | — | — |
| t3-control | 172 | 34.6 | 70.0 / −29.7 / 63.4 | 44 | 51 | 0.3535 | 0.0573 | 0.405 | 0.8575 | 63 min |
| t3-noanchor-yn | 169 | 34.5 | 64.4 / −24.3 / 63.4 | 41 | 54 | 0.3582 | 0.0361 | 0.419 | 0.845 | 91 min |
| t4-teacher27b | 171 | 39.4 | 68.2 / −13.5 / 63.4 | 37 | 52 | 0.3515 | 0.0517 | 0.392 | 0.850 | 59 min |
| t34-combined | 169 | **40.6** | 66.5 / −8.1 / 63.4 | 35 | 52 | 0.3492 | 0.0439 | 0.419 | 0.8525 | 59 min |

Read: the 27B teacher is what moves yes/no decisiveness (band 48 → 37); dropping the anchor adds a little (→ 35).
More training alone (t3-control) helps choice, barely moves yes/no. Single seed on 74 yes/no items: differences of
a few items are within noise, hence seeds 1 and 2 of t4 and t34 (launched 12:19Z, GPU0-3). Exploratory, not confirmatory.

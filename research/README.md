# Research

This folder holds the record of the experiments behind the reference recipe, v1 to v20,
and the tools that keep it. Since v9, every run is preregistered: its predictions, and the
failure that would refute them, are committed before training. The outcome is appended
after the run, and nothing before it is edited. A run replaces the reference only when it
meets its own bar. v13, v14, v16, v17 and v18 missed their bars and were promoted by
decision ([history.md#experiments-since-v7](history.md#experiments-since-v7)); the v16 to
v18 preregistrations record the decision in Decision after the outcome. The current results are in [evaluation/README.md](../evaluation/README.md).

- [`preregistrations/`](preregistrations/README.md): the 14 frozen preregistrations, v9 to
  v20, with a map of the paths they name and the configuration of each run.
- [`history.md`](history.md): the narrative by theme, v1 to v19: what moved the benchmark,
  what did not, and what is left.
- [`generations.md`](generations.md): the dated ledger, one entry per generation from v1 to
  v19, with the files each came from. It was written from the private development
  history, which is not in this repository.
- [`next/STAGE_A.md`](next/STAGE_A.md): the next-model plan's engineering stage (unseen-v2,
  the A1/A2/A3 recipe changes, the stage configs) and the commands and fixed decision rules
  for Stages B to E.
- [`data/`](data/): the saved per-case results, as CSV.
- [`figures/`](figures/) and [`scripts/`](scripts/): the figures, and the scripts that collect
  the results and draw them.
- [`../configs/experiments/`](../configs/experiments/): the training configuration of each
  preregistered run from v11 on.

To propose an experiment, write its preregistration first, in the form of the files in
`preregistrations/`, and commit it before you train. Paths are relative to the repository
root unless they are links.

## Preregistration practice

Since v9, every run has been pre-registered (see [Preregistrations](#preregistrations)):
predictions and the failure that would refute them, committed before training, and
outcomes appended without editing what came before. That is why v13 and v14 are recorded as
missing their bars rather than as wins, and why the negatives in
[the experiment history](history.md) can be believed. One measurement is still missing: a
**seed replicate**. McNemar covers which tasks were sampled, not training noise, and
eight runs since v7 span 143-161 tasks around its 154. (As of v14. Six retrains on AWS
later measured the retrain noise, SD 3.2 tasks: see
[Retraining on AWS](../evaluation/results.md#retraining-on-aws). The v19 seed replicate was stopped
before it gave results.)

## Preregistrations

A JevBench score is the number of the 231 public tasks answered correctly. "Bar missed"
means that a pre-registered prediction failed, so by the rule the earlier model stayed.
The files are frozen and name the paths of the layout they were written in;
[preregistrations/README.md](preregistrations/README.md) maps each old path to the new one.

| Version | Purpose | Outcome | Preregistration |
| --- | --- | --- | --- |
| v9 | Synthetic stated-rule execution | Bar missed. Learned the generator, JevBench 152. | [PREREGISTRATION-v9.md](preregistrations/PREREGISTRATION-v9.md) |
| v10 | Minimal pairs for cross-reference and requirement checking | Bar missed. Targeted families fell, JevBench 145. | [PREREGISTRATION-v10.md](preregistrations/PREREGISTRATION-v10.md) |
| v11 | Teaching the model to read the question (arms v11a and v11b) | Both arms lost on JevBench (143 and 144). | [PREREGISTRATION-v11.md](preregistrations/PREREGISTRATION-v11.md) |
| v12 | Distilling a frozen Qwen3.5-4B | Bar missed. JevBench 146. | [PREREGISTRATION-v12.md](preregistrations/PREREGISTRATION-v12.md) |
| v13 | A Qwen3.5-2B-Base torso | Bar missed (157 against 162). Promoted by decision. | [PREREGISTRATION-v13.md](preregistrations/PREREGISTRATION-v13.md) |
| v14 | Multi-step documents with a teacher | Bar missed by 2 tasks (161). Promoted by decision. | [PREREGISTRATION-v14.md](preregistrations/PREREGISTRATION-v14.md) |
| v15 | Rules applied to a user's situation (ShARC, ConditionalQA) | Bar missed. JevBench 160. | [PREREGISTRATION-v15.md](preregistrations/PREREGISTRATION-v15.md) |
| v16 | Generated document questions | Bar missed (163). Promoted by decision. | [PREREGISTRATION-v16.md](preregistrations/PREREGISTRATION-v16.md) |
| v17 | Replay toward v14 | Bar missed on HotpotQA (164). Promoted by decision. | [PREREGISTRATION-v17.md](preregistrations/PREREGISTRATION-v17.md) |
| v18 | Generated questions on the weak skills | Bar missed on HotpotQA by one question (164). Promoted by decision. | [PREREGISTRATION-v18.md](preregistrations/PREREGISTRATION-v18.md) |
| v19 | Answer adequacy | All four predictions held (167). Became the reference recipe by its own rule. | [PREREGISTRATION-v19.md](preregistrations/PREREGISTRATION-v19.md) |
| v19-calmix | Calibrating v19 on a broader held-out mix, no training | Rule failed on NLL. v19's temperatures stay. | [PREREGISTRATION-v19-calmix.md](preregistrations/PREREGISTRATION-v19-calmix.md) |
| v19-seed1 | The seed replicate of v19 | Stopped at step 1,420. No results taken. | [PREREGISTRATION-v19-seed1.md](preregistrations/PREREGISTRATION-v19-seed1.md) |
| v20 | Reading the question, and confidence | Four predictions failed (169 at the 4096 window). v19 stays the reference recipe. | [PREREGISTRATION-v20.md](preregistrations/PREREGISTRATION-v20.md) |

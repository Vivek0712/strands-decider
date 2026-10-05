# unseen-v2

The second unseen-family set: 9,000 questions, a third per question type, from 11 families
no Strands Decider trains on, split once into dev (40%) and test (60%) within each family.
It extends [unseen/](../unseen/README.md) (1,050 rows), which was too small to resolve a
3-point difference. A local measure, not a JevBench score.

| Family | Yes/no | Choice | Score | Gold |
| --- | ---: | ---: | ---: | --- |
| StrategyQA (v1) | 400 | | | dataset, `ChilleD/StrategyQA` train |
| RuleTaker depth 5 (v1) | 400 | | | `strands-decider data build`'s held-out file |
| CommonsenseQA 2.0 (new) | 400 | | | dataset, dev file |
| CommonsenseQA (v1) | | 400 | | dataset, validation |
| ARC-Challenge (v1) | | 400 | | dataset, test |
| HellaSwag (new) | | 400 | | dataset, validation |
| STS-B (v1) | | | 600 | dataset, test, 6 levels |
| chess | 450 | 450 | 600 | python-chess rules: check, legality, gives check, mate in one, material, king mobility |
| arithmetic | 450 | 450 | 600 | integer expressions and word problems; how A compares with B (5 levels) |
| calendar | 450 | 450 | 600 | `datetime`: weekday, weekend, leap year, order, offsets, the gap between two dates (5 levels) |
| seating | 450 | 450 | 600 | five seats, true clues with exactly one arrangement: who sits where, left of, how many to the left |
| **Total** | **3,000** | **3,000** | **3,000** | |

The generator families ([`generators.py`](generators.py)) pick a question template, then
a gold label uniformly, and draw until a row has it, so labels are balanced within every
template; a score item too close to a level boundary is not drawn. Excluded on purpose:
BoolQ and MNLI are training sources (so no BoolQ-hard or e-SNLI), and nothing comes from
JevBench or Image JevBench. `tests/test_unseen_v2.py` holds both, and that no family is a
training family. The sources, revisions, licences and file hashes are in
[data/sources.md](../../data/sources.md).

| Script | What it does |
| --- | --- |
| [`build.py`](build.py) | Builds the rows from one seed; writes a manifest (revisions, file sha256s, library versions, the output's sha256) beside them |
| [`generators.py`](generators.py) | The four generator families |
| [`score.py`](score.py) | v1's grading (yes/no band, chance-corrected credit, score by expected level) plus NLL, yes/no Brier, band mass, score RPS = (1/(K-1))·Σ(F_k − 1[gold ≤ k])², top-probability ECE; `--split`; `--vs`: paired by item, bootstrap 95% intervals, `--resample-runs` for items × seeds |

```bash
pip install "strands-decider[unseen]"     # datasets and python-chess
python evaluation/unseen_v2/build.py --out data/unseen_v2.jsonl           # ~2 min on a laptop CPU
python evaluation/unseen/run.py --rows data/unseen_v2.jsonl --checkpoint CKPT --out reports/unseen_v2/NAME
python evaluation/unseen_v2/score.py reports/unseen_v2/NAME --split dev --vs reports/unseen_v2/BASE
```

Seed 0 with the pinned inputs gives 9,000 rows (dev: 1,201 yes/no, 1,205 choice, 1,194
score) whose sha256 the manifest records; a rebuild must reproduce it before any model is
read on the set. Model selection reads `--split dev`; `test` is read once per stage.

python-chess is GPL-3.0. It is a build-time tool in the `unseen` extra, never imported by
the package; the rows hold positions and rule facts, and none of its code is redistributed.

## Reading the numbers

At 3,000 rows per type, the per-type competence has a standard error of about 2-3 points
on dev; read differences through `--vs`, which pairs runs item by item. The thresholded
competences are noisy; NLL, Brier and RPS move with less noise and are reported beside them.

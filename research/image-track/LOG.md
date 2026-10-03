# Image track log (exp/image)

Goal: Strands Decider (v19, 2B) with `--vision` up the 2B class on Image JevBench, without
benchmark gaming. Private research branch; nothing here is posted upstream.

Conventions
- Every run gets an entry *before* it starts (config, commit, seed, command, data) and its
  results *after* (metrics, wall time, cost, artifact paths). Failed runs stay in the log.
- Evaluation = `evaluation/vision/run.py` (strands system): NaturalBench shard 0 groups
  0-299 (1200 items), POPE adversarial first 600 questions, the 128 rebuilt Image JevBench
  preview items (60 "exact": ArxivQA, CLEVR-HOPE, Geometry3K x 20), plus the image-removed
  ("blind") pass. Image temperatures are fitted only on NaturalBench groups 300-599
  (`--nb-start 300`), which no evaluation uses.
- Per-item outputs for every run: `/Users/vivekrajaps/vision-decider/checkpoints/image-runs/<run>/`.
- Pinned: v19 `StrandsAgents/strands-decider-2B-hobson-v19@bb282d78`, base
  `Qwen/Qwen3.5-2B-Base@b1485b2f`, NaturalBench `@ba41a7d5`, POPE `@4db12766`,
  Multimodal-Mind2Web `@1b4c6a8c`. transformers 5.18.0, peft 0.21.2, torch 2.7.1+cu126,
  PIL image processor (pinned in-tree).
- Cost ledger at the end.

## Hosts

| host | offer | GPU | $/h | used for |
|---|---|---|---|---|
| vast 53990925 (CZ) | 49793506 | 1x H100 SXM 80GB, driver 610.57 | 2.75 | I1 evaluations, dataset build, dedupe, training smoke test |

## I1 — pixel budget and image temperatures (no training)

Code (commit 0191d24): `vision.fit_image(img, long_side, max_pixels)` scales down (never
up) to at most `max_pixels`, keeping the aspect ratio (floor rounding so the area stays
within budget; the long-side path keeps its old rounding, so 448 px reproduces exactly).
`VisionEngineConfig.image_max_pixels`, `load_vision_engine(image_max_pixels=)`, and
`StrandsDeciderConfig.image_temperature_by_kind` (image questions use it where fitted,
else the text temperatures; requests without images never do). `run.py` gains
`--max-pixels`, `--nb-start`, `--temps`, `--no-blind`; the image-removed pass always uses
the text temperatures (it is a text-only request to the server).
`research/image-track/temps.py` fits per-kind temperatures by NLL on held-out rows and
rescores a run exactly from its stored probabilities (logits = T0 * log p).

Runs (all v19, H100, `research/image-track/remote/i1.sh`, commit 0191d24):

| run | resize | items |
|---|---|---|
| i1-v19-448 | long side 448 | eval + blind |
| i1-v19-px400k | budget 400k px (no long-side cap) | eval + blind |
| cal-v19-448 | long side 448 | NB groups 300-599, no blind |
| cal-v19-px400k | budget 400k | NB groups 300-599, no blind |
| i1-v19-px800k | budget 800k | eval, no blind |
| cal-v19-px800k | budget 800k | NB groups 300-599, no blind |

Wall (H100, no fla kernels): eval+blind 322 s at 448, 455 s at 400k; held-out NB 233 s / 227 s.

### I1 results (v19, no training)

Image temperatures fitted on held-out NB groups 300-599 (`temps.py fit`), then applied to
the evaluation rows (`temps.py apply`; image-removed rows keep the text temperatures).

| setting | NB acc | NB G-Acc | NB ECE | POPE acc | POPE Brier | POPE ECE | IJB exact | IJB all | IJB ECE | blind conf NB/POPE | blind ECE NB/POPE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 448 px (baseline, reproduces the brief) | 0.784 | 0.327 | 0.014 | 0.877 | 0.202 | 0.072 | 37/60 | 0.430 | 0.107 | 0.652 / 0.725 | 0.152 / 0.225 |
| 448 px + image T (noul 0.810, choice 0.775) | 0.784 | 0.327 | 0.019 | 0.877 | 0.196 | 0.061 | 37/60 | 0.430 | 0.101 | same | same |
| 400k px | 0.788 | 0.347 | 0.029 | 0.858 | 0.209 | 0.082 | 40/60 | 0.555 | 0.089 | same | same |
| **400k px + image T (noul 0.759, choice 0.790)** | 0.788 | 0.347 | 0.029 | 0.858 | 0.203 | 0.057 | **40/60** | 0.555 | **0.072** | same | same |

Per family (IJB acc), 448 -> 400k: ArxivQA 0.55 -> 0.60, CLEVR-HOPE 0.85 -> 0.90, FinQA
0.15 -> 0.20, Geometry3K 0.45 -> 0.50, Mind2Web 0.25 -> 0.17, ScreenSpot 0.33 -> 0.69.
POPE's COCO photos (640x480 = 307k px) are not downscaled at 400k, and v19 loses 1.8 points
there at full resolution; everything else gains. The image temperatures are sharper than
the text ones for noul (0.76-0.81 vs 0.91) and change accuracy nowhere (monotone per row).

Budget 800k (v19, no image T; its held-out fit was stopped to free the host): NB acc 0.792,
G-Acc 0.357, ECE 0.027; POPE 0.858 / Brier 0.209 / ECE 0.082; IJB exact **41/60**, IJB all
0.617, IJB ECE 0.057; ArxivQA 0.65, CLEVR 0.90, FinQA 0.20, Geometry 0.50, Mind2Web 0.33,
ScreenSpot 0.83. More pixels keep helping IJB (screens most) at twice the image tokens.

**I1 pick: 400k-pixel budget + image temperatures** (best IJB exact, IJB ECE, NB G-Acc;
POPE -1.8 pt is the cost; 400k is the budget the strongest 4B entry serves, 800k is a
further +1 exact at 2x tokens). Training runs use the 400k budget.

Host 1 cost: 09:01:20-09:35:51 UTC = 0.575 h x $2.75 = **$1.58**.

## Training data (I2)

Built on the GPU host by `research/image-track/builders/*.py` via `remote/build.sh`
(`remote/fetch.sh` downloads the raw sources). Rows are the in-tree `Example` plus
`images`, `ablation`, `source`, `source_id`, `pair_id`. Yes/no questions are rendered
60% as nouls with the server's default criteria and 40% as yes/no choices.

| file | rows | images | source / licence | content |
|---|---|---|---|---|
| vqa.jsonl | 11,600 | 10,134 | VQAv2 train annotations + complementary pairs (CC BY 4.0); COCO **train2014** images | minimal pairs (same question, two images, different answers): 3,500 yes/no pairs, 800 number pairs, 1,500 other pairs (shared 4-5 option set), >=7/10 annotator agreement |
| count.jsonl | 4,400 | 3,716 (all within the VQA set) | COCO train2014 instances (CC BY 4.0) | how-many (2,600; non-crowd, no tiny instances, <=8), at-least-N (1,000), presence with co-occurrence hard negatives (800) |
| m2w.jsonl | 6,200 | 6,200 | Multimodal-Mind2Web **train** split only (HF card: openrail; Mind2Web CC BY 4.0) | viewport crop around the target, 3-5 letter markers (box+tag or circled letter), "Browser goal ... Next labelled element for action OP?" / "Goal: click X Which labelled marker should be clicked?"; website `budget` excluded (all 12 rebuilt preview M2W items are budget.com, test_task) |
| charts.jsonl | 7,667 | 2,599 | own matplotlib renders | extremes, comparisons, printed values, differences, peaks, trends, grouped bars, pies, 2x2 panels; numeric distractors placed so the gold's rank is uniform |
| docs.jsonl | 5,789 | 1,500 | own PIL renders | invoices/receipts/POs/quotes, forms (checkboxes, signatures), multi-year financial tables (change/sum/value), some "scanned" |
| tabfact.jsonl | 2,484 | 1,271 | TabFact train tables only (MIT; tables Wikipedia CC BY-SA), rendered by us | "is this statement supported by the table" yes/no |
| text replay | 3,000 sampled | - | v19's own committed training rows (data/synthetic/generated_v16, generated_v18, adequacy_gen; Apache-2.0) | protects text behaviour |
| image-removed copies | ~15% of image rows with <=9 options | - | derived | weight 0, KL-only to the frozen torso's text-only reading |

sha256: charts 1c915218…, count 09a27833…, docs eceece7a…, m2w 37905fb5…, tabfact 7db5268e…,
vqa 02afc86f… (`/root/data`, not committed; rebuild is deterministic from the seeds).

GQA, PlotQA and AITW were not used (time): PlotQA's role is taken by our own chart renders.

### Dedupe evidence (`research/image-track/dedupe.py`)

- COCO: every COCO training image is `COCO_train2014_*` (asserted); POPE's 500
  adversarial images are val2014; id overlap **0**.
- pHash (64-bit) of all 21,704 training images vs 1,828 evaluation images (NaturalBench
  shard 0 groups 0-599 = 1,200 images, all 500 POPE adversarial images, 128 preview
  items). First pass found 6 Mind2Web crops within Hamming <=6 of three preview M2W items
  (same budget.com pages); the website was then excluded at build time. Second pass:
  **0** training images within Hamming 6; closest pairs: one at 8 (COCO 288654 vs NB
  group 177 image 1 — inspected: a snowy park vs an insect on leather, unrelated), 15 at 10.
  Report: `/root/data/dedupe_report.json` (copied to the run artifacts).
- No NaturalBench, POPE, MMBench or Image JevBench item, image or question is in any
  training file; the unlisted v0.2 preview page was not used.

## Training runs

(entries below)

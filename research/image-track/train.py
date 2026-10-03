"""Continue Strands Decider (v19) on images: LoRA + pointer head trainable, vision frozen.

    python research/image-track/train.py research/image-track/configs/<run>.yaml

Starts from a published text checkpoint loaded with `VisionDeciderModel.load` (its
adapter mapped onto the multimodal torso), so step 0 is exactly v19 `--vision`.

Loss per row (summed over a step's rows, divided by the step's row count):
  CE(gold)                               labelled rows (score rows: ordinal-smoothed)
  + kl_frozen_weight * KL(frozen || student)   labelled rows: the untouched torso's
                                          option-number reading, with the image in the prompt
  + kl_only_weight   * KL(frozen || student)   image-removed copies (weight 0): no label,
                                          only the frozen reading of the text-only prompt
Training forwards at temperature 1 (raw logits); per-kind image temperatures are fitted
afterwards on held-out items (temps.py). Option order is reshuffled every time a row is
drawn and noul/choice share one renderer with the server (prompting.render_question,
vision.render_image_state). Images are resized as at inference (fit_image with the
run's long side / pixel budget). Text replay rows (v19's own committed training rows)
protect text behaviour.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any

import torch
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from torch.utils.data import DataLoader, Dataset

from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import Example
from strands_decider.prompting import render_question
from strands_decider.vision import (
    VisionDeciderModel,
    expand_image_tokens,
    fit_image,
    image_tokens_for_grid,
    qwen_base,
    render_image_state,
)

V19, V19_REV = "StrandsAgents/strands-decider-2B-hobson-v19", "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
MAX_KL_OPTIONS = 9  # the frozen readout scores the single-token digits 1-9


# ---- data ----------------------------------------------------------------------------


def load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(cfg["seed"])
    root = cfg["data_root"]
    rows: list[dict[str, Any]] = []
    for f in cfg["train_files"]:
        n = cfg.get("caps", {}).get(os.path.basename(f))
        part = [json.loads(x) for x in open(os.path.join(root, f), encoding="utf-8") if x.strip()]
        if n and len(part) > n:
            rng.shuffle(part)
            part = part[:n]
        rows += part
    drop = set()
    if cfg.get("dedupe_drop"):
        drop = {x.strip() for x in open(cfg["dedupe_drop"]) if x.strip()}
        before = len(rows)
        rows = [r for r in rows if not any(i in drop for i in r.get("images", []))]
        print(f"[image-train] dedupe dropped {before - len(rows)} rows", flush=True)
    for f in cfg.get("text_replay_files", []):
        part = [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        for r in part:
            r.update(images=[], ablation=False, source="text_replay",
                     source_id=f"text-{hashlib.md5(json.dumps(r['state'])[:2000].encode()).hexdigest()[:12]}")
        rows += part
    # Text replay is sampled to `text_replay_n` rows in all.
    text = [r for r in rows if r["source"] == "text_replay"]
    img = [r for r in rows if r["source"] != "text_replay"]
    rng.shuffle(text)
    text = text[: cfg.get("text_replay_n", len(text))]
    abl = []
    frac = cfg.get("ablation_fraction", 0.0)
    if frac > 0:
        for r in img:
            if len(r["options"]) <= MAX_KL_OPTIONS and rng.random() < frac:
                abl.append({**r, "images": [], "ablation": True, "weight": 0.0, "pair_id": None,
                            "task": r["task"] + "/ablation"})
    return img + abl + text


def split(rows: list[dict[str, Any]], frac: float, seed: int) -> tuple[list, list]:
    """By source_id: a row, its pair half and its ablation copy stay on one side."""
    def key(r):  # pairs share pair_id; tie their sources together
        return r.get("pair_id") or r["source_id"]
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    ks = sorted(groups)
    random.Random(seed).shuffle(ks)
    nval = int(len(ks) * frac)
    val_keys = set(ks[:nval])
    # ablation copies carry pair_id None; route them by source_id of their original
    src_val = {r["source_id"] for k in val_keys for r in groups[k]}
    tr, va = [], []
    for r in rows:
        (va if (key(r) in val_keys or r["source_id"] in src_val) else tr).append(r)
    return tr, [r for r in va if not r.get("ablation")]


class Batches(Dataset):
    """Each item is one collated micro-batch (index lists fixed up front)."""

    def __init__(self, rows, batches, collate):
        self.rows, self.batches, self.collate = rows, batches, collate

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.collate([self.rows[j] for j in self.batches[i]], seed=i)


class ImageCollator:
    def __init__(self, tok, proc, cfg: dict[str, Any], train: bool):
        self.tok, self.proc, self.cfg, self.train = tok, proc, cfg, train
        self.ccfg = CollatorConfig(max_length=cfg["max_length"], num_slots=24, head_type="pointer",
                                   shuffle_options=cfg.get("shuffle_options", True),
                                   ordinal_smoothing=cfg.get("ordinal_smoothing", 0.1), seed=0)

    def __call__(self, rows: list[dict[str, Any]], seed: int = 0) -> dict[str, torch.Tensor] | None:
        from PIL import Image

        base = SystemOneCollator(self.tok, self.ccfg, train=self.train)
        base.rng = random.Random(self.cfg["seed"] * 1_000_003 + seed * 7 + self.cfg.get("epoch", 0))
        imgs, per_row = [], []
        for r in rows:
            k = []
            for p in r.get("images", []):
                im = Image.open(os.path.join(self.cfg["data_root"], p))
                imgs.append(fit_image(im, self.cfg["image_long_side"], self.cfg["image_max_pixels"]))
                k.append(1)
            per_row.append(len(k))
        mm: dict[str, torch.Tensor] = {}
        counts: list[int] = []
        if imgs:
            out = self.proc(images=imgs, return_tensors="pt")
            counts = image_tokens_for_grid(out["image_grid_thw"].tolist(), self.proc.merge_size)
            mm = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
        ids, opts, labels, dists, n_slots, weights, abl = [], [], [], [], [], [], []
        c = 0
        for r, n_img in zip(rows, per_row):
            ex = Example.from_dict(r)
            order = base._option_order(ex)
            instr = base._instruction(ex)
            rq = render_question(ex.to_question(instr), option_order=order)
            prompt = expand_image_tokens(render_image_state(ex.state, n_img), counts[c:c + n_img]) + rq.text
            c += n_img
            enc = self.tok(prompt, return_offsets_mapping=True)
            if len(enc["input_ids"]) > self.cfg["max_length"]:
                raise ValueError(f"row over max_length ({len(enc['input_ids'])}): {r['task']}")
            ids.append(enc["input_ids"])
            opts.append(base.option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text)))
            labels.append(base._remap_label(ex.label, order))
            d = base._target_distribution(ex, ex.label, ex.n_options, order)
            dists.append(d)
            n_slots.append(ex.n_options)
            weights.append(0.0 if r.get("ablation") else 1.0)
            abl.append(bool(r.get("ablation")))
        pad = self.tok.pad_token_id
        L = max(len(x) for x in ids)
        width = max(len(o) for o in opts)
        out = {
            "input_ids": torch.tensor([x + [pad] * (L - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids]),
            "opt_idx": torch.tensor([o + [-1] * (width - len(o)) for o in opts]),
            "labels": torch.tensor(labels),
            "n_slots": torch.tensor(n_slots),
            "weights": torch.tensor(weights),
            "ablation": torch.tensor(abl),
        }
        if any(d is not None for d in dists):
            out["label_dist"] = torch.stack([
                d if d is not None else torch.nn.functional.one_hot(torch.tensor(l), 24).float()
                for d, l in zip(dists, labels)])[:, :width]
        out.update(mm)
        return out


def plan_batches(rows, lengths, max_rows: int, max_tokens: int, seed: int) -> list[list[int]]:
    """Length-grouped micro-batches under a token budget, in shuffled order."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    out = []
    for m in range(0, len(idx), 2000):  # sort within mega-chunks: grouped, still mixed
        chunk = sorted(idx[m:m + 2000], key=lambda i: lengths[i])
        cur: list[int] = []
        for i in chunk:
            longest = max([lengths[j] for j in cur] + [lengths[i]])
            if cur and (len(cur) >= max_rows or longest * (len(cur) + 1) > max_tokens):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def est_length(r: dict[str, Any], img_tokens: int) -> int:
    text = json.dumps(r["state"]) + r["instructions"] + json.dumps(r["options"])
    return len(text) // 3 + 60 + img_tokens * len(r.get("images", []))


# ---- training ------------------------------------------------------------------------


def row_losses(model, batch, kl_w: float, kl_only_w: float):
    mm = {k: batch[k] for k in ("pixel_values", "image_grid_thw") if k in batch}
    out = model(batch["input_ids"], batch["attention_mask"], batch["n_slots"], opt_idx=batch["opt_idx"],
                temperature=1.0, **mm)
    lp = out["log_probs"]
    safe = torch.where(torch.isinf(lp), torch.zeros_like(lp), lp)
    if "label_dist" in batch:
        ce = -(batch["label_dist"] * safe).sum(-1)
    else:
        ce = -safe.gather(1, batch["labels"].view(-1, 1)).squeeze(1)
    ce = ce * batch["weights"]
    kl = torch.zeros_like(ce)
    if kl_w > 0 or kl_only_w > 0:
        qwen_base(model.torso).rope_deltas = None
        ref, elig = model.frozen_slot_log_probs(batch["input_ids"], batch["attention_mask"], batch["n_slots"], **mm)
        if ref.numel():
            ref = ref[:, : lp.shape[-1]]
            valid = torch.isfinite(ref) & torch.isfinite(lp)
            p = ref.exp().masked_fill(~valid, 0.0)
            diff = (ref - lp).masked_fill(~valid, 0.0)
            per = (p * diff).sum(-1) * elig.float()
            coef = torch.where(batch["ablation"], torch.full_like(per, kl_only_w), torch.full_like(per, kl_w))
            kl = coef * per
    qwen_base(model.torso).rope_deltas = None
    return ce, kl, lp


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    n = correct = 0
    nll = 0.0
    by: dict[str, list] = {}
    for batch in loader:
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, _, lp = row_losses(model, batch, 0.0, 0.0)
        ok = (lp.argmax(-1) == batch["labels"]).float()
        n += len(ok)
        correct += float(ok.sum())
        nll += float(ce.sum())
    model.train()
    return {"val_acc": correct / max(1, n), "val_nll": nll / max(1, n), "val_n": n}


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda"
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    from transformers import Qwen2VLImageProcessorPil

    ckpt = cfg.get("init_checkpoint", V19)
    if ckpt == V19:
        ckpt = snapshot_download(V19, revision=V19_REV)
    model = VisionDeciderModel.load(ckpt)
    v19_temps = dict(model.config.temperature_by_kind)
    model.to(device)
    for p in model.head.parameters():
        p.requires_grad_(True)
    for p in qwen_base(model.torso).visual.parameters():
        p.requires_grad_(False)
    if cfg.get("gradient_checkpointing", True):
        qwen_base(model.torso).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[image-train] trainable params {trainable:,}", flush=True)
    model.train()

    rows = load_rows(cfg)
    tr, va = split(rows, cfg.get("val_fraction", 0.03), seed)
    manifest = {"n_train": len(tr), "n_val": len(va),
                "by_source_train": {}, "ablation_train": sum(bool(r.get("ablation")) for r in tr)}
    for r in tr:
        manifest["by_source_train"][r["source"]] = manifest["by_source_train"].get(r["source"], 0) + 1
    print(f"[image-train] {json.dumps(manifest)}", flush=True)
    json.dump(manifest, open(os.path.join(out_dir, "data_manifest.json"), "w"), indent=2)

    proc = Qwen2VLImageProcessorPil.from_pretrained(model.config.base_model)
    img_tok = cfg.get("est_image_tokens", 400)
    lens_tr = [est_length(r, img_tok) for r in tr]
    lens_va = [est_length(r, img_tok) for r in va]
    collate = ImageCollator(model.tokenizer, proc, cfg, train=True)
    vcollate = ImageCollator(model.tokenizer, proc, cfg, train=False)
    mb_rows, mb_tok = cfg.get("micro_rows", 16), cfg.get("micro_tokens", 12000)
    batches = []
    for ep in range(cfg.get("epochs", 1)):
        batches += [(ep, b) for b in plan_batches(tr, lens_tr, mb_rows, mb_tok, seed * 100 + ep)]
    rows_per_step = cfg.get("rows_per_step", 32)
    total_rows = sum(len(b) for _, b in batches)
    total_steps = max(1, total_rows // rows_per_step)
    print(f"[image-train] {len(batches)} micro-batches, ~{total_steps} steps", flush=True)
    ds = Batches(tr, [b for _, b in batches], collate)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=cfg.get("workers", 8),
                        prefetch_factor=4, persistent_workers=False)
    vds = Batches(va, plan_batches(va, lens_va, mb_rows, mb_tok, 1)[: cfg.get("val_batches", 60)], vcollate)
    vloader = DataLoader(vds, batch_size=None, num_workers=4)

    head_p = [p for p in model.head.parameters() if p.requires_grad]
    lora_p = [p for n, p in model.torso.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for p in head_p if p.ndim > 1], "lr": cfg["head_lr"], "weight_decay": cfg.get("weight_decay", 0.01)},
        {"params": [p for p in head_p if p.ndim <= 1], "lr": cfg["head_lr"], "weight_decay": 0.0},
        {"params": lora_p, "lr": cfg["lr"], "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8)
    warm = max(1, int(total_steps * cfg.get("warmup_ratio", 0.03)))

    def lam(s):
        if s < warm:
            return s / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    hist = []
    t0 = time.time()
    step, rows_in_step = 0, 0
    acc_ce = acc_kl = 0.0
    log_rows = 0
    pending: list[torch.Tensor] = []
    kl_w, kl_only_w = cfg.get("kl_frozen_weight", 0.3), cfg.get("kl_only_weight", 1.0)
    if cfg.get("eval_at_start", False):
        hist.append({"step": 0, **evaluate(model, vloader, device)})
        print(f"[image-train] eval @0 {hist[-1]}", flush=True)
    for batch in loader:
        if batch is None:
            continue
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, kl, _ = row_losses(model, batch, kl_w, kl_only_w)
        n = len(ce)
        # Sum now, normalise by the step's row count at the step: rows weigh equally.
        (ce.sum() + kl.sum()).div(rows_per_step).backward()
        acc_ce += float(ce.sum())
        acc_kl += float(kl.sum())
        rows_in_step += n
        log_rows += n
        if rows_in_step >= rows_per_step:
            if rows_in_step != rows_per_step:  # rescale to the true row count
                for p in head_p + lora_p:
                    if p.grad is not None:
                        p.grad.mul_(rows_per_step / rows_in_step)
            torch.nn.utils.clip_grad_norm_(head_p + lora_p, cfg.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rows_in_step = 0
            if step % cfg.get("log_every", 20) == 0:
                e = {"step": step, "ce": acc_ce / log_rows, "kl": acc_kl / log_rows,
                     "lr": sched.get_last_lr()[-1], "elapsed_s": round(time.time() - t0),
                     "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                hist.append(e)
                print(f"[image-train] {json.dumps(e)} / {total_steps}", flush=True)
                acc_ce = acc_kl = 0.0
                log_rows = 0
            if cfg.get("eval_every") and step % cfg["eval_every"] == 0:
                hist.append({"step": step, **evaluate(model, vloader, device)})
                print(f"[image-train] eval {hist[-1]}", flush=True)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
    hist.append({"step": step, **evaluate(model, vloader, device), "final": True})
    print(f"[image-train] final {hist[-1]} wall {time.time() - t0:.0f}s", flush=True)
    # Text temperatures stay v19's; image temperatures are fitted afterwards (temps.py).
    model.config.temperature_by_kind = v19_temps
    model.config.image_temperature_by_kind = {}
    model.save_pretrained(out_dir)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    json.dump(cfg, open(os.path.join(out_dir, "train_config.json"), "w"), indent=2)
    print(f"[image-train] saved {out_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

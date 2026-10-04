"""Image training on a Gemma 4 torso (src/strands_decider/vision_train.py), on the tiny
random-weight Gemma 4 of tests/tiny_bases.py; nothing is downloaded: collated rows carry
Gemma's image tensors, and three steps of training lower the loss and save a checkpoint
whose adapter loads back onto exactly the text checkpoint's modules.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tiny_bases import save_gemma4, save_gemma4_checkpoint  # noqa: E402

from strands_decider.vision import VisionDeciderModel  # noqa: E402
from strands_decider.vision_train import (  # noqa: E402
    ImageCollator,
    VisionTrainConfig,
    row_losses,
    train,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "image"))
import common  # noqa: E402


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    base = save_gemma4(str(tmp_path_factory.mktemp("tiny-gemma4")), images=True)
    return save_gemma4_checkpoint(base, str(tmp_path_factory.mktemp("gemma-text-ckpt")))


def _lora_names(model) -> set[str]:
    return {n.split(".lora_")[0] for n, _ in model.torso.named_parameters() if "lora_A" in n}


def _image_rows(root, n: int = 6) -> None:
    rng = random.Random(0)
    os.makedirs(root / "img", exist_ok=True)
    rows = []
    for k in range(n):
        name = f"img/{k}.png"
        Image.new("RGB", (64 + 16 * k, 48), (40 * k % 255, 90, 160)).save(root / name)
        kw = dict(source="synthetic", images=[name], source_id=f"img-{k}")
        rows.append(common.yesno(f"Is image {k} blue?", k % 2 == 0, rng, task="blue", pair_id=f"pair-{k // 2}", **kw))
        rows.append(common.row("choice", "Which colour?", [["red", ""], ["green", ""], ["blue", ""]], 2,
                               task="colour", **kw))
    common.write(str(root / "rows.jsonl"), rows)


def _cfg(root, **over) -> VisionTrainConfig:
    return VisionTrainConfig(data_root=str(root), train_files=["rows.jsonl"], image_long_side=0,
                             image_max_pixels=4096, workers=0, **over)


def test_collated_rows_carry_gemma_images(ckpt, tmp_path):
    _image_rows(tmp_path)
    model = VisionDeciderModel.load(ckpt, trainable=True)
    cfg = _cfg(tmp_path, ablation_fraction=1.0)
    from strands_decider.vision_train import load_rows

    rows = [r for r in load_rows(cfg) if r["task"] in ("colour", "colour/ablation")][:4]
    batch = ImageCollator(model.tokenizer, model.image_prompt(), cfg, train=True)(rows, index=1)
    assert {"pixel_values", "image_position_ids"} <= set(batch) and "image_grid_thw" not in batch
    image_id = model.torso.config.image_token_id
    n_images = sum(bool(r["images"]) for r in rows)
    assert batch["pixel_values"].shape[0] == n_images
    for i, r in enumerate(rows):
        assert bool((batch["input_ids"][i] == image_id).any()) == bool(r["images"])
    ce, kl, lp = row_losses(model, batch, 0.3, 1.0)
    assert lp.shape == (len(rows), 3) and torch.isfinite(ce).all() and torch.isfinite(kl).all()
    copy = batch["ablation"]
    assert torch.all(ce[copy] == 0) and torch.all(kl[copy] > 0)


def test_three_steps_of_image_training_reduce_the_loss_and_round_trip(ckpt, tmp_path):
    _image_rows(tmp_path)
    out = train(_cfg(tmp_path, init_from=ckpt, init_revision=None, rows_per_step=12, micro_rows=12,
                     micro_tokens=100_000, epochs=3, max_steps=3, lr=1e-2, head_lr=1e-2, log_every=1,
                     eval_every=0, val_fraction=0.0, shuffle_options=False, kl_frozen_weight=0.0,
                     output_dir=str(tmp_path / "out")))
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        ce = [h["ce"] for h in json.load(fh) if "ce" in h]
    assert len(ce) == 3 and ce[-1] < ce[0]
    saved = load_file(os.path.join(out, "lora", "adapter_model.safetensors"))
    assert saved and all(".language_model." in k for k in saved)  # multimodal keys, loaded as they are
    before = load_file(os.path.join(ckpt, "lora", "adapter_model.safetensors"))
    assert any(not torch.equal(v, saved[k.replace("base_model.model.", "base_model.model.language_model.", 1)])
               for k, v in before.items())
    assert _lora_names(VisionDeciderModel.load(out)) == _lora_names(VisionDeciderModel.load(ckpt))

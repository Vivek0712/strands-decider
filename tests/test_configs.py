"""Every shipped config loads, and the v19 route's configs cannot drift from their records.

No test read a config before this one, so a renamed or removed TrainConfig field failed
only on the GPU host. configs/train.yaml is what the released v19 weights trained from;
its header claims it equals configs/experiments/v19.yaml except three keys, and
configs/train-parent.yaml claims the same of v14.yaml. These tests hold both claims,
comparing the parsed configs, not the text.

tests/fixtures/v19-p5-run1/ holds the hobson_config.json and train_config.json that the
v19 checkpoint saved: hyperparameters and relative paths only. A saved train_config.json
is a reproduction path (`strands-decider train --config <ckpt>/train_config.json`).
"""

import dataclasses
import glob
import os

import pytest

from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.train import TrainConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# configs/vision/ holds image-training configs (strands_decider.vision_train), the rest TrainConfig's.
VISION = sorted(glob.glob(os.path.join(ROOT, "configs", "vision", "*.yaml")))
CONFIGS = sorted(set(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True)) - set(VISION))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "v19-p5-run1")


def _load(*parts):
    return dataclasses.asdict(TrainConfig.from_yaml(os.path.join(ROOT, *parts)))


def _differing_keys(a, b):
    return {k for k in a if a[k] != b[k]}


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_every_config_loads(path):
    # from_yaml rejects unknown keys, so this also fails when a field is removed or renamed.
    TrainConfig.from_yaml(path)


@pytest.mark.parametrize("path", VISION, ids=[os.path.relpath(p, ROOT) for p in VISION])
def test_every_vision_config_loads(path):
    from strands_decider.vision_train import VisionTrainConfig

    VisionTrainConfig.from_yaml(path)


def test_train_yaml_is_v19_except_three_keys():
    train, v19 = _load("configs", "train.yaml"), _load("configs", "experiments", "v19.yaml")
    assert _differing_keys(train, v19) == {"teacher_file", "output_dir", "max_length"}


def test_train_parent_yaml_is_v14_except_two_keys():
    parent, v14 = _load("configs", "train-parent.yaml"), _load("configs", "experiments", "v14.yaml")
    assert _differing_keys(parent, v14) == {"teacher_file", "output_dir"}
    # The file that `training/recipe.sh teacher` writes and run_recipe.sh check_teacher checksums.
    assert parent["teacher_file"] == "data/teacher_multistep_v14_train.jsonl"


def test_v19_seed1_is_v19_except_seed_and_output_dir():
    seed1, v19 = _load("configs", "experiments", "v19-seed1.yaml"), _load("configs", "experiments", "v19.yaml")
    assert _differing_keys(seed1, v19) == {"seed", "output_dir"}


def test_v19_saved_configs_load():
    StrandsDeciderConfig.from_json(os.path.join(FIXTURE, "hobson_config.json"))
    saved = dataclasses.asdict(TrainConfig.from_yaml(os.path.join(FIXTURE, "train_config.json")))
    # The released run was configs/train.yaml under training/run_recipe.sh FAST=1, which
    # turns gradient checkpointing off and precomputes the frozen-KL reference (speed only).
    assert _differing_keys(saved, _load("configs", "train.yaml")) == {"gradient_checkpointing", "precompute_frozen_kl"}


@pytest.mark.parametrize("seed", [1, 2])
def test_v19_yn27b_seeds_differ_only_in_seed_and_output_dir(seed):
    rep, ref = (_load("configs", "experiments", f"v19-yn27b{s}.yaml") for s in (f"-seed{seed}", ""))
    assert _differing_keys(rep, ref) == {"seed", "output_dir"}
    assert rep["seed"] == seed


@pytest.mark.parametrize("seed", [1, 2])
def test_v19_images_seeds_differ_only_in_seed_and_output_dir(seed):
    from strands_decider.vision_train import VisionTrainConfig

    rep, ref = (dataclasses.asdict(VisionTrainConfig.from_yaml(os.path.join(ROOT, "configs", "vision", f"v19-images{s}.yaml")))
                for s in (f"-seed{seed}", ""))
    assert _differing_keys(rep, ref) == {"seed", "output_dir"}
    assert rep["seed"] == seed


BAKEOFF = ("qwen35-2b", "minicpm5-2b", "gemma4-e2b")


def test_bakeoff_configs_differ_only_in_the_base():
    ref = _load("configs", "experiments", "bakeoff", "qwen35-2b.yaml")
    for name in BAKEOFF[1:]:
        other = _load("configs", "experiments", "bakeoff", f"{name}.yaml")
        assert _differing_keys(other, ref) == {"base_model", "base_model_revision", "lora_targets", "output_dir"}
    # From the raw base, on v19-yn27b's data and losses
    yn = _load("configs", "experiments", "v19-yn27b.yaml")
    assert ref["continue_from"] is None and ref["init_from"] is None
    assert ref["base_model"] == yn["base_model"] and ref["lora_targets"] == yn["lora_targets"]
    same = {"train_files", "kl_frozen_weight", "kl_frozen_skip_kinds", "head_type", "pointer_dim", "lora_r",
            "lora_alpha", "lr", "head_lr", "micro_batch_size", "grad_accum", "max_length", "seed"}
    assert all(ref[k] == yn[k] for k in same)
    assert ref["teacher_file"] == "data/teacher_t4.jsonl"


def _vision(name: str) -> dict:
    from strands_decider.vision_train import VisionTrainConfig

    return dataclasses.asdict(VisionTrainConfig.from_yaml(os.path.join(ROOT, "configs", "vision", f"{name}.yaml")))


def test_gemma_images_is_v19_images_except_its_start_and_token_estimate():
    """The Gemma image arm is variant A on the same data: image-removed copies, 400k pixels."""
    gemma, v19 = _vision("gemma4-e2b-images"), _vision("v19-images")
    assert _differing_keys(gemma, v19) == {"init_from", "init_revision", "est_image_tokens", "output_dir"}
    assert gemma["ablation_fraction"] == 0.15 and gemma["image_max_pixels"] == 400_000
    # starts from the Gemma bake-off's text checkpoint unless given another at run time
    assert gemma["init_from"] == _load("configs", "experiments", "bakeoff", "gemma4-e2b.yaml")["output_dir"]
    assert gemma["init_revision"] is None


@pytest.mark.parametrize("seed", [1, 2])
def test_gemma_images_seeds_differ_only_in_seed_and_output_dir(seed):
    rep, ref = _vision(f"gemma4-e2b-images-seed{seed}"), _vision("gemma4-e2b-images")
    assert _differing_keys(rep, ref) == {"seed", "output_dir"}
    assert rep["seed"] == seed and rep["output_dir"] == f"{ref['output_dir']}-seed{seed}"


def test_vision_train_overrides_init_from_at_run_time():
    from strands_decider.vision_train import VisionTrainConfig, with_overrides

    cfg = VisionTrainConfig.from_yaml(os.path.join(ROOT, "configs", "vision", "gemma4-e2b-images.yaml"))
    over = with_overrides(cfg, ["init_from=/ckpt/gemma-text", "max_steps=3"])
    assert (over.init_from, over.max_steps, over.seed) == ("/ckpt/gemma-text", 3, 0)
    with pytest.raises(SystemExit):
        with_overrides(cfg, ["init_frm=/x"])

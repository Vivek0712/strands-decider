"""The next-model stage configs (configs/next/) cannot drift from their design, and the step
benchmark (training/bench_steps.py) reads the trainer's log.

Stage D is a 2^3 factorial: every pair of cells must differ in exactly the keys of the arms
whose switches differ (and the output directory), so a main effect is that arm and nothing
else. Stage B's pilots are the base bake-off's recipe on another base; Stage E is the
control cell on the pilot's checkpoint for a full epoch. (test_configs.py already loads every
config and holds the Stage E seed replicate.)
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import re
import sys

import pytest

from strands_decider.train import TrainConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_KEYS = {
    0: {"train_files"},                                          # A1
    1: {"score_rps_weight", "ordinal_smoothing"},                # A2
    2: {"teacher_file", "pool_teacher_files", "pool_alpha"},     # A3
}
CELLS = ["".join(b) for b in itertools.product("01", repeat=3)]


def _load(*parts):
    return dataclasses.asdict(TrainConfig.from_yaml(os.path.join(ROOT, *parts)))


def _cell(name):
    return _load("configs", "next", "stage-d", f"cell-{name}.yaml")


def _diff(a, b):
    return {k for k in a if a[k] != b[k]}


def test_stage_d_has_exactly_the_eight_cells():
    assert sorted(os.listdir(os.path.join(ROOT, "configs", "next", "stage-d"))) == [f"cell-{c}.yaml" for c in CELLS]


@pytest.mark.parametrize("a, b", list(itertools.combinations(CELLS, 2)))
def test_cells_differ_only_in_their_arm_switches(a, b):
    want = {"output_dir"}.union(*(ARM_KEYS[j] for j in range(3) if a[j] != b[j]))
    assert _diff(_cell(a), _cell(b)) == want


def test_the_control_cell_is_v20_continued_from_its_soup_for_1500_steps():
    c0, v20 = _cell("000"), _load("configs", "experiments", "strands-decider-2B-hobson-v20.yaml")
    assert _diff(c0, v20) == {"continue_from", "max_steps", "output_dir"}
    assert (c0["continue_from"], c0["max_steps"]) == ("checkpoints/strands-decider-2B-hobson-v20", 1500)
    for name in CELLS:
        c = _cell(name)
        assert (c["seed"], c["init_seed"], c["max_steps"]) == (0, 0, 1500)
        assert c["output_dir"] == f"checkpoints/next-d-{name}"


def test_each_arm_switches_what_it_says():
    off, a1, a2, a3 = _cell("000"), _cell("100"), _cell("010"), _cell("001")
    # A1 appends the families after v20's files, so every teacher index still holds
    assert a1["train_files"] == [*off["train_files"], "data/families_a1.jsonl"]
    assert (a2["score_rps_weight"], a2["ordinal_smoothing"]) == (1.0, 0.0)
    assert (off["score_rps_weight"], off["ordinal_smoothing"]) == (0.0, 0.1)
    assert a3["pool_teacher_files"] == ["data/teacher_pool_qwen35-27b.jsonl"] and a3["pool_alpha"] == 0.5
    assert a3["teacher_file"] == "data/replay_v14_multistep.jsonl"  # v20's file without the 27B part
    assert off["pool_teacher_files"] == [] and off["teacher_file"] == "data/teacher_yn_qwen35-27b.jsonl"


def test_stage_b_pilots_are_the_bakeoff_recipe_on_another_base():
    ref = _load("configs", "experiments", "bakeoff", "qwen35-2b.yaml")
    pilots = {n: _load("configs", "next", "stage-b", f"pilot-{n}.yaml") for n in ("4b-base", "4b-instruct", "2b-base")}
    assert _diff(pilots["2b-base"], ref) == {"max_steps", "output_dir"}
    for n in ("4b-base", "4b-instruct"):
        assert _diff(pilots[n], ref) == {"base_model", "base_revision", "max_steps", "output_dir"}
        assert re.fullmatch(r"[0-9a-f]{40}", pilots[n]["base_revision"])
    assert pilots["4b-base"]["base_model"] == "Qwen/Qwen3.5-4B-Base"
    assert pilots["4b-instruct"]["base_model"] == "Qwen/Qwen3.5-4B"
    assert {p["max_steps"] for p in pilots.values()} == {1500}


def test_stage_e_is_the_control_cell_on_the_pilot_for_a_full_epoch():
    e, c0 = _load("configs", "next", "stage-e", "4b.yaml"), _cell("000")
    assert _diff(e, c0) == {"continue_from", "base_model", "max_steps", "output_dir"}
    assert (e["continue_from"], e["max_steps"]) == ("checkpoints/next-b-pilot-4b", 3738)


def test_no_config_names_a_host_path_or_bucket():
    for path in [os.path.join(d, f) for d, _, fs in os.walk(os.path.join(ROOT, "configs", "next")) for f in fs]:
        text = open(path, encoding="utf-8").read()
        assert not re.search(r"s3://|/home/|/Users/|/root/|arn:aws", text), path


# ---- training/bench_steps.py -----------------------------------------------------------------


@pytest.fixture
def bench():
    sys.path.insert(0, os.path.join(ROOT, "training"))
    try:
        import bench_steps

        yield bench_steps
    finally:
        sys.path.remove(os.path.join(ROOT, "training"))


def test_bench_reads_the_last_logged_rate(bench):
    log = ("[strands-decider] frozen-KL reference for 1,200 micro-batches in 41 s\n"
           "[strands-decider] epoch 0 step 20/100 loss 0.5 lr 1e-4 0.41 step/s peak_mem 12.3GiB\n"
           "[strands-decider] epoch 0 step 100/100 loss 0.4 lr 1e-5 0.57 step/s peak_mem 12.4GiB\n")
    assert bench.parse(log) == {"last_step": 100, "step_per_s": 0.57, "peak_mem_gib": 12.4,
                                "frozen_kl_precompute_s": 41}


def test_bench_trains_a_config_for_a_few_steps(bench, tmp_path, capsys):
    transformers = pytest.importorskip("transformers")
    if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
        pytest.skip("the tiny Qwen3.5 needs transformers >= 5.18")
    from tiny_qwen35 import save_base, save_checkpoint

    from strands_decider.data.format import Example, write_jsonl

    ckpt = save_checkpoint(save_base(str(tmp_path / "base")), str(tmp_path / "ckpt"))
    rows = [Example("noul", f"Order {k} late.", "Late?", [["false", ""], ["true", ""]], k % 2) for k in range(8)]
    write_jsonl(str(tmp_path / "t.jsonl"), rows)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"train_files: ['{tmp_path / 't.jsonl'}']\ncontinue_from: '{ckpt}'\nhead_type: pointer\n"
                   "max_length: 256\nmicro_batch_size: 2\ngrad_accum: 1\nval_fraction: 0.25\n")
    bench.main(["--config", str(cfg), "--steps", "2", "--json", str(tmp_path / "r.json")])
    import json

    r = json.loads((tmp_path / "r.json").read_text())
    assert r["steps"] == 2 and r["last_step"] == 2 and r["step_per_s"] > 0 and set(r["kernels"]) == {
        "causal_conv1d", "flash_linear_attention"}
    assert not os.path.exists(r["output_dir"])  # the scratch checkpoint is removed

"""Arms A2 and A3 of the next-model plan: the ranked probability score for score rows
(TrainConfig.score_rps_weight) and the fixed two-teacher loss (data/teacher_pool.py,
TrainConfig.pool_teacher_files / pool_alpha). The arithmetic on its own, then the loss on
the tiny Qwen3.5 of tests/tiny_qwen35.py, then a few CPU training steps (nothing is
downloaded). Both are off by default, and off they leave the loss exactly as it was."""

from __future__ import annotations

import json
import math
import os
import random
import re

import pytest
import torch
from unseen_v2 import score as unseen_score

from strands_decider.data import teacher_pool as TP
from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import Example, write_jsonl
from strands_decider.modeling import ranked_probability_score
from strands_decider.train import TrainConfig

YN = [["false", "no"], ["true", "yes"]]


# ---- recalibration and the pool -----------------------------------------------------------


def test_recalibrate():
    p = [0.7, 0.2, 0.1, 0.0]
    assert TP.recalibrate(p, 1.0) == pytest.approx(p)
    flat, sharp = TP.recalibrate(p, 2.0), TP.recalibrate(p, 0.5)
    assert max(flat) < 0.7 < max(sharp) and flat[3] == 0 and sum(flat) == pytest.approx(1)


def test_fit_recovers_the_temperature_of_an_overconfident_teacher():
    rng = random.Random(0)
    rows = []
    for _ in range(4000):
        q = rng.uniform(0.05, 0.95)
        gold = int(rng.random() < q)
        rows.append((TP.recalibrate([1 - q, q], 0.5), gold))  # reports q sharpened by T = 0.5
    t = TP.fit_temperature(rows)
    assert t == pytest.approx(2.0, rel=0.15)  # undoing it takes T = 2
    assert TP.nll(rows, t) < TP.nll(rows) and TP.ece(rows, t) < TP.ece(rows)


def _corpus(tmp_path, n=40):
    exs = [Example("noul", f"s{i}", "q?", YN, i % 2, task="t") for i in range(n)]
    exs += [Example("choice", f"c{i}", "which?", [["a", ""], ["b", ""], ["c", ""]], i % 3, task="u")
            for i in range(n)]
    write_jsonl(str(tmp_path / "a.jsonl"), exs[:n])
    write_jsonl(str(tmp_path / "b.jsonl"), exs[n:])
    return [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")], exs


def _teacher(path, exs, sure=0.9, skip=()):
    with open(path, "w", encoding="utf-8") as fh:
        for i, ex in enumerate(exs):
            if i in skip:
                continue
            k = ex.n_options
            p = [(1 - sure) / (k - 1)] * k
            p[ex.label if i % 4 else (ex.label + 1) % k] = sure  # wrong on every fourth row
            fh.write(json.dumps({"i": i, "probs": p}) + "\n")
    return str(path)


def test_fit_writes_temperatures_per_kind_from_a_held_out_split(tmp_path):
    files, exs = _corpus(tmp_path)
    t = _teacher(tmp_path / "t.jsonl", exs)
    TP.main(["fit", "--teacher", t, "--train-files", *files, "--heldout-fraction", "0.5"])
    cal = json.loads((tmp_path / "t.jsonl.calibration.json").read_text())
    assert set(cal["temperature_by_kind"]) == {"noul", "choice"} and cal["train_files"] == files
    rep = cal["heldout_report_before_after"]
    assert sum(r["heldout_rows"] for r in rep.values()) == 40
    assert all(r["nll"][1] <= r["nll"][0] for r in rep.values())  # recalibration never hurts held-out NLL
    assert cal["temperature_by_kind"]["noul"] > 1  # 90% sure but right 75%: flatten
    with pytest.raises(ValueError, match="does not match"):
        TP.fit(t, files[:1])


def test_the_pool_averages_recalibrated_teachers_and_checks_their_files(tmp_path):
    files, exs = _corpus(tmp_path)
    t1 = _teacher(tmp_path / "t1.jsonl", exs, 0.9)
    t2 = _teacher(tmp_path / "t2.jsonl", exs, 0.6, skip={0, 1})
    with pytest.raises(FileNotFoundError, match="fit"):
        TP.load_pool([t1], exs, files)
    for t, temps in ((t1, {"noul": 2.0, "choice": 1.5}), (t2, {"noul": 1.0})):
        with open(TP.calibration_path(t), "w", encoding="utf-8") as fh:
            json.dump({"train_files": files, "temperature_by_kind": temps}, fh)
    pool = TP.load_pool([t1, t2], exs, [*files, "data/a1.jsonl"])  # A1 appended after: fine
    assert len(pool) == len(exs)
    p1, p2 = TP.recalibrate([0.1, 0.9], 2.0), [0.4, 0.6]
    assert pool[5] == pytest.approx([(a + b) / 2 for a, b in zip(p1, p2, strict=True)])
    assert pool[1] == pytest.approx(p1)  # only the first teacher labelled row 1
    q1 = TP.recalibrate([0.05, 0.9, 0.05], 1.5)  # the first teacher's choice temperature
    assert pool[41] == pytest.approx([(a + b) / 2 for a, b in zip(q1, [0.2, 0.6, 0.2], strict=True)])
    with pytest.raises(ValueError, match="does not begin"):
        TP.load_pool([t1], exs, files[::-1])


# ---- RPS ------------------------------------------------------------------------------------


def test_rps_matches_the_evaluation_definition_and_ignores_padding_and_reversal():
    rng = random.Random(1)
    for _ in range(20):
        k, width = rng.randint(2, 6), 7
        p = [rng.random() + 0.01 for _ in range(k)]
        p = [x / sum(p) for x in p]
        g = rng.randrange(k)
        lp = torch.full((1, width), -math.inf)
        lp[0, :k] = torch.tensor(p).log()
        got = ranked_probability_score(lp, torch.tensor([g]), torch.tensor([k]))
        assert float(got) == pytest.approx(unseen_score.rps(p, g), abs=1e-6)
        rev = torch.full((1, width), -math.inf)
        rev[0, :k] = torch.tensor(p[::-1]).log()
        assert float(ranked_probability_score(rev, torch.tensor([k - 1 - g]), torch.tensor([k]))) == \
            pytest.approx(float(got), abs=1e-6)


# ---- the collator ---------------------------------------------------------------------------


class _Tok:
    pad_token_id = 0

    def __call__(self, texts, **kw):
        n = max(len(t) for t in texts)
        ids = torch.tensor([[1] * len(t) + [0] * (n - len(t)) for t in texts])
        return {"input_ids": ids, "attention_mask": (ids > 0).long()}


def test_the_pool_follows_the_option_shuffle():
    ex = Example("choice", "s", "which?", [["a", ""], ["b", ""], ["c", ""]], 2, task="t")
    ex.pool = [0.1, 0.2, 0.7]
    coll = SystemOneCollator(_Tok(), CollatorConfig(num_slots=4, seed=3), train=True)
    for _ in range(5):
        b = coll([ex, Example("noul", "s", "q?", YN, 1)])
        lab = int(b["labels"][0])
        assert float(b["pool"][0, lab]) == pytest.approx(0.7)  # the gold option's pool mass moves with it
        assert b["has_pool"].tolist() == [True, False] and float(b["pool"][1].sum()) == 0


# ---- the loss on a real (tiny) model, and training -------------------------------------------

transformers = pytest.importorskip("transformers")
NEW_ENOUGH = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) >= (5, 18)
needs_tiny = pytest.mark.skipif(not NEW_ENOUGH, reason="the tiny Qwen3.5 needs transformers >= 5.18")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from tiny_qwen35 import save_base, save_checkpoint

    base = save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))
    return save_checkpoint(base, str(tmp_path_factory.mktemp("ckpt")))


def _batch(model):
    rows = [Example("noul", "The parcel arrived late.", "Was it late?", YN, 1, task="t"),
            Example("choice", "Refund please.", "Which team?", [["billing", ""], ["sales", ""], ["ops", ""]], 0),
            Example("score", "Pretty good.", "How good?", [["0", "bad"], ["1", "ok"], ["2", "good"], ["3", "great"]], 2)]
    rows[0].pool = [0.3, 0.7]
    rows[1].pool = [0.5, 0.25, 0.25]
    cfg = CollatorConfig(num_slots=24, head_type="pointer", ordinal_smoothing=0.0, max_length=256)
    b = SystemOneCollator(model.tokenizer, cfg, train=False)(rows)
    return b


def _forward(model, b, **kw):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], n_slots=b["n_slots"],
                 opt_idx=b["opt_idx"], labels=b["labels"], **kw)


@needs_tiny
def test_the_two_terms_are_exactly_the_documented_loss(ckpt):
    from strands_decider.modeling import StrandsDeciderModel

    model = StrandsDeciderModel.load(ckpt)
    model.eval()
    b = _batch(model)
    with torch.no_grad():
        plain = _forward(model, b)
        off = _forward(model, b, pool=b["pool"], has_pool=b["has_pool"], pool_alpha=0.0,
                       rps_rows=b["kind_id"] == 2, rps_weight=0.0)
        assert torch.equal(plain["loss"], off["loss"])  # off is the old loss, exactly
        lp = plain["log_probs"]
        nll = -lp[torch.arange(3), b["labels"]]
        a = 0.4
        mixed = _forward(model, b, pool=b["pool"], has_pool=b["has_pool"], pool_alpha=a)
        kl = []
        for i in range(2):
            q = b["pool"][i, : int(b["n_slots"][i])]
            kl.append(float((q * (q.log() - lp[i, : len(q)])).sum()))
        want = ((1 - a) * nll[0] + a * kl[0] + (1 - a) * nll[1] + a * kl[1] + nll[2]) / 3
        assert float(mixed["loss"]) == pytest.approx(float(want), rel=1e-5)
        rps = _forward(model, b, rps_rows=b["kind_id"] == 2, rps_weight=2.0)
        p = lp[2, :4].exp().tolist()
        want = (nll.sum() + 2.0 * unseen_score.rps(p, int(b["labels"][2]))) / 3
        assert float(rps["loss"]) == pytest.approx(float(want), rel=1e-5)


def _rows(n):
    out = []
    for k in range(n):
        out.append(Example("noul", f"Order {k} shipped late.", "Was it late?", YN, 1, task="late"))
        out.append(Example("score", f"Review {k}: fine.", "Rating?", [["0", "bad"], ["1", "ok"], ["2", "good"]],
                           k % 3, task="stars"))
    return out


@needs_tiny
def test_training_with_both_arms_runs_and_records_them(ckpt, tmp_path):
    from strands_decider.train import train

    write_jsonl(str(tmp_path / "train.jsonl"), _rows(4))
    write_jsonl(str(tmp_path / "val.jsonl"), _rows(1))
    files = [str(tmp_path / "train.jsonl")]
    t = _teacher(tmp_path / "t.jsonl", _rows(4)[::2], 0.8)  # only the yes/no rows are labelled ...
    lines = [json.loads(x) for x in open(t, encoding="utf-8")]
    with open(t, "w", encoding="utf-8") as fh:  # ... at their indices in the train file
        fh.writelines(json.dumps({"i": 2 * r["i"], "probs": r["probs"]}) + "\n" for r in lines)
    TP.main(["fit", "--teacher", t, "--train-files", *files, "--heldout-fraction", "0.5"])
    cfg = TrainConfig(continue_from=ckpt, train_files=files, val_files=[str(tmp_path / "val.jsonl")],
                      head_type="pointer", max_length=256, micro_batch_size=2, grad_accum=1, max_steps=2,
                      lr=1e-2, head_lr=1e-2, log_every=1, eval_every=0, output_dir=str(tmp_path / "out"),
                      pool_teacher_files=[t], pool_alpha=0.5, score_rps_weight=1.0, ordinal_smoothing=0.0)
    out = train(cfg)
    saved = json.load(open(os.path.join(out, "train_config.json"), encoding="utf-8"))
    assert saved["pool_alpha"] == 0.5 and saved["score_rps_weight"] == 1.0
    model_cfg = json.load(open(os.path.join(out, "strands_decider_config.json"), encoding="utf-8"))
    assert model_cfg["ordinal_smoothing"] == 0.0  # the run's smoothing, not the parent's 0.1

    # A row may not carry both teacher targets.
    with open(tmp_path / "old.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"i": 0, "probs": [0.5, 0.5]}) + "\n")
    with pytest.raises(ValueError, match="both a teacher_file"):
        train(TrainConfig(**{**cfg.__dict__, "teacher_file": str(tmp_path / "old.jsonl"), "teacher_weight": 1.0}))


@pytest.mark.parametrize("over, match", [({"pool_alpha": 0.5}, "go together"),
                                         ({"pool_teacher_files": ["x"]}, "go together"),
                                         ({"pool_teacher_files": ["x"], "pool_alpha": 1.5}, r"\[0, 1\]"),
                                         ({"score_rps_weight": -1.0}, ">= 0")])
def test_bad_arm_settings_are_refused(over, match):
    from strands_decider.train import train

    with pytest.raises(ValueError, match=match):
        train(TrainConfig(continue_from="unused", **over))


def test_both_arms_default_off():
    cfg = TrainConfig()
    assert (cfg.pool_teacher_files, cfg.pool_alpha, cfg.score_rps_weight) == ([], 0.0, 0.0)

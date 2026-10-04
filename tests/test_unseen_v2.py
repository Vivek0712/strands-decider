"""unseen-v2 (evaluation/unseen_v2/): the generator families' gold, the build's counts,
splits and exclusions, the v2 measures against v1's grading, the paired bootstrap, the
Stage D factorial decision and the Stage C rank check. Offline: synthetic public rows,
fake deciders; the chess family needs python-chess (the `unseen` extra) and is skipped
without it."""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import sys

import numpy as np
import pytest
from unseen import build as B1
from unseen import score as S1
from unseen_v2 import adapters as A
from unseen_v2 import build as B
from unseen_v2 import factorial as F
from unseen_v2 import generators as G
from unseen_v2 import rank_check as RC
from unseen_v2 import score as S
from unseen_v2 import spearman as SP

from strands_decider.data.format import Example

CHEAP = ("arithmetic", "calendar", "seating")
YN = [["false", ""], ["true", ""]]


# ---- generators -----------------------------------------------------------------------


@pytest.mark.parametrize("family", CHEAP)
@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_generated_rows_are_examples_and_reproducible(family, kind):
    rows = G.draw(family, kind, 30, random.Random(3))
    assert rows == G.draw(family, kind, 30, random.Random(3))
    assert rows != G.draw(family, kind, 30, random.Random(4))
    for r in rows:
        ex = Example.from_dict(r)
        assert ex.kind == kind and r["task"] == f"unseen/{family}" and r["template"] in G.TEMPLATES[family][kind]
        assert len({o[0] for o in r["options"]}) == len(r["options"])  # options are named by text
    assert len({repr((r["state"], r["instructions"], r["options"])) for r in rows}) == len(rows)


def test_labels_are_balanced_within_a_template():
    rows = G.draw("seating", "noul", 400, random.Random(0))
    for t in ("seat", "left_of"):
        labels = [r["label"] for r in rows if r["template"] == t]
        assert 0.4 < sum(labels) / len(labels) < 0.6
    levels = [r["label"] for r in G.draw("calendar", "score", 500, random.Random(0))]
    assert all(60 < levels.count(k) < 140 for k in range(5))


def _value(expr: str) -> int:
    assert re.fullmatch(r"[\d+\-*/() ]+", expr)
    return round(eval(expr))  # digits and operators only, checked above


def test_arithmetic_gold_recomputed():
    rng = random.Random(1)
    for r in G.draw("arithmetic", "noul", 60, rng) + G.draw("arithmetic", "choice", 60, rng):
        if r["template"] != "expression":
            continue
        value = _value(r["state"].splitlines()[0].removeprefix("Expression: "))
        if r["kind"] == "noul":
            claim = int(re.search(r"equal (-?\d+)", r["instructions"]).group(1))
            assert r["label"] == int(claim == value)
        else:
            assert int(r["options"][r["label"]][0]) == value
    for r in G.draw("arithmetic", "score", 60, rng):
        a, b = (_value(line.split(" = ")[1]) for line in r["state"].splitlines()[:2])
        assert r["label"] == sum(a / b > c for c in G.RATIO_CUTS)


def _zeller(d: dt.date) -> int:
    """Monday = 0, by Zeller's congruence (independent of datetime's own weekday)."""
    y, m = (d.year - 1, d.month + 12) if d.month < 3 else (d.year, d.month)
    h = (d.day + 13 * (m + 1) // 5 + y + y // 4 - y // 100 + y // 400) % 7  # 0 = Saturday
    return (h + 5) % 7


def _parse(text: str) -> dt.date:
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(text)


def test_calendar_gold_recomputed():
    rng = random.Random(2)
    for r in G.draw("calendar", "choice", 80, rng):
        if r["template"] == "weekday":
            d = _parse(r["state"].removeprefix("Date: ").split(" (")[0])
            assert r["label"] == _zeller(d)
    for r in G.draw("calendar", "score", 80, rng):
        a, b = (_parse(line.split(": ", 1)[1]) for line in r["state"].splitlines())
        assert r["label"] == sum(abs((b - a).days) >= c for c in G.GAP_CUTS)


def test_seating_puzzles_have_one_arrangement_and_the_gold_follows_it():
    rng = random.Random(5)
    for _ in range(20):
        names, seat, state = G._puzzle(rng)
        assert sorted(seat.values()) == [1, 2, 3, 4, 5]
        assert all(name in state for name in names)
    for r in G.draw("seating", "score", 40, rng):
        assert 0 <= r["label"] <= 4


def test_chess_gold_recomputed_from_the_position():
    chess = pytest.importorskip("chess")
    rng = random.Random(0)
    for r in G.draw("chess", "noul", 24, rng):
        board = chess.Board(r["state"].split("FEN): ")[1].splitlines()[0])
        if r["template"] == "check":
            assert r["label"] == int(board.is_check())
    for r in G.draw("chess", "score", 24, rng):
        board = chess.Board(r["state"].split("FEN): ")[1].splitlines()[0])
        if r["template"] == "material":
            diff = sum(G.PIECE_VALUES[p.piece_type] * (1 if p.color else -1) for p in board.piece_map().values())
            assert r["label"] == (0 if diff <= -3 else 1 if diff < 0 else 2 if diff == 0 else 3 if diff < 3 else 4)


# ---- build ----------------------------------------------------------------------------


def _public():
    return {
        "strategyqa": [{"question": f"q{i}", "answer": i % 2 == 0} for i in range(50)],
        "commonsenseqa": [{"question": f"c{i}", "answerKey": "A",
                           "choices": {"label": ["A", "B"], "text": [f"x{i}", f"y{i}"]}} for i in range(50)],
        "arc_challenge": [{"question": f"a{i}", "answerKey": "B",
                           "choices": {"label": ["A", "B", "C"], "text": ["1", "2", "3"]}} for i in range(50)],
        "stsb": [{"sentence1": "s", "sentence2": "t", "score": i / 50} for i in range(50)],
        "csqa2": [json.dumps({"question": f"claim {i}", "answer": "yes" if i % 3 else "no"}) + "\n" for i in range(50)],
        "hellaswag": [{"activity_label": "Cooking", "ctx": f"ctx {i}", "endings": [f"e{i}{j}" for j in range(4)],
                       "label": str(i % 4)} for i in range(50)] + [
                      {"activity_label": "x", "ctx": "dup", "endings": ["a", "a", "b", "c"], "label": "0"}],
    }


SMALL = {"noul": {"strategyqa": 10, "ruletaker_d5": 10, "csqa2": 10, **{g: 10 for g in CHEAP}},
         "choice": {"commonsenseqa": 10, "arc_challenge": 10, "hellaswag": 10, **{g: 10 for g in CHEAP}},
         "score": {"stsb": 10, **{g: 10 for g in CHEAP}}}


def _held(tmp_path):
    held = tmp_path / "holdout.jsonl"
    held.write_text("".join(json.dumps({"task": "ruletaker_d5", "kind": "noul", "state": str(i),
                                        "instructions": "q", "options": YN, "label": i % 2}) + "\n" for i in range(80)))
    return str(held)


def test_build_counts_splits_and_determinism(tmp_path):
    rows = B.build(_held(tmp_path), counts=SMALL, public=_public())
    tasks = [r["task"] for r in rows]
    assert {t: tasks.count(t) for t in set(tasks)} == {
        f"unseen/{f}": 10 * (3 if f in CHEAP else 1) for kind in SMALL.values() for f in kind}
    for t in set(tasks):
        dev = sum(r["split"] == "dev" for r in rows if r["task"] == t)
        assert dev == round(B.DEV_FRACTION * tasks.count(t))
    assert B.build(_held(tmp_path), counts=SMALL, public=_public()) == rows
    assert B.build(_held(tmp_path), seed=1, counts=SMALL, public=_public()) != rows
    for r in rows:
        Example.from_dict(r)
    hs = [r for r in rows if r["task"] == "unseen/hellaswag"]
    assert all(r["state"] != "Activity: x\nContext: dup" for r in hs)  # a repeated ending keeps it out


def test_the_default_counts_are_three_thousand_per_type():
    assert {k: sum(v.values()) for k, v in B.COUNTS.items()} == {"noul": 3000, "choice": 3000, "score": 3000}


def training_sources() -> tuple[set[str], set[str]]:
    """The recipe names and Hugging Face ids data/sources.md lists with the role `train`."""
    text = open(B.__file__.replace("evaluation/unseen_v2/build.py", "data/sources.md"), encoding="utf-8").read()
    names, ids = set(), set()
    for line in text.split("## Short-task datasets")[1].splitlines():
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) == 4 and cells[3] == "train":
            names |= {n.strip(" `") for n in cells[0].split(",")}
            ids.add(cells[1].split(" ")[0].strip("`"))
    return names, ids


def test_no_excluded_family_and_no_training_source_in_unseen_v2():
    families = {f for kind in B.COUNTS.values() for f in kind}
    assert not {f for f in families if any(x in f for x in B.EXCLUDED)}
    names, ids = training_sources()
    assert {"boolq", "mnli_entail", "ruletaker_d0"} <= names and "google/boolq" in ids  # the right table
    assert "ruletaker_d5" not in names  # held out of training by design
    assert not families & names
    assert not {repo for repo, *_ in B.FILES.values()} & ids
    assert not set(B1.REVISIONS) & ids


# ---- score ----------------------------------------------------------------------------


def _p(i, kind, gold, probs, task="t", split="dev"):
    return {"id": i, "task": task, "kind": kind, "gold": gold, "probs": probs, "split": split}


def _random_rows(seed, n=90):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        kind = ("noul", "choice", "score")[i % 3]
        k = {"noul": 2, "choice": 4, "score": 5}[kind]
        p = [rng.random() + 1e-3 for _ in range(k)]
        rows.append(_p(i, kind, rng.randrange(k), [x / sum(p) for x in p], task=f"f{i % 4}"))
    return rows


def test_v2_measures_reproduce_v1_grading():
    rows = _random_rows(0)
    m = S.measures(S.features(rows).sum(axis=0))
    per, intelligence = S1.competence(rows)
    assert float(m["intelligence"]) == pytest.approx(intelligence)
    for k, v in per.items():
        assert float(m[f"competence_{k}"]) == pytest.approx(v)
    assert float(m["ece_top"]) == pytest.approx(S1.ece(rows))
    yn = [r for r in rows if r["kind"] == "noul"]
    assert float(m["band_mass"]) == pytest.approx(sum(map(S1.in_band, yn)) / len(yn))


def test_rps_is_normalised_and_proper_on_examples():
    assert S.rps([1, 0, 0, 0, 0], 0) == 0.0
    assert S.rps([0, 0, 0, 0, 1], 0) == pytest.approx(1.0)  # every cumulative level wrong
    assert S.rps([0, 1, 0, 0, 0], 0) == pytest.approx(1 / 4)  # one level off, K = 5
    assert S.rps([0.5, 0, 0, 0, 0.5], 2) == pytest.approx(0.25)
    q = [0.1, 0.6, 0.3]
    honest = sum(qi * S.rps(q, g) for g, qi in enumerate(q))
    assert all(sum(qi * S.rps(p, g) for g, qi in enumerate(q)) > honest
               for p in ([0, 1, 0], [0.2, 0.5, 0.3], [1 / 3] * 3))


def test_paired_bootstrap_sees_a_real_difference_and_not_a_null_one():
    base = [_p(i, "noul", 1, [0.5, 0.5]) for i in range(60)] + [_p(60 + i, "choice", 0, [0.6, 0.4]) for i in range(60)]
    sure = [_p(i, "noul", 1, [0.1, 0.9]) for i in range(60)] + base[60:]
    vs = S.paired_bootstrap([sure], [base], resamples=300)
    assert vs["competence_noul"]["diff"] == pytest.approx(200.0) and vs["competence_noul"]["excludes_0"]
    assert vs["band_mass"]["diff"] == pytest.approx(-1.0)
    null = S.paired_bootstrap([base, base], [base], resamples=300, resample_runs=True)
    assert all(v["diff"] == 0 and v["ci95"] == [0, 0] for v in null.values())


def test_runs_must_answer_the_same_items():
    with pytest.raises(ValueError, match="same items"):
        S.paired_bootstrap([_random_rows(0)], [_random_rows(0, n=60)], resamples=10)


def test_split_selection():
    rows = [_p(0, "noul", 1, [0.1, 0.9], split="dev"), _p(1, "noul", 1, [0.1, 0.9], split="test")]
    assert [r["id"] for r in S.select(rows, "test")] == [1] and len(S.select(rows, "all")) == 2
    with pytest.raises(ValueError, match="no split"):
        S.select([{k: v for k, v in rows[0].items() if k != "split"}], "dev")


# ---- Stage D factorial ------------------------------------------------------------------


def _write(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d)


def test_factorial_adopts_only_the_arm_that_moves_its_metric(tmp_path):
    def cell(a1, a3):
        out = []
        for i in range(120):
            kind = ("noul", "choice", "score")[i % 3]
            if kind == "noul":
                sure = (i // 3) % 10 < (6 if a3 else 3)  # A3 makes more yes/no answers decisive
                out.append(_p(i, kind, 1, [0.1, 0.9] if sure else [0.5, 0.5]))
            elif kind == "choice":
                out.append(_p(i, kind, 0, [0.7, 0.3] if (i // 3) % 10 < 5 + a1 else [0.3, 0.7]))
            else:
                out.append(_p(i, kind, 0, [0.6, 0.2, 0.2]))
        return out

    specs = [f"{a}{b}{c}={_write(tmp_path, f'c{a}{b}{c}', cell(int(a), int(c)))}"
             for a in "01" for b in "01" for c in "01"]
    F.main([*(x for s in specs for x in ("--cell", s)), "--resamples", "300", "--json", str(tmp_path / "f.json")])
    report = json.loads((tmp_path / "f.json").read_text())
    d = report["decision"]
    assert d["A3"]["adopt"] and not d["A2"]["adopt"]
    assert d["A2"]["effect"]["diff"] == 0
    with pytest.raises(SystemExit, match="missing"):
        F.cells(["000=x"])


def test_a_drop_elsewhere_blocks_adoption():
    eff = {"intelligence": {"diff": 3.0, "ci95": [1, 5], "excludes_0": True},
           "competence_noul": {"diff": -1.5, "ci95": [-3, 0], "excludes_0": False},
           "competence_choice": {"diff": 0.0, "ci95": [0, 0], "excludes_0": False},
           "competence_score": {"diff": 9.0, "ci95": [5, 12], "excludes_0": True}}
    d = F.decide({"A1": eff, "A2": eff, "A3": eff}, None)
    assert not d["A1"]["adopt"] and d["A1"]["drops_over_1"] == {"competence_noul": -1.5}
    public = {f"{a}{b}{c}": 30.0 - 3 * int(b) for a in "01" for b in "01" for c in "01"}
    flat = {k: {"diff": 0.5, "ci95": [0.1, 1], "excludes_0": True} for k in eff}
    d = F.decide({"A1": flat, "A2": flat, "A3": flat}, public)
    assert d["A1"]["adopt"] and not d["A2"]["adopt"] and "public_intelligence" in d["A2"]["drops_over_1"]


# ---- Stage C rank check ------------------------------------------------------------------


def test_spearman_handles_ties_and_matches_the_definition():
    assert float(SP.spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40]))) == pytest.approx(1.0)
    assert float(SP.spearman(np.array([4.0, 3, 2, 1]), np.array([10.0, 20, 30, 40]))) == pytest.approx(-1.0)
    assert SP.ranks(np.array([[5.0, 1, 5, 3]])).tolist() == [[3.5, 1, 3.5, 2]]


def test_rank_check_runs_a_system_and_spearman_reads_it(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    exs = [Example("noul", f"s{i}", "q?", YN, i % 2, task="unseen/x") for i in range(30)]
    rows.write_text("".join(json.dumps({**json.loads(e.to_json()), "split": "dev"}) + "\n" for e in exs))
    systems = tmp_path / "systems.json"
    systems.write_text(json.dumps({"systems": [{"name": f"s{k}", "adapter": "python:fake_sys:make"} for k in range(3)]}))
    (tmp_path / "fake_sys.py").write_text(
        "import os\n"
        "def make():\n"
        "    skill = int(os.environ['SD_RANK_REPO'][-1])\n"
        "    return lambda state, q: {'noul': 0.9 if int(state[1:]) % 3 < skill else 0.5}\n")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setenv("SD_RANK_REPO", "")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="--revision"):
        RC.main(["run", "--rows", str(rows), "--out", str(out), "--systems", str(systems),
                 "--repo", f"s0={tmp_path}/r0"])
    RC.main(["run", "--rows", str(rows), "--out", str(out), "--systems", str(systems),
             *(x for k in range(3) for x in ("--repo", f"s{k}={tmp_path}/r{k}", "--revision", f"s{k}=abc")),
             "--only", "s0", "s1", "s2"])
    assert (out / "DO_NOT_TRAIN.md").exists()
    assert json.loads((out / "s1" / "system.json").read_text())["revision"] == "abc"
    csv = tmp_path / "official.csv"
    csv.write_text("system,official\ns0,10\ns1,20\ns2,30\nmissing,40\n")
    SP.main(["--official", str(csv), "--dir", str(out), "--resamples", "100", "--json", str(tmp_path / "r.json")])
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["n"] == 3 and r["missing_runs"] == ["missing"] and r["rho"] == pytest.approx(1.0) and r["passes"]


def test_adapters_normalise_answer_shapes():
    yn = {"type": "noul", "criteria": {"false": None, "true": None}}
    assert A.normalise({"noul": 0.7}, yn) == {"noul": 0.7}
    assert A.normalise({"probabilities": {"yes": 0.2, "no": 0.8}}, yn) == {"noul": 0.2}
    ch = {"type": "choice", "criteria": {"red": None, "blue": None}}
    assert A.normalise({"probs": {"blue": 3, "red": 1}}, ch) == {"probabilities": {"red": 0.25, "blue": 0.75}}
    sc = {"type": "score", "criteria": ["a", "b", "c"]}
    assert A.normalise({"distribution": [0.2, 0.2, 0.6]}, sc)["probabilities"]["2"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        A.normalise({"distribution": [1.0]}, sc)


def test_stage_e_calibration_mix_is_half_in_family_half_unseen_per_kind():
    from types import SimpleNamespace as NS

    from unseen_v2 import calibrate_mixed as CM

    unseen = [NS(kind=k, i=i) for k in ("noul", "choice") for i in range(50)]
    fam = {"a": [NS(kind="noul", i=i) for i in range(30)] + [NS(kind="score", i=i) for i in range(9)],
           "b": [NS(kind="noul", i=i) for i in range(40)] + [NS(kind="choice", i=i) for i in range(12)]}
    ours, theirs = CM.mix(fam, unseen, per_kind=40)
    for kind in ("noul", "choice", "score"):
        assert sum(e.kind == kind for e in ours) == sum(e.kind == kind for e in theirs)
    assert sum(e.kind == "noul" for e in ours) == 40 and sum(e.kind == "choice" for e in ours) == 12
    assert not any(e.kind == "score" for e in ours)  # no unseen score rows to match them
    assert CM.mix(fam, unseen, per_kind=40) == (ours, theirs)

"""The A1 training families (strands_decider.data.families): reproducible, balanced rows of
all three types in several formats, gold that replays from the rendered state, and no
overlap with any unseen-v2 family (evaluation/unseen_v2/), which they must never leak into."""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import pytest
from unseen_v2 import build as UB
from unseen_v2 import generators as UG

from strands_decider.data import families as A1
from strands_decider.data.families import state as state_family
from strands_decider.data.format import Example


@pytest.fixture(scope="module")
def rows():
    return A1.build(60, seed=0)


def test_every_family_gives_every_type_reproducibly(rows):
    assert Counter((e.task, e.kind) for e in rows) == {
        (f"a1/{f}", k): 60 for f in A1.FAMILIES for k in A1.KINDS}
    assert [e.to_json() for e in A1.build(60, seed=0)] == [e.to_json() for e in rows]
    assert [e.to_json() for e in A1.build(60, seed=1)] != [e.to_json() for e in rows]
    for e in rows:
        Example.from_dict(json.loads(e.to_json()))
        assert e.instruction_variants and len({o[0] for o in e.options}) == len(e.options)


def test_a_family_draws_the_same_rows_whatever_else_is_built():
    alone = A1.build(20, seed=0, families=["lead_rubric"])
    both = A1.build(20, seed=0, families=["state_tracking", "lead_rubric"])
    assert sorted(e.to_json() for e in alone) == sorted(e.to_json() for e in both if e.task == "a1/lead_rubric")


def test_labels_are_balanced_and_formats_vary(rows):
    for f in A1.FAMILIES:
        yn = [e.label for e in rows if e.task == f"a1/{f}" and e.kind == "noul"]
        assert 0.3 < sum(yn) / len(yn) < 0.7
    shapes = Counter(type(e.state).__name__ for e in rows)
    assert shapes["dict"] > 0 and shapes["str"] > 0
    assert any(isinstance(e.state, str) and "{" in e.state and "\n" not in e.state.split(": ", 1)[-1][:40]
               for e in rows)  # compact JSON appears


def test_state_tracking_gold_replays_from_the_rendered_log():
    """Read the rules, start state and log back out of the text, whatever format they were
    drawn in, and replay them."""
    rng = random.Random(7)
    for _ in range(200):
        r = state_family.generate(rng, "choice", "state")
        s = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
        rules = {ev: (set(a.split(" or ")), to) for ev, a, to in
                 re.findall(r'"?(\w+)"?(?::| is) ?"?from ([\w ]+?) to (\w+)', s)}
        log = re.findall(r'"entity": ?"([\w-]+)", ?\s*"event": ?"(\w+)"', s) or [
            (who, ev) for ev, who in re.findall(r"^\d+\. (\w+) ([\w-]+)$", s, re.M)]
        assert rules and log
        start = re.search(r"starts in state '(\w+)'", s).group(1)
        now: dict[str, str] = {}
        for who, ev in log:
            cur = now.get(who, start)
            allowed, to = rules[ev]
            now[who] = to if cur in allowed else cur
        who = re.search(r"is (\S+) in after", r["instructions"]).group(1)
        assert r["options"][r["label"]][0] == now.get(who, start)


# ---- never an unseen-v2 family -------------------------------------------------------------


def test_no_a1_family_is_an_unseen_v2_family():
    unseen = {f for kind in UB.COUNTS.values() for f in kind} | set(UG.FAMILIES)
    assert not set(A1.FAMILIES) & unseen
    for f in A1.FAMILIES:
        assert not any(u in f or f in u for u in unseen)


def test_no_a1_question_is_asked_in_unseen_v2(rows):
    rng = random.Random(0)
    unseen_questions = {r["instructions"] for fam in ("arithmetic", "calendar", "seating")
                        for kind in A1.KINDS for r in UG.draw(fam, kind, 40, rng)}
    a1_questions = {q for e in rows for q in e.all_instructions()}
    assert not unseen_questions & a1_questions
    text = " ".join(json.dumps(e.state) for e in rows).lower()
    for marker in ("fen)", "chess", "seat", "weekday", "leap year", "expression:", "hellaswag", "strategyqa"):
        assert marker not in text
    assert all(e.task.startswith("a1/") for e in rows)


def test_eval_file_never_repeats_a_training_row(tmp_path):
    A1.main(["--out", str(tmp_path / "t.jsonl"), "--eval-out", str(tmp_path / "e.jsonl"), "--n", "40", "--n-eval", "40"])
    train = {line for line in (tmp_path / "t.jsonl").read_text().splitlines()}
    held = (tmp_path / "e.jsonl").read_text().splitlines()
    keys = {A1.key(Example.from_dict(json.loads(x))) for x in train}
    assert held and not any(A1.key(Example.from_dict(json.loads(x))) in keys for x in held)

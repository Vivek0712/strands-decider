"""The local scorers of evaluation/: the JevBench v1.5-rule proxy (jevbench/v15_proxy.py)
and the paired image comparison (vision/compare.py). Standard library and synthetic rows."""
from __future__ import annotations

import pytest
from jevbench import v15_proxy as P
from vision import compare as C


def _yn(task: str, p_yes: float, correct: bool, family: str = "policy") -> dict:
    return {"task_id": task, "family": family, "probs": {"yes": p_yes, "no": 1 - p_yes},
            "predicted": "yes" if p_yes >= 0.5 else "no", "correct": correct, "ordinal_ev": None}


def _choice(task: str, correct: bool, n: int = 4) -> dict:
    return {"task_id": task, "family": "routing", "probs": {f"o{i}": 1 / n for i in range(n)},
            "correct": correct, "ordinal_ev": None}


def _score(task: str, correct: bool) -> dict:
    return {"task_id": task, "family": "rubric", "probs": {str(i): 0.2 for i in range(5)},
            "correct": correct, "ordinal_ev": 2.0}


def test_yes_no_answers_inside_the_band_count_as_wrong():
    assert P.task_score(_yn("a", 0.85, True)) == 1.0
    assert P.task_score(_yn("a", 0.70, True)) == -1.0  # right, but an abstention under v1.5
    assert P.task_score(_yn("a", 0.80, True)) == 1.0  # the band is open: 0.8 is decisive
    assert P.task_score(_yn("a", 0.20, True)) == 1.0
    assert P.task_score(_yn("a", 0.21, True)) == -1.0
    assert P.task_score(_choice("c", True)) == 1.0
    assert P.task_score(_choice("c", False)) == pytest.approx(-1 / 3)  # chance-corrected


def test_proxy_weighs_the_three_types_equally():
    rows = [_yn("a", 0.9, True), _yn("b", 0.5, True), _choice("c", True), _score("d", False)]
    per, intelligence = P.proxy(rows)
    assert per == {"noul": 0.0, "choice": 100.0, "score": -25.0}
    assert intelligence == pytest.approx(25.0)
    m = P.metrics(rows)
    assert (m["yes_no"], m["yes_no_in_band"], m["band_correct"], m["n_correct"]) == (2, 1, 1, 3)
    assert m["band_by_family"] == {"policy": 1}


def test_paired_bootstrap_vs_a_baseline():
    base = [_yn(f"t{i}", 0.5, True) for i in range(20)] + [_choice(f"c{i}", True) for i in range(20)]
    same = P.paired_bootstrap([base], [base], resamples=200)
    assert all(v == {"diff": 0.0, "ci95": [0.0, 0.0]} for v in same.values())
    decisive = [_yn(f"t{i}", 0.9, True) for i in range(20)] + base[20:]
    got = P.paired_bootstrap([decisive, base], [base], resamples=200)
    assert got["yes_no_in_band"]["diff"] == -10.0  # the mean of -20 and 0 over the two seeds
    assert got["intelligence_proxy"]["ci95"][0] > 0
    with pytest.raises(ValueError):
        P.paired_bootstrap([base[:-1]], [base], resamples=10)


def _nb(group: int, rights: list[bool], blind_conf: float = 0.5) -> list[dict]:
    rows = []
    for k, ok in enumerate(rights):
        rows.append({"id": f"nb-{group}-{k}", "bench": "naturalbench", "group": group, "q": k // 2, "i": k % 2,
                     "gold": 0 if ok else 1, "probs": [0.7, 0.3]})
    return rows


def test_image_comparison_counts_groups_and_items():
    # group 2 is incomplete (3 of its 4 answers), so G-Acc leaves it out
    arm = {"naturalbench": _nb(0, [True] * 4) + _nb(1, [True, True, True, False]) + _nb(2, [True] * 3)}
    base = {"naturalbench": _nb(0, [True, False, True, True]) + _nb(1, [True, False, False, False])
            + _nb(2, [False] * 3)}
    got = C.paired_bootstrap([arm], [base], resamples=200)
    assert got["naturalbench_acc"]["diff"] == round(10 / 11 - 4 / 11, 4)
    assert got["naturalbench_g_acc"]["diff"] == pytest.approx(0.5)  # group 0 all right only in the arm
    same = C.paired_bootstrap([arm], [arm], resamples=50)
    assert same["naturalbench_g_acc"] == {"diff": 0.0, "ci95": [0.0, 0.0]}

"""Adapters for running public board systems on unseen-v2 (Stage C), evaluation only.

evaluation/unseen/run.py runs any decider through `--adapter python:MODULE:NAME`, where
MODULE.NAME() returns `ask(state, question) -> answer` in System One JSON. The factories
here are those NAMEs for the systems whose published code we have run before; each reads the
system's own download (a directory, pinned to a revision when it was fetched) from an
environment variable, so no host path lives in this file:

  decider        Mapika/decider-2b and Mapika/decider-4b (tag v2): `decider.infer.Decider`,
                 `system_one(state, {"q": question})`; SD_RANK_REPO = the download
  flymy          FlyMy decision-2b-preview: `model.load().decide(state, question)`, run from
                 the download directory; SD_RANK_REPO = the download. Its permission covers
                 running it and publishing measurements, nothing else: never train on it.

A system served behind the System One API needs no adapter: `--adapter http://HOST:PORT`.
Any other system gets its factory here once its own inference code has been read at the
pinned revision (systems.json lists what is known).

`normalise` maps the answer shapes seen so far (System One `noul`/`probabilities`, or a
`probs`/`distribution` dict or list) onto a System One answer, so run.py reads them alike.

NEVER use these systems' answers as training data or teacher targets: Stage C measures rank
agreement only, and evaluation/unseen/run.py keeps only probabilities and gold, no rows.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

Ask = Callable[[Any, dict[str, Any]], dict[str, Any]]


def _repo() -> str:
    repo = os.environ.get("SD_RANK_REPO")
    if not repo:
        raise SystemExit("set SD_RANK_REPO to the system's download directory (pinned revision)")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def keys(question: dict[str, Any]) -> list[str]:
    """The answer keys a question's probabilities use: option names, or level indices."""
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def normalise(answer: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """`answer` as a System One answer: `noul` = P(yes) for yes/no, else `probabilities`
    keyed by option name or level index, renormalised over the question's options."""
    if question["type"] == "noul":
        if isinstance(answer.get("noul"), (int, float)):
            return {"noul": float(answer["noul"])}
        p = answer.get("probabilities") or answer.get("probs") or {}
        yes = p.get("true", p.get("yes"))
        if yes is None:
            raise ValueError(f"no P(yes) in {sorted(answer)}")
        return {"noul": float(yes)}
    ks = keys(question)
    p = answer.get("probabilities") or answer.get("probs") or answer.get("distribution")
    if p is None:
        raise ValueError(f"no probabilities in {sorted(answer)}")
    vals = [float(x) for x in p] if isinstance(p, list) else [float(p.get(k, 0.0)) for k in ks]
    if len(vals) != len(ks):
        raise ValueError(f"{len(vals)} probabilities for {len(ks)} options")
    total = sum(vals) or 1.0
    return {"probabilities": {k: v / total for k, v in zip(ks, vals, strict=True)}}


def decider() -> Ask:
    from decider.infer import Decider  # the system's own package, from SD_RANK_REPO

    d = Decider(_repo())
    return lambda state, q: normalise(d.system_one(state, {"q": q})["answers"]["q"], q)


def flymy() -> Ask:
    repo = _repo()
    os.chdir(repo)  # its loader reads files relative to the download
    import model  # the system's own module, from SD_RANK_REPO

    m = model.load()
    return lambda state, q: normalise(m.decide(state, q), q)

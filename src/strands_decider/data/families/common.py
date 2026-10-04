"""What every A1 family shares: rendering a state in several formats, the row shape, and
drawing rows with balanced gold labels."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..format import Example

Row = dict[str, Any]
# (rng, kind, template) -> a row, or None when this draw does not fit the template
Generator = Callable[[random.Random, str, str], "Row | None"]

YES_NO = [["false", "No, it does not hold."], ["true", "Yes, it holds."]]
FORMATS = ("json", "compact_json", "key_value", "prose")


def render(rng: random.Random, title: str, record: Mapping[str, Any]) -> Any:
    """`record` in one of four formats, drawn at random: a JSON object (the prompt renders
    it indented), compact JSON on one line, `key: value` lines, or sentences. The same
    facts in different dress, so a head cannot key on one layout."""
    style = rng.choice(FORMATS)
    if style == "json":
        return {title: dict(record)}
    if style == "compact_json":
        return f"{title}: " + json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    if style == "key_value":
        return f"{title}\n" + "\n".join(f"{k}: {_text(v)}" for k, v in record.items())
    return f"{title}. " + " ".join(_sentence(k, v) for k, v in record.items())


def _sentence(key: str, value: Any) -> str:
    """One fact as a sentence: "The order_age_days is 12." or, when the value is itself a
    sentence (a rule), "condition_1: Claims must be ...". Keys stay verbatim, since rules
    name the fields they test."""
    text = _text(value)
    if text.endswith("."):
        return f"{key}: {text}"
    return f"The {key} is {text}."


def _text(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return ", ".join(map(_text, v)) if v else "none"
    return str(v)


def join(*parts: Any) -> Any:
    """Several rendered parts as one state: dicts merged when every part is a dict,
    otherwise text separated by blank lines (dicts as indented JSON)."""
    if all(isinstance(p, dict) for p in parts):
        out: dict[str, Any] = {}
        for p in parts:
            out.update(p)
        return out
    return "\n\n".join(p if isinstance(p, str) else json.dumps(p, indent=2, ensure_ascii=False) for p in parts)


def row(kind: str, state: Any, instructions: Sequence[str], options: list[list[str]], label: int,
        family: str) -> Row:
    """A row in the corpus format; the first phrasing is canonical, the rest are variants."""
    return {"kind": kind, "state": state, "instructions": instructions[0], "options": options, "label": label,
            "task": f"a1/{family}", "instruction_variants": list(instructions[1:])}


def levels(descriptions: Sequence[str]) -> list[list[str]]:
    return [[str(i), d] for i, d in enumerate(descriptions)]


def named(names: Sequence[str]) -> list[list[str]]:
    return [[n, n] for n in names]


def draw(gen: Generator, templates: Mapping[str, int], kind: str, n: int, rng: random.Random,
         max_tries: int = 2000) -> list[Example]:
    """`n` rows of one family and kind, labels balanced within each template: a template
    is picked by weight, then a target label position uniformly, and the generator is asked
    until it yields that label. A row repeated exactly is drawn again."""
    names, weights = zip(*templates.items(), strict=True)
    out: list[Example] = []
    seen: set[str] = set()
    while len(out) < n:
        template = rng.choices(names, weights)[0]
        target = rng.random()
        for _ in range(max_tries):
            r = gen(rng, kind, template)
            if r is None:
                continue
            key = repr((r["state"], r["instructions"], r["options"]))
            if key not in seen and r["label"] == int(target * len(r["options"])):
                seen.add(key)
                out.append(Example.from_dict(r))
                break
        else:
            raise RuntimeError(f"{kind}/{template}: no row with the target label in {max_tries} draws")
    return out

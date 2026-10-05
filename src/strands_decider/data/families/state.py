"""State tracking: entities move through a small state machine, driven by an event log.

Each domain has states and allowed transitions; an event not allowed from an entity's
current state is rejected and changes nothing (the rules say so in the state). The gold
follows by replaying the log. Asked as: is X in state S (yes/no), which state is X in
(choice), how many of the four entities end in state S (score).
"""

from __future__ import annotations

import random
from typing import Any

from .common import YES_NO, Row, join, levels, named, render, row

FAMILY = "state_tracking"
# domain -> (entity noun, id prefix, start state, {event: ({from states}, to state)})
DOMAINS: dict[str, tuple[str, str, str, dict[str, tuple[set[str], str]]]] = {
    "orders": ("order", "ORD", "created", {
        "pay": ({"created"}, "paid"), "ship": ({"paid"}, "shipped"), "deliver": ({"shipped"}, "delivered"),
        "cancel": ({"created", "paid"}, "cancelled"), "return": ({"delivered"}, "returned")}),
    "tickets": ("ticket", "TCK", "open", {
        "assign": ({"open"}, "in_progress"), "resolve": ({"in_progress"}, "resolved"),
        "reopen": ({"resolved", "closed"}, "open"), "close": ({"resolved"}, "closed"),
        "escalate": ({"open", "in_progress"}, "escalated"), "deescalate": ({"escalated"}, "in_progress")}),
    "devices": ("device", "DEV", "offline", {
        "boot": ({"offline"}, "online"), "shutdown": ({"online", "maintenance"}, "offline"),
        "service": ({"online", "offline"}, "maintenance"), "restore": ({"maintenance"}, "online"),
        "retire": ({"offline"}, "retired")}),
    "subscriptions": ("subscription", "SUB", "trial", {
        "convert": ({"trial"}, "active"), "lapse": ({"active"}, "past_due"),
        "settle": ({"past_due"}, "active"), "cancel": ({"trial", "active", "past_due"}, "cancelled"),
        "pause": ({"active"}, "paused"), "resume": ({"paused"}, "active")}),
}
COUNT = ["none of them", "exactly one", "exactly two", "exactly three", "all four"]


def _states(domain: str) -> list[str]:
    _, _, start, events = DOMAINS[domain]
    return list(dict.fromkeys([start, *(to for _, to in events.values())]))


def scenario(rng: random.Random) -> tuple[str, list[str], dict[str, str], Any]:
    """A domain, four entity ids, their final states, and the rendered state."""
    domain = rng.choice(sorted(DOMAINS))
    noun, prefix, start, events = DOMAINS[domain]
    ids = [f"{prefix}-{n}" for n in rng.sample(range(100, 999), 4)]
    now = dict.fromkeys(ids, start)
    log = []
    for _ in range(rng.randint(5, 14)):
        who, ev = rng.choice(ids), rng.choice(sorted(events))
        allowed, to = events[ev]
        if now[who] in allowed:
            now[who] = to
        log.append({"entity": who, "event": ev})
    rules = {ev: f"from {' or '.join(sorted(a))} to {to}" for ev, (a, to) in sorted(events.items())}
    header = (f"Every {noun} starts in state '{start}'. Events move a {noun} between states as the rules "
              f"say; an event not allowed from the {noun}'s current state is rejected and changes nothing.")
    style = rng.randrange(2)
    if style == 0:
        events_part: Any = {"event_log": log}
    else:
        events_part = "Event log, in order:\n" + "\n".join(
            f"{i + 1}. {e['event']} {e['entity']}" for i, e in enumerate(log))
    state = join(header, render(rng, "Transition rules", rules), events_part)
    return domain, ids, now, state


def generate(rng: random.Random, kind: str, template: str) -> Row | None:
    domain, ids, now, state = scenario(rng)
    noun = DOMAINS[domain][0]
    states = _states(domain)
    if kind == "noul":
        who = rng.choice(ids)
        s = now[who] if rng.random() < 0.5 else rng.choice(states)
        return row("noul", state, [f"After all the events, is {who} in state '{s}'?",
                                   f"Does {who} end up in the '{s}' state?",
                                   f"Once the log has been applied, is the {noun} {who} '{s}'?"],
                   YES_NO, int(now[who] == s), FAMILY)
    if kind == "choice":
        who = rng.choice(ids)
        return row("choice", state, [f"Which state is {who} in after all the events?",
                                     f"What is the final state of {who}?"],
                   named(states), states.index(now[who]), FAMILY)
    s = rng.choice(states)
    n = sum(v == s for v in now.values())
    return row("score", state, [f"How many of the four {noun}s end in state '{s}'?",
                                f"Count the {noun}s whose final state is '{s}'."], levels(COUNT), n, FAMILY)


TEMPLATES = {"noul": {"state": 1}, "choice": {"state": 1}, "score": {"count": 1}}

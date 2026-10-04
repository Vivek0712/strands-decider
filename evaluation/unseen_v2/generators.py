"""The generator-backed families of unseen-v2: every gold label is computed by a program.

Four families, each asked as all three question types, none of them a training family
(data/sources.md, and src/strands_decider/data/families/ for the A1 generators;
tests/test_unseen_v2.py holds the disjointness):

  chess       positions reached by seeded random play (python-chess, rules only, no
              engine): is the side to move in check, is a move legal, which move gives
              check / is legal / mates in one, the material balance, the king's mobility
  arithmetic  integer expressions and short word problems: is a stated value right,
              which value is right, how two quantities compare (5 ordinal levels)
  calendar    proleptic Gregorian dates (datetime): weekdays, leap years, order, offsets,
              the gap between two dates (5 ordinal levels)
  seating     a row of five seats and true clues with exactly one arrangement: who sits
              where, is a statement about the arrangement true, how many sit left of X

Each generator is a function `(rng, kind, template) -> row or None`; None means the draw did not
fit (a rare position, a value too close to a level boundary) and the caller draws again.
`draw` picks a question template, then a gold label uniformly, and redraws until the
generator yields it, so labels are balanced within every template: yes/no half and half,
choice positions uniform, score levels uniform. Everything follows from one `random.Random`, so a seed gives the same rows.

python-chess (GPL-3.0) is a build-time tool here, not a dependency of the package: the
rows record positions and rule facts, and nothing of python-chess is redistributed.
"""

from __future__ import annotations

import datetime as dt
import itertools
import random
from collections.abc import Callable
from typing import Any

Row = dict[str, Any]
Generator = Callable[[random.Random, str, str], "Row | None"]

YES_NO = [["false", "The answer is no."], ["true", "The answer is yes."]]


def _levels(descriptions: list[str]) -> list[list[str]]:
    return [[str(i), d] for i, d in enumerate(descriptions)]


def _row(kind: str, state: Any, instructions: str, options: list[list[str]], label: int, family: str,
         template: str) -> Row:
    return {"kind": kind, "state": state, "instructions": instructions, "options": options,
            "label": label, "task": f"unseen/{family}", "template": template}


def _choice(rng: random.Random, right: str, wrong: list[str], n: int = 4) -> tuple[list[list[str]], int] | None:
    """`right` among `n - 1` distinct wrong answers, in a random order: (options, label)."""
    pool = list(dict.fromkeys(w for w in wrong if w != right))
    if len(pool) < n - 1:
        return None
    names = [*rng.sample(pool, n - 1), right]
    rng.shuffle(names)
    return [[x, x] for x in names], names.index(right)


# ---- chess ---------------------------------------------------------------------------------

PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
MATERIAL = ["Black is clearly ahead (by 3 points or more)", "Black is slightly ahead (by 1 or 2 points)",
            "material is level", "White is slightly ahead (by 1 or 2 points)",
            "White is clearly ahead (by 3 points or more)"]
MOBILITY = ["the king has no legal move", "exactly one", "exactly two", "exactly three", "four or more"]


def _position(rng: random.Random, plies: tuple[int, int] = (8, 70)) -> Any:
    """A position reached by random play that favours captures and checks, so that
    positions are tactical rather than quiet; never a finished game."""
    import chess

    board = chess.Board()
    for _ in range(rng.randint(*plies)):
        moves = list(board.legal_moves)
        weights = [4 if board.is_capture(m) or board.gives_check(m) else 1 for m in moves]
        board.push(rng.choices(moves, weights)[0])
        if board.is_game_over():
            board.pop()
            break
    return board


def _board_text(board: Any) -> str:
    import chess

    ranks = str(board).splitlines()
    grid = "\n".join(f"{8 - i} {r}" for i, r in enumerate(ranks)) + "\n  a b c d e f g h"
    side = "White" if board.turn == chess.WHITE else "Black"
    return (f"Chess position (FEN): {board.fen()}\n{grid}\n"
            f"Upper-case letters are White pieces, lower-case Black. {side} is to move.")


def _describe(board: Any, move: Any) -> str:
    import chess

    piece = board.piece_at(move.from_square)
    name = PIECE_NAMES[piece.piece_type] if piece else "piece"
    text = f"the {name} on {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
    if move.promotion:
        text += f", promoting to a {PIECE_NAMES[move.promotion]}"
    return text


def _reach(board: Any, frm: int) -> list[int]:
    """The squares the piece on `frm` would reach on an otherwise empty board."""
    import chess

    piece = board.piece_at(frm)
    empty = chess.Board(None)
    empty.set_piece_at(frm, piece)
    squares = list(empty.attacks(frm))
    if piece.piece_type == chess.PAWN:
        step = 8 if piece.color == chess.WHITE else -8
        squares += [s for s in (frm + step, frm + 2 * step) if 0 <= s < 64]
    return squares


def _illegal_moves(board: Any, rng: random.Random, k: int = 12) -> list[Any]:
    """Moves of the side to move's own pieces that are not legal: a wrong geometry, a
    blocked path, or one that leaves the king in check."""
    import chess

    legal = set(board.legal_moves)
    own = [s for s, p in board.piece_map().items() if p.color == board.turn]
    # pseudo-legal moves that leave or put the king in check come first: the hard kind
    out: list[Any] = [m for m in board.pseudo_legal_moves if m not in legal]
    rng.shuffle(out)
    out = out[: k // 2]
    for _ in range(200):
        if len(out) >= k:
            break
        frm = rng.choice(own)
        # Mostly a square the piece could reach on an empty board (a blocked or
        # self-exposing move), sometimes any square (a wrong geometry).
        reach = _reach(board, frm)
        to = rng.choice(reach) if reach and rng.random() < 0.8 else rng.randrange(64)
        target = board.piece_at(to)
        if to == frm or (target is not None and target.color == board.turn):
            continue
        piece = board.piece_at(frm)
        promo = (chess.QUEEN if piece is not None and piece.piece_type == chess.PAWN
                 and chess.square_rank(to) in (0, 7) else None)
        mv = chess.Move(frm, to, promotion=promo)
        if mv not in legal and mv not in out:
            out.append(mv)
    return out


def chess_family(rng: random.Random, kind: str, t: str) -> Row | None:
    import chess

    board = _position(rng)
    side = "White" if board.turn == chess.WHITE else "Black"
    legal = list(board.legal_moves)
    state = _board_text(board)
    if kind == "noul":
        if t == "check":
            return _row("noul", state, f"Is {side}'s king in check in this position?", YES_NO,
                        int(board.is_check()), "chess", t)
        if t == "legal":
            if rng.random() < 0.5:
                mv, ok = rng.choice(legal), 1
            else:
                bad = _illegal_moves(board, rng, 1)
                if not bad:
                    return None
                mv, ok = bad[0], 0
            return _row("noul", state, f"Is moving {_describe(board, mv)} a legal move for {side} here?",
                        YES_NO, ok, "chess", t)
        mv = rng.choice(legal)
        return _row("noul", state, f"If {side} plays {_describe(board, mv)}, does that move give check?",
                    YES_NO, int(board.gives_check(mv)), "chess", t)
    if kind == "choice":
        if t == "gives_check":
            checks = [m for m in legal if board.gives_check(m)]
            quiet = [m for m in legal if not board.gives_check(m)]
            if not checks or len(quiet) < 3:
                return None
            right = rng.choice(checks)
            picked = _choice(rng, _describe(board, right), [_describe(board, m) for m in quiet])
            question = f"Which of these moves by {side} gives check?"
        elif t == "legal":
            bad = _illegal_moves(board, rng)
            picked = _choice(rng, _describe(board, rng.choice(legal)), [_describe(board, m) for m in bad])
            question = f"Which of these moves is legal for {side} in this position?"
        else:
            mates = []
            for m in legal:
                board.push(m)
                if board.is_checkmate():
                    mates.append(m)
                board.pop()
            if len(mates) != 1:
                return None
            others = [m for m in legal if m != mates[0]]
            picked = _choice(rng, _describe(board, mates[0]), [_describe(board, m) for m in others])
            question = f"Which of these moves by {side} delivers checkmate?"
        if picked is None:
            return None
        return _row("choice", state, question, picked[0], picked[1], "chess", t)
    if t == "material":
        diff = sum(PIECE_VALUES[p.piece_type] * (1 if p.color == chess.WHITE else -1)
                   for p in board.piece_map().values())
        level = 0 if diff <= -3 else 1 if diff < 0 else 2 if diff == 0 else 3 if diff < 3 else 4
        return _row("score", state, "Counting pawn 1, knight 3, bishop 3, rook 5 and queen 9, how does the "
                    "material stand?", _levels(MATERIAL), level, "chess", t)
    king = board.king(board.turn)
    n = sum(m.from_square == king for m in legal)
    return _row("score", state, f"How many legal moves does {side}'s king have?", _levels(MOBILITY),
                min(n, 4), "chess", t)


# ---- arithmetic ----------------------------------------------------------------------------

RATIO = ["A is far smaller than B (less than half of B)", "A is somewhat smaller than B (half to nine tenths)",
         "A and B are about equal (within ten percent)", "A is somewhat larger than B (1.1 to 2 times B)",
         "A is far larger than B (more than twice B)"]
RATIO_CUTS = (0.5, 1 / 1.1, 1.1, 2.0)  # level boundaries on A / B


def _expression(rng: random.Random) -> tuple[str, int]:
    """A small integer expression and its value."""
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 19)
    shape = rng.randrange(5)
    if shape == 0:
        return f"{a} + {b} * {c}", a + b * c
    if shape == 1:
        return f"({a} + {b}) * {c}", (a + b) * c
    if shape == 2:
        return f"{a} * {c} - {b}", a * c - b
    if shape == 3:
        d = rng.randint(2, 12)
        return f"{a * d} / {d} + {b}", a + b
    return f"{a} - {b} - {c}", a - b - c


WORD = [
    ("A warehouse holds {a} crates of {b} bottles each and ships {c} bottles. How many bottles remain?",
     lambda a, b, c: a * b - c),
    ("A runner covers {a} laps of a {b}-metre track, then walks another {c} metres. How many metres in total?",
     lambda a, b, c: a * b + c),
    ("A printer has {c} sheets and prints {a} reports of {b} pages each. How many sheets are left over?",
     lambda a, b, c: c - a * b),
    ("{a} people share the cost of {b} tickets at {c} dollars each. How much is the total bill in dollars?",
     lambda a, b, c: b * c),
]


def _word_problem(rng: random.Random) -> tuple[str, int]:
    template, f = rng.choice(WORD)
    a, b = rng.randint(3, 40), rng.randint(3, 60)
    c = rng.randint(a * b + 1, a * b + 400) if "left over" in template else rng.randint(5, 500)
    return template.format(a=a, b=b, c=c), f(a, b, c)


def _near(value: int, rng: random.Random) -> list[int]:
    """Plausible wrong values: off by one or ten, digits swapped, a sign or carry slip."""
    s = str(abs(value))
    swapped = int(s[::-1]) * (1 if value >= 0 else -1) if len(s) > 1 else value + 2
    cands = {value + 1, value - 1, value + 10, value - 10, swapped, -value, value + rng.choice([2, 3, 9, 11]),
             value * 2 if value else 7, value + 100}
    cands.discard(value)
    return sorted(cands)


def arithmetic_family(rng: random.Random, kind: str, t: str) -> Row | None:
    worded = t == "word"
    text, value = _word_problem(rng) if worded else _expression(rng)
    state = text if worded else f"Expression: {text}\n(* is multiplication and / exact division; usual precedence.)"
    if kind == "noul":
        claim = value if rng.random() < 0.5 else rng.choice(_near(value, rng))
        question = (f"Is the answer {claim}?" if worded else f"Does the expression equal {claim}?")
        return _row("noul", state, question, YES_NO, int(claim == value), "arithmetic", t)
    if kind == "choice":
        picked = _choice(rng, str(value), [str(x) for x in _near(value, rng)], n=rng.choice([4, 5]))
        if picked is None:
            return None
        return _row("choice", state, "What is the answer?" if worded else "What is the value of the expression?",
                    picked[0], picked[1], "arithmetic", t)
    expr_a, a = _expression(rng)
    expr_b, b = _expression(rng)
    if a <= 0 or b <= 0:
        return None
    ratio = a / b
    if any(abs(ratio - cut) / cut < 0.04 for cut in RATIO_CUTS):
        return None  # too close to a boundary to be a fair item
    level = sum(ratio > cut for cut in RATIO_CUTS)
    return _row("score", f"A = {expr_a}\nB = {expr_b}\n(usual precedence; / is exact division)",
                "How does A compare with B?", _levels(RATIO), level, "arithmetic", t)


# ---- calendar ------------------------------------------------------------------------------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GAP = ["less than a week apart", "one to four weeks apart", "one to six months apart (29 to 182 days)",
       "six months to two years apart (183 to 730 days)", "more than two years apart"]
GAP_CUTS = (7, 29, 183, 731)  # a gap of d days is at level sum(d >= cut)


def _date(rng: random.Random) -> dt.date:
    return dt.date(1900, 1, 1) + dt.timedelta(days=rng.randrange(200 * 365))


def _fmt(d: dt.date, rng: random.Random, style: int | None = None) -> str:
    style = rng.randrange(3) if style is None else style
    if style == 0:
        return d.isoformat()
    if style == 1:
        return f"{d.day} {d.strftime('%B')} {d.year}"
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def calendar_family(rng: random.Random, kind: str, t: str) -> Row | None:
    d = _date(rng)
    if kind == "noul":
        if t == "weekend":
            return _row("noul", f"Date: {_fmt(d, rng)}", "Does this date fall on a Saturday or a Sunday?",
                        YES_NO, int(d.weekday() >= 5), "calendar", t)
        if t == "before":
            e = d + dt.timedelta(days=rng.choice([-1, 1]) * rng.randint(1, 900))
            return _row("noul", f"First date: {_fmt(d, rng)}\nSecond date: {_fmt(e, rng)}",
                        "Is the first date earlier than the second?", YES_NO, int(d < e), "calendar", t)
        if t == "leap":
            y = rng.choice([rng.randrange(1600, 2400), rng.randrange(16, 24) * 100])
            return _row("noul", f"Year: {y} (Gregorian calendar)", "Is this a leap year?", YES_NO,
                        int(_leap(y)), "calendar", t)
        n = rng.randint(1, 120)
        e = d + dt.timedelta(days=n)
        month = e.strftime("%B") if rng.random() < 0.5 else (d + dt.timedelta(days=n + 31)).strftime("%B")
        return _row("noul", f"Start date: {_fmt(d, rng)}\nDays added: {n}",
                    f"Does the resulting date fall in {month}?", YES_NO, int(e.strftime("%B") == month),
                    "calendar", t)
    if kind == "choice":
        if t == "weekday":
            return _row("choice", f"Date: {_fmt(d, rng)} (Gregorian calendar)", "What day of the week is it?",
                        [[w, w] for w in WEEKDAYS], d.weekday(), "calendar", t)
        n = rng.randint(2, 400)
        e = d + dt.timedelta(days=n)
        style = rng.randrange(3)  # one style for every option
        wrong = [_fmt(e + dt.timedelta(days=k), rng, style) for k in (-1, 1, 7, -7, 30, -30, 365)]
        picked = _choice(rng, _fmt(e, rng, style), wrong)
        if picked is None:
            return None
        return _row("choice", f"Start date: {_fmt(d, rng)}\nDays added: {n}", "What is the resulting date?",
                    picked[0], picked[1], "calendar", t)
    gap = rng.choice([rng.randint(0, 6), rng.randint(7, 28), rng.randint(29, 182), rng.randint(183, 730),
                      rng.randint(731, 4000)])
    if any(abs(gap - cut) <= 1 for cut in GAP_CUTS):
        return None
    e = d + dt.timedelta(days=gap)
    first, second = (d, e) if rng.random() < 0.5 else (e, d)
    return _row("score", f"Date A: {_fmt(first, rng)}\nDate B: {_fmt(second, rng)}",
                "How far apart are the two dates?", _levels(GAP), sum(gap >= c for c in GAP_CUTS),
                "calendar", t)


# ---- seating -------------------------------------------------------------------------------

NAMES = ["Ada", "Bram", "Chen", "Dalia", "Emeka", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kemal", "Lena",
         "Mira", "Nils", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tomas"]
SEATS = 5
LEFT = ["no one", "one person", "two people", "three people", "four people"]


def _clue(rng: random.Random, seat: dict[str, int], names: list[str]) -> tuple[str, Callable[[dict[str, int]], bool]]:
    a, b = rng.sample(names, 2)
    k = rng.randrange(6)
    if k == 0:
        return (f"{a} sits somewhere to the left of {b}." if seat[a] < seat[b] else
                f"{a} sits somewhere to the right of {b}.",
                (lambda s, a=a, b=b, lt=seat[a] < seat[b]: (s[a] < s[b]) == lt))
    if k == 1 and abs(seat[a] - seat[b]) == 1:
        return f"{a} and {b} sit next to each other.", lambda s, a=a, b=b: abs(s[a] - s[b]) == 1
    if k == 1:
        return f"{a} and {b} do not sit next to each other.", lambda s, a=a, b=b: abs(s[a] - s[b]) != 1
    if k == 2:
        return f"{a} sits in seat {seat[a]}.", lambda s, a=a, n=seat[a]: s[a] == n
    if k == 3:
        end = seat[a] in (1, SEATS)
        return (f"{a} sits at one end of the row." if end else f"{a} does not sit at either end of the row.",
                lambda s, a=a, end=end: (s[a] in (1, SEATS)) == end)
    if k == 4 and seat[b] - seat[a] == 1:
        return f"{a} sits immediately to the left of {b}.", lambda s, a=a, b=b: s[b] - s[a] == 1
    gap = abs(seat[a] - seat[b]) - 1
    return (f"Exactly {gap} {'person sits' if gap == 1 else 'people sit'} between {a} and {b}.",
            lambda s, a=a, b=b, g=gap: abs(s[a] - s[b]) - 1 == g)


def _puzzle(rng: random.Random) -> tuple[list[str], dict[str, int], str]:
    """Five people, a hidden seating, and true clues added until exactly one seating fits."""
    names = rng.sample(NAMES, SEATS)
    order = rng.sample(names, SEATS)
    seat = {n: i + 1 for i, n in enumerate(order)}
    every = [dict(zip(p, range(1, SEATS + 1), strict=True)) for p in itertools.permutations(names)]
    clues: list[str] = []
    alive = every
    while len(alive) > 1 and len(clues) < 12:
        text, test = _clue(rng, seat, names)
        kept = [s for s in alive if test(s)]
        if len(kept) < len(alive) and text not in clues:
            clues.append(text)
            alive = kept
    if len(alive) != 1:
        return _puzzle(rng)
    state = (f"Five people, {', '.join(sorted(names))}, sit in a row of seats numbered 1 to 5 from left to "
             "right, one person per seat.\nClues:\n" + "\n".join(f"- {c}" for c in clues))
    return names, seat, state


def seating_family(rng: random.Random, kind: str, t: str) -> Row | None:
    names, seat, state = _puzzle(rng)
    by_seat = {v: k for k, v in seat.items()}
    if kind == "noul":
        a, b = rng.sample(names, 2)
        if t == "seat":
            n = seat[a] if rng.random() < 0.5 else rng.choice([s for s in range(1, SEATS + 1) if s != seat[a]])
            return _row("noul", state, f"Does {a} sit in seat {n}?", YES_NO, int(seat[a] == n), "seating", t)
        return _row("noul", state, f"Does {a} sit to the left of {b}?", YES_NO, int(seat[a] < seat[b]),
                    "seating", t)
    if kind == "choice":
        n = rng.randint(1, SEATS)
        options = [[x, x] for x in sorted(names)]
        return _row("choice", state, f"Who sits in seat {n}?", options, sorted(names).index(by_seat[n]),
                    "seating", t)
    a = rng.choice(names)
    return _row("score", state, f"How many people sit to the left of {a}?", _levels(LEFT), seat[a] - 1,
                "seating", t)


FAMILIES: dict[str, Generator] = {
    "chess": chess_family,
    "arithmetic": arithmetic_family,
    "calendar": calendar_family,
    "seating": seating_family,
}
# The question templates of each family and kind, with their weights: `draw` picks the
# template first, so a template that rarely yields one label does not crowd out the others.
TEMPLATES: dict[str, dict[str, dict[str, int]]] = {
    "chess": {"noul": {"check": 1, "legal": 1, "gives_check": 1},
              "choice": {"gives_check": 4, "legal": 4, "mate_in_one": 1},
              "score": {"material": 1, "king_mobility": 1}},
    "arithmetic": {"noul": {"expression": 2, "word": 1}, "choice": {"expression": 2, "word": 1},
                   "score": {"compare": 1}},
    "calendar": {"noul": {"weekend": 1, "before": 1, "leap": 1, "offset_month": 1},
                 "choice": {"weekday": 1, "offset": 1}, "score": {"gap": 1}},
    "seating": {"noul": {"seat": 1, "left_of": 1}, "choice": {"who": 1}, "score": {"count_left": 1}},
}


def draw(family: str, kind: str, n: int, rng: random.Random, max_tries: int = 2000) -> list[Row]:
    """`n` rows of `family` asked as `kind`, gold labels balanced within each template:
    before each row a template is picked by weight and a target label position drawn
    uniformly, and the generator is asked again until it yields that label. A row repeated
    exactly is drawn again."""
    gen = FAMILIES[family]
    names, weights = zip(*TEMPLATES[family][kind].items(), strict=True)
    out: list[Row] = []
    seen: set[str] = set()
    while len(out) < n:
        template = rng.choices(names, weights)[0]
        target = rng.random()  # a uniform position among the options, fixed before the draw
        for _ in range(max_tries):
            row = gen(rng, kind, template)
            if row is None:
                continue
            key = repr((row["state"], row["instructions"], row["options"]))
            if key in seen:
                continue
            if row["label"] == int(target * len(row["options"])):
                seen.add(key)
                out.append(row)
                break
        else:
            raise RuntimeError(f"{family}/{kind}/{template}: no row with the target label in {max_tries} draws")
    return out

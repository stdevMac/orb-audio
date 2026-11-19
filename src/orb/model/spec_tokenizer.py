"""Compact spec tokenizer for the planner.

The planner predicts a `Spec` (duration, texture, mood, and timed events), not
a full Score. The deterministic compiler (`grammar.score_from_spec`) turns the
Spec into a Score.

Numbers the caption states are encoded as decimal digits, not bins. A bin makes
the model learn decimal-to-bin arithmetic; digits make it a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..grammar import (
    GAIN_LEVELS,
    MAX_EVENTS,
    MOODS,
    NO_BED,
    PAN_LABELS,
    Spec,
    SpecEvent,
    gain_index,
    gain_value,
    pan_index,
    pan_value,
    quantize_time,
)
from ..score import EVENT_ROLES, TEXTURES

# Byte tokens occupy 0..255.
PAD = 256
BOS = 257
EOS = 258
SEP = 259

SPEC = 260
END_SPEC = 261
DUR = 262
TEX = 263
MOOD = 264
EVENTS = 265
END_EVENTS = 266
EVENT = 267
T = 268
D = 269
G = 270
P = 271
S = 272

DIGIT = 273  # 10
GAIN = 283  # 16
PAN = 299  # 9
HEX = 308  # 16
TEXVAL = 324  # 8: the original textures
MOODVAL = 332  # 10
ROLEVAL = 342  # one per EVENT_ROLES entry (47 since v0.3)
TEX_BASE_SLOTS = 8
# Textures after the first 8 (ocean, stream, forest, night, traffic since
# v0.3) take ids after the roles, so every v0.1/v0.2 token keeps its id.
TEXEXT = ROLEVAL + len(EVENT_ROLES)
# "No bed" (texture "none") gets one token after the extended textures, so a
# free-text prompt like "a notification ding" can plan a dry cue (v0.4). A
# planner trained before v0.4 has no such row; `SpecGrammar(no_bed=False)`
# keeps it out of the mask.
NOBED = TEXEXT + max(0, len(TEXTURES) - TEX_BASE_SLOTS)  # 394
VOCAB_SIZE = NOBED + 1  # 395; 394 in v0.3, 358 in v0.2, 352 in v0.1

DIGITS = tuple(range(DIGIT, DIGIT + 10))
GAINS = tuple(range(GAIN, GAIN + GAIN_LEVELS))
PANS = tuple(range(PAN, PAN + len(PAN_LABELS)))
HEXS = tuple(range(HEX, HEX + 16))


def tex_token(i: int) -> int:
    return TEXVAL + i if i < TEX_BASE_SLOTS else TEXEXT + i - TEX_BASE_SLOTS


def tex_from_token(tok: int) -> int | None:
    if TEXVAL <= tok < TEXVAL + min(TEX_BASE_SLOTS, len(TEXTURES)):
        return tok - TEXVAL
    if TEXEXT <= tok < TEXEXT + len(TEXTURES) - TEX_BASE_SLOTS:
        return tok - TEXEXT + TEX_BASE_SLOTS
    return None


TEXVALS = tuple(tex_token(i) for i in range(len(TEXTURES)))
MOODVALS = tuple(range(MOODVAL, MOODVAL + len(MOODS)))
ROLEVALS = tuple(range(ROLEVAL, ROLEVAL + len(EVENT_ROLES)))

TEX_TO_ID = {t: i for i, t in enumerate(TEXTURES)}
MOOD_TO_ID = {m: i for i, m in enumerate(MOODS)}
ROLE_TO_ID = {r: i for i, r in enumerate(EVENT_ROLES)}


def _digits(n: int, width: int) -> list[int]:
    s = f"{int(n):0{width}d}"
    if len(s) != width:
        raise ValueError(f"{n} does not fit in {width} digits")
    return [DIGIT + int(c) for c in s]


def _hex(n: int) -> list[int]:
    s = f"{int(n):02x}"
    return [HEX + int(c, 16) for c in s]


@dataclass
class EncodedSpecPair:
    input_ids: list[int]
    labels: list[int]
    weights: list[float]


class SpecTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id = PAD
    bos_id = BOS
    eos_id = EOS
    sep_id = SEP

    def encode_caption(self, text: str, add_special: bool = True) -> list[int]:
        body = list(text.encode("utf-8"))
        return ([BOS] + body + [EOS]) if add_special else body

    def prompt(self, text: str) -> list[int]:
        return [BOS] + list(text.encode("utf-8")) + [SEP]

    def encode_spec(self, spec: Spec, add_special: bool = True) -> list[int]:
        out: list[int] = []
        if add_special:
            out.append(BOS)
        if spec.texture not in TEX_TO_ID and spec.texture != NO_BED:
            raise ValueError(f"texture {spec.texture!r} cannot be tokenized")
        if spec.bed_db is not None:
            raise ValueError("bed_db cannot be tokenized: the planner always emits the default bed level")
        if any(ev.params for ev in spec.events):
            raise ValueError("event params cannot be tokenized: the planner always emits the role defaults")
        out += [SPEC, DUR] + _digits(round(spec.duration * 100), 5)
        tex = NOBED if spec.texture == NO_BED else tex_token(TEX_TO_ID[spec.texture])
        out += [TEX, tex, MOOD, MOODVAL + MOOD_TO_ID[spec.mood], EVENTS]
        for ev in spec.events:
            out += [EVENT, ROLEVAL + ROLE_TO_ID[ev.role], T] + _digits(round(ev.t * 100), 4)
            out += [D] + _digits(round(ev.dur * 100), 4)
            out += [G, GAIN + gain_index(ev.gain_db), P, PAN + pan_index(ev.pan), S] + _hex(ev.seed)
        out += [END_EVENTS, END_SPEC]
        if add_special:
            out.append(EOS)
        return out

    def decode_spec(self, ids: list[int]) -> Spec:
        toks = [int(i) for i in ids if int(i) not in (PAD, BOS, EOS)]
        p = 0

        def take(expected: int | None = None) -> int:
            nonlocal p
            if p >= len(toks):
                raise ValueError("unexpected end of spec tokens")
            tok = toks[p]
            p += 1
            if expected is not None and tok != expected:
                raise ValueError(f"expected {expected}, got {tok} at {p - 1}")
            return tok

        def digits(width: int) -> int:
            nonlocal p
            value = 0
            for _ in range(width):
                tok = take()
                if not (DIGIT <= tok < DIGIT + 10):
                    raise ValueError(f"expected digit, got {tok}")
                value = value * 10 + (tok - DIGIT)
            return value

        def one_of(lo: int, hi: int, what: str) -> int:
            tok = take()
            if not (lo <= tok < hi):
                raise ValueError(f"expected {what}, got {tok}")
            return tok - lo

        take(SPEC)
        take(DUR)
        duration = digits(5) / 100.0
        take(TEX)
        tex_tok = take()
        tex_i = tex_from_token(tex_tok)
        if tex_i is None and tex_tok != NOBED:
            raise ValueError(f"expected texture, got {tex_tok}")
        texture = NO_BED if tex_tok == NOBED else TEXTURES[tex_i]
        take(MOOD)
        mood = MOODS[one_of(MOODVAL, MOODVAL + len(MOODS), "mood")]
        take(EVENTS)

        events: list[SpecEvent] = []
        while p < len(toks) and toks[p] != END_EVENTS:
            take(EVENT)
            role = EVENT_ROLES[one_of(ROLEVAL, ROLEVAL + len(EVENT_ROLES), "role")]
            take(T)
            t = digits(4) / 100.0
            take(D)
            dur = digits(4) / 100.0
            take(G)
            gain = gain_value(one_of(GAIN, GAIN + GAIN_LEVELS, "gain"))
            take(P)
            pan = pan_value(one_of(PAN, PAN + len(PAN_LABELS), "pan"))
            take(S)
            seed = 0
            for _ in range(2):
                tok = take()
                if not (HEX <= tok < HEX + 16):
                    raise ValueError(f"expected hex digit, got {tok}")
                seed = seed * 16 + (tok - HEX)
            if len(events) >= MAX_EVENTS:
                raise ValueError(f"more than {MAX_EVENTS} events")
            events.append(SpecEvent(role, t, dur, gain, pan, seed))

        take(END_EVENTS)
        take(END_SPEC)
        spec = Spec(quantize_time(duration), texture, mood, events)
        for ev in spec.events:
            if ev.t + ev.dur > spec.duration + 1e-6:
                raise ValueError(f"event {ev.role} at {ev.t} exceeds duration {spec.duration}")
        return spec

    def encode_pair(self, caption: str, spec: Spec, max_len: int | None = None) -> EncodedSpecPair:
        cap = list(caption.encode("utf-8"))
        body = self.encode_spec(spec, add_special=False)
        input_ids = [BOS] + cap + [SEP] + body + [EOS]
        labels = [-100] * (1 + len(cap) + 1) + body + [EOS]
        weights = [0.0] * (1 + len(cap) + 1) + [1.0] * len(body) + [1.0]
        if max_len is not None:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            weights = weights[:max_len]
        return EncodedSpecPair(input_ids, labels, weights)


class SpecGrammar:
    """Constrained-decoding mask for the Spec body.

    Structural tokens, categorical ranges, and `t + dur <= duration` are all
    enforced during generation, so a decoded Spec is always valid.
    """

    def __init__(self, no_bed: bool = True) -> None:
        self.no_bed = no_bed  # False for planners trained before the NOBED token
        self._bedless = False
        self.state = "START"
        self._left = 0
        self._events = 0
        self._value = 0
        self._duration_cs = 0
        self._t_cs = 0

    def _digits_below(self, limit: int) -> set[int]:
        """Digits that keep the final value <= limit, given the prefix so far."""
        remaining = self._left - 1
        scale = 10**remaining
        return {DIGIT + d for d in range(10) if (self._value * 10 + d) * scale <= limit}

    def allowed(self) -> set[int]:
        s = self.state
        if s == "START":
            return {SPEC}
        if s == "DUR":
            return {DUR}
        if s == "DUR_DIGITS":
            return self._digits_below(99999)
        if s == "TEX":
            return {TEX}
        if s == "TEX_VAL":
            return set(TEXVALS) | ({NOBED} if self.no_bed else set())
        if s == "MOOD":
            return {MOOD}
        if s == "MOOD_VAL":
            return set(MOODVALS)
        if s == "EVENTS":
            return {EVENTS}
        if s == "EVENT_OR_END":
            if self._bedless and self._events == 0:
                return {EVENT}  # a cue with no bed needs at least one event
            return {EVENT, END_EVENTS} if self._events < MAX_EVENTS else {END_EVENTS}
        if s == "ROLE":
            return set(ROLEVALS)
        if s == "T":
            return {T}
        if s == "T_DIGITS":
            return self._digits_below(max(0, self._duration_cs - 2))
        if s == "D":
            return {D}
        if s == "D_DIGITS":
            return self._digits_below(max(0, self._duration_cs - self._t_cs))
        if s == "G":
            return {G}
        if s == "GAIN_VAL":
            return set(GAINS)
        if s == "P":
            return {P}
        if s == "PAN_VAL":
            return set(PANS)
        if s == "S":
            return {S}
        if s == "HEX_DIGITS":
            return set(HEXS)
        if s == "END_SPEC":
            return {END_SPEC}
        if s == "DONE":
            return {EOS}
        raise AssertionError(f"bad grammar state {s!r}")

    def advance(self, token: int) -> None:
        if token not in self.allowed():
            raise ValueError(f"token {token} not allowed in state {self.state}")
        s = self.state
        if s == "START":
            self.state = "DUR"
        elif s == "DUR":
            self.state, self._left, self._value = "DUR_DIGITS", 5, 0
        elif s == "DUR_DIGITS":
            self._value = self._value * 10 + (token - DIGIT)
            self._left -= 1
            if self._left == 0:
                self._duration_cs = self._value
                self._value = 0
                self.state = "TEX"
        elif s == "TEX":
            self.state = "TEX_VAL"
        elif s == "TEX_VAL":
            self._bedless = token == NOBED
            self.state = "MOOD"
        elif s == "MOOD":
            self.state = "MOOD_VAL"
        elif s == "MOOD_VAL":
            self.state = "EVENTS"
        elif s == "EVENTS":
            self.state = "EVENT_OR_END"
        elif s == "EVENT_OR_END":
            if token == EVENT:
                self._events += 1
                self.state = "ROLE"
            else:
                self.state = "END_SPEC"
        elif s == "ROLE":
            self.state = "T"
        elif s == "T":
            self.state, self._left, self._value = "T_DIGITS", 4, 0
        elif s == "T_DIGITS":
            self._value = self._value * 10 + (token - DIGIT)
            self._left -= 1
            if self._left == 0:
                self._t_cs = self._value
                self._value = 0
                self.state = "D"
        elif s == "D":
            self.state, self._left, self._value = "D_DIGITS", 4, 0
        elif s == "D_DIGITS":
            self._value = self._value * 10 + (token - DIGIT)
            self._left -= 1
            if self._left == 0:
                self._value = 0
                self.state = "G"
        elif s == "G":
            self.state = "GAIN_VAL"
        elif s == "GAIN_VAL":
            self.state = "P"
        elif s == "P":
            self.state = "PAN_VAL"
        elif s == "PAN_VAL":
            self.state = "S"
        elif s == "S":
            self.state, self._left = "HEX_DIGITS", 2
        elif s == "HEX_DIGITS":
            self._left -= 1
            if self._left == 0:
                # Always pass through EVENT_OR_END so END_EVENTS is emitted,
                # even at MAX_EVENTS; allowed() blocks further EVENT tokens.
                self.state = "EVENT_OR_END"
        elif s == "END_SPEC":
            self.state = "DONE"

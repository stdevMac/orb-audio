"""Deterministic score grammar.

The caption must determine the score. A small `Spec` (duration, texture, mood,
and a list of timed events) is sampled, the score is a deterministic function
of that spec, and the caption is generated from the score. The model can learn
the mapping instead of the dataset.

v1 is SFX + texture only. Tempo, key and mode are deliberately absent from the
Spec: with music off they do not affect the audio, so putting them in the Spec
would inflate exact-match without being verifiable. They return in v2 with a
version token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .score import (
    AMBIENCE_TEXTURES,
    EVENT_ROLES,
    TEXTURES,
    Event,
    MusicTrack,
    Note,
    Score,
    Section,
    Texture,
)

MOODS = (
    "calm",
    "warm",
    "playful",
    "hopeful",
    "bright",
    "tense",
    "dark",
    "eerie",
    "neutral",
    "intense",
)
MOOD_ENERGY = {
    "calm": 0.35,
    "warm": 0.50,
    "playful": 0.60,
    "hopeful": 0.65,
    "bright": 0.70,
    "tense": 0.75,
    "dark": 0.80,
    "eerie": 0.70,
    "neutral": 0.55,
    "intense": 0.90,
}
SECTION_ENERGY = {"intro": 0.35, "build": 0.60, "peak": 0.90, "outro": 0.40}
SECTION_DENSITY = {"intro": 0.25, "build": 0.50, "peak": 0.75, "outro": 0.35}

# role: (character, apex fraction, default gain dB, default duration s)
EVENT_STYLE = {
    "whoosh": ("fast, airy", 0.05, -8.0, 0.80),
    "impact": ("short, deep", 0.00, -6.0, 0.70),
    "ui_pop": ("soft, rounded", 0.00, -14.0, 0.16),
    "riser": ("dark, tense", 0.92, -20.0, 2.00),
    "swell": ("wide, shimmering", 0.90, -18.0, 1.50),
    "click": ("tight, clean", 0.00, -16.0, 0.10),
    "bell": ("bright, metallic", 0.10, -14.0, 1.40),
    "sub_drop": ("deep, clean", 0.10, -12.0, 1.00),
    "glitch": ("digital, broken", 0.00, -16.0, 0.20),
    "noise_sweep": ("wide, airy", 0.50, -18.0, 1.50),
    "thunder": ("deep, rolling", 0.02, -6.0, 6.00),
    "door_slam": ("heavy, wooden", 0.00, -8.0, 0.90),
    "knock": ("dry, wooden", 0.00, -12.0, 0.70),
    "footsteps": ("steady, close", 0.00, -14.0, 3.00),
    "water_drop": ("small, liquid", 0.00, -14.0, 0.35),
    "explosion": ("huge, rumbling", 0.00, -4.0, 3.00),
    "bird_chirp": ("bright, lively", 0.00, -14.0, 3.00),
    "crickets": ("steady, nocturnal", 0.00, -18.0, 4.00),
    "wave_crash": ("rolling, foamy", 0.25, -8.0, 5.00),
    "wind_gust": ("rushing, cold", 0.40, -10.0, 4.00),
    "splash": ("wet, sudden", 0.00, -10.0, 1.20),
    "frog": ("croaky, wet", 0.00, -14.0, 3.00),
    "owl": ("soft, hollow", 0.00, -16.0, 3.00),
    "glass_break": ("sharp, shattering", 0.00, -8.0, 1.60),
    "clock_tick": ("dry, steady", 0.00, -16.0, 4.00),
    "phone_ring": ("ringing, insistent", 0.00, -12.0, 4.00),
    "keyboard_typing": ("busy, clicky", 0.00, -16.0, 4.00),
    "mouse_click": ("small, plastic", 0.00, -16.0, 0.30),
    "camera_shutter": ("crisp, mechanical", 0.00, -14.0, 0.60),
    "paper_rustle": ("crisp, papery", 0.00, -16.0, 2.00),
    "cork_pop": ("festive, popping", 0.00, -12.0, 1.00),
    "coin_drop": ("bright, jingling", 0.00, -14.0, 1.50),
    "metal_clang": ("hard, ringing", 0.00, -10.0, 2.50),
    "wood_creak": ("slow, creaking", 0.50, -14.0, 2.00),
    "heartbeat": ("deep, pulsing", 0.00, -10.0, 4.00),
    "applause": ("warm, clapping", 0.30, -12.0, 5.00),
    "pouring_water": ("liquid, filling", 0.00, -14.0, 3.00),
    "can_opening": ("fizzy, snappy", 0.00, -12.0, 1.00),
    "clock_alarm": ("loud, ringing", 0.00, -14.0, 3.00),
    "punch": ("hard, meaty", 0.00, -8.0, 0.50),
    "sword_clash": ("steely, ringing", 0.00, -8.0, 1.50),
    "gunshot": ("loud, cracking", 0.00, -8.0, 1.50),
    "car_horn": ("loud, brassy", 0.00, -10.0, 1.00),
    "siren": ("wailing, urgent", 0.00, -14.0, 6.00),
    "engine": ("low, rumbling", 0.00, -12.0, 4.00),
    "church_bell": ("deep, tolling", 0.00, -10.0, 6.00),
    "fireworks": ("booming, crackling", 0.30, -10.0, 4.00),
}

# The bed. Every texture is filtered noise placed `bed_db` below the
# foreground's active loudness. The planner always emits a bed at the default
# level (its Spec has no slot for the level); "none" is reachable by writing a
# Spec directly or with a bed-less grammar caption (see `parse_caption`), and
# `bed_db` only by writing a Spec.
#
# The default was -30 dB (the level of every training score, kept as
# `TRAIN_BED_DB`). At -30 the bed sat at -40..-48 dBFS before the first event
# of a sparse cue, an audible hiss under UI sounds. At -40 it is ambience you
# notice in the gaps, not a noise floor.
NO_BED = "none"
BED_CHOICES = TEXTURES + (NO_BED,)
DEFAULT_BED_DB = -40.0
TRAIN_BED_DB = -30.0
BED_DB_RANGE = (-60.0, -6.0)
# The scene beds (ocean, forest, night...) are the sound you asked for, not
# a noise floor: their default sits much closer to the events.
AMBIENCE_BED_DB = -14.0


# Per-event renderer parameters a hand-written Spec may set in `params`.
# Like `bed_db` they are render-only: the caption grammar and the planner
# cannot express them, so a planner Spec never carries any. role -> name ->
# (min, max, what it does). A role that is not listed takes no params.
_PITCH = "pitch of the fundamental, Hz"
_RING = "ring time (T60) of the fundamental, s"
EVENT_PARAMS: dict[str, dict[str, tuple[float, float, str]]] = {
    "bell": {"f0": (110.0, 2400.0, _PITCH), "t60": (0.05, 8.0, _RING)},
    "ui_pop": {"f0": (200.0, 4000.0, _PITCH), "t60": (0.01, 1.0, _RING)},
    "click": {"f0": (500.0, 8000.0, _PITCH), "t60": (0.005, 0.5, _RING)},
    "impact": {"f0": (30.0, 400.0, _PITCH), "t60": (0.05, 3.0, _RING)},
    "sub_drop": {
        "f0": (40.0, 400.0, "pitch the drop starts on, Hz"),
        "f_end": (20.0, 200.0, "pitch the drop lands on, Hz"),
    },
    "church_bell": {
        "f0": (100.0, 1000.0, "strike note, Hz: half the bell's nominal. Unset, the seed picks the bell."),
        "ring": (0.2, 4.0, "scales every partial's ring time (1 = as cast)"),
        "hardness": (0.0, 1.0, "clapper hardness: 0 felt, 0.5 the shipped iron clapper, 1 a hard, bright blow"),
        "tierce": (0.55, 0.67, "the third above the prime, as a ratio to the nominal: 0.6 minor (a church bell), 0.63 major"),
        "beat": (0.0, 0.01, "how far apart each partial's doublet sits, as a fraction of its frequency (0.0012 as cast)"),
    },
}


def event_param_errors(role: str, params: object) -> list[str]:
    """Why `params` is not valid for `role`; empty when it is (or is None)."""
    if params is None:
        return []
    if not isinstance(params, dict):
        return ["params: must be an object of name -> number"]
    allowed = EVENT_PARAMS.get(role, {})
    errors = []
    for name in sorted(params, key=str):
        value = params[name]
        if name not in allowed:
            takes = f"takes {', '.join(sorted(allowed))}" if allowed else "takes no params"
            errors.append(f"params.{name}: unknown for role {role!r} (it {takes})")
            continue
        lo, hi, _ = allowed[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            errors.append(f"params.{name}: must be a finite number")
        elif not lo <= value <= hi:
            errors.append(f"params.{name}: {value:g} is outside [{lo:g}, {hi:g}]")
    return errors


def default_bed_db(texture: str) -> float:
    return AMBIENCE_BED_DB if texture in AMBIENCE_TEXTURES else DEFAULT_BED_DB

PAN_LABELS = ("L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4")
GAIN_LEVELS = 16  # -30 dB .. 0 dB in 2 dB steps
MAX_EVENTS = 8


def quantize_time(x: float) -> float:
    return round(float(x) * 100.0) / 100.0


def quantize_gain(x: float) -> float:
    return float(np.clip(round(float(x) / 2.0) * 2.0, -30.0, 0.0))


def pan_index(pan: float) -> int:
    return int(round((float(np.clip(pan, -1.0, 1.0)) + 1.0) * 0.5 * (len(PAN_LABELS) - 1)))


def pan_value(index: int) -> float:
    return float(index) / (len(PAN_LABELS) - 1) * 2.0 - 1.0


def pan_label(pan: float) -> str:
    return PAN_LABELS[pan_index(pan)]


def gain_index(gain_db: float) -> int:
    return int(np.clip(round((float(gain_db) + 30.0) / 2.0), 0, GAIN_LEVELS - 1))


def gain_value(index: int) -> float:
    return float(index) * 2.0 - 30.0


@dataclass
class SpecEvent:
    role: str
    t: float
    dur: float
    gain_db: float
    pan: float
    seed: int
    # Renderer parameters for this one event (see `EVENT_PARAMS`), e.g.
    # {"f0": 440.0} to put a bell on a note. None = the role's defaults.
    params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.role not in EVENT_ROLES:
            raise ValueError(f"unknown event role {self.role!r}")
        if self.seed < 0 or self.seed > 255:
            raise ValueError("event seed must be in [0, 255]")
        errors = event_param_errors(self.role, self.params)
        if errors:
            raise ValueError("; ".join(errors))
        if self.params is not None:
            self.params = {str(k): float(v) for k, v in self.params.items()} or None


@dataclass
class Spec:
    """What the planner emits, and what an agent may write by hand.

    `texture` is one of `TEXTURES`, or `"none"` for no bed at all. `bed_db`
    is the bed level relative to the foreground (None = `DEFAULT_BED_DB`).
    The planner only ever produces a texture from `TEXTURES` with
    `bed_db=None` and events without `params`; those options exist for
    hand-written Specs.
    """

    duration: float
    texture: str
    mood: str
    events: list[SpecEvent] = field(default_factory=list)
    bed_db: float | None = None

    def __post_init__(self) -> None:
        if self.texture not in BED_CHOICES:
            raise ValueError(f"unknown texture {self.texture!r}")
        if self.mood not in MOODS:
            raise ValueError(f"unknown mood {self.mood!r}")
        if len(self.events) > MAX_EVENTS:
            raise ValueError(f"at most {MAX_EVENTS} events")
        if self.texture == NO_BED and not self.events:
            raise ValueError('texture "none" needs at least one event (the cue would be silent)')
        if self.bed_db is not None:
            if isinstance(self.bed_db, bool) or not isinstance(self.bed_db, (int, float)):
                raise ValueError("bed_db must be a number")
            if not np.isfinite(self.bed_db) or not (BED_DB_RANGE[0] <= self.bed_db <= BED_DB_RANGE[1]):
                raise ValueError(f"bed_db must be in [{BED_DB_RANGE[0]:g}, {BED_DB_RANGE[1]:g}] dB")
            if self.texture == NO_BED:
                raise ValueError('bed_db has no effect with texture "none"; drop one of them')
            self.bed_db = float(self.bed_db)


# Scene templates: some training Specs are small, plausible scenes rather than
# independent random events, so the planner sees the new natural and foley
# roles in the orders people ask for them ("a storm", "someone walks up, the
# door slams, a knock"). The caption grammar is unchanged; only which Specs
# are sampled.
SCENES = ("storm", "arrival", "cave", "battle",
          "beach", "forest_morning", "night_camp", "kitchen", "office", "city_street", "sword_fight", "haunted_house")
SCENE_P = 0.25

# Each later scene: (textures, moods, [(role, how many (lo, hi), where (lo, hi)
# as fractions of the cue)]). Events land in random order inside their window.
_SCENE_RECIPES = {
    "beach": (("ocean",), ("calm", "warm", "hopeful"),
              [("wave_crash", (0, 2), (0.0, 0.8)), ("bird_chirp", (0, 2), (0.0, 0.9)),
               ("wind_gust", (0, 1), (0.0, 0.7)), ("splash", (0, 1), (0.1, 0.9))]),
    "forest_morning": (("forest",), ("calm", "warm", "hopeful", "bright"),
                       [("bird_chirp", (1, 3), (0.0, 0.85)), ("wind_gust", (0, 1), (0.0, 0.6)),
                        ("wood_creak", (0, 1), (0.1, 0.8)), ("footsteps", (0, 1), (0.0, 0.7))]),
    "night_camp": (("night", "fire"), ("calm", "eerie", "warm"),
                   [("crickets", (0, 1), (0.0, 0.3)), ("owl", (0, 2), (0.1, 0.9)),
                    ("frog", (0, 2), (0.0, 0.9)), ("wood_creak", (0, 1), (0.2, 0.8))]),
    "kitchen": (("room_cold", "city_hum"), ("warm", "neutral", "playful"),
                [("pouring_water", (0, 1), (0.0, 0.6)), ("can_opening", (0, 1), (0.0, 0.8)),
                 ("cork_pop", (0, 1), (0.0, 0.8)), ("glass_break", (0, 1), (0.3, 0.9)),
                 ("metal_clang", (0, 1), (0.0, 0.9)), ("water_drop", (0, 2), (0.0, 0.9))]),
    "office": (("room_cold", "city_hum"), ("neutral", "calm", "tense"),
               [("keyboard_typing", (1, 1), (0.0, 0.4)), ("mouse_click", (0, 2), (0.0, 0.9)),
                ("phone_ring", (0, 1), (0.2, 0.8)), ("paper_rustle", (0, 1), (0.0, 0.9)),
                ("clock_tick", (0, 1), (0.0, 0.5)), ("camera_shutter", (0, 1), (0.0, 0.9))]),
    "city_street": (("traffic", "city_hum"), ("neutral", "tense", "intense"),
                    [("car_horn", (0, 2), (0.0, 0.9)), ("siren", (0, 1), (0.0, 0.6)),
                     ("engine", (0, 2), (0.0, 0.8)), ("footsteps", (0, 1), (0.0, 0.7)),
                     ("church_bell", (0, 1), (0.0, 0.6))]),
    "sword_fight": (("wind", "fire", "room_cold"), ("intense", "tense", "dark"),
                    [("sword_clash", (2, 4), (0.0, 0.9)), ("whoosh", (0, 2), (0.0, 0.9)),
                     ("punch", (0, 2), (0.0, 0.9)), ("footsteps", (0, 1), (0.0, 0.6))]),
    "haunted_house": (("drone", "wind", "room_cold"), ("eerie", "dark", "tense"),
                      [("wood_creak", (1, 2), (0.0, 0.8)), ("door_slam", (0, 1), (0.2, 0.9)),
                       ("clock_tick", (0, 1), (0.0, 0.4)), ("heartbeat", (0, 1), (0.1, 0.7)),
                       ("owl", (0, 1), (0.0, 0.9)), ("footsteps", (0, 1), (0.0, 0.7))]),
}
_SCENE_ANCHOR = {"storm": "thunder", "arrival": "footsteps", "cave": "water_drop", "battle": "explosion",
                 "beach": "wave_crash", "forest_morning": "bird_chirp", "night_camp": "owl", "kitchen": "pouring_water",
                 "office": "keyboard_typing", "city_street": "car_horn", "sword_fight": "sword_clash",
                 "haunted_house": "wood_creak"}


def _event(rng: np.random.Generator, role: str, t: float, duration: float,
           dur: float | None = None, seed: int | None = None) -> SpecEvent | None:
    _, _, gain0, dur0 = EVENT_STYLE[role]
    if dur is None:
        dur = dur0 * rng.uniform(0.7, 1.4)
    t = quantize_time(max(0.0, t))
    dur = quantize_time(np.clip(dur, 0.02, min(9.99, duration - t)))
    if t >= duration or dur < 0.05:
        return None
    gain = quantize_gain(gain0 + rng.uniform(-4.0, 4.0))
    pan = pan_value(pan_index(rng.uniform(-1.0, 1.0)))
    return SpecEvent(role, t, dur, gain, pan, int(rng.integers(0, 256)) if seed is None else int(seed))


def sample_scene(rng: np.random.Generator, kind: str | None = None) -> Spec:
    """A Spec shaped like a small scene (see `SCENES`)."""
    kind = str(rng.choice(SCENES)) if kind is None else kind
    duration = float(rng.integers(12, 61)) / 2.0  # 6-30 s
    out: list[SpecEvent | None] = []
    if kind == "storm":
        texture = str(rng.choice(("rain", "wind")))
        mood = str(rng.choice(("dark", "tense", "eerie", "calm")))
        t = rng.uniform(0.0, 0.25 * duration)
        for _ in range(int(rng.integers(1, 4))):
            out.append(_event(rng, "thunder", t, duration, dur=rng.uniform(4.0, 9.0)))
            t += rng.uniform(3.0, 8.0)
        if rng.random() < 0.3:
            out.append(_event(rng, "water_drop", rng.uniform(0.0, duration - 0.5), duration))
    elif kind == "arrival":
        texture = str(rng.choice(("room_cold", "city_hum", "rain", "wind")))
        mood = str(rng.choice(("tense", "dark", "neutral", "eerie")))
        t = rng.uniform(0.0, 1.0)
        walk = rng.uniform(2.0, 5.0)
        out.append(_event(rng, "footsteps", t, duration, dur=walk))
        t += walk + rng.uniform(0.1, 0.6)
        out.append(_event(rng, "door_slam", t, duration))
        t += rng.uniform(0.8, 2.5)
        out.append(_event(rng, "knock", t, duration))
        if rng.random() < 0.3:
            out.append(_event(rng, str(rng.choice(("riser", "sub_drop", "swell"))), rng.uniform(min(t, duration), max(t, duration - 1.0)), duration))
    elif kind == "cave":
        texture = str(rng.choice(("drone", "wind", "room_cold")))
        mood = str(rng.choice(("eerie", "calm", "dark")))
        for _ in range(int(rng.integers(2, 7))):
            out.append(_event(rng, "water_drop", rng.uniform(0.0, duration - 0.5), duration))
    elif kind in _SCENE_RECIPES:
        textures, moods, parts = _SCENE_RECIPES[kind]
        texture = str(rng.choice(textures))
        mood = str(rng.choice(moods))
        for role, (lo, hi), (a, b) in parts:
            for _ in range(int(rng.integers(lo, hi + 1))):
                out.append(_event(rng, role, rng.uniform(a, b) * duration, duration))
    else:  # battle
        texture = str(rng.choice(("fire", "wind", "sci_fi", "city_hum")))
        mood = str(rng.choice(("intense", "tense", "dark")))
        t = rng.uniform(0.0, 0.3 * duration)
        for _ in range(int(rng.integers(1, 4))):
            out.append(_event(rng, "explosion", t, duration))
            t += rng.uniform(1.5, 6.0)
        for _ in range(int(rng.integers(0, 3))):
            role = str(rng.choice(("impact", "whoosh", "footsteps", "glitch", "sub_drop")))
            out.append(_event(rng, role, rng.uniform(0.0, duration - 0.5), duration))
    events = sorted((e for e in out if e is not None), key=lambda e: e.t)[:MAX_EVENTS]
    if not events:  # every event fell off the end; keep the scene's anchor
        events = [_event(rng, _SCENE_ANCHOR[kind], 0.0, duration)]
    return Spec(duration, texture, mood, events)


def sample_spec(rng: np.random.Generator) -> Spec:
    if rng.random() < SCENE_P:
        return sample_scene(rng)
    # Half-second steps from 2 s to 30 s: agents ask for short UI cues as well
    # as long beds, and the planner must copy any of them.
    duration = float(rng.integers(4, 61)) / 2.0
    texture = str(rng.choice(TEXTURES))
    mood = str(rng.choice(MOODS))
    # Agents ask for a single sound as often as for a full scene, so the
    # planner must see 0 (pure ambience) through MAX_EVENTS.
    n_events = int(rng.integers(0, MAX_EVENTS + 1))
    roles = [str(x) for x in rng.choice(EVENT_ROLES, size=max(1, n_events), replace=False)][:n_events]

    events: list[SpecEvent] = []
    for role in roles:
        _, _, gain0, dur0 = EVENT_STYLE[role]
        dur = quantize_time(np.clip(dur0 * rng.uniform(0.7, 1.4), 0.02, min(9.99, duration)))
        t = quantize_time(rng.uniform(0.0, max(0.0, duration - dur)))
        if t + dur > duration:
            t = quantize_time(duration - dur)
        gain = quantize_gain(gain0 + rng.uniform(-4.0, 4.0))
        pan = pan_value(pan_index(rng.uniform(-1.0, 1.0)))
        seed = int(rng.integers(0, 256))
        events.append(SpecEvent(role, t, dur, gain, pan, seed))

    events.sort(key=lambda e: e.t)
    return Spec(duration, texture, mood, events)


def _chord(root: int, mode: str, degree: int) -> list[int]:
    scale = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return [root + 12 * ((degree + k) // 7) + scale[(degree + k) % 7] for k in (0, 2, 4)]


def score_from_spec(spec: Spec, with_music: bool = False) -> Score:
    n_sections = 4
    sec_dur = spec.duration / n_sections
    roles = ("intro", "build", "peak", "outro")
    energy_scale = MOOD_ENERGY.get(spec.mood, 0.6)
    sections = [
        Section(
            i * sec_dur,
            sec_dur,
            role,
            min(1.0, SECTION_ENERGY[role] * energy_scale / 0.7),
            SECTION_DENSITY[role],
        )
        for i, role in enumerate(roles)
    ]

    music: list[MusicTrack] = []
    if with_music:
        # v2. v1 has no tempo/key/mode in the Spec, so music uses fixed defaults.
        root = 48
        bar = 4.0 * 60.0 / 120.0
        n_bars = max(1, int(spec.duration // bar))
        pad_notes: list[Note] = []
        bass_notes: list[Note] = []
        for b in range(n_bars):
            t = b * bar
            chord = _chord(root, "minor", (0, 5, 3, 4)[b % 4])
            for midi in chord:
                pad_notes.append(Note(t, bar * 0.98, midi, 0.45))
            bass_notes.append(Note(t, bar * 0.90, chord[0] - 12, 0.60))
        music = [
            MusicTrack("pad", pad_notes, gain_db=-17.0),
            MusicTrack("bass", bass_notes, gain_db=-15.0),
        ]

    if spec.texture == NO_BED:
        textures: list[Texture] = []
    else:
        bed_db = default_bed_db(spec.texture) if spec.bed_db is None else float(spec.bed_db)
        textures = [Texture(0.0, spec.duration, spec.texture, gain_db=bed_db, width=0.7)]

    events: list[Event] = []
    for se in spec.events:
        character, apex_frac, _, _ = EVENT_STYLE[se.role]
        dur = min(se.dur, spec.duration - se.t)
        if dur <= 0.02:
            continue
        apex = se.t + apex_frac * dur
        events.append(
            Event(
                se.t,
                se.role,
                dur,
                character,
                apex=apex,
                params=dict(se.params or {}),
                gain_db=se.gain_db,
                pan=se.pan,
                seed=se.seed,
            )
        )

    return Score(
        duration=spec.duration,
        sample_rate=48000,
        seed=0,
        tempo=120.0,
        key="C",
        mode="minor",
        mood=spec.mood,
        target_lufs=-20.0,
        sections=sections,
        music=music,
        textures=textures,
        events=events,
    )


def sample_score(rng: np.random.Generator, with_music: bool = True) -> Score:
    return score_from_spec(sample_spec(rng), with_music=with_music)


def caption(score: Score) -> str:
    """A short, agent-style description that determines the score."""
    parts = [f"{score.duration:.2f}-second {score.mood} cue"]
    tex = ", ".join(sorted({t.character.replace("_", " ") for t in score.textures}))
    # Training scores always have a bed, so "no bed" never reached the
    # planner; `parse_caption` reads it, and `orb.api.generate` renders such a
    # caption without the planner.
    parts.append(f"{tex} bed" if tex else "no bed")
    evs = []
    for e in score.events:
        role = e.role.replace("_", " ")
        evs.append(
            f"{role} @{e.t:.2f} {e.dur:.2f}s {e.gain_db:+.0f}dB {pan_label(e.pan)} #{e.seed:02x}"
        )
    if evs:
        parts.append("with " + ", ".join(evs))
    return ", ".join(parts)


def spec_from_score(score: Score) -> Spec:
    """Inverse of `score_from_spec` for the fields the Spec carries."""
    events = [
        SpecEvent(e.role, quantize_time(e.t), quantize_time(e.dur), quantize_gain(e.gain_db), e.pan, e.seed)
        for e in score.events
    ]
    if not score.textures:
        return Spec(score.duration, NO_BED, score.mood, events)
    tex = score.textures[0]
    # The training scores' level means "the planner's bed", which the Spec
    # writes as None (rendered at today's default), so held-out Specs still
    # compare equal to planner output.
    defaults = (DEFAULT_BED_DB, TRAIN_BED_DB) + ((AMBIENCE_BED_DB,) if tex.character in AMBIENCE_TEXTURES else ())
    bed_db = None if tex.gain_db in defaults else float(tex.gain_db)
    return Spec(score.duration, tex.character, score.mood, events, bed_db)


_ROLE_WORDS = "|".join(sorted((r.replace("_", " ") for r in EVENT_ROLES), key=len, reverse=True))
_TEX_WORDS = "|".join(sorted((t.replace("_", " ") for t in TEXTURES), key=len, reverse=True))
_EVENT_RE = re.compile(
    rf"({_ROLE_WORDS}) @(\d+\.\d{{2}}) (\d+\.\d{{2}})s ([+-]\d+)dB ({'|'.join(PAN_LABELS)}) #([0-9a-f]{{2}})"
)
_CAPTION_RE = re.compile(
    rf"(\d+\.\d{{2}})-second ({'|'.join(MOODS)}) cue"
    rf"(?:, (no|{_TEX_WORDS}) bed)?"
    rf"(?:, with ((?:{_EVENT_RE.pattern})(?:, (?:{_EVENT_RE.pattern}))*))?"
)


def parse_caption(text: str) -> Spec | None:
    """Read a caption written in the exact grammar of :func:`caption`.

    Returns None for anything else (free text, typos, extra clauses). A caption
    with ``no bed`` or no bed clause at all gives ``texture="none"``; the
    planner never saw such captions, so they are rendered from this Spec
    directly. Raises ValueError when the caption parses but its values are not
    a valid Spec (e.g. an event past the end of the cue).
    """
    m = _CAPTION_RE.fullmatch(text.strip())
    if m is None:
        return None
    duration, mood, bed, ev_text = m.group(1), m.group(2), m.group(3), m.group(4)
    texture = NO_BED if bed in (None, "no") else bed.replace(" ", "_")
    events = []
    for em in _EVENT_RE.finditer(ev_text or ""):
        role, t, dur, gain, pan, seed = em.groups()
        events.append(
            SpecEvent(
                role.replace(" ", "_"),
                float(t),
                float(dur),
                float(int(gain)),
                pan_value(PAN_LABELS.index(pan)),
                int(seed, 16),
            )
        )
    spec = Spec(float(duration), texture, mood, events)
    for e in spec.events:
        if not (0.0 <= e.t < spec.duration):
            raise ValueError(f"{e.role} @{e.t:.2f} is outside the {spec.duration:.2f} s cue")
    return spec

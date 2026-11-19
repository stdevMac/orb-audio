"""Everyday, nature and action sounds: the v0.3 library.

Each role is a small physical or acoustic model in numpy, built the same way
as the foley roles in `foley.py` (and rendered through the same
`render_foley` path: high-pass, smooth low-pass to 12 kHz after every
envelope and attack, peak normalisation, a dark room or outdoor reverb, fades
to exactly zero, constant-power pan). Only sounds that are not voices are
here: a dog, a cat, a laugh or a cough needs recordings or a learned model.

The seed picks a documented variant (`VARIANTS`) and seeds the noise. Every
model is deterministic.
"""

from __future__ import annotations

import functools

import numpy as np

from ..score import Event
from .dsp import LN1000, TAU, bandlimit, saturate
from .foley import _attack, _burst, _fade_out, _fftconv, _modes_ir, _norm, _pulse, _ring, _smooth, _bubble

ROLES = (
    "bird_chirp",
    "crickets",
    "wave_crash",
    "wind_gust",
    "splash",
    "frog",
    "owl",
    "glass_break",
    "clock_tick",
    "phone_ring",
    "keyboard_typing",
    "mouse_click",
    "camera_shutter",
    "paper_rustle",
    "cork_pop",
    "coin_drop",
    "metal_clang",
    "wood_creak",
    "heartbeat",
    "applause",
    "pouring_water",
    "can_opening",
    "clock_alarm",
    "punch",
    "sword_clash",
    "gunshot",
    "car_horn",
    "siren",
    "engine",
    "church_bell",
    "fireworks",
)

VARIANTS = {
    "bird_chirp": "seed % 4 = song: 0 sparrow chips, 1 trill, 2 warbled phrase, 3 two-note whistle",
    "crickets": "seed % 3 = 0 one cricket close, 1 two crickets, 2 a field chorus",
    "wave_crash": "seed % 3 = 0 small wave on sand, 1 breaking wave, 2 big wave on rocks",
    "wind_gust": "seed % 3 = 0 low buffeting gust, 1 howling gust, 2 gust through leaves",
    "splash": "seed % 3 = 0 small stone, 1 something heavier, 2 a big jump into water",
    "frog": "seed % 3 = 0 two-part ribbit, 1 deep bullfrog, 2 small tree frog peeps",
    "owl": "seed % 2 = 0 hoo hoo-hoo (tawny/great horned), 1 a single soft hoot",
    "glass_break": "seed % 3 = 0 a glass dropped on a hard floor, 1 a window pane, 2 a bottle smashed",
    "clock_tick": "seed % 3 = 0 wall clock (1 tick/s, tick-tock), 1 pocket watch (fast, small), 2 grandfather clock (slow, deep case)",
    "phone_ring": "seed % 4 = 0 classic bell phone (2 s on, 4 s off), 1 classic UK double ring, 2 modern marimba ringtone, 3 phone vibrating on a table",
    "keyboard_typing": "seed % 3 = 0 laptop keys, 1 mechanical keyboard, 2 old typewriter",
    "mouse_click": "seed % 2 = 0 single click, 1 double click",
    "camera_shutter": "seed % 3 = 0 DSLR single frame, 1 DSLR burst, 2 film camera with wind-on lever",
    "paper_rustle": "seed % 2 = 0 handling a sheet, 1 crumpling a sheet into a ball",
    "cork_pop": "seed % 2 = 0 wine cork, 1 champagne cork with fizz",
    "coin_drop": "seed % 3 = 0 one coin bouncing, 1 a coin spinning down, 2 a handful of coins into a dish",
    "metal_clang": "seed % 3 = 0 metal pipe, 1 bin lid or sheet, 2 anvil hit",
    "wood_creak": "seed % 3 = 0 creaking door, 1 floorboard, 2 old ship's timber",
    "heartbeat": "seed % 3 = tempo: 0 resting (62 bpm), 1 nervous (88 bpm), 2 racing (120 bpm)",
    "applause": "seed % 3 = 0 a few people, 1 a room, 2 a big audience",
    "pouring_water": "seed % 2 = 0 into a glass (pitch rises as it fills), 1 from a jug into a pot",
    "can_opening": "seed % 2 = 0 soda can, 1 beer can (bigger hiss)",
    "clock_alarm": "seed % 2 = 0 twin-bell wind-up alarm clock, 1 digital beeping alarm",
    "punch": "seed % 3 = 0 body punch, 1 face punch, 2 heavy movie punch",
    "sword_clash": "seed % 3 = 0 blade clash, 1 sword drawn from its sheath, 2 two quick clashes",
    "gunshot": "seed % 3 = 0 pistol, 1 rifle, 2 distant shot",
    "car_horn": "seed % 3 = 0 one honk, 1 two short honks, 2 long honk",
    "siren": "seed % 3 = 0 wail, 1 yelp, 2 two-tone hi-lo",
    "engine": "seed % 3 = 0 car idling, 1 car revving, 2 car passing by",
    "church_bell": "seed % 2 = 0 one strike, 1 tolling (a strike every 2.5 s)",
    "fireworks": "seed % 3 = 0 one shell (whistle and burst), 1 crackling shell, 2 distant show",
}

# f0 / width / reverb: what the mixer passes as `p`. A role may override the
# reverb per variant (outdoor vs room) by writing into `p`.
PARAMS: dict[str, dict[str, float]] = {
    "bird_chirp": dict(width=0.3, reverb=0.18, reverb_decay=0.9),
    "crickets": dict(width=0.5, reverb=0.12, reverb_decay=0.8),
    "wave_crash": dict(width=0.8, reverb=0.0),
    "wind_gust": dict(width=0.8, reverb=0.0),
    "splash": dict(width=0.4, reverb=0.14, reverb_decay=0.7),
    "frog": dict(width=0.2, reverb=0.16, reverb_decay=0.8),
    "owl": dict(width=0.2, reverb=0.3, reverb_decay=1.4),
    "glass_break": dict(width=0.6, reverb=0.2, reverb_decay=0.6),
    "clock_tick": dict(width=0.0, reverb=0.06, reverb_decay=0.4),
    "phone_ring": dict(width=0.2, reverb=0.16, reverb_decay=0.5),
    "keyboard_typing": dict(width=0.3, reverb=0.05, reverb_decay=0.3),
    "mouse_click": dict(width=0.0, reverb=0.03, reverb_decay=0.25),
    "camera_shutter": dict(width=0.1, reverb=0.05, reverb_decay=0.3),
    "paper_rustle": dict(width=0.4, reverb=0.06, reverb_decay=0.3),
    "cork_pop": dict(width=0.2, reverb=0.14, reverb_decay=0.5),
    "coin_drop": dict(width=0.2, reverb=0.16, reverb_decay=0.6),
    "metal_clang": dict(width=0.4, reverb=0.2, reverb_decay=1.0),
    "wood_creak": dict(width=0.3, reverb=0.16, reverb_decay=0.6),
    "heartbeat": dict(width=0.0, reverb=0.0),
    "applause": dict(width=0.9, reverb=0.2, reverb_decay=0.9),
    "pouring_water": dict(width=0.2, reverb=0.1, reverb_decay=0.4),
    "can_opening": dict(width=0.1, reverb=0.08, reverb_decay=0.4),
    "clock_alarm": dict(width=0.1, reverb=0.12, reverb_decay=0.5),
    "punch": dict(width=0.2, reverb=0.1, reverb_decay=0.5),
    "sword_clash": dict(width=0.4, reverb=0.18, reverb_decay=0.9),
    "gunshot": dict(width=0.5, reverb=0.3, reverb_decay=1.2),
    "car_horn": dict(width=0.2, reverb=0.18, reverb_decay=0.9),
    "siren": dict(width=0.3, reverb=0.22, reverb_decay=1.2),
    "engine": dict(width=0.3, reverb=0.08, reverb_decay=0.6),
    # ring / hardness / tierce / beat are the bell as cast; a Spec's params may move them
    # (`orb.grammar.EVENT_PARAMS`). `f0` has no default: unset, the seed picks the bell.
    "church_bell": dict(width=0.4, reverb=0.4, reverb_decay=2.0, ring=1.0, hardness=0.5, tierce=0.6, beat=0.0012),
    "fireworks": dict(width=0.7, reverb=0.25, reverb_decay=1.6),
}

# Roles that are a stable pitched tone by design (a whistle, a horn, a bell,
# a cricket's song). Their high band carries real harmonics, so the
# "no whistle line" and "no buzzing envelope" checks that guard the noisy
# roles do not apply; the Nyquist check still does.
TONAL = ("bird_chirp", "crickets", "owl", "phone_ring", "clock_alarm", "car_horn", "siren",
         "church_bell", "metal_clang", "sword_clash", "glass_break", "coin_drop", "frog")


# -- helpers -----------------------------------------------------------------


def _n(ev: Event, sr: int) -> int:
    return max(1, int(ev.dur * sr))


def _tone(freq: np.ndarray, sr: int, phase: float = 0.0) -> np.ndarray:
    """A sine following the instantaneous frequency curve `freq` (Hz)."""
    return np.sin(phase + TAU * np.cumsum(freq) / sr)


def _hann_env(n: int, at: int, length: int, rise: float = 0.3) -> np.ndarray:
    """0 outside [at, at+length); a raised-cosine rise over `rise` of it, then
    a raised-cosine fall over the rest."""
    out = np.zeros(n)
    k = max(2, min(length, n - at))
    if at >= n or k < 2:
        return out
    a = max(1, int(rise * k))
    b = max(1, k - a)
    e = np.ones(k)
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(a) + 0.5) / a)
    e[a:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 0.5) / b)
    out[at : at + k] = e
    return out


def _strike(n: int, sr: int, at: int, freqs, t60s, gains, rng: np.random.Generator,
            contact: float = 0.0008, noise: float = 0.0) -> np.ndarray:
    """A struck object: a short force pulse (and optional noise) into a bank of
    modes. Unlike `_ring`, the force is not differentiated, so a contact this
    short already carries the highs."""
    exc = _pulse(n, at, max(2, int(contact * sr)))
    if noise > 0:
        exc = exc + noise * _burst(rng, n, at, int(4 * contact * sr) + 2, contact, sr, rise=0.0005)
    return _fftconv(exc, _modes_ir(freqs, t60s, gains, sr, n), n)


def _grains(rng: np.random.Generator, n: int, times: np.ndarray, amps: np.ndarray, sr: int) -> np.ndarray:
    """Impulses at `times` (s) with `amps`, random sign."""
    out = np.zeros(n)
    idx = (np.asarray(times) * sr).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(out, idx[ok], np.asarray(amps)[ok] * rng.choice((-1.0, 1.0), int(ok.sum())))
    return out


def _decorrelate(x: np.ndarray, sr: int, crng: np.random.Generator, taps: int = 6, span: float = 0.012) -> np.ndarray:
    """A second ear: a few short, weak, darkened echoes (a little space), so a
    mono source gets some width without a comb."""
    n = len(x)
    out = x.copy()
    for _ in range(taps):
        d = int(crng.uniform(0.002, span) * sr)
        if d < n:
            out[d:] += crng.uniform(0.05, 0.15) * x[: n - d]
    return out


def _bounces(first_gap: float, ratio: float, count: int, crng: np.random.Generator) -> np.ndarray:
    """Times of a bouncing object: each gap shrinks by `ratio`."""
    t = [0.0]
    g = first_gap
    for _ in range(count - 1):
        t.append(t[-1] + g * crng.uniform(0.92, 1.08))
        g *= ratio
        if g < 0.004:
            break
    return np.array(t)


def _bubble_into(out: np.ndarray, sr: int, at: int, f0: float, damp_scale: float, rise: float, amp: float) -> None:
    """`amp * foley._bubble(...)` added into `out` in place, computing only the
    samples the bubble occupies (a stream has thousands of them)."""
    n = len(out)
    if at >= n or at < 0:
        return
    d = (0.043 * f0 + 0.0014 * f0**1.5) * damp_scale
    k = min(n - at, int(LN1000 / d * 1.2 * sr))
    t = np.arange(k) / sr
    ph = TAU * np.cumsum(f0 * (1.0 + rise * d * t)) / sr
    a = min(k, max(2, int(0.002 * sr)))
    env = np.exp(-d * t)
    env[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, a))
    out[at : at + k] += amp * np.sin(ph) * env


def _harmonic(f: np.ndarray, sr: int, amps, phase: float = 0.0) -> np.ndarray:
    """A sum of harmonics k*f with amplitudes `amps` (k = 1..), each muted
    above 0.45 sr."""
    ph = phase + TAU * np.cumsum(f) / sr
    out = np.zeros(len(f))
    for k, a in enumerate(amps, start=1):
        if a == 0:
            continue
        mask = (k * f) < 0.45 * sr
        out += a * np.sin(k * ph) * mask
    return out


# =============================================================================
# Nature
# =============================================================================


def _anim_env(k: int, rise: float, power: float = 1.0) -> np.ndarray:
    """A k-sample bump: raised-cosine rise over `rise` of it, cos^2 fall over
    the rest, both reaching exactly 0 at the ends."""
    a = int(np.clip(rise * k, 1, k - 1))
    b = k - a
    e = np.empty(k)
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(a) + 0.5) / a)
    e[a:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 0.5) / b)
    return e**power if power != 1.0 else e


def _anim_lowpass(x: np.ndarray, sr: int, hi: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth-magnitude low-pass on a power-of-two FFT (the
    far-off layers of the animal roles: air absorption)."""
    n = len(x)
    nfft = 1 << (n + 2048 - 1).bit_length()
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return np.fft.irfft(np.fft.rfft(x, nfft) / np.sqrt(1.0 + (f / hi) ** (2 * order)), nfft)[:n]


def _anim_add(out: np.ndarray, at: int, y: np.ndarray) -> None:
    """Add `y` into `out` starting at sample `at`, clipped to the buffer."""
    n = len(out)
    if at < 0 or at >= n:
        return
    k = min(len(y), n - at)
    out[at : at + k] += y[:k]


def _bird_note(sr: int, kind: str, f_a: float, f_b: float, length: float, crng: np.random.Generator,
               h2: float = 0.2) -> np.ndarray:
    """One syllable from the syrinx: a fast frequency sweep (birds sweep
    kilohertz in tens of ms), its 2nd/3rd harmonics, an asymmetric amplitude
    bump. `kind`: down (steep falling chip), up, arch (inverted U), whistle
    (near steady), buzz (fast FM, so sidebands: a rough, burry note)."""
    k = max(16, int(length * sr))
    u = np.arange(k) / (k - 1)
    t = np.arange(k) / sr
    if kind == "down":
        f = f_b + (f_a - f_b) * (1.0 - u) ** 2.2
        env = _anim_env(k, 0.12)
    elif kind == "up":
        f = f_a + (f_b - f_a) * u**1.6
        env = _anim_env(k, 0.6)
    elif kind == "arch":
        f = f_a * (1.0 - u) + f_b * u + crng.uniform(0.12, 0.3) * max(f_a, f_b) * np.sin(np.pi * u) ** 1.3
        env = _anim_env(k, 0.35)
    elif kind == "buzz":
        f = (f_a + (f_b - f_a) * u) * (1.0 + crng.uniform(0.04, 0.08) * np.sin(TAU * crng.uniform(90, 170) * t))
        env = _anim_env(k, 0.2, 0.6)
    else:  # whistle
        f = f_a + (f_b - f_a) * u
        env = _anim_env(k, 0.12, 0.5)
    f = f * (1.0 + 0.005 * np.sin(TAU * crng.uniform(25, 60) * t + crng.uniform(0, TAU)))
    ph = TAU * np.cumsum(f) / sr + crng.uniform(0, TAU)
    y = np.sin(ph) + h2 * np.sin(2 * ph + 0.4) + 0.3 * h2 * np.sin(3 * ph + 1.1) * (3 * f < 0.45 * sr)
    # Louder where the sweep is higher: the syrinx's output rises with tension.
    return y * env * (f / f.mean()) ** 0.7


def _bird_song(n: int, sr: int, kind: int, crng: np.random.Generator, start: float, base: float,
               pause: float = 1.0) -> np.ndarray:
    """One bird singing song type `kind` (see VARIANTS) from `start` s."""
    out = np.zeros(n)
    dur = n / sr
    t = start
    while t < dur - 0.08:
        at = int(t * sr)
        if kind == 0:  # sparrow: steep "chip"s and arched "cheep"s, in irregular runs
            for _ in range(int(crng.integers(2, 7))):
                if crng.random() < 0.55:
                    L, fa = crng.uniform(0.025, 0.045), crng.uniform(5200, 6600) * base
                    y = _bird_note(sr, "down", fa, fa * crng.uniform(0.55, 0.7), L, crng, h2=0.15)
                else:
                    L, fa = crng.uniform(0.05, 0.09), crng.uniform(2900, 3700) * base
                    y = _bird_note(sr, "arch", fa, fa * crng.uniform(0.8, 1.0), L, crng, h2=crng.uniform(0.3, 0.5))
                _anim_add(out, at, crng.uniform(0.6, 1.0) * y)
                at += int((L + crng.uniform(0.06, 0.2)) * sr)
            t = at / sr + crng.uniform(0.15, 0.7) * pause
        elif kind == 1:  # trill: an intro note, then a fast run of identical down-chips
            fa = crng.uniform(5200, 6400) * base
            _anim_add(out, at, 0.7 * _bird_note(sr, "whistle", fa * 0.72, fa * 0.68, 0.09, crng, h2=0.05))
            at += int(crng.uniform(0.14, 0.2) * sr)
            reps = int(crng.integers(12, 26))
            L = crng.uniform(0.018, 0.028)
            period = 1.0 / crng.uniform(12, 18)
            drop = crng.uniform(0.5, 0.65)
            for r in range(reps):
                g = min(1.0, 0.45 + r / 5.0) * crng.uniform(0.85, 1.0)
                _anim_add(out, at, g * _bird_note(sr, "down", fa, fa * drop, L, crng, h2=0.18))
                at += int(period * crng.uniform(0.97, 1.03) * sr)
            t = at / sr + crng.uniform(0.6, 1.4) * pause
        elif kind == 2:  # warbled phrase: varied slurred notes, a burry one now and then
            for _ in range(int(crng.integers(4, 9))):
                shape = str(crng.choice(("down", "up", "arch", "arch", "buzz")))
                L = crng.uniform(0.05, 0.13) if shape != "down" else crng.uniform(0.03, 0.06)
                fa = crng.uniform(2200, 4200) * base
                fb = float(np.clip(fa * crng.uniform(0.65, 1.4), 1800, 6000))
                _anim_add(out, at, crng.uniform(0.55, 1.0) * _bird_note(sr, shape, fa, fb, L, crng, h2=crng.uniform(0.12, 0.35)))
                at += int((L + crng.uniform(0.025, 0.07)) * sr)
            t = at / sr + crng.uniform(0.4, 1.0) * pause
        else:  # two-note whistle "fee-bee": a pure high note, then one ~15% lower
            f1 = crng.uniform(3700, 4200) * base
            _anim_add(out, at, _bird_note(sr, "whistle", f1 * 1.01, f1 * 0.985, crng.uniform(0.24, 0.3), crng, h2=0.04))
            at += int(crng.uniform(0.34, 0.4) * sr)
            f2 = f1 * crng.uniform(0.82, 0.87)
            _anim_add(out, at, 0.85 * _bird_note(sr, "whistle", f2 * 1.03, f2, crng.uniform(0.26, 0.34), crng, h2=0.04))
            t = at / sr + 0.3 + crng.uniform(0.9, 1.6) * pause
    return out


def _bird_chirp(ev, sr, rng, crng, p, seed):
    """Birdsong outdoors: the singer close, plus one or two other birds
    farther off (quieter, darker from air absorption, their own timing), as
    in any real recording of birds."""
    n = _n(ev, sr)
    kind = seed % 4
    near = _bird_song(n, sr, kind, crng, 0.01, crng.uniform(0.88, 1.12))
    far_l = np.zeros(n)
    far_r = np.zeros(n)
    for _ in range(int(crng.integers(1, 3))):
        other = int(crng.choice((0, 1, 2)))
        far = _bird_song(n, sr, other, crng, crng.uniform(0.1, 1.2), crng.uniform(0.8, 1.15), pause=1.6)
        pos = crng.uniform(0.1, 0.9)
        g = crng.uniform(0.12, 0.28)
        far_l += g * (1.0 - pos) * 2 * far
        far_r += g * pos * 2 * far
    cut = crng.uniform(4500, 6500)  # air absorption takes the highs off the far birds
    p["reverb"], p["reverb_decay"] = 0.22, 1.1
    return near + _anim_lowpass(far_l, sr, cut), _decorrelate(near, sr, crng) + _anim_lowpass(far_r, sr, cut)


def _cricket_voice(n: int, sr: int, crng: np.random.Generator, fc: float, trill: bool, h2: float) -> np.ndarray:
    """One male cricket. Each wing closure drags the scraper over the file's
    teeth at the harp's own frequency (an escapement), so a pulse is a burst
    of a near-pure carrier (falling slightly as the stroke slows) with weak
    harmonics. Field crickets group 3-5 pulses into a chirp; tree and ground
    crickets trill (pulses on and on)."""
    out = np.zeros(n)
    dur = n / sr
    rate = crng.uniform(25, 34) if not trill else crng.uniform(38, 60)
    L = (0.55 if not trill else 0.5) / rate
    k = int(L * sr)
    tt = np.arange(k) / sr
    f = fc * (1.0 + 0.02 * (0.5 - tt / L))
    ph = TAU * np.cumsum(f) / sr
    pulse = (np.sin(ph) + h2 * np.sin(2 * ph + 0.5) + 0.3 * h2 * np.sin(3 * ph + 1.3)) * _anim_env(k, 0.25, 0.8)
    if trill:
        level = 0.35 + 0.65 * _smooth(crng, n, sr, crng.uniform(0.3, 0.8))
        tm = crng.uniform(0.0, 0.3)
        while tm < dur:
            at = int(tm * sr)
            _anim_add(out, at, level[min(at, n - 1)] * crng.uniform(0.85, 1.0) * pulse)
            tm += crng.uniform(0.97, 1.03) / rate
        return out
    per = int(crng.integers(3, 6))
    shape = 0.65 + 0.35 * np.sin(np.pi * (np.arange(per) + 0.6) / (per + 0.2))
    chirp_period = 1.0 / crng.uniform(1.6, 3.0)
    tm = crng.uniform(0.0, chirp_period)
    while tm < dur:
        for j in range(per):
            _anim_add(out, int((tm + j / rate) * sr), shape[j] * crng.uniform(0.88, 1.0) * pulse)
        tm += chirp_period * crng.uniform(0.92, 1.08)
    return out


def _crickets(ev, sr, rng, crng, p, seed):
    """Crickets at night: 0 one field cricket close, 1 two answering each
    other, 2 a field chorus; always with fainter distant singers behind, as a
    real field is never just one insect."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    if kind == 0:
        near = [(1.0, 0.5)]
    elif kind == 1:
        near = [(1.0, 0.3), (crng.uniform(0.5, 0.8), 0.72)]
    else:
        near = [(crng.uniform(0.45, 1.0), crng.uniform(0.15, 0.85)) for _ in range(3)]
    far_count, far_level = ((3, 0.07), (4, 0.1), (10, 0.22))[kind]
    voices = [(g, pos, crng.uniform(4300, 4900), False, 0.18) for g, pos in near]
    for _ in range(far_count):
        trill = bool(crng.random() < 0.45)
        fc = crng.uniform(2800, 3800) if trill else crng.uniform(4100, 5000)
        voices.append((far_level * crng.uniform(0.4, 1.0), crng.uniform(0.05, 0.95), fc, trill, 0.06))
    left = np.zeros(n)
    right = np.zeros(n)
    for g, pos, fc, trill, h2 in voices:
        y = g * _cricket_voice(n, sr, crng, fc, trill, h2)
        left += y * (1.0 - pos) * 2
        right += y * pos * 2
    p["reverb"], p["reverb_decay"] = (0.1, 0.15, 0.25)[kind], 0.9
    fo = _fade_out(n, sr, min(0.2, 0.3 * dur))
    return left * fo, right * fo


def _wash(rng, n, sr, lo, hi, order=2):
    return _norm(bandlimit(rng.standard_normal(n), sr, lo, hi, order=order))


def _water_fast(m):
    """The smallest 2^a 3^b 5^c >= m: an FFT length numpy does quickly (a
    clip length with a large prime factor costs ten times as much)."""
    best = 1 << max(0, int(m - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < m:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


_WATER_MAGS: dict = {}


def _water_mag(m, sr, lo, hi, order):
    """The Butterworth-shaped band magnitude of `dsp.bandlimit` on an
    m-point rfft grid (cached: the same few bands are used over and over)."""
    key = (m, sr, lo, hi, order)
    mag = _WATER_MAGS.get(key)
    if mag is None:
        f = np.fft.rfftfreq(m, 1.0 / sr)
        mag = np.ones_like(f)
        if hi is not None:
            mag /= np.sqrt(1.0 + (f / max(hi, 1.0)) ** (2 * order))
        if lo is not None:
            mag /= np.sqrt(1.0 + (np.maximum(f, 1e-9) / max(lo, 1.0)) ** (-2 * order))
        if len(_WATER_MAGS) > 64:
            _WATER_MAGS.clear()
        _WATER_MAGS[key] = mag
    return mag


def _water_band(x, sr, lo=None, hi=None, order=2, pad=2048):
    """`dsp.bandlimit` (same zero-phase magnitude, same padding against
    wrap-around) at a fast FFT length."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = _water_fast(n + 2 * pad)
    return np.fft.irfft(np.fft.rfft(np.pad(x, (pad, 0)), m) * _water_mag(m, sr, lo, hi, order), m)[pad : pad + n]


def _water_noise(rng, n, sr, lo, hi, order=2):
    """Band-limited Gaussian noise, peak-normalised: drawn directly as a
    random spectrum (one inverse FFT), circular so it needs no padding."""
    m = _water_fast(n)
    spec = rng.standard_normal(m // 2 + 1) + 1j * rng.standard_normal(m // 2 + 1)
    return _norm(np.fft.irfft(spec * _water_mag(m, sr, lo, hi, order), m)[:n])


def _water_bubbles(n, sr, times, f0s, amps, damp=1.0, rise=0.1, dec=2, floor=5.5):
    """Many Minnaert bubbles at once: f(t) = f0 (1 + rise d t), amplitude
    exp(-d t), d = damp (0.043 f0 + 0.0014 f0^1.5) (van den Doel 2005), the
    same model as `foley._bubble`. A stream or a breaking wave has thousands
    of them, so they are drawn in vectorised batches (grouped by length) at
    sr/dec and band-limited back up to sr; each one is cut `floor` nepers
    (48 dB) down and opens over 1 ms."""
    times = np.asarray(times, dtype=np.float64)
    f0s = np.broadcast_to(np.asarray(f0s, dtype=np.float64), times.shape)
    amps = np.broadcast_to(np.asarray(amps, dtype=np.float64), times.shape)
    sr2 = sr / dec
    n2 = _water_fast(-(-n // dec) + 64)
    out2 = np.zeros(n2)
    # the glide must stay under the reduced Nyquist
    f0s = np.minimum(f0s, 0.42 * sr2 / (1.0 + floor * rise))
    ok = (times >= 0) & (times < n / sr) & (amps > 0) & (f0s > 20)
    times, f0s, amps = times[ok], f0s[ok], amps[ok]
    if len(times):
        d = (0.043 * f0s + 0.0014 * f0s**1.5) * damp
        L = np.maximum(16, np.ceil(floor / d * sr2)).astype(int)
        cls = np.ceil(np.log2(L)).astype(int)
        start = (times * sr2).astype(int)
        r = max(2, int(0.001 * sr2))
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
        for c in np.unique(cls):
            idx = np.nonzero(cls == c)[0]
            Lc = 1 << int(c)
            tt = np.arange(Lc) / sr2
            step = max(1, 1_500_000 // Lc)
            for b in range(0, len(idx), step):
                j = idx[b : b + step]
                dj = d[j, None]
                y = np.sin(TAU * f0s[j, None] * (tt + 0.5 * rise * dj * tt * tt)) * np.exp(-dj * tt) * amps[j, None]
                y[:, :r] *= ramp
                pos = start[j, None] + np.arange(Lc)
                m = pos < n2
                out2 += np.bincount(pos[m], weights=y[m], minlength=n2)
    N = n2 * dec
    Z = np.zeros(N // 2 + 1, dtype=complex)
    Y = np.fft.rfft(out2)
    Z[: len(Y)] = Y
    return np.fft.irfft(Z, N)[:n] * dec


def _water_times(crng, rate, sr, cap=200_000):
    """Poisson event times (s) for a rate curve `rate` (events/s per sample)."""
    cum = np.cumsum(rate) / sr
    total = float(cum[-1]) if len(cum) else 0.0
    k = int(min(cap, crng.poisson(max(total, 0.0))))
    if k == 0:
        return np.zeros(0)
    u = np.sort(crng.uniform(0.0, total, k))
    return np.interp(u, cum, np.arange(len(rate)) / sr)


def _water_crackle(rng, crng, n, sr, rate, lo, hi, alpha=1.6, order=2):
    """Crackle: impulses at Poisson times following `rate`, heavy-tailed
    (Pareto `alpha`) amplitudes, random sign, band-limited to [lo, hi]. The
    sound of many droplets landing, bubbles bursting, pebbles knocking."""
    tt = _water_times(crng, rate, sr)
    amps = np.minimum(crng.pareto(alpha, len(tt)) + 1.0, 30.0)
    g = _grains(rng, n, tt, amps, sr)
    return _water_band(g, sr, lo, hi, order=order)


def _water_radii(crng, k, r_lo, r_hi, slope=2.5):
    """Bubble radii (mm) from a power law ~ r^-slope on [r_lo, r_hi]
    (breaking waves and plunging jets: slope 2-3.5)."""
    u = crng.uniform(0.0, 1.0, k)
    a = 1.0 - slope
    return (r_lo**a + u * (r_hi**a - r_lo**a)) ** (1.0 / a)


def _water_resonate(x, sr, f1, peaks, q, gain, n_fft=1024, hop=256):
    """Filter `x` through a moving air column: a quarter-wave pipe (open at
    the rim, closed by the water) has peaks at the odd multiples of f1(t).
    `f1` is a curve over the clip; `peaks` the number of odd modes; `q` their
    sharpness; `gain` their height over the dry level."""
    from .dsp import istft, stft

    X, n = stft(x, n_fft, hop)
    nf = X.shape[0]
    fr = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f = np.interp(np.arange(nf) * hop, np.arange(len(f1)), f1)[:, None]
    G = np.ones((nf, len(fr)))
    for k in range(peaks):
        fk = (2 * k + 1) * f
        G += gain / (k + 1) ** 0.7 / (1.0 + ((fr[None, :] - fk) / (fk / q)) ** 2)
    return istft(X * G, n_fft, hop, length=n)


def _wave_crash(ev, sr, rng, crng, p, seed):
    """One wave reaching the shore. Surf is the sound of air in water: the
    crest spills (a growing hiss over a low rumble), the lip plunges and
    traps a cloud of large bubbles (a broadband roar with a low thump), the
    foam runs up the beach as a long wash of tiny bubbles fizzing and
    bursting, and the water drains back: a soft hiss over sand, a clatter of
    rolling pebbles on a stony beach."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    size = (0.55, 1.0, 1.35)[kind]
    t_b = min(0.3 * dur, crng.uniform(0.3, 0.55) * (0.7 + 0.3 * size))
    after = np.maximum(t - t_b, 0.0)
    pre = np.clip(t / max(t_b, 1e-3), 0.0, 1.0) ** 2
    # the roar: up over 60-150 ms, gone in ~0.4-0.9 s
    tau_r = 0.25 + 0.45 * size
    roar_env = _attack(n, sr, t_b - 0.02, 0.06 + 0.07 * size) * np.exp(-after / tau_r)
    # the wash: rises with the run-up, fades slowly with a few surges
    tau_w = max(0.6, 0.5 * dur)
    surge = 0.7 + 0.3 * _smooth(crng, n, sr, 2.5)
    wash_env = (_attack(n, sr, t_b, 0.25) * np.exp(-after / tau_w) * surge + 0.25 * pre) / 1.25
    # backwash: the water drains back once the run-up stops
    t_bk = t_b + crng.uniform(0.35, 0.5) * dur
    bk = np.clip((t - t_bk) / 0.5, 0.0, None)
    back_env = bk * np.exp(1.0 - bk)
    tail = _fade_out(n, sr, min(0.25 * dur, 0.8))

    # trapped air: big bubbles in the plunge (roar), small ones in the foam
    def cloud(rate_env, rate, r_lo, r_hi, slope, amp_pow, damp):
        tt = _water_times(crng, rate * rate_env, sr)
        r = _water_radii(crng, len(tt), r_lo, r_hi, slope)
        a = r**amp_pow * np.interp(tt, t, rate_env) * crng.uniform(0.3, 1.0, len(tt))
        return _water_bubbles(n, sr, tt, 3260.0 / r, a, damp=damp, rise=0.05, dec=4 if r_lo > 1.0 else 2)

    big = _norm(cloud(roar_env, 900 * size, 1.5, 14.0, 2.2, 0.8, 1.4))
    small = _norm(cloud(wash_env, 500.0, 0.6, 2.2, 2.8, 0.5, 1.2))
    thump = _water_band(_burst(rng, n, int(t_b * sr), int(0.5 * sr), 0.06 + 0.06 * size, sr, rise=0.02), sr, 30.0, 160.0, order=2)
    thump = (0.12 if kind == 0 else 0.3 * size) * _norm(thump)

    def chan(common):
        # turbulence: brown-ish rumble and a pinkish roar
        rumble = _water_noise(rng, n, sr, 60.0, 400.0) * (0.5 * pre + roar_env + 0.4 * wash_env)
        roar_n = _water_noise(rng, n, sr, 250.0, 3000.0) * roar_env
        # the plunge hits: crackle of impacts, dense
        splat = _water_crackle(rng, crng, n, sr, 6000.0 * roar_env + 1.0, 300.0, 7000.0, alpha=1.8)
        # the foam: many tiny bursts, a hiss, all fading with the wash
        fizz = _water_crackle(rng, crng, n, sr, 20000.0 * wash_env + 1.0, 1500.0, 9000.0, alpha=3.0)
        hiss = _water_noise(rng, n, sr, 400.0, 8000.0) * wash_env
        out = common + (0.35 * rumble + 0.9 * size * roar_n + 0.45 * size * _norm(splat)
               + 0.5 * hiss + 0.3 * _norm(fizz))
        if kind == 2:
            # pebbles rolled back down the beach: hard knocks, a clatter
            clat = _water_crackle(rng, crng, n, sr, 900.0 * back_env + 1.0, 900.0, 6500.0, alpha=1.3, order=4)
            out = out + 0.8 * _norm(clat) * np.minimum(1.0, 4.0 * back_env)
        else:
            # sand: the drain is a soft fizzing hiss
            drain = _water_crackle(rng, crng, n, sr, 5000.0 * back_env + 1.0, 2500.0, 9000.0, alpha=2.5)
            out = out + 0.25 * _norm(drain) * back_env
        # foam and distance soften the top: surf falls ~4-6 dB/octave above 1-2 kHz
        return _water_band(out, sr, None, 4500.0, order=1)

    common = 0.45 * size * big + 0.4 * small + thump
    return chan(common) * tail, chan(common) * tail


def _air_fastlen(m: int) -> int:
    """The smallest 2^a 3^b 5^c >= m (an FFT length without a large prime)."""
    best = 1 << (m - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            k = p35 << max(0, (-(-m // p35) - 1).bit_length())
            best = min(best, k)
            p35 *= 3
        p5 *= 5
    return best


def _air_band(x, sr, *bands, split=False, pad=2048):
    """`dsp.bandlimit` for one or more (lo, hi, order, gain) bands at once,
    summed, on the last axis of `x` (so a (2, n) stereo pair filters in one
    pass), with an FFT length free of large primes. Same zero-phase
    Butterworth magnitudes and padding as `bandlimit`. `split=True` returns
    each band on its own (one forward FFT for all of them)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    m = _air_fastlen(n + 2 * pad)
    xp = np.zeros(x.shape[:-1] + (m,))
    xp[..., pad : pad + n] = x
    X = np.fft.rfft(xp, axis=-1)
    f = np.fft.rfftfreq(m, 1.0 / sr)
    mags = []
    for lo, hi, order, gain in bands:
        b = np.full_like(f, gain)
        if hi is not None:
            b /= np.sqrt(1.0 + (f / max(hi, 1.0)) ** (2 * order))
        if lo is not None:
            b /= np.sqrt(1.0 + (np.maximum(f, 1e-9) / max(lo, 1.0)) ** (-2 * order))
        mags.append(b)
    if split:
        return [np.fft.irfft(X * b, m, axis=-1)[..., pad : pad + n] for b in mags]
    return np.fft.irfft(X * np.sum(mags, axis=0), m, axis=-1)[..., pad : pad + n]


def _air_narrow(rng, n, sr, f, bw):
    """Narrow-band noise of about `bw` Hz around the frequency curve `f` (Hz):
    complex noise low-passed to bw/2 and turned up to f. A resonance driven
    by turbulence (a howl over an edge, a firework whistle), not a pure tone."""
    z = _air_band(rng.standard_normal((2, n)), sr, (None, 0.5 * bw, 2, 1.0))
    z = z[0] + 1j * z[1]
    z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)
    return np.real(z * np.exp(1j * TAU * np.cumsum(f) / sr))


def _air_doppler(x, sr, dist, c=343.0):
    """`x` as heard from `dist(t)` metres away: the travel time dist/c is a
    moving delay (the Doppler shift falls out of it), relative to the closest
    point so nothing is pushed out of the clip."""
    n = len(x)
    i = np.arange(n, dtype=np.float64)
    d = (dist - dist.min()) / c * sr
    return np.interp(i - d, i, x, left=0.0)


def _air_absorb(x, sr, near, f_dark=2500.0):
    """Air absorption: the far source keeps its lows and loses its highs;
    `near(t)` in [0, 1] is how much of the top survives."""
    dark = _air_band(x, sr, (None, f_dark, 2, 1.0))
    return dark + np.clip(near, 0.0, 1.0) * (x - dark)


def _wind_gust(ev, sr, rng, crng, p, seed):
    """Wind is turbulence, not a tone: a breeze that never quite stops, one
    gust swelling over it, and eddies at every time scale (slow surges, quick
    buffets). The faster the air, the louder, and the brighter through the
    edge hiss: rumble ~ u^1.5, rushing band ~ u^1.3, hiss ~ u^2.5. The
    howling gust adds narrow resonances over edges whose pitch follows the
    speed (Strouhal), and only above a threshold; the leafy gust adds the rustle of leaves, many
    tiny random contacts whose rate climbs steeply with the speed."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    peak = crng.uniform(0.3, 0.7) * dur
    gust = np.where(t < peak, np.sin(0.5 * np.pi * t / peak) ** 2,
                    np.cos(0.5 * np.pi * np.clip((t - peak) / max(dur - peak, 1e-3), 0, 1)) ** 2)
    base = crng.uniform(0.55, 0.7)  # the wind that is always there
    eddies = 0.5 * _smooth(crng, n, sr, 0.9) + 0.3 * _smooth(crng, n, sr, 2.5) + 0.2 * _smooth(crng, n, sr, 7.0) - 0.5
    u = np.clip(base + (1.0 - base) * gust + 0.6 * eddies, 0.05, None)
    u = u / u.max()
    shape = _attack(n, sr, 0.0, min(0.15, 0.05 * dur)) * _fade_out(n, sr, min(0.3, 0.1 * dur))
    if kind == 0:  # pressure slaps: eddies the size of the listener's head hitting it
        buffet = 0.3 + 2.0 * _smooth(crng, n, sr, 5.0) ** 4
    else:
        buffet = 0.7 + 0.3 * _smooth(crng, n, sr, 3.0)
    f0 = crng.uniform(380.0, 620.0)
    ratio = crng.uniform(1.35, 1.6)
    whistle = np.clip((u - 0.4) / 0.6, 0.0, 1.0) ** 1.5 * (0.6 + 0.4 * _smooth(crng, n, sr, 1.5))

    rumble, rush, hiss = (y / np.abs(y).max(axis=-1, keepdims=True) for y in _air_band(
        rng.standard_normal((2, n)), sr, (20.0, 220.0, 2, 1.0), (150.0, 1200.0, 2, 1.0), (900.0, 3000.0, 2, 1.0), split=True))
    rumble = rumble * u**1.5 * buffet
    rush = rush * u**1.3  # the broadband rush is steadier than the rumble
    hiss = hiss * u**2.5
    if kind == 0:
        out = 1.0 * rumble + 0.3 * rush + 0.04 * hiss
    elif kind == 1:
        f = f0 * (0.55 + 0.6 * u)
        howl = _air_narrow(rng, n, sr, f, 16.0) + 0.45 * _air_narrow(rng, n, sr, f * ratio, 24.0)
        out = 0.8 * rumble + 0.45 * rush + 0.07 * hiss + 0.3 * _norm(howl) * whistle
    else:
        lam = 2500.0 * u**3  # leaf contacts per second
        hits = rng.random((2, n)) < lam / sr
        g = np.zeros((2, n))
        g[hits] = rng.lognormal(0.0, 0.6, int(hits.sum())) * rng.choice((-1.0, 1.0), int(hits.sum()))
        ticks, sheet = _air_band(np.stack([g, rng.standard_normal((2, n))]), sr, (1200.0, 7000.0, 2, 1.0), (1500.0, 6000.0, 2, 1.0), split=True)
        ticks = ticks[0] / np.abs(ticks[0]).max()
        sheet = sheet[1] / np.abs(sheet[1]).max() * u**3 * (0.5 + 0.5 * _smooth(crng, n, sr, 18.0))
        out = 0.8 * (0.8 * rumble + 0.45 * rush + 0.07 * hiss) + 0.3 * ticks + 0.15 * sheet
    return out[0] * shape, out[1] * shape


def _splash(ev, sr, rng, crng, p, seed):
    """Something enters water. The impact is a slap (a short broadband
    crack and, for anything heavy, the thump of displaced water); a crown of
    water is thrown up and hisses; the cavity behind the object pinches off
    into a big bubble (the "plunk") and a gurgle of smaller ones; then the
    spray falls back as a patter of droplets, each a click and often a tiny
    bubble. A big jump throws a jet that collapses into a second splash."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    size = (0.35, 0.75, 1.4)[kind]
    t0 = 0.008
    a0 = int(t0 * sr)
    after = np.maximum(t - t0, 0.0)
    slap = _water_band(_burst(rng, n, a0, int(0.12 * sr), 0.003 + 0.008 * size, sr, rise=0.0007), sr, 200.0, 9000.0, order=2)
    thump = _water_band(_burst(rng, n, a0, int(0.35 * sr), 0.015 + 0.05 * size, sr, rise=0.003), sr, 45.0, 400.0, order=2)
    # the crown: a spray hiss
    crown_env = _attack(n, sr, t0, 0.006) * np.exp(-after / (0.03 + 0.1 * size))
    crown = _water_noise(rng, n, sr, 700.0, 9000.0) * crown_env
    # cavity pinch-off: radius grows with the object
    t_c = t0 + (0.03 + 0.1 * size) * crng.uniform(0.85, 1.15)
    r_c = (5.0, 10.0, 26.0)[kind] * crng.uniform(0.85, 1.15)
    plunk = _water_bubbles(n, sr, [t_c], [3260.0 / r_c], [1.0], damp=0.8, rise=0.25, dec=2)
    gk = int((25 + 70 * size) * crng.uniform(0.8, 1.2))
    gt = t_c - 0.01 + crng.gamma(1.5, 0.05 + 0.08 * size, gk)
    gr = _water_radii(crng, gk, 1.0, 4.0 + 6.0 * size, 2.0)
    gurgle = _water_bubbles(n, sr, gt, 3260.0 / gr, gr**0.7 * np.exp(-(gt - t_c) / (0.15 + 0.2 * size)), damp=1.1, rise=0.15)
    # fallback: droplet impacts, a hump of patter over ~0.1 .. 0.4 + 0.6 size s
    t_p = 0.06 + 0.08 * size
    u = after / t_p
    hump = np.where(u < 1.0, u**2, np.exp(-(after - t_p) / (0.1 + 0.25 * size))) * (t > t0 + 0.02)
    if kind == 2:
        # the Worthington jet falls back: a second, smaller splash
        t2 = t0 + crng.uniform(0.32, 0.42)
        u2 = np.maximum(t - t2, 0.0) / 0.25
        hump = hump + 0.7 * u2 * np.exp(1.0 - u2) * (t > t2)
        slap = slap + 0.35 * _water_band(_burst(rng, n, int(t2 * sr), int(0.1 * sr), 0.012, sr, rise=0.002), sr, 300.0, 8000.0, order=2)
    rate = (250.0 + 500.0 * size) * hump
    pat = _water_crackle(rng, crng, n, sr, rate + 0.5, 600.0, 9000.0, alpha=1.5)
    dt = _water_times(crng, 0.35 * rate, sr)
    dr = _water_radii(crng, len(dt), 0.5, 2.0, 2.0)
    drops = _water_bubbles(n, sr, dt, 3260.0 / dr, crng.uniform(0.2, 1.0, len(dt)) * np.interp(dt, t, hump), damp=1.0, rise=0.2)
    slosh = _water_noise(rng, n, sr, 250.0, 1500.0) * _attack(n, sr, t0, 0.04) * np.exp(-after / (0.12 + 0.25 * size))
    mono = (_norm(slap) + (0.15 + 0.3 * size) * _norm(thump) + 0.55 * crown
            + (0.35 + 0.25 * size) * _norm(plunk) + 0.5 * _norm(gurgle)
            + 0.35 * _norm(pat) + 0.25 * _norm(drops) + 0.3 * slosh)
    mono = mono * _fade_out(n, sr, min(0.2, 0.15 * dur))
    return mono, _decorrelate(mono, sr, crng)


def _frog_glottis(n: int, sr: int, crng: np.random.Generator, notes, jitter: float = 0.02,
                  shimmer: float = 0.15) -> np.ndarray:
    """A frog's larynx: air pushed from the lungs makes glottal pulses at the
    vocal-fold rate f0, and a note may gate them into pulse groups (the
    rattle of a "rib", the growl of a bullfrog). Returns the impulse train;
    `_frog_sac` rings the throat and vocal sac with it.

    `notes`: (t0 s, length s, f0 start, f0 end, pulse rate Hz (0 = none),
    pulse depth 0-1, rise fraction, amplitude)."""
    exc = np.zeros(n)
    for t0, L, fa, fb, pr, depth, rise, amp in notes:
        at, k = int(t0 * sr), int(L * sr)
        if at >= n or k < 32:
            continue
        k = min(k, n - at)
        u = np.arange(k) / k
        f0 = (fa + (fb - fa) * u) * (1.0 + jitter * (2.0 * _smooth(crng, k, sr, 80.0) - 1.0))
        idx = np.nonzero(np.diff(np.floor(np.cumsum(f0) / sr)) > 0)[0] + 1
        env = _anim_env(k, rise, 0.7)
        if pr > 0:
            env = env * (1.0 - depth + depth * np.sin(np.pi * pr * u * L) ** 4)
        exc[at + idx] += amp * env[idx] * crng.uniform(1.0 - shimmer, 1.0, len(idx))
    return exc


def _frog_sac(exc: np.ndarray, sr: int, formants, width: float) -> np.ndarray:
    """Each glottal pulse is a smooth puff `width` s long (wider = darker),
    which rings the throat and inflated vocal sac: a bank of resonances
    (Hz, t60 s, gain)."""
    fr, t60, g = zip(*formants)
    ir = _modes_ir(fr, t60, g, sr, len(exc))
    k = max(2, int(width * sr))
    ir = np.convolve(ir, np.sin(np.pi * (np.arange(k) + 0.5) / k))
    return _fftconv(exc, ir, len(exc))


def _frog_pulse(sr: int, fd: float, length: float, crng: np.random.Generator, low: float = 0.35) -> np.ndarray:
    """One call pulse of a tree / chorus frog: a short burst of the dominant
    carrier (the larynx's vibration, tuned by the vocal sac), a weaker band
    an octave down and a faint 2nd harmonic, rising fast and dying away."""
    k = max(16, int(length * sr))
    t = np.arange(k) / sr
    ph = TAU * np.cumsum(fd * (1.0 + 0.04 * t / length)) / sr + crng.uniform(0, TAU)
    y = np.sin(ph) + 0.15 * np.sin(2 * ph + 0.5) + low * np.sin(0.5 * ph + 0.9)
    env = (1.0 - np.exp(-t / 0.0012)) * np.exp(-t / (0.4 * length))
    return y * env * _fade_out(k, sr, 0.25 * length)


def _frog_ribbit(n: int, sr: int, crng: np.random.Generator, out: np.ndarray, start: float, gain: float) -> None:
    """A chorus frog calling "rib-bit" over and over: two pulse trains (a
    frog's note is a run of discrete pulses, 50-100 per second, not a held
    tone), the second faster and a little higher; each note swells."""
    dur = n / sr
    fd = crng.uniform(1700, 2400)
    rate = crng.uniform(55, 75)
    period = crng.uniform(0.6, 1.0)
    t = start
    while t < dur - 0.25 or (t == start and t < dur - 0.05):
        cs = t
        for f, r, count, off in ((fd, rate, int(crng.integers(5, 9)), 0.0),
                                 (fd * crng.uniform(1.04, 1.1), rate * crng.uniform(1.1, 1.3), int(crng.integers(8, 13)), None)):
            if off is None:
                t += crng.uniform(0.03, 0.06)
            for j in range(count):
                a = gain * (0.45 + 0.55 * (j + 1) / count) * crng.uniform(0.85, 1.0)
                _anim_add(out, int(t * sr), a * _frog_pulse(sr, f * crng.uniform(0.98, 1.02), 0.7 / r, crng))
                t += crng.uniform(0.95, 1.05) / r
        t = max(t + 0.15, cs + period * crng.uniform(0.85, 1.2))


def _frog_peeps(n: int, sr: int, crng: np.random.Generator, out: np.ndarray, start: float, gain: float) -> None:
    """A spring peeper: a short, pure, upslurred "peep" near 3 kHz about
    once a second (a tiny frog, a single high whistle of a call)."""
    dur = n / sr
    f0 = crng.uniform(2650, 3100)
    period = crng.uniform(0.7, 1.3)
    t = start
    while t < dur - 0.15 or (t == start and t < dur - 0.05):
        L = crng.uniform(0.08, 0.13)
        k = int(L * sr)
        u = np.arange(k) / k
        ph = TAU * np.cumsum(f0 * (0.86 + 0.14 * u**0.7)) / sr
        y = (np.sin(ph) + 0.08 * np.sin(2 * ph)) * _anim_env(k, 0.5, 0.8)
        _anim_add(out, int(t * sr), gain * crng.uniform(0.85, 1.0) * y)
        t += period * crng.uniform(0.85, 1.15)


def _frog(ev, sr, rng, crng, p, seed):
    """Frogs at a pond: 0 a chorus frog's two-part "rib-bit" (two pulse
    trains), 1 a bullfrog's deep "rumm" (a low glottal rate, a growl, a low
    sac resonance plus a ~1.3 kHz one), 2 spring peepers' short rising
    "peep"s. Frogs call in choruses: the near caller is joined by neighbours
    of its kind farther off (quieter, duller), each on its own clock."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    others = (int(crng.integers(2, 4)), 1, int(crng.integers(5, 9)))[kind]
    near = np.zeros(n)
    far_l, far_r = np.zeros(n), np.zeros(n)
    if kind == 1:
        s_near, s_far = crng.uniform(0.92, 1.08), crng.uniform(0.9, 1.1)
        for c in range(1 + others):
            f0 = crng.uniform(95, 125)
            t = 0.03 if c == 0 else crng.uniform(0.9, 1.6)
            notes = []
            for _ in range(int(crng.integers(2, 4)) if c == 0 else 2):
                if t + 0.1 > dur:
                    break
                L = crng.uniform(0.5, 0.75)
                notes.append((t, min(L, dur - t - 0.02), f0 * 1.03, f0 * 0.95, crng.uniform(18, 28), 0.6, 0.12, 1.0))
                t += L + crng.uniform(0.3, 0.55)
            exc = _frog_glottis(n, sr, crng, notes, jitter=0.05, shimmer=0.4)
            if c == 0:
                near = exc
            else:
                g, pos = crng.uniform(0.25, 0.4), crng.uniform(0.05, 0.95)
                far_l += g * (1.0 - pos) * 2 * exc
                far_r += g * pos * 2 * exc

        def sac(s):
            return ((210 * s, 0.035, 1.0), (430 * s, 0.025, 0.45), (1300 * s, 0.012, 0.7), (2300 * s, 0.008, 0.18))

        near = _frog_sac(near, sr, sac(s_near), 0.0016)
        far_l = _frog_sac(far_l, sr, sac(s_far), 0.0016)
        far_r = _frog_sac(far_r, sr, sac(s_far), 0.0016)
    else:
        call = _frog_ribbit if kind == 0 else _frog_peeps
        call(n, sr, crng, near, 0.02, 1.0)
        for _ in range(others):
            y = np.zeros(n)
            call(n, sr, crng, y, crng.uniform(0.0, 0.9), crng.uniform(0.2, 0.45) if kind == 0 else crng.uniform(0.2, 0.9))
            pos = crng.uniform(0.05, 0.95)
            far_l += (1.0 - pos) * 2 * y
            far_r += pos * 2 * y
    scale = 1.0 / max(float(np.max(np.abs(near))), 1e-9)
    near *= scale
    # The neighbours are farther off: duller (air absorption).
    left = near + _anim_lowpass(far_l * scale, sr, 5000.0)
    right = _decorrelate(near, sr, crng) + _anim_lowpass(far_r * scale, sr, 5000.0)
    p["reverb"], p["reverb_decay"] = 0.18, 0.9
    return left, right


def _owl_env(k: int, sr: int, attack: float, release: float) -> np.ndarray:
    """Raised-cosine attack, a held middle, raised-cosine release."""
    a = int(np.clip(attack * sr, 1, k // 2))
    r = int(np.clip(release * sr, 1, k - a))
    e = np.ones(k)
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(a) + 0.5) / a)
    e[k - r :] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    return e


def _owl_hoot(sr: int, f0: float, L: float, crng, rng, tremolo: float = 0.0) -> np.ndarray:
    """One hoot. An owl hoots with its bill shut and throat puffed: a low,
    near-sinusoidal syrinx tone (weak harmonics), a soft swell in, a small
    rise into pitch at the onset and a droop at the end, a slow wobble, and
    breath turbulence that rides the airflow (noise modulated by the tone, so
    it sits around the harmonics, not as hiss). The breath speaks a moment
    before the tone does: the aspirated "h" of "h'hoo"."""
    k = max(64, int(L * sr))
    t = np.arange(k) / sr
    u = np.arange(k) / k
    f = f0 * (0.9 + 0.1 * (1.0 - np.exp(-t / 0.06))) * (1.0 - 0.08 * u**2.5)
    f = f * (1.0 + 0.006 * (2.0 * _smooth(crng, k, sr, 10.0) - 1.0))
    ph = TAU * np.cumsum(f) / sr + crng.uniform(0, TAU)
    tone = np.sin(ph) + 0.3 * np.sin(2 * ph + 0.6) + 0.12 * np.sin(3 * ph + 1.2) + 0.05 * np.sin(4 * ph + 2.0)
    breath = _norm(_anim_lowpass(rng.standard_normal(k), sr, 500.0, order=4)) * (0.5 + 0.5 * np.sin(ph))
    y = tone + 0.2 * breath
    env = _owl_env(k, sr, min(0.09, 0.3 * L), min(0.16, 0.4 * L))
    if tremolo > 0:  # the tawny's quavering last note
        env *= 1.0 - tremolo * (0.5 + 0.5 * np.sin(TAU * crng.uniform(9, 13) * t)) * np.clip((u - 0.25) / 0.3, 0, 1)
    m = int(0.07 * sr)
    h = m // 2
    hb = _norm(_anim_lowpass(rng.standard_normal(m), sr, 900.0, order=4)) * np.sin(np.pi * np.arange(m) / m) ** 2 * 0.25
    y = np.concatenate([hb[:h], y * env])
    y[h:m] += hb[h:m]
    return y


def _owl(ev, sr, rng, crng, p, seed):
    """Owl hoots at night, a little way off in the trees: 0 a hooting phrase
    (great horned "hoo, h'hoo-hoo, hoo, hoo" or tawny "hooo... hu, hu-hooooo"
    with its quaver), 1 a single soft hoot. A phrase too long for the slot is
    squeezed to fit."""
    n = _n(ev, sr)
    dur = n / sr
    if seed % 2 == 0:
        if crng.random() < 0.5:  # great horned
            f0 = crng.uniform(270, 340)
            pattern = ((0.0, 0.30, 0.85, 0.0), (0.45, 0.15, 0.7, 0.0), (0.62, 0.42, 1.0, 0.0),
                       (1.25, 0.30, 0.75, 0.0), (1.72, 0.34, 0.7, 0.0))
        else:  # tawny
            f0 = crng.uniform(380, 460)
            pattern = ((0.0, 0.6, 1.0, 0.0), (1.15, 0.12, 0.6, 0.0), (1.38, 0.1, 0.6, 0.0), (1.52, 0.6, 0.9, 0.45))
        gap = crng.uniform(2.5, 3.5)
    else:
        f0 = crng.uniform(300, 380)
        pattern = ((0.0, crng.uniform(0.5, 0.75), 1.0, 0.0),)
        gap = crng.uniform(2.2, 3.2)
    span = pattern[-1][0] + pattern[-1][1]
    sc = min(1.0, (dur - 0.08) / span) if dur > 0.12 else 0.0
    mono = np.zeros(n)
    start = 0.03
    while sc > 0 and start + span * sc < dur:
        for off, L, a, trem in pattern:
            _anim_add(mono, int((start + off * sc) * sr), a * _owl_hoot(sr, f0 * crng.uniform(0.99, 1.01), L * sc, crng, rng, trem))
        start += span * sc + gap
    p["reverb"], p["reverb_decay"] = 0.22, 1.3
    return mono, _decorrelate(mono, sr, crng, span=0.03)


# =============================================================================
# Everyday
# =============================================================================


def _bells_place(out, at, x, amp=1.0):
    """Add `amp * x` into `out` from sample `at` on, clipped to `out`."""
    n = len(out)
    if at >= n or at < 0 or amp == 0:
        return
    k = min(len(x), n - at)
    out[at : at + k] += amp * x[:k]


def _bells_modes(freqs, t60s, gains, sr, length=None):
    """The impulse response of a bank of damped sine modes, each computed only
    for as long as it rings (to -66 dB)."""
    t60s = np.maximum(np.asarray(t60s, dtype=float), 1e-3)
    big = int(1.1 * t60s.max() * sr) + 1
    m = big if length is None else min(length, big)
    out = np.zeros(m)
    for f, d, g in zip(freqs, t60s, gains):
        if not 20.0 < f < 0.45 * sr or g == 0:
            continue
        k = min(m, int(1.1 * d * sr) + 1)
        out[:k] += g * _bells_mode(f, d, sr, k)
    return out


def _bells_mode(f, t60, sr, k, block=2048):
    """exp(-LN1000 t / t60) sin(2 pi f t) for k samples, cheaply: one block
    of complex exponentials, then each later block is the first one times a
    single complex factor (no per-sample exp or sin)."""
    z = (-LN1000 / max(t60, 1e-3) + 1j * TAU * f) / sr
    b = min(block, k)
    base = np.exp(z * np.arange(b))
    nb = -(-k // b)
    steps = np.exp(z * b * np.arange(nb))
    return (steps[:, None] * base[None, :]).ravel()[:k].imag


def _bells_clicks(n, sr, times, amps, lo, rng):
    """Contact clicks: an impulse high-passed at `lo` (zero phase), placed at
    each of `times` (s) with `amps` and a random sign. The same as band
    limiting a train of `_grains`, without a filter over the whole clip."""
    h = 512
    kern = np.zeros(2 * h)
    kern[h] = 1.0
    kern = bandlimit(kern, sr, lo, None, order=2, pad=h) * np.hanning(2 * h)
    out = np.zeros(n + 2 * h)
    signs = rng.choice((-1.0, 1.0), len(times))
    for tm, a, sg in zip(times, amps, signs):
        at = int(tm * sr)
        if 0 <= at < n:
            out[at : at + 2 * h] += a * sg * kern
    return out[h : h + n]


def _bells_doublets(freqs, t60s, crng, split):
    """Every partial of a real (slightly asymmetric) round body is a pair of
    modes a little apart: they beat. Returns the two halves separately so
    the ears can hear different mixes of them."""
    freqs = np.asarray(freqs, dtype=float)
    d = freqs * split * crng.uniform(0.3, 1.0, len(freqs))
    fa, fb = freqs - 0.5 * d, freqs + 0.5 * d
    ta = np.asarray(t60s, dtype=float) * crng.uniform(0.9, 1.1, len(freqs))
    tb = np.asarray(t60s, dtype=float) * crng.uniform(0.9, 1.1, len(freqs))
    return fa, fb, ta, tb


def _glass_frag(sr, crng, f0, t60):
    """One glass fragment: a small, stiff, high-Q plate with a few inharmonic
    modes, the higher ones dying sooner."""
    r = np.array((1.0, crng.uniform(1.3, 1.55), crng.uniform(1.9, 2.4), crng.uniform(2.6, 3.2), crng.uniform(3.4, 4.2)))
    g = np.array((1.0, 0.8, 0.6, 0.45, 0.3)) * crng.uniform(0.4, 1.0, 5)
    return _bells_modes(f0 * r, t60 * np.array((1.0, 0.8, 0.6, 0.5, 0.4)), g, sr)


def _glass_break(ev, sr, rng, crng, p, seed):
    """Glass shattering. The blow cracks the glass in a few milliseconds (a
    sharp broadband crack and the body's own ring, cut short as it fails);
    then the pieces land, each fragment a small high-Q plate that rings at
    its own pitch every time it hits the floor and bounces (big pieces lower
    and longer, splinters high and short); last, a trickle of tiny bits
    tinkling to rest. A pane's pieces fall first and crash down a few tenths
    of a second later."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    t0 = 0.004
    at0 = int(t0 * sr)
    crack = np.zeros(n)
    _bells_place(crack, at0, bandlimit(_burst(rng, int(0.08 * sr), 0, int(0.07 * sr), (0.01, 0.016, 0.012)[kind], sr, rise=0.0003), sr, 500.0, None, order=2, pad=1024))
    fb = (crng.uniform(1500, 2800), crng.uniform(500, 1000), crng.uniform(800, 1500))[kind]
    body = np.zeros(n)
    _bells_place(body, at0, _bells_modes(fb * np.array((1.0, 2.1, 3.6, 5.2)), (0.08, 0.06, 0.045, 0.035), (1, 0.7, 0.5, 0.3), sr))
    thud = np.zeros(n)
    _bells_place(thud, at0, bandlimit(_burst(rng, int(0.2 * sr), 0, int(0.1 * sr), (0.012, 0.01, 0.025)[kind], sr, rise=0.002), sr, 60.0, 500.0, order=2, pad=2048))

    left = np.zeros(n)
    right = np.zeros(n)
    contacts = []
    count = (45, 60, 45)[kind]
    land = crng.uniform(0.28, 0.42)  # pane: the fall to the ground
    for _ in range(count):
        size = crng.random() ** 1.6  # most fragments are small
        f0 = np.exp(np.log(7800.0) * (1.0 - size) + np.log((2200.0, 1500.0, 1300.0)[kind]) * size) * crng.uniform(0.9, 1.1)
        t60 = (0.025 + 0.3 * size) * crng.uniform(0.7, 1.3)
        ir = _glass_frag(sr, crng, f0, t60)
        if kind == 1 and crng.random() < 0.7:
            tm = t0 + land + crng.gamma(2.0, 0.035)
        else:
            tm = t0 + crng.gamma(1.3, 0.05)
        gap = crng.uniform(0.04, 0.15) * (0.6 + 0.8 * size)
        ratio = crng.uniform(0.45, 0.65)
        a = crng.uniform(0.4, 1.0) * (0.5 + size) * np.exp(-max(0.0, tm - t0 - (land if kind == 1 else 0.0)) / 0.5)
        w = crng.uniform(0.15, 0.85)
        for _b in range(int(crng.integers(2, 6))):
            if tm > dur - 0.06:
                break
            at = int(tm * sr)
            _bells_place(left, at, ir, a * (1.0 - w))
            _bells_place(right, at, ir, a * w)
            contacts.append((tm, a))
            tm += gap * crng.uniform(0.85, 1.15)
            gap *= ratio
            a *= crng.uniform(0.35, 0.6)
    # splinters: dozens of tiny bits, very high and very short
    tiny = [_glass_frag(sr, crng, crng.uniform(6500, 9000), crng.uniform(0.008, 0.025)) for _ in range(6)]
    span = (0.5, 0.8, 0.6)[kind]
    for _ in range((140, 220, 160)[kind]):
        tm = t0 + ((land if kind == 1 and crng.random() < 0.6 else 0.0) + crng.gamma(1.2, span / 4.0))
        if tm > dur - 0.04:
            continue
        a = crng.uniform(0.1, 0.35) * np.exp(-(tm - t0) / (1.2 * span))
        w = crng.uniform(0.1, 0.9)
        ir = tiny[int(crng.integers(0, 6))]
        _bells_place(left, int(tm * sr), ir, a * (1.0 - w))
        _bells_place(right, int(tm * sr), ir, a * w)
        contacts.append((tm, 0.5 * a))
    # the hit itself is a click (a few hundred microseconds of contact)
    ct = np.array([c[0] for c in contacts])
    ca = np.array([c[1] for c in contacts])
    clicks = _bells_clicks(n, sr, ct, ca, 2500.0, rng)
    m = max(np.abs(left).max(), np.abs(right).max(), 1e-12)
    clicks = 0.9 * clicks / max(np.abs(clicks).max(), 1e-12)
    base = 0.9 * _norm(crack) + 0.3 * _norm(body) + (0.45 if kind == 2 else 0.25) * _norm(thud) + clicks
    return base + 1.4 * left / m, base + 1.4 * right / m


def _clock_tick(ev, sr, rng, crng, p, seed):
    """The escapement: an escape-wheel tooth landing on a steel pallet is a
    cluster of two or three tiny metal impacts a few ms apart (lock, impulse,
    drop), broadband and dead within ~10 ms. The brass plates of the
    movement ring on briefly with a few high, sparse partials, and the case
    the movement is screwed into radiates most of it, so the tick sits in
    the 0.5-2 kHz range (as in real recordings) rather than up at 8 kHz. Tick and tock come from the two pallets, so each has
    its own fixed character, and a clock is rarely in perfect beat. A pocket
    watch beats 5 times a second with almost no case; a grandfather clock
    beats once a second into a tall wooden case."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    period = (1.0, 0.2, 1.0)[kind]
    band = ((1000.0, 6500.0), (2000.0, 8000.0), (800.0, 5500.0))[kind]
    t60 = (0.012, 0.006, 0.015)[kind]
    # case: (lo, hi, t60, modes, level against the escapement)
    case = ((400.0, 2200.0, 0.02, 10, 0.8), (1200.0, 4000.0, 0.006, 6, 0.2), (150.0, 1500.0, 0.05, 12, 1.0))[kind]
    # the movement's plates: (lo, hi, t60, partials, level)
    plate = ((2500.0, 8000.0, 0.05, 7, 0.3), (3500.0, 8500.0, 0.03, 5, 0.3), (1800.0, 6500.0, 0.05, 8, 0.2))[kind]
    gap = ((0.002, 0.01), (0.0006, 0.0016), (0.003, 0.012))[kind]
    movement = int(crng.integers(1 << 31))
    beat = crng.uniform(0.0, 0.04) if kind != 1 else 0.0  # out of beat
    pal = [int(crng.integers(1 << 31)) for _ in range(2)]
    out = np.zeros(n)
    tm = crng.uniform(0.005, 0.03)
    i = 0
    while tm < dur - 0.04:
        tock = i % 2
        at = int(tm * sr)
        lag = 0.0
        for j in range(2 if kind == 1 else 3):
            g = _desk_hit(sr, rng, 0.012 if kind == 1 else 0.02, band, t60, 16, contact=0.00006,
                          noise=1.5, noise_tau=0.002, mseed=pal[tock] + j)
            a = (1.0, crng.uniform(0.3, 0.55), crng.uniform(0.12, 0.3))[j]
            _desk_add(out, at + int(lag * sr), g, a * (0.8 if tock else 1.0))
            lag += crng.uniform(*gap)
        cg = _desk_hit(sr, rng, case[2] * 3.0, case[:2], case[2], case[3], contact=0.0004,
                       noise=0.3, noise_tau=0.001, mseed=pal[tock] + 9)
        _desk_add(out, at, cg, case[4] * crng.uniform(0.9, 1.0))
        pg = _desk_hit(sr, rng, plate[2] * 2.5, plate[:2], plate[2], plate[3], contact=0.0001,
                       noise=0.5, noise_tau=0.0003, mseed=movement + tock)
        _desk_add(out, at, pg, plate[4] * crng.uniform(0.85, 1.0))
        tm += period * crng.uniform(0.998, 1.002) * (1.0 + (beat if tock else -beat))
        i += 1
    return out, out


def _bell_ring(n, sr, at, length, f, rng, crng, rate=20.0, partials=None):
    """An electromechanical bell: a clapper striking a metal gong `rate`
    times per second for `length` s (one train of blows into the gong's
    modes)."""
    partials = partials or (1.0, 2.76, 5.4, 8.9)
    t60 = (0.6, 0.35, 0.2, 0.12)
    w = max(2, int(0.0004 * sr))
    pulse = np.sin(np.pi * (np.arange(w) + 0.5) / w)
    m = 4 * w + 2
    tt = np.arange(m) / sr
    rise = np.minimum(1.0, 0.5 - 0.5 * np.cos(np.pi * np.minimum(tt / 0.0005, 1.0)))
    exc = np.zeros(n)
    for i in range(int(length * rate)):
        a_ = at + int(i / rate * sr)
        if a_ >= n:
            break
        a = crng.uniform(0.85, 1.0)
        _bells_place(exc, a_, pulse, a)
        _bells_place(exc, a_, 0.2 * rng.standard_normal(m) * np.exp(-tt / 0.0004) * rise, a)
    return _fftconv(exc, _bells_modes(f * np.array(partials), t60, (1.0, 0.6, 0.35, 0.2), sr), n)


def _phone_vibrate(n, sr, rng, crng):
    """A phone vibrating on a table: an eccentric-mass motor (~150-190 Hz)
    spins up in ~60 ms and shakes the phone, which chatters against the
    tabletop once per turn: a click train through the phone's case and the
    table, plus the motor's own hum. Bursts of ~0.8 s."""
    dur = n / sr
    fm = crng.uniform(150, 190)
    table = _bells_modes(crng.uniform(170, 260) * np.array((1.0, 2.3, 3.9, 6.1)), (0.1, 0.07, 0.05, 0.035), (1, 0.7, 0.5, 0.3), sr)
    case = _bells_modes(crng.uniform(900, 1400) * np.array((1.0, 1.8, 2.9, 4.3)), (0.02, 0.015, 0.012, 0.01), (1, 0.6, 0.4, 0.25), sr)
    on = crng.uniform(0.7, 1.0)
    cycle = on + crng.uniform(0.8, 1.1)
    exc = np.zeros(n)
    hum = np.zeros(n)
    start = 0.01
    while start < dur - 0.2:
        at = int(start * sr)
        k = min(int(on * sr), n - at - int(0.02 * sr))
        if k < int(0.1 * sr):
            break
        tt = np.arange(k) / sr
        spin = 1.0 - np.exp(-tt / 0.06)  # spin-up
        stop = np.exp(-np.clip(tt - (k / sr - 0.06), 0.0, None) / 0.02)  # coasts down when cut
        f = fm * spin * (0.6 + 0.4 * stop) * (1.0 + 0.01 * np.sin(TAU * 3.0 * tt))
        ph = np.cumsum(f) / sr
        idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0] + 1
        amp = (spin[idx] * stop[idx]) ** 2 * crng.uniform(0.6, 1.0, len(idx))
        np.add.at(exc, at + idx, amp)
        hum[at : at + k] += _harmonic(f, sr, (1.0, 0.35, 0.2, 0.1)) * spin * stop
        start += cycle
    rattle = _fftconv(exc, 0.5 * table + np.pad(case, (0, max(0, len(table) - len(case))))[: len(table)], n)
    mono = _norm(rattle) + 0.3 * _norm(hum)
    return bandlimit(mono, sr, 80.0, 6000.0, order=2)


def _phone_ring(ev, sr, rng, crng, p, seed):
    """0/1: an electromechanical bell phone (a clapper flicking between two
    small gongs ~20 times a second, in the national cadence); 2: a marimba
    ringtone through a phone's small loudspeaker; 3: vibrating on a table."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 4
    mono = np.zeros(n)
    if kind in (0, 1):
        # Two bells side by side, hit alternately by one clapper.
        f1, f2 = crng.uniform(1050, 1250), crng.uniform(1350, 1550)
        if kind == 0:
            cadence = [(0.0, 2.0)]
            cycle = 6.0
        else:
            cadence = [(0.0, 0.4), (0.6, 0.4)]
            cycle = 3.0
        start = 0.01
        while start < dur - 0.2:
            for off, L in cadence:
                at = int((start + off) * sr)
                L = min(L, dur - start - off - 0.05)
                if L <= 0.05:
                    continue
                mono += _bell_ring(n, sr, at, L, f1, rng, crng, rate=20.0)
                mono += _bell_ring(n, sr, at + int(0.025 * sr), L, f2, rng, crng, rate=20.0)
            start += cycle
    elif kind == 2:
        # A modern ringtone: a soft marimba-like arpeggio, repeated, played
        # by a phone's tiny loudspeaker (no bass, a little strained).
        root = crng.uniform(520, 640)
        steps = [0, 4, 7, 12, 7, 4, 0, 12] if crng.random() < 0.5 else [0, 7, 12, 16, 12, 7, 12, 19]
        note = 0.13
        bars = {}
        w = int(0.0015 * sr)
        mallet = np.sin(np.pi * (np.arange(w) + 0.5) / w)  # a soft mallet's contact
        start = 0.01
        while start < dur - 0.2:
            for j, s in enumerate(steps):
                at = int((start + j * note) * sr)
                if at >= n - int(0.05 * sr):
                    break
                f = root * 2 ** (s / 12)
                if f not in bars:
                    bars[f] = np.convolve(_bells_modes((f, 4.0 * f, 9.2 * f), (0.35, 0.08, 0.03), (1.0, 0.22, 0.05), sr), mallet)
                _bells_place(mono, at, bars[f])
            start += len(steps) * note + 0.6
        mono = bandlimit(mono, sr, 550.0, 7000.0, order=2)
        mono = saturate(1.4 * mono / max(np.abs(mono).max(), 1e-12), 1.8)
    else:
        mono = _phone_vibrate(n, sr, rng, crng)
    return mono, _decorrelate(mono, sr, crng)


# -- desk sounds: small, fast, close-miked mechanical transients --------------
#
# A clock escapement, a key switch, a mouse micro-switch, a shutter: each is a
# small stiff part hitting another in well under a millisecond. What reaches
# the mic is a broadband crack that dies in a few milliseconds (many dense,
# heavily damped modes, not a few clean ones: clean modes ringing for tens of
# ms are what makes a click read as a wood block), then a weaker, slightly
# longer knock of whatever housing it sits in. These helpers build one hit on
# a short local buffer and drop it into the output, so a few hundred hits
# stay cheap (no full-length FFT per hit).


def _desk_ir(sr, mrng, k, band, t60, count):
    """`count` random damped modes, log-spread over `band` (Hz), t60 ~ `t60` s,
    peak 1."""
    t = np.arange(k) / sr
    fs = np.exp(mrng.uniform(np.log(band[0]), np.log(band[1]), count))
    ds = t60 * mrng.uniform(0.5, 1.5, count)
    gs = mrng.uniform(0.4, 1.0, count)
    ph = mrng.uniform(0.0, TAU, count)
    return _norm((gs[:, None] * np.exp(-LN1000 * t[None, :] / ds[:, None])
                  * np.sin(TAU * fs[:, None] * t[None, :] + ph[:, None])).sum(0))


@functools.lru_cache(maxsize=128)
def _desk_ir_fixed(sr, mseed, k, band, t60, count):
    """`_desk_ir` for a part that is struck again and again (a pallet, a key,
    a switch): the same modes every time, computed once."""
    ir = _desk_ir(sr, np.random.default_rng(mseed), k, band, t60, count)
    ir.flags.writeable = False
    return ir


def _desk_noise(rng, n, sr, lo, hi):
    """Band-limited noise, peak 1, on a power-of-two FFT (noise has no edges
    to wrap, and an awkward length makes `bandlimit` slow)."""
    m = 1 << max(4, (n - 1).bit_length())
    f = np.fft.rfftfreq(m, 1.0 / sr)
    mag = 1.0 / np.sqrt(1.0 + (f / hi) ** 4) / np.sqrt(1.0 + (np.maximum(f, 1e-9) / lo) ** -4)
    return _norm(np.fft.irfft(np.fft.rfft(rng.standard_normal(m)) * mag, m)[:n])


def _desk_hit(sr, rng, length, band, t60, count=14, contact=0.00008, noise=1.0, noise_tau=0.0006,
              mseed=None, body=None):
    """One small impact, `length` s long, peak 1: a contact pulse plus a
    noise crack into `count` damped modes over `band`. `mseed` fixes the
    modes (the same part struck again sounds the same; only the crack
    varies); `body` = (lo, hi, t60, count, gain) adds the housing it is
    mounted in."""
    k = max(32, int(length * sr))
    t = np.arange(k) / sr
    exc = np.zeros(k)
    c = max(2, int(contact * sr))
    exc[:c] = np.sin(np.pi * (np.arange(c) + 0.5) / c)
    if noise > 0:
        exc += noise * rng.standard_normal(k) * np.exp(-t / noise_tau)
    band = (float(band[0]), float(band[1]))
    if mseed is None:
        ir = _desk_ir(sr, rng, k, band, t60, count)
    else:
        ir = _desk_ir_fixed(sr, int(mseed), k, band, float(t60), count)
    if body is not None:
        blo, bhi, bt60, bcount, bgain = body
        ir = ir + bgain * _desk_ir_fixed(sr, int(mseed or 0) + 1, k, (float(blo), float(bhi)), float(bt60), bcount)
    m = 1 << (2 * k - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(exc, m) * np.fft.rfft(ir, m), m)[:k]
    f = max(2, k // 6)  # the local buffer ends on zero
    y[-f:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, f))
    return _norm(y)


def _desk_add(out, at, g, a=1.0):
    """Mix the local buffer `g` into `out` from sample `at` (clipped at the end)."""
    if 0 <= at < len(out):
        k = min(len(g), len(out) - at)
        out[at : at + k] += a * g[:k]


def _desk_swish(rng, n, sr, env, lo, hi):
    """Friction noise (a thumb on a lever, paper sliding) following `env`."""
    return _desk_noise(rng, n, sr, lo, hi) * env


def _keyboard_typing(ev, sr, rng, crng, p, seed):
    """Typing: each keystroke is a keycap snapping down (a bright plastic
    click, a few ms), bottoming out on the plate a moment later (a duller,
    lower knock of the keyboard deck) and, when the finger lifts 60-120 ms
    later, a softer return click. Keys across the board differ a little. The
    rhythm is human: quick runs inside a word, the space bar louder and lower
    with its stabiliser rattle, and pauses. A typewriter is slower and much
    louder: a type bar slaps the platen through the paper, the carriage
    escapement steps after it, and the margin bell rings at the end of a
    line."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    left = np.zeros(n)
    right = np.zeros(n)
    keys = [int(crng.integers(1 << 31)) for _ in range(8)]  # key positions on the deck
    deck = int(crng.integers(1 << 31))
    # click band, click t60, click buffer, deck (lo, hi, t60, modes), deck level, release level, median gap
    cl_band, cl_t60, cl_len, deck_b, deck_lvl, rel_lvl, med = (
        ((1800.0, 8000.0), 0.0025, 0.012, (300.0, 2200.0, 0.012, 10), 0.5, 0.35, 0.13),  # laptop scissor keys
        ((2000.0, 9000.0), 0.003, 0.015, (180.0, 1500.0, 0.02, 12), 0.85, 0.55, 0.12),  # mechanical
        ((900.0, 7000.0), 0.006, 0.03, (120.0, 1100.0, 0.05, 14), 1.0, 0.3, 0.2),  # typewriter
    )[kind]
    line = int(crng.integers(14, 30))
    tm = crng.uniform(0.01, 0.06)
    word = int(crng.integers(2, 9))
    i = 0
    chars = 0
    while tm < dur - 0.12:
        space = i == word
        at = int(tm * sr)
        a = crng.uniform(0.55, 1.0) * (1.25 if space else 1.0)
        w = 0.5 + (0.0 if space else crng.uniform(-0.3, 0.3))
        stroke = np.zeros(int(0.2 * sr))
        esc = _desk_hit(sr, rng, 0.02, (1500.0, 8000.0), 0.004, 12, noise=1.2, noise_tau=0.0004,
                        mseed=deck + 1) if kind == 2 else None
        if kind == 2 and space:
            _desk_add(stroke, 0, esc, 0.45)  # no type bar: only the carriage steps
        else:
            mr = keys[int(crng.integers(0, 8))] + (77 if space else 0)
            band = (cl_band[0] * (0.6 if space else 1.0), cl_band[1] * (0.85 if space else 1.0))
            click = _desk_hit(sr, rng, cl_len, band, cl_t60, 16, noise=1.5, noise_tau=0.0005,
                              mseed=mr)
            # the cap scrapes its guide and the switch leaf chatters: a few
            # weaker contacts smeared over the next 4-15 ms
            for _ in range(int(crng.integers(1, 4))):
                _desk_add(stroke, int(crng.uniform(0.001, 0.008) * sr) if kind < 2 else int(crng.uniform(0.001, 0.004) * sr),
                          _desk_hit(sr, rng, cl_len, band, cl_t60, 12, noise=1.5, noise_tau=0.0008, mseed=mr + 3),
                          crng.uniform(0.25, 0.6))
            dk = _desk_hit(sr, rng, deck_b[2] * 3.0, (deck_b[0] * (0.7 if space else 1.0), deck_b[1]), deck_b[2],
                           deck_b[3], contact=0.0006, noise=0.6, noise_tau=0.0015,
                           mseed=deck + (5 if space else 0))
            bot = int(crng.uniform(0.002, 0.007) * sr) if kind < 2 else int(crng.uniform(0.0, 0.002) * sr)
            _desk_add(stroke, 0, click, 0.8 if kind == 0 else 1.0)
            _desk_add(stroke, bot, dk, deck_lvl * (1.3 if space else 1.0))
            if space and kind < 2:  # the stabiliser wire rattles a few ms later
                _desk_add(stroke, bot + int(crng.uniform(0.004, 0.009) * sr),
                          _desk_hit(sr, rng, 0.01, (2500.0, 8000.0), 0.002, 10, noise=1.0, noise_tau=0.0003), 0.35)
            if kind == 2:  # the carriage escapement steps after the slug lands
                _desk_add(stroke, bot + int(crng.uniform(0.018, 0.03) * sr), esc, 0.4)
            rel = int(crng.uniform(0.06, 0.12) * sr)
            _desk_add(stroke, rel, _desk_hit(sr, rng, cl_len, (band[0] * 1.2, band[1]), cl_t60 * 0.8, 12,
                                             noise=1.0, noise_tau=0.0004, mseed=mr),
                      rel_lvl * crng.uniform(0.6, 1.0))
        _desk_add(left, at, stroke, a * (1 - w))
        _desk_add(right, at, stroke, a * w)
        chars += 1
        if kind == 2 and chars == line and tm < dur - 0.8:
            # end of line: the margin bell (a small struck bell: it does ring)
            fb = crng.uniform(2200.0, 2600.0)
            b = _norm(_strike(n, sr, at + int(0.05 * sr), (fb, fb * 2.76), (0.6, 0.12),
                              (1.0, 0.25), rng, 0.0002))
            left += 0.2 * b
            right += 0.2 * b
        if space:
            word = int(crng.integers(2, 9))
            i = 0
            tm += crng.uniform(0.12, 0.3) * med / 0.12
        else:
            i += 1
            tm += float(np.clip(crng.lognormal(np.log(med), 0.45), 0.045, 0.5))
        if crng.random() < 0.05:
            tm += crng.uniform(0.35, 1.0)  # a pause to think
    return left, right


def _mouse_click(ev, sr, rng, crng, p, seed):
    """A mouse button: the finger pushes the plastic lever onto a tiny
    snap-action micro-switch, whose spring flips with a sharp metal snap
    (contact and a bounce within a millisecond) and the shell knocks once;
    the release 50-100 ms later is the same snap, softer. The whole click is
    over in a few ms: bright, dry, tiny."""
    n = _n(ev, sr)
    out = np.zeros(n)
    sw = int(crng.integers(1 << 31))  # one switch, one shell
    shell = int(crng.integers(1 << 31))
    presses = [0.004]
    if seed % 2 == 1:
        presses.append(0.004 + crng.uniform(0.11, 0.16))
    for tm in presses:
        hold = crng.uniform(0.06, 0.1) if seed % 2 == 0 else crng.uniform(0.045, 0.07)
        for up, (rel, lvl) in enumerate(((0.0, 1.0), (hold, 0.55))):
            at = int((tm + rel) * sr)
            snap = _desk_hit(sr, rng, 0.008, (2500.0, 9500.0), 0.0018, 16, contact=0.00005, noise=1.5,
                             noise_tau=0.00025, mseed=sw + up)
            body = _desk_hit(sr, rng, 0.02, (900.0, 3500.0), 0.005, 10, contact=0.0003, noise=0.4,
                             noise_tau=0.0008, mseed=shell + up)
            a = lvl * crng.uniform(0.85, 1.0)
            _desk_add(out, at, snap, a)
            _desk_add(out, at + int(crng.uniform(0.0003, 0.0009) * sr), snap, 0.4 * a)
            _desk_add(out, at, body, (0.45 if up == 0 else 0.25) * a)
    return out, out


def _camera_shutter(ev, sr, rng, crng, p, seed):
    """A camera firing. A DSLR is two mechanical clacks: the mirror flips up
    and slaps its stop as the first curtain runs, then the second curtain
    closes and the mirror drops back (a few tens of ms later, a touch
    deeper), and the motor recocks the shutter with a short geared whirr. In
    burst the cycle repeats ~7 times a second. A film camera's focal-plane
    shutter is a lighter double click; the wind-on lever then drags a ratchet
    through its stroke and springs back."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    out = np.zeros(n)
    mech = int(crng.integers(1 << 31))
    body = (300.0, 2500.0, 0.012, 12, 0.7)

    def clack(at, a, which, bright=1.0):
        g = _desk_hit(sr, rng, 0.035, (1200.0 * bright, 9000.0), 0.004, 18, contact=0.00015, noise=1.5,
                      noise_tau=0.0006, mseed=mech + which, body=body)
        _desk_add(out, at, g, a)
        _desk_add(out, at + int(crng.uniform(0.002, 0.005) * sr), g, 0.3 * a)  # it bounces off its stop

    def whirr(at, length):
        k = int(length * sr)
        if at >= n or k < 8:
            return
        u = np.arange(k) / k
        env = np.sin(np.pi * u) ** 0.6
        # gear mesh: noise roughly modulated at 160-260 Hz (too irregular to buzz)
        mod = 0.6 + 0.4 * _smooth(rng, k, sr, crng.uniform(160.0, 260.0))
        w = _desk_noise(rng, k, sr, 500.0, 4500.0) * mod * env
        _desk_add(out, at, _norm(w), 0.18)

    if kind == 1:
        tm = 0.004
        fps = crng.uniform(6.0, 8.0)
        while tm < dur - 0.13:
            at = int(tm * sr)
            sp = crng.uniform(0.035, 0.05)
            clack(at, crng.uniform(0.9, 1.0), 0)
            clack(at + int(sp * sr), crng.uniform(0.8, 0.9), 1, 0.8)
            whirr(at + int((sp + 0.006) * sr), crng.uniform(0.06, 0.08))
            tm += 1.0 / fps * crng.uniform(0.99, 1.01)
    elif kind == 0:
        sp = crng.uniform(0.04, 0.07)
        clack(int(0.004 * sr), 1.0, 0)
        clack(int((0.004 + sp) * sr), 0.85, 1, 0.8)
        whirr(int((0.01 + sp) * sr), crng.uniform(0.07, 0.1))
    else:
        sp = crng.uniform(0.012, 0.025)
        clack(int(0.004 * sr), 0.9, 0, 1.4)
        clack(int((0.004 + sp) * sr), 0.6, 1, 1.4)
        # wind-on: ratchet teeth at an uneven pace (the thumb speeds up and
        # slows down), a friction drag, then the lever springs home
        st = 0.004 + crng.uniform(0.18, 0.26)
        span = min(crng.uniform(0.2, 0.28), max(0.05, dur - st - 0.08))
        tt = st
        while tt < st + span:
            u = (tt - st) / span
            _desk_add(out, int(tt * sr), _desk_hit(sr, rng, 0.01, (2500.0, 9000.0), 0.0018, 10, noise=1.0,
                                                    noise_tau=0.0003), 0.22 * crng.uniform(0.6, 1.0))
            tt += crng.uniform(0.007, 0.02) * (1.2 - 0.6 * np.sin(np.pi * u))
        a0, k = int(st * sr), int(span * sr)
        if a0 < n:
            env = np.zeros(n)
            m = min(k, n - a0)
            env[a0 : a0 + m] = np.sin(np.pi * np.linspace(0, 1, m))
            out += 0.07 * _desk_swish(rng, n, sr, env, 800.0, 6000.0)
        clack(int((st + span + crng.uniform(0.01, 0.03)) * sr), 0.35, 2, 1.6)
    return out, _decorrelate(out, sr, crng, taps=3, span=0.006)


def _paper_rustle(ev, sr, rng, crng, p, seed):
    """Paper crackles: a stiff sheet in the hands buckles in many small snaps
    (each a crack a fraction of a ms long, of very uneven strength), in
    gestures: a burst while the hands move, then a pause. Under the crackle
    the surfaces slide (a soft swish) and a big flex makes the whole sheet
    go "thwap" (a low, damped flutter). Handling a sheet is sparse, with
    swishes and the odd flex; crumpling is dense, louder snaps that thin out
    as the ball gets tight."""
    n = _n(ev, sr)
    dur = n / sr
    crumple = seed % 2 == 1
    t = np.arange(n) / sr
    gest = []  # (start, length, intensity)
    tm = crng.uniform(0.0, 0.08)
    while tm < dur - 0.1:
        ln = crng.uniform(0.25, 0.7) if crumple else crng.uniform(0.2, 0.6)
        gest.append((tm, min(ln, dur - tm - 0.02), crng.uniform(0.5, 1.0)))
        tm += ln + (crng.uniform(0.02, 0.15) if crumple else crng.uniform(0.1, 0.5))
    act = np.zeros(n)
    for g0, gl, gi in gest:
        u = np.clip((t - g0) / max(gl, 1e-3), 0.0, 1.0)
        act = np.maximum(act, gi * np.where((t >= g0) & (t <= g0 + gl), np.sin(np.pi * u) ** 0.7, 0.0))
    if crumple:
        act *= 1.0 - 0.45 * t / dur  # the ball tightens

    def chan(r):
        out = np.zeros(n)
        cnt = int((650.0 if crumple else 140.0) * dur)
        tt = r.uniform(0.0, dur, cnt)
        tt = tt[r.random(cnt) < np.interp(tt, t, act) ** 1.5]
        amps = np.minimum(r.pareto(1.8, len(tt)) + 0.15, 25.0)
        # snap shapes by size: small ones bright and tiny, big ones lower and longer
        shapes = [_desk_hit(sr, r, 0.006 + 0.002 * k, (r.uniform(900.0, 3000.0), 9500.0), 0.0008 + 0.0004 * k, 10,
                            contact=0.00004, noise=2.0, noise_tau=0.00015 + 0.0001 * k) for k in range(6)]
        rank = np.empty(len(tt), int)
        rank[np.argsort(amps)] = np.arange(len(tt))
        sign = r.choice((-1.0, 1.0), len(tt))
        for j in range(len(tt)):
            _desk_add(out, int(tt[j] * sr), shapes[min(5, 6 * rank[j] // max(1, len(tt)))], amps[j] * sign[j])
        swish = _desk_swish(r, n, sr, act ** 2, 400.0, 5000.0)
        flex = np.zeros(n)
        for g0, gl, gi in gest:
            if r.random() < (0.5 if crumple else 0.8):
                fl = _desk_hit(sr, r, 0.12, (90.0, 450.0), 0.05, 8, contact=0.004, noise=0.5, noise_tau=0.01)
                _desk_add(flex, int((g0 + r.uniform(0.0, 0.6) * gl) * sr), fl, gi)
        return _norm(out) + (0.12 if crumple else 0.22) * swish + (0.15 if crumple else 0.3) * flex

    a = _attack(n, sr, 0.0, 0.01)
    return chan(rng) * a, chan(np.random.default_rng(int(rng.integers(1 << 31)))) * a


def _cork_pop(ev, sr, rng, crng, p, seed):
    """A cork leaving a bottle. While the cork slides out it squeaks against
    the glass (stick-slip, a few short rubbery chirps). When it clears the
    lip, the pressure difference equalises in one pulse (air rushing in
    behind a wine cork, gas rushing out of champagne) that sets the air in
    the neck and headspace ringing as a Helmholtz resonator. The opening
    grows as the cork leaves, so the pitch sweeps up over a few ms, and the
    open neck radiates so well that the ring dies in a couple of cycles:
    a hollow 'pop', not a struck tone. Champagne adds a sharper report, the
    hiss of the gas gushing out of the neck, and the fizz of bubbles
    bursting at the surface.
    """
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    champ = seed % 2 == 1
    t0 = min(0.3 * dur, crng.uniform(0.12, 0.2) if not champ else crng.uniform(0.03, 0.06))
    at = int(t0 * sr)
    tt = np.maximum(t - t0, 0.0)
    mono = np.zeros(n)

    if not champ:
        # the cork twisting out: two or three short rubbery squeaks
        for _ in range(int(crng.integers(2, 4))):
            s0 = crng.uniform(0.0, max(0.01, t0 - 0.06))
            L = crng.uniform(0.02, 0.05)
            k = int(L * sr)
            a0 = int(s0 * sr)
            if a0 + k >= n or k < 8:
                continue
            u = np.arange(k) / k
            rate = crng.uniform(250, 450) * (1.0 + 0.3 * u)
            ph = np.cumsum(rate) / sr
            pul = np.zeros(k)
            pul[np.nonzero(np.diff(np.floor(ph), prepend=0.0) > 0)[0]] = 1.0
            sq = np.zeros(n)
            sq[a0 : a0 + k] = pul * np.sin(np.pi * u) * crng.uniform(0.5, 1.0)
            mono += 0.12 * _norm(_fftconv(sq, _modes_ir((1400.0, 2300.0, 3300.0), (0.006, 0.004, 0.003), (1.0, 0.6, 0.3), sr, n), n))

    # the pulse: a smooth 0.5-1 ms push (a suck for wine), then the neck's
    # air ringing with Q ~3 while the pitch rises as the opening clears
    w = (0.0009, 0.0005)[champ]
    push = _pulse(n, at, max(2, int(w * sr))) * (-1.0 if not champ else 1.0)
    f0 = (crng.uniform(330, 480), crng.uniform(420, 600))[champ]
    f = f0 * (0.6 + 0.4 * (1.0 - np.exp(-tt / 0.004)))
    d = np.pi * f0 / 3.0
    ring = np.sin(TAU * np.cumsum(f) / sr) * np.exp(-d * tt) * (t >= t0) * _attack(n, sr, t0, 0.0004)
    pop = _norm(ring) + 0.5 * _norm(bandlimit(push, sr, 80.0, None, order=2))
    # the rim and the cork's edge: a short dull tick of air turbulence
    tick = bandlimit(_burst(rng, n, at, int(0.02 * sr), (0.002, 0.003)[champ], sr, rise=0.0004), sr, 300.0, 4000.0, order=2)
    mono += pop + (0.35, 0.7)[champ] * _norm(tick)

    if champ:
        # gas gushing out of the neck: a hiss that swells in 20 ms and fades
        # over half a second, darker as it slows
        g_env = (1.0 - np.exp(-tt / 0.02)) * np.exp(-tt / crng.uniform(0.3, 0.5)) * (t >= t0)
        gush = bandlimit(rng.standard_normal(n), sr, 1500.0, 7000.0, order=2) * g_env
        # the fizz: thousands of tiny bubbles bursting at the surface, random
        # clicks (each a small, very short burst) thinning out over seconds
        rate = 2500.0 * np.exp(-tt / 0.9) * (t >= t0 + 0.02)
        cnt = int(rate.sum() / sr) + 1
        times = rng.choice(n, size=cnt, p=rate / rate.sum()) / sr
        fz = _grains(rng, n, times, crng.uniform(0.1, 1.0, cnt) ** 2, sr)
        fizz = bandlimit(fz, sr, 2500.0, 8000.0, order=2)
        mono = 1.4 * mono + 0.22 * _norm(gush) + 0.18 * _norm(fizz)
    return mono, _decorrelate(mono, sr, crng)


def _coin_ir(sr, crng, size, t60):
    """A coin: a thin free metal disc. Its first modes, (2,0), (0,1) and
    (3,0), sit at 1 : 1.73 : 2.33, high up (real coins ring ~5-12 kHz; the
    ones under our 9 kHz band carry it). Every mode is a doublet, split by
    the relief stamped on the faces: the shimmer of a ringing coin."""
    f1 = crng.uniform(5000, 6200) / size
    freqs = f1 * np.array((1.0, 1.73, 2.33)) * crng.uniform(0.99, 1.01, 3)
    fa, fb, ta, tb = _bells_doublets(freqs, t60 * np.array((1.0, 0.7, 0.5)), crng, 0.004)
    g = np.array((1.0, 0.55, 0.45))
    return _bells_modes(np.r_[fa, fb], np.r_[ta, tb], np.r_[g, g * crng.uniform(0.5, 1.0, 3)], sr)


def _coin_drop(ev, sr, rng, crng, p, seed):
    """Coins on a hard surface. A dropped coin bounces on its rim: each
    landing is a pair of contacts (rim, then face) and a high 'ching'; the
    coin rings freely between bounces; the bounces come quicker (restitution
    ~0.55), then it rattles on the spot, faster and faster as it falls flat,
    and stops dead. A spun coin is Euler's disk: the contact runs round the
    rim at a rate that climbs steeply towards the end, a rising whirr. A
    handful in a dish is a clatter of coins on china and on each other."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    t = np.arange(n) / sr
    floor = _bells_modes(crng.uniform(300, 500) * np.array((1.0, 2.2, 3.7)), (0.05, 0.035, 0.025), (1, 0.6, 0.4), sr)
    clicks_t, clicks_a = [], []
    mono = np.zeros(n)

    def settle(t_start, a0, length, r0, rmax, exc):
        """Rattle to rest: contacts at a rate climbing like Euler's disk."""
        i0 = int(t_start * sr)
        i1 = min(n - int(0.03 * sr), int((t_start + length) * sr))
        if i1 - i0 < 10:
            return t_start
        tt = (np.arange(i1 - i0)) / sr
        rate = np.minimum(rmax, r0 * (length / np.maximum(length - tt, 1e-3)) ** 0.6)
        ph = np.cumsum(rate) / sr
        idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0] + 1
        amp = a0 * (0.5 + 0.5 * (tt[idx] / length) ** 2) * crng.uniform(0.5, 1.0, len(idx))
        np.add.at(exc, i0 + idx, amp)
        clicks_t.extend(t_start + tt[idx])
        clicks_a.extend(0.6 * amp)
        return t_start + tt[-1]

    if kind in (0, 1):
        size = crng.uniform(0.9, 1.25)
        free = _coin_ir(sr, crng, size, crng.uniform(0.5, 0.9))
        damped = _coin_ir(sr, crng, size, crng.uniform(0.04, 0.07))
        exc_free = np.zeros(n)
        exc_damp = np.zeros(n)
        if kind == 0:
            tm, a = 0.005, 1.0
            gap = crng.uniform(0.16, 0.26)
            ratio = crng.uniform(0.5, 0.62)
            for _b in range(int(crng.integers(4, 7))):
                if tm > dur - 0.3:
                    break
                for dt, f in ((0.0, 1.0), (crng.uniform(0.003, 0.012), crng.uniform(0.3, 0.6))):
                    at = int((tm + dt) * sr)
                    exc_free[at] += a * f
                    clicks_t.append(tm + dt)
                    clicks_a.append(a * f)
                tm += gap * crng.uniform(0.9, 1.1)
                gap *= ratio
                a *= crng.uniform(0.55, 0.75)
                if gap < 0.025:
                    break
            t_set = tm
            end = settle(t_set, 0.5 * a + 0.25, min(crng.uniform(0.35, 0.7), dur - t_set - 0.08), crng.uniform(18, 30), 140.0, exc_damp)
        else:
            exc_free[int(0.005 * sr)] = 0.8
            clicks_t.append(0.005)
            clicks_a.append(0.8)
            t_set = 0.02
            T = crng.uniform(0.7, 0.85) * dur
            end = settle(0.05, 0.3, T - 0.05, crng.uniform(16, 24), 240.0, exc_damp)
            # the spinning coin keeps some of its ring: the rim drives it
            exc_free[: int(end * sr)] += 0.25 * exc_damp[: int(end * sr)]
        ring = _fftconv(exc_free, free, n)
        # once it rattles or lies flat on the floor, the contact damps the ring
        ring *= np.where(t < t_set, 1.0, np.exp(-np.maximum(t - t_set, 0.0) / 0.12))
        ring *= np.where(t < end, 1.0, np.exp(-np.maximum(t - end, 0.0) / 0.015))
        # the final clack as it drops flat
        at = int(end * sr) + 1
        if at < n:
            exc_damp[at] += 0.5
            clicks_t.append(end)
            clicks_a.append(0.6)
        mono = ring + _fftconv(exc_damp, damped, n)
        body = _fftconv(exc_free + exc_damp, floor, n)
        mono = mono + 0.25 * body * np.abs(mono).max() / max(np.abs(body).max(), 1e-12)
    else:
        dish = _bells_modes(crng.uniform(900, 1300) * np.array((1.0, 2.2, 3.6, 5.1)), (0.35, 0.25, 0.15, 0.1), (1, 0.6, 0.4, 0.25), sr)
        k = int(crng.integers(6, 11))
        coins = [_coin_ir(sr, crng, crng.uniform(0.85, 1.3), crng.uniform(0.15, 0.35)) for _ in range(k)]
        exc_dish = np.zeros(n)
        for c in range(k):
            tm = 0.005 + crng.gamma(1.5, 0.035)
            a = crng.uniform(0.5, 1.0)
            gap = crng.uniform(0.03, 0.08)
            for _b in range(int(crng.integers(2, 5))):
                if tm > dur - 0.05:
                    break
                at = int(tm * sr)
                _bells_place(mono, at, coins[c], a)
                # it hits another coin as often as the dish
                if crng.random() < 0.5:
                    _bells_place(mono, at, coins[int(crng.integers(0, k))], 0.5 * a)
                else:
                    exc_dish[at] += a
                clicks_t.append(tm)
                clicks_a.append(a)
                tm += gap * crng.uniform(0.8, 1.2)
                gap *= 0.55
                a *= crng.uniform(0.35, 0.6)
        for _ in range(int(crng.integers(3, 7))):  # settling: a few late clinks
            tm = crng.uniform(0.3, min(1.2, dur - 0.1))
            a = crng.uniform(0.05, 0.15)
            _bells_place(mono, int(tm * sr), coins[int(crng.integers(0, k))], a)
            clicks_t.append(tm)
            clicks_a.append(a)
        d = _fftconv(exc_dish, dish, n)
        mono = mono + 0.35 * d * np.abs(mono).max() / max(np.abs(d).max(), 1e-12)
    clicks = _bells_clicks(n, sr, clicks_t, clicks_a, 2000.0, rng)
    mono = mono / max(np.abs(mono).max(), 1e-12) + 0.35 * _norm(clicks)
    return mono, _decorrelate(mono, sr, crng)


def _metal_clang(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    kind = seed % 3
    at = int(0.003 * sr)
    if kind == 0:  # a pipe: free-free bar modes (1, 2.76, 5.40, 8.93)
        f = crng.uniform(380, 620)
        freqs = f * np.array((1.0, 2.756, 5.404, 8.933, 13.34))
        t60 = np.array((2.2, 1.6, 1.0, 0.6, 0.35))
        g = (1.0, 0.8, 0.55, 0.35, 0.2)
    elif kind == 1:  # sheet / bin lid: dense, inharmonic, quicker
        f = crng.uniform(180, 300)
        freqs = f * np.sort(crng.uniform(1.0, 18.0, 14))
        t60 = crng.uniform(0.3, 0.9, 14)
        g = crng.uniform(0.3, 1.0, 14)
    else:  # anvil: a bright ping, long
        f = crng.uniform(900, 1200)
        freqs = f * np.array((1.0, 1.58, 2.33, 3.1, 4.4))
        t60 = np.array((2.8, 1.8, 1.1, 0.7, 0.4))
        g = (1.0, 0.6, 0.45, 0.3, 0.2)
    ring = _strike(n, sr, at, freqs * crng.uniform(0.99, 1.01, len(freqs)), t60, g, rng, 0.0004, noise=0.4)
    hit = bandlimit(_burst(rng, n, at, int(0.02 * sr), 0.003, sr, rise=0.0005), sr, 400.0, 8000.0, order=4)
    mono = _norm(ring) + 0.35 * _norm(hit)
    beat = 1.0 + 0.15 * np.sin(TAU * crng.uniform(1.5, 4.0) * np.arange(n) / sr)
    return mono * beat, _decorrelate(mono, sr, crng) * beat


def _wood_creak(ev, sr, rng, crng, p, seed):
    """Stick-slip friction: a train of tiny slips whose rate wanders (the
    pitch of the creak), each one ringing the wood's modes."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    rate0, spread = ((90, 140), (45, 70), (25, 60))[kind]
    t = np.arange(n) / sr
    act = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** (0.6 if kind != 1 else 1.2)
    if kind == 1:
        act = act * (0.3 + 0.7 * _smooth(crng, n, sr, 3.0))
    rate = rate0 + spread * _smooth(crng, n, sr, 1.5 if kind != 1 else 4.0)
    ph = np.cumsum(rate) / sr
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    amps = act[idx] * crng.uniform(0.6, 1.0, len(idx))
    slips = np.zeros(n)
    slips[idx] = amps
    f = crng.uniform(380, 520) * (0.7 if kind == 2 else 1.0)
    freqs = f * np.array((1.0, 1.9, 2.7, 3.8, 5.2))
    body = _fftconv(slips, _modes_ir(freqs, (0.04, 0.03, 0.025, 0.018, 0.012), (1, 0.7, 0.5, 0.35, 0.2), sr, n), n)
    grit = bandlimit(slips * rng.standard_normal(n), sr, 1500.0, 6000.0, order=2)
    mono = _norm(body) + 0.15 * _norm(grit)
    return mono, _decorrelate(mono, sr, crng)


def _heartbeat(ev, sr, rng, crng, p, seed):
    """Heart sounds as heard on the chest. S1 ('lub') is the mitral then the
    tricuspid valve closing 20-35 ms apart: a low, heavily damped thump
    (25-120 Hz, ~0.1 s). S2 ('dub'), the aortic then the pulmonary valve, is
    shorter, a little higher and crisper, split wider on inspiration. S1 to
    S2 is the systole (~0.35 s at rest, shorter as the heart races); the beat
    interval breathes with respiration (sinus arrhythmia). The chest wall and
    the stethoscope pass only the low end."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    bpm = (62.0, 88.0, 120.0)[kind] * crng.uniform(0.97, 1.03)
    period = 60.0 / bpm
    systole = max(0.2, 0.5 - 0.0024 * bpm)
    resp = crng.uniform(3.5, 5.0)
    rph = crng.uniform(0.0, TAU)
    rsa = (0.04, 0.025, 0.012)[kind]
    f_s1, f_s2 = crng.uniform(34, 44), crng.uniform(50, 62)
    mono = np.zeros(n)
    tex = np.zeros(n)

    def thump(at, f, tau, amp):
        k = min(n - at, int(7 * tau * sr))
        if k < 10:
            return
        tt = np.arange(k) / sr
        env = (1.0 - np.exp(-tt / 0.004)) * np.exp(-tt / tau)
        ph = TAU * np.cumsum(f * (1.0 + 0.15 * np.exp(-tt / 0.02))) / sr
        y = np.sin(ph) * env + 0.45 * np.sin(2.3 * ph + 0.7) * env * np.exp(-tt / (0.5 * tau))
        mono[at : at + k] += amp * y
        tex[at : at + k] += amp * env

    tm = 0.03
    while tm < dur - 0.2:
        insp = 0.5 + 0.5 * np.sin(TAU * tm / resp + rph)
        a = crng.uniform(0.88, 1.0) * (1.0 - 0.08 * insp)
        thump(int(tm * sr), f_s1 * crng.uniform(0.95, 1.05), crng.uniform(0.022, 0.03), a)
        thump(int((tm + crng.uniform(0.02, 0.035)) * sr), f_s1 * crng.uniform(1.1, 1.25), 0.02, 0.55 * a)
        t2 = tm + systole * crng.uniform(0.98, 1.02)
        if t2 < dur - 0.1:
            thump(int(t2 * sr), f_s2 * crng.uniform(0.95, 1.05), crng.uniform(0.014, 0.018), 0.75 * a)
            split = crng.uniform(0.012, 0.022) + 0.025 * insp
            thump(int((t2 + split) * sr), f_s2 * crng.uniform(1.1, 1.3), 0.012, 0.4 * a)
        tm += period * (1.0 + rsa * np.sin(TAU * tm / resp + rph)) * crng.uniform(0.99, 1.01)
    # a little of the valves' broadband texture, low-passed by the chest
    noise = bandlimit(rng.standard_normal(n), sr, 30.0, 250.0, order=2)
    mono = _norm(mono) + 0.25 * _norm(noise * tex)
    mono = bandlimit(mono, sr, 22.0, 400.0, order=2)
    return mono, mono


def _applause(ev, sr, rng, crng, p, seed):
    """Applause is many independent hand claps. Each clap is a ~1 ms puff of
    air squeezed from between the palms that rings the cavity between the
    hands (0.7-2.8 kHz: low for cupped hands, high for flat palms) for a few
    ms. Each person keeps their own hand shape and claps steadily at 2.5-5.5
    Hz (a few % jitter); the crowd starts within a fraction of a second and
    trails off person by person; now and then a hand lands differently. A few people stay distinct; a big audience
    becomes a dense, flat roar of clicks in a hall."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    people = (int(crng.integers(3, 7)), int(crng.integers(20, 35)), int(crng.integers(150, 230)))[kind]
    n_cfg = min(people, 24)
    L = int(0.02 * sr)
    tt = np.arange(L) / sr
    bank = np.zeros((n_cfg, L))
    r = max(2, int(0.0003 * sr))
    for c in range(n_cfg):
        fr = np.exp(crng.uniform(np.log(700.0), np.log(2800.0)))
        exc = crng.standard_normal(L) * np.exp(-tt / crng.uniform(0.0007, 0.0016))
        exc[:r] *= np.linspace(0.0, 1.0, r)
        res = np.exp(-tt / crng.uniform(0.00025, 0.0007)) * np.sin(TAU * fr * tt)  # a heavily damped cavity (Q ~ 1-6)
        ring = np.convolve(exc, res)[:L]
        clap = 0.75 * _norm(ring) + 0.35 * _norm(_air_band(exc, sr, (250.0, None, 2, 1.0)))
        clap *= np.minimum(1.0, (L - np.arange(L)) / (0.3 * L))  # end on zero
        bank[c] = clap / max(np.sqrt(np.mean(clap**2)), 1e-12)
    cfg = np.arange(people) % n_cfg  # each person's usual hand shape
    rate = crng.uniform(2.5, 4.5, people) if kind == 0 else crng.uniform(3.0, 5.5, people)
    onset = crng.gamma(2.0, 0.06 if kind else 0.1, people)
    stop = np.minimum(dur * crng.uniform(0.72, 0.97, people), dur - L / sr - 0.01)
    level = crng.lognormal(0.0, 0.35, people)
    w = crng.uniform(0.1, 0.9, people) if kind else crng.uniform(0.25, 0.75, people)
    m = int(dur * 6.0) + 4
    iv = (1.0 / rate)[:, None] * (1.0 + 0.04 * crng.standard_normal((people, m)))
    times = onset[:, None] + np.cumsum(iv, axis=1) - iv[:, :1]
    ok = times < stop[:, None]
    amp = level[:, None] * np.exp(0.15 * crng.standard_normal((people, m)))
    amp = amp * np.minimum(1.0, 0.4 + 0.6 * (times - onset[:, None]) / 0.4)  # the first claps are tentative
    who = np.broadcast_to(np.arange(people)[:, None], times.shape)[ok]
    shape = (cfg[who] + (crng.random(len(who)) < 0.3) * crng.integers(1, n_cfg, len(who))) % n_cfg  # and its slips
    at = (times[ok] * sr).astype(int)
    a = amp[ok]
    out = []
    for g in (np.sqrt(1.0 - w[who]), np.sqrt(w[who])):
        y = np.zeros(n + L)
        for s in range(0, len(at), 800):  # place the claps in chunks (bounded memory)
            idx = (at[s : s + 800, None] + np.arange(L)).ravel()
            val = ((a[s : s + 800] * g[s : s + 800])[:, None] * bank[shape[s : s + 800]]).ravel()
            y += np.bincount(idx, weights=val, minlength=n + L)[: n + L]
        out.append(y[:n])
    p["reverb"], p["reverb_decay"] = ((0.16, 0.6), (0.22, 0.9), (0.3, 1.5))[kind]
    fade = _fade_out(n, sr, min(0.3, 0.1 * dur))
    return out[0] * fade, out[1] * fade


def _pouring_water(ev, sr, rng, crng, p, seed):
    """A stream of water poured into a vessel. The falling jet entrains air
    where it meets the surface: a dense cloud of small bubbles ringing at
    their Minnaert pitches (a few hundred Hz to a few kHz) and a crackle of
    droplet impacts, both following the flow. Into a glass, the air column
    above the water (a pipe closed by the water) colours it all, and its
    resonance climbs as the glass fills; a jug into a pot is a thicker,
    lower, glugging stream. At the end the stream thins and the last drops
    fall."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    glass = seed % 2 == 0
    t_on = 0.02
    t_off = max(0.4, dur - crng.uniform(0.25, 0.45))
    flow = _attack(n, sr, t_on, 0.06) * (1.0 - _attack(n, sr, t_off, 0.18))
    wob = 0.8 + 0.2 * _smooth(crng, n, sr, 4.0)
    if not glass:
        # glugging: air bubbles back into the jug through the spout choke the
        # stream a few times a second, at irregular moments
        gt = np.cumsum(crng.uniform(0.14, 0.3, int(dur / 0.14) + 2))
        g = np.zeros(n)
        for tg in gt:
            g += np.exp(-0.5 * ((t - tg) / 0.035) ** 2)
        wob = wob * (1.0 - 0.35 * np.clip(g, 0.0, 1.0))
    flow = flow * wob
    # into an empty glass the stream first hits the bottom: no bubbles yet
    wet = _attack(n, sr, t_on, 0.35) if glass else _attack(n, sr, t_on, 0.1)
    rate_b = (1100.0 if glass else 900.0) * flow * wet
    tt = _water_times(crng, rate_b, sr)
    r = _water_radii(crng, len(tt), *((0.6, 3.5) if glass else (1.0, 7.0)), 2.2)
    a = r**0.6 * np.interp(tt, t, flow) * crng.uniform(0.2, 1.0, len(tt))
    bub = _water_bubbles(n, sr, tt, 3260.0 / r, a, damp=1.3, rise=0.15, dec=2)
    crack = _water_crackle(rng, crng, n, sr, 3500.0 * flow + 1.0, 400.0 if glass else 250.0, 10000.0, alpha=1.7)
    turb = _water_noise(rng, n, sr, 200.0 if glass else 120.0, 9000.0 if glass else 6000.0) * flow
    mono = _norm(bub) + 0.9 * _norm(crack) + 0.35 * turb
    # the last drops after the stream stops
    k = int(crng.integers(2, 5))
    dt = t_off + 0.12 + np.cumsum(crng.uniform(0.05, 0.12, k))
    dr = crng.uniform(1.0, 2.5, k)
    mono = mono + 0.5 * _water_bubbles(n, sr, dt, 3260.0 / dr, crng.uniform(0.3, 0.8, k) * np.exp(-np.arange(k) * 0.4), damp=0.9, rise=0.2)
    if glass:
        # air column: depth 12 cm -> 2 cm, end correction 0.6 x 3.5 cm radius
        fill = np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)
        L = 0.12 - 0.10 * fill + 0.021
        mono = _water_resonate(mono, sr, 343.0 / (4.0 * L), peaks=3, q=7.0, gain=4.0)
    else:
        # a wide pot: a broad, low, slowly rising cavity colour
        fill = np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)
        L = 0.16 - 0.04 * fill + 0.07
        mono = _water_resonate(mono, sr, 343.0 / (4.0 * L), peaks=3, q=3.0, gain=1.5)
    mono = mono * _fade_out(n, sr, 0.05)
    return mono, _decorrelate(mono, sr, crng)


def _can_opening(ev, sr, rng, crng, p, seed):
    """Pull tab: a fingertip lifts the tab (a small tick), the lever cracks
    the scored lid open (a sharp crack with a short metallic ring) and the
    pressurised gas bursts out through the gap (a loud, bright hiss that
    dies as the pressure drops, spitting a few droplets), the tab folds the
    flap in (a duller clunk) and drops back (a tick), then the drink fizzes:
    countless tiny bubbles bursting at the surface."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    beer = seed % 2 == 1
    t_lift = 0.02 + crng.uniform(0.0, 0.03)
    lift = _strike(n, sr, int(t_lift * sr), (3100.0, 4700.0, 6900.0), (0.006, 0.004, 0.003), (1, 0.6, 0.4), rng, 0.0002, noise=0.8)
    t_c = t_lift + crng.uniform(0.07, 0.14)
    ac = int(t_c * sr)
    lid = np.array((2600.0, 4100.0, 5600.0, 7300.0)) * crng.uniform(0.92, 1.08)
    crack = _strike(n, sr, ac, lid, (0.025, 0.018, 0.012, 0.008), (1, 0.8, 0.6, 0.4), rng, 0.00015, noise=1.5)
    snap = _water_band(_burst(rng, n, ac, int(0.006 * sr), 0.0012, sr, rise=0.0003), sr, 800.0, 10000.0, order=2)
    # the jet: brightest at the burst, darkening and dying as pressure falls
    tau = 0.13 if beer else 0.08
    aft = np.maximum(t - t_c, 0.0)
    env = _attack(n, sr, t_c, 0.0015) * (np.exp(-((aft / (2.2 * tau)) ** 2)) * (0.6 + 0.4 * np.exp(-aft / tau)) + 0.03 * np.exp(-aft / (4.0 * tau)))
    flick = 0.75 + 0.25 * _smooth(crng, n, sr, 60.0)
    hiss = _water_band(rng.standard_normal(n), sr, 700.0, 10000.0, order=1) * env * flick
    spit = _water_crackle(rng, crng, n, sr, 2500.0 * env + 1.0, 1500.0, 9000.0, alpha=1.6)
    t_f = t_c + crng.uniform(0.08, 0.16)
    fold = _strike(n, sr, int(t_f * sr), (1350.0, 2400.0, 3800.0), (0.02, 0.015, 0.01), (1, 0.6, 0.4), rng, 0.0004, noise=0.6)
    t_back = t_f + crng.uniform(0.25, 0.45)
    back = _strike(n, sr, int(t_back * sr), (2900.0, 4400.0), (0.008, 0.005), (1, 0.6), rng, 0.0002, noise=0.5)
    # effervescence: bursts at the surface, thinning out
    fz = _attack(n, sr, t_c + 0.03, 0.08) * np.exp(-np.maximum(t - t_c, 0.0) / (0.9 if beer else 0.6))
    fizz = _water_crackle(rng, crng, n, sr, 400.0 * fz + 1.0, 2500.0, 9500.0, alpha=1.5)
    ft = _water_times(crng, 350.0 * fz, sr)
    fb = _water_bubbles(n, sr, ft, crng.uniform(2500.0, 6000.0, len(ft)), crng.uniform(0.2, 1.0, len(ft)) * np.interp(ft, t, fz), damp=1.2, rise=0.1)
    mono = (0.25 * _norm(lift) + 0.4 * _norm(crack) + 0.35 * _norm(snap) + _norm(hiss)
            + 0.3 * _norm(spit) * env + 0.35 * _norm(fold) + 0.2 * _norm(back)
            + 0.05 * _norm(fizz) + 0.03 * _norm(fb))
    mono = mono * _fade_out(n, sr, min(0.2, 0.15 * dur))
    return mono, _decorrelate(mono, sr, crng, taps=3)


def _clock_alarm(ev, sr, rng, crng, p, seed):
    """0: a twin-bell wind-up alarm clock. A hammer on a spring arm
    oscillates between two small domed bells, striking each ~12-15 times a
    second (25-30 hits/s in all); the bells ring on between strikes, so it
    is a continuous, rattling, beating metallic ring that slows a touch as
    the spring runs down. 1: a digital alarm: a piezo disc driven by a
    2048 Hz square wave (its resonance near 4 kHz makes the 2nd harmonic
    the loudest), in bursts of 2-4 beeps each second."""
    n = _n(ev, sr)
    dur = n / sr
    mono = np.zeros(n)
    if seed % 2 == 0:
        f1 = crng.uniform(1700, 2400)
        f2 = f1 * crng.uniform(1.1, 1.3)
        fh = crng.uniform(12.0, 15.0)
        ratios = np.array((1.0, 1.72, 2.65, 3.75))
        excs, irs = [], []
        for f in (f1, f2):
            fr = f * ratios * crng.uniform(0.98, 1.02, 4)
            fa, fb, ta, tb = _bells_doublets(fr, np.array((0.9, 0.6, 0.4, 0.28)) * crng.uniform(0.7, 1.2), crng, 0.006)
            g = np.array((0.7, 0.6, 1.0, 0.6)) * crng.uniform(0.6, 1.0, 4)
            irs.append(_bells_modes(np.r_[fa, fb], np.r_[ta, tb], np.r_[g, 0.7 * g], sr))
            excs.append(np.zeros(n))
        tm = 0.005
        hits_t = []
        while tm < dur - 0.03:
            per = (1.0 / fh) * (1.0 + 0.05 * tm / dur) * crng.uniform(0.96, 1.04)
            for b, off in ((0, 0.0), (1, crng.uniform(0.45, 0.55))):
                th = tm + off * per
                if th < dur - 0.03:
                    excs[b][int(th * sr)] += crng.uniform(0.55, 1.0)
                    hits_t.append(th)
            tm += per
        ring = _fftconv(excs[0], irs[0], n) + crng.uniform(0.7, 1.0) * _fftconv(excs[1], irs[1], n)
        # the hammer's metal-on-metal tick
        ht = np.array(hits_t)
        tick = _bells_clicks(n, sr, ht, crng.uniform(0.5, 1.0, len(ht)), 2500.0, rng)
        # the frame and the loose hammer arm buzz along with every blow
        rattle = rng.standard_normal(n) * _fftconv(excs[0] + excs[1], np.exp(-np.arange(int(0.012 * sr)) / (0.003 * sr)), n)
        rattle = bandlimit(rattle, sr, 1500.0, None, order=2)
        mono = _norm(ring) + 0.3 * _norm(tick) + 0.12 * _norm(rattle)
    else:
        f = 2048.0 * crng.uniform(0.98, 1.02)
        beeps = int(crng.integers(2, 5))
        on = crng.uniform(0.05, 0.08)
        gap = crng.uniform(0.11, 0.14)
        start = 0.005
        while start < dur - 0.1:
            for j in range(beeps):
                at = int((start + j * gap) * sr)
                k = int(on * sr)
                if at + k >= n:
                    break
                tt = np.arange(k) / sr
                y = saturate(_harmonic(np.full(k, f), sr, (0.4, 1.0, 0.35, 0.25)), 1.5)
                y *= np.sin(np.pi * np.clip(tt / 0.004, 0, 0.5)) * np.cos(0.5 * np.pi * np.clip((tt - on + 0.005) / 0.005, 0, 1))
                mono[at : at + k] += y
            start += 1.0
    return mono, _decorrelate(mono, sr, crng, taps=3)


# =============================================================================
# Action and game
# =============================================================================


def _punch_reso(x, sr, f, q):
    """`x` through one damped resonance at `f` Hz with quality `q` (a body
    cavity, a skin patch): its impulse response convolved by FFT."""
    n = len(x)
    d = np.pi * f / q  # amplitude decay rate, 1/s
    k = min(n, int(7.0 / d * sr) + 2)
    t = np.arange(k) / sr
    ir = np.exp(-d * t) * np.sin(TAU * f * t)
    return _fftconv(x, ir, n)


def _punch(ev, sr, rng, crng, p, seed):
    """A fist landing, the way film foley records it: the swish of the arm
    and sleeve coming in, the skin slap of the knuckles (a sharp, bright,
    mid-heavy crack), and the thud of the mass behind it, a low, heavily
    damped boom of the torso or skull that is over in a tenth of a second.
    A body punch is mostly thud; a face punch is mostly slap with the crunch
    of bone and cartilage; the heavy movie punch is a long whoosh into a
    bigger slap and a deep thud.
    """
    n = _n(ev, sr)
    t = np.arange(n) / sr
    kind = seed % 3  # 0 body, 1 face, 2 heavy movie punch
    t0 = (crng.uniform(0.07, 0.1), crng.uniform(0.06, 0.09), crng.uniform(0.17, 0.22))[kind]
    t0 = min(t0, 0.3 * n / sr)
    at = int(t0 * sr)

    # 1. the swish: air and cloth torn past by the arm, a noise band whose
    # centre climbs as the fist speeds up, cut off by the hit
    u = np.clip(t / t0, 0.0, 1.0)
    swish_env = np.sin(0.5 * np.pi * u) ** 3 * (t < t0)
    noise = rng.standard_normal(n)
    sw_lo = bandlimit(noise, sr, 250.0, 900.0, order=2)
    sw_hi = bandlimit(noise, sr, 900.0, 3500.0, order=2)
    swish = (sw_lo * (1.0 - u) + sw_hi * u) * swish_env

    # 2. the slap: the knuckles meeting skin, a very short broadband crack
    # coloured by the skin patch (a broad 1-2.5 kHz hump) and a lower smack
    tau_s = (0.006, 0.004, 0.007)[kind]
    crack = _burst(rng, n, at, int(0.05 * sr), tau_s, sr, rise=0.0003)
    slap = (bandlimit(crack, sr, 500.0, 7000.0, order=2)
            + 0.8 * _punch_reso(crack, sr, crng.uniform(1100, 2400), 3.0)
            + 0.5 * _punch_reso(crack, sr, crng.uniform(450, 750), 2.5))

    # 3. the thud: the force of the hit (a 4-8 ms push) into the body, a low
    # resonance with Q ~2 plus the mass's broadband low knock
    width = (0.007, 0.004, 0.009)[kind]
    push = _pulse(n, at, int(width * sr))
    f_body = (crng.uniform(85, 120), crng.uniform(150, 210), crng.uniform(60, 85))[kind]
    thud = _punch_reso(push, sr, f_body, 1.8) + 0.6 * _punch_reso(push, sr, f_body * 2.3, 2.5)
    knock = bandlimit(_burst(rng, n, at, int(0.12 * sr), (0.025, 0.015, 0.04)[kind], sr, rise=0.001), sr, 40.0, 400.0, order=2)
    thud = _norm(thud) + 0.6 * _norm(knock)

    lv = ((1.0, 0.6, 0.15), (0.7, 1.0, 0.12), (1.1, 0.9, 0.3))[kind]  # thud, slap, swish
    mono = lv[0] * thud + lv[1] * _norm(slap) + lv[2] * _norm(swish)
    if kind == 1:
        # bone and cartilage: a few tiny cracks in the first 20 ms
        cnt = int(crng.integers(5, 12))
        cr = _grains(rng, n, t0 + crng.gamma(1.5, 0.004, cnt), crng.uniform(0.3, 1.0, cnt), sr)
        mono += 0.25 * _norm(bandlimit(cr, sr, 1200.0, 5000.0, order=2))
    # the cloth and the body settle: a short soft rustle after the hit
    settle = bandlimit(_burst(rng, n, at + int(0.01 * sr), int(0.15 * sr), 0.04, sr, rise=0.01), sr, 400.0, 3000.0, order=2)
    mono += 0.06 * _norm(settle)
    mono = saturate(_norm(mono) * (1.2, 1.1, 1.6)[kind], 1.3)
    return mono, _decorrelate(mono, sr, crng, taps=3)


def _blade(n, sr, at, rng, crng):
    f = crng.uniform(1100, 1700)
    freqs = f * np.array((1.0, 1.37, 2.13, 2.71, 3.66, 4.9))
    t60 = np.array((1.2, 0.9, 0.7, 0.5, 0.35, 0.25)) * crng.uniform(0.8, 1.2)
    return _strike(n, sr, at, freqs, t60, (1.0, 0.8, 0.6, 0.45, 0.3, 0.2), rng, 0.0002, noise=0.7)


def _sword_clash(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    if kind == 1:
        # drawn from a sheath: a rising scrape ("shing") ending in a ring
        L = min(0.45, 0.6 * dur)
        u = np.clip(t / L, 0, 1)
        scrape = np.zeros(n)
        noise = rng.standard_normal(n)
        blk = int(0.03 * sr)
        for s in range(0, n, blk):
            e = min(n, s + 2 * blk)
            c = 1800 + 2600 * float(u[s])
            scrape[s:e] += bandlimit(noise[s:e], sr, c * 0.8, c * 1.2, order=2) * np.hanning(e - s)
        env = np.sin(0.5 * np.pi * u) ** 2 * np.exp(-np.maximum(t - L, 0) / 0.05)
        ring = _blade(n, sr, int(L * sr), rng, crng)
        mono = 0.8 * _norm(scrape) * env + 0.7 * _norm(ring)
    else:
        hits = (0.004,) if kind == 0 else (0.004, crng.uniform(0.18, 0.3))
        mono = np.zeros(n)
        for h in hits:
            at = int(h * sr)
            ring = _blade(n, sr, at, rng, crng)
            hit = bandlimit(_burst(rng, n, at, int(0.03 * sr), 0.004, sr, rise=0.0005), sr, 1000.0, 9000.0, order=4)
            grind = bandlimit(_burst(rng, n, at, int(0.15 * sr), 0.04, sr), sr, 2500.0, 7000.0, order=4)
            mono += _norm(ring) + 0.5 * _norm(hit) + 0.2 * _norm(grind)
    return mono, _decorrelate(mono, sr, crng)


def _gun_nwave(n, sr, at, length):
    """A pressure N-wave: a near-instant rise (0.1 ms), a linear fall through
    zero to the same negative peak over `length` s, and a 0.1 ms return."""
    out = np.zeros(n)
    k = max(4, int(length * sr))
    r = max(2, int(0.0001 * sr))
    if at + k + 2 * r >= n:
        return out
    up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    out[at : at + r] = up
    out[at + r : at + r + k] = np.linspace(1.0, -1.0, k)
    out[at + r + k : at + 2 * r + k] = -up[::-1]
    return out


def _gun_blast(n, sr, at, length, decay):
    """The muzzle blast: a Friedlander pulse (instant rise, decay through a
    longer, shallower negative phase) of positive phase `length` s."""
    out = np.zeros(n)
    k = min(n - at, int((length * 6 + 0.002) * sr))
    if k < 8:
        return out
    t = np.arange(k) / sr
    r = max(2, int(0.00012 * sr))
    w = (1.0 - t / length) * np.exp(-decay * t / length)
    w[:r] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    out[at : at + k] = w
    return out


def _gunshot(ev, sr, rng, crng, p, seed):
    """A real shot, recorded outdoors. The muzzle blast is a Friedlander
    pressure pulse (a 0.1 ms rise, a 1-2 ms positive phase, a longer negative
    one) with the turbulent roar of the propellant gas behind it; a rifle's
    supersonic bullet adds its own sharp N-wave crack just ahead. The ground
    answers within a few ms, then the surroundings: a handful of discrete,
    darkened echoes (buildings, a treeline) over half a second and a diffuse
    rolling tail that dies over 1-2 s. A distant shot loses its highs to the
    air, so the crack is gone, the blast is a dull thud and the echoes and
    the roll carry most of the energy."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3  # 0 pistol, 1 rifle, 2 distant
    t0 = 0.004
    at = int(t0 * sr)

    T = (crng.uniform(0.0011, 0.0016), crng.uniform(0.0016, 0.0024), crng.uniform(0.002, 0.003))[kind]
    blast = _gun_blast(n, sr, at, T, 2.2)
    # propellant gas: a roar of turbulent noise, loudest in the first 10 ms
    tau = (0.012, 0.018, 0.02)[kind]
    gas = bandlimit(_burst(rng, n, at, int(0.25 * sr), tau, sr, rise=0.0003), sr, 180.0, 6500.0, order=2)
    src = _norm(blast) + (0.55, 0.5, 0.45)[kind] * _norm(gas)
    if kind == 1:
        # the bullet's crack arrives ahead of the muzzle blast
        lead = int(crng.uniform(0.0015, 0.004) * sr)
        crack = _gun_nwave(n, sr, at, crng.uniform(0.00025, 0.0004))
        src = np.roll(src, lead) + 0.8 * crack
    if kind == 2:
        src = bandlimit(src, sr, 60.0, crng.uniform(1100.0, 1600.0), order=2)
    # the mic and the preamp take the hit: a hard, slightly asymmetric clip
    src = saturate(_norm(src) * (2.2, 2.6, 1.3)[kind], 2.0)
    # the ground reflection, a few ms behind and a little darker
    g = int(crng.uniform(0.002, 0.006) * sr)
    src = src + 0.5 * np.concatenate((np.zeros(g), bandlimit(src, sr, None, 4000.0, order=2)[: n - g]))

    # The surroundings, per ear. Discrete echoes from big reflectors (each
    # lower-passed by distance), and a diffuse roll: noise shaped by a slow
    # swell and an exponential decay, low-passed the way air takes the highs
    # off a long path.
    T60 = (0.8, 1.0, 1.6)[kind] * crng.uniform(0.9, 1.1)
    roll_lp = (2600.0, 2200.0, 1100.0)[kind]
    roll_lvl = (0.1, 0.12, 0.35)[kind]

    def space(r):
        er = np.zeros(n)
        k_echo = int(crng.integers(4, 7))
        for i in range(k_echo):
            d = crng.uniform(0.04, 0.55 if kind != 2 else 0.9)
            dd = int(d * sr)
            if dd < n:
                er[dd:] += crng.uniform(0.12, 0.3) * np.exp(-d / 0.4) * src[: n - dd]
        er = bandlimit(er, sr, 90.0, 2400.0 if kind != 2 else 1000.0, order=2)
        env = (1.0 - np.exp(-np.maximum(t - t0, 0) / (0.03 if kind != 2 else 0.08))) * np.exp(-LN1000 * np.maximum(t - t0, 0) / T60)
        roll = bandlimit(r.standard_normal(n), sr, 70.0, roll_lp, order=2) * env
        roll *= 0.7 + 0.3 * _smooth(crng, n, sr, 6.0)
        return er + roll_lvl * _norm(roll)

    direct = src * (1.0 if kind != 2 else 0.6)
    p["reverb"], p["reverb_decay"] = (0.15, 0.18, 0.3)[kind], (1.0, 1.2, 1.8)[kind]
    fade = _fade_out(n, sr, min(0.25, 0.3 * dur))
    return (direct + space(rng)) * fade, (direct + space(crng)) * fade


def _car_horn(ev, sr, rng, crng, p, seed):
    """Two diaphragm horns a third apart (around 400 and 500 Hz): a buzzy,
    harmonic-rich tone through the horn's flare."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    f1 = crng.uniform(390, 430)
    f2 = f1 * crng.uniform(1.2, 1.26)
    honks = {0: [(0.01, min(0.45, dur - 0.05))], 1: [(0.01, 0.16), (0.27, 0.22)], 2: [(0.01, dur - 0.08)]}[kind]
    mono = np.zeros(n)
    amps = [1.0 / k ** 0.9 for k in range(1, 18)]
    for st, L in honks:
        at = int(st * sr)
        k = int(max(0.05, L) * sr)
        if at + k >= n:
            k = n - at - 1
        if k < 100:
            continue
        tt = np.arange(k) / sr
        wob = 1.0 + 0.002 * np.sin(TAU * 6 * tt)
        y = _harmonic(f1 * wob, sr, amps) + 0.9 * _harmonic(f2 * wob, sr, amps, phase=1.0)
        env = np.clip(tt / 0.012, 0, 1) * np.clip((L - tt) / 0.02, 0, 1)
        mono[at : at + k] += y * env
    # the flare: a broad resonance around 2-3 kHz, little below 300 Hz
    mono = bandlimit(mono, sr, 300.0, 4500.0, order=2) + 0.6 * bandlimit(mono, sr, 1800.0, 3200.0, order=2)
    mono = saturate(_norm(mono) * 1.2, 1.3)
    p["reverb"], p["reverb_decay"] = 0.18, 0.9
    return mono, _decorrelate(mono, sr, crng, span=0.03)


def _siren(ev, sr, rng, crng, p, seed):
    """An emergency-vehicle siren heard from the street. The driver is a
    square-ish wave pushed hard through a horn loudspeaker (odd harmonics
    strong, the horn's distortion adds even ones, a broad 1-2.5 kHz horn
    resonance). The pitch pattern is the variant: a wail (a slow rise and a
    slower fall), a yelp (a fast triangle sweep), a two-tone hi-lo. Then the
    street: the vehicle approaches, passes or wails from a distance (travel
    delay so the pitch Doppler-shifts, 1/r level, air absorption), facades
    send back a few late echoes, and its own engine rumbles faintly."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    if kind == 0:  # wail
        period = crng.uniform(4.0, 5.5)
        lo, hi = crng.uniform(600, 720), crng.uniform(1350, 1650)
        u = ((t + crng.uniform(0.0, 0.25) * period) % period) / period
        s = np.where(u < 0.42, np.sin(0.5 * np.pi * u / 0.42), np.cos(0.5 * np.pi * (u - 0.42) / 0.58) ** 1.3)
        f = lo + (hi - lo) * s
    elif kind == 1:  # yelp
        rate = crng.uniform(2.6, 3.4)
        lo, hi = crng.uniform(650, 800), crng.uniform(1500, 1750)
        u = (t * rate + crng.uniform(0, 1)) % 1.0
        tri = 1.0 - np.abs(2.0 * u - 1.0)
        f = lo + (hi - lo) * (0.5 - 0.5 * np.cos(np.pi * tri)) ** 0.7
    else:  # two-tone hi-lo
        lo, hi = ((440.0, 587.0), (460.0, 610.0), (392.0, 523.0))[int(crng.integers(3))]
        half = crng.uniform(0.5, 0.65)
        f = np.where((t / half).astype(int) % 2 == 0, hi, lo)
        k = int(0.015 * sr)
        f = np.convolve(np.pad(f, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    amps = (1.0, 0.25, 0.55, 0.16, 0.34, 0.1, 0.22, 0.07, 0.14, 0.05, 0.09)
    y = _harmonic(f, sr, amps)
    y = saturate(y * 1.4, 1.4)
    y = _air_band(y, sr, (350.0, 7000.0, 2, 1.0), (1000.0, 2600.0, 2, 0.5))
    # the street
    mode = int(crng.integers(3))
    v = crng.uniform(10.0, 17.0)
    if mode == 0:  # approaching and passing
        d0 = crng.uniform(10.0, 20.0)
        x = v * (t - crng.uniform(0.6, 0.85) * dur)
    elif mode == 1:  # passing by, then going away
        d0 = crng.uniform(10.0, 20.0)
        x = v * (t - crng.uniform(0.2, 0.4) * dur)
    else:  # a block or two away, moving slowly across
        d0 = crng.uniform(60.0, 120.0)
        x = 0.4 * v * (t - 0.5 * dur)
    dist = np.sqrt(x**2 + d0**2)
    y = _air_absorb(y, sr, (25.0 / dist) ** 0.7) * (d0 / dist) ** 0.8
    y = _air_doppler(y, sr, dist)
    echo = _air_band(y, sr, (None, 3000.0, 2, 1.0))
    wet = y.copy()
    for _ in range(int(crng.integers(2, 5))):
        d = int(crng.uniform(0.04, 0.3) * sr)
        wet[d:] += crng.uniform(0.15, 0.35) * echo[: n - d]
    rumble = _norm(_air_band(rng.standard_normal(n), sr, (40.0, 300.0, 2, 1.0))) * (d0 / dist) * 0.05
    env = _attack(n, sr, 0.0, 0.12) * _fade_out(n, sr, min(0.5, 0.2 * dur))
    mono = (_norm(wet) + rumble) * env
    p["reverb"], p["reverb_decay"] = 0.25, 1.4
    return mono, _decorrelate(mono, sr, crng, span=0.04)


def _engine(ev, sr, rng, crng, p, seed):
    """A four-cylinder four-stroke car engine. Each firing opens an exhaust
    valve: a steep pressure pulse (blowdown) of about a tenth of the firing
    interval, plus a burst of combustion noise that is a little different for
    each cylinder (so the sound has the firing rate rpm*4/120 Hz, a full
    harmonic series, and the half orders of a real, slightly unbalanced
    engine). The pulses ring the exhaust pipe (an open pipe, ~65-95 Hz and its
    harmonics) and leave through a muffler that opens up with the throttle;
    the intake roars under load, the block and valve train add a steady
    mechanical noise, a little louder on each firing. Idle
    wobbles around 800 rpm; revving sweeps up hard on the throttle and falls
    back off it; passing by puts the car on a road (tyre noise) and moves it
    past the listener: travel delay (Doppler), 1/r, air absorption, pan."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    cyl = 4
    idle = crng.uniform(720, 880)
    if kind == 0:
        rpm = idle * (1 + 0.03 * (_smooth(crng, n, sr, 1.2) - 0.5) + 0.012 * (_smooth(crng, n, sr, 7.0) - 0.5))
        thr = np.full(n, 0.12)
    elif kind == 1:
        rpm = np.full(n, idle)
        thr = np.full(n, 0.1)
        blips = 1 if dur < 2.5 else int(crng.integers(1, 3))
        for i in range(blips):
            t0 = (0.1 + 0.45 * i + crng.uniform(0.0, 0.08)) * dur if blips > 1 else crng.uniform(0.12, 0.25) * dur
            up = crng.uniform(0.3, 0.5) * min(1.0, dur / 3.0)
            tau = crng.uniform(0.35, 0.6) * min(1.0, dur / 3.0)
            top = crng.uniform(3500, 5500)
            r = np.clip((t - t0) / up, 0.0, 1.0)
            curve = np.where(t < t0 + up, 1.0 - (1.0 - r) ** 2, np.exp(-np.maximum(t - t0 - up, 0.0) / tau)) * (t >= t0)
            rpm = np.maximum(rpm, idle + (top - idle) * curve)
            thr = np.maximum(thr, np.where((t >= t0) & (t < t0 + up), 1.0, 0.0))
        k = int(0.04 * sr)
        thr = np.convolve(np.pad(thr, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    else:
        rpm = crng.uniform(1900, 2700) * (1 + 0.02 * (_smooth(crng, n, sr, 1.0) - 0.5))
        thr = np.full(n, 0.45)
    fire = rpm * cyl / 120.0
    ph = np.cumsum(fire) / sr
    k = ph.astype(int)
    frac = ph - k
    kk = int(k[-1]) + 2
    a = ((1.0 + 0.2 * crng.standard_normal(cyl))[np.arange(kk) % cyl]
         * np.exp((0.1 if kind == 0 else 0.05) * crng.standard_normal(kk)))[k]
    since = frac / fire  # seconds since this firing
    pulse = (1.0 - np.exp(-frac / 0.05)) * np.exp(-frac / 0.25)
    lb = int(0.012 * sr)
    tb = np.arange(lb) / sr
    bark_bank = _air_band(crng.standard_normal(cyl * lb), sr, (150.0, 3000.0, 2, 1.0)).reshape(cyl, lb) * np.exp(-tb / 0.005)
    i = (since * sr).astype(int)
    bark = np.where(i < lb, bark_bank[k % cyl, np.minimum(i, lb - 1)], 0.0)
    exc = a * (pulse + 0.2 * bark / max(np.abs(bark_bank).max(), 1e-12))
    exc = exc - np.mean(exc)
    fp = crng.uniform(65.0, 95.0)
    harm = np.array((1, 2, 3, 4, 5, 6, 7, 9, 11), dtype=float)
    pipe = _fftconv(exc, _modes_ir(fp * harm, 0.16 / harm**0.5, 1.0 / harm**0.7, sr, n), n)
    ex = _norm(exc) + 0.5 * _norm(pipe)
    lo, mid, top = _air_band(ex, sr, (25.0, 350.0, 2, 1.0), (350.0, 1500.0, 2, 1.0), (1500.0, 5000.0, 2, 1.0), split=True)
    exhaust = lo + (0.6 + 0.6 * thr) * mid + (0.2 + 0.4 * thr) * top
    lt = int(0.002 * sr)
    tick_bank = _air_band(crng.standard_normal(cyl * lt), sr, (2000.0, 6000.0, 2, 1.0)).reshape(cyl, lt) * np.exp(-np.arange(lt) / sr / 0.0006)
    j = (((frac + 0.5) % 1.0) / fire * sr).astype(int)
    tick = np.where(j < lt, tick_bank[k % cyl, np.minimum(j, lt - 1)], 0.0)
    intake, mech = (_norm(b) for b in _air_band(rng.standard_normal(n), sr, (200.0, 1500.0, 2, 1.0), (400.0, 3000.0, 2, 1.0), split=True))
    intake = intake * (0.5 + 0.5 * np.exp(-frac / 0.3)) * thr**2
    mech = mech * (0.7 + 0.3 * np.exp(-frac / 0.2)) * (0.6 + 0.4 * thr)  # block, valvetrain, combustion roar
    mono = _norm(exhaust) + 0.02 * _norm(tick) + 0.25 * intake + 0.06 * mech
    mono = mono * (0.55 + 0.45 * thr) * (rpm / rpm.max()) ** 0.3
    if kind == 2:
        v = crng.uniform(12.0, 20.0)
        d0 = crng.uniform(4.0, 8.0)
        x = v * (t - crng.uniform(0.4, 0.6) * dur)
        dist = np.sqrt(x**2 + d0**2)
        tyres = _norm(_air_band(rng.standard_normal(n), sr, (300.0, 2500.0, 2, 0.3), (30.0, 200.0, 2, 0.25))) * 0.4
        mono = _norm(mono) + tyres
        mono = _air_doppler(_air_absorb(mono, sr, d0 / dist) * (d0 / dist), sr, dist)
        side = x / dist
        env = _attack(n, sr, 0.0, 0.05) * _fade_out(n, sr, min(0.4, 0.2 * dur))
        return mono * np.sqrt(0.5 * (1 - side)) * env, mono * np.sqrt(0.5 * (1 + side)) * env
    mono = mono * _attack(n, sr, 0.0, 0.1) * _fade_out(n, sr, min(0.4, 0.2 * dur))
    return mono, _decorrelate(mono, sr, crng, span=0.02)


def _church_bell(ev, sr, rng, crng, p, seed):
    """A tower bell heard from the street. A tuned bronze bell's partials,
    relative to the nominal: hum 0.25, prime 0.5, tierce 0.6 (the minor
    third that gives a bell its sound), quint 0.75, nominal 1, then the
    superquint 1.5, the octave nominal 2 and a forest of upper partials to
    4-5x. The iron clapper's blow excites the upper partials strongly, and
    they die within a second or two while the hum rings on for tens of
    seconds. Every partial is a doublet (the bell is not perfectly round),
    so they beat slowly, and the two ears hear different mixes of each pair.

    `p` may carry a Spec's params (`orb.grammar.EVENT_PARAMS`): `f0` puts the
    strike note on a pitch, `ring` scales the decay, `hardness` is the
    clapper, `tierce` the bell's third and `beat` the width of the doublets.
    Each one unset leaves the bell exactly as the seed casts it, and every
    random draw is made either way, so a param changes only what it names."""
    n = _n(ev, sr)
    dur = n / sr
    fN = crng.uniform(520, 980)
    if "f0" in p:
        # The strike note is heard an octave under the nominal, where a tuned
        # bell's prime sits (the nominal, superquint and octave nominal are
        # the 2nd, 3rd and 4th harmonics of it).
        fN = 2.0 * float(p["f0"])
    tierce = float(p.get("tierce", 0.6))
    ratios = np.array((0.25, 0.5, tierce, 0.75, 1.0, 1.25, 1.335, 1.5, 2.0, 2.5, 2.67, 3.0, 3.36, 4.0, 4.5, 5.0))
    ratios = ratios * crng.uniform(0.993, 1.007, len(ratios))
    ratios[0] *= crng.uniform(0.97, 1.03)  # old bells: hum and prime are rarely true
    ratios[1] *= crng.uniform(0.98, 1.02)
    ratios[4] = 1.0
    size = (700.0 / fN) ** 0.8
    t60 = np.array((24, 12, 9, 6, 6, 3, 3, 3.5, 2.6, 1.8, 1.6, 1.4, 1.1, 0.9, 0.7, 0.55)) * size
    t60 = t60 * float(p.get("ring", 1.0))
    g = np.array((0.35, 0.5, 0.75, 0.25, 1.0, 0.2, 0.3, 0.55, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.12, 0.08))
    g = g * crng.uniform(0.75, 1.25, len(g))
    hard = float(p.get("hardness", 0.5))
    if hard != 0.5:
        # A harder clapper is in contact for less time, so more of the blow
        # goes into the partials above the nominal; felt leaves them out.
        g = g * np.where(ratios > 1.0, (0.15 + 1.7 * hard) ** np.log2(np.maximum(ratios, 1.0)), 1.0)
    fa, fb, ta, tb = _bells_doublets(fN * ratios, t60, crng, float(p.get("beat", 0.0012)))
    # each ear hears its own blend of the two halves of every doublet
    u_l = crng.uniform(0.25, 0.75, len(g))
    u_r = np.clip(u_l + crng.uniform(-0.2, 0.2, len(g)), 0.1, 0.9)
    ir_l = np.zeros(n)
    ir_r = np.zeros(n)
    for i in range(len(g)):
        for f, d, ul, ur in ((fa[i], ta[i], u_l[i], u_r[i]), (fb[i], tb[i], 1.0 - u_l[i], 1.0 - u_r[i])):
            k = min(n, int(1.1 * d * sr) + 1)
            m = g[i] * _bells_mode(f, d, sr, k)
            ir_l[:k] += ul * m
            ir_r[:k] += ur * m
    if hard < 0.5:
        # ...and a soft clapper lands slower: up to 12 ms for felt.
        k = min(n, max(2, int(0.024 * (0.5 - hard) * sr)))
        ir_l[:k] *= np.linspace(0.0, 1.0, k)
        ir_r[:k] *= np.linspace(0.0, 1.0, k)
    strikes = [0.005] if seed % 2 == 0 else list(np.arange(0.005, max(0.01, dur - 0.5), 2.5))
    amps = crng.uniform(0.85, 1.0, len(strikes))
    # the clapper's clank: iron on bronze, a few ms
    k = int(0.05 * sr)
    clank = np.zeros(n)
    for st in strikes:
        _bells_place(clank, int(st * sr), bandlimit(_burst(rng, k, 0, int(0.02 * sr), 0.003, sr), sr, 1000.0, 6000.0, order=2, pad=1024))
    fo = _fade_out(n, sr, min(1.5, 0.3 * dur))
    out = []
    for ir in (ir_l, ir_r):
        y = np.zeros(n)
        for st, a in zip(strikes, amps):
            _bells_place(y, int(st * sr), ir, a)
        out.append((_norm(y) + 0.05 * (2.0 * hard) ** 2 * _norm(clank)) * fo)
    return out[0], out[1]


def _fireworks(ev, sr, rng, crng, p, seed):
    """Aerial shells. A mortar thumps and the shell climbs (whistling, if it
    is a whistle shell); at the top the bursting charge sends out a shock: a
    short N-wave, a sharp crack over the whole band near by, duller far away
    (a longer N-wave, air absorption). What follows is the outdoors: echoes
    from buildings and terrain arriving over a second or two, and a rumbling
    tail whose highs die first. A crackling shell's stars then pop by the
    hundred, each a tiny N-wave. A distant show is many shells, some in quick
    salvos, with overlapping echoes and faint launch thumps."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    # stereo buffers, one per component, filtered once at the end
    crack = np.zeros((2, n))  # already band-limited per shell
    echo = np.zeros((2, n))
    body = np.zeros((2, n))
    launch = np.zeros((2, n))
    lift = np.zeros((2, n))
    crackle = np.zeros((2, n))
    whistle = np.zeros(n)
    tails = np.zeros((3, 2, n))  # low / mid / high tail envelopes
    taus = (0.9, 0.45, 0.15)
    ln = int(0.03 * sr)

    def shell(t_launch, t_burst, whistling, crackling, level, far):
        g = level * np.sqrt(np.array((1.0, 0.0)) + np.array((-1.0, 1.0)) * crng.uniform(0.2, 0.8))[:, None]
        if 0.0 <= t_launch < dur:
            la = int(t_launch * sr)
            launch[:] += (0.25 if far < 0.5 else 0.12) * g * _burst(rng, n, la, int(0.12 * sr), 0.025, sr, rise=0.002)
            lift[:] += 0.06 * g * _burst(rng, n, la, int(0.4 * sr), 0.12, sr, rise=0.01)
            if whistling:
                u = np.clip((t - t_launch) / (t_burst - t_launch), 0, 1)
                f = crng.uniform(1100, 1500) + crng.uniform(800, 1300) * u
                gate = np.sin(np.pi * np.clip(u * 1.05, 0, 1)) ** 0.5 * (t >= t_launch) * (t < t_burst)
                whistle[:] += _air_narrow(rng, n, sr, f, 60.0) * gate
        if t_burst >= dur - 0.15:
            return
        b = int(t_burst * sr)
        k = max(4, int((0.0015 + 0.004 * far) * sr))
        nw = np.zeros(ln + k)
        nw[:k] = np.linspace(1.0, -1.0, k)
        nw = _air_band(nw, sr, (30.0, 9000.0 - 6000.0 * far, 2, 1.0))
        m = min(len(nw), n - b)
        crack[:, b : b + m] += g * nw[:m]
        for d in np.sort(crng.uniform(0.08, 1.3, int(crng.integers(3, 6)))):
            j = b + int(d * sr)
            if j < n:
                m = min(len(nw), n - j)
                echo[:, j : j + m] += crng.uniform(0.15, 0.4) / (1.0 + d) * g * nw[:m]
        body[:] += (0.35 - 0.15 * far) * g * _burst(rng, n, b, int(0.4 * sr), 0.06, sr, rise=0.002)
        after = np.maximum(t - t_burst, 0.0)
        grow = (1.0 - np.exp(-after / 0.03)) * (t >= t_burst) * (0.25 + 0.3 * far)
        for i, (tau, a) in enumerate(zip(taus, (1.0, 0.8, 0.4 * (1.0 - far)))):
            tails[i] += a * g * grow * np.exp(-after / tau)
        if crackling:
            cnt = int(crng.integers(250, 450)) if far < 0.5 else int(crng.integers(60, 140))
            tt = t_burst + crng.uniform(0.25, 0.5) + crng.gamma(2.2, 0.3, cnt)
            amp = crng.lognormal(0.0, 0.5, cnt) * np.exp(-(tt - t_burst) / 2.0)
            side = crng.uniform(0.0, 1.0, cnt)
            for c, gc in enumerate((np.sqrt(1.0 - side), np.sqrt(side))):
                crackle[c] += level * _grains(rng, n, tt, amp * gc, sr)

    if kind == 2:  # a distant show
        tm = crng.uniform(0.0, 0.3)
        while tm < dur - 1.0:
            far = crng.uniform(0.5, 1.0)
            shell(tm - crng.uniform(1.2, 2.0), tm, False, crng.random() < 0.3, crng.uniform(0.4, 1.0), far)
            if crng.random() < 0.3:  # a salvo
                shell(-1.0, tm + crng.uniform(0.05, 0.2), False, False, crng.uniform(0.3, 0.8), far)
            tm += crng.uniform(0.4, 1.3)
        p["reverb"], p["reverb_decay"] = 0.3, 2.0
        hi_crackle = 5000.0
    else:
        flight = crng.uniform(0.9, 1.4) * min(1.0, dur / 3.0)
        shell(0.02, 0.02 + flight, kind == 0 or crng.random() < 0.5, kind == 1, 1.0, crng.uniform(0.0, 0.25))
        p["reverb"], p["reverb_decay"] = 0.25, 1.6
        hi_crackle = 8000.0

    def peak(x):
        return max(float(np.abs(x).max()), 1e-12)

    out = crack / peak(crack)
    lows = _air_band(np.stack([launch, body]), sr, (40.0, 280.0, 2, 1.0))
    out += 0.25 * lows[0] / peak(lows[0]) + 0.35 * lows[1] / peak(lows[1])
    if np.abs(lift).max() > 0:
        y = _air_band(lift, sr, (500.0, 3000.0, 2, 1.0))
        out += 0.06 * y / peak(y)
    e = _air_band(echo, sr, (None, 2500.0, 2, 1.0))
    out += 0.5 * e / peak(crack)
    noise = _air_band(rng.standard_normal((2, n)), sr, (40.0, 400.0, 2, 1.0), (300.0, 1500.0, 2, 1.0), (1500.0, 5000.0, 2, 1.0), split=True)
    tail = sum(b / peak(b) * env for b, env in zip(noise, tails))
    out += tail
    if np.abs(crackle).max() > 0:
        k = max(4, int(0.0005 * sr))
        cr = np.stack([np.convolve(c, np.linspace(1.0, -1.0, k), mode="same") for c in crackle])
        cr = _air_band(cr, sr, (600.0, hi_crackle, 2, 1.0))
        out += (0.45 if kind != 2 else 0.25) * cr / peak(cr)
    if np.abs(whistle).max() > 0:
        out += 0.2 * whistle / peak(whistle)
    fo = _fade_out(n, sr, min(0.8, 0.25 * dur))
    return out[0] * fo, out[1] * fo


FNS = {
    "bird_chirp": _bird_chirp,
    "crickets": _crickets,
    "wave_crash": _wave_crash,
    "wind_gust": _wind_gust,
    "splash": _splash,
    "frog": _frog,
    "owl": _owl,
    "glass_break": _glass_break,
    "clock_tick": _clock_tick,
    "phone_ring": _phone_ring,
    "keyboard_typing": _keyboard_typing,
    "mouse_click": _mouse_click,
    "camera_shutter": _camera_shutter,
    "paper_rustle": _paper_rustle,
    "cork_pop": _cork_pop,
    "coin_drop": _coin_drop,
    "metal_clang": _metal_clang,
    "wood_creak": _wood_creak,
    "heartbeat": _heartbeat,
    "applause": _applause,
    "pouring_water": _pouring_water,
    "can_opening": _can_opening,
    "clock_alarm": _clock_alarm,
    "punch": _punch,
    "sword_clash": _sword_clash,
    "gunshot": _gunshot,
    "car_horn": _car_horn,
    "siren": _siren,
    "engine": _engine,
    "church_bell": _church_bell,
    "fireworks": _fireworks,
}

assert set(FNS) == set(ROLES) == set(VARIANTS) == set(PARAMS)

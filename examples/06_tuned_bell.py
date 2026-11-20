#!/usr/bin/env python3
"""Put a sound on a note.

A role's pitch is normally fixed (the synth `bell` rings at 880 Hz) or left to
the seed (each `church_bell` is cast somewhere between C4 and B4). A cue that
sits over music has to follow its key, so a Spec event can carry `params`:
`f0` is the note, and the church bell also takes `ring` (how long it rings),
`hardness` (the clapper), `tierce` (its minor or major third) and `beat`.

Three bells toll a D minor triad one at a time, then together, harder and
longer. No model and no torch: params are Spec-only, like `bed_db`.

    python examples/06_tuned_bell.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.api import render_spec, save_wav  # noqa: E402
from orb.tools import validate_spec  # noqa: E402

NOTES = {"D4": 293.66, "F4": 349.23, "A4": 440.00}


def bell(t: float, note: str, seed: int, **params: float) -> dict:
    # An even seed is a single strike; the seed also casts the bell (its
    # detuned hum and prime, the weight of each partial).
    return {"role": "church_bell", "t": t, "dur": 12.0 - t, "gain_db": -6.0, "pan": 0.0, "seed": seed,
            "params": {"f0": NOTES[note], **params}}


SPEC = {
    "duration": 12.0,
    "texture": "none",
    "mood": "calm",
    "events": [
        bell(0.5, "D4", 0),
        bell(2.5, "F4", 2),
        bell(4.5, "A4", 4),
        bell(7.0, "D4", 0, hardness=0.9, ring=1.5),
        bell(7.0, "F4", 2, hardness=0.9, ring=1.5),
        bell(7.0, "A4", 4, hardness=0.9, ring=1.5),
    ],
}


def main() -> None:
    report = validate_spec(SPEC)
    assert report["valid"], report["errors"]
    audio = render_spec(report["spec"])
    save_wav(audio, ROOT / "out/example_06_tuned_bells.wav")

    for ev in report["spec"]["events"][:3]:
        print(f"church_bell @ {ev['t']:.1f} s  strike note {ev['params']['f0']:.2f} Hz")
    print(f"then all three at 7.0 s, hardness 0.9, ring x1.5   {audio.shape}  {audio.dtype}")
    for w in report["warnings"]:
        print(f"note     {w}")
    print("wrote    out/example_06_tuned_bells.wav")


if __name__ == "__main__":
    main()

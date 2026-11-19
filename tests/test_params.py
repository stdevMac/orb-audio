"""Per-event params: a hand-written Spec can put a pitched sound on a note.

The planner and the caption grammar know nothing about params (like `bed_db`
they are render-only), so the default path must not move: a Spec without
params renders exactly what it rendered before, and a param set to its
default is the same audio as leaving it out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orb.api import dict_to_spec, render_spec, spec_to_dict
from orb.grammar import EVENT_PARAMS, Spec, SpecEvent, event_param_errors, score_from_spec
from orb.render.events import ROLE_PARAMS, render_event
from orb.score import EVENT_ROLES
from orb.tools import describe_grammar, dispatch, validate_spec

SR = 48000


def _one(role: str, params: dict | None = None, seed: int = 0, dur: float = 3.0) -> np.ndarray:
    ev = {"role": role, "t": 0.0, "dur": dur, "gain_db": -6.0, "pan": 0.0, "seed": seed}
    if params is not None:
        ev["params"] = params
    spec = dict_to_spec({"duration": dur, "texture": "none", "mood": "neutral", "events": [ev]})
    (event,) = score_from_spec(spec).events
    return render_event(event, SR, seed=0)


def _peak_hz(audio: np.ndarray, near: float, span: float = 0.04, start: float = 0.3, length: float = 1.5) -> float:
    """The strongest spectral peak within `span` of `near`."""
    x = audio.mean(axis=1)[int(start * SR) : int((start + length) * SR)]
    x = x * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x, 1 << 20))
    freqs = np.fft.rfftfreq(1 << 20, 1.0 / SR)
    band = (freqs > near * (1 - span)) & (freqs < near * (1 + span))
    return float(freqs[band][np.argmax(spec[band])])


def _cents(f: float, ref: float) -> float:
    return 1200.0 * math.log2(f / ref)


def _centroid(audio: np.ndarray, seconds: float) -> float:
    x = audio.mean(axis=1)[: int(seconds * SR)]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return float((freqs * spec).sum() / spec.sum())


def _seconds_to_fall(audio: np.ndarray, db: float = 40.0) -> float:
    env = np.abs(audio).max(axis=1)
    hop = SR // 20
    blocks = env[: len(env) // hop * hop].reshape(-1, hop).max(axis=1)
    below = np.where(blocks < blocks.max() * 10 ** (-db / 20))[0]
    return float(below[0] * hop / SR) if len(below) else float("inf")


# -- the registry ------------------------------------------------------------


def test_registry_names_real_roles_and_sane_ranges():
    for role, params in EVENT_PARAMS.items():
        assert role in EVENT_ROLES
        for name, (lo, hi, doc) in params.items():
            assert lo < hi and doc
            default = ROLE_PARAMS[role].get(name)
            # A param either has a default inside its range, or none at all
            # (the seed picks it): never a default the validator would reject.
            assert default is None or lo <= default <= hi, (role, name, default)


def test_describe_grammar_lists_params_with_defaults():
    roles = describe_grammar()["roles"]
    assert roles["bell"]["params"]["f0"] == {
        "range": [110.0, 2400.0],
        "default": 880,
        "description": EVENT_PARAMS["bell"]["f0"][2],
    }
    # The church bell's note has no default: unset, the seed picks the bell.
    assert roles["church_bell"]["params"]["f0"]["default"] is None
    assert roles["church_bell"]["params"]["ring"]["default"] == 1.0
    assert "params" not in roles["whoosh"]


# -- the default path does not move ------------------------------------------


@pytest.mark.parametrize("role", sorted(EVENT_PARAMS))
def test_empty_params_is_no_params(role):
    assert np.array_equal(_one(role), _one(role, {}))


@pytest.mark.parametrize("role", sorted(EVENT_PARAMS))
def test_a_param_at_its_default_is_the_same_audio(role):
    defaults = {k: ROLE_PARAMS[role][k] for k in EVENT_PARAMS[role] if k in ROLE_PARAMS[role]}
    assert defaults
    for seed in (0, 1, 7):
        assert np.array_equal(_one(role, seed=seed), _one(role, defaults, seed=seed)), (role, seed)


def test_planner_spec_serialises_as_before():
    spec = Spec(4.0, "wind", "bright", [SpecEvent("bell", 1.0, 1.4, -14.0, 0.25, 99)])
    assert "params" not in spec_to_dict(spec)["events"][0]


def test_params_roundtrip_through_the_dict():
    data = {
        "duration": 4.0,
        "texture": "none",
        "mood": "calm",
        "events": [{"role": "church_bell", "t": 0.0, "dur": 3.0, "gain_db": -6.0, "pan": 0.0, "seed": 4,
                    "params": {"f0": 440, "ring": 2.0}}],
    }
    back = spec_to_dict(dict_to_spec(data))
    assert back["events"][0]["params"] == {"f0": 440.0, "ring": 2.0}
    assert spec_to_dict(dict_to_spec(back)) == back


# -- pitch -------------------------------------------------------------------


@pytest.mark.parametrize("hz", [220.0, 293.66, 440.0, 659.26])
def test_bell_lands_on_the_asked_pitch(hz):
    audio = _one("bell", {"f0": hz, "t60": 3.0}, dur=2.5)
    assert abs(_cents(_peak_hz(audio, hz), hz)) < 1.0


@pytest.mark.parametrize("hz", [261.63, 293.66, 369.99, 440.0])
@pytest.mark.parametrize("seed", [0, 2, 6])
def test_church_bell_strike_note_is_half_its_nominal(hz, seed):
    """`f0` is the strike note; the nominal partial sits an octave up, whatever bell the seed casts."""
    # With the doublets closed the nominal is a single line, exactly on 2 * f0.
    pure = _one("church_bell", {"f0": hz, "beat": 0.0}, seed=seed, dur=4.0)
    assert abs(_cents(_peak_hz(pure, 2 * hz), 2 * hz)) < 0.5
    # As cast, each partial is a pair straddling its frequency by at most
    # 0.06 % (about a cent), and the louder half is the peak a tuner reads.
    cast = _one("church_bell", {"f0": hz}, seed=seed, dur=4.0)
    assert abs(_cents(_peak_hz(cast, 2 * hz), 2 * hz)) < 2.0


def test_church_bell_f0_changes_the_note_and_nothing_random():
    """Asking for the note the seed would have picked gives the seed's own bell back."""
    # Recover the seed's own nominal from the as-cast render, then ask for it.
    as_cast = _one("church_bell", seed=2, dur=4.0)
    spec = np.abs(np.fft.rfft(as_cast.mean(axis=1)[int(0.3 * SR) : int(1.8 * SR)] * np.hanning(int(1.5 * SR)), 1 << 20))
    freqs = np.fft.rfftfreq(1 << 20, 1.0 / SR)
    band = (freqs > 520) & (freqs < 980)
    nominal = float(freqs[band][np.argmax(spec[band])])
    asked = _one("church_bell", {"f0": nominal / 2}, seed=2, dur=4.0)
    # Same bell to within the FFT's resolution of the recovered nominal.
    assert abs(_cents(_peak_hz(asked, nominal), nominal)) < 1.0
    a, b = as_cast.mean(axis=1), asked.mean(axis=1)
    assert np.corrcoef(np.abs(a[:SR]), np.abs(b[:SR]))[0, 1] > 0.9


def test_sub_drop_and_impact_follow_f0():
    low, high = _one("impact", {"f0": 50.0}, dur=1.0), _one("impact", {"f0": 200.0}, dur=1.0)
    assert _centroid(low, 0.5) < _centroid(high, 0.5)
    drop = _one("sub_drop", {"f0": 220.0, "f_end": 55.0}, dur=1.5)
    assert drop.shape == (int(1.5 * SR), 2) and np.isfinite(drop).all()


# -- the church bell's other dials ------------------------------------------


def test_ring_scales_the_decay():
    short, cast, long = (_seconds_to_fall(_one("church_bell", {"f0": 440.0, "ring": r}, dur=20.0)) for r in (0.35, 1.0, 2.5))
    assert short < cast < long
    assert long > 2 * short


def test_hardness_brightens_the_blow_and_felt_lands_slower():
    felt, cast, hard = (_one("church_bell", {"f0": 440.0, "hardness": h}, dur=4.0) for h in (0.05, 0.5, 1.0))
    assert _centroid(felt, 0.15) < _centroid(cast, 0.15) < _centroid(hard, 0.15)
    rise = lambda a: int(np.argmax(np.abs(a).max(axis=1) >= 0.9 * np.abs(a).max()))  # noqa: E731
    assert rise(felt) > rise(hard)


def test_tierce_moves_only_the_third():
    minor = _one("church_bell", {"f0": 440.0, "tierce": 0.6, "beat": 0.0}, dur=4.0)
    major = _one("church_bell", {"f0": 440.0, "tierce": 0.63, "beat": 0.0}, dur=4.0)
    assert not np.array_equal(minor, major)
    # the nominal is where it was
    assert abs(_cents(_peak_hz(major, 880.0), 880.0)) < 1.0


def test_tuned_render_is_deterministic():
    spec = {"duration": 4.0, "texture": "none", "mood": "calm",
            "events": [{"role": "church_bell", "t": 0.2, "dur": 3.5, "seed": 3, "gain_db": -6.0, "pan": 0.0,
                        "params": {"f0": 293.66, "ring": 1.5, "hardness": 0.8}}]}
    assert np.array_equal(render_spec(spec), render_spec(spec))


# -- validation --------------------------------------------------------------


def test_event_param_errors():
    assert event_param_errors("bell", None) == []
    assert event_param_errors("bell", {"f0": 440}) == []
    assert "unknown" in event_param_errors("bell", {"ring": 2.0})[0]
    assert "takes no params" in event_param_errors("whoosh", {"f0": 440})[0]
    assert "outside" in event_param_errors("bell", {"f0": 20000})[0]
    assert "finite" in event_param_errors("bell", {"f0": float("nan")})[0]
    assert "finite" in event_param_errors("bell", {"f0": True})[0]
    assert "finite" in event_param_errors("bell", {"f0": "440"})[0]
    assert "object" in event_param_errors("bell", [440])[0]


def test_spec_event_rejects_bad_params():
    with pytest.raises(ValueError, match="unknown for role"):
        SpecEvent("whoosh", 0.0, 1.0, -6.0, 0.0, 0, {"f0": 440.0})
    with pytest.raises(ValueError, match="outside"):
        dict_to_spec({"duration": 2.0, "texture": "none", "mood": "calm",
                      "events": [{"role": "church_bell", "t": 0.0, "dur": 1.0, "params": {"tierce": 0.9}}]})
    assert SpecEvent("bell", 0.0, 1.0, -6.0, 0.0, 0, {}).params is None


def test_validate_spec_reports_params_by_path_and_warns():
    base = {"duration": 3.0, "texture": "none", "mood": "calm"}
    bad = validate_spec({**base, "events": [{"role": "bell", "t": 0.0, "dur": 1.0, "params": {"f0": 5, "nope": 1}}]})
    assert not bad["valid"]
    assert any(e.startswith("events[0].params.f0:") for e in bad["errors"])
    assert any(e.startswith("events[0].params.nope:") for e in bad["errors"])

    good = validate_spec({**base, "events": [{"role": "bell", "t": 0.0, "dur": 1.0, "params": {"f0": 440}}]})
    assert good["valid"] and good["spec"]["events"][0]["params"] == {"f0": 440.0}
    assert any("render-only" in w and "params" in w for w in good["warnings"])

    plain = validate_spec({**base, "events": [{"role": "bell", "t": 0.0, "dur": 1.0}]})
    assert "params" not in plain["spec"]["events"][0]
    assert not any("params" in w for w in plain["warnings"])


def test_tokenizer_refuses_params():
    pytest.importorskip("torch")  # orb.model imports torch; everything above runs on numpy alone
    from orb.model.spec_tokenizer import SpecTokenizer

    spec = Spec(4.0, "wind", "calm", [SpecEvent("bell", 1.0, 1.4, -14.0, 0.0, 9, {"f0": 440.0})])
    with pytest.raises(ValueError, match="params cannot be tokenized"):
        SpecTokenizer().encode_spec(spec)


def test_render_tool_takes_params_and_names_differ(tmp_path, monkeypatch):
    monkeypatch.setenv("ORB_OUTPUT_DIR", str(tmp_path))
    base = {"duration": 3.0, "texture": "none", "mood": "calm"}
    a = dispatch("orb_render_spec", {"spec": {**base, "events": [{"role": "church_bell", "t": 0.0, "dur": 2.5, "seed": 0, "params": {"f0": 293.66}}]}})
    b = dispatch("orb_render_spec", {"spec": {**base, "events": [{"role": "church_bell", "t": 0.0, "dur": 2.5, "seed": 0, "params": {"f0": 440.0}}]}})
    assert a["path"] != b["path"]  # the content hash covers params
    assert a["spec"]["events"][0]["params"] == {"f0": 293.66}

"""ORB as a tool for agents, framework-agnostic.

This module is the single source of truth for ORB's agent tools. It holds:

* ``TOOLS``: tool definitions (name, description, JSON Schema for the input),
  usable as-is with any tool-use API;
* :func:`anthropic_tools` / :func:`openai_tools`: the same definitions in the
  shape each API expects;
* :func:`dispatch`: run a tool call and get a JSON-serialisable dict back;
* :func:`run_tool`: the same, but never raises, and returns text for the
  ``tool_result`` / ``tool`` message of your agent loop;
* :func:`validate_spec`: a strict, friendly validator for agent-authored Specs.

The MCP server (``orb.mcp_server``) is a thin transport around this module.

Only the ``orb_generate`` tool with a ``caption`` needs torch and the planner
weights, and torch is imported lazily, on first use. Everything else is
numpy-only.

Minimal agent loop (any LLM, any SDK)::

    from orb.tools import anthropic_tools, run_tool

    tools = anthropic_tools()            # pass to the model
    # ... the model answers with a tool call (name, input) ...
    text, is_error = run_tool(name, input)
    # ... send `text` back as the tool result ...
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .grammar import (
    AMBIENCE_BED_DB,
    BED_CHOICES,
    BED_DB_RANGE,
    DEFAULT_BED_DB,
    EVENT_PARAMS,
    EVENT_STYLE,
    MAX_EVENTS,
    MOODS,
    NO_BED,
    PAN_LABELS,
    Spec,
    SpecEvent,
    event_param_errors,
)
from .render.events import ROLE_PARAMS as _ROLE_PARAMS
from .render.foley import VARIANTS as _FOLEY_VARIANTS
from .render.natural import VARIANTS as _NATURAL_VARIANTS
from .score import AMBIENCE_TEXTURES, EVENT_ROLES, TEXTURES

FOLEY_VARIANTS = {**_FOLEY_VARIANTS, **_NATURAL_VARIANTS}

SAMPLE_RATE = 48000

# Hard limits: outside these the call fails. The planner was trained on
# 2-30 s; the DSP renderer is fine outside that, so it is only a warning.
MIN_DURATION = 0.1
MAX_DURATION = 60.0
PLANNER_MIN_DURATION = 2.0
PLANNER_MAX_DURATION = 30.0
MIN_GAIN_DB = -60.0
MAX_GAIN_DB = 0.0
MAX_VARIANTS = 32

_SPEC_KEYS = {"duration", "texture", "mood", "events", "bed_db"}
_EVENT_KEYS = {"role", "t", "dur", "gain_db", "pan", "seed", "params"}


class OrbToolError(ValueError):
    """A tool call failed in a way the agent can fix (bad input, missing weights)."""


# -- JSON schemas ------------------------------------------------------------

EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "One timed sound event. Times are in seconds from the start of the cue.",
    "properties": {
        "role": {"type": "string", "enum": list(EVENT_ROLES), "description": "What kind of sound."},
        "t": {"type": "number", "minimum": 0, "description": "Onset time in seconds. Exact to the sample."},
        "dur": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Length in seconds; t + dur must be <= duration. Defaults per role.",
        },
        "gain_db": {
            "type": "number",
            "minimum": MIN_GAIN_DB,
            "maximum": MAX_GAIN_DB,
            "description": (
                "Level relative to the other events, in dB (0 = loudest). The whole cue is "
                "loudness-normalised, so this sets balance, not absolute volume. Defaults per role."
            ),
        },
        "pan": {"type": "number", "minimum": -1, "maximum": 1, "description": "-1 left, 0 centre, 1 right."},
        "seed": {
            "type": "integer",
            "minimum": 0,
            "maximum": 255,
            "description": (
                "Picks one take of this event. Same seed, same sound. For the natural and foley roles "
                "it also picks the variant: thunder distance (seed % 4: 0 close crack .. 3 far rumble), "
                "footsteps surface and pace, water drop drip or plop, door weight, explosion distance."
            ),
        },
        "params": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": (
                "Renderer parameters for this one event, for the roles that take them (see "
                "roles.<role>.params in orb_describe_grammar): for example {\"f0\": 440} puts a bell or a "
                "church bell on A4, so a cue can follow the key of the music under it. Spec only: a "
                "caption cannot express params."
            ),
        },
    },
    "required": ["role", "t"],
    "additionalProperties": False,
}

SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "An ORB Spec: the complete, editable description of a sound cue. The audio is a pure, "
        "deterministic function of it."
    ),
    "properties": {
        "duration": {
            "type": "number",
            "minimum": MIN_DURATION,
            "maximum": MAX_DURATION,
            "description": "Cue length in seconds (planner range 2-30; the DSP renderer accepts 0.1-60).",
        },
        "texture": {
            "type": "string",
            "enum": list(BED_CHOICES),
            "description": (
                "The ambience bed under the events (filtered noise). Use \"none\" for no bed at all, "
                "so the cue is digital silence between events: the right choice for UI sounds, "
                "notifications and anything that must loop or layer cleanly."
            ),
        },
        "bed_db": {
            "type": "number",
            "minimum": BED_DB_RANGE[0],
            "maximum": BED_DB_RANGE[1],
            "description": (
                f"Bed level in dB relative to the events' loudness (default {DEFAULT_BED_DB:g}). "
                "Lower is quieter. Not allowed with texture \"none\"."
            ),
        },
        "mood": {"type": "string", "enum": list(MOODS), "description": "Overall energy and colour."},
        "events": {"type": "array", "maxItems": MAX_EVENTS, "items": EVENT_SCHEMA},
    },
    "required": ["duration", "texture", "mood"],
    "additionalProperties": False,
}

_OUT = {
    "output_dir": {
        "type": "string",
        "description": (
            "Sub-directory for the WAV file(s), relative to the server's output root "
            "($ORB_OUTPUT_DIR or ./orb_out). Paths that resolve outside the root are rejected."
        ),
    },
    "name": {
        "type": "string",
        "description": "Base file name without extension. Defaults to a content hash of the Spec.",
    },
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "orb_generate",
        "description": (
            "Make a sound cue (48 kHz stereo WAV) and return its file path, the exact Spec it was "
            "rendered from, and an audio audit. Give EITHER a `spec` (exact, numpy-only, preferred "
            "when you know the timings) OR a `caption` in the ORB caption grammar (uses the 5M-param "
            "planner; needs torch and the planner weights). A grammar caption with `no bed` is "
            "read exactly, without the planner or torch. Call orb_describe_grammar first if you "
            "have not seen the roles, textures and moods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "string",
                    "description": (
                        "e.g. '6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a', "
                        "or '2.00-second playful cue, no bed, with ui pop @0.50 0.16s -12dB C #03' "
                        "(no bed: silence between events, no planner needed)"
                    ),
                },
                "spec": SPEC_SCHEMA,
                "seed": {"type": "integer", "minimum": 0, "default": 0, "description": "Render seed."},
                "detail": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Apply the optional CC BY-NC detail model if its weights are present. Slow: "
                        "roughly 80-95 s for a 6 s cue on CPU. The file name gets a _detail suffix."
                    ),
                },
                **_OUT,
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orb_render_spec",
        "description": (
            "Render a Spec to a WAV file. Deterministic, numpy-only, no model, about 10x faster "
            "than real time on a laptop CPU. Returns the path, the normalised Spec and an audit "
            "(peak, rms_dbfs, dc, clipping, passes)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": SPEC_SCHEMA,
                "seed": {"type": "integer", "minimum": 0, "default": 0, "description": "Render seed."},
                **_OUT,
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orb_variants",
        "description": (
            "Render N non-repeating takes of one Spec: same timing and structure, different "
            "texture and synthesis seeds. Use for game SFX pools and A/B choices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": SPEC_SCHEMA,
                "n": {"type": "integer", "minimum": 1, "maximum": MAX_VARIANTS, "default": 4},
                "base_seed": {"type": "integer", "minimum": 0, "default": 0},
                "vary_event_seeds": {
                    "type": "boolean",
                    "default": True,
                    "description": "Also change each event's seed per take, for bigger differences.",
                },
                **_OUT,
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orb_describe_grammar",
        "description": (
            "Return everything needed to author a Spec directly: the event roles (with their "
            "character and default duration/gain), textures, moods, limits, the caption grammar, "
            "the Spec JSON Schema and an example. Takes no arguments."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orb_validate_spec",
        "description": (
            "Check a Spec without rendering it. Returns valid, a list of errors (each with the "
            "path of the bad field), warnings, the normalised Spec (defaults filled, events "
            "sorted) and its canonical caption."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"spec": {"type": "object", "description": "The Spec to check."}},
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)


def anthropic_tools() -> list[dict[str, Any]]:
    """Tool definitions for the Anthropic Messages API (``tools=``)."""
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOLS]


def openai_tools() -> list[dict[str, Any]]:
    """Tool definitions for OpenAI-style function calling (``tools=``)."""
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in TOOLS
    ]


# -- validation --------------------------------------------------------------


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_spec(data: Any) -> dict[str, Any]:
    """Validate and normalise an agent-authored Spec.

    Returns ``{"valid", "errors", "warnings", "spec", "caption"}``. ``spec`` is
    the normalised dict (role defaults filled in for missing ``dur``/``gain_db``,
    ``pan`` 0, ``seed`` derived from the event index, events sorted by time) and
    is ``None`` when there are errors. Errors are strings prefixed with a path
    such as ``events[2].t``, so an agent can fix them one by one.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["spec: must be a JSON object"], "warnings": [], "spec": None, "caption": None}

    for k in sorted(set(data) - _SPEC_KEYS):
        errors.append(f"{k}: unknown field (allowed: {', '.join(sorted(_SPEC_KEYS))})")

    duration = data.get("duration")
    if not _num(duration):
        errors.append("duration: required number of seconds")
        duration = None
    elif not (MIN_DURATION <= duration <= MAX_DURATION):
        errors.append(f"duration: {duration} is outside [{MIN_DURATION}, {MAX_DURATION}] s")
        duration = None
    elif not (PLANNER_MIN_DURATION <= duration <= PLANNER_MAX_DURATION):
        warnings.append(
            f"duration: {duration} s is outside the planner's 2-30 s range; the DSP render is fine, "
            "but the caption will not round-trip through the planner"
        )

    texture = data.get("texture")
    if texture not in BED_CHOICES:
        errors.append(f"texture: {texture!r} is not one of {list(BED_CHOICES)}")
    bed_db = data.get("bed_db")
    if bed_db is not None:
        if not _num(bed_db) or not (BED_DB_RANGE[0] <= bed_db <= BED_DB_RANGE[1]):
            errors.append(f"bed_db: must be a number in [{BED_DB_RANGE[0]:g}, {BED_DB_RANGE[1]:g}]")
            bed_db = None
        elif texture == NO_BED:
            errors.append('bed_db: has no effect with texture "none"; drop one of them')
            bed_db = None
    mood = data.get("mood")
    if mood not in MOODS:
        errors.append(f"mood: {mood!r} is not one of {list(MOODS)}")

    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        errors.append("events: must be a list")
        raw_events = []
    if len(raw_events) > MAX_EVENTS:
        errors.append(f"events: {len(raw_events)} events, at most {MAX_EVENTS} allowed")

    events: list[dict[str, Any]] = []
    has_params = False
    for i, ev in enumerate(raw_events):
        p = f"events[{i}]"
        if not isinstance(ev, dict):
            errors.append(f"{p}: must be an object")
            continue
        for k in sorted(set(ev) - _EVENT_KEYS):
            errors.append(f"{p}.{k}: unknown field (allowed: {', '.join(sorted(_EVENT_KEYS))})")
        role = ev.get("role")
        if role not in EVENT_ROLES:
            errors.append(f"{p}.role: {role!r} is not one of {list(EVENT_ROLES)}")
            continue
        _, _, gain0, dur0 = EVENT_STYLE[role]

        t = ev.get("t")
        if not _num(t) or t < 0:
            errors.append(f"{p}.t: required number >= 0")
            continue
        dur = ev.get("dur", dur0)
        if not _num(dur) or dur <= 0:
            errors.append(f"{p}.dur: must be a number > 0")
            continue
        if duration is not None and t + dur > duration + 1e-6:
            if "dur" not in ev and t < duration:
                # The agent left dur to the default; fit it instead of failing.
                dur = round(duration - t, 2)
                warnings.append(f"{p}.dur: default {dur0} s trimmed to {dur} s to fit the cue")
            else:
                errors.append(
                    f"{p}: ends at {t + dur:.3f} s, after the cue ends at {duration} s "
                    f"(move t earlier, shorten dur, or lengthen duration)"
                )
                continue
        if dur <= 0.02:
            warnings.append(f"{p}.dur: {dur} s is too short to render; the event will be silent")
        gain = ev.get("gain_db", gain0)
        if not _num(gain) or not (MIN_GAIN_DB <= gain <= MAX_GAIN_DB):
            errors.append(f"{p}.gain_db: must be a number in [{MIN_GAIN_DB}, {MAX_GAIN_DB}]")
            continue
        pan = ev.get("pan", 0.0)
        if not _num(pan) or not (-1.0 <= pan <= 1.0):
            errors.append(f"{p}.pan: must be a number in [-1, 1]")
            continue
        seed = ev.get("seed", (17 * i + 1) % 256)
        if isinstance(seed, float) and seed.is_integer():
            seed = int(seed)
        if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed <= 255):
            errors.append(f"{p}.seed: must be an integer in [0, 255]")
            continue
        params = ev.get("params")
        param_errors = event_param_errors(role, params)
        if param_errors:
            errors.extend(f"{p}.{e}" for e in param_errors)
            continue
        event = {"role": role, "t": float(t), "dur": float(dur), "gain_db": float(gain), "pan": float(pan), "seed": seed}
        if params:
            event["params"] = {k: float(v) for k, v in params.items()}
            has_params = True
        events.append(event)

    if texture == NO_BED and not raw_events:
        errors.append('texture: "none" needs at least one event (the cue would be silent)')

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "spec": None, "caption": None}

    if bed_db is not None:
        warnings.append(
            "bed_db is render-only: the caption cannot express it, so it will not round-trip "
            "through the planner"
        )
    if has_params:
        warnings.append(
            "event params are render-only: the caption cannot express them, so they will not "
            "round-trip through the planner"
        )
    events.sort(key=lambda e: e["t"])
    spec = {"duration": float(duration), "texture": texture, "mood": mood, "events": events}
    if bed_db is not None:
        spec["bed_db"] = float(bed_db)
    return {"valid": True, "errors": [], "warnings": warnings, "spec": spec, "caption": spec_caption(spec)}


def spec_caption(spec: dict[str, Any]) -> str:
    """The canonical caption for a (valid, normalised) Spec, in the planner's grammar.

    Values are snapped to the caption grid (0.01 s, 2 dB, nine pan positions),
    so off-grid Specs get the nearest caption.
    """
    from .grammar import caption, score_from_spec

    s = Spec(
        spec["duration"],
        spec["texture"],
        spec["mood"],
        [SpecEvent(e["role"], e["t"], e["dur"], e["gain_db"], e["pan"], e["seed"]) for e in spec["events"]],
        spec.get("bed_db"),
    )
    return caption(score_from_spec(s, with_music=False))


def _require_valid(spec: Any) -> dict[str, Any]:
    report = validate_spec(spec)
    if not report["valid"]:
        raise OrbToolError("invalid spec:\n  " + "\n  ".join(report["errors"]))
    return report


# -- rendering helpers -------------------------------------------------------


def output_root() -> Path:
    """The directory every WAV is written under: ``$ORB_OUTPUT_DIR`` or ``./orb_out``."""
    return Path(os.environ.get("ORB_OUTPUT_DIR") or "orb_out").expanduser().resolve()


def _output_dir(arg: str | None) -> Path:
    """Resolve a per-call ``output_dir`` and confine it to :func:`output_root`.

    The argument comes from the model, so it may only pick a sub-directory of
    the configured root. A relative path is joined onto the root. An absolute
    path is accepted only if it already lies inside the root. ``..`` and
    symlinks are resolved before the check.
    """
    root = output_root()
    if arg is None or arg == "":
        d = root
    else:
        if not isinstance(arg, str):
            raise OrbToolError("output_dir: must be a string")
        p = Path(arg).expanduser()
        d = (p if p.is_absolute() else root / p).resolve()
        if d != root and root not in d.parents:
            raise OrbToolError(
                f"output_dir: {arg!r} is outside the output root {str(root)!r}; pass a path "
                "relative to it (the root is set by --output-dir or $ORB_OUTPUT_DIR)"
            )
    d.mkdir(parents=True, exist_ok=True)
    # Re-check after creation in case a component was a symlink created meanwhile.
    real = d.resolve()
    if real != root and root not in real.parents:
        raise OrbToolError(f"output_dir: {arg!r} resolves outside the output root")
    return real


def _file_name(name: str | None, spec: dict[str, Any], seed: int, suffix: str = "", detail: bool = False) -> str:
    """Base file name. Hash-based names cover the Spec, the seed and the detail flag."""
    if name:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(name)).name).strip("._") or "cue"
        if base.lower().endswith(".wav"):
            base = base[:-4]
    else:
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
        base = f"orb_{digest}_s{seed}"
    if detail:
        base += "_detail"
    return f"{base}{suffix}.wav"


def _render_to_file(spec: dict[str, Any], seed: int, path: Path, detail=None) -> dict[str, Any]:
    from .api import render_spec, save_wav
    from .eval.audit import audit_waveform, passes

    audio = render_spec(spec, detail=detail, seed=int(seed))
    save_wav(audio, path)
    audit = audit_waveform(audio, SAMPLE_RATE)
    audit = {k: round(float(v), 6) for k, v in audit.items()}
    audit["passes"] = bool(passes(audit))
    return {"path": str(path), "duration_s": round(audio.shape[0] / SAMPLE_RATE, 4), "audit": audit}


def _seed_arg(args: dict[str, Any], key: str = "seed") -> int:
    seed = args.get(key, 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise OrbToolError(f"{key}: must be an integer >= 0")
    return seed


# -- lazily loaded models ----------------------------------------------------

_MODELS: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _planner():
    with _MODEL_LOCK:
        if "planner" not in _MODELS:
            try:
                import torch  # noqa: F401
            except ImportError as e:
                raise OrbToolError(
                    "caption mode needs torch (pip install 'orb-audio[train]'). Or pass a `spec` instead; "
                    "the spec path is numpy-only."
                ) from e
            from .api import load_planner, resolve_planner_path

            # Same lookup as orb.api.generate: $ORB_PLANNER, ./weights, the
            # checkout, planner.safetensors, then the Hub (if huggingface_hub
            # is installed).
            try:
                path = resolve_planner_path()
            except FileNotFoundError as e:
                raise OrbToolError(
                    f"caption mode needs the planner weights: {e} Or pass a `spec` instead; it needs no model."
                ) from e
            _MODELS["planner"] = load_planner(path)
        return _MODELS["planner"]


def _detail():
    with _MODEL_LOCK:
        if "detail" not in _MODELS:
            from .api import load_detail, resolve_detail_path

            try:
                path = resolve_detail_path()
            except FileNotFoundError as e:
                raise OrbToolError(
                    "detail=true needs the CC BY-NC detail weights (set ORB_DETAIL); they are a "
                    f"separate, research-only download. {e}"
                ) from e
            _MODELS["detail"] = load_detail(path)
        return _MODELS["detail"]


# -- the tools ---------------------------------------------------------------


def _tool_generate(args: dict[str, Any]) -> dict[str, Any]:
    caption, spec_in = args.get("caption"), args.get("spec")
    if (caption is None) == (spec_in is None):
        raise OrbToolError("pass exactly one of `caption` or `spec`")
    seed = _seed_arg(args)
    warnings: list[str] = []
    if caption is not None:
        if not isinstance(caption, str) or not caption.strip():
            raise OrbToolError("caption: must be a non-empty string")
        from .api import plan, spec_to_dict
        from .grammar import parse_caption

        try:
            direct = parse_caption(caption)
        except ValueError:  # grammar-shaped but not a valid Spec: the planner reads it
            direct = None
        if direct is not None and direct.texture == NO_BED:
            # The planner never saw a bed-less caption and would add a bed.
            spec_in = spec_to_dict(direct)
            source = "caption"
        else:
            tokenizer, model = _planner()
            spec_in = spec_to_dict(plan(caption, tokenizer, model))
            source = "planner"
    else:
        source = "spec"
    try:
        report = _require_valid(spec_in)
    except OrbToolError as e:
        if source != "planner":
            raise
        raise OrbToolError(
            f"{e}\nThe planner only reads the caption grammar (see orb_describe_grammar); free text "
            "is out of distribution. Write the caption in the grammar, or pass a `spec`."
        ) from e
    warnings += report["warnings"]
    spec = report["spec"]
    detail = _detail() if args.get("detail") else None
    path = _output_dir(args.get("output_dir")) / _file_name(args.get("name"), spec, seed, detail=detail is not None)
    out = _render_to_file(spec, seed, path, detail=detail)
    return {
        **out,
        "sample_rate": SAMPLE_RATE,
        "source": source,
        "spec": spec,
        "caption": report["caption"],
        "seed": seed,
        "detail": detail is not None,
        "warnings": warnings,
    }


def _tool_render_spec(args: dict[str, Any]) -> dict[str, Any]:
    report = _require_valid(args.get("spec"))
    seed = _seed_arg(args)
    spec = report["spec"]
    path = _output_dir(args.get("output_dir")) / _file_name(args.get("name"), spec, seed)
    out = _render_to_file(spec, seed, path)
    return {
        **out,
        "sample_rate": SAMPLE_RATE,
        "spec": spec,
        "caption": report["caption"],
        "seed": seed,
        "warnings": report["warnings"],
    }


def _tool_variants(args: dict[str, Any]) -> dict[str, Any]:
    report = _require_valid(args.get("spec"))
    spec = report["spec"]
    n = args.get("n", 4)
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= MAX_VARIANTS):
        raise OrbToolError(f"n: must be an integer in [1, {MAX_VARIANTS}]")
    base = _seed_arg(args, "base_seed")
    vary = bool(args.get("vary_event_seeds", True))
    out_dir = _output_dir(args.get("output_dir"))
    stem = _file_name(args.get("name"), spec, base)[:-4]
    takes = []
    for i in range(n):
        seed = base + i
        take = json.loads(json.dumps(spec))
        if vary:
            for j, ev in enumerate(take["events"]):
                ev["seed"] = (ev["seed"] + 37 * i + 11 * j * (i > 0)) % 256
        r = _render_to_file(take, seed, out_dir / f"{stem}_v{i:02d}.wav")
        takes.append({**r, "seed": seed, "event_seeds": [e["seed"] for e in take["events"]]})
    return {"n": n, "spec": spec, "variants": takes, "sample_rate": SAMPLE_RATE, "warnings": report["warnings"]}


EXAMPLE_SPEC: dict[str, Any] = {
    "duration": 6.0,
    "texture": "sci_fi",
    "mood": "bright",
    "events": [
        {"role": "riser", "t": 0.5, "dur": 1.5, "gain_db": -18.0, "pan": 0.0, "seed": 12},
        {"role": "whoosh", "t": 1.8, "dur": 0.8, "gain_db": -8.0, "pan": -0.5, "seed": 40},
        {"role": "impact", "t": 2.0, "dur": 0.7, "gain_db": -4.0, "pan": 0.0, "seed": 7},
        {"role": "bell", "t": 2.0, "dur": 1.4, "gain_db": -14.0, "pan": 0.25, "seed": 99},
    ],
}


def _describe_params(role: str) -> dict[str, Any]:
    """The params a role takes: range, what each does, and the default.

    A default of None means the role has no single default for it: the seed
    picks the value (a church bell's note), which is not the same as zero.
    """
    return {
        name: {"range": [lo, hi], "default": _ROLE_PARAMS[role].get(name), "description": doc}
        for name, (lo, hi, doc) in EVENT_PARAMS[role].items()
    }


def describe_grammar() -> dict[str, Any]:
    """Everything an agent needs to author a Spec without the planner."""
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "roles": {
            role: {
                "character": EVENT_STYLE[role][0],
                "apex_fraction": EVENT_STYLE[role][1],
                "default_gain_db": EVENT_STYLE[role][2],
                "default_dur": EVENT_STYLE[role][3],
                **({"seed_variants": FOLEY_VARIANTS[role]} if role in FOLEY_VARIANTS else {}),
                **({"params": _describe_params(role)} if role in EVENT_PARAMS else {}),
            }
            for role in EVENT_ROLES
        },
        "textures": list(TEXTURES),
        "bed": {
            "none": 'texture "none": no bed, digital silence between events (UI sounds, layering)',
            "bed_db": {"default": DEFAULT_BED_DB, "range": list(BED_DB_RANGE),
                       "scene_bed_default": AMBIENCE_BED_DB},
            "scene_beds": {t: "a scene built from physical models (not filtered noise), "
                              "placed close to the events" for t in AMBIENCE_TEXTURES},
        },
        "moods": list(MOODS),
        "limits": {
            "max_events": MAX_EVENTS,
            "duration_s": [MIN_DURATION, MAX_DURATION],
            "planner_duration_s": [PLANNER_MIN_DURATION, PLANNER_MAX_DURATION],
            "gain_db": [MIN_GAIN_DB, MAX_GAIN_DB],
            "pan": [-1.0, 1.0],
            "seed": [0, 255],
            "max_variants": MAX_VARIANTS,
        },
        "notes": [
            "Timing is exact: an event at t=1.25 starts at sample 60000.",
            "apex_fraction is where in the event the peak lands: an impact peaks at t, a riser near t + 0.92*dur. "
            "To land a riser's peak on a hit at time T, set t = T - 0.92*dur.",
            "The cue is loudness-normalised as a whole; gain_db sets the balance between events.",
            f"The texture bed runs for the whole cue, bed_db (default {DEFAULT_BED_DB:g}) below the events. Set "
            'texture to "none" for no bed (silence between events), or lower bed_db for a quieter one.',
            "Same Spec + same seed = bit-identical audio.",
            "A few roles take per-event params (roles.<role>.params): set params.f0 to put a bell, church bell, "
            "pop, click, impact or sub drop on a pitch, so a cue can follow the key of the music it sits over. "
            "Params are Spec-only: a caption cannot express them.",
        ],
        "caption_grammar": (
            "{duration:.2f}-second {mood} cue, {texture} bed[, with {role} @{t:.2f} {dur:.2f}s "
            "{gain:+.0f}dB {pan} #{seed:02x}, ...]  (role/texture underscores become spaces; "
            "gain even dB in [-30, 0]; duration in 0.5 s steps). '{texture} bed' may be 'no bed': "
            "such a caption is read exactly, without the planner."
        ),
        "pan_labels": {label: round(i / (len(PAN_LABELS) - 1) * 2 - 1, 2) for i, label in enumerate(PAN_LABELS)},
        "spec_schema": SPEC_SCHEMA,
        "example_spec": EXAMPLE_SPEC,
        "example_caption": spec_caption(EXAMPLE_SPEC),
    }


def _tool_describe(args: dict[str, Any]) -> dict[str, Any]:
    return describe_grammar()


def _tool_validate(args: dict[str, Any]) -> dict[str, Any]:
    if "spec" not in args:
        raise OrbToolError("spec: required")
    return validate_spec(args["spec"])


_HANDLERS = {
    "orb_generate": _tool_generate,
    "orb_render_spec": _tool_render_spec,
    "orb_variants": _tool_variants,
    "orb_describe_grammar": _tool_describe,
    "orb_validate_spec": _tool_validate,
}


def dispatch(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one tool call. Returns a JSON-serialisable dict.

    Raises :class:`OrbToolError` for input the agent can fix, and ``KeyError``
    for an unknown tool name.
    """
    if name not in _HANDLERS:
        raise KeyError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")
    args = dict(arguments or {})
    allowed = set(next(t for t in TOOLS if t["name"] == name)["input_schema"]["properties"])
    extra = sorted(set(args) - allowed)
    if extra:
        raise OrbToolError(f"unknown argument(s) {extra} for {name}; allowed: {sorted(allowed)}")
    return _HANDLERS[name](args)


def run_tool(name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Run a tool call for an agent loop: returns ``(json_text, is_error)`` and never raises."""
    try:
        return json.dumps(dispatch(name, arguments)), False
    except (OrbToolError, KeyError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        return json.dumps({"error": msg}), True
    except Exception as e:  # pragma: no cover - surfaced to the agent, not swallowed silently
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True

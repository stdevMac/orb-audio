"""Public API for ORB.

Two entry points, one for each path through the engine.

Model-free DSP (needs only numpy)::

    from orb.api import render_spec
    audio = render_spec({
        "duration": 6.0, "texture": "wind", "mood": "bright",   # or "none"
        "events": [{"role": "whoosh", "t": 1.0, "dur": 0.9,
                    "gain_db": -6.0, "pan": -0.5, "seed": 42}],
    })

Full pipeline (needs torch and the planner weights)::

    from orb.api import generate
    out = generate("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a")
    out["audio"]   # (n, 2) float32, 48 kHz
    out["spec"]    # the planned Spec as a dict

Every function is deterministic for a given input and seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .eval.audit import audit_waveform, passes
from .grammar import NO_BED, Spec, SpecEvent, parse_caption, score_from_spec
from .render.mix import render as _render_score

SAMPLE_RATE = 48000
# ``None`` means "find the weights": see :func:`resolve_planner_path`.
DEFAULT_PLANNER: str | None = None
DEFAULT_DETAIL: str | None = None
PLANNER_HUB_REPO = "marcosmaceo/orb-planner"


# -- spec <-> dict ---------------------------------------------------------


def spec_to_dict(spec: Spec) -> dict[str, Any]:
    """A JSON-serialisable view of a Spec.

    ``bed_db`` and an event's ``params`` are included only when they are set,
    so a planner Spec (which never sets either) serialises exactly as before.
    """
    out = {
        "duration": float(spec.duration),
        "texture": spec.texture,
        "mood": spec.mood,
        "events": [
            {
                "role": e.role,
                "t": float(e.t),
                "dur": float(e.dur),
                "gain_db": float(e.gain_db),
                "pan": float(e.pan),
                "seed": int(e.seed),
                **({"params": dict(e.params)} if e.params else {}),
            }
            for e in spec.events
        ],
    }
    if spec.bed_db is not None:
        out["bed_db"] = float(spec.bed_db)
    return out


def dict_to_spec(data: dict[str, Any]) -> Spec:
    """Inverse of :func:`spec_to_dict`.

    Besides the planner's fields, a hand-written Spec may set ``texture`` to
    ``"none"`` (no bed: digital silence between events) and ``bed_db``, the
    bed level relative to the foreground in [-60, -6] dB (default -40). An
    event may carry ``params``, renderer parameters for that one event (see
    ``orb.grammar.EVENT_PARAMS``): ``{"f0": 440.0}`` puts a bell on a note.
    """
    bed_db = data.get("bed_db")
    if bed_db is not None and (isinstance(bed_db, bool) or not isinstance(bed_db, (int, float))):
        raise ValueError("bed_db must be a number")
    events = [
        SpecEvent(
            role=e["role"],
            t=float(e["t"]),
            dur=float(e["dur"]),
            gain_db=float(e.get("gain_db", -12.0)),
            pan=float(e.get("pan", 0.0)),
            seed=int(e.get("seed", 0)),
            params=e.get("params"),
        )
        for e in data.get("events", [])
    ]
    return Spec(
        duration=float(data["duration"]),
        texture=str(data["texture"]),
        mood=str(data["mood"]),
        events=events,
        bed_db=None if bed_db is None else float(bed_db),
    )


# -- models ----------------------------------------------------------------

# The repository root when running from a checkout (src/orb/api.py -> ../..).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _weight_dirs() -> list[Path]:
    """Local folders searched for weights: ./weights, then the checkout's weights/."""
    dirs = [Path.cwd() / "weights", _REPO_ROOT / "weights"]
    out: list[Path] = []
    for d in dirs:
        if d.resolve() not in [o.resolve() for o in out]:
            out.append(d)
    return out


def _in_dir(d: Path, name: str) -> Path | None:
    """``<name>.pt``, else ``<name>.safetensors`` with its ``config.json``, in ``d``."""
    pt = d / f"{name}.pt"
    if pt.is_file():
        return pt
    st = d / f"{name}.safetensors"
    if st.is_file() and (d / "config.json").is_file():
        return st
    return None


def _explicit(path: str | Path, name: str, source: str) -> Path:
    p = Path(path).expanduser()
    if p.is_dir():
        found = _in_dir(p, name)
        if found is None:
            raise FileNotFoundError(
                f"{source} is a folder without {name}.pt or {name}.safetensors + config.json: {p}"
            )
        return found
    if not p.is_file():
        raise FileNotFoundError(f"{source} points at a file that does not exist: {p}")
    return p


def _hub_snapshot(repo_id: str, name: str) -> Path:
    """Download ``<name>.safetensors`` + ``config.json`` from the Hub (cached)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=repo_id, allow_patterns=[f"{name}.safetensors", "config.json"])
    )


def resolve_planner_path(path: str | Path | None = None) -> Path:
    """Find the planner weights.

    Order: ``path`` -> ``$ORB_PLANNER`` -> ``./weights/planner.pt`` -> the
    checkout's ``weights/planner.pt`` -> ``planner.safetensors`` + ``config.json``
    in those folders -> a Hugging Face Hub download of ``marcosmaceo/orb-planner``
    (only if ``huggingface_hub`` is installed). An explicit path or env var that
    does not exist is an error, never silently skipped.
    """
    import os

    if path is not None:
        return _explicit(path, "planner", "planner path")
    env = os.environ.get("ORB_PLANNER")
    if env:
        return _explicit(env, "planner", "$ORB_PLANNER")
    dirs = _weight_dirs()
    for d in dirs:
        if (d / "planner.pt").is_file():
            return d / "planner.pt"
    for d in dirs:
        found = _in_dir(d, "planner")
        if found is not None:
            return found
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        hub_error = "huggingface_hub is not installed"
    else:
        try:
            found = _in_dir(_hub_snapshot(PLANNER_HUB_REPO, "planner"), "planner")
            if found is not None:
                return found
            hub_error = f"{PLANNER_HUB_REPO} has no planner.safetensors + config.json"
        except Exception as e:  # network, auth, offline mode
            hub_error = f"download from {PLANNER_HUB_REPO} failed: {e}"
    raise FileNotFoundError(
        "ORB planner weights not found (searched $ORB_PLANNER, "
        + ", ".join(str(d) for d in dirs)
        + f"; {hub_error}). Fix: pip install 'orb-audio[hub]' to download them from "
        f"https://huggingface.co/{PLANNER_HUB_REPO}, or pass planner_path=/ set ORB_PLANNER "
        "to a planner.pt / planner.safetensors. Rendering a Spec dict with render_spec() "
        "needs no weights."
    )


def resolve_detail_path(path: str | Path | None = None) -> Path:
    """Find the optional detail weights (CC BY-NC-4.0, never auto-downloaded).

    Order: ``path`` -> ``$ORB_DETAIL`` -> ``./weights/detail.pt`` -> the
    checkout's ``weights/detail.pt`` -> ``detail.safetensors`` + ``config.json``
    in those folders.
    """
    import os

    if path is not None:
        return _explicit(path, "detail", "detail path")
    env = os.environ.get("ORB_DETAIL")
    if env:
        return _explicit(env, "detail", "$ORB_DETAIL")
    dirs = _weight_dirs()
    for d in dirs:
        if (d / "detail.pt").is_file():
            return d / "detail.pt"
    for d in dirs:
        found = _in_dir(d, "detail")
        if found is not None:
            return found
    raise FileNotFoundError(
        "ORB detail weights not found (searched $ORB_DETAIL, "
        + ", ".join(str(d) for d in dirs)
        + "). They are optional, CC BY-NC-4.0 (non-commercial) and not downloaded "
        "automatically: get them from https://huggingface.co/marcosmaceo/orb-detail and "
        "pass detail_path= or set ORB_DETAIL. Everything works without them."
    )


def load_planner(path: str | Path | None = DEFAULT_PLANNER, device: str = "cpu"):
    """Load the planner from a ``.pt`` checkpoint or a ``.safetensors`` file.

    ``path=None`` finds the weights with :func:`resolve_planner_path`.
    Returns ``(tokenizer, model)``.
    """
    import torch

    from .model.planner import Planner
    from .model.spec_tokenizer import SpecTokenizer

    path = resolve_planner_path(path)
    tokenizer = SpecTokenizer()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        cfg = json.loads((path.parent / "config.json").read_text())
        model = Planner(cfg.get("vocab_size", tokenizer.vocab_size), **cfg["model"])
        model.load_state_dict(load_file(str(path), device="cpu"))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # The checkpoint's own vocabulary: older planners (v0.3: 394 tokens)
        # load next to a newer tokenizer and simply never emit the new ids.
        model = Planner(int(ckpt["model"]["tok_emb.weight"].shape[0]), **ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return tokenizer, model


def load_detail(path: str | Path | None = DEFAULT_DETAIL, device: str = "cpu"):
    """Load the optional detail model from ``.pt`` or ``.safetensors``.

    The detail weights are CC BY-NC-4.0 and are not part of the open-source
    release; see ``weights/README.md``. ``path=None`` searches locally with
    :func:`resolve_detail_path`; they are never downloaded automatically.
    """
    import torch

    from .model.detail import DetailUNet

    path = resolve_detail_path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        cfg = json.loads((path.parent / "config.json").read_text())
        model = DetailUNet(**cfg["model"])
        model.load_state_dict(load_file(str(path), device="cpu"))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = DetailUNet(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["generator"])
    model.to(device).eval()
    return model


# -- generation ------------------------------------------------------------


def plan(caption: str, tokenizer, planner, device: str = "cpu") -> Spec:
    """Caption -> Spec with constrained decoding, so the result is always valid."""
    import torch

    from .model.spec_tokenizer import NOBED, SpecGrammar

    prompt = tokenizer.prompt(caption)
    out = planner.generate(
        torch.tensor([prompt], device=device),
        max_new_tokens=1024,
        temperature=1e-6,
        top_p=1.0,
        eos_id=tokenizer.eos_id,
        grammar=SpecGrammar(no_bed=planner.tok_emb.num_embeddings > NOBED),
    )
    return tokenizer.decode_spec(out[0, len(prompt) :].tolist())


def render_spec(
    spec: Spec | dict[str, Any],
    detail=None,
    strength: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Spec -> stereo float32 audio at 48 kHz.

    ``detail`` is an optional model from :func:`load_detail`. Without it this
    is pure numpy and needs no torch.
    """
    if isinstance(spec, dict):
        spec = dict_to_spec(spec)
    score = score_from_spec(spec, with_music=False)
    score.seed = int(seed)
    score.validate()
    if detail is None:
        return np.asarray(_render_score(score), dtype=np.float32)
    # The physical-model roles and scene beds bypass the detail stage (see
    # `render.mix.apply_detail_routed`); everything else gets it as before.
    from .render.mix import apply_detail_routed, render_stems

    audio = apply_detail_routed(render_stems(score), detail, seed=int(seed), strength=float(strength))
    return np.asarray(audio, dtype=np.float32)


def generate(
    caption: str,
    planner_path: str | Path | None = DEFAULT_PLANNER,
    detail_path: str | Path | None = None,
    device: str = "cpu",
    strength: float = 1.0,
    seed: int = 0,
    planner=None,
    detail=None,
) -> dict[str, Any]:
    """Caption -> audio, spec, and audit metrics in one call.

    Pass ``planner``/``detail`` to reuse already-loaded models, or leave them
    out and the weights are loaded: ``planner_path=None`` finds the planner
    (see :func:`resolve_planner_path`, which may download it from the Hub).
    The detail stage runs only when ``detail_path`` or ``detail`` is given; a
    ``detail_path`` that does not exist raises instead of being skipped.

    A grammar caption with no bed (``"..., no bed, with ui pop @0.60 ..."``)
    is rendered from :func:`orb.grammar.parse_caption` without the planner,
    which never saw bed-less captions and would add a noise bed to them.
    ``out["planned"]`` says which path ran.
    """
    try:
        direct = parse_caption(caption)
    except ValueError:  # grammar-shaped but not a valid Spec: let the planner read it
        direct = None
    planned = direct is None or direct.texture != NO_BED
    if detail is None and detail_path is not None:
        detail = load_detail(detail_path, device)
    if planned:
        if planner is None:
            tokenizer, model = load_planner(planner_path, device)
        else:
            tokenizer, model = planner
        spec = plan(caption, tokenizer, model, device)
    else:
        spec = direct
    audio = render_spec(spec, detail=detail, strength=strength, seed=seed)
    audit = audit_waveform(audio, SAMPLE_RATE)
    audit["passes"] = passes(audit)
    return {
        "audio": audio,
        "sample_rate": SAMPLE_RATE,
        "spec": spec_to_dict(spec),
        "audit": audit,
        "planned": planned,
    }


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float array to a 16-bit PCM WAV.

    Samples are rounded to the nearest step, not truncated toward zero
    (truncation adds a signal-correlated error on tails that decay through the
    last bits). No dither is added: exact digital silence stays exactly zero.
    """
    import wave

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())

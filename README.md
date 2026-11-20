<div align="center">

# ORB

**Sound as code. A from-scratch audio engine for agents: a 5.1M-parameter planner and a NumPy DSP renderer.**

Caption in, 48 kHz stereo out: exact to the frame, editable as JSON, no samples, no pretrained weights.

[![tests](https://github.com/stdevMac/orb-audio/actions/workflows/tests.yml/badge.svg)](https://github.com/stdevMac/orb-audio/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Demo page](https://img.shields.io/badge/demo-GitHub%20Pages-222?logo=github)](https://stdevmac.github.io/orb-audio/)
<br>
[![HF model](https://img.shields.io/badge/%F0%9F%A4%97%20model-orb--planner-ffcc4d)](https://huggingface.co/marcosmaceo/orb-planner)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20space-try%20it%20live-ff9d00)](https://huggingface.co/spaces/marcosmaceo/orb)
[![HF dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-orb--planner--data-ffcc4d)](https://huggingface.co/datasets/marcosmaceo/orb-planner-data)

**[Watch the demo](https://stdevmac.github.io/orb-audio/)** ·
**[Try it live](https://huggingface.co/spaces/marcosmaceo/orb)** ·
**[Use it from an agent](#for-agents)** ·
**[How it works](docs/architecture.md)**

<a href="https://stdevmac.github.io/orb-audio/">
  <img src="site/assets/orb-demo-poster.jpg" alt="ORB demo video: captions and Spec edits turning into sound, cut to motion design. Click to watch on the project page." width="100%">
</a>

<sub>Click the image to watch the demo on the project page, or download <a href="site/assets/orb-demo.mp4">the MP4</a>.</sub>

</div>

```
caption ──► planner (5.1M) ──► Spec (JSON) ──► DSP render ──► [detail (9.1M)] ──► WAV + audit
```

- the open engine is a **5.1M-parameter planner (20 MB)** plus NumPy DSP, 48 kHz stereo;
  the bracketed detail model is optional, separate and non-commercial (see below)
- planner: **100% exact** on the held-out grammar captions, 0 decode failures; since v8 it also reads plain words (58% of our free-text prompts and 70% of a blind set planned fully right)
- the DSP path needs **only NumPy**: no torch, no weights, no GPU
- timing is a property of the score, so it is **exact**
- every output is an editable **`Spec`**, not a blob
- built for **agents**: ships an MCP server, `orb-mcp`
- **Apache-2.0** code and planner weights

## The idea

A sample is a **file**. This is a **function**.

That one change is the whole pitch. Because the sound is generated from a typed
score, you can move an impact from 5.00 s to 5.40 s by editing a number, ask
for 500 non-repeating variants from one spec, diff a sound design in git, and
ship the whole engine in about 20 MB of weights with no sample clearance.

It does not try to beat a recorded sound library on timbre. It aims to be
the programmable layer underneath one: the transitions, UI ticks, ambience beds
and variations that make up most of a project's audio.

![The ORB web demo: a caption in, a waveform and an editable Spec out](docs/demo.png)

## Quickstart

```bash
git clone https://github.com/stdevMac/orb-audio
cd orb-audio
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# numpy only, no weights, no torch
python examples/04_no_torch.py

# the full pipeline (needs torch)
pip install -e ".[train]"
python examples/01_generate.py
```

`generate()` finds the planner on its own: an explicit `planner_path`, then `$ORB_PLANNER`,
then `./weights/planner.pt`, then the clone's `weights/` folder (`planner.pt`, or
`planner.safetensors` + `config.json`), and finally a download of
[marcosmaceo/orb-planner](https://huggingface.co/marcosmaceo/orb-planner) from the Hugging
Face Hub if `huggingface_hub` is installed. If none of those work it raises an error that
says what to install or pass.

```python
from orb.api import generate, save_wav

out = generate("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a")
out["audio"]   # (n, 2) float32 at 48 kHz
out["spec"]    # {'duration': 6.0, 'texture': 'wind', 'mood': 'bright', 'events': [...]}
out["audit"]   # peak, rms_dbfs, dc, click_score, passes, ...
save_wav(out["audio"], "whoosh.wav")
```

> The planner weights in `weights/` are stored with Git LFS. If `weights/planner.pt`
> is a tiny text file (an LFS pointer), run `git lfs install && git lfs pull`.

**Installed with pip (no clone)?** Install from GitHub with the `hub` extra (the package
name is `orb-audio`; the import name is `orb`) and the same `generate(caption)` call
downloads `planner.safetensors` + `config.json` from the Hub on first use and caches them:

```bash
pip install "orb-audio[train,hub] @ git+https://github.com/stdevMac/orb-audio"
```

Offline, download the two files yourself and pass `planner_path=` (a file or the folder
holding them) or set `ORB_PLANNER`.

No torch? Skip the planner and render a Spec directly:

```python
from orb.api import render_spec, save_wav

audio = render_spec({
    "duration": 6.0, "texture": "wind", "mood": "bright",
    "events": [{"role": "whoosh", "t": 1.0, "dur": 0.9, "gain_db": -6.0, "pan": -0.5, "seed": 42}],
})
save_wav(audio, "whoosh.wav")
```

From the shell:

```bash
orb-generate "8.00-second tense cue, drone bed, with riser @1.00 2.40s -16dB C #51" --out riser.wav
```

The project page with a local, server-side engine (the same page as the
[Hugging Face Space](https://huggingface.co/spaces/marcosmaceo/orb): generate, download,
shape scores by hand):

```bash
python web/server.py --port 8765       # http://127.0.0.1:8765
```

## For agents

Most generative audio hands an agent a WAV file it cannot inspect or edit. ORB hands it
**a JSON document it can reason about**: every event has a role, a start time, a
duration, a gain, a pan and a seed. An agent can plan a cue, render it, read the audit,
nudge one number and re-render. The result is deterministic, so the same Spec always
gives the same audio.

ORB ships a [Model Context Protocol](https://modelcontextprotocol.io) server, so any
MCP client (Claude Code, Claude Desktop, Cursor and others) can drive it as a set of tools:

```bash
pip install "orb-audio[mcp,train] @ git+https://github.com/stdevMac/orb-audio"
orb-mcp            # stdio MCP server; see docs/agents.md for client config
```

Spec mode needs only numpy. Caption mode needs the planner: from a clone it reads
`weights/planner.pt`; with a pip install, download
[`planner.safetensors` + `config.json`](https://huggingface.co/marcosmaceo/orb-planner) and point
the server at them with `orb-mcp --planner /path/to/planner.safetensors` (or `$ORB_PLANNER`).
Without `--output-dir`, rendered WAVs land in `./orb_out/`.

The server lives in [`src/orb/mcp_server.py`](src/orb/mcp_server.py). Setup, tool reference and
example agent workflows are in [`docs/agents.md`](docs/agents.md).

Why this matters for agents and motion design:

- **Sync by construction.** Events land at exact timestamps, so a cue can be generated *from*
  an animation's keyframes rather than nudged to fit afterwards.
- **Edit, don't regenerate.** "Make the impact later and quieter" is a two-field diff, not a new sample.
- **Variants for free.** Change the seeds and you get non-repeating takes of the same design.
- **Checkable output.** Every render returns an audit (peak, DC, clicks, silence before the first event).

## Examples

| File | Needs torch? | Shows |
|---|---|---|
| [`examples/01_generate.py`](examples/01_generate.py) | yes | caption → WAV, full pipeline |
| [`examples/02_edit_spec.py`](examples/02_edit_spec.py) | no | change one number, re-render |
| [`examples/03_variants.py`](examples/03_variants.py) | no | N non-repeating takes from one Spec |
| [`examples/04_no_torch.py`](examples/04_no_torch.py) | no | the shippable DSP engine |
| [`examples/05_agent_tool.py`](examples/05_agent_tool.py) | no | the agent tool layer, called directly |
| [`examples/06_tuned_bell.py`](examples/06_tuned_bell.py) | no | per-event `params`: church bells on a D minor triad |

A 12-second trailer cue:

```
12.00-second intense cue, drone bed, with riser @1.00 2.50s -18dB C #55, whoosh @4.20 1.20s -10dB L3 #77, impact @5.00 0.90s -4dB C #66, sub drop @5.00 1.60s -8dB C #88, noise sweep @6.50 2.00s -16dB R2 #97, impact @9.50 0.90s -4dB C #99, bell @9.50 2.00s -12dB R1 #3d
```

## The Spec

The planner emits a small typed object, and the renderer is a pure function of it.

```json
{
  "duration": 12.0,
  "texture": "drone",
  "mood": "intense",
  "events": [
    {"role": "riser",    "t": 1.0, "dur": 2.5, "gain_db": -18.0, "pan": 0.0,   "seed": 85},
    {"role": "impact",   "t": 5.0, "dur": 0.9, "gain_db": -4.0,  "pan": 0.0,   "seed": 102},
    {"role": "sub_drop", "t": 5.0, "dur": 1.6, "gain_db": -8.0,  "pan": 0.0,   "seed": 136}
  ]
}
```

- **47 roles**: whoosh, impact, ui_pop, riser, swell, click, bell, sub_drop, glitch, noise_sweep, plus 37 physically modelled natural, foley and everyday roles:
  - foley: thunder, door_slam, knock, footsteps, water_drop, explosion
  - nature: bird_chirp, crickets, wave_crash, wind_gust, splash, frog, owl
  - home and office: glass_break, clock_tick, clock_alarm, phone_ring, keyboard_typing, mouse_click, camera_shutter, paper_rustle, cork_pop, coin_drop, pouring_water, can_opening, wood_creak, metal_clang
  - people and action: heartbeat, applause, punch, sword_clash, gunshot, fireworks
  - street: car_horn, siren, engine, church_bell

  For these the seed picks the variant (thunder `seed % 4` is distance, footsteps `seed % 3` the floor, siren `seed % 3` wail/yelp/hi-lo, phone_ring `seed % 4` bell/UK/ringtone/vibrate, …; the full list is `orb.render.natural.VARIANTS` and `orb.render.foley`).
- **13 textures**: room_cold, wind, city_hum, vinyl, drone, rain, fire, sci_fi (filtered-noise beds) and ocean, stream, forest, night, traffic (scene beds built from the physical models: waves breaking, a brook over stones, leaves and birds, crickets with a frog or an owl, cars passing). Scene beds sit at −14 dB under the events by default; the noise beds at −40 dB.
- **10 moods**: calm, warm, playful, hopeful, bright, tense, dark, eerie, neutral, intense
- up to **8 events**, duration 2–30 s
- the texture bed sits at −40 dB relative to the events by default. In a Spec dict, set
  `"bed_db"` to move it, or `"texture": "none"` for no bed at all (clean UI sounds). In a
  caption, write `no bed` in place of the bed clause; such a caption is rendered exactly,
  without the planner.
- a few roles take per-event **`params`** in a Spec dict, so a sound can sit on a note:
  `{"role": "church_bell", "t": 1.0, "dur": 6.0, "params": {"f0": 293.66}}` puts the bell's
  strike note on D4. `bell`, `ui_pop`, `click`, `impact` and `sub_drop` take `f0` as well;
  the church bell also takes `ring` (how long it rings), `hardness` (the clapper), `tierce`
  (its minor or major third) and `beat`. Left out, every sound is exactly what it was. Like
  `bed_db`, params are Spec-only: a caption cannot express them. `orb_describe_grammar`
  lists each role's params with their ranges and defaults.

The caption grammar is documented in [`examples/README.md`](examples/README.md).

## Results

**Planner** (v8): 5.10M parameters, 6 layers, d=256, byte-level tokenizer (395 tokens), constrained decoding. It reads both the caption grammar and plain words.

| Metric | Value |
|---|---|
| Held-out exact match (500) | **500/500 (100%)** |
| …captions using an everyday role (v7) | 390/390 (100%) |
| Larger held-out set (2,000) | 2000/2000 (100%) |
| Previous held-out set (v6, 500) | 500/500 (100%) (v6 planner: 98.8%) |
| Plain-text prompts, our set (209) | 58.4% fully right (v7 + keyword helper: 45.0%) |
| Plain-text prompts, blind set (40) | 70.0% fully right (v7 + keyword helper: 77.5%) |
| Per role / per scene bed | 100% for all 47 roles and 13 textures |
| Decode failures | 0 |

The grammar is regular, so a perfect held-out score means the parser is solved inside the grammar. Plain words are harder. The most common miss is an invented extra role for a noun ORB has no sound for, and counts and long multi-clause prompts are weak. When exactness matters, write a grammar caption or pass a Spec dict.

**DSP renderer**: on 50 held-out cues, 100% audit pass, max peak 0.983, max |DC| 1.5e-4, no clipping. The event stem is silent before the first event (−77.6 dB); the full mix carries the texture bed from the first sample (about −50 to −60 dBFS on sparse cues at the default −40 dB) unless the Spec uses `"texture": "none"` or the caption says `no bed`, which are digitally silent between events.

**Detail model** (optional, separate download): detail v6, a 9.06M complex-STFT U-Net. It passes the no-noise artifact gate (no tonal stripes, no added high-frequency energy, +0.063 dB in quiet frames) but does **not** meet its restoration gate (10% MR-STFT improvement; v6 reaches +0.84%), so treat it as a subtle texture stage. The DSP path is the shippable engine. Full numbers are in [`RESULTS.md`](RESULTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md).

**Size in context.** Both models together are 14.16M parameters (the open planner alone is 5.10M). Open text-to-audio models are around a billion: AudioLDM 2 is 1.1B in total and AudioLDM 2-large 1.5B ([diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2)), Stable Audio Open 1.0 is about 1B ([model card](https://huggingface.co/stabilityai/stable-audio-open-1.0)), and AudioGen-medium is 1.5B ([model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/AUDIOGEN_MODEL_CARD.md)). That is roughly 70–105× more than ORB with the detail model, and 200–300× more than the planner. They are open-vocabulary models and ORB is not, so this compares footprint, not capability.

## Where to get it

| What | Where | License |
|---|---|---|
| Code, DSP engine, MCP server | [github.com/stdevMac/orb-audio](https://github.com/stdevMac/orb-audio) | Apache-2.0 |
| Planner weights (5.1M) | [`weights/planner.pt`](weights/) (Git LFS) · `planner.safetensors` + `config.json` on the [GitHub release](https://github.com/stdevMac/orb-audio/releases) and [HF: marcosmaceo/orb-planner](https://huggingface.co/marcosmaceo/orb-planner) | Apache-2.0 |
| Planner training data (300k rows) + held-out sets | [HF: marcosmaceo/orb-planner-data](https://huggingface.co/datasets/marcosmaceo/orb-planner-data) | Apache-2.0 |
| Live demo | [HF Space: marcosmaceo/orb](https://huggingface.co/spaces/marcosmaceo/orb) | Apache-2.0 |
| Project page + demo video | [stdevmac.github.io/orb-audio](https://stdevmac.github.io/orb-audio/) · `orb-demo.mp4` on the GitHub release | Apache-2.0, except third-party media in the film (see [NOTICE](NOTICE)) |
| Detail model (9.1M) | [HF: marcosmaceo/orb-detail](https://huggingface.co/marcosmaceo/orb-detail) | **CC BY-NC-4.0**, research only |

The detail model was trained on [ESC-50](https://github.com/karolpiczak/ESC-50), which is
CC BY-NC 3.0, so those weights cannot be Apache-2.0; they are published under CC BY-NC-4.0. They are not included in this
repository. See [`weights/README.md`](weights/README.md).

## What this is not

- **Not a sample library replacement.** A recorded door slam beats a synthesized one. The natural, foley and everyday roles are physical models (damped mode banks for doors, bells and metal, Minnaert bubbles for water, arrival trains for thunder, swept tonal syllables for birds), so they read as the real thing in a mix, but up close they are cleaner and more regular than a field recording. They are rendered by DSP only: the detail model is skipped for them, because it was trained on degraded ESC-50 clips and only adds risk to a clean physical model.
- **Not an open-vocabulary text-to-audio model.** Since v8 the planner reads plain words ("rain on a tin roof with distant thunder"), but it can only plan with its 47 roles and 13 textures. For anything else it picks the nearest role or invents an extra one. For exact results an agent (or you) writes a grammar caption or the Spec.
- **Not a music generator.** Music is deferred to v2.
- **Not real-time.** A 12 s cue renders in about a second on CPU with the DSP path; the detail stage takes tens of seconds.
- **No voices or vocal animals.** Speech, laughter, crying, coughing, snoring, dogs, cats, cows, roosters and other vocal animals need recordings or a learned model; a physical model of a vocal tract does not sound like one. ORB leaves them out rather than ship a fake. Birdsong, crickets, frogs and owls are in, because they are whistles, stridulation and simple resonators.

## Repo layout

```
src/orb/
  api.py            public API: generate, render_spec, plan, weight lookup
  tools.py          agent tool layer (JSON in, JSON out) behind the MCP server
  mcp_server.py     MCP server for agents (orb-mcp)
  score.py          the typed Score
  grammar.py        Spec, captions, deterministic compiler
  render/           DSP: events, textures, music, mix, detail
  model/            planner, detail U-Net, tokenizer, grammar mask
  data/             datasets
  train/            training loops and losses
  eval/             audit and metrics
examples/           five runnable examples (05 is the agent tool layer)
scripts/            data generation, training, evaluation, packaging, release
web/                local demo server and UI
site/               GitHub Pages project page and demo video
docs/               architecture, score format, data, training, eval, agents
weights/            planner weights (Apache-2.0, Git LFS)
```

Further reading: [`docs/architecture.md`](docs/architecture.md) (how it works),
[`docs/score.md`](docs/score.md) (the Score schema),
[`docs/agents.md`](docs/agents.md) (MCP server and agent workflows),
[`docs/training.md`](docs/training.md) (GPU runbook),
[`docs/eval.md`](docs/eval.md) (gates),
[`docs/internal/`](docs/internal/) (original plan and pre-build red-team).

## Training

The planner trains on procedurally generated data in under an hour on one GPU.
The shipped v8 recipe, warm-started from `weights/planner_v7.pt`:

```bash
python scripts/gen_scores.py --n 150000 --out data/scores_150k_v7.jsonl --no-music --seed 2
python scripts/gen_freetext.py --n 150000 --seed 11 --out data/freetext_150k.jsonl
cat data/scores_150k_v7.jsonl data/freetext_150k.jsonl > data/planner_v8_train.jsonl
python scripts/train_spec.py --config configs/planner_v8.json

python scripts/eval_spec.py --checkpoint runs/planner_v8/best.pt --data data/heldout_v7.jsonl --n 500
python scripts/eval_freetext.py --planner runs/planner_v8/best.pt                      # 209 prompts
python scripts/eval_freetext.py --planner runs/planner_v8/best.pt --data data/freetext_blind.jsonl
```

The data files rebuild byte for byte from these seeds. GPU training is not
bit-deterministic, so a rerun lands on equivalent numbers rather than identical weights. The held-out and free-text eval sets in `data/` are the exact
ones behind [RESULTS.md](RESULTS.md).

See [`docs/training.md`](docs/training.md) for the GPU runbook and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a role or texture.

## Citation

```bibtex
@software{orb2026,
  title  = {ORB: a tiny, from-scratch audio engine for agents},
  author = {Marcos Maceo},
  year   = {2026},
  url    = {https://github.com/stdevMac/orb-audio},
  note   = {Version 0.1.0}
}
```

## Acknowledgements

Development was AI-assisted (Anthropic Claude). The architecture, training
runs, evaluation and release were prepared in a single assisted session. The
implementation draws on the following work:

- Ronneberger, Fischer & Brox. *U-Net.* MICCAI 2015.
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
- Wu & He. *Group Normalization.* ECCV 2018.
- Yamamoto, Song & Kim. *Parallel WaveGAN* (multi-resolution STFT loss). ICASSP 2020.
- Miyato et al. *Spectral Normalization for GANs.* ICLR 2018.
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
- Piczak. *ESC: Dataset for Environmental Sound Classification.* ACM MM 2015.

## License

Code and planner weights: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
third-party components.

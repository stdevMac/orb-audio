# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-event `params` in a hand-written Spec, so a pitched sound can follow the key of the music it sits over: `{"role": "church_bell", "t": 1.0, "dur": 6.0, "params": {"f0": 293.66}}` puts the bell's strike note on D4. `bell`, `ui_pop`, `click`, `impact` and `sub_drop` take `f0` (and `t60` or `f_end`); `church_bell` takes `f0`, `ring`, `hardness`, `tierce` and `beat`. `orb.grammar.EVENT_PARAMS` is the registry, `orb_describe_grammar` lists each role's params with ranges and defaults, and `orb_validate_spec` rejects unknown or out-of-range ones by path. Like `bed_db`, params are render-only: the caption grammar and the planner do not express them, and the tokenizer refuses a Spec that carries any. A Spec without params renders bit-identically to before, and a param at its default is the same audio as leaving it out. `examples/06_tuned_bell.py`, `tests/test_params.py`.

### Changed

- Realism pass on the natural, foley and everyday sounds: 30 roles rebuilt from the physics of the real sound and checked against ESC-50 recordings and two audio judges (RESULTS.md). Thunder is a crack, then rumble arriving from the rest of the bolt and a long low tail, instead of a whoosh. Big gains for church bell, heartbeat, siren, owl, birds, applause, fireworks, gunshot and the desk sounds. Variant contract unchanged, no planner retrain. The site's pre-rendered library is re-rendered with the new sounds.

## [0.1.0] — 2026-11-16

First public release.

### Added

- Planner v8 reads plain words: "rain on a tin roof with distant thunder" now goes straight to a Spec. Vocabulary 394 → 395, adding the `NOBED` token (texture `none`). Warm-started from v7 on 150k grammar pairs plus 150k synthetic free-text pairs (`orb.freetext`, `scripts/gen_freetext.py`), 40k steps. Free-text pass rate on our 209-prompt set: 58.4%, against 45.0% for v7 behind the keyword helper. On a 40-prompt blind set: 70.0% raw, against 77.5% for the helper. Grammar held-out sets: unchanged at 100%. v7 is kept as `weights/planner_v7.pt`. See RESULTS.md.
- `scripts/eval_freetext.py`, `data/freetext_eval.jsonl` (frozen) and `data/freetext_blind.jsonl`; `scripts/check_web_planner.py --free N` adds plain-text prompts to the web parity check.
- Site: plain words go to the planner first. The keyword helper's caption is shown as a "second opinion" that you can run with one click. The int8 planner download stays at 5.1 MB.
- Planner v7 (vocabulary 358 → 394, 150k pairs, 30k steps): 500/500 and 2000/2000 exact on the v7 held-out sets, 500/500 on the v6 set, 0 decode failures, every role 100%. v6 kept as `weights/planner_v6.pt`.
- Detail gate now passes on all 32 cues (foley bypasses the detail model).
- Everyday sound library (`orb.render.natural`): 31 more physically modelled
  roles, appended after the first sixteen so no existing seed moves:
  bird_chirp, crickets, wave_crash, wind_gust, splash, frog, owl, glass_break,
  clock_tick, phone_ring, keyboard_typing, mouse_click, camera_shutter,
  paper_rustle, cork_pop, coin_drop, metal_clang, wood_creak, heartbeat,
  applause, pouring_water, can_opening, clock_alarm, punch, sword_clash,
  gunshot, car_horn, siren, engine, church_bell, fireworks. The seed picks the
  variant (see `orb.render.natural.VARIANTS`). Measured against ESC-50 where a
  class exists (RESULTS.md).
- Five scene beds built from real events, not shaped noise
  (`orb.render.ambience`): `ocean`, `stream`, `forest`, `night`, `traffic`,
  appended after the eight original textures. They default to −14 dB under
  the events (the noise beds stay at −40 dB).
- DSP-only routing: the 37 physically modelled roles and the 5 scene beds
  bypass the detail model by default (`orb.render.mix.DSP_ONLY_ROLES`,
  `apply_detail_routed`). Cues that use only the original roles take exactly
  the old path.
- Grammar scenes: beach, forest morning, night camp, kitchen, office, city
  street, sword fight, haunted house; 25% of training captions are scenes.
- Site: an "Everyday" library group, new "Nature & foley" scenes, eight more
  example chips, and prompt-helper words for the new sounds and beds.
- `tests/test_natural.py`.

- Six physically modelled natural and foley roles in `orb.render.foley`:
  `thunder`, `door_slam`, `knock`, `footsteps`, `water_drop`, `explosion`.
  The seed picks the variant (distance, floor, door weight, drip/plop).
- Grammar scenes (storm, arrival, cave, battle) in `orb.grammar.sample_scene`;
  20% of generated training captions are scenes.
- Planner v6: same architecture, vocabulary 352 → 358. 98.8% exact on 500
  held-out captions (98.5% on captions using a new role), 99.2% on 2,000,
  0 decode failures. The previous planner is kept as `weights/planner_v5.pt`.
- Site: a "Nature & foley" library group (thunderstorm, footsteps → door → knock,
  cave drips, explosion), "Thunderstorm" and "Footsteps, door, knock" chips, and
  prompt-helper recipes that use the real roles instead of impact/sub-drop fakes.
- `tests/test_render_baseline.py`: 150 existing renders pinned bit for bit;
  `tests/test_foley.py`.

- Typed `Score` / `Spec` schema with validation and JSON serialisation.
- Deterministic DSP renderers at 48 kHz: the 10 core SFX roles, 8 noise texture
  beds, and the music renderer (music is deferred to v2).
- Compact Spec planner: byte-level tokenizer and constrained decoding. The first
  version (v5, 5.08M parameters) reached 96.6% exact on 500 held-out captions;
  the shipped planner is v8 (below).
- Optional detail model (detail v6): 9.06M complex-STFT U-Net with bilinear
  resize-conv upsampling, a smoothed-mask residual above 2 kHz that tapers off
  between 8 and 9.5 kHz, a quiet-frame gate and an energy gate; no noise channel.
  It passes the no-noise artifact gate (`scripts/eval_detail_artifacts.py`).
- `orb.api` public surface: `generate`, `render_spec`, `plan`, `load_planner`,
  `load_detail`, `spec_to_dict`, `dict_to_spec`, `save_wav`.
- `orb-generate` command-line tool.
- `orb-mcp`: a Model Context Protocol server so agents can plan, render and
  edit Specs as tools (`pip install "orb-audio[mcp] @ git+https://github.com/stdevMac/orb-audio"`
  until the package is on PyPI; see `docs/agents.md`).
- `orb.tools` agent tool layer and `examples/05_agent_tool.py`.
- Planner weight lookup (`resolve_planner_path`): `planner_path` -> `$ORB_PLANNER` ->
  `./weights/` -> the clone's `weights/` (`.pt`, or `.safetensors` + `config.json`) ->
  a cached Hugging Face Hub download of `marcosmaceo/orb-planner` when `huggingface_hub`
  is installed (`orb-audio[hub]`), so `generate(caption)` works after a pip install.
  The CC BY-NC detail weights are looked up locally only (`$ORB_DETAIL`), never downloaded.
- Spec options for the background bed: `"texture": "none"` renders no bed, and
  `"bed_db"` sets the bed level relative to the events (default −40 dB). A grammar
  caption can say `no bed`; `generate()`, `orb-generate` and `orb_generate` read such
  a caption exactly and render it without the planner (`out["planned"]` is False).
- Noise pass: the default bed is 10 dB quieter (−40 dB below the events, was −30);
  noise attacks and the reverb IR are band-limited; continuous roles (whoosh, riser,
  swell, noise_sweep) fade in after apex alignment, so a whoosh no longer starts with a
  click; `save_wav` rounds to the nearest 16-bit step instead of truncating. See
  RESULTS.md, "Noise pass".
- Local web demo (`web/`) with pre-rendered examples and live generation, and
  a Hugging Face Space (`hf/space/`) that runs the planner + DSP path on CPU.
- GitHub Pages project page (`site/`) with a demo video.
- Planner weights as safetensors on the Hugging Face Hub and as GitHub release
  assets (with the demo video), plus the v8 planner training data (300k rows) as a Hub
  dataset.
- Training, evaluation, and packaging scripts.
- Release tooling: `scripts/release_check.sh` (local preflight) and
  `scripts/release_publish.sh` (GitHub + Hugging Face publish, dry run by
  default).
- Test suite (`pytest -q`) and CI workflow.
- Packaging as `orb-audio` (the name `orb` is taken on PyPI); the import name
  is `orb`. Install from GitHub for now:
  `pip install "orb-audio @ git+https://github.com/stdevMac/orb-audio"`. Design notes from the build live in `docs/internal/`.

### Fixed

- `SpecTokenizer.encode_spec` raised a bare `KeyError` on `texture="none"`;
  it now raises a `ValueError` that says no-bed Specs bypass the planner.
- Background hiss: the bed absorbed the mix normalisation gain on sparse cues,
  `np.interp` left a flat white-noise tail above the last spectral control
  point, and grains were unfiltered white bursts. See `RESULTS.md`.
- DC offset from one-sided transients and slow texture LFOs.
- Constrained decoding skipped `END_EVENTS` at the event limit.
- Planner event distribution excluded 0 and 1 event, so single-sound captions
  emitted zeros.
- `torch.istft` window device mismatch during CUDA inference.
- Detail inference fed whole clips to a model trained on 256-frame crops.

### Known limitations

- The detail model does not meet its restoration gate (10% MR-STFT; v6 reaches
  +0.84%); it is an optional texture stage, and the DSP path is the shippable engine.
- The detail weights are CC BY-NC-4.0 and are distributed separately.
- Music is not implemented (v2).
- Plain-word prompts are best-effort: the planner may invent the nearest role
  for a noun ORB has no sound for, and counts and long multi-clause prompts are
  weak. Pass a grammar caption or a Spec dict when exactness matters.
- Renderer rough edges, planned for 0.1.1. Fixing them changes the bit-identical
  render baselines, so they wait for a point release:
  - `pan` is ignored on stereo-wide roles (bell, whoosh and others).
  - `click` has a faint sizzle above 10 kHz.
  - A long `sub_drop` can end on a soft thump.

[Unreleased]: https://github.com/stdevMac/orb-audio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stdevMac/orb-audio/releases/tag/v0.1.0

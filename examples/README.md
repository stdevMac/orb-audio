# Examples

Run any of these from the repository root. They all write into `out/`.

```bash
python examples/01_generate.py      # caption -> WAV (planner + DSP)
python examples/02_edit_spec.py     # change one number, re-render
python examples/03_variants.py      # N non-repeating takes from one Spec
python examples/04_no_torch.py      # pure DSP, numpy only, no weights
python examples/05_agent_tool.py    # an agent scores a UI animation via orb.tools
python examples/06_tuned_bell.py    # three church bells on a D minor triad (event params)
```

| Example | Needs torch? | Needs weights? | Shows |
|---|---|---|---|
| `01_generate.py` | yes | planner | the full pipeline |
| `02_edit_spec.py` | no | no | the sound is data |
| `03_variants.py` | no | no | infinite non-repeating takes |
| `04_no_torch.py` | no | no | the shippable DSP engine |
| `05_agent_tool.py` | no | no | the agent tool layer (`orb.tools`, as used by `orb-mcp`) |
| `06_tuned_bell.py` | no | no | per-event `params`: a sound on a note |

## The caption grammar

The planner was trained on one caption format. Stay on the grid and it
reproduces the Spec exactly:

```
{duration:.2f}-second {mood} cue, {texture} bed[, with {role} @{t:.2f} {dur:.2f}s {gain:+.0f}dB {pan} #{seed:02x}, ...]
```

- duration: 2.0–30.0 in **0.5 s** steps
- `t` and `dur`: **0.01 s** steps, `t + dur <= duration`
- gain: **even** dB from −30 to 0
- pan: `L4 L3 L2 L1 C R1 R2 R3 R4`
- at most 8 events
- moods: calm, warm, playful, hopeful, bright, tense, dark, eerie, neutral, intense
- textures: room_cold, wind, city_hum, vinyl, drone, rain, fire, sci_fi
- roles: whoosh, impact, ui pop, riser, swell, click, bell, sub drop, glitch, noise sweep,
  thunder, door slam, knock, footsteps, water drop, explosion (in a caption, underscores
  become spaces). For the last six the seed picks the variant: thunder `#..` mod 4 is
  distance (0 close, 3 far), footsteps mod 3 the floor, explosion mod 3 the distance.
- `{texture} bed` may be `no bed` (silence between events, for UI sounds). The planner
  never saw such captions, so `generate()` reads them exactly and skips the planner:
  `2.00-second playful cue, no bed, with ui pop @0.50 0.16s -12dB C #03`

Examples:

```
17.50-second tense cue, room cold bed, with footsteps @0.39 3.55s -14dB R1 #2c, door slam @4.52 0.81s -6dB R2 #bc, knock @5.82 0.49s -8dB L2 #23
```


```
12.00-second intense cue, drone bed, with riser @1.00 2.50s -18dB C #55, whoosh @4.20 1.20s -10dB L3 #77, impact @5.00 0.90s -4dB C #66, sub drop @5.00 1.60s -8dB C #88, noise sweep @6.50 2.00s -16dB R2 #97, impact @9.50 0.90s -4dB C #99, bell @9.50 2.00s -12dB R1 #3d
```

If a caption produces the wrong Spec, it is usually the planner's ~3.4%
single-digit copy error, not the renderer. Pass a Spec dict to `render_spec`
when you need exactness.

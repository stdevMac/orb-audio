# ORB for agents

ORB is small, fast and deterministic, and its input is a typed JSON document.
That makes it a good tool for an LLM agent. The agent writes the sound design
as data, and ORB turns that data into a WAV file in milliseconds, with the
timing exact to the sample.

You can use it in two ways:

- **MCP server** (`orb-mcp`): for Claude Code, Claude Desktop, IDEs and any
  other MCP client.
- **Plain tool definitions** (`orb.tools`): JSON Schemas plus a dispatcher, for
  any tool-use loop you write yourself (Anthropic, OpenAI-style, or your own).

Both run the same code. `orb-mcp` is only a transport around `orb.tools`.

## The five tools

| tool | input | what it does | needs torch |
|---|---|---|---|
| `orb_describe_grammar` | none | returns roles (with `character`, `apex_fraction` and defaults), textures, moods, limits, the Spec JSON Schema and a worked example | no |
| `orb_validate_spec` | `spec` | checks a Spec without rendering it; returns `valid`, path-prefixed `errors`, `warnings`, the normalised `spec` and its canonical `caption` | no |
| `orb_render_spec` | `spec`, `seed?`, `output_dir?`, `name?` | Spec → WAV; returns `path`, `duration_s`, `sample_rate`, `spec`, `caption`, `audit` | no |
| `orb_variants` | `spec`, `n` (1–32), `base_seed?`, `vary_event_seeds?` | n different takes of the same design, written as `<name>_v00.wav`, `<name>_v01.wav`, ... | no |
| `orb_generate` | `caption` **or** `spec`, `seed?`, `detail?` | the full pipeline; a caption goes through the 5M planner first | only for `caption` / `detail` |

Every successful result is JSON and includes the absolute WAV path, the exact
Spec that was rendered and an audio `audit` (peak, RMS, DC offset, click score
and a `passes` flag). Errors come back as tool errors, not crashes. Their messages
point at the problem (`events[2]: ends at 4.400 s, after the cue ends at 4 s
(move t earlier, shorten dur, or lengthen duration)`), so the model can fix the Spec and try again.

Every file goes under one **output root**: `--output-dir`, else
`$ORB_OUTPUT_DIR`, else `./orb_out`. The root is set by whoever starts the
server, not by the model. A tool call's `output_dir` only picks a
sub-directory of the root: a relative path is joined onto it, and a path that
resolves outside it (an absolute path elsewhere, `..`, or a symlink pointing
out) is rejected with a tool error. `name` is reduced to a plain file name.

Without a `name`, the file name is a hash of the Spec and the seed, with a
`_detail` suffix when `detail` is on. The same inputs always map to the same
file, so rendering the same thing twice overwrites that file with identical
audio. Nothing is cached: every call renders again. Plain renders take
milliseconds, so this costs little.

## Wiring

Install ORB with the MCP extra:

```bash
# numpy + the official MCP SDK
pip install "orb-audio[mcp] @ git+https://github.com/stdevMac/orb-audio"
# add the `train` extra (torch) for caption mode
pip install "orb-audio[mcp,train] @ git+https://github.com/stdevMac/orb-audio"
orb-mcp --help
```

`orb-mcp` is the same as `python -m orb.mcp_server`. The flags are:

- `--output-dir DIR`: the output root. Tool calls cannot write outside it.
- `--planner PATH`: planner weights for caption mode (`planner.pt`, or
  `planner.safetensors` with its `config.json` next to it). The default is
  `$ORB_PLANNER`, then `weights/planner.pt`. After a pip install there is no
  `weights/` folder: download `planner.safetensors` + `config.json` from
  [marcosmaceo/orb-planner](https://huggingface.co/marcosmaceo/orb-planner) and pass
  this flag.
- `--builtin`: forces the dependency-free server (see below).

### Claude Code

```bash
claude mcp add orb -- orb-mcp
# or pin the output folder
claude mcp add orb -e ORB_OUTPUT_DIR="$PWD/sfx" -- orb-mcp
# or from a source checkout, without installing the script
claude mcp add orb -- /path/to/cool-sounds/.venv/bin/python -m orb.mcp_server
```

To share the server with everyone on a project, put this in a `.mcp.json` at the
repo root:

```json
{
  "mcpServers": {
    "orb": {
      "command": "orb-mcp",
      "args": ["--output-dir", "assets/sfx"]
    }
  }
}
```

Then ask: *"Score the onboarding animation in `src/Onboarding.tsx` with ORB:
a riser into the logo landing, whooshes on the three cards, a click on the CTA
at 2.8 s."*

### Claude Desktop

Edit `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "orb": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "orb.mcp_server"],
      "env": {
        "ORB_OUTPUT_DIR": "/Users/you/Music/orb",
        "ORB_PLANNER": "/absolute/path/to/weights/planner.pt"
      }
    }
  }
}
```

Desktop does not start servers with your shell's `PATH`, so use absolute paths.
Restart Desktop after you edit the file.

### No SDK? It still works

If the `mcp` package is not installed, `orb-mcp` falls back to a small built-in
JSON-RPC stdio server (`orb.mcp_server.serve_builtin`). It has no dependencies,
speaks `initialize`, `ping`, `tools/list` and `tools/call`, and returns the same
tool schemas and results. A numpy-only install is therefore a working MCP
server. Pass `--builtin` to force the fallback even when the SDK is installed.

## Any tool-use loop

`orb.tools` exports the schemas in two dialects, plus one function that never
raises:

```python
from orb.tools import anthropic_tools, openai_tools, run_tool

anthropic_tools()  # [{"name", "description", "input_schema"}, ...]
openai_tools()     # [{"type": "function", "function": {"name", "description", "parameters"}}, ...]
text, is_error = run_tool("orb_render_spec", {"spec": {...}})  # JSON string + error flag
```

A complete agent with the Anthropic SDK (`pip install anthropic`):

```python
import anthropic
from orb.tools import anthropic_tools, run_tool

client = anthropic.Anthropic()
messages = [{"role": "user", "content": (
    "Make a 3 second 'achievement unlocked' sting for a game: a swell into a bright "
    "impact at 1.0 s with a bell on top. Then give me 4 variants of it."
)}]

while True:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        tools=anthropic_tools(),
        messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break
    results = []
    for block in response.content:
        if block.type == "tool_use":
            text, is_error = run_tool(block.name, block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": text, "is_error": is_error})
    messages.append({"role": "user", "content": results})

print(next(b.text for b in response.content if b.type == "text"))
```

Return every `tool_result` from one turn together in a single user message.
Leave `tool_choice` at its default. The model calls `orb_describe_grammar`
by itself when it needs the vocabulary.

[`examples/05_agent_tool.py`](../examples/05_agent_tool.py) runs the same loop
offline. A small deterministic function stands in for the model, so it needs no
API key and no torch.

## Spec as contract

Most generative audio is a prompt in and a waveform out. If the result is
wrong, you change the prompt and hope. ORB adds a step in the middle: the
**Spec**, a typed document that says what plays, when, how loud and where.

```json
{"duration": 4.0, "texture": "room_cold", "mood": "bright",
 "events": [{"role": "impact", "t": 1.2, "dur": 0.5, "gain_db": -6, "pan": 0.0, "seed": 21}]}
```

For an agent, this changes the job:

- **Exact timing.** An event at `t = 1.25` starts at sample 60 000. If the UI
  framework says the card lands at 1.6 s, the whoosh lands at 1.6 s. Nothing
  has to "listen and adjust".
- **Deterministic.** The same Spec and seed give bit-identical audio. A render is a
  build step, not a gamble: the same inputs always write the same file.
- **Diffable and reviewable.** A sound design is about 1 kB of JSON next to the
  code it scores. A reviewer can read "the impact moved from 1.20 to 1.25 s" in a
  pull request.
- **Checkable before it costs anything.** `orb_validate_spec` returns precise,
  path-prefixed errors. The usual loop is draft, validate, fix, render, and it
  converges in one or two turns.
- **Balance, not level.** The final mix is loudness-normalised, so `gain_db`
  sets how loud events are *relative to each other*. A -6 dB impact over a
  -20 dB riser reads as "the hit dominates" at any output level.
- **No model in the loop unless you want one.** The agent is the planner. It
  writes Specs directly, and rendering is numpy-only. The 5M planner is there
  for grammar captions and plain words (since v8), and `orb_generate` returns the Spec it planned so the
  agent can edit it.

The rule of thumb is that **the agent decides what and when, and ORB decides
how it sounds.**

## Recipes

### A UI sound kit

UI sounds are short and are heard hundreds of times. Make them tiny, and make
several takes of each so repeats do not sound mechanical.

```python
kit = {
    "tap":     ("click",  0.10, -12, "neutral"),
    "toggle":  ("ui_pop", 0.16, -12, "playful"),
    "error":   ("glitch", 0.25, -10, "tense"),
    "success": ("bell",   0.90, -12, "bright"),
}
for name, (role, dur, gain, mood) in kit.items():
    spec = {"duration": max(0.3, dur + 0.15), "texture": "none", "mood": mood,
            "events": [{"role": role, "t": 0.0, "dur": dur, "gain_db": gain}]}
    run_tool("orb_variants", {"spec": spec, "n": 4, "name": name, "output_dir": "ui"})
```

`"texture": "none"` renders the events with no background bed, which is what UI
sounds want. Any other texture adds a quiet ambience bed under the whole cue, by default
40 dB below the events; set `"bed_db"` in the Spec to move it (Spec only). In caption
mode, write `no bed` in place of the bed clause (`2.00-second playful cue, no bed, with ui
pop @0.50 0.16s -12dB C #03`): such a caption is read exactly, without the planner. `duration` can be as short as 0.1 s. The planner's 2–30 s range applies only to
captions, and a shorter rendered Spec only produces a warning. Omitted fields
take the role defaults from `orb_describe_grammar`.

### A trailer cue

A trailer hit is a build that peaks on a frame. A riser peaks at
`apex_fraction = 0.92` of its length, so to land its peak on a hit at time `T`,
start it at `t = T - 0.92 * dur`.

```python
T, rise = 3.0, 2.5                      # the hit, and how long the build is
spec = {"duration": 6.0, "texture": "drone", "mood": "intense", "events": [
    {"role": "riser",    "t": round(T - 0.92 * rise, 2), "dur": rise, "gain_db": -18, "pan": 0.0, "seed": 3},
    {"role": "whoosh",   "t": T - 0.15, "dur": 0.6, "gain_db": -10, "pan": -0.5, "seed": 8},
    {"role": "impact",   "t": T,        "dur": 0.9, "gain_db": -2,  "pan": 0.0, "seed": 1},
    {"role": "sub_drop", "t": T,        "dur": 1.6, "gain_db": -8,  "pan": 0.0, "seed": 4},
    {"role": "bell",     "t": T + 0.02, "dur": 2.0, "gain_db": -16, "pan": 0.25, "seed": 77},
]}
run_tool("orb_render_spec", {"spec": spec, "name": "trailer_hit"})
```

`swell` (apex 0.90) and `noise_sweep` (apex 0.50) work the same way. Read
`apex_fraction` from the grammar rather than hard-coding it.

### Sync to motion keyframes

Motion tools and UI frameworks already know when things move. Map each keyframe
to a role, put the event on the keyframe's time, and pan it to where it is on
screen.

```python
ROLE_FOR = {"anticipation": "riser", "land": "impact", "slide_in": "whoosh",
            "tap": "click", "confirm": "bell"}
events = [{"role": ROLE_FOR[k["kind"]], "t": k["t"], "dur": k["dur"], "pan": k["x"]}
          for k in timeline["keyframes"]]
```

[`examples/05_agent_tool.py`](../examples/05_agent_tool.py) does this end to end:

- It scores a 4 s onboarding animation and lands a riser's apex on the logo
  landing.
- It lets validation catch a bell that runs past the end, then fixes it.
- It renders the cue and a pool of four tap takes.
- It writes `onboarding.sound.json`, which records the timeline, the Spec, the
  audit and a binding from each keyframe to its sound.

If you move a keyframe and re-run, the sound moves with it.

### A bell in the key of the music

A cue that sits over music has to follow its key, and a role's pitch is otherwise fixed
(the synth `bell` rings at 880 Hz) or left to the seed (each `church_bell` is cast
somewhere between C4 and B4). An event's `params` puts it on a note:

```python
D4, F4, A4 = 293.66, 349.23, 440.00
spec = {"duration": 10.0, "texture": "none", "mood": "calm", "events": [
    {"role": "church_bell", "t": 0.5, "dur": 9.0, "seed": 0, "params": {"f0": D4}},
    {"role": "church_bell", "t": 2.5, "dur": 7.0, "seed": 2, "params": {"f0": F4}},
    {"role": "church_bell", "t": 4.5, "dur": 5.0, "seed": 4,
     "params": {"f0": A4, "hardness": 0.9, "ring": 1.5}},
]}
run_tool("orb_render_spec", {"spec": spec, "name": "d_minor_bells"})
```

| Role | Params |
|---|---|
| `bell`, `ui_pop`, `click`, `impact` | `f0` (Hz), `t60` (ring time, s) |
| `sub_drop` | `f0`, `f_end` (Hz: where the drop starts and lands) |
| `church_bell` | `f0` (strike note, Hz), `ring` (x decay), `hardness` (0 felt .. 1 hard), `tierce` (0.6 minor, 0.63 major), `beat` (doublet width) |

`orb_describe_grammar` returns the same table under `roles.<role>.params`, with each
param's range and default. A default of `null` means there is none: unset, the seed picks
the value (a church bell's note). `orb_validate_spec` rejects a param the role does not
take, or one outside its range, with the event's path.

- **`f0` on a church bell is its strike note**, the pitch a listener hears: half the bell's
  nominal. The seed still casts the bell (how far its hum and prime are from true, the
  weight of each partial), so two seeds on one note are two different bells on that note.
- **A param left out changes nothing.** A Spec without params renders what it always
  rendered, and a param set to its default is the same audio as leaving it out.
- **Spec only.** Like `bed_db`, params have no place in the caption grammar, so the planner
  never emits them and a caption will not carry them. Validation says so in a warning.

[`examples/06_tuned_bell.py`](../examples/06_tuned_bell.py) tolls a D minor triad this way.

## Limits

- Up to 8 events per Spec, 47 roles and 13 textures (`orb_describe_grammar` lists them, with each role's variants). A cue lasts 0.1–60 s when rendered
  from a Spec, or 2–30 s when planned from a caption.
- ORB makes sound effects, everyday sounds and ambience beds. It does not make music, speech or vocal animals (dogs, cats, laughter, coughs): those need recordings.
- The 37 physically modelled roles and the scene beds (ocean, stream, forest, night, traffic) are always rendered by DSP; `detail: true` only touches the original ten roles and eight noise beds.
- Caption mode needs torch and the planner weights (see `--planner` above). `detail: true` needs the
  separately downloaded `weights/detail.pt`, which is **CC BY-NC-4.0**. See
  [`MODEL_CARD.md`](../MODEL_CARD.md). The Spec path uses no learned weights,
  so its output carries no model license terms.
- `detail: true` is slow: about 80–95 s for a 6 s cue on CPU. Some MCP clients
  time out tool calls well before that, so raise the client's tool timeout if
  you use it. The Spec path takes milliseconds.
- The server writes files only under its output root (see above). Per-call
  `output_dir` values that resolve outside it are refused, and names are
  reduced to plain file names. The server does not play audio. Your client or
  app does that.

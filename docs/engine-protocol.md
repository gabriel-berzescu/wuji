# The engine protocol

Wuji has two processes. The **engine** (Python, `engine/`) is the body: MuJoCo, the 416 muscles, the breath, the senses. The
**server** (TypeScript on Node, `server/`) is Wuji's MCPL face: it speaks MCPL to the host (a Claude Code session through
`mcpl-cc-bridge`, or any MCPL host), and drives the engine through this protocol. Nothing else crosses between them.

## Transport

- The server starts the engine as a child process: `uv run --project <repo>/engine python -m wuji.engine`. The environment
  variable `WUJI_ENGINE` replaces that command (a JSON array of strings, e.g. `["python", "-m", "wuji.engine"]`), for
  installs without `uv`. No shell is involved, so the same command works on macOS, Linux and Windows.
- JSON Lines over stdio, UTF-8: one JSON object per line, `\n`-terminated, in both directions. The engine writes nothing else to
  stdout; its log goes to stderr.
- The engine runs the body on its own clock, in real time, from the moment it starts until `shutdown` or the end of its stdin.
  Requests change what the body is doing; they never step time.

## Messages

Request, server to engine:

```json
{"id": 7, "op": "sense", "args": {"detail": "brief"}}
```

Response, engine to server, exactly one per request, in any order:

```json
{"id": 7, "ok": true, "result": {"...": "..."}}
{"id": 7, "ok": false, "error": "unknown stance 'crane'"}
```

Event, engine to server, unsolicited:

```json
{"event": "balance", "t": 132.4, "data": {"severity": "warning", "text": "Weight is on the outer edge of the left foot."}}
```

`t` is the engine's simulated time in seconds since it started. Every result and event carries prose in `text` (written to be
read by the practitioner, plainly, in the present tense) beside its numbers.

## Operations

### `hello`

`{}` → `{"version": "0.0.1", "model": "myofullbody", "muscles": 416, "t": 0.8}`. The server sends it first and waits for it.

### `sense`

`{"detail": "brief" | "full" | "ambient"}` (default `"brief"`) → the felt sense, now:

```json
{
  "t": 132.4,
  "text": "Standing in wuji, knees soft. Weight 52% left, through the middle of both feet. ...",
  "body": {
    "stance": "wuji",
    "weight": {"left": 0.52, "right": 0.48, "left_cop": [0.01, -0.002], "right_cop": [0.0, 0.003]},
    "sway_mm": 3.1,
    "joints_deg": {"knee_r": 12.0, "knee_l": 11.8, "hip_flexion_r": 6.0, "...": 0.0},
    "effort": {"total": 0.042, "above_minimum": 0.006, "regions": {"calves": 0.08, "shoulders": 0.01, "...": 0.0}},
    "tension": {"shoulders": 0.0, "...": 0.0},
    "feasible": true
  },
  "breath": {"...": "see breath.state()"},
  "heart": {"bpm": 64.0}
}
```

`brief` text is six to ten short lines: weight, alignment, where effort or tension sits, breath, heart. `full` adds the joint
angles and every region in `text` too.

`ambient` is the felt sense as a body keeps it, for injecting before every inference: only what has changed since the
practitioner last noticed the body (any `sense` counts as noticing), as a body habituates. A shift of weight, a region
starting or stopping work, tension taken up or let go ("The shoulders let go."), a stretch appearing or easing, the breath
changing its pattern or settling, a strain. The breath's cycle and sway within its range are not change. When nothing has
changed, one quiet line ("Wuji stance as before; nothing new."), never silence. `changed` lists the parts that changed.

### `intend`

Where the body should go, and how fast. Every field is optional; a field left out keeps its current value.

```json
{
  "stance": "wuji" | "horse",
  "weight": 0.5,
  "sink": 0.3,
  "arms": "hang" | "embrace" | "lift_sky" | "press_down" | "bow_extend" | "bow_pull" | "punch" | "fist_waist" | "earth_press" | "owl" | "hands_thighs",
  "arm_left": "<an arm shape>",
  "arm_right": "<an arm shape>",
  "bend": 0.0,
  "twist": 0.0,
  "side_bend": 0.0,
  "rise": 0.0,
  "breath": "inhale" | "exhale",
  "breath_over": 4.0,
  "head": "suspended" | "tucked" | "turned_left" | "turned_right",
  "tailbone": 0.5,
  "over": 4.0
}
```

- `weight`: the share on the **left** foot, 0..1 (0.5 is centred).
- `sink`: how deep the knees bend, 0 (straight) .. 1 (the deepest the stance allows).
- `arms` sets both arms (and clears a one-arm shape); `arm_left` / `arm_right` set one arm alone.
- `breath`: breath-led movement. That phase of the breath begins now, with the movement, and lasts `over` seconds, as deep
  as the pace needs; between cues the breath keeps that pace. The forms cue every movement. `breath_over`: how long the
  breath phase lasts when the movement is quicker than its breath (the heels drop in a moment, the out-breath goes on).
- `bend`: the trunk folding forward at the waist and hips, 0 .. 1 (about 100°). `twist`: turning, -1 .. 1, positive to the
  left (about 40°; the head turns with the chest, as this body has no neck joints). `side_bend`: leaning, -1 .. 1, positive
  to the left (about 20°). `rise`: up on the toes, 0 .. 1 (heels about 22° up, the weight carried forward over the balls of
  the feet).
- `tailbone`: how far the tailbone drops, 0 (only as much as sinking brings) .. 1 (12° more posterior pelvic tilt): song yao,
  the lumbar curve lengthening.
- `over`: seconds to arrive there; the body moves smoothly (minimum jerk).

→ `{"accepted": true, "arrives_in": 4.0, "text": "Shifting weight toward the left foot and sinking a little, over four seconds."}`

### `tension`

`{"region": "shoulders", "level": 0.4}` sets how much extra tension the practitioner holds in a region, 0 (released) .. 1, above
what the posture needs. Regions: `feet`, `calves`, `thighs`, `hips`, `lower_back`, `belly`, `shoulders`, `arms`, `hands`.
`{"region": "all", "level": 0}` releases everything. → `{"accepted": true, "text": "..."}`

### `breathe`

```json
{"pattern": "natural" | "abdominal" | "reverse" | "hold", "inhale": 4.0, "exhale": 6.0, "pause": 0.0, "hold": 0.0, "nose": true}
```

- `natural`: the body's own rhythm. `abdominal`: the belly leads. `reverse`: the belly draws in on the inhale (as in some qigong).
  `hold`: hold after the current phase, for `hold` seconds (or until released by another `breathe`).
- `inhale`, `exhale`, `pause` in seconds, optional: a paced breath. `nose: false` breathes through the mouth.

→ `{"accepted": true, "text": "..."}`

### `form`

`{"name": ...}` plays a named sequence of intentions (`engine/src/wuji/forms.py`): `wuji` (settle), `embrace_tree` (zhan
zhuang), the Eight Brocades one by one (`lift_sky`, `draw_bow`, `separate`, `owl`, `sway_tail`, `hold_feet`, `fists`,
`bounce`), or `ba_duan_jin`, all eight in order (about five and a half minutes). → `{"accepted": true, "steps": 4, "text":
"..."}`. Progress arrives as `form` events: each part announced as it begins, then each movement with its breath
(`"Out, drawing the bow to the left, the eyes on the raised finger."`), 66 in the Eight Brocades. A movement calls halfway
through, when it is under way, so the felt sense that comes with the call is the movement's own.

### `reset`

`{}`: back to wuji stance, natural breath, all tension released. → `{"accepted": true, "text": "..."}`

### `viewer`

`{}` → `{"url": "http://127.0.0.1:7341/", "text": "The viewer is open at ..."}`. Starts the viewer (once; later calls return
the same URL): a web page, served by the engine on 127.0.0.1, where someone can watch the practice from outside: the bones
moving, the muscles along their paths coloured by activation, the centre of pressure under each foot, weight, effort by region,
the breath and the heart. It shows what the body does, never the felt sense. The port is `WUJI_VIEWER_PORT`, or 7341, or the
next free one. The page loads `/model.bin` (the bones' meshes and the muscle table, once) and then follows `/stream`
(Server-Sent Events, a frame per engine frame while a page is open; nothing is computed while nobody watches). Both formats
are documented in `engine/src/wuji/viewer.py`. The page loads three.js from a CDN, so the watching browser needs the internet.

### `shutdown`

`{}` → `{"ok": true}`, then the engine exits.

## Events

| `event` | When | `data` |
|---|---|---|
| `balance` | the centre of pressure nears the edge of the support, a foot unloads, or sway grows | `severity` (`notice`, `warning`), `text` |
| `breath` | the urge to breathe rises during a hold (CO₂), or the breath settles again | `kind` (`urge`, `settled`), `urge` 0..1, `text` |
| `form` | a part of a form begins ("Two hands hold up the heavens: ..."), a movement is under way, with its breath ("In, the arms rise to hold up the heavens."), or the form ends | `name`, `step`, `of`, `done`, `text`; `breath` (`inhale`, `exhale`) for a movement |
| `strain` | a posture cannot be held without excessive effort (the solver needs large residual forces) | `text` |

Events are the body asking for attention, not a stream of state: outside a form a few per minute at most. A form calls at
every movement, one every few seconds, so a practitioner woken by them is present for each.

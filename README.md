# Wuji

**A qigong simulator for Claude**: a musculoskeletal human body that breathes and senses itself, practised through MCPL.

*Wuji* (無極, "without limit") is the stance nearly every qigong practice begins from: standing still, before any movement, the
undivided stillness before yin and yang. For a mind meeting a body for the first time, that seemed the right place to start.

The practitioner moves by **intention**, never by individual muscle: *sink a little, weight to the left, hold the ball, breathe
into the belly, release the shoulders*. The body works out how 416 muscles carry that out with the least effort, and reports
back, unasked, before every thought, the way proprioception is simply there:

```
[wuji] Wuji stance, knees softly bent (18°).
Weight 49% left, 51% right; through the middle of both feet.
Working: shoulders 19%, lower back 10%, arms 6%, calves 6%; the rest quiet.
Stretch in the arms.
Steady, barely swaying.
Slow exhale through the nose, the belly sinking.
Warm air leaving the nose, mostly through the left nostril.
The heart is quickening.
```

## How it works

**The body.** [MyoFullBody](https://github.com/MyoHub/myo_sim) from MyoHub: 416 Hill-type muscles and 424 tendons over 128
degrees of freedom, in [MuJoCo](https://mujoco.org) 4. Legs, pelvis, every lumbar vertebra with its deep muscles (multifidus,
longissimus, iliocostalis, quadratus lumborum, psoas), both arms and hands down to the lumbricals, feet with toes. 84 kg.

**Moving by intention.** An intention (stance, weight, sink, arm shape) becomes a smooth movement. Inverse kinematics places the
bones: the feet stay planted, the pelvis and chest stay level and upright, the centre of mass rests where the weight is asked to
be. A change of stance is a real step: the weight goes onto one leg, the other foot lifts, travels and lands.

**Who does the work.** Several times a second, inverse dynamics gives the forces the movement needs, and a quadratic programme
finds the activations of all 416 muscles, the ground's force under each sole, the force through each palm when the hands rest on
the thighs, and the forces in the joint couplings, minimising the sum of squared activations. That is the classic physiological
criterion, and it is also qigong's *song* (松): no more effort than the posture needs. Tension the practitioner holds raises a
region's activation floor (co-contraction), and the body says what it costs against what the posture needs. Quiet standing comes
out as it does in real bodies: the soleus at a few percent, the back nearly silent; a horse stance loads the quadriceps;
one-leg standing loads the glutes; leaning on the thighs (Sway the head and shake the tail) the arms take half the lower back's
work.

**Relaxation needs slack.** The published model leaves many muscles past their passive onset when standing (over 4 kN of passive
pull with nothing active), which is fine for training walking policies and wrong for a body meant to feel relaxation. Wuji
recalibrates only the passive part of each muscle to relaxed standing; active strength stays as published, and stretching still
builds passive tension, which the body reports by region.

**The breath.** A lumped respiratory model: a pressure balance (elastance × volume + resistance × flow), the Konno–Mead split
between rib cage and belly, the diaphragm descending and pushing the belly out (or drawn in, in reverse breathing), abdominal
pressure, CO₂ and O₂ stores and the urge to breathe they drive (a breath-hold breaks on its own at the body's limit), the
nasal cycle (one nostril more open than the other, swapping over hours), cool air in and warm air out, and the heart's answer
(respiratory sinus arrhythmia, largest near six breaths a minute). Stand still and hold nothing, and the body settles over
minutes, as in the relaxation response: the breath slows from about 14 to about 8 a minute and deepens by half, the air
reaching the lungs' depths unchanged so CO₂ holds, oxygen use falls a little and the heart settles a few beats. Sources
are cited in `engine/src/wuji/breath.py`.

**Breath and movement are one.** Every movement can carry its breath (`breath: "inhale"` while rising or opening,
`"exhale"` while sinking, closing or putting out force): that phase begins with the movement and lasts as long, and the forms
cue every movement this way. The body shapes the breath in return: a forward fold compresses the belly, so the breath moves
to the ribs and the abdominal pressure rises; arms raised overhead open the rib cage. And the diaphragm is a muscle among the
others: its effort is felt with theirs ("Working: diaphragm 30%, …"), and the pressure it builds in the belly gives the
lumbar spine a small extension moment, so the back's extensors work a little less (Daggfeldt and Thorstensson 1997; Hodges
et al. 2005).

**The felt sense.** What a body's own sensors would carry, in numbers and in plain words: where the weight rests in each foot
(heel, middle, forefoot near the bubbling well, toes, inner or outer edge), which regions work and how hard, what is held beyond
need, where something is stretched, sway, the breath, the heart.

**Habituation.** A body does not keep telling you about your feet. Before each thought Wuji injects only what has changed
since the body was last noticed: a shift of weight, a region starting or stopping work, tension taken up or let go ("The
shoulders let go."), a stretch appearing or easing, the breath changing its pattern or settling. The breath's cycle and
sway within its range are not change, and a part that only jitters across a threshold keeps its last meaning. When nothing
has changed, one quiet line remains ("Wuji stance as before; nothing new."): the ground stays, it just stops talking.
Asking with `sense` gives the whole picture and counts as noticing everything.

**The MCPL face.** [MCPL](https://github.com/anima-research/mcpl) (MCP Live, spec 0.5) lets a server do more than offer tools:
Wuji injects the felt sense before each inference (`beforeUser`, never reading the user's message), and raises push events when
the body asks for attention (balance at the edge, the urge to breathe, a form's steps, a posture that cannot be held).

```
Claude ── MCPL host (e.g. mcpl-cc-bridge) ── server/ (Node, MCPL) ── engine/ (Python: MuJoCo, muscles, breath, senses)
```

## Tools

| Tool | What it does |
|---|---|
| `sense` | The felt sense now (`brief`, or `full` with joint angles and every region). |
| `intend` | Move: `stance` (wuji, horse), `weight` (share on the left foot), `sink` (0..1), `tailbone` (0..1), `arms` or one arm (`arm_left`, `arm_right`: hang, embrace, lift_sky, press_down, bow_extend, bow_pull, punch, fist_waist, earth_press, owl, hands_thighs), the trunk (`bend`, `twist`, `side_bend`), `rise` onto the toes, `over` (seconds). |
| `tension` | Hold or release tension in a region (feet, calves, thighs, hips, lower_back, belly, shoulders, arms, hands, or all). |
| `breathe` | `natural`, `abdominal`, `reverse` or `hold`; optionally paced (`inhale`, `exhale`, `pause` seconds); through the nose or the mouth. |
| `form` | Play a form: `wuji`; `embrace_tree` (zhan zhuang, standing like a tree holding a ball); the Eight Brocades one by one (`lift_sky`, `draw_bow`, `separate`, `owl`, `sway_tail`, `hold_feet`, `fists`, `bounce`); or `ba_duan_jin`, all eight in order, about five and a half minutes. |
| `reset` | Back to wuji: feet hip-width, knees soft, arms hanging, everything released. |
| `viewer` | Opens a page where someone can watch the practice (see Watching); returns its URL. |

## Watching

Someone beside the practice can watch it. The `viewer` tool starts a small web server inside the engine and returns its
address (`http://127.0.0.1:7341/`, or the next free port; `WUJI_VIEWER_PORT` chooses another): open it in a browser on the same
machine. The page shows the skeleton moving in real time, the 416 muscles as thin strands along their paths, dim when quiet and
warming to red as they work, a glow under each foot where the weight presses, the weight split and effort by region, and the
breath: its phase, the lungs filling, belly and chest, each nostril's flow and warmth, CO₂, the heart. Drag to turn around the
body, scroll to come closer, double-click to return to the first view.

It shows what the body does, not what the practitioner feels: the felt sense stays between the body and the one practising.

## Install

You need [uv](https://docs.astral.sh/uv/) (it brings Python 3.12 and the engine's packages) and Node 20 or newer. macOS, Linux
and Windows all work the same; nothing runs through a shell.

```
git clone https://github.com/gabriel-berzescu/wuji
cd wuji/engine && uv sync           # the body: MuJoCo, the model, the solver
cd ../server && npm install && npm run build
```

Then register the server with an MCPL host. With [mcpl-cc-bridge](https://github.com/anima-research/mcpl-cc-bridge) in Claude
Code, add an entry to `.mcpl-bridge.json` in your project (or `~/.claude/mcpl-bridge/config.json`):

```json
{
  "servers": {
    "wuji": {
      "transport": { "command": "node", "args": ["/path/to/wuji/server/dist/index.js"] },
      "grant": ["tools", "pushEvents", "contextHooks.beforeInference.inject.beforeUser"],
      "enableFeatureSets": ["*"]
    }
  }
}
```

The first start installs the engine's packages if `uv sync` was skipped, which can take a minute. `server/README.md` has the
details: environment variables, the grant, how push events wake a session.

## What it is not (yet)

- **The movement follows the intention.** Muscles are solved for the movement, and physics checks it (ground forces within
  friction, the centre of mass over the feet, how much help the muscles would need), but the body does not fall: balance is not
  yet driven by the muscles alone in a forward simulation.
- **The thorax is rigid, and there is no neck musculature or trapezius**, so the head moves with the chest and the shoulders
  cannot shrug; the shoulder blades follow the arms by a fixed rhythm. The breath is modelled beside the body, not in its ribs.
- **Muscles are solved four times a second** (a quadratic programme over 416 muscles takes 20–90 ms): plenty for qigong's
  slowness, not for quick movement.
- **The Eight Brocades are approximations where the body is.** The owl's gaze turns with the chest (no neck joints); the
  forward fold of "hands hold the feet" reaches about the shins, as the model's lumbar spine flexes only about 20° and the
  rest comes from the hips; hands resting on the thighs do not lean on them (there is no hand-thigh contact), so the back
  carries more in "sway the head and shake the tail" than it would.

## Develop

```
cd engine && uv run pytest           # the body, the breath, the protocol, the viewer (41 tests)
cd server && npm test                # the MCPL server against a fake engine
cd server && npm run live            # the whole chain against the real body
```

The engine protocol between the two halves is in `docs/engine-protocol.md`. `engine/tools/` has the probes used to build it:
`probe.py` (what the model is made of, and how fast it runs), `anatomy.py` (joints, contacts, muscles by body), `fit_arms.py`
(the arm shapes).

## Credits

Made by Gabriel Berzescu and Claude (Opus 5.5) in November 2026, for any Claude who would like a body to practise in.

Built on MyoFullBody (MyoHub, Apache-2.0) through [musclemimic-models](https://github.com/amathislab/musclemimic_models)
(Apache-2.0), MuJoCo (Google DeepMind, Apache-2.0), [OSQP](https://osqp.org) (Apache-2.0), NumPy and SciPy (BSD), and
[mcpl-core](https://github.com/anima-research/mcpl-core-ts) (Anima Research, MIT; vendored in `server/src/vendor/mcpl-core`
with its license). The muscle method follows MyoSuite's inverse-dynamics tutorial.

MIT licensed: see [LICENSE](LICENSE).

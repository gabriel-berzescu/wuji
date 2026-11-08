# wuji-server

Wuji's MCPL face. It runs the engine (the simulated body, `../engine`) as a child process and offers
it to Claude through an MCPL host such as [mcpl-cc-bridge](https://github.com/anima-research/mcpl-cc-bridge):

- **Tools**: `sense`, `intend`, `tension`, `breathe`, `form`, `reset`, `viewer`, one per engine operation
  (see `../docs/engine-protocol.md`). Results are the engine's prose first, then the rest of its
  answer as compact JSON.
- **Felt sense** (`wuji.feltSense`): before each turn, a brief sense of the body (six to ten lines)
  is injected before the user message, tagged `[wuji]`. Proprioception you never have to ask for.
  It never reads the user's message, and it never delays a turn: if the engine has not answered
  within 1.5 s, nothing is injected.
- **Body signals** (`wuji.signals`): the engine's `balance`, `breath`, `form` and `strain` events
  arrive as `push/event`s, tagged `wuji:<event>` (and `wuji:warning` / `wuji:notice` for balance).

MCPL spec 0.5 (`capabilities.experimental.mcpl`). MIT.

## Requirements

Node 20 or newer, and [uv](https://docs.astral.sh/uv/) for the engine (or see `WUJI_ENGINE`).
Works the same on macOS, Linux and Windows: nothing goes through a shell.

## Build and test

```
cd server
npm install
npm run build
npm test
```

The tests run against `test/fake-engine.mjs`, a small stand-in that speaks the engine protocol
with canned answers, so they need neither Python nor MuJoCo.

Run it by hand (it speaks MCPL on stdin/stdout and logs to stderr):

```
node dist/index.js
```

## Register it with mcpl-cc-bridge

Add an entry to `.mcpl-bridge.json` in your project (or `~/.claude/mcpl-bridge/config.json`).
Use absolute paths; on Windows, forward slashes work.

```json
{
  "servers": {
    "wuji": {
      "transport": {
        "command": "node",
        "args": ["/Users/gabriel/git-repos/wuji/server/dist/index.js"]
      },
      "grant": ["tools", "pushEvents", "contextHooks.beforeInference.inject.beforeUser"]
    }
  }
}
```

The tools then appear in Claude Code as `wuji__sense`, `wuji__intend`, and so on. Body signals
wake the session like any channel delivery, which needs the bridge's channel lane
(`--dangerously-load-development-channels plugin:mcpl-bridge@mcpl-bridge-dev` while the plugin is
unpublished); the tools and the felt sense work without it.

### Grants

| Path | What it enables | Without it |
|---|---|---|
| `tools` | the seven tools | the tools still answer: they are plain MCP and need no MCPL grant |
| `contextHooks.beforeInference.inject.beforeUser` | `wuji.feltSense` | no felt sense before turns |
| `pushEvents` | `wuji.signals` | no body signals |

The server advertises only the `beforeUser` injection leaf, not `observe`, so even the bridge's
default grant (which includes `observe`) never hands it the user's text. Absence is denial: until
the host's first `featureSets/update` Request arrives, and for anything that Request leaves out,
the feature set stays off. A malformed policy grants nothing; a `featureSets/update` Notification
can only narrow. The receipt names each disabled feature set and what is missing.

## Configuration (environment)

| Variable | Default | |
|---|---|---|
| `WUJI_ENGINE` | `["uv","run","--project","<repo>/engine","python","-m","wuji.engine"]` | the engine command, as a JSON array (no shell). Name an executable (`python3`, `C:/…/python.exe`), not a `.cmd` or `.bat` |
| `WUJI_SENSE_BUDGET_MS` | `1500` | how long the felt sense may wait for the engine (the bridge gives hooks 5 s) |
| `WUJI_FELT_SENSE` | `ambient` | what is injected before each turn: `ambient`, only what changed since the body was last noticed (one quiet line when nothing did), or `brief`, the whole picture every time |
| `WUJI_TOOL_TIMEOUT_MS` | `20000` | how long a tool call may wait for the engine, including for it to start |
| `WUJI_ENGINE_START_TIMEOUT_MS` | `300000` | how long a new engine has to answer `hello` (a first `uv run` may install packages) |
| `WUJI_PUSH_MIN_INTERVAL_MS` | `1000` | at most one signal of each kind per interval; a newer one waiting replaces an older |
| `WUJI_PUSH_MAX_PER_MINUTE` | `40` | across all kinds (a form calls at every movement, one every few seconds; at most about 25 a minute, in the bounce) |
| `WUJI_RESTART_MIN_MS` / `WUJI_RESTART_MAX_MS` | `500` / `30000` | engine restart backoff (doubling; reset after a minute of uptime) |

The bridge's `transport.env` sets these for one server.

## How it works

- **Engine**: started at launch, sent `hello` first; requests carry ids and responses are matched by
  id. If it dies, the requests it held fail (the tool result says so), and it is restarted with
  backoff and greeted again; tool calls made meanwhile wait for it. On exit (host gone, SIGINT,
  SIGTERM) the server sends `shutdown`, closes the engine's stdin, and kills it if it has not gone
  within 3 s.
- **Push events**: `featureSet: "wuji.signals"`, `eventId` `wuji-<run>-<event>-<t>` (stable for one
  engine event: the engine run, the kind, the simulated time), `origin` with the event's fields,
  `payload.content` the event's `text`.

## mcpl-core

`src/vendor/mcpl-core/` is a copy of [`@animalabs/mcpl-core`](https://github.com/anima-research/mcpl-core-ts)'s
`src/` at main (commit and version in `VENDOR.json`), compiled by this package's `tsc`, the way
mcpl-cc-bridge vendors it. So the server has no runtime dependencies, installs anywhere without
access to anima-research's packages, and runs the library's main tip rather than the last
published release. To refresh it from a checkout beside the wuji repo:

```
npm run vendor:mcpl-core            # from ../../mcpl-core-ts
npm run vendor:mcpl-core -- <path>  # or from elsewhere (or set MCPL_CORE_TS)
```

Then build and test.

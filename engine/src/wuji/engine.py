"""The engine process: the body and the breath in real time, spoken to over JSON Lines on stdio (docs/engine-protocol.md).

    uv run python -m wuji.engine

Bones are placed 25 times a second, the muscles re-solved four times a second, the breath stepped every frame. Requests are
answered between frames; events are rare by design.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time

import numpy as np

from . import __version__
from .anatomy import REGION_ORDER
from .body import Body
from .senses import Senses

FRAME = 1 / 25
MUSCLE_EVERY = 0.25

FORMS = {
    # name: [(seconds after start, operation, args)], each a step of the form
    "wuji": [
        (0, "intend", {"stance": "wuji", "weight": 0.5, "sink": 0.25, "arms": "hang", "over": 4}),
        (0, "breathe", {"pattern": "natural"}),
        (4, "tension", {"region": "all", "level": 0}),
    ],
    "embrace_tree": [  # zhan zhuang: standing like a tree, holding a ball
        (0, "intend", {"stance": "wuji", "sink": 0.4, "over": 4}),
        (4, "intend", {"arms": "embrace", "over": 5}),
        (9, "breathe", {"pattern": "abdominal", "inhale": 4, "exhale": 6}),
    ],
    "lift_sky": [  # Ba Duan Jin, first brocade: two hands hold up the heavens, three times
        (0, "intend", {"stance": "wuji", "sink": 0.3, "arms": "hang", "over": 3}),
        (0, "breathe", {"pattern": "natural", "inhale": 4, "exhale": 4}),
        (3, "intend", {"arms": "lift_sky", "sink": 0.05, "over": 4}),
        (7, "intend", {"arms": "press_down", "sink": 0.3, "over": 4}),
        (11, "intend", {"arms": "lift_sky", "sink": 0.05, "over": 4}),
        (15, "intend", {"arms": "press_down", "sink": 0.3, "over": 4}),
        (19, "intend", {"arms": "lift_sky", "sink": 0.05, "over": 4}),
        (23, "intend", {"arms": "hang", "sink": 0.25, "over": 4}),
        (27, "breathe", {"pattern": "natural"}),
    ],
}


def _load_breath():
    try:
        from .breath import Breath  # noqa: PLC0415
        return Breath()
    except Exception as e:  # the breath is built separately; the body can stand without it
        print(f"[wuji] no breath model ({e}); breathing is silent", file=sys.stderr)
        return None


class Engine:
    def __init__(self, out=sys.stdout) -> None:
        self.out = out
        self.lock = threading.Lock()
        self.body = Body()
        self.senses = Senses(self.body)
        self.breath = _load_breath()
        self.form: dict | None = None
        self.running = True
        self.last_event: dict[str, float] = {}
        self.strain_since: float | None = None
        self.since_muscles = 0.0

    # ---------- output ----------

    def send(self, obj: dict) -> None:
        line = json.dumps(obj, separators=(",", ":"), default=_jsonable)
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()

    def event(self, kind: str, data: dict, every: float = 10.0) -> None:
        """An event, unless one of its kind (and severity) went out less than `every` seconds ago."""
        key = kind + str(data.get("severity", data.get("kind", "")))
        if self.body.t - self.last_event.get(key, -1e9) < every:
            return
        self.last_event[key] = self.body.t
        self.send({"event": kind, "t": round(self.body.t, 2), "data": data})

    # ---------- the frame ----------

    def frame(self, dt: float) -> None:
        b = self.body
        b.step(dt)
        self.senses.note_sway()
        self.since_muscles += dt
        if self.since_muscles >= MUSCLE_EVERY:
            b.solve_muscles(self.since_muscles)
            self.since_muscles = 0.0
            self.watch()
        if self.breath is not None:
            regions = self.senses_effort()
            for ev in self.breath.step(dt, effort=min(1.0, 8 * regions), belly_tension=b.tension["belly"]) or []:
                self.event(ev.get("event", "breath"), ev.get("data", {}), every=5.0)
            st = self.breath.state()
            # the breath moves the body: the trunk sways a little forward and back with each breath (mm)
            b.breath_shift = -0.0015 * (st.get("lung_volume_l", 3.0) - 2.75)
        self.run_form()

    def senses_effort(self) -> float:
        s = self.body.solution
        a = self.body.a
        return float((s.act * a.f0).sum() / a.f0.sum()) if s is not None else 0.0

    def watch(self) -> None:
        """Balance and strain: the body asking for attention."""
        b, s = self.body, self.body.solution
        if s is None:
            return
        fz = {k: max(0.0, float(v["force"][2])) for k, v in s.grf.items()}
        total = sum(fz.values()) or 1.0
        for side, f in fz.items():
            if f / total < 0.2 or not b.loaded.get(side, True):
                continue
            lo, hi = b.posture_solver.heel_to_toe[side]
            frac = float((s.grf[side]["cop"] @ np.array([0.0, -1.0, 0.0]) - lo) / (hi - lo))
            if frac < 0.1 or frac > 0.88:
                where = "heel" if frac < 0.1 else "toes"
                self.event("balance", {"severity": "warning", "text": f"Weight on the very edge of the {side} {where}: close to tipping."})
        strained = s.root_residual > 20 or s.joint_residual > 10
        if strained and self.strain_since is None:
            self.strain_since = b.t
        elif not strained:
            self.strain_since = None
        if self.strain_since is not None and b.t - self.strain_since > 1.0:
            self.event("strain", {"text": "This posture asks more than the muscles can give; something has to change."}, every=20.0)

    def run_form(self) -> None:
        f = self.form
        if f is None:
            return
        steps = FORMS[f["name"]]
        while f["next"] < len(steps) and self.body.t - f["t0"] >= steps[f["next"]][0]:
            _, op, args = steps[f["next"]]
            text = self.apply(op, args)
            f["next"] += 1
            self.event("form", {"name": f["name"], "step": f["next"], "of": len(steps), "done": False, "text": text}, every=0.0)
        if f["next"] >= len(steps) and not self.body.moving:
            self.event("form", {"name": f["name"], "step": len(steps), "of": len(steps), "done": True,
                                "text": f"The form {f['name'].replace('_', ' ')} is complete."}, every=0.0)
            self.form = None

    # ---------- requests ----------

    def apply(self, op: str, args: dict) -> str:
        b = self.body
        if op == "intend":
            return b.intend(stance=args.get("stance"), weight=args.get("weight"), sink=args.get("sink"),
                            arms=args.get("arms"), over=args.get("over", 3.0)) + _head_note(args.get("head"))
        if op == "tension":
            return b.set_tension(args.get("region", "all"), float(args.get("level", 0.0)))
        if op == "breathe":
            if self.breath is None:
                return "The breath is not modelled in this build."
            return self.breath.set_pattern(args.get("pattern", "natural"), inhale=args.get("inhale"), exhale=args.get("exhale"),
                                           pause=args.get("pause"), hold=args.get("hold"), nose=args.get("nose", True))
        raise ValueError(f"unknown operation {op!r}")

    def handle(self, req: dict) -> dict:
        op, args = req.get("op"), req.get("args") or {}
        b = self.body
        if op == "hello":
            return {"version": __version__, "model": "myofullbody", "muscles": b.m.nu, "t": round(b.t, 2)}
        if op == "sense":
            detail = args.get("detail", "brief")
            felt = self.breath.felt(detail) if self.breath is not None else None
            return self.senses.read(self.breath.state() if self.breath is not None else None, felt, detail)
        if op in ("intend", "tension", "breathe"):
            self.form = None if op == "intend" else self.form
            return {"accepted": True, "text": self.apply(op, args), **({"arrives_in": args.get("over", 3.0)} if op == "intend" else {})}
        if op == "form":
            name = args.get("name")
            if name not in FORMS:
                raise ValueError(f"unknown form {name!r} (known: {', '.join(FORMS)})")
            self.form = {"name": name, "t0": b.t, "next": 0}
            return {"accepted": True, "steps": len(FORMS[name]), "text": f"Beginning {name.replace('_', ' ')}."}
        if op == "reset":
            self.form = None
            text = b.reset()
            if self.breath is not None:
                self.breath.set_pattern("natural")
            return {"accepted": True, "text": text}
        if op == "shutdown":
            self.running = False
            return {"ok": True}
        raise ValueError(f"unknown operation {op!r}")

    def serve(self, lines: "queue.Queue[str | None]") -> None:
        last = time.monotonic()
        while self.running:
            start = time.monotonic()
            dt = min(0.1, start - last)
            last = start
            self.frame(dt)
            while True:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    break
                if line is None:  # stdin closed: the server is gone
                    self.running = False
                    break
                self.respond(line)
            time.sleep(max(0.0, FRAME - (time.monotonic() - start)))

    def respond(self, line: str) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[wuji] bad request line: {e}", file=sys.stderr)
            return
        try:
            self.send({"id": req.get("id"), "ok": True, "result": self.handle(req)})
        except Exception as e:  # an error in a request answers that request; the body goes on
            self.send({"id": req.get("id"), "ok": False, "error": str(e)})


def _head_note(head) -> str:
    if head in (None, "suspended"):
        return ""
    return " (This body has no neck joints: the head stays aligned with the chest.)"


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(f"not JSON: {type(x)}")


def main() -> None:
    lines: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            if line.strip():
                lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    engine = Engine()
    print(f"[wuji] engine {__version__} ready: {engine.body.m.nu} muscles", file=sys.stderr)
    engine.serve(lines)


if __name__ == "__main__":
    main()

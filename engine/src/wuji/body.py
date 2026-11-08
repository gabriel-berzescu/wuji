"""The living body: where it is going, how it moves there, and what its muscles are doing about it, moment by moment.

An intention (stance, weight, sink, arms) becomes a smooth movement (minimum jerk). A change of stance is a real step: the
weight goes onto the staying leg, the other foot lifts, travels and is set down, and the weight comes back. Every frame the
posture solver places the bones; a few times a second the muscle solver works out who carries the load. Holding tension in a
region raises its muscles' activation floor (co-contraction), and the cost of it is measured against what the posture needs.
Quiet standing is never still: the centre of mass wanders a few millimetres, and the breath moves it too.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .anatomy import Anatomy, REGION_ORDER, calibrate_rest, load_model
from .muscles import MuscleSolver, Solution
from .posture import ARMS, STANCES, Posture, PostureSolver

TENSION_FLOOR = 0.15  # activation a fully held region keeps (co-contraction), on top of what the posture needs
FOOT_LIFT = 0.06  # how high a stepping foot rises (m)


def _min_jerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class _Segment:
    """One smooth movement: from a pose to a pose, over a duration. A step has three phases."""
    t0: float
    duration: float
    start: dict
    end: dict
    step_foot: str | None = None  # the foot that lifts, when the stance changes


class Body:
    def __init__(self, seed: int = 0) -> None:
        self.m = load_model()
        self.a = Anatomy(self.m)
        self.posture_solver = PostureSolver(self.a)
        calibrate_rest(self.m, self.posture_solver.solve(Posture(sink=0.0), iterations=60))
        self.muscles = MuscleSolver(self.a)
        self.rng = np.random.default_rng(seed)

        self.posture = Posture(stance="wuji", weight=0.5, sink=0.25, arms="hang")
        self.pose = self._pose_of(self.posture)
        self.segments: list[_Segment] = []
        self.tension = {r: 0.0 for r in REGION_ORDER}
        self.t = 0.0
        self.sway = np.zeros(2)
        self._sway_v = np.zeros(2)
        self.breath_shift = 0.0  # set by the breath, metres along FORWARD

        self.q = self.posture_solver.solve(self.posture, feet=self.pose["feet"], knee_deg=self.pose["knee"],
                                           arms=self.pose["arms"], iterations=80)
        self.q_prev = self.q.copy()
        self.qvel = np.zeros(self.m.nv)
        self.qvel_prev = np.zeros(self.m.nv)
        self.loaded = {"left": True, "right": True}
        self.solution: Solution | None = None
        self.minimum: Solution | None = None  # the same moment without held tension
        self._minimum_at = -1e9
        self.minimum_effort: float | None = None
        self.solve_muscles()

    # ---------- intentions ----------

    def _pose_of(self, p: Posture, feet: dict | None = None) -> dict:
        return {"weight": p.weight, "knee": p.sink * STANCES[p.stance]["max_knee"], "arms": dict(ARMS[p.arms]),
                "feet": dict(feet or self.posture_solver.feet_for(p.stance)), "lift": {"left": 0.0, "right": 0.0}}

    def intend(self, stance=None, weight=None, sink=None, arms=None, over: float = 3.0) -> str:
        """Plans the movement to a new posture; returns what will happen, in a sentence."""
        if stance is not None and stance not in STANCES:
            raise ValueError(f"unknown stance {stance!r} (known: {', '.join(STANCES)})")
        if arms is not None and arms not in ARMS:
            raise ValueError(f"unknown arm shape {arms!r} (known: {', '.join(ARMS)})")
        if weight is not None and not 0.0 <= weight <= 1.0:
            raise ValueError("weight is the share on the left foot, 0..1")
        if sink is not None and not 0.0 <= sink <= 1.0:
            raise ValueError("sink is 0 (straight) .. 1 (the deepest the stance allows)")
        over = max(0.5, float(over))
        target = self.posture.merged(stance=stance, weight=weight, sink=sink, arms=arms)
        start = self._current_pose()
        self.segments.clear()
        t = self.t
        if target.stance != self.posture.stance:
            new_feet = self.posture_solver.feet_for(target.stance)
            # step the left foot first, then the right: each step shifts the weight away, lifts, places, returns; then
            # the body settles into the new stance. The steps share 75% of the time asked for, the settling the rest.
            steps = [s for s in ("left", "right") if abs(new_feet[s] - start["feet"][s]) >= 1e-3]
            feet = dict(start["feet"])
            pose = start
            for side in steps:
                feet[side] = new_feet[side]
                end = {**pose, "feet": dict(feet), "weight": 0.5}
                self.segments.append(_Segment(t, 0.75 * over / len(steps), pose, end, step_foot=side))
                t += 0.75 * over / len(steps)
                pose = end
            end = self._pose_of(target, feet=new_feet)
            self.segments.append(_Segment(t, 0.25 * over, pose, end))
            what = f"stepping into {target.stance}"
        else:
            self.segments.append(_Segment(t, over, start, self._pose_of(target)))
            what = "moving"
        self.posture = target
        parts = []
        if weight is not None:
            parts.append(f"weight to {round(weight * 100)}% left")
        if sink is not None:
            parts.append(f"sinking to {sink:.0%} of the stance's depth")
        if arms is not None:
            parts.append(f"arms to {arms.replace('_', ' ')}")
        return f"{what.capitalize()}{': ' + ', '.join(parts) if parts else ''}, over {over:g} seconds."

    def set_tension(self, region: str, level: float) -> str:
        level = float(min(1.0, max(0.0, level)))
        regions = REGION_ORDER if region == "all" else (region,)
        for r in regions:
            if r not in self.tension:
                raise ValueError(f"unknown region {region!r} (known: {', '.join(REGION_ORDER)}, all)")
            self.tension[r] = level
        self.minimum = None
        if level == 0:
            return "Releasing everything." if region == "all" else f"Releasing the {region.replace('_', ' ')}."
        return f"Holding the {region.replace('_', ' ')} at {level:.0%} extra tension."

    def reset(self) -> str:
        self.set_tension("all", 0.0)
        self.intend(stance="wuji", weight=0.5, sink=0.25, arms="hang", over=3.0)
        return "Returning to wuji: feet hip-width, knees soft, arms hanging, everything released."

    # ---------- time ----------

    def _current_pose(self) -> dict:
        return self._pose_at(self.t)

    def _pose_at(self, t: float) -> dict:
        while self.segments and t >= self.segments[0].t0 + self.segments[0].duration:
            self.pose = self.segments.pop(0).end
        if not self.segments or t < self.segments[0].t0:
            return self.pose
        seg = self.segments[0]
        s = (t - seg.t0) / seg.duration
        if seg.step_foot is None:
            return _blend(seg.start, seg.end, _min_jerk(s))
        # a step: weight off the stepping foot (0-35%), the foot travels in an arc (35-65%), weight back (65-100%)
        side = seg.step_foot
        off = {**seg.start, "weight": 0.0 if side == "left" else 1.0}
        if s < 0.35:
            return _blend(seg.start, off, _min_jerk(s / 0.35))
        if s < 0.65:
            u = _min_jerk((s - 0.35) / 0.3)
            pose = {**off, "feet": dict(seg.start["feet"]), "lift": {"left": 0.0, "right": 0.0}}
            pose["feet"][side] = seg.start["feet"][side] + u * (seg.end["feet"][side] - seg.start["feet"][side])
            pose["lift"][side] = FOOT_LIFT * np.sin(np.pi * u)
            return pose
        landed = {**off, "feet": dict(seg.end["feet"])}
        return _blend(landed, seg.end, _min_jerk((s - 0.65) / 0.35))

    @property
    def moving(self) -> bool:
        return bool(self.segments)

    def step(self, dt: float) -> None:
        """Advances the body dt seconds: sway, pose, bones (the muscles are solved separately, less often)."""
        if dt < 1e-4:
            return
        self.t += dt
        # postural sway: a slow random walk pulled back to centre (a few mm, as in quiet standing), plus the breath
        k, c = 0.6, 1.2
        self._sway_v += (-k * self.sway - c * self._sway_v) * dt + self.rng.normal(0, 0.004, 2) * np.sqrt(dt)
        self.sway += self._sway_v * dt
        pose = self._pose_at(self.t)
        now = Posture(stance=self.posture.stance, weight=pose["weight"], sink=0.0, arms=self.posture.arms)
        sway = np.array([0.5 * self.sway[0], self.sway[1] + self.breath_shift])  # less side to side than front to back
        self.q_prev = self.q
        self.q = self.posture_solver.solve(now, feet=pose["feet"], lift=pose["lift"], knee_deg=pose["knee"],
                                           arms=pose["arms"], iterations=6, sway=sway)
        self.loaded = {s: pose["lift"][s] < 0.005 for s in ("left", "right")}
        qvel = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, qvel, dt, self.q_prev, self.q)
        self.qvel_prev, self.qvel = self.qvel, 0.7 * self.qvel + 0.3 * qvel  # light smoothing of numerical velocity
        self.current = pose

    def solve_muscles(self, dt: float = 0.05) -> Solution:
        """Who carries the load right now (a few times a second)."""
        qacc = (self.qvel - self.qvel_prev) / max(dt, 1e-3)
        floor = np.array([TENSION_FLOOR * self.tension[r] for r in self.a.muscle_region])
        solution = self.muscles.solve(self.q, self.qvel, qacc, floor=floor, loaded=getattr(self, "loaded", None))
        if not solution.ok and self.solution is not None:
            return self.solution  # a moment the solver could not settle: keep the last good one
        self.solution = solution
        # with tension held, what would the posture need on its own? (solved again, at most once a second)
        if not any(self.tension.values()):
            self.minimum = solution
        elif self.minimum is None or self.t - self._minimum_at > 1.0:
            prev = self.muscles.prev.copy()
            minimum = self.muscles.solve(self.q, self.qvel, qacc, loaded=self.loaded)
            self.muscles.prev = prev
            if minimum.ok:
                self.minimum, self._minimum_at = minimum, self.t
        self.minimum_effort = self.minimum.effort if self.minimum is not None else None
        return self.solution


def _blend(a: dict, b: dict, s: float) -> dict:
    lerp = lambda x, y: x + s * (y - x)
    return {
        "weight": lerp(a["weight"], b["weight"]),
        "knee": lerp(a["knee"], b["knee"]),
        "arms": {k: lerp(a["arms"][k], b["arms"][k]) for k in a["arms"]},
        "feet": {k: lerp(a["feet"][k], b["feet"][k]) for k in a["feet"]},
        "lift": {k: lerp(a["lift"][k], b["lift"][k]) for k in a["lift"]},
    }

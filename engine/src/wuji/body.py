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
from .passive import Felt, PassiveTissues
from .posture import ARMS, RISE_MAX, STANCES, TRUNK_MAX, Posture, PostureSolver, pelvic_tilt

# Abdominal pressure above rest pushes the lumbar spine toward extension: about 0.15 N m per cmH2O here (the pressure acts
# on roughly 250 cm2 of the abdominal cavity's roof, a few centimetres in front of the spine; Daggfeldt and Thorstensson
# 1997 measured the moment it can make; Hodges, Eriksson, Shirley and Gandevia 2005 its effect on spinal stiffness).
IAP_MOMENT = 0.15  # N m per cmH2O above PAB_REST
PAB_REST = 8.0

TENSION_FLOOR = 0.15  # activation a fully held region keeps (co-contraction), on top of what the posture needs
FOOT_LIFT = 0.06  # how high a stepping foot rises (m)
STEP_WIDE = (0.25, 0.60)  # a step landing the feet this far apart (m) sinks on the staying leg: not at all .. fully
STEP_SINK, STEP_REACH = 55.0, 5.0  # the staying knee's bend and the reaching knee's, in the widest step (deg)
HAND_CONTACT = 0.03  # a resting palm this close to its thigh's surface (m) is on it, and can press: the fingers reach 8-10 cm
# beyond the palm's middle, so a palm a few centimetres off still touches (the far hand at the bottom of the sway's lean)


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
        self.tissues = PassiveTissues(self.a)
        self.passive_felt: list[Felt] = []
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
        self.abdominal_pressure = PAB_REST  # set by the breath, cmH2O

        self.q = self.posture_solver.solve(self.posture, feet=self.pose["feet"], knee_deg=self.pose["knee"],
                                           arms=self.pose["arms_l"], arms_right=self.pose["arms_r"], iterations=80,
                                           rest=self.pose["rest"])
        self.q_prev = self.q.copy()
        self.qvel = np.zeros(self.m.nv)
        self.qvel_prev = np.zeros(self.m.nv)
        self.loaded = {"left": True, "right": True}
        self.hands = self._hands_on_thighs(self.pose)
        self.solution: Solution | None = None
        self.minimum: Solution | None = None  # the same moment without held tension
        self._minimum_at = -1e9
        self.minimum_effort: float | None = None
        self.solve_muscles()

    # ---------- intentions ----------

    def _pose_of(self, p: Posture, feet: dict | None = None) -> dict:
        knee = p.sink * STANCES[p.stance]["max_knee"]
        return {"weight": p.weight, "knee": knee, "tilt": pelvic_tilt(knee, p.tailbone),
                "arms_l": dict(ARMS[p.arm_left or p.arms]), "arms_r": dict(ARMS[p.arm_right or p.arms]),
                "bend": p.bend * TRUNK_MAX["bend"], "twist": p.twist * TRUNK_MAX["twist"],
                "side": p.side_bend * TRUNK_MAX["side"], "rise": p.rise * RISE_MAX,
                "feet": dict(feet or self.posture_solver.feet_for(p.stance)), "lift": {"left": 0.0, "right": 0.0},
                "rest": {s: float((getattr(p, f"arm_{s}") or p.arms) == "hands_thighs") for s in ("left", "right")}}

    def intend(self, stance=None, weight=None, sink=None, arms=None, tailbone=None, arm_left=None, arm_right=None,
               bend=None, twist=None, side_bend=None, rise=None, over: float = 3.0) -> str:
        """Plans the movement to a new posture; returns what will happen, in a sentence."""
        if stance is not None and stance not in STANCES:
            raise ValueError(f"unknown stance {stance!r} (known: {', '.join(STANCES)})")
        for name, shape in (("arms", arms), ("arm_left", arm_left), ("arm_right", arm_right)):
            if shape is not None and shape not in ARMS:
                raise ValueError(f"unknown arm shape {shape!r} for {name} (known: {', '.join(ARMS)})")
        for name, v, lo in (("bend", bend, 0.0), ("rise", rise, 0.0), ("twist", twist, -1.0), ("side_bend", side_bend, -1.0)):
            if v is not None and not lo <= v <= 1.0:
                raise ValueError(f"{name} runs {lo:g} .. 1")
        if weight is not None and not 0.0 <= weight <= 1.0:
            raise ValueError("weight is the share on the left foot, 0..1")
        if sink is not None and not 0.0 <= sink <= 1.0:
            raise ValueError("sink is 0 (straight) .. 1 (the deepest the stance allows)")
        if tailbone is not None and not 0.0 <= tailbone <= 1.0:
            raise ValueError("tailbone is 0 (as the knees bring it) .. 1 (dropped as far as it goes)")
        over = max(0.5, float(over))
        target = self.posture.merged(stance=stance, weight=weight, sink=sink, arms=arms, tailbone=tailbone,
                                     arm_left=arm_left, arm_right=arm_right, bend=bend, twist=twist, side_bend=side_bend,
                                     rise=rise)
        start = self._current_pose()
        self.segments.clear()
        t = self.t
        if target.stance != self.posture.stance:
            # as Ba Duan Jin steps: the left foot alone steps out to the new width (or back in), the right staying where
            # it is, and the body settles between the feet. (Stepping both feet out made the second step a one-legged
            # lunge, the body's weight over a foot already 30 cm out to the side: more than muscles and floor could hold.)
            # The step shifts the weight onto the right foot, lifts, places, comes back: 75% of the time asked for, the
            # settling the rest.
            width = 2 * STANCES[target.stance]["half_width"]
            new_feet = {"right": start["feet"]["right"], "left": start["feet"]["right"] + width}
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
        if tailbone is not None:
            parts.append("the tailbone dropping, the waist lengthening" if tailbone > 0 else "the pelvis returning to neutral")
        for side, shape in (("left", arm_left), ("right", arm_right)):
            if shape is not None:
                parts.append(f"the {side} arm to {shape.replace('_', ' ')}")
        if bend is not None:
            parts.append(f"folding forward {bend * TRUNK_MAX['bend']:.0f}°" if bend > 0 else "the trunk rising upright")
        if twist is not None:
            parts.append(f"turning {abs(twist) * TRUNK_MAX['twist']:.0f}° to the {'left' if twist > 0 else 'right'}" if twist
                         else "turning back to the front")
        if side_bend is not None:
            parts.append(f"leaning {abs(side_bend) * TRUNK_MAX['side']:.0f}° to the {'left' if side_bend > 0 else 'right'}"
                         if side_bend else "the trunk upright again")
        if rise is not None:
            parts.append("rising onto the toes" if rise > 0 else "the heels coming down")
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
        # a step: weight off the stepping foot (0-35%), the foot travels in an arc (35-65%), weight back (65-100%). A wide
        # step sinks on the staying leg and reaches out with the stepping one nearly straight, as a person steps out into
        # a horse stance: otherwise the reaching foot is beyond the leg's length while the weight is over the other foot.
        side = seg.step_foot
        other = "right" if side == "left" else "left"
        span = max(abs(seg.end["feet"][side] - seg.start["feet"][other]),  # the feet's distance as the step lands,
                   abs(seg.start["feet"][side] - seg.start["feet"][other]))  # or as it lifts off (stepping back in)
        wide = float(np.clip((span - STEP_WIDE[0]) / (STEP_WIDE[1] - STEP_WIDE[0]), 0.0, 1.0))
        knee0 = seg.start["knee"]
        reaching = {other: knee0 + wide * max(0.0, STEP_SINK - knee0), side: knee0 + wide * (STEP_REACH - knee0)}
        off = {**seg.start, "weight": 0.0 if side == "left" else 1.0}
        if s < 0.35:
            u = _min_jerk(s / 0.35)
            pose = _blend(seg.start, off, u)
            return {**pose, "knees": {k: knee0 + u * (v - knee0) for k, v in reaching.items()}}
        if s < 0.65:
            u = _min_jerk((s - 0.35) / 0.3)
            pose = {**off, "feet": dict(seg.start["feet"]), "lift": {"left": 0.0, "right": 0.0}, "knees": reaching}
            pose["feet"][side] = seg.start["feet"][side] + u * (seg.end["feet"][side] - seg.start["feet"][side])
            pose["lift"][side] = FOOT_LIFT * np.sin(np.pi * u)
            return pose
        landed = {**off, "feet": dict(seg.end["feet"])}
        u = _min_jerk((s - 0.65) / 0.35)
        pose = _blend(landed, seg.end, u)
        return {**pose, "knees": {k: v + u * (seg.end["knee"] - v) for k, v in reaching.items()}}

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
                                           arms=pose["arms_l"], arms_right=pose["arms_r"], iterations=6, sway=sway,
                                           tilt_deg=pose["tilt"], rise_deg=pose["rise"],
                                           trunk={"bend": pose["bend"], "twist": pose["twist"], "side": pose["side"]},
                                           rest=pose["rest"], knees=pose.get("knees"))
        tiptoe = pose["rise"] > 3.0
        self.loaded = {s: (False if pose["lift"][s] >= 0.005 else "forefoot" if tiptoe else True) for s in ("left", "right")}
        self.hands = self._hands_on_thighs(pose)
        qvel = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, qvel, dt, self.q_prev, self.q)
        self.qvel_prev, self.qvel = self.qvel, 0.7 * self.qvel + 0.3 * qvel  # light smoothing of numerical velocity
        self.current = pose

    def _hands_on_thighs(self, pose: dict) -> dict:
        """side -> True when that hand rests on its thigh (the arm in hands_thighs, the palm on the thigh's surface)."""
        gap = self.posture_solver.hand_gap
        return {s: bool(pose["rest"][s] > 0 and abs(gap.get(s, np.inf)) < HAND_CONTACT) for s in ("left", "right")}

    def abdominal_support(self) -> np.ndarray:
        """The extension moment the belly's pressure gives the lumbar spine, as a generalised force."""
        out = np.zeros(self.m.nv)
        out[self.a.dadr["flex_extension"]] = IAP_MOMENT * max(0.0, self.abdominal_pressure - PAB_REST)
        return out

    def lumbar_flexion(self) -> float:
        """How far the lumbar spine is folded, L1 to the sacrum (degrees, flexion positive)."""
        return self.tissues.lumbar.flexion(self.a, self.q)

    def lumbar_passive(self) -> float:
        """The extension moment the back of the lumbar spine gives as it is folded (N m, at each level)."""
        return next(f.moment for f in self.tissues(self.q)[1] if f.name == self.tissues.lumbar.name)

    def external_support(self) -> np.ndarray:
        """What holds the body besides the muscles and the ground, as generalised forces: the belly's pressure and the
        passive tissues (ligaments, capsules, fascia; passive.py). Remembers what the tissues are doing (passive_felt)."""
        tau, self.passive_felt = self.tissues(self.q)
        return self.abdominal_support() + tau

    def breath_shaping(self) -> tuple[float, float]:
        """How the body shapes the breath now: (belly compression, 0..1, from folding forward; rib opening, 0..1, from
        the arms raised overhead)."""
        pose = getattr(self, "current", None) or self.pose
        fold = min(1.0, pose["bend"] / 100.0)
        elev = 0.5 * (pose["arms_l"]["shoulder_elv"] + pose["arms_r"]["shoulder_elv"])
        return 0.7 * fold, min(1.0, max(0.0, (elev - 60.0) / 100.0))

    def solve_muscles(self, dt: float = 0.05) -> Solution:
        """Who carries the load right now (a few times a second)."""
        qacc = (self.qvel - self.qvel_prev) / max(dt, 1e-3)
        floor = np.array([TENSION_FLOOR * self.tension[r] for r in self.a.muscle_region])
        external = self.external_support()
        solution = self.muscles.solve(self.q, self.qvel, qacc, floor=floor, loaded=getattr(self, "loaded", None),
                                      hands=getattr(self, "hands", None), external=external)
        if not solution.ok and self.solution is not None:
            return self.solution  # a moment the solver could not settle: keep the last good one
        self.solution = solution
        # with tension held, what would the posture need on its own? (solved again, at most once a second)
        if not any(self.tension.values()):
            self.minimum = solution
        elif self.minimum is None or self.t - self._minimum_at > 1.0:
            prev = self.muscles.prev.copy()
            minimum = self.muscles.solve(self.q, self.qvel, qacc, loaded=self.loaded, hands=self.hands, external=external)
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
        "tilt": lerp(a["tilt"], b["tilt"]),
        "arms_l": {k: lerp(a["arms_l"][k], b["arms_l"][k]) for k in a["arms_l"]},
        "arms_r": {k: lerp(a["arms_r"][k], b["arms_r"][k]) for k in a["arms_r"]},
        "bend": lerp(a["bend"], b["bend"]),
        "twist": lerp(a["twist"], b["twist"]),
        "side": lerp(a["side"], b["side"]),
        "rise": lerp(a["rise"], b["rise"]),
        "feet": {k: lerp(a["feet"][k], b["feet"][k]) for k in a["feet"]},
        "lift": {k: lerp(a["lift"][k], b["lift"][k]) for k in a["lift"]},
        "rest": {k: lerp(a["rest"][k], b["rest"][k]) for k in a["rest"]},
    }

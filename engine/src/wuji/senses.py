"""The felt sense: what the body reports about itself, in numbers and in plain words.

Built from what a body's own sensors would carry: the soles' pressure (where the weight rests in each foot), the muscles'
effort (spindles and tendon organs, summarised by region as the share of the region's strength in use), passive stretch
(muscles pulled past their resting length), joint angles, sway, and the breath and heart. Words are short, present tense,
without "I" or "you".
"""
from __future__ import annotations

import mujoco
import numpy as np

from .anatomy import FORWARD, LEFTWARD, REGION_ORDER, REGION_WORDS
from .body import Body

QUIET = 0.03  # a region using less than this share of its strength is quiet


def _where_along(frac: float) -> str:
    """Where along the sole the weight rests: heel, middle, forefoot (near Yongquan, the bubbling well), toes."""
    if frac < 0.28:
        return "heel"
    if frac < 0.42:
        return "middle"
    if frac < 0.62:
        return "forefoot"
    return "toes"


BOTH = {"heel": "both heels", "middle": "the middle of both feet", "forefoot": "both forefeet, near the bubbling wells",
        "toes": "the toes of both feet"}
ONE = {"heel": "its heel", "middle": "its middle", "forefoot": "its forefoot, near the bubbling well", "toes": "its toes"}


def region_effort(act: np.ndarray, f0: np.ndarray, sel: np.ndarray) -> float:
    """How hard a region works: its muscles' activations, root-mean-squared and weighted by strength, so a few muscles
    working hard show (as they would be felt) instead of being averaged away by the idle ones."""
    return float(np.sqrt((f0[sel] * act[sel] ** 2).sum() / f0[sel].sum()))


def _deg(body: Body, joint: str) -> float:
    return float(np.degrees(body.q[body.a.qadr[joint]]))


class Senses:
    def __init__(self, body: Body) -> None:
        self.body = body
        self.sway_history: list[np.ndarray] = []
        self.noticed: dict[str, object] = {}  # what the practitioner last noticed: each part's meaning (see ambient)
        self.noticed_raw: dict[str, object] = {}  # and the numbers behind it, for hysteresis

    def note_sway(self) -> None:
        b = self.body
        self.sway_history.append(np.array([0.5 * b.sway[0], b.sway[1] + b.breath_shift]))
        del self.sway_history[:-100]

    def read(self, breath_state: dict | None = None, breath_felt: list[str] | None = None, detail: str = "brief") -> dict:
        b = self.body
        a, m = b.a, b.m
        s = b.solution
        d = b.muscles.d  # holds the lengths of the last muscle solve

        # soles
        weight = {}
        fz = {k: max(0.0, float(v["force"][2])) for k, v in s.grf.items()}
        total = sum(fz.values()) or 1.0
        feet_text = {}
        b.posture_solver.d.qpos[:] = b.q
        mujoco.mj_kinematics(m, b.posture_solver.d)
        pd = b.posture_solver.d
        for side in ("left", "right"):
            share = fz.get(side, 0.0) / total
            weight[side] = round(share, 3)
            if side not in s.grf or share < 0.02:
                feet_text[side] = None
                continue
            cop = s.grf[side]["cop"]
            lo, hi = b.posture_solver.heel_to_toe[side]
            frac = float((cop @ FORWARD - lo) / (hi - lo))
            pts = np.array([pd.xpos[p.body] + pd.xmat[p.body].reshape(3, 3) @ p.local for p in a.sole if p.foot == side])
            mid = pts.mean(axis=0) @ LEFTWARD
            inward = (cop @ LEFTWARD - mid) * (-1 if side == "left" else 1)  # toward the body's midline
            edge = "inner" if inward > 0.02 else "outer" if inward < -0.02 else ""
            weight[f"{side}_along"] = round(frac, 3)
            weight[f"{side}_inward_mm"] = round(1000 * inward, 1)
            feet_text[side] = (_where_along(frac), edge)

        # effort by region
        f0 = a.f0
        regions = {r: round(region_effort(s.act, f0, a.muscle_region == r), 4) for r in REGION_ORDER}
        need = b.minimum if b.minimum is not None else s
        needed = {r: round(region_effort(need.act, f0, a.muscle_region == r), 4) for r in REGION_ORDER}
        passive = np.array([-mujoco.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i], m.actuator_acc0[i],
                                                   m.actuator_biasprm[i, :9]) for i in range(m.nu)])
        stretch = {r: round(float(passive[a.muscle_region == r].sum() / f0[a.muscle_region == r].sum()), 4) for r in REGION_ORDER}
        held = {r: v for r, v in b.tension.items() if v > 0}
        minimum = b.minimum_effort or s.effort or 1e-9
        ratio = s.effort / max(minimum, 1e-9)

        sway = np.array(self.sway_history) if self.sway_history else np.zeros((1, 2))
        sway_mm = float(1000 * np.linalg.norm(sway - sway.mean(axis=0), axis=1).max())
        joints = {k: round(_deg(b, k), 1) for k in ("knee_angle_l", "knee_angle_r", "hip_flexion_l", "hip_flexion_r",
                                                     "ankle_angle_l", "ankle_angle_r", "flex_extension")}
        feasible = s.root_residual < 20 and s.joint_residual < 10

        # each part of the felt sense: (key, its meaning, its words). The meaning is what counts as a change for `ambient`:
        # buckets coarse enough that fluctuation (the breath's cycle, sway within its range) is not change.
        parts: list[tuple[str, object, str]] = []
        knee = 0.5 * (joints["knee_angle_l"] + joints["knee_angle_r"])
        bend = "straight" if knee < 4 else "softly bent" if knee < 20 else "bent" if knee < 45 else "deeply bent"
        parts.append(("stance", (b.posture.stance, bend, b.moving),
                      f"{b.posture.stance.capitalize()} stance, knees {bend} ({knee:.0f}°){', moving' if b.moving else ''}."))
        lt, rt = feet_text["left"], feet_text["right"]
        edges = lambda ft: f", on the {ft[1]} edge" if ft[1] else ""
        if lt and rt:
            if lt == rt:
                where = f"through {BOTH[lt[0]]}{edges(lt)}"
            else:
                where = f"the left foot on {ONE[lt[0]]}{edges(lt)}, the right on {ONE[rt[0]]}{edges(rt)}"
            words = f"Weight {weight['left']:.0%} left, {weight['right']:.0%} right; {where}."
        else:
            on, ft = ("left", lt) if lt else ("right", rt)
            words = f"All the weight on the {on} foot, on {ONE[ft[0]]}{edges(ft)}; the other foot is free."
        parts.append(("weight", (round(weight["left"] * 20), lt, rt), words))
        working = sorted(((v, r) for r, v in regions.items() if v >= QUIET), reverse=True)
        if working:
            parts.append(("working", tuple((r, round(v * 100 / 3)) for v, r in working[:4]),
                          "Working: " + ", ".join(f"{REGION_WORDS[r]} {v:.0%}" for v, r in working[:4]) + "; the rest quiet."))
        else:
            parts.append(("working", (), "Every region quiet: the posture rests on the bones."))
        for r, v in held.items():
            parts.append((f"held:{r}", (round(v, 2), round(regions[r] * 100 / 3)),
                          f"Holding the {REGION_WORDS[r]} ({v:.0%} tension): working at {regions[r]:.0%} of their strength "
                          f"where the posture needs {needed[r]:.0%}."))
        stretched = [r for r, v in sorted(stretch.items(), key=lambda kv: -kv[1]) if v > 0.02][:2]
        if stretched:
            parts.append(("stretch", tuple(stretched), "Stretch in the " + " and ".join(REGION_WORDS[r] for r in stretched) + "."))
        parts.append(("sway", 0 if sway_mm < 3 else 1 if sway_mm < 8 else 2,
                      "Steady, barely swaying." if sway_mm < 3 else f"Swaying gently, about {sway_mm:.0f} mm."))
        if not feasible:
            parts.append(("strain", True, "Strained: the muscles cannot quite hold this."))
        if breath_felt:
            bs = breath_state or {}
            meaning = (bs.get("pattern"), bs.get("paced"), round((bs.get("rate_bpm") or 0) / 3), (bs.get("settled") or 0) > 0.6,
                       round((bs.get("urge") or 0) * 4), bs.get("mouth"), (bs.get("co2_mmhg") or 40) < 32)
            parts.append(("breath", meaning, chr(10).join(breath_felt)))  # its lines stay lines

        # hysteresis: a part that jitters across a bucket's edge without really moving keeps the meaning last noticed
        raw = {"weight": weight["left"], "sway": sway_mm, "working": regions}
        close = {
            "weight": lambda old, new: abs(old - new) < 0.04,
            "sway": lambda old, new: abs(old - new) < 1.5,
            "working": lambda old, new: all(abs(old[r] - new[r]) < 0.015 for r in new),
        }
        for i, (key, meaning, words) in enumerate(parts):
            if key in close and key in self.noticed and key in self.noticed_raw and self.noticed[key] != meaning                     and close[key](self.noticed_raw[key], raw[key]):
                if key != "weight" or meaning[1:] == self.noticed[key][1:]:  # the feet's zones must not have changed
                    parts[i] = (key, self.noticed[key], words)
        changed = [key for key, meaning, _ in parts if self.noticed.get(key, None) != meaning]
        if detail == "ambient":
            lines = self._ambient(parts, b.posture.stance)
        else:
            lines = [words for _, _, words in parts]
        for key, meaning, _ in parts:
            if self.noticed.get(key) != meaning or key not in self.noticed_raw:
                self.noticed_raw[key] = raw.get(key)
        self.noticed = {key: meaning for key, meaning, _ in parts}

        if detail == "full":
            lines.append("Joints: knees {:.0f}°/{:.0f}°, hips {:.0f}°/{:.0f}°, ankles {:.0f}°/{:.0f}°, lumbar {:+.0f}° (left/right)."
                         .format(joints["knee_angle_l"], joints["knee_angle_r"], joints["hip_flexion_l"], joints["hip_flexion_r"],
                                 joints["ankle_angle_l"], joints["ankle_angle_r"], joints["flex_extension"]))
            lines.append("All regions: " + ", ".join(f"{r.replace('_', ' ')} {v:.1%}" for r, v in regions.items()) + ".")

        heart = {"bpm": (breath_state or {}).get("heart_bpm")}
        return {
            "t": round(b.t, 2),
            "text": "\n".join(lines),
            "body": {
                "stance": b.posture.stance,
                "moving": b.moving,
                "weight": weight,
                "sway_mm": round(sway_mm, 1),
                "joints_deg": joints,
                "effort": {"total": round(s.effort, 6), "times_needed": round(ratio, 2), "regions": regions, "needed": needed},
                "stretch": stretch,
                "tension": dict(b.tension),
                "feasible": feasible,
                "residual": {"root": round(s.root_residual, 2), "joint": round(s.joint_residual, 2)},
            },
            "breath": breath_state or {},
            "heart": heart,
            "changed": changed,
        }

    def _ambient(self, parts: list, stance: str) -> list[str]:
        """What has changed since the practitioner last noticed the body, as a body habituates: a shift of weight, a region
        starting or stopping work, tension taken up or let go, a stretch, the breath changing its pattern or settling. The
        breath's cycle and sway within its range are not change. Nothing changed: one quiet line, never silence."""
        before = self.noticed
        if not before:
            return [words for _, _, words in parts]
        lines = [words for key, meaning, words in parts if before.get(key, None) != meaning]
        now = {key for key, _, _ in parts}
        for key in before:
            if key in now:
                continue
            if key.startswith("held:"):
                lines.append(f"The {REGION_WORDS[key[5:]]} let go.")
            elif key == "stretch":
                lines.append("The stretch has eased.")
            elif key == "strain":
                lines.append("The strain has passed.")
        return lines or [f"{stance.capitalize()} stance as before; nothing new."]

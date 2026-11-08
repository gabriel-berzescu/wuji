"""Where the bones go: a posture (stance, weight, sink, arms) becomes a full set of joint angles, by inverse kinematics.

The feet stay planted where they stand; the knees bend to the asked depth; the pelvis stays level (the tailbone drops a little
as the knees bend), the chest stays upright, and the centre of mass sits over the feet where the weight is asked to be. The
arms take their shape directly from joint angles. Every pose honours the model's joint couplings (vertebrae, shoulder rhythm).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .anatomy import FORWARD, LEFTWARD, UP, Anatomy

DEG = np.pi / 180

STANCES = {
    # half the distance between the feet's centres (m), the deepest knee bend at sink 1 (deg), how far the toes turn out (deg)
    "wuji": {"half_width": 0.085, "max_knee": 45.0},
    "horse": {"half_width": 0.20, "max_knee": 75.0},
}

# Arm shapes: shoulder elevation, plane of elevation (0 out to the side, 90 forward), rotation, elbow flexion, forearm
# (pro/sup), wrist flexion (deg), and how much the fingers curl. The shoulder and elbow angles were fitted to where the hands
# and elbows should be relative to the shoulder (tools/fit_arms.py): hang beside the thighs; embrace holding a ball in front of
# the chest, elbows rounded; lift_sky overhead; press_down in front of the hips.
ARMS = {
    "hang": {"shoulder_elv": 6, "elv_angle": 0, "shoulder_rot": 0, "elbow_flex": 0, "pro_sup": 0, "flexion": 0, "curl": 0.2},
    "embrace": {"shoulder_elv": 47, "elv_angle": 41, "shoulder_rot": 34, "elbow_flex": 90, "pro_sup": -20, "flexion": 0, "curl": 0.25},
    "lift_sky": {"shoulder_elv": 178, "elv_angle": 10, "shoulder_rot": 0, "elbow_flex": 9, "pro_sup": 80, "flexion": -40, "curl": 0.0},
    "press_down": {"shoulder_elv": 16, "elv_angle": -29, "shoulder_rot": 0, "elbow_flex": 52, "pro_sup": 80, "flexion": -45, "curl": 0.0},
    # for the Eight Brocades (curl 2 is a fist)
    "bow_extend": {"shoulder_elv": 89, "elv_angle": -2, "shoulder_rot": 0, "elbow_flex": 0, "pro_sup": 0, "flexion": -40, "curl": 0.6},
    "bow_pull": {"shoulder_elv": 86, "elv_angle": 10, "shoulder_rot": 23, "elbow_flex": 130, "pro_sup": 0, "flexion": 0, "curl": 2.0},
    "punch": {"shoulder_elv": 77, "elv_angle": 100, "shoulder_rot": 0, "elbow_flex": 0, "pro_sup": 80, "flexion": 0, "curl": 2.0},
    "fist_waist": {"shoulder_elv": 44, "elv_angle": -81, "shoulder_rot": 0, "elbow_flex": 85, "pro_sup": -80, "flexion": 0, "curl": 2.0},
    "earth_press": {"shoulder_elv": 11, "elv_angle": 23, "shoulder_rot": 0, "elbow_flex": 0, "pro_sup": 80, "flexion": -45, "curl": 0.0},
    "owl": {"shoulder_elv": 9, "elv_angle": -15, "shoulder_rot": -27, "elbow_flex": 0, "pro_sup": -60, "flexion": 0, "curl": 0.1},
    "hands_thighs": {"shoulder_elv": 19, "elv_angle": 77, "shoulder_rot": 0, "elbow_flex": 16, "pro_sup": 0, "flexion": 0, "curl": 0.3},
}


@dataclass
class Posture:
    stance: str = "wuji"
    weight: float = 0.5  # share on the left foot
    sink: float = 0.25  # 0 straight .. 1 the deepest the stance allows
    arms: str = "hang"
    tailbone: float = 0.0  # 0 .. 1: how far the tailbone drops (posterior pelvic tilt), beyond what sinking brings
    arm_left: str | None = None  # one arm's own shape (None: as `arms`)
    arm_right: str | None = None
    bend: float = 0.0  # 0 .. 1: the trunk folding forward (at the waist and the hips), up to 100 degrees
    twist: float = 0.0  # -1 .. 1: the trunk turning, positive to the left, up to 40 degrees
    side_bend: float = 0.0  # -1 .. 1: the trunk leaning, positive to the left, up to 20 degrees
    rise: float = 0.0  # 0 .. 1: up on the toes, the heels raised up to 22 degrees

    def merged(self, **changes) -> "Posture":
        out = replace(self, **{k: v for k, v in changes.items() if v is not None})
        if changes.get("arms") is not None:  # both arms given: a one-arm shape goes, unless given again
            out.arm_left, out.arm_right = changes.get("arm_left"), changes.get("arm_right")
        return out


TRUNK_MAX = {"bend": 100.0, "twist": 40.0, "side": 20.0}  # degrees at 1
RISE_MAX = 22.0  # degrees of heel raise at 1


class PostureSolver:
    LEG_JOINTS = ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle")
    SPINE_JOINTS = ("flex_extension", "lat_bending", "axial_rotation")

    def __init__(self, anatomy: Anatomy) -> None:
        self.a = anatomy
        m = self.m = anatomy.m
        self.d = mujoco.MjData(m)
        self.root = anatomy.body["Full Body"]
        self.pelvis = anatomy.body["pelvis"]
        self.thorax = anatomy.body["thorax"]
        dofs = list(range(6))
        for side in ("r", "l"):
            dofs += [anatomy.dadr[f"{j}_{side}"] for j in self.LEG_JOINTS]
        dofs += [anatomy.dadr[j] for j in self.SPINE_JOINTS]
        self.dofs = np.array(dofs)

        # standing: the soles on the floor, the body as the model stands it
        q = m.qpos0.copy()
        mujoco.mj_kinematics(m, self.d)
        self.d.qpos[:] = q
        mujoco.mj_kinematics(m, self.d)
        sole_z = min(self._sole_world(p)[2] for p in anatomy.sole)
        q[2] -= sole_z
        self.q_stand = q
        self.d.qpos[:] = q
        mujoco.mj_kinematics(m, self.d)
        self.pelvis_quat0 = self.d.xquat[self.pelvis].copy()
        self.thorax_quat0 = self.d.xquat[self.thorax].copy()
        self.heel_to_toe = {}
        self.foot0 = {}
        self.toes0 = {}
        for side in ("left", "right"):
            b = anatomy.body["calcn_" + side[0]]
            self.foot0[side] = (b, self.d.xpos[b].copy(), self.d.xquat[b].copy())
            t = anatomy.body["toes_" + side[0]]
            self.toes0[side] = (t, self.d.xpos[t].copy(), self.d.xquat[t].copy())
            pts = np.array([self._sole_world(p) for p in anatomy.sole if p.foot == side])
            along = pts @ FORWARD
            self.heel_to_toe[side] = (float(along.min()), float(along.max()))
        self.q = q.copy()

    def _sole_world(self, p) -> np.ndarray:
        return self.d.xpos[p.body] + self.d.xmat[p.body].reshape(3, 3) @ p.local

    def feet_for(self, stance: str) -> dict:
        """Each foot's sideways position (m, along LEFTWARD, from the body's midline) in a stance."""
        half = STANCES[stance]["half_width"]
        return {"left": half, "right": -half}

    def foot_poses(self, feet: dict, lift: dict | None = None, rise_deg: float = 0.0) -> dict:
        """Planted (or lifted) foot poses: each foot moved sideways from where it stands, up by its lift (m), and its heel
        raised by rise_deg, pivoting on the ball of the foot (the toes stay flat on the floor)."""
        centre = 0.5 * (self.foot0["left"][1] + self.foot0["right"][1]) @ LEFTWARD
        poses = {}
        for side in ("left", "right"):
            b, pos, quat = self.foot0[side]
            shift = LEFTWARD * (centre + feet[side] - pos @ LEFTWARD) + UP * (lift or {}).get(side, 0.0)
            p, qt = pos + shift, quat.copy()
            if rise_deg > 0:
                pivot = self.toes0[side][1] + shift
                r = _axis_quat(LEFTWARD, rise_deg * DEG)  # about the left-right axis: the heel goes up
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, r)
                p = pivot + rot.reshape(3, 3) @ (p - pivot)
                qt = _quat_mul(r, qt)
            poses[side] = (b, p, qt)
        return poses

    def trunk_target(self, bend_deg: float, twist_deg: float, side_deg: float) -> np.ndarray:
        """The chest's orientation: bent forward, turned (positive to the left) and leaning (positive to the left)."""
        q = _quat_mul(_axis_quat(LEFTWARD, bend_deg * DEG), self.thorax_quat0)
        q = _quat_mul(_axis_quat(FORWARD, -side_deg * DEG), q)
        return _quat_mul(_axis_quat(UP, twist_deg * DEG), q)

    def solve(self, posture: Posture, feet: dict | None = None, iterations: int = 30, sway: np.ndarray | None = None,
              lift: dict | None = None, knee_deg: float | None = None, arms: dict | None = None,
              tilt_deg: float | None = None, arms_right: dict | None = None, trunk: dict | None = None,
              rise_deg: float = 0.0) -> np.ndarray:
        """Joint angles (qpos) for a posture, warm-started from the last solution.

        feet: sideways foot positions (default: the stance's); lift: how high each foot is raised (stepping); knee_deg: the
        knee bend (default: from sink and the stance); arms: explicit arm angles (default: the posture's arm shape), for
        the left arm too when arms_right is given; trunk: {"bend", "twist", "side"} in degrees; rise_deg: heels raised.
        """
        m, d, a = self.m, self.d, self.a
        stance = STANCES[posture.stance]
        feet = self.foot_poses(feet or self.feet_for(posture.stance), lift, rise_deg)
        trunk = trunk or {}
        bend = trunk.get("bend", 0.0)
        knee = (posture.sink * stance["max_knee"] if knee_deg is None else knee_deg) * DEG
        tilt = (pelvic_tilt(knee / DEG, posture.tailbone) if tilt_deg is None else tilt_deg) * DEG
        # a deep bend folds at the hips as much as at the waist: the pelvis tips forward with it
        pelvis_target = _quat_mul(_axis_quat(LEFTWARD, 0.55 * bend * DEG - tilt), self.pelvis_quat0)
        pelvis_target = _quat_mul(_axis_quat(UP, 0.25 * trunk.get("twist", 0.0) * DEG), pelvis_target)
        thorax_target = self.trunk_target(bend, trunk.get("twist", 0.0), trunk.get("side", 0.0))
        lx, rx = feet["left"][1] @ LEFTWARD, feet["right"][1] @ LEFTWARD
        com_target = np.array([rx + posture.weight * (lx - rx), self._com_ahead(feet, rise_deg)])
        if sway is not None:
            com_target = com_target + sway

        q = self.q
        q_free = self.q_stand
        for _ in range(iterations):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs, weights = [], [], []

            def task(J, e, w):
                rows.append(J[:, self.dofs] if J.ndim == 2 else J[None, self.dofs])
                errs.append(np.atleast_1d(e))
                weights.append(np.full(np.atleast_1d(e).shape, w))

            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for side, (b, pos, quat) in feet.items():
                mujoco.mj_jacBody(m, d, jp, jr, b)
                task(jp.copy(), pos - d.xpos[b], 100.0)
                task(jr.copy(), _rot_err(quat, d.xquat[b]), 30.0)
            mujoco.mj_jacBody(m, d, jp, jr, self.pelvis)
            task(jr.copy(), _rot_err(pelvis_target, d.xquat[self.pelvis]), 10.0)
            mujoco.mj_jacBody(m, d, jp, jr, self.thorax)
            task(jr.copy(), _rot_err(thorax_target, d.xquat[self.thorax]), 10.0)
            for side in ("left", "right"):  # the toes stay flat on the floor, also on tiptoe
                t, _, tquat = self.toes0[side]
                mujoco.mj_jacBody(m, d, jp, jr, t)
                task(jr.copy(), _rot_err(tquat, d.xquat[t]), 10.0 if rise_deg > 0 else 1.0)
            jc = np.zeros((3, m.nv))
            mujoco.mj_jacSubtreeCom(m, d, jc, self.root)
            com = d.subtree_com[self.root]
            task(np.stack([jc.T @ LEFTWARD, jc.T @ FORWARD]), com_target - np.array([com @ LEFTWARD, com @ FORWARD]), 30.0)
            for side in ("r", "l"):
                row = np.zeros(m.nv)
                row[a.dadr["knee_angle_" + side]] = 1.0
                task(row, knee - q[a.qadr["knee_angle_" + side]], 10.0)
            # everything else stays near standing, gently
            reg = np.zeros((len(self.dofs) - 6, m.nv))
            e = np.zeros(len(self.dofs) - 6)
            for i, v in enumerate(self.dofs[6:]):
                reg[i, v] = 1.0
                jnt = m.dof_jntid[v]
                e[i] = q_free[m.jnt_qposadr[jnt]] - q[m.jnt_qposadr[jnt]]
            task(reg, e, 0.05)

            J = np.vstack(rows)
            err = np.concatenate(errs)
            W = np.concatenate(weights)
            JW = J.T * W
            dq_sub = np.linalg.solve(JW @ J + 1e-4 * np.eye(len(self.dofs)), JW @ err)
            dq = np.zeros(m.nv)
            dq[self.dofs] = dq_sub
            mujoco.mj_integratePos(m, q, dq, 1.0)
            self._clip(q)
            a.couple(q)
            if np.linalg.norm(dq_sub) < 1e-6:
                break
        self.q = q
        left = arms if arms is not None else ARMS[posture.arms]
        self.set_arms(q, left, arms_right if arms_right is not None else left)
        return q.copy()

    def _com_ahead(self, feet, rise_deg: float = 0.0) -> float:
        """Where along the feet the weight rests: a little ahead of the ankles, toward the bubbling well (Yongquan); on
        tiptoe, over the balls of the feet, the only part still on the floor."""
        lo, hi = self.heel_to_toe["left"]
        return lo + (0.38 + 0.34 * min(1.0, rise_deg / 8.0)) * (hi - lo)

    def _clip(self, q: np.ndarray) -> None:
        m = self.m
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                adr = m.jnt_qposadr[j]
                q[adr] = np.clip(q[adr], *m.jnt_range[j])

    def set_arms(self, q: np.ndarray, s_left: dict, s_right: dict | None = None) -> None:
        """Sets the arms from shapes' angles (deg; see ARMS): the right arm from s_right (default: the same as the left).
        The model mirrors its joints, so the same angles give mirrored arms."""
        a = self.a
        for side in ("r", "l"):
            s = s_left if side == "l" else (s_right or s_left)
            for joint in ("shoulder_elv", "elv_angle", "shoulder_rot", "elbow_flex", "pro_sup", "flexion"):
                q[a.qadr[f"{joint}_{side}"]] = s[joint] * DEG
            for f in range(2, 6):
                q[a.qadr[f"mcp{f}_flexion_{side}"]] = s["curl"] * 40 * DEG  # curl 2: a fist (the joints' limits hold it)
                q[a.qadr[f"pm{f}_flexion_{side}"]] = s["curl"] * 50 * DEG
                q[a.qadr[f"md{f}_flexion_{side}"]] = s["curl"] * 25 * DEG
        self._clip(q)
        a.couple(q)


def pelvic_tilt(knee_deg: float, tailbone: float) -> float:
    """Posterior pelvic tilt (deg): the tailbone drops a little as the knees bend, and further when asked (song yao: the
    lumbar curve lengthens and the waist hangs instead of holding)."""
    return 6.0 * min(1.0, knee_deg / 45.0) + 12.0 * min(1.0, max(0.0, tailbone))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def _rot_err(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """The rotation (world frame, axis * angle) that takes `current` to `target`."""
    res = np.zeros(3)
    mujoco.mju_subQuat(res, target, current)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, current)
    return rot.reshape(3, 3) @ res

"""Where the bones go: a posture (stance, weight, sink, arms) becomes a full set of joint angles, by inverse kinematics.

The feet stay planted where they stand; the knees bend to the asked depth; the pelvis stays level (the tailbone drops a little
as the knees bend), the chest stays upright, and the centre of mass sits over the feet where the weight is asked to be. The
arms take their shape directly from joint angles, except hands resting on the thighs: those go onto the thigh, by inverse
kinematics of the arm. Every pose honours the model's joint couplings (vertebrae, shoulder rhythm).
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
# the chest, elbows rounded; lift_sky overhead; press_down in front of the hips. hands_thighs holds the hands in front of
# the thighs, palms back; whenever a thigh is within reach, place_hands lays the palm on it.
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
    "hands_thighs": {"shoulder_elv": 19, "elv_angle": 77, "shoulder_rot": 0, "elbow_flex": 16, "pro_sup": 80, "flexion": 0, "curl": 0.3},
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

# Hands resting on the thighs: each palm goes to the front of its own thigh, as near the knee as the arm reaches with the
# elbow soft (no lower than 10 cm above the knee joint, no higher than 8 cm below the hip), the palm turned to face it.
REST_LOW = -0.30  # m along the femur from the hip (the knee joint is at -0.40)
REST_HIGH = -0.08
REST_ELBOW = 20.0  # deg: the elbow's softness when the arm reaches its lowest
ARM_JOINTS = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex", "pro_sup", "deviation", "flexion")


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

        # the arms, for resting the hands on the thighs: their joints, and how far a palm reaches from the shoulder
        self.arm_qadr = {side: np.array([anatomy.qadr[f"{j}_{side[0]}"] for j in ARM_JOINTS]) for side in ("left", "right")}
        self.arm_range = {side: m.jnt_range[[anatomy.jnt[f"{j}_{side[0]}"] for j in ARM_JOINTS]] for side in ("left", "right")}
        self.arm_couplings = {}  # the couplings an arm's joints drive (the shoulder rhythm), directly or through another
        for side, adr in self.arm_qadr.items():
            driven, chain = set(adr.tolist()), []
            for _ in range(3):
                chain = [c for c in anatomy.couplings if c[1] in driven]
                driven |= {c[0] for c in chain}
            self.arm_couplings[side] = chain
        self.arm_warm: dict = {}  # each resting arm's last angles, to start from
        self.hand_gap: dict = {}  # each resting palm's gap to its thigh's surface (m), after the last solve
        reach = q.copy()
        self.set_arms(reach, {**ARMS["hang"], "elbow_flex": REST_ELBOW})
        self.d.qpos[:] = reach
        mujoco.mj_kinematics(m, self.d)
        self.reach = {side: float(np.linalg.norm(self._palm(side)[0] - self.d.xpos[anatomy.body[f"humerus_{side[0]}"]]))
                      for side in ("left", "right")}

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
              rise_deg: float = 0.0, rest: dict | None = None) -> np.ndarray:
        """Joint angles (qpos) for a posture, warm-started from the last solution.

        feet: sideways foot positions (default: the stance's); lift: how high each foot is raised (stepping); knee_deg: the
        knee bend (default: from sink and the stance); arms: explicit arm angles (default: the posture's arm shape), for
        the left arm too when arms_right is given; trunk: {"bend", "twist", "side"} in degrees; rise_deg: heels raised;
        rest: side -> 0..1, how far each hand has gone onto its thigh (default, when the arms come from the posture: 1
        for an arm in hands_thighs).
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
        if rest is None and arms is None:
            rest = {s: float((getattr(posture, f"arm_{s}") or posture.arms) == "hands_thighs") for s in ("left", "right")}
        self.hand_gap = self.place_hands(q, rest or {})
        return q.copy()

    def _palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """A palm's middle and the way it faces (world), from the kinematics in self.d."""
        p = self.a.palm[side]
        R = self.d.xmat[p.hand].reshape(3, 3)
        return self.d.xpos[p.hand] + R @ p.local, R @ p.normal

    def _thigh_spot(self, side: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Where a hand rests on its thigh: the world point, the thigh's outward normal there, and how well it is within
        reach (1 within, falling to 0 two centimetres beyond). On the front, as near the knee as the palm reaches from the
        shoulder with the elbow soft, between REST_LOW and REST_HIGH; out of reach, the nearest point of that stretch."""
        a, d = self.a, self.d
        f = a.palm[side].femur
        F, R = d.xpos[f], d.xmat[f].reshape(3, 3)
        (y_hi, y_lo), (r_hi, r_lo) = a.thigh.radius_at
        k = (r_hi - r_lo) / (y_hi - y_lo)  # the surface is a line here: front(y) = shoulder + A + y B
        A = F + R @ np.array([r_hi - k * y_hi, 0.0, 0.0]) - d.xpos[a.body[f"humerus_{side[0]}"]]
        B = R @ np.array([k, 1.0, 0.0])
        bb, ab, reach = B @ B, A @ B, self.reach[side]
        y = float(np.clip(-ab / bb, REST_LOW, REST_HIGH))  # the nearest to the shoulder
        dist = float(np.linalg.norm(A + y * B))
        if dist <= reach:  # within reach: go down to where the reachable stretch begins
            y = max(REST_LOW, (-ab - np.sqrt(ab * ab - bb * (A @ A - reach ** 2))) / bb)
        return F + R @ a.thigh.front(y), R[:, 0], float(np.clip(1.0 - (dist - reach) / 0.02, 0.0, 1.0))

    def _thigh_gap(self, side: str) -> float:
        """How far a palm is from its thigh's surface (m; negative: pressed into the soft tissue), from self.d."""
        a, d = self.a, self.d
        f = a.palm[side].femur
        loc = d.xmat[f].reshape(3, 3).T @ (self._palm(side)[0] - d.xpos[f])
        if not a.thigh.knee_y < loc[1] < 0.0:
            return np.inf
        return float(np.hypot(loc[0], loc[2]) - a.thigh.front(loc[1])[0])

    def place_hands(self, q: np.ndarray, rest: dict, iterations: int = 3) -> dict:
        """Brings each resting hand (rest > 0) from where its arm shape holds it toward its spot on the thigh, by inverse
        kinematics of the arm alone (the legs and trunk are already placed), keeping near the shape. A thigh out of reach
        is not strained for: the hand stays nearer its shape. Returns each resting palm's gap to its thigh's surface (m)."""
        m, d, a = self.m, self.d, self.a
        gaps = {}
        for side in ("left", "right"):
            s = float(rest.get(side, 0.0))
            if s <= 0:
                self.arm_warm.pop(side, None)
                continue
            adr, (lo, hi) = self.arm_qadr[side], self.arm_range[side].T
            shape = q[adr].copy()
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            start = self._palm(side)[0]
            spot, normal, within = self._thigh_spot(side)
            s *= within
            if s <= 0:
                self.arm_warm.pop(side, None)
                gaps[side] = self._thigh_gap(side)
                continue
            target = start + s * (spot - start)
            warm = self.arm_warm.get(side)
            x = q[adr].copy() if warm is None else warm.copy()
            chain = self.arm_couplings[side]
            qq = q.copy()

            def palm_at(x):
                qq[adr] = x
                a.couple(qq, chain)
                d.qpos[:] = qq
                mujoco.mj_kinematics(m, d)
                p, n = self._palm(side)
                # the palm's position (cm) and its facing, turned toward the thigh as the hand arrives
                return np.concatenate([100 * p, 3 * s * n])

            goal = np.concatenate([100 * target, -3 * s * normal])
            for _ in range(iterations if warm is not None else 4 * iterations):  # warm: a few steps a frame suffice
                y0 = palm_at(x)
                J = np.stack([(palm_at(x + 1e-4 * e) - y0) / 1e-4 for e in np.eye(len(x))], axis=1)
                reg = 0.1  # per rad: stay near the shape where the task leaves freedom
                dx = np.linalg.solve(J.T @ J + reg * np.eye(len(x)), J.T @ (goal - y0) + reg * (shape - x))
                x = np.clip(x + dx, lo, hi)
                if np.abs(dx).max() < 1e-3:
                    break
            q[adr] = x
            a.couple(q, chain)
            self.arm_warm[side] = x.copy()
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            gaps[side] = self._thigh_gap(side)
        return gaps

    def _com_ahead(self, feet, rise_deg: float = 0.0) -> float:
        """Where along the feet the weight rests: a little ahead of the ankles, toward the bubbling well (Yongquan); on
        tiptoe, over the balls of the feet, the only part still on the floor."""
        lo, hi = self.heel_to_toe["left"]
        return lo + (0.38 + 0.34 * min(1.0, rise_deg / 8.0)) * (hi - lo)

    def _clip(self, q: np.ndarray) -> None:
        m = self.m
        for j in range(m.njnt):
            if m.jnt_limited[j] and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
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

"""What the body is made of, in the terms a practitioner feels: muscle regions and sides, the soles, the joint couplings.

MyoFullBody (MyoHub, Apache-2.0, via the `musclemimic-models` package): 416 Hill-type muscles over 128 degrees of freedom.
Its world: z up, the body faces -y, its right side is -x.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import musclemimic_models
import numpy as np

UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([0.0, -1.0, 0.0])
LEFTWARD = np.array([1.0, 0.0, 0.0])

# Muscle name roots (side suffix removed) by region. Every one of the model's 208 roots is listed exactly once.
REGIONS: dict[str, tuple[str, ...]] = {
    "feet": ("edl", "ehl", "fdl", "fhl"),
    "calves": ("soleus", "gasmed", "gaslat", "tibant", "tibpost", "perbrev", "perlong"),
    "thighs": ("recfem", "vasint", "vaslat", "vasmed", "bflh", "bfsh", "semimem", "semiten", "grac", "sart",
               "addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid", "addmagProx"),
    "hips": ("glmax1", "glmax2", "glmax3", "glmed1", "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "piri", "tfl",
             "iliacus", "psoas", "Ps_"),
    "lower_back": ("IL_", "LTpL_", "LTpT_", "MF_", "QL_"),
    "belly": ("rect_abd", "EO", "IO"),
    "shoulders": ("DELT", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ", "PECM", "LAT", "CORB"),
    "arms": ("BIClong", "BICshort", "TRIlat", "TRIlong", "TRImed", "BRA", "BRD", "ANC", "PT", "PQ", "SUP", "FCR", "FCU",
             "ECRL", "ECRB", "ECU", "PL"),
    "hands": ("EDC", "EDM", "EIP", "FDS", "FDP", "LU_RB", "UI_UB", "RI", "APL", "EPB", "EPL", "FPL", "OP"),
}
REGION_ORDER = tuple(REGIONS)

# Plain names for the felt sense.
REGION_WORDS = {
    "feet": "feet", "calves": "calves", "thighs": "thighs", "hips": "hips (glutes, psoas: the kua)",
    "lower_back": "lower back", "belly": "belly", "shoulders": "shoulders", "arms": "arms", "hands": "hands",
    "diaphragm": "diaphragm",
}


def _root(name: str) -> tuple[str, str]:
    """A muscle's root name and side ('left' / 'right'); the right arm's muscles carry no suffix."""
    for suffix, side in (("_left", "left"), ("_l", "left"), ("_r", "right")):
        if name.endswith(suffix):
            return name[: -len(suffix)], side
    return name, "right"


def region_of(root: str) -> str:
    for region, keys in REGIONS.items():
        for k in keys:
            if root == k or (k.endswith("_") and root.startswith(k)) or (root.startswith(k) and re.fullmatch(re.escape(k) + r"\d*", root)):
                return region
    raise KeyError(f"muscle {root!r} belongs to no region")


@dataclass(frozen=True)
class SolePoint:
    """A point on a sole that can bear weight: the end of a foot capsule, at its bottom."""
    foot: str  # 'left' | 'right'
    body: int  # the body it rides on (calcn or toes)
    local: np.ndarray  # position in that body's frame
    where: str  # 'heel' | 'ball' | 'toe'


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(musclemimic_models.get_xml_path("myofullbody")))


def calibrate_rest(m: mujoco.MjModel, q_rest: np.ndarray, onset: float = 1.0) -> dict:
    """Moves each muscle's passive onset so that, in relaxed standing (q_rest), no muscle is past it.

    The published model puts many muscles beyond their passive onset in its own standing pose (4,174 N of passive pull in
    all, the rectus abdominis at 104% of its maximum force and each middle deltoid at 62%, with zero activation), which is
    harmless for training walking policies but wrong for a body meant to feel relaxation. MuJoCo calibrates a muscle's
    passive part (biasprm) and active part (gainprm) separately, so only the passive range is shifted: active strength
    stays as published, and stretching beyond rest still builds passive tension. Returns {muscle index: shift} for the
    muscles moved.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = q_rest
    mujoco.mj_fwdPosition(m, d)
    moved = {}
    for i in range(m.nu):
        lr0, lr1 = m.actuator_lengthrange[i]
        r0, r1 = m.actuator_biasprm[i, 0:2]
        norm = r0 + (d.actuator_length[i] - lr0) * (r1 - r0) / (lr1 - lr0)
        if norm > onset:
            m.actuator_biasprm[i, 0:2] -= norm - onset
            moved[i] = float(norm - onset)
    return moved


class Anatomy:
    def __init__(self, m: mujoco.MjModel) -> None:
        self.m = m
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)
        self.muscles = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
        self.muscle_side = []
        self.muscle_region = []
        for n in self.muscles:
            root, side = _root(n)
            self.muscle_side.append(side)
            self.muscle_region.append(region_of(root))
        self.muscle_side = np.array(self.muscle_side)
        self.muscle_region = np.array(self.muscle_region)
        self.f0 = m.actuator_gainprm[:, 2].copy()  # peak isometric force, N

        self.jnt = {name(mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self.body = {name(mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
        self.qadr = {n: int(m.jnt_qposadr[j]) for n, j in self.jnt.items()}
        self.dadr = {n: int(m.jnt_dofadr[j]) for n, j in self.jnt.items()}

        # joint couplings: q1 - q1_0 = c0 + c1 (q2 - q2_0) + ... (q2 absent: q1 locked at q1_0 + c0)
        self.couplings = []
        for e in range(m.neq):
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            self.couplings.append((int(m.jnt_qposadr[j1]), int(m.jnt_qposadr[j2]) if j2 >= 0 else -1, m.eq_data[e, :5].copy()))
        self.coupled_qadr = {c[0] for c in self.couplings}

        self.sole = self._sole_points()

    def couple(self, qpos: np.ndarray) -> None:
        """Sets every coupled joint from its master, in place (the spine's vertebrae, the shoulder rhythm, the patellae)."""
        q0 = self.m.qpos0
        for _ in range(2):  # chains of couplings settle in two passes
            for a1, a2, c in self.couplings:
                x = qpos[a2] - q0[a2] if a2 >= 0 else 0.0
                qpos[a1] = q0[a1] + c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))

    def _sole_points(self) -> list[SolePoint]:
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        points = []
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE or not re.fullmatch(r"[rl]_(foot_col[13]|bofoot_col[12])", n):
                continue
            foot = "right" if n.startswith("r_") else "left"
            body = int(m.geom_bodyid[g])
            radius, half = m.geom_size[g, :2]
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            for sgn in (-1.0, 1.0):
                world = d.geom_xpos[g] + sgn * half * axis - radius * UP
                local = d.xmat[body].reshape(3, 3).T @ (world - d.xpos[body])
                ahead = float(np.dot(world - d.xpos[self.body["calcn_" + foot[0]]], FORWARD))
                where = "toe" if "bofoot" in n and ahead > 0.15 else "ball" if "bofoot" in n or ahead > 0.1 else "heel"
                points.append(SolePoint(foot, body, local, where))
        return points

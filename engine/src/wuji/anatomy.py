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


# Muscle strength, calibrated to people (calibrate_strength). The published forces come from muscle volumes at a high specific
# tension (the legs from Rajagopal et al. 2016), and they are uneven against measured strength: the model's maximal torques
# were, against healthy men of its size (STRENGTH_TESTS), plantarflexion 406 N m against 160, knee flexion 348 against 120,
# hip extension 507 against 237, but shoulder flexion 51 against 77. A muscle's effort is its share of its strength, so with
# the published forces the calves read about 10% on tiptoe where EMG reads 25-40% MVC. Each group's factor below multiplies
# its muscles' force; tools/fit_strength.py fits them all at once to every test (least squares in log torque, each factor
# pulled weakly toward 1). A muscle crossing two joints keeps one factor, set where it does most of its work and checked at
# the other joint: the hamstrings are fitted to knee flexion and hip extension together, the gastrocnemii go with the
# plantarflexors (their knee flexion is in the knee flexion test), the rectus femoris with the quadriceps (its hip flexion in
# the hip flexion test). Muscles in no group (the feet's and hands' own, the forearm's wrist and finger muscles, the neck's,
# the diaphragm's) keep their published force: no norm here constrains them.
STRENGTH: dict[str, tuple[tuple[str, ...], float, str]] = {
    # group: (muscle roots, force factor, the tests that set it)
    "plantarflexors": (("soleus", "gasmed", "gaslat", "tibpost", "perlong", "perbrev", "fdl", "fhl"), 0.40, "plantarflexion"),
    "dorsiflexors": (("tibant", "edl", "ehl"), 0.56, "dorsiflexion"),
    "quadriceps": (("recfem", "vasint", "vaslat", "vasmed"), 0.77, "knee extension (rectus femoris: also hip flexion)"),
    "hamstrings": (("bflh", "bfsh", "semimem", "semiten"), 0.29, "knee flexion and hip extension, fitted together"),
    "gluteus_maximus": (("glmax1", "glmax2", "glmax3"), 0.59, "hip extension"),
    "hip_abductors": (("glmed1", "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "tfl", "piri"), 0.63, "hip abduction"),
    "hip_adductors": (("addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid", "addmagProx", "grac"), 0.73,
                      "no adduction norm; through their share of hip flexion and extension"),
    "hip_flexors": (("iliacus", "psoas", "Ps_", "sart"), 1.02, "hip flexion"),
    "back_extensors": (("IL_", "LTpL_", "LTpT_", "MF_", "QL_"), 1.03, "trunk extension"),
    "abdominals": (("rect_abd", "EO", "IO"), 0.81, "trunk flexion"),
    "shoulder": (("DELT", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ", "PECM", "LAT", "CORB"), 1.44,
                 "shoulder flexion and abduction"),
    "elbow_flexors": (("BIClong", "BICshort", "BRA", "BRD"), 0.75, "elbow flexion"),
    "elbow_extensors": (("TRIlat", "TRIlong", "TRImed", "ANC"), 1.27, "elbow extension"),
}

# Maximal voluntary isometric joint torques of healthy men, for this body: 84.3 kg, about 1.78 m, taken as 40 years old.
# Each test: (joint, direction (+1: the joint's positive sense), pose (deg; every other joint as in the model's reference
# pose), norm N m, accepted range N m, source). The model's torque is all the agonists' at full activation together.
# Harbo, Brincks & Andersen 2012 (Eur J Appl Physiol 112:267-275, doi:10.1007/s00421-011-1975-3): 93 men, Biodex System 3,
# seated, isometric at neutral joint positions; their regression on age, height and mass (Tables 1-2) is evaluated for this
# body, and the range is their 95% prediction interval (+-1.96 residual SD, Tables 3-4).
# Anderson, Madigan & Nussbaum 2007 (J Biomech 40:3105-3113, doi:10.1016/j.jbiomech.2007.03.022): peak isometric torque as a
# fraction of body weight x height (C1), times 84.3 kg x 9.81 x 1.78 m = 1472 N m.
STRENGTH_TESTS: dict[str, tuple] = {
    # Stafilidis & Sickinger 2019 (PLoS One 14:e0219840): knee straight, ankle neutral, 14 men of 77 kg, 127-173 N m with
    # the seat's position; Anderson 2007: C1 0.095 -> 140 N m (knee 50 deg flexed, the gastrocnemii shortened); Harbo 2012
    # isokinetic at 60 deg/s -> 120 N m (below isometric)
    "ankle plantarflexion": ("ankle_angle_r", -1, {"ankle_angle_r": 0}, 160.0, (110.0, 220.0)),
    "ankle dorsiflexion": ("ankle_angle_r", +1, {"ankle_angle_r": 0}, 44.0, (26.0, 62.0)),  # Harbo 2012 ("ankle extension")
    # Harbo 2012 (265, 173-356); Sarabon, Kozinc & Perman 2021 meta-analysis (Front Physiol 12:767941): men 3.19 N m/kg at
    # 50-70 deg knee flexion, hip ~90 deg -> 269 N m
    "knee extension": ("knee_angle_r", -1, {"knee_angle_r": 60, "hip_flexion_r": 90}, 265.0, (173.0, 356.0)),
    # Sarabon 2021: men 1.39 N m/kg mid-range -> 117 N m; Anderson 2007: C1 0.087 -> 128 N m
    "knee flexion": ("knee_angle_r", +1, {"knee_angle_r": 45, "hip_flexion_r": 90}, 120.0, (80.0, 170.0)),
    # Anderson 2007: standing, peak near 53 deg hip flexion, C1 0.161 -> 237 N m; Harbo 2012 isokinetic -> 191 N m
    "hip extension": ("hip_flexion_r", -1, {"hip_flexion_r": 50, "knee_angle_r": 10}, 237.0, (160.0, 320.0)),
    # Harbo 2012 (167, 111-223); Anderson 2007: C1 0.113 -> 166 N m, peak near neutral, standing
    "hip flexion": ("hip_flexion_r", +1, {"hip_flexion_r": 0}, 167.0, (111.0, 223.0)),
    # Whiler et al. 2017 (Physiother Can 69:212-216): side-lying, 0 deg, Biodex; men 140 +- 44 N m at 73.6 kg -> 1.90 N m/kg
    "hip abduction": ("hip_adduction_r", -1, {"hip_adduction_r": 0}, 160.0, (100.0, 230.0)),
    # Anders et al. 2026 (J Funct Morphol Kinesiol 11:212): standing, 124 men of 78.5 kg, 241 N m; seated with the pelvis
    # fixed, men read about half that (Pan et al. 2025, Eur J Med Res 30:471: 118 N m), hence the wide range
    "trunk extension": ("flex_extension", +1, {"flex_extension": 0}, 240.0, (150.0, 330.0)),
    # Pan et al. 2025: seated, flexion 87 N m = 0.74 of extension; applied to the standing extension: 175 N m (inferred)
    "trunk flexion": ("flex_extension", -1, {"flex_extension": 0}, 175.0, (87.0, 260.0)),
    # Aronis et al. 2025 (J Funct Morphol Kinesiol 11:20): standing, elbow straight, 90 deg elevation, men of 88 kg, 77 +- 14
    "shoulder flexion": ("shoulder_elv_r", +1, {"elv_angle_r": 90, "shoulder_elv_r": 90}, 77.0, (50.0, 105.0)),
    # Harbo 2012 (65, 41-89); posture "neutral": taken as 45 deg elevation in the scapular plane
    "shoulder abduction": ("shoulder_elv_r", +1, {"elv_angle_r": 30, "shoulder_elv_r": 45}, 65.0, (41.0, 89.0)),
    # elbow at 90 deg, forearm neutral: Harbo 2012 53 N m (29-77); men pooled in Kotte et al. 2018 (Shoulder Elbow
    # 10:207-215) 77 +- 15 N m
    "elbow flexion": ("elbow_flex_r", +1, {"elbow_flex_r": 90}, 65.0, (40.0, 95.0)),
    # Kotte et al. 2018: Askew 1987 41 +- 11 N m, men pooled 48 +- 6 N m; Harbo 2012 isokinetic -> 50 N m
    "elbow extension": ("elbow_flex_r", -1, {"elbow_flex_r": 90}, 45.0, (30.0, 65.0)),
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


@dataclass(frozen=True)
class Palm:
    """The middle of a palm, where a hand rests on its thigh: a point on the hand and the way the palm faces (hand frame)."""
    hand: int  # the body it rides on (the third metacarpal)
    local: np.ndarray  # position in that body's frame
    normal: np.ndarray  # out of the palm, in that body's frame
    femur: int  # the thigh it can rest on, the same side's


@dataclass(frozen=True)
class Thigh:
    """The front of a thigh, in the femur's frame (x forward, y up the bone, z to the body's right): the soft tissue is the
    femur's two collision capsules, 8 cm in radius at the top and 6.7 cm above the knee. front(y) is the point on the
    front surface at height y (m, negative down the bone; the knee joint is at y = -0.40)."""
    radius_at: tuple  # ((y top, y low), (radius there, radius there)): the surface tapers linearly between the capsules
    knee_y: float

    def front(self, y: float) -> np.ndarray:
        return np.array([float(np.interp(y, self.radius_at[0][::-1], self.radius_at[1][::-1])), y, 0.0])


def load_model(calibrated: bool = True) -> mujoco.MjModel:
    """The body. calibrated: with its muscle strengths matched to people's (calibrate_strength); False gives the model as
    published."""
    m = mujoco.MjModel.from_xml_path(str(musclemimic_models.get_xml_path("myofullbody")))
    if calibrated:
        calibrate_strength(m)
    return m


def roots_of(roots: tuple[str, ...], muscles: list[str]) -> np.ndarray:
    """Indices of the muscles whose root is among roots (a root ending in '_' matches as a prefix, as in REGIONS)."""
    out = []
    for i, n in enumerate(muscles):
        r = _root(n)[0]
        if any(r == k or (k.endswith("_") and r.startswith(k)) or re.fullmatch(re.escape(k) + r"\d*", r) for k in roots):
            out.append(i)
    return np.array(out, dtype=int)


def calibrate_strength(m: mujoco.MjModel) -> dict:
    """Scales each muscle group's force so the body's maximal joint torques are those of a healthy man of its size.

    The published strengths are uneven against people (see STRENGTH). Each group's peak isometric force (gainprm[2]) and, in
    proportion, its passive force scale (biasprm[2], which MuJoCo keeps apart from the active one; MyoSuite's converter
    writes the same muscle parameters to both) are multiplied by the group's factor, so a muscle's passive tension stays the
    same share of its strength. Lengths are untouched, so calibrate_rest's onsets, the force-length curves and every moment
    arm stay as they were. Returns {group: factor}.
    """
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    seen = np.zeros(m.nu, dtype=bool)
    for group, (roots, factor, _why) in STRENGTH.items():
        idx = roots_of(roots, names)
        if len(idx) == 0 or seen[idx].any():
            raise ValueError(f"strength group {group!r} matches no muscle or overlaps another group")
        seen[idx] = True
        m.actuator_gainprm[idx, 2] *= factor
        m.actuator_biasprm[idx, 2] *= factor
    return {g: v[1] for g, v in STRENGTH.items()}


def calibrate_rest(m: mujoco.MjModel, q_rest: np.ndarray, onset: float = 1.0) -> dict:
    """Moves each muscle's passive onset so that, in relaxed standing (q_rest), no muscle is past it.

    The published model puts many muscles beyond their passive onset in its own standing pose (4,174 N of passive pull in
    all, the rectus abdominis at 104% of its maximum force and each middle deltoid at 62%, with zero activation), which is
    harmless for training walking policies but wrong for a body meant to feel relaxation. MuJoCo calibrates a muscle's
    passive part (biasprm) and active part (gainprm) separately, so only the passive range is shifted: strength is untouched
    here (calibrate_strength sets it), and stretching beyond rest still builds passive tension. Returns {muscle index:
    shift} for the muscles moved.
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
        # the palms: 2 cm in front of the third metacarpal's middle (the palm's soft tissue); the model mirrors the left
        # hand's frame, so the left palm faces the hand's +z and the right palm its -z
        self.palm = {side: Palm(self.body[f"thirdmc_{side[0]}"], np.array([0.0, 0.0, 0.02 * sgn]), np.array([0.0, 0.0, sgn]),
                                self.body[f"femur_{side[0]}"]) for side, sgn in (("left", 1.0), ("right", -1.0))}
        self.thigh = self._thigh()

    def _thigh(self) -> Thigh:
        m = self.m
        caps = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"l_femur{k}_col") for k in (1, 2)]
        knee = m.body_pos[self.body["tibia_l"]]
        return Thigh(((float(m.geom_pos[caps[0], 1]), float(m.geom_pos[caps[1], 1])),
                      (float(m.geom_size[caps[0], 0]), float(m.geom_size[caps[1], 0]))), float(knee[1]))

    def couple(self, qpos: np.ndarray, couplings: list | None = None) -> None:
        """Sets every coupled joint from its master, in place (the spine's vertebrae, the shoulder rhythm, the patellae);
        only the couplings given, when given."""
        q0 = self.m.qpos0
        for _ in range(2):  # chains of couplings settle in two passes
            for a1, a2, c in self.couplings if couplings is None else couplings:
                x = qpos[a2] - q0[a2] if a2 >= 0 else 0.0
                qpos[a1] = q0[a1] + c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))

    def joint_torques(self, q: np.ndarray, joint: str) -> tuple[np.ndarray, np.ndarray]:
        """Each muscle's torque about a joint, held still at pose q: (at full activation, its active part; at rest, its
        passive part), N m, positive in the joint's positive direction. Couplings are followed: a muscle crossing the knee
        also pulls on the patella and the knee's coupled translations, and all of it counts toward the knee's torque
        (by virtual work, through the couplings' slopes)."""
        m = self.m
        d = getattr(self, "_d", None) or mujoco.MjData(m)
        self._d = d
        h = 1e-4
        qp, qm = q.copy(), q.copy()
        qp[self.qadr[joint]] += h
        qm[self.qadr[joint]] -= h
        self.couple(qp)
        self.couple(qm)
        slope = np.zeros(m.nv)  # each dof's velocity per unit of the joint's
        for j in range(m.njnt):
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                slope[m.jnt_dofadr[j]] = (qp[m.jnt_qposadr[j]] - qm[m.jnt_qposadr[j]]) / (2 * h)
        d.qpos[:] = q
        self.couple(d.qpos)
        mujoco.mj_fwdPosition(m, d)
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        arm = moment @ slope
        gain = np.array([mujoco.mju_muscleGain(d.actuator_length[i], 0.0, m.actuator_lengthrange[i], m.actuator_acc0[i],
                                               m.actuator_gainprm[i, :9]) for i in range(m.nu)])
        bias = np.array([mujoco.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i], m.actuator_acc0[i],
                                               m.actuator_biasprm[i, :9]) for i in range(m.nu)])
        return arm * gain, arm * bias

    def strength_pose(self, angles: dict) -> np.ndarray:
        """A pose for a strength test: the model's reference pose with the given joints set (degrees), couplings followed."""
        q = self.m.qpos0.copy()
        for k, v in angles.items():
            q[self.qadr[k]] = np.radians(v)
        self.couple(q)
        return q

    def max_torque(self, test: str) -> tuple[float, float]:
        """A strength test of STRENGTH_TESTS on the right side: the largest isometric torque the agonists make together at
        full activation, and the passive torque in the same direction (N m)."""
        joint, sign, angles = STRENGTH_TESTS[test][:3]
        active, passive = self.joint_torques(self.strength_pose(angles), joint)
        active = sign * active
        return float(active[active > 0].sum()), float(sign * passive.sum())

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

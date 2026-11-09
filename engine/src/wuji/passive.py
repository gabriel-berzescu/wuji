"""What holds the joints besides the muscles: ligaments, joint capsules, fascia, discs, and the muscles' own connective
tissue stretched.

The model has no ligaments, and MuJoCo's passive muscle curve gives a muscle about 2% of its strength at the end of a
joint's range (anatomy.calibrate_rest also moved the passive onsets to relaxed standing), so without these the muscles
carry everything, even where a real body hangs on its tissues: the back near full flexion (flexion-relaxation), the hip in
extension on its iliofemoral ligament, the hamstrings in a toe-touch. Each element here is a measured passive
moment-angle relation acting on one or more joints. It reaches the muscle solver as a generalised force, as the
abdominal pressure does, and it is felt as stretch in a region, in proportion to how near the end of its range it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .anatomy import Anatomy

STRETCH_FELT = 0.1  # an element at the end of its range is felt like a muscle group pulling a tenth of its strength


@dataclass(frozen=True)
class Felt:
    """One element now: its moment (N m, positive the way it pushes, described by the element) and how near the end of
    its range it is (0 slack .. 1 at the end), felt as stretch in a region."""
    name: str
    region: str
    moment: float
    share: float


class LumbarFlexion:
    """The back of the lumbar spine resists folding: its ligaments (supraspinous, interspinous, the ligamenta flava, the
    capsules), the thoracolumbar fascia, the discs, and the stretched back muscles' own connective tissue. Little at first,
    steeply near the end of the range: near full flexion they carry the trunk and the back muscles fall silent (the
    flexion-relaxation phenomenon: Floyd and Silver 1955; McGill and Kippers 1994; Dolan, Mannion and Adams 1994, who put
    the passive share of the extensor moment at full flexion near 100 N m). One element, the same moment across each
    lumbar level:
        M = A (exp(B x) - 1), x = lumbar flexion (L1-S1) / its standing maximum;
    about 10 N m at half the range, 45 at four fifths, 120 at full flexion."""
    name = "lumbar flexion"
    region = "lower_back"
    JOINTS = ("flex_extension", "L4_L5_FE", "L3_L4_FE", "L2_L3_FE", "L1_L2_FE")
    FLEX_MAX = 55.0  # degrees, L1-S1 in standing full flexion
    A, B = 1.49, 4.4  # N m, steepness
    FULL = A * (math.exp(B) - 1.0)  # N m at full flexion

    def flexion(self, a: Anatomy, q: np.ndarray) -> float:
        """How far the lumbar spine is folded, L1 to the sacrum (degrees, flexion positive)."""
        return -float(np.degrees(sum(q[a.qadr[j]] for j in self.JOINTS)))

    def __call__(self, a: Anatomy, q: np.ndarray, tau: np.ndarray) -> list[Felt]:
        x = max(0.0, self.flexion(a, q)) / self.FLEX_MAX
        moment = self.A * (math.exp(self.B * min(x, 1.2)) - 1.0)
        for j in self.JOINTS:  # extension is positive on these joints
            tau[a.dadr[j]] += moment
        return [Felt(self.name, self.region, moment, min(1.0, moment / self.FULL))]


class LumbarBendAndTwist:
    """The trunk's passive resistance to bending sideways and to twisting, measured in vivo with the muscles relaxed
    (McGill, Seguin and Bennett 1994, Spine 19:696; exponentials fitted to their stiffnesses as tabulated by Watt et al.
    2022):
        side bending  M = 2.44 (exp(0.106 deg) - 1): 4.6 N m at 10 degrees, 18 at 20;
        twisting      M = 1.55 (exp(0.0724 deg) - 1): 2.2 N m at 12 degrees, 7.3 at 24.
    These are the whole trunk's; this body's chest is rigid, so all its bending and twisting happens in the lumbar joints,
    and the measured curves act there, the same moment across each level, against the way it is bent. Side bending is felt
    in the lower back (quadratus lumborum, the fascia), twisting along the belly (the obliques)."""
    BEND = ("lat_bending", "L4_L5_LB", "L3_L4_LB", "L2_L3_LB", "L1_L2_LB")
    TWIST = ("axial_rotation", "L4_L5_AR", "L3_L4_AR", "L2_L3_AR", "L1_L2_AR")
    BEND_FIT, TWIST_FIT = (2.44, 0.106), (1.55, 0.0724)  # N m, per degree
    BEND_END, TWIST_END = 25.0, 30.0  # degrees: the end of each range, where the share felt reaches 1

    def __call__(self, a: Anatomy, q: np.ndarray, tau: np.ndarray) -> list[Felt]:
        felt = []
        for name, joints, (k, b), end, region in (
                ("lumbar side bending", self.BEND, self.BEND_FIT, self.BEND_END, "lower_back"),
                ("lumbar twisting", self.TWIST, self.TWIST_FIT, self.TWIST_END, "belly")):
            angle = float(np.degrees(sum(q[a.qadr[j]] for j in joints)))
            moment = k * (math.exp(b * min(abs(angle), 1.3 * end)) - 1.0)
            for j in joints:
                tau[a.dadr[j]] -= math.copysign(moment, angle)
            felt.append(Felt(name, region, moment, min(1.0, moment / (k * (math.exp(b * end) - 1.0)))))
        return felt


def riener_edrich(hip: float, knee: float, ankle: float) -> tuple[float, float, float]:
    """Passive hip, knee and ankle moments (N m) of the relaxed leg: Riener and Edrich 1999, J Biomech 32:539,
    doi:10.1016/S0021-9290(99)00009-3 (coefficients as reproduced identically by Gasparutto, Jacquelin and Dumas 2018 and
    by Prasad et al. 2025). Degrees: hip flexion, knee flexion and ankle PLANTARflexion positive; each moment positive
    toward its joint's positive angle. The joints are coupled through the two-joint muscles: the hamstrings pull the knee
    into flexion as the hip folds, the gastrocnemius as the ankle dorsiflexes. Not zero at neutral, rightly: a relaxed hip
    rests near 27 degrees of flexion, a knee near 18, a foot near 20 of plantarflexion."""
    h, k, a = float(np.clip(hip, -30, 110)), float(np.clip(knee, -5, 130)), float(np.clip(ankle, -40, 50))
    m_hip = math.exp(1.4655 - 0.0034 * k - 0.0750 * h) - math.exp(1.3403 - 0.0226 * k + 0.0305 * h) + 8.072
    m_knee = (math.exp(1.800 - 0.0460 * a - 0.0352 * k + 0.0217 * h)
              - math.exp(-3.971 - 0.0004 * a + 0.0495 * k - 0.0128 * h) - 4.820 + math.exp(2.220 - 0.150 * k))
    m_ankle = math.exp(2.1016 - 0.0843 * a - 0.0176 * k) - math.exp(-7.9763 + 0.1949 * a + 0.0008 * k) - 1.792
    return m_hip, m_knee, m_ankle


def hip_abduction_moment(abduction: float, flexion: float) -> float:
    """The relaxed hip's passive abducting moment (N m), abduction and hip flexion in degrees: Amankwah, Triolo and Kirsch
    2004, J Rehabil Res Dev 41:15, doi:10.1682/JRRD.2004.01.0015, their able-bodied fit (measured from about 3 degrees of
    adduction to 35 of abduction; beyond that it is extrapolated)."""
    r = math.pi / 180.0
    ab, fl = float(np.clip(abduction, -20, 40)), float(np.clip(flexion, -10, 100))
    return 16.0 * math.exp((-4.5 * ab - 0.27 * fl) * r) - 4.8 * math.exp((2.3 * ab - 0.021 * fl) * r)


class Legs:
    """Each leg's passive tissues: Riener and Edrich's hip, knee and ankle moments (the capsules and ligaments, the
    muscles' own stretch, the two-joint hamstrings and gastrocnemius), and the hip's passive resistance to adduction and
    abduction (the iliotibial band, the capsule, the abductors stretched), Amankwah's fit as measured: it rests near 10
    degrees of abduction (11 N m abducting at neutral), as the hip's resting position is abducted, just as Riener and
    Edrich's hip, knee and ankle rest away from zero.

    Felt as stretch where the tissue is drawn long: the back of the thigh as the hip folds (the hamstrings), the front of
    the hip as it extends (the iliofemoral ligament, the hip flexors), the calves as the ankle dorsiflexes, the front of the
    thigh as the knee bends deeply, the outside of the hip in adduction; each measured from how it is in quiet standing,
    so standing itself is not felt as stretch.

    The ankle's curve is taken at half (ANKLE). Riener and Edrich measured young men lying relaxed; at full strength their
    calves would carry quiet standing almost alone (16 of the ~20 N m each ankle needs), leaving the soleus at 1-3% of its
    strength where standing people's works at 5-15% (EMG), and in a horse stance, the shins inclined near 30 degrees, they
    would out-pull the load so that the shins (tibialis anterior) had to hold at 60%, which practitioners do not feel.
    At half, the soleus works in quiet standing as in people, and the horse stance rests in the thighs."""
    ANKLE = 0.5  # the share of Riener and Edrich's ankle moment applied (see above)
    # felt stretch: (moment in quiet standing, moment at the end of the range), N m, signed as each Felt's moment
    HAMSTRINGS = (0.0, 72.0)  # -M_hip: the hip folded 100 degrees, knees straight
    HIP_FRONT = (8.9, 47.6)  # M_hip: the hip extended 30 degrees
    CALVES = (7.8, 50.0)  # M_ankle (at ANKLE), plantarflexing: the ankle dorsiflexed 30 degrees
    THIGH_FRONT = (0.0, 30.0)  # -M_knee: the knee bent deeply
    HIP_SIDE = (11.2, 50.0)  # the abducting moment: the hip adducted about 12 degrees

    def __call__(self, a: Anatomy, q: np.ndarray, tau: np.ndarray) -> list[Felt]:
        felt = []
        share = lambda m, ref: max(0.0, min(1.0, (m - ref[0]) / (ref[1] - ref[0])))
        for s in ("l", "r"):
            hip = float(np.degrees(q[a.qadr[f"hip_flexion_{s}"]]))
            knee = float(np.degrees(q[a.qadr[f"knee_angle_{s}"]]))
            dorsi = float(np.degrees(q[a.qadr[f"ankle_angle_{s}"]]))  # this body: dorsiflexion positive
            m_hip, m_knee, m_ankle = riener_edrich(hip, knee, -dorsi)
            m_ankle *= self.ANKLE
            tau[a.dadr[f"hip_flexion_{s}"]] += m_hip
            tau[a.dadr[f"knee_angle_{s}"]] += m_knee
            tau[a.dadr[f"ankle_angle_{s}"]] -= m_ankle  # plantarflexing is negative on this body's ankle
            adduction = float(np.degrees(q[a.qadr[f"hip_adduction_{s}"]]))
            m_abd = hip_abduction_moment(-adduction, hip)
            tau[a.dadr[f"hip_adduction_{s}"]] -= m_abd  # abducting is negative on this body's hip
            side = "left" if s == "l" else "right"
            felt += [Felt(f"{side} hamstrings", "thighs", -m_hip, share(-m_hip, self.HAMSTRINGS)),
                     Felt(f"{side} hip front", "hips", m_hip, share(m_hip, self.HIP_FRONT)),
                     Felt(f"{side} calf", "calves", m_ankle, share(m_ankle, self.CALVES)),
                     Felt(f"{side} thigh front", "thighs", -m_knee, share(-m_knee, self.THIGH_FRONT)),
                     Felt(f"{side} hip side", "hips", m_abd, share(m_abd, self.HIP_SIDE))]
        return felt


class PassiveTissues:
    """All the passive elements: tissues(q) -> (generalised forces, nv; what each element is doing)."""

    def __init__(self, anatomy: Anatomy, elements: list | None = None) -> None:
        self.a = anatomy
        self.lumbar = LumbarFlexion()
        self.elements = elements if elements is not None else [self.lumbar, LumbarBendAndTwist(), Legs()]

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, list[Felt]]:
        tau = np.zeros(self.a.m.nv)
        felt = [f for e in self.elements for f in e(self.a, q, tau)]
        return tau, felt

    @staticmethod
    def stretch(felt: list[Felt]) -> dict[str, float]:
        """region -> stretch felt from the passive tissues (in the felt sense's units: a share of strength)."""
        out: dict[str, float] = {}
        for f in felt:
            out[f.region] = out.get(f.region, 0.0) + STRETCH_FELT * f.share
        return out

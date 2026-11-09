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

    def __call__(self, a: Anatomy, q: np.ndarray, tau: np.ndarray) -> Felt:
        x = max(0.0, self.flexion(a, q)) / self.FLEX_MAX
        moment = self.A * (math.exp(self.B * min(x, 1.2)) - 1.0)
        for j in self.JOINTS:  # extension is positive on these joints
            tau[a.dadr[j]] += moment
        return Felt(self.name, self.region, moment, min(1.0, moment / self.FULL))


class PassiveTissues:
    """All the passive elements: tissues(q) -> (generalised forces, nv; what each element is doing)."""

    def __init__(self, anatomy: Anatomy, elements: list | None = None) -> None:
        self.a = anatomy
        self.lumbar = LumbarFlexion()
        self.elements = elements if elements is not None else [self.lumbar]

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, list[Felt]]:
        tau = np.zeros(self.a.m.nv)
        felt = [e(self.a, q, tau) for e in self.elements]
        return tau, felt

    @staticmethod
    def stretch(felt: list[Felt]) -> dict[str, float]:
        """region -> stretch felt from the passive tissues (in the felt sense's units: a share of strength)."""
        out: dict[str, float] = {}
        for f in felt:
            out[f.region] = out.get(f.region, 0.0) + STRETCH_FELT * f.share
        return out

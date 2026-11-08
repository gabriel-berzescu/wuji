"""Fits the strength factors in anatomy.STRENGTH: each muscle group's force multiplier, so the body's maximal isometric joint
torques (anatomy.STRENGTH_TESTS) match those measured in healthy men of its size.

    uv run python tools/fit_strength.py

The model as published is measured first: each test's torque is the sum, over the muscle groups, of the agonists' torques
at full activation (velocity zero, at the test's pose). The factors then minimise the squared log-ratio of model to norm
over all tests, with a weak pull of each factor toward 1 (a group no test constrains keeps its published strength).
Muscles crossing two joints belong to one group with one factor, so a test that loads both (the hamstrings in hip extension
and in knee flexion) is fitted jointly. Prints the factors to copy into STRENGTH, and the before / after table.
"""
import sys

import numpy as np
from scipy.optimize import least_squares

from wuji.anatomy import STRENGTH, STRENGTH_TESTS, Anatomy, load_model, roots_of

sys.stdout.reconfigure(encoding="utf-8")
REGULARISE = 0.15  # weight of log(factor) against a test's log error

m = load_model(calibrated=False)
a = Anatomy(m)
groups = list(STRENGTH)
member = np.full(m.nu, -1)
for g, name in enumerate(groups):
    member[roots_of(STRENGTH[name][0], a.muscles)] = g

# contribution[t, g]: the published model's torque in test t from group g's agonists (N m); other: muscles in no group
tests = list(STRENGTH_TESTS)
contribution = np.zeros((len(tests), len(groups)))
other = np.zeros(len(tests))
for t, test in enumerate(tests):
    joint, sign, angles = STRENGTH_TESTS[test][:3]
    active, _ = a.joint_torques(a.strength_pose(angles), joint)
    active = sign * active
    for i in np.where(active > 0)[0]:
        if member[i] >= 0:
            contribution[t, member[i]] += active[i]
        else:
            other[t] += active[i]
norm = np.array([STRENGTH_TESTS[t][3] for t in tests])


def residual(logf):
    model = contribution @ np.exp(logf) + other
    return np.concatenate([np.log(model / norm), REGULARISE * logf])


fit = least_squares(residual, np.zeros(len(groups)))
factor = np.round(np.exp(fit.x), 2)
before = contribution.sum(axis=1) + other
after = contribution @ factor + other
print("factors (copy into anatomy.STRENGTH):")
for g, f in zip(groups, factor):
    print(f"  {g:22s} {f:.2f}   (now {STRENGTH[g][1]:.2f})")
print(f"\n{'test':32s} {'before':>7s} {'norm':>6s} {'range':>11s} {'after':>7s}")
for t, test in enumerate(tests):
    lo, hi = STRENGTH_TESTS[test][4]
    flag = "" if lo <= after[t] <= hi else "  <- outside"
    print(f"{test:32s} {before[t]:7.0f} {norm[t]:6.0f} {lo:5.0f}-{hi:<5.0f} {after[t]:7.0f}{flag}")

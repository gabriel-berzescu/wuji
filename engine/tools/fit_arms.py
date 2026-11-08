"""Fits the arm shapes in posture.ARMS: the shoulder and elbow angles that put the hand and elbow where they should be.

    uv run python tools/fit_arms.py [shape ...]

Targets are relative to the right shoulder (humerus origin), in metres: forward, outward (away from the midline), up.
The forearm, wrist and finger settings (pro_sup, flexion, curl) are chosen by hand in posture.ARMS.
"""
import sys

import mujoco
import numpy as np
from scipy.optimize import minimize

from wuji.anatomy import FORWARD, LEFTWARD, Anatomy, load_model
from wuji.posture import DEG, Posture, PostureSolver

SHAPES = {
    "hang": dict(hand=(0.03, 0.10, -0.60), elbow=(0.0, 0.04, -0.29), x0=(5, 0, 0, 5)),
    "embrace": dict(hand=(0.40, -0.04, -0.12), elbow=(0.16, 0.15, -0.22), x0=(60, 70, 20, 80)),  # a big ball, hands ~26 cm apart
    "lift_sky": dict(hand=(0.05, -0.06, 0.62), elbow=(0.03, 0.02, 0.30), x0=(160, 90, 0, 5)),
    "press_down": dict(hand=(0.18, 0.12, -0.45), elbow=(0.04, 0.10, -0.28), x0=(15, 60, 0, 30)),
    # the Eight Brocades
    "bow_extend": dict(hand=(0.05, 0.60, 0.02), elbow=(0.02, 0.30, 0.01), x0=(90, 5, 0, 5)),  # arm out to the side, shoulder high
    "bow_pull": dict(hand=(0.20, -0.04, -0.02), elbow=(-0.02, 0.27, -0.02), x0=(85, -10, 30, 140)),  # drawing the string to the chest
    "punch": dict(hand=(0.60, -0.08, -0.12), elbow=(0.31, -0.03, -0.06), x0=(80, 90, 0, 5)),  # fist forward, slowly
    "fist_waist": dict(hand=(0.02, 0.06, -0.42), elbow=(-0.18, 0.08, -0.25), x0=(30, -60, 0, 100)),  # fist resting at the waist
    "earth_press": dict(hand=(0.08, 0.12, -0.52), elbow=(0.0, 0.08, -0.29), x0=(10, 40, 0, 20)),  # one hand presses the earth
    "owl": dict(hand=(-0.02, 0.13, -0.59), elbow=(-0.01, 0.06, -0.29), x0=(8, -10, -30, 5)),  # arms down, turned outward
    "hands_thighs": dict(hand=(0.30, 0.04, -0.52), elbow=(0.10, 0.12, -0.27), x0=(30, 60, 0, 30)),  # resting on the thighs
}
JOINTS = ("shoulder_elv", "elv_angle", "shoulder_rot", "elbow_flex")

m = load_model()
a = Anatomy(m)
ps = PostureSolver(a)
q0 = ps.solve(Posture(sink=0.0), iterations=60)
d = ps.d


def place(x):
    q = q0.copy()
    for k, v in zip(JOINTS, x):
        q[a.qadr[k + "_r"]] = v * DEG
    ps._clip(q)
    a.couple(q)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    shoulder = d.xpos[a.body["humerus_r"]]
    rel = lambda p: np.array([(p - shoulder) @ FORWARD, -(p - shoulder) @ LEFTWARD, (p - shoulder)[2]])
    return rel(d.xpos[a.body["ulna_r"]]), rel(d.xpos[a.body["thirdmc_r"]]), q


for name, s in SHAPES.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    cost = lambda x: np.sum((place(x)[1] - s["hand"]) ** 2) + 0.5 * np.sum((place(x)[0] - s["elbow"]) ** 2)
    r = minimize(cost, s["x0"], method="Nelder-Mead", options={"maxiter": 800, "xatol": 0.5, "fatol": 1e-6})
    _, hand, q = place(r.x)
    angles = {k: round(float(np.degrees(q[a.qadr[k + "_r"]]))) for k in JOINTS}
    print(f"{name:12s} {angles}  hand {np.round(hand * 100).tolist()} cm, error {np.sqrt(r.fun) * 100:.1f} cm")

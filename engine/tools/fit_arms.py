"""Fits the arm shapes in posture.ARMS: the shoulder and elbow angles that put the hand and elbow where they should be.

    uv run python tools/fit_arms.py

Targets are relative to the right shoulder (humerus origin), in metres: forward, outward (away from the midline), up.
"""
import mujoco
import numpy as np
from scipy.optimize import minimize

from wuji.anatomy import FORWARD, LEFTWARD, Anatomy, load_model
from wuji.posture import DEG, Posture, PostureSolver

SHAPES = {
    "hang": dict(hand=(0.03, 0.10, -0.60), elbow=(0.0, 0.04, -0.29), x0=(5, 0, 0, 5)),
    "embrace": dict(hand=(0.38, -0.10, -0.12), elbow=(0.18, 0.12, -0.20), x0=(60, 70, 20, 80)),
    "lift_sky": dict(hand=(0.05, -0.06, 0.62), elbow=(0.03, 0.02, 0.30), x0=(160, 90, 0, 5)),
    "press_down": dict(hand=(0.18, 0.12, -0.45), elbow=(0.04, 0.10, -0.28), x0=(15, 60, 0, 30)),
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
    cost = lambda x: np.sum((place(x)[1] - s["hand"]) ** 2) + 0.5 * np.sum((place(x)[0] - s["elbow"]) ** 2)
    r = minimize(cost, s["x0"], method="Nelder-Mead", options={"maxiter": 600, "xatol": 0.5, "fatol": 1e-6})
    _, hand, q = place(r.x)
    angles = {k: round(float(np.degrees(q[a.qadr[k + "_r"]])), 1) for k in JOINTS}
    print(f"{name:11s} {angles}  hand {np.round(hand * 100).tolist()} cm, error {np.sqrt(r.fun) * 100:.1f} cm")

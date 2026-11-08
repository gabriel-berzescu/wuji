"""Loads the MyoFullBody model and reports what it is made of and how fast it runs on this machine.

    uv run python tools/probe.py
"""
import time
from collections import Counter

import mujoco
import musclemimic_models
import numpy as np

path = musclemimic_models.get_xml_path("myofullbody")
t = time.perf_counter()
m = mujoco.MjModel.from_xml_path(str(path))
print(f"loaded {path.name} in {time.perf_counter() - t:.2f} s (MuJoCo {mujoco.__version__})")
d = mujoco.MjData(m)

muscle = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
print(f"bodies {m.nbody}, joints {m.njnt}, dofs nv={m.nv}, qpos nq={m.nq}, actuators {m.nu} ({muscle.sum()} muscles), "
      f"tendons {m.ntendon}, geoms {m.ngeom}, sensors {m.nsensor}, timestep {m.opt.timestep * 1000:.1f} ms, "
      f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}")
print("joint types:", dict(Counter(mujoco.mjtJoint(j).name for j in m.jnt_type)))
free = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
print("free joints:", free)

# body regions, from the body names the muscles pull on: a rough map of where the 416 are
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("first actuators:", names[:12])
print("last actuators:", names[-12:])
bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
print("bodies:", ", ".join(bodies))

# speed: a standing body doing nothing, stepped forward; then inverse dynamics and moment arms
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
n = 500
t = time.perf_counter()
for _ in range(n):
    mujoco.mj_step(m, d)
dt = (time.perf_counter() - t) / n
print(f"mj_step: {dt * 1e3:.3f} ms per step = {m.opt.timestep / dt:.1f}x real time; ncon after {n} steps: {d.ncon}")

mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
t = time.perf_counter()
for _ in range(n):
    mujoco.mj_inverse(m, d)
print(f"mj_inverse: {(time.perf_counter() - t) / n * 1e3:.3f} ms")

# actuator moment arms (nu x nv), what static optimisation needs
t = time.perf_counter()
for _ in range(50):
    mujoco.mj_fwdPosition(m, d)
    moment = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
print(f"moment arms: {(time.perf_counter() - t) / 50 * 1e3:.3f} ms, matrix {moment.shape}, nonzero {np.count_nonzero(moment)}")

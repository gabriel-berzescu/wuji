"""Prints the body's joints (with ranges), keyframes, ground and foot geometry, and the muscles grouped by the body they act on.

    uv run python tools/anatomy.py
"""
from collections import defaultdict

import mujoco
import musclemimic_models
import numpy as np

m = mujoco.MjModel.from_xml_path(str(musclemimic_models.get_xml_path("myofullbody")))
d = mujoco.MjData(m)
name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or f"#{i}"

print("== joints (name, type, range deg/m, qpos0, body)")
for j in range(m.njnt):
    t = mujoco.mjtJoint(m.jnt_type[j]).name.replace("mjJNT_", "")
    lo, hi = m.jnt_range[j]
    adr = m.jnt_qposadr[j]
    scale = 1 if t == "SLIDE" else 180 / np.pi
    rng = f"[{lo * scale:.0f}, {hi * scale:.0f}]" if m.jnt_limited[j] else "free"
    q0 = m.qpos0[adr] * scale if t in ("HINGE", "SLIDE") else m.qpos0[adr:adr + 7].round(3).tolist()
    print(f"  {name(mujoco.mjtObj.mjOBJ_JOINT, j)}: {t} {rng} q0={q0 if not isinstance(q0, float) else round(q0, 1)} body={name(mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j])}")

print("== keyframes:", m.nkey, [name(mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)])
print("== equality constraints:", m.neq, [name(mujoco.mjtObj.mjOBJ_EQUALITY, e) for e in range(m.neq)][:20])

print("== geoms that collide (contype/conaffinity nonzero)")
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        print(f"  {name(mujoco.mjtObj.mjOBJ_GEOM, g)}: {mujoco.mjtGeom(m.geom_type[g]).name} size={m.geom_size[g].round(3).tolist()} "
              f"body={name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])} contype={m.geom_contype[g]} conaff={m.geom_conaffinity[g]}")

mujoco.mj_forward(m, d)
print("== standing height at qpos0: pelvis z", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].round(3),
      "head z", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")].round(3),
      "calcn_r", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")].round(3),
      "toes_r", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "toes_r")].round(3))
print("total mass", m.body_mass.sum().round(1), "kg")

print("== muscles by the joints they cross (first joint's body)")
mujoco.mj_fwdPosition(m, d)
moment = np.zeros((m.nu, m.nv))
mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
dof_body = [name(mujoco.mjtObj.mjOBJ_BODY, m.dof_bodyid[v]) for v in range(m.nv)]
groups = defaultdict(list)
for a in range(m.nu):
    bodies = sorted({dof_body[v] for v in np.nonzero(moment[a])[0]})
    groups[" + ".join(bodies[:3]) + (" ..." if len(bodies) > 3 else "")].append(name(mujoco.mjtObj.mjOBJ_ACTUATOR, a))
for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"  [{len(v)}] {k}: {', '.join(v[:8])}{' ...' if len(v) > 8 else ''}")
print("== muscle properties: F0 range", m.actuator_gainprm[:, 2].min().round(1), "-", m.actuator_gainprm[:, 2].max().round(1),
      "| ctrlrange", m.actuator_ctrlrange[0].tolist(), "| dynprm (tau act/deact)", m.actuator_dynprm[0, :2].tolist())

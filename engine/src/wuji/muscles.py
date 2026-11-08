"""Who does the work: the muscle activations that produce a motion, found by static optimisation.

For each moment, inverse dynamics gives the generalised forces the motion needs (MuJoCo's mj_inverse, with every constraint
switched off so nothing is hidden in them). A quadratic programme (OSQP) then finds, all at once:

- the activation of each of the 416 muscles, between what the practitioner holds (tension) and 1,
- the ground's force at each loaded point of the soles, inside a friction pyramid,
- reserve forces for anything the muscles cannot do (heavily penalised: their size says how strained the posture is),

The joint couplings (the vertebrae following the lumbar spine, the shoulder rhythm, the patellae) are folded into the
equations rather than given force variables: everything is expressed in the independent coordinates, so a force on a coupled
joint reaches the joint that drives it through the coupling's slope (q_dot = T q_dot_independent; the equations are T^T(...)).
Ground forces are solved in hectonewtons, so all the variables have similar sizes and the solver converges quickly.

minimising the sum of squared activations (the classic physiological criterion, Crowninshield & Brand 1981; also qigong's
song: no more effort than the posture needs) plus a little change from the last moment (activation is not instantaneous).
A muscle's force is linear in its activation at a given length and velocity: F = gain(L, V) * a + bias(L) (MuJoCo's Hill
model), so the problem is a true QP. This follows MyoSuite's inverse-dynamics tutorial, with the contacts added as variables.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import osqp
from scipy import sparse

from .anatomy import Anatomy

MU = 0.9  # sole friction


@dataclass
class Solution:
    act: np.ndarray  # activations (nu)
    force: np.ndarray  # muscle forces, N (positive = pulling)
    grf: dict  # foot -> {"force": world N (3,), "cop": world point (3,)} (absent when the foot bears nothing)
    root_residual: float  # N or N m: what had to hold the body up besides muscles and ground (should be ~0)
    joint_residual: float  # largest reserve torque on a joint, N m
    effort: float  # mean squared activation
    ok: bool


class MuscleSolver:
    W_ACT = 1.0
    W_SMOOTH = 0.5
    W_FORCE = 1e-3  # per hN squared: just enough to spread the load over the sole
    W_ROOT = 1e2
    W_RESERVE = 1.0
    F_UNIT = 100.0  # ground forces are solved in hectonewtons

    def __init__(self, anatomy: Anatomy) -> None:
        self.a = anatomy
        self.m = anatomy.m
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.prev = np.zeros(m.nu)
        self.couplings = []  # (slave dof, master dof or -1, master qpos address, polynomial coefficients)
        for e in range(m.neq):
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            self.couplings.append((int(m.jnt_dofadr[j1]), int(m.jnt_dofadr[j2]) if j2 >= 0 else -1,
                                   int(m.jnt_qposadr[j2]) if j2 >= 0 else -1, m.eq_data[e, :5].copy()))
        slaves = {c[0] for c in self.couplings}
        self.free = np.array([v for v in range(m.nv) if v not in slaves])  # independent coordinates

    def _reduction(self, qpos: np.ndarray) -> np.ndarray:
        """T (nv x n_free): full joint velocities from the independent ones, through the couplings' slopes."""
        m = self.m
        rows = {v: np.eye(len(self.free))[k] for k, v in enumerate(self.free)}
        pending = list(self.couplings)
        while pending:
            rest = []
            for s, mdof, madr, c in pending:
                if mdof < 0:
                    rows[s] = np.zeros(len(self.free))
                elif mdof in rows:
                    x = qpos[madr] - m.qpos0[madr]
                    rows[s] = (c[1] + x * (2 * c[2] + x * (3 * c[3] + x * 4 * c[4]))) * rows[mdof]
                else:
                    rest.append((s, mdof, madr, c))
            if len(rest) == len(pending):
                raise RuntimeError("circular joint couplings")
            pending = rest
        return np.array([rows[v] for v in range(m.nv)])

    def solve(self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray, floor: np.ndarray | None = None,
              loaded: dict | None = None) -> Solution:
        """floor: per-muscle lower bound on activation (tension held). loaded: foot -> bool, whether the foot is on the ground."""
        m, d, a = self.m, self.d, self.a
        nu, nv = m.nu, m.nv
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos, qvel, qacc
        saved = m.opt.disableflags
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        mujoco.mj_inverse(m, d)
        m.opt.disableflags = saved
        tau = d.qfrc_inverse.copy()

        # muscles: generalised force = M^T (gain * a + bias)
        moment = np.zeros((nu, nv))
        mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        gain = np.empty(nu)
        bias = np.empty(nu)
        for i in range(nu):
            prm = m.actuator_gainprm[i, :9]
            lr = m.actuator_lengthrange[i]
            gain[i] = mujoco.mju_muscleGain(d.actuator_length[i], d.actuator_velocity[i], lr, m.actuator_acc0[i], prm)
            bias[i] = mujoco.mju_muscleBias(d.actuator_length[i], lr, m.actuator_acc0[i], m.actuator_biasprm[i, :9])
        A_mus = moment.T * gain
        b_mus = moment.T @ bias

        # ground: the loaded sole points
        pts = [p for p in a.sole if (loaded or {}).get(p.foot, True)]
        jacs, worlds = [], []
        jp = np.zeros((3, nv))
        for p in pts:
            w = d.xpos[p.body] + d.xmat[p.body].reshape(3, 3) @ p.local
            mujoco.mj_jac(m, d, jp, None, w, p.body)
            jacs.append(jp.copy())
            worlds.append(w)
        nc = len(pts)

        # everything in the independent coordinates: T^T (A_mus a + Jc^T f + ...) + r = T^T (tau - b_mus)
        T = self._reduction(qpos)
        nr = T.shape[1]
        # variables: [a (nu), f in hN (3 nc), r (nr)]
        nf = 3 * nc
        n = nu + nf + nr
        Jc = np.hstack([J.T for J in jacs]) * self.F_UNIT if nc else np.zeros((nv, 0))
        A_eq = sparse.hstack([sparse.csc_matrix(T.T @ A_mus), sparse.csc_matrix(T.T @ Jc), sparse.identity(nr)])
        rhs = T.T @ (tau - b_mus)

        rows, lo, hi = [A_eq], [rhs], [rhs]
        # friction pyramids
        if nc:
            fr = sparse.lil_matrix((5 * nc, n))
            for k in range(nc):
                ix, iy, iz = nu + 3 * k, nu + 3 * k + 1, nu + 3 * k + 2
                fr[5 * k, iz] = 1.0
                fr[5 * k + 1, ix], fr[5 * k + 1, iz] = 1.0, -MU
                fr[5 * k + 2, ix], fr[5 * k + 2, iz] = -1.0, -MU
                fr[5 * k + 3, iy], fr[5 * k + 3, iz] = 1.0, -MU
                fr[5 * k + 4, iy], fr[5 * k + 4, iz] = -1.0, -MU
            rows.append(fr.tocsc())
            lo.append(np.tile([0, -np.inf, -np.inf, -np.inf, -np.inf], nc))
            hi.append(np.tile([np.inf, 0, 0, 0, 0], nc))
        # activation bounds
        rows.append(sparse.hstack([sparse.identity(nu), sparse.csc_matrix((nu, n - nu))]))
        lo.append(np.zeros(nu) if floor is None else np.clip(floor, 0, 1))
        hi.append(np.ones(nu))

        w_r = np.full(nr, self.W_RESERVE)
        w_r[:6] = self.W_ROOT  # the root's six coordinates are the first independent ones
        P = sparse.diags(np.concatenate([np.full(nu, 2 * (self.W_ACT + self.W_SMOOTH)), np.full(nf, 2 * self.W_FORCE),
                                         2 * w_r])).tocsc()
        q = np.concatenate([-2 * self.W_SMOOTH * self.prev, np.zeros(nf + nr)])
        A = sparse.vstack(rows).tocsc()
        solver = osqp.OSQP()
        solver.setup(P, q, A, np.concatenate(lo), np.concatenate(hi), verbose=False, polishing=True, eps_abs=1e-5,
                     eps_rel=1e-5, max_iter=20000, warm_starting=True, adaptive_rho_interval=25)
        res = solver.solve(raise_error=False)
        self.status = res.info.status
        ok = res.info.status in ("solved", "solved inaccurate") and res.x is not None and np.all(np.isfinite(res.x))
        x = res.x if ok else np.zeros(n)
        act = np.clip(x[:nu], 0, 1)
        f = x[nu:nu + nf].reshape(-1, 3) * self.F_UNIT
        r = x[nu + nf:]
        if ok:
            self.prev = act

        grf = {}
        for foot in ("left", "right"):
            idx = [k for k, p in enumerate(pts) if p.foot == foot]
            if not idx:
                continue
            F = f[idx].sum(axis=0)
            fz = f[idx, 2]
            cop = (np.array(worlds)[idx] * fz[:, None]).sum(axis=0) / max(fz.sum(), 1e-9)
            grf[foot] = {"force": F, "cop": cop}
        force = -(gain * act + bias)
        return Solution(act=act, force=force, grf=grf, root_residual=float(np.linalg.norm(r[:6])),
                        joint_residual=float(np.abs(r[6:]).max()), effort=float(np.mean(act ** 2)), ok=ok)

"""Who does the work: the muscle activations that produce a motion, found by static optimisation.

For each moment, inverse dynamics gives the generalised forces the motion needs (MuJoCo's mj_inverse, with every constraint
switched off so nothing is hidden in them). A quadratic programme (OSQP) then finds, all at once:

- the activation of each of the 416 muscles, between what the practitioner holds (tension) and 1,
- the ground's force at each loaded point of the soles, inside a friction pyramid,
- the force between each palm and its thigh when the hands rest on the thighs (an internal contact: equal and opposite on
  the hand and the femur at the same point, pushing only, inside a friction pyramid around the thigh's surface normal), so
  the arms can carry part of the trunk when it leans on them,
- reserve forces for anything the muscles cannot do (heavily penalised: their size says how strained the posture is),

The joint couplings (the vertebrae following the lumbar spine, the shoulder rhythm, the patellae) are folded into the
equations rather than given force variables: everything is expressed in the independent coordinates, so a force on a coupled
joint reaches the joint that drives it through the coupling's slope (q_dot = T q_dot_independent; the equations are T^T(...)).
Contact forces are solved in hectonewtons, so all the variables have similar sizes and the solver converges quickly.

minimising the sum of squared activations (the usual quadratic form of Crowninshield & Brand's 1981 physiological
criterion; also qigong's song: no more effort than the posture needs) plus a little change from the last moment
(activation is not instantaneous). Crowninshield & Brand cubed the muscle stress, for endurance: a muscle near its maximum
costs far more than its share, so the load spreads before any one muscle is driven hard. The squares alone let one strong
muscle run near its maximum to spare a dozen small ones (the latissimus at 90% when the hands bear the trunk's lean on the
thighs), so activation past EASE costs again, steeply (a soft ceiling: a slack variable per muscle, still a QP).
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
MU_HAND = 0.6  # a palm on the thigh: skin on cloth


@dataclass
class Solution:
    act: np.ndarray  # activations (nu)
    force: np.ndarray  # muscle forces, N (positive = pulling)
    grf: dict  # foot -> {"force": world N (3,), "cop": world point (3,)} (absent when the foot bears nothing)
    hands: dict  # side -> {"force": world N (3,), the thigh's push on the palm, "point": world (3,)} (absent: not resting)
    root_residual: float  # N or N m: what had to hold the body up besides muscles and ground (should be ~0)
    joint_residual: float  # largest reserve torque on a joint, N m
    effort: float  # mean squared activation
    ok: bool


class MuscleSolver:
    W_ACT = 1.0
    W_SMOOTH = 0.5
    W_FORCE = 1e-3  # per hN squared: just enough to spread the load over the sole (the palms' pressing costs muscle anyway)
    W_ROOT = 1e2
    W_RESERVE = 1.0
    F_UNIT = 100.0  # contact forces are solved in hectonewtons
    EASE = 0.5  # activation past this costs W_HARD more per unit squared (the endurance criterion's character)
    W_HARD = 10.0

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

    def _qp(self, p_diag, q, A, lo, hi) -> np.ndarray | None:
        solver = osqp.OSQP()
        solver.setup(sparse.diags(p_diag).tocsc(), q, A, lo, hi, verbose=False, polishing=True, eps_abs=1e-5,
                     eps_rel=1e-5, max_iter=20000, warm_starting=True, adaptive_rho_interval=25)
        res = solver.solve(raise_error=False)
        self.status = res.info.status
        ok = res.info.status in ("solved", "solved inaccurate") and res.x is not None and np.all(np.isfinite(res.x))
        return res.x if ok else None

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
              loaded: dict | None = None, external: np.ndarray | None = None, hands: dict | None = None) -> Solution:
        """floor: per-muscle lower bound on activation (tension held). loaded: foot -> True (on the ground), "forefoot"
        (on tiptoe) or False (lifted). external: generalised forces (nv) acting besides muscles and ground, which the
        muscles then need not provide (the abdominal pressure's extension moment on the lumbar spine). hands: side -> True
        when that palm rests on its thigh and can press on it."""
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
        def bears(p):  # loaded: foot -> True (the whole sole), "forefoot" (on tiptoe: not the heel) or False (lifted)
            on = (loaded or {}).get(p.foot, True)
            return on is True or (on == "forefoot" and p.where != "heel")
        pts = [p for p in a.sole if bears(p)]
        jacs, worlds = [], []
        jp = np.zeros((3, nv))
        for p in pts:
            w = d.xpos[p.body] + d.xmat[p.body].reshape(3, 3) @ p.local
            mujoco.mj_jac(m, d, jp, None, w, p.body)
            jacs.append(jp.copy())
            worlds.append(w)
        nc = len(pts)

        # hands on the thighs: the thigh pushes the palm (J_hand^T f) and the palm the thigh (-J_femur^T f), at the palm;
        # the normal points out of the thigh, from the femur's axis through the palm; one tangent runs along the thigh
        rests, hjacs, hpoints, frames = [], [], [], []
        jf = np.zeros((3, nv))
        for side in ("left", "right"):
            if not (hands or {}).get(side):
                continue
            p = a.palm[side]
            w = d.xpos[p.hand] + d.xmat[p.hand].reshape(3, 3) @ p.local
            mujoco.mj_jac(m, d, jp, None, w, p.hand)
            mujoco.mj_jac(m, d, jf, None, w, p.femur)
            R = d.xmat[p.femur].reshape(3, 3)
            radial = R.T @ (w - d.xpos[p.femur]) * np.array([1.0, 0.0, 1.0])
            rests.append(side)
            hjacs.append(jp - jf)
            hpoints.append(w)
            nrm = R @ radial / max(np.linalg.norm(radial), 1e-9)
            frames.append((nrm, R[:, 1], np.cross(nrm, R[:, 1])))
        nh = len(rests)

        # everything in the independent coordinates: T^T (A_mus a + Jc^T f + ...) + r = T^T (tau - b_mus)
        T = self._reduction(qpos)
        nr = T.shape[1]
        # variables: [a (nu), f in hN (3 nc soles, then 3 nh palms), r (nr)]
        nf = 3 * (nc + nh)
        n = nu + nf + nr
        Jc = np.hstack([J.T for J in jacs + hjacs]) * self.F_UNIT if nc + nh else np.zeros((nv, 0))
        A_eq = sparse.hstack([sparse.csc_matrix(T.T @ A_mus), sparse.csc_matrix(T.T @ Jc), sparse.identity(nr)])
        rhs = T.T @ (tau - b_mus - (external if external is not None else 0.0))

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
        if nh:  # the palms: the same pyramid, around each thigh's normal, with an edge along the thigh
            fr = sparse.lil_matrix((5 * nh, n))
            for k, (nrm, t1, t2) in enumerate(frames):
                c = nu + 3 * (nc + k)
                for row, v in enumerate((nrm, t1 - MU_HAND * nrm, -t1 - MU_HAND * nrm, t2 - MU_HAND * nrm,
                                         -t2 - MU_HAND * nrm)):
                    fr[5 * k + row, c:c + 3] = v
            rows.append(fr.tocsc())
            lo.append(np.tile([0, -np.inf, -np.inf, -np.inf, -np.inf], nh))
            hi.append(np.tile([np.inf, 0, 0, 0, 0], nh))
        # activation bounds
        rows.append(sparse.hstack([sparse.identity(nu), sparse.csc_matrix((nu, n - nu))]))
        lo.append(np.zeros(nu) if floor is None else np.clip(floor, 0, 1))
        hi.append(np.ones(nu))

        w_r = np.full(nr, self.W_RESERVE)
        w_r[:6] = self.W_ROOT  # the root's six coordinates are the first independent ones
        p_diag = np.concatenate([np.full(nu, 2 * (self.W_ACT + self.W_SMOOTH)), np.full(nf, 2 * self.W_FORCE), 2 * w_r])
        q = np.concatenate([-2 * self.W_SMOOTH * self.prev, np.zeros(nf + nr)])
        A, lo, hi = sparse.vstack(rows).tocsc(), np.concatenate(lo), np.concatenate(hi)
        x = self._qp(p_diag, q, A, lo, hi)
        if x is not None and self.W_HARD > 0 and x[:nu].max() > self.EASE + 1e-3:
            # some muscle past EASE: solve again with the soft ceiling (slack s per muscle: a - s <= EASE, s >= 0, costing
            # W_HARD s^2). When none is past it the first answer is already the optimum with the ceiling, so ordinary
            # postures pay nothing for it.
            eye, zeros = sparse.identity(nu), lambda r, c: sparse.csc_matrix((r, c))
            A1 = sparse.vstack([sparse.hstack([A, zeros(A.shape[0], nu)]),
                                sparse.hstack([eye, zeros(nu, n - nu), -eye]),
                                sparse.hstack([zeros(nu, n), eye])]).tocsc()
            x1 = self._qp(np.concatenate([p_diag, np.full(nu, 2 * self.W_HARD)]), np.concatenate([q, np.zeros(nu)]), A1,
                          np.concatenate([lo, np.full(nu, -np.inf), np.zeros(nu)]),
                          np.concatenate([hi, np.full(nu, self.EASE), np.full(nu, np.inf)]))
            x = x1[:n] if x1 is not None else x
        ok = x is not None
        x = x if ok else np.zeros(n)
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
        palms = {side: {"force": f[nc + k], "point": hpoints[k]} for k, side in enumerate(rests)}
        force = -(gain * act + bias)
        return Solution(act=act, force=force, grf=grf, hands=palms, root_residual=float(np.linalg.norm(r[:6])),
                        joint_residual=float(np.abs(r[6:]).max()), effort=float(np.mean(act ** 2)), ok=ok)

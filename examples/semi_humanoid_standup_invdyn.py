import argparse
import os
import sys
import time

import numpy as np
import pinocchio

import crocoddyl


URDF_PATH = "/home/xhz/crocoddyl/asset/Semi_Taks_T1/Semi_Taks_T1.urdf"
MESH_DIR = "/home/xhz/crocoddyl/asset/Semi_Taks_T1"

LH_FRAME = "left_wrist_pitch_link"
RH_FRAME = "right_wrist_pitch_link"
LSH_FRAME = "left_shoulder_pitch_link"
RSH_FRAME = "right_shoulder_pitch_link"
TORSO_FRAME = "torso_link"
BASE_FRAME = "base_link"
WAIST_JNAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

DIRECTIONS = {
    "front":       (+1.20, +0.00),
    "front_right": (+0.85, +0.45),
    "front_left":  (+0.85, -0.45),
    "back":        (-1.00, +0.00),
    "back_right":  (-0.70, +0.45),
    "back_left":   (-0.70, -0.45),
}

DT = 2e-2
T_RAMP_UP = 20
T_RISE = 90
T_RAMP_DOWN = 80
T_RETURN = 60

Q_PARK_FRAC = 0.78

HAND_F_PEAK = 40.0
BASE_F_OFFLOAD = 80.0
FRICTION_MU = 0.5
TABLE_Z = 0.0

W_HAND_REACH = 1e3
W_TORSO = 5e2
W_WAIST_TAU = 4e1
W_WAIST_TAU_RELEASE = 1e-1
W_TOTAL_TAU = 1e-3
W_ACCEL = 2.5e-2
W_ACCEL_RELEASE = 3.0
W_FORCE = 2e1
W_FRICTION = 1e1
W_BASE_FZ_OFFLOAD = 3e-2
W_BASE_F_LIGHT = 1e-4
W_POSE = 8e2
W_REG = 1e-3
W_LIM = 1e2
W_TERMINAL = 80.0
W_TERMINAL_V = 50.0


def parse_args():
    p = argparse.ArgumentParser(description="Semi-humanoid stand-up OCP")
    p.add_argument("--dir", choices=list(DIRECTIONS), default="back",
                   help="initial fall direction preset: "
                        "front, front_right, front_left, back, back_right, back_left")
    p.add_argument("--pitch", type=float, default=None,
                   help="override initial waist_pitch (rad)")
    p.add_argument("--roll", type=float, default=None,
                   help="override initial waist_roll (rad)")
    p.add_argument("--headless", action="store_true",
                   help="skip meshcat visualization")
    p.add_argument("--once", action="store_true",
                   help="play visualization once instead of looping")
    p.add_argument("--plot", action="store_true",
                   help="show solver convergence plot")
    p.add_argument("--iters", type=int, default=80,
                   help="max solver iterations")
    p.add_argument("--force", type=float, default=HAND_F_PEAK,
                   help="peak hand pressing force (N)")
    return p.parse_args()


def body_frame(model, name):
    return model.getFrameId(name, pinocchio.FrameType.BODY)


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * a)


class Builder:
    def __init__(self, model, pitch, roll, hand_force):
        self.rmodel = model
        self.rdata = model.createData()
        self.state = crocoddyl.StateMultibody(model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nv = self.state.nv
        self.nq = model.nq
        self.nu_act = self.actuation.nu

        self.id_base = body_frame(model, BASE_FRAME)
        self.id_LH = model.getFrameId(LH_FRAME)
        self.id_RH = model.getFrameId(RH_FRAME)
        self.id_LSh = model.getFrameId(LSH_FRAME)
        self.id_RSh = model.getFrameId(RSH_FRAME)
        self.id_torso = model.getFrameId(TORSO_FRAME)

        self.waist_v = [model.joints[model.getJointId(j)].idx_v for j in WAIST_JNAMES]
        self.waist_q = [model.joints[model.getJointId(j)].idx_q for j in WAIST_JNAMES]
        actuated = list(model.names)[2:]
        self.arm_v = [
            model.joints[model.getJointId(n)].idx_v
            for n in actuated if n not in WAIST_JNAMES
        ]
        self.arm_q = [
            model.joints[model.getJointId(n)].idx_q
            for n in actuated if n not in WAIST_JNAMES
        ]

        self.pitch = pitch
        self.roll = roll
        self.hand_force = hand_force
        q_init = self._q_start()
        self.qN = pinocchio.neutral(model)

        lsh0 = self._fk(q_init, self.id_LSh).translation
        rsh0 = self._fk(q_init, self.id_RSh).translation
        tgt_LH_pos = np.array([lsh0[0], lsh0[1], TABLE_Z])
        tgt_RH_pos = np.array([rsh0[0], rsh0[1], TABLE_Z])

        self.q0 = self._ik_hands_to_table(q_init, tgt_LH_pos, tgt_RH_pos)
        self.x0 = np.concatenate([self.q0, np.zeros(self.nv)])
        self.xN = np.concatenate([self.qN, np.zeros(self.nv)])

        self.M_base = self._fk(self.q0, self.id_base)
        self.M_torso_0 = self._fk(self.q0, self.id_torso)
        self.M_torso_N = self._fk(self.qN, self.id_torso)
        self.M_LH = self._fk(self.q0, self.id_LH)
        self.M_RH = self._fk(self.q0, self.id_RH)
        self.tgt_LH = pinocchio.SE3(self.M_LH.rotation, self.M_LH.translation.copy())
        self.tgt_RH = pinocchio.SE3(self.M_RH.rotation, self.M_RH.translation.copy())

        self.waist_tau_w = np.zeros(self.nu_act)
        for j in WAIST_JNAMES:
            self.waist_tau_w[model.joints[model.getJointId(j)].idx_v - 6] = W_WAIST_TAU
        self.total_tau_w = np.full(self.nu_act, W_TOTAL_TAU)

        maxf = sys.float_info.max
        self.xlb = np.hstack([
            -maxf * np.ones(6),
            model.lowerPositionLimit[7:],
            -maxf * np.ones(self.nv),
        ])
        self.xub = np.hstack([
            +maxf * np.ones(6),
            model.upperPositionLimit[7:],
            +maxf * np.ones(self.nv),
        ])

        self.sw_q = np.concatenate([np.zeros(6), 0.01 * np.ones(self.nv - 6)])
        self.sw_v_base = np.ones(self.nv)

        print(
            f"start pitch={pitch:+.2f} roll={roll:+.2f} | "
            f"LH=({self.tgt_LH.translation[0]:+.3f},"
            f"{self.tgt_LH.translation[1]:+.3f},"
            f"{self.tgt_LH.translation[2]:+.3f}) "
            f"RH=({self.tgt_RH.translation[0]:+.3f},"
            f"{self.tgt_RH.translation[1]:+.3f},"
            f"{self.tgt_RH.translation[2]:+.3f}) "
            f"F_peak={hand_force:.1f}N"
        )

    def _fk(self, q, fid):
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q)
        pinocchio.updateFramePlacement(self.rmodel, self.rdata, fid)
        return self.rdata.oMf[fid].copy()

    def _q_start(self):
        q = pinocchio.neutral(self.rmodel)
        jp = self.rmodel.getJointId("waist_pitch_joint")
        jr = self.rmodel.getJointId("waist_roll_joint")
        q[self.rmodel.joints[jp].idx_q] = self.pitch
        q[self.rmodel.joints[jr].idx_q] = self.roll
        return q

    def _ik_hands_to_table(self, q, tgt_LH, tgt_RH):
        left_arm_v = []
        right_arm_v = []
        arm_q_idx = []
        for n in self.rmodel.names:
            if (n.startswith("left_") or n.startswith("right_")) and "joint" in n:
                jid = self.rmodel.getJointId(n)
                arm_q_idx.append(self.rmodel.joints[jid].idx_q)
                if n.startswith("left_"):
                    left_arm_v.append(self.rmodel.joints[jid].idx_v)
                else:
                    right_arm_v.append(self.rmodel.joints[jid].idx_v)

        q = q.copy()
        best_q = q.copy()
        best_err = float("inf")
        step = 0.55
        damp = 1e-3
        for _ in range(800):
            pinocchio.forwardKinematics(self.rmodel, self.rdata, q)
            pinocchio.computeJointJacobians(self.rmodel, self.rdata, q)
            pinocchio.updateFramePlacements(self.rmodel, self.rdata)
            err_L = tgt_LH - self.rdata.oMf[self.id_LH].translation
            err_R = tgt_RH - self.rdata.oMf[self.id_RH].translation
            e = float(np.linalg.norm(err_L) + np.linalg.norm(err_R))
            if e < best_err:
                best_err = e
                best_q = q.copy()
            if e < 1e-6:
                break
            J_L = pinocchio.computeFrameJacobian(
                self.rmodel, self.rdata, q, self.id_LH,
                pinocchio.LOCAL_WORLD_ALIGNED,
            )[:3]
            J_R = pinocchio.computeFrameJacobian(
                self.rmodel, self.rdata, q, self.id_RH,
                pinocchio.LOCAL_WORLD_ALIGNED,
            )[:3]
            J_L_arm = np.zeros((3, self.nv))
            J_R_arm = np.zeros((3, self.nv))
            J_L_arm[:, left_arm_v] = J_L[:, left_arm_v]
            J_R_arm[:, right_arm_v] = J_R[:, right_arm_v]
            J = np.vstack([J_L_arm, J_R_arm])
            err = np.concatenate([err_L, err_R])
            dv = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            nrm = float(np.linalg.norm(dv))
            if nrm > 0.6:
                dv = dv * (0.6 / nrm)
            q = pinocchio.integrate(self.rmodel, q, dv * step)
            for iq in arm_q_idx:
                lo = self.rmodel.lowerPositionLimit[iq]
                hi = self.rmodel.upperPositionLimit[iq]
                if np.isfinite(lo) and np.isfinite(hi):
                    q[iq] = float(np.clip(q[iq], lo, hi))
        return best_q

    def _xref(self, waist_a, arm_a):
        q = self.q0.copy()
        for iq in self.waist_q:
            q[iq] = (1.0 - waist_a) * self.q0[iq]
        for iq in self.arm_q:
            q[iq] = (1.0 - arm_a) * self.q0[iq]
        return np.concatenate([q, np.zeros(self.nv)])

    def warm_start_xs(self, T_total):
        xs = []
        qpark = Q_PARK_FRAC
        T_PHASE_C = T_RAMP_UP + T_RISE + T_RAMP_DOWN
        for k in range(T_total + 1):
            if k <= T_RAMP_UP:
                xs.append(self._xref(0.0, 0.0))
            elif k <= T_RAMP_UP + T_RISE:
                a = (k - T_RAMP_UP) / T_RISE
                xs.append(self._xref(qpark * smoothstep(a), 0.0))
            elif k <= T_PHASE_C:
                xs.append(self._xref(qpark, 0.0))
            else:
                a = (k - T_PHASE_C) / T_RETURN
                s = smoothstep(a)
                xs.append(self._xref(qpark + (1.0 - qpark) * s, s))
        return xs

    def _lock_w(self, w_q=0.0, w_v=0.0, a_q=0.0, a_v=0.0):
        w = np.zeros(2 * self.nv)
        for iv in self.waist_v:
            w[iv] = w_q
            w[self.nv + iv] = w_v
        for iv in self.arm_v:
            w[iv] = a_q
            w[self.nv + iv] = a_v
        return w

    def _node(
        self,
        contact=False,
        hand_tgt=None,
        torso_tgt=None,
        f_hand=None,
        f_base_offload=None,
        xref=None,
        lock=None,
        lock_s=1.0,
        reg_s=1.0,
        term_v=False,
        accel_w=W_ACCEL,
        waist_tau_w=W_WAIST_TAU,
    ):
        nu = self.nv + 6 + (6 if contact else 0)
        cts = crocoddyl.ContactModelMultiple(self.state, nu)
        cs = crocoddyl.CostModelSum(self.state, nu)

        cts.addContact(
            "base",
            crocoddyl.ContactModel6D(
                self.state, self.id_base, self.M_base,
                pinocchio.LOCAL_WORLD_ALIGNED, nu, np.array([0.0, 50.0]),
            ),
        )
        if f_base_offload is not None:
            base_fref = pinocchio.Force(
                np.concatenate([
                    np.array([0.0, 0.0, f_base_offload]),
                    np.zeros(3),
                ])
            )
            cs.addCost(
                "base_f",
                crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ResidualModelContactForce(
                        self.state, self.id_base, base_fref, 6, nu, False,
                    ),
                ),
                W_BASE_FZ_OFFLOAD,
            )
        else:
            cs.addCost(
                "base_f",
                crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ResidualModelContactForce(
                        self.state, self.id_base, pinocchio.Force.Zero(),
                        6, nu, False,
                    ),
                ),
                W_BASE_F_LIGHT,
            )

        if contact:
            hand_pin = {
                "LH": (self.id_LH, self.tgt_LH.translation),
                "RH": (self.id_RH, self.tgt_RH.translation),
            }
            for name, (fid, p_ref) in hand_pin.items():
                cts.addContact(
                    name,
                    crocoddyl.ContactModel3D(
                        self.state, fid, p_ref,
                        pinocchio.LOCAL_WORLD_ALIGNED, nu,
                        np.array([150.0, 25.0]),
                    ),
                )
                cone = crocoddyl.FrictionCone(np.eye(3), FRICTION_MU, 4, False)
                cs.addCost(
                    name + "_fric",
                    crocoddyl.CostModelResidual(
                        self.state,
                        crocoddyl.ActivationModelQuadraticBarrier(
                            crocoddyl.ActivationBounds(cone.lb, cone.ub)
                        ),
                        crocoddyl.ResidualModelContactFrictionCone(
                            self.state, fid, cone, nu, False,
                        ),
                    ),
                    W_FRICTION,
                )
                fv = np.zeros(3) if f_hand is None else f_hand
                cs.addCost(
                    name + "_f",
                    crocoddyl.CostModelResidual(
                        self.state,
                        crocoddyl.ResidualModelContactForce(
                            self.state, fid,
                            pinocchio.Force(np.concatenate([fv, np.zeros(3)])),
                            3, nu, False,
                        ),
                    ),
                    W_FORCE,
                )

        if hand_tgt is not None:
            for fid, M in hand_tgt.items():
                cs.addCost(
                    f"hand_track_{fid}",
                    crocoddyl.CostModelResidual(
                        self.state,
                        crocoddyl.ActivationModelWeightedQuad(
                            np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) ** 2
                        ),
                        crocoddyl.ResidualModelFramePlacement(
                            self.state, fid, M, nu,
                        ),
                    ),
                    W_HAND_REACH,
                )

        if torso_tgt is not None:
            cs.addCost(
                "torso",
                crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ActivationModelWeightedQuad(
                        np.array([0.1, 0.1, 1.0, 0.3, 1.0, 0.3]) ** 2
                    ),
                    crocoddyl.ResidualModelFramePlacement(
                        self.state, self.id_torso, torso_tgt, nu,
                    ),
                ),
                W_TORSO,
            )

        waist_w = np.zeros(self.nu_act)
        for j in WAIST_JNAMES:
            waist_w[self.rmodel.joints[self.rmodel.getJointId(j)].idx_v - 6] = waist_tau_w
        cs.addCost(
            "waist_tau",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(waist_w ** 2),
                crocoddyl.ResidualModelJointEffort(
                    self.state, self.actuation,
                    np.zeros(self.nu_act), nu, False,
                ),
            ),
            1.0,
        )
        cs.addCost(
            "total_tau",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(self.total_tau_w ** 2),
                crocoddyl.ResidualModelJointEffort(
                    self.state, self.actuation,
                    np.zeros(self.nu_act), nu, False,
                ),
            ),
            1.0,
        )
        cs.addCost(
            "accel",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ResidualModelJointAcceleration(self.state, nu),
            ),
            accel_w,
        )

        sw_v = np.full(self.nv, W_TERMINAL_V) if term_v else self.sw_v_base.copy()
        sw = np.concatenate([self.sw_q, sw_v])
        xref_use = self.xN if xref is None else xref
        cs.addCost(
            "reg",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(sw ** 2),
                crocoddyl.ResidualModelState(self.state, xref_use, nu),
            ),
            W_REG * reg_s,
        )
        cs.addCost(
            "lim",
            crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(self.xlb, self.xub, 1.0)
                ),
                crocoddyl.ResidualModelState(self.state, self.xN, nu),
            ),
            W_LIM,
        )

        if lock is not None and np.any(lock):
            cs.addCost(
                "lock",
                crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ActivationModelWeightedQuad(lock ** 2),
                    crocoddyl.ResidualModelState(self.state, xref_use, nu),
                ),
                W_POSE * lock_s,
            )

        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, cts, cs,
            ),
            DT,
        )

    def build(self):
        models = []

        lock_press = self._lock_w(w_q=4.0, w_v=1.0)
        lock_rise = self._lock_w(w_q=8.0, w_v=2.0)
        lock_unload = self._lock_w(w_q=25.0, w_v=25.0, a_q=2.0, a_v=2.0)
        lock_return = self._lock_w(w_q=12.0, w_v=6.0, a_q=5.0, a_v=3.0)
        lock_term = self._lock_w(w_q=20.0, w_v=8.0, a_q=15.0, a_v=8.0)

        qpark = Q_PARK_FRAC
        F = self.hand_force
        Bf = BASE_F_OFFLOAD
        T_PHASE_C = T_RAMP_UP + T_RISE + T_RAMP_DOWN

        for k in range(T_RAMP_UP):
            a = (k + 1) / T_RAMP_UP
            models.append(self._node(
                contact=True,
                f_hand=np.array([0.0, 0.0, -a * F]),
                f_base_offload=Bf,
                xref=self._xref(0.0, 0.0),
                lock=lock_press,
            ))

        for k in range(T_RISE):
            a = (k + 1) / T_RISE
            s = qpark * smoothstep(a)
            M_t = pinocchio.SE3.Interpolate(self.M_torso_0, self.M_torso_N, s)
            models.append(self._node(
                contact=True,
                torso_tgt=M_t,
                f_hand=np.array([0.0, 0.0, -F]),
                f_base_offload=Bf,
                xref=self._xref(s, 0.0),
                lock=lock_rise,
            ))

        M_park = pinocchio.SE3.Interpolate(self.M_torso_0, self.M_torso_N, qpark)
        for k in range(T_RAMP_DOWN):
            a = (k + 1) / T_RAMP_DOWN
            models.append(self._node(
                contact=True,
                torso_tgt=M_park,
                f_hand=np.array([0.0, 0.0, -(1.0 - a) * F]),
                f_base_offload=Bf * (1.0 - a),
                xref=self._xref(qpark, 0.0),
                lock=lock_unload,
                accel_w=W_ACCEL_RELEASE,
                waist_tau_w=W_WAIST_TAU_RELEASE,
            ))

        for k in range(T_RETURN):
            a = (k + 1) / T_RETURN
            s_w = qpark + (1.0 - qpark) * smoothstep(a)
            arm_a = smoothstep(a)
            M_t = pinocchio.SE3.Interpolate(self.M_torso_0, self.M_torso_N, s_w)
            models.append(self._node(
                contact=False,
                torso_tgt=M_t,
                xref=self._xref(s_w, arm_a),
                lock=lock_return,
                reg_s=10.0,
                term_v=(a > 0.5),
                accel_w=W_ACCEL_RELEASE,
                waist_tau_w=W_WAIST_TAU_RELEASE,
            ))

        terminal = self._node(
            contact=False,
            torso_tgt=self.M_torso_N,
            xref=self.xN,
            lock=lock_term,
            reg_s=W_TERMINAL,
            term_v=True,
        )
        return crocoddyl.ShootingProblem(self.x0, models, terminal)


def evaluate(b, solver):
    rmodel = b.rmodel
    nv = b.nv
    nq = b.nq
    waist_v = b.waist_v

    xs = list(solver.xs)
    us = list(solver.us)
    qdots = np.stack([x[nq:] for x in xs[1:]])
    pk_waist_qdot = float(np.max(np.abs(qdots[:, waist_v])))

    problem = solver.problem
    problem.calc(xs, us)
    taus, fz_hand, fz_base, accels = [], [], [], []
    for k in range(problem.T):
        diff = getattr(problem.runningDatas[k], "differential", None)
        if diff is None:
            continue
        taus.append(np.array(diff.multibody.actuation.tau, copy=True))
        try:
            cs = diff.multibody.contacts.contacts.todict()
            if "base" in cs:
                fz_base.append(float(cs["base"].f.linear[2]))
            for n in ("LH", "RH"):
                if n in cs:
                    fz_hand.append(float(cs[n].f.linear[2]))
        except Exception:
            pass
        try:
            accels.append(np.array(diff.xout[:nv], copy=True))
        except Exception:
            pass

    if taus:
        T = np.stack(taus)
        wt = T[:, waist_v]
        rms_per = [float(np.sqrt(np.mean(wt[:, i] ** 2))) for i in range(3)]
        pk_per = [float(np.max(np.abs(wt[:, i]))) for i in range(3)]
        rms_arm = float(np.sqrt(np.mean(T[:, b.arm_v] ** 2)))
    else:
        rms_per = pk_per = [float("nan")] * 3
        rms_arm = float("nan")

    qf = xs[-1][:nq]
    vf = xs[-1][nq:]
    Mf = b._fk(qf, b.id_torso)

    return {
        "cost": float(solver.cost),
        "pk_waist_qdot": pk_waist_qdot,
        "rms_per": rms_per,
        "pk_per": pk_per,
        "rms_arm": rms_arm,
        "pk_fz_hand": float(np.max(np.abs(fz_hand))) if fz_hand else 0.0,
        "pk_fz_base": float(np.max(np.abs(fz_base))) if fz_base else 0.0,
        "rms_a": float(np.sqrt(np.mean(np.stack(accels) ** 2))) if accels else float("nan"),
        "z_err": float(Mf.translation[2] - b.M_torso_N.translation[2]),
        "pitch_err": float(pinocchio.log3(b.M_torso_N.rotation.T @ Mf.rotation)[1]),
        "final_v": float(np.max(np.abs(vf))),
        "final_arm_q_dev": float(np.max(np.abs([qf[i] for i in b.arm_q]))),
        "final_waist_q_dev": float(np.max(np.abs([qf[i] for i in b.waist_q]))),
    }


def print_metrics(m):
    ry, rr, rp = m["rms_per"]
    py, pr, pp = m["pk_per"]
    print(
        f"[final] cost={m['cost']:.1f} |q̇|w={m['pk_waist_qdot']:.2f} "
        f"τ(rms/pk) y={ry:.2f}/{py:.1f} r={rr:.2f}/{pr:.1f} p={rp:.2f}/{pp:.1f} "
        f"arm_τ={m['rms_arm']:.2f} "
        f"fz: hand={m['pk_fz_hand']:.1f}N base={m['pk_fz_base']:.1f}N "
        f"|a|={m['rms_a']:.2f} | final: w|q|={m['final_waist_q_dev']*1e3:.1f}mrad "
        f"a|q|={m['final_arm_q_dev']*1e3:.1f}mrad |v|={m['final_v']:.3f}"
    )


def display(model, b, solver, args):
    try:
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError as e:
        print(f"display skipped: {e}")
        return
    try:
        coll = pinocchio.buildGeomFromUrdf(
            model, URDF_PATH, pinocchio.GeometryType.COLLISION,
            package_dirs=MESH_DIR,
        )
        vis = pinocchio.buildGeomFromUrdf(
            model, URDF_PATH, pinocchio.GeometryType.VISUAL,
            package_dirs=MESH_DIR,
        )
        viz = MeshcatVisualizer(model, coll, vis)
        viz.initViewer(open=True)
        viz.loadViewerModel(rootNodeName="semi_taks")
    except Exception as e:
        print(f"display skipped ({type(e).__name__}: {e})")
        return

    try:
        import meshcat.geometry as mg

        th = 0.05
        viz.viewer["table"].set_object(
            mg.Box([3.0, 3.0, th]),
            mg.MeshLambertMaterial(color=0x9F7A4F, reflectivity=0.3),
        )
        Tp = np.eye(4)
        Tp[:3, 3] = [0.0, 0.0, TABLE_Z - 0.5 * th]
        viz.viewer["table"].set_transform(Tp)
        for name, M in (("LH_target", b.tgt_LH), ("RH_target", b.tgt_RH)):
            viz.viewer[name].set_object(
                mg.Sphere(0.02),
                mg.MeshLambertMaterial(color=0xFF3030),
            )
            Tm = np.eye(4)
            Tm[:3, 3] = M.translation
            viz.viewer[name].set_transform(Tm)
    except Exception as e:
        print(f"(table sketch skipped: {type(e).__name__}: {e})")

    xs = list(solver.xs)
    print(f"meshcat playback: {len(xs)} frames, dt={DT}s. Ctrl+C to stop.")
    try:
        while True:
            for x in xs:
                viz.display(x[: model.nq])
                time.sleep(DT)
            if args.once:
                break
            time.sleep(0.8)
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    args = parse_args()
    p, r = DIRECTIONS[args.dir]
    if args.pitch is not None:
        p = args.pitch
    if args.roll is not None:
        r = args.roll

    model = pinocchio.buildModelFromUrdf(URDF_PATH)
    print(f"nq={model.nq} nv={model.nv}")

    b = Builder(model, p, r, args.force)
    problem = b.build()
    print(
        f"T={problem.T} (ramp_up={T_RAMP_UP}, rise={T_RISE}, "
        f"ramp_down={T_RAMP_DOWN}, return={T_RETURN})"
    )

    solver = crocoddyl.SolverIntro(problem)
    solver.th_minImprove = 1e-2
    cbs = [crocoddyl.CallbackVerbose()]
    if args.plot:
        cbs.append(crocoddyl.CallbackLogger())
    solver.setCallbacks(cbs)

    xs = b.warm_start_xs(problem.T)
    us = problem.quasiStatic(xs[:problem.T])
    print(f"SOLVE iters={args.iters}")
    try:
        ok = solver.solve(xs, us, args.iters, False)
    except KeyboardInterrupt:
        ok = False
        print("\ninterrupted; using best-so-far.")
    print(f"converged={ok} iters={solver.iter} cost={solver.cost:.2f}")

    try:
        print_metrics(evaluate(b, solver))
    except Exception as e:
        print(f"metrics skipped: {type(e).__name__}: {e}")

    if not args.headless:
        display(model, b, solver, args)

    if args.plot:
        log = solver.getCallbacks()[-1]
        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps,
        )

    return solver, b


if __name__ == "__main__":
    main()

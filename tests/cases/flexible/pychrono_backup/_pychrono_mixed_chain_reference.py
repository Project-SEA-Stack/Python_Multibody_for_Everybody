# -*- coding: utf-8 -*-
r"""Independent, third-party-software cross-validation of FSM's mixed_chain
(`Examples_flexible_FSM/flex_mixed_chain.py`, EXAMPLE T in
FSM_table_generator.py) using REAL PyChrono.

This script is NOT part of the normal pytest suite and cannot be imported
or collected by it: PyChrono is only installed in a separate conda
environment named `chrono` (see below), not in the `multibody_env`
environment the rest of this repository's tests run under. Instead, this
script is a one-off, manually-run verification tool.

Modeling approach
------------------
Unlike the double-pendulum reference scripts in this same folder (which
reconstruct each flexible link as a continuous `ChElementBeamEuler` FEA
beam), this system is reconstructed as PLAIN RIGID BODIES + explicit
joints + spring forces -- mirroring FSM's own "rigid segments + discrete
springs" discretization exactly, since mixed_chain's topology (one axial
member modeled as 5 prismatic-jointed bodies + tension springs, one
bending member modeled as 5 revolute-jointed bodies + torsion springs) is
already a discrete idealization in FSM itself, not an approximation of a
continuous beam. This sidesteps the EI_BOOST/geometric-coupling issue the
double-pendulum scripts need entirely.

Topology (exact arrays copied from Examples_flexible_FSM/flex_mixed_chain.py,
0-indexed bodies 0..10 = FSM bodies 1..11):
    ground --R(hub)--> body0 --P+spring--> body1 --P+spring--> ... --> body5
      --R+spring(bending)--> body6 --R+spring--> body7 --R+spring--> body8
      --R+spring--> body9 --R(free, no spring)--> body10(payload)

Axial joints use `ChLinkLockPrismatic` (slides along the joint frame's
LOCAL Z axis -- confirmed empirically, NOT X) + `ChLinkTSDA` (translational
spring-damper, connecting each pair of bodies' own CG points -- directly
implements FSM's CG-to-CG tension springs). Bending joints use
`ChLinkLockRevolute` + a MANUALLY computed-and-applied torsion-spring
torque each step (no `ChLinkRSDA` rotational spring-damper exists in this
PyChrono version). Initial body positions/angles below are copied directly
from FSM's own solved initial configuration (`flex_mixed_chain.py`'s
Table 1/2, zero gravity) -- this is a pure dynamics-propagation comparison
starting from an identical IC, not an independent static solve.

Validated result (dt=2e-6, 5.0s span -- see "stiffness-ratio notes" below
for why such a small dt is required): hub_theta/payload trajectories track
FSM closely across all ~28 oscillation cycles (final max pointwise error:
hub 0.00797 rad, payload_z 0.01143 m); total energy is NOT exactly
conserved (rises to ~3.27x its initial value by t=5.0s, with the growth
RATE decaying over time rather than compounding) -- confirmed via per-body
error analysis that this spurious energy hides almost entirely in the
UNTRACKED intermediate bending bodies (7-10% relative error there) rather
than the tracked hub/payload endpoints (~0.5% relative error), consistent
with being a genuine numerical artifact of the extreme stiffness ratio
(>30,000x across the axial springs, >400,000x across body masses) rather
than a modeling bug.

Stiffness-ratio / dt notes (see repo memory for the full A/B investigation)
-----------------------------------------------------------------------------
dt=2e-4 diverges outright by t~0.05s. dt=2e-5 (10x finer) is stable for a
few hundred ms but then accumulates GROWING high-frequency noise (adding
2% critical damping on the axial TSDA springs bounds, but does not
eliminate, this growth). Of four candidate fixes tested head-to-head on a
short window (HHT numerical damping with adaptive step, HHT with FIXED
step, single-component mass-scaling, and reducing dt by 10x), only
reducing dt to 2e-6 (plain fixed-step Trapezoidal, no damping needed)
actually worked -- confirming this is a genuine time-resolution/aliasing
problem, not a damping or single-stiff-component issue. A 2% critical TSDA
damping term is still included below as a secondary safeguard (harmless at
this resolution) but is no longer load-bearing for stability at dt=2e-6.

How to run this script
-----------------------
PyChrono lives in a separate conda env (`chrono`), and importing it
requires prepending its DLL directories to PATH first:

    $env:PATH = "C:\Users\<user>\Anaconda3\envs\chrono;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Library\bin;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Scripts;" + $env:PATH
    C:\Users\<user>\Anaconda3\envs\chrono\python.exe _pychrono_mixed_chain_reference.py

NOTE: a full 5.0s run at dt=2e-6 takes on the order of 2 hours of
wall-clock time (2.5M steps, each with manual per-step spring-force
application in Python) -- the script checkpoints progress to the output
.npz every ~5% of steps (with a `complete` boolean flag inside) so partial
results can be inspected if interrupted. Reduce TSPAN below for a quicker
smoke-test run.
"""
import math
import os
import time

import numpy as np
import pychrono as chrono

NBODIES = 11
types = ['R', 'P', 'P', 'P', 'P', 'P', 'R', 'R', 'R', 'R', 'R']
parent_cg_to_joint = [[0.1, 0.0], [0.05, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0],
                       [0.0, 0.0], [0.075, 0.0], [0.075, 0.0], [0.075, 0.0], [0.075, 0.0]]
m0 = [0.05, 0.4, 0.4, 0.4, 0.4, 1e-06, 0.15, 0.15, 0.15, 0.15, 0.2]
J0 = [0.001, 0.0013333333333333337, 0.0013333333333333337, 0.0013333333333333337,
      0.0013333333333333337, 1e-06, 0.00028125, 0.00028125, 0.00028125, 0.00028125, 0.001]

# TensionSpring (rest_length, k) for the 5 prismatic joints (joints[1..5], 0-indexed)
tension_l0 = [0.15000000000000002, 0.2, 0.2, 0.2, 0.1]
tension_k = [10000000.0, 5000000.0, 5000000.0, 5000000.0, 10000000.0]

# TorsionSpring (theta0, k) for the 4 bending joints bodies[5]-bodies[6] ..
# bodies[8]-bodies[9]. Joint bodies[9]-bodies[10] (body 10->11, the payload
# hinge) has NO spring (free hinge).
torsion_theta0 = [0.05, 0.0, 0.0, 0.0]
torsion_k = [666.6666666666666, 333.3333333333333, 333.3333333333333, 333.3333333333333]

# Initial configuration (com_positions[0], angle_positions[0]) copied directly
# from FSM's own solved IC for this model (Examples_flexible_FSM/flex_mixed_chain.py),
# so Chrono starts from an IDENTICAL state to FSM rather than re-solving its own.
POS0 = np.array([
    [0.15, -0.0], [0.3, -0.0], [0.5, -0.0], [0.7, -0.0], [0.9, -0.0], [1.0, -0.0],
    [1.07490627, -0.00374844], [1.22443785, -0.01498438], [1.37322099, -0.03367975],
    [1.52088382, -0.05978781], [1.59438881, -0.07468801],
])
THETA0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.0])


def fsm_rotate(theta, vec):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([vec[0] * c + vec[1] * s, -vec[0] * s + vec[1] * c])


def chrono_quat(theta_fsm):
    return chrono.Q_from_AngAxis(-theta_fsm, chrono.ChVectorD(0, 0, 1))


def fsm_angle_of(body):
    v = body.TransformDirectionLocalToParent(chrono.ChVectorD(1, 0, 0))
    return -math.atan2(v.y, v.x)


def build_system():
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, 0, 0))  # gVec=0 in FSM model -- no gravity
    system.SetSolver(chrono.ChSolverSparseQR())
    system.SetTimestepperType(chrono.ChTimestepper.Type_TRAPEZOIDAL)

    ground = chrono.ChBody()
    ground.SetBodyFixed(True)
    system.Add(ground)

    bodies = []
    for i in range(NBODIES):
        b = chrono.ChBody()
        b.SetPos(chrono.ChVectorD(POS0[i, 0], POS0[i, 1], 0))
        b.SetRot(chrono_quat(THETA0[i]))
        b.SetPos_dt(chrono.ChVectorD(0, 0, 0))
        b.SetWvel_par(chrono.ChVectorD(0, 0, 0))
        b.SetMass(m0[i])
        b.SetInertiaXX(chrono.ChVectorD(J0[i], J0[i], J0[i]))
        system.Add(b)
        bodies.append(b)

    prev_body = ground
    prev_theta = 0.0
    prev_pos = np.array([0.0, 0.0])
    tension_idx = 0
    tsda_list = []
    for j in range(NBODIES):
        child = bodies[j]
        joint_pos_fsm = prev_pos + fsm_rotate(prev_theta, parent_cg_to_joint[j])
        joint_pos_chrono = chrono.ChVectorD(joint_pos_fsm[0], joint_pos_fsm[1], 0)

        if types[j] == 'R':
            link = chrono.ChLinkLockRevolute()
            link.Initialize(prev_body, child, chrono.ChCoordsysD(joint_pos_chrono, chrono.ChQuaternionD(1, 0, 0, 0)))
            system.Add(link)
        else:  # 'P'
            slide_dir_fsm = fsm_rotate(prev_theta, [1.0, 0.0])
            slide_dir_chrono = chrono.ChVectorD(slide_dir_fsm[0], slide_dir_fsm[1], 0)
            quat = chrono.Q_from_Vect_to_Vect(chrono.ChVectorD(0, 0, 1), slide_dir_chrono)
            link = chrono.ChLinkLockPrismatic()
            link.Initialize(prev_body, child, chrono.ChCoordsysD(joint_pos_chrono, quat))
            system.Add(link)

            tsda = chrono.ChLinkTSDA()
            tsda.Initialize(prev_body, child, False, prev_body.GetPos(), child.GetPos(), False,
                             tension_l0[tension_idx])
            tsda.SetSpringCoefficient(tension_k[tension_idx])
            # Small (2% of critical) damping -- found via a trial sweep to help
            # suppress a growing numerical instability from the extreme
            # stiffness ratio at body idx 5 (m=1e-6 next to k=1e7). No longer
            # load-bearing for stability once dt=2e-6 is used (see module
            # docstring) but kept as a harmless secondary safeguard.
            k_spring = tension_k[tension_idx]
            m1 = prev_body.GetMass() if prev_body is not ground else 1e30
            m2 = child.GetMass()
            mu = (m1 * m2) / (m1 + m2)
            c_crit = 2.0 * (k_spring * mu) ** 0.5
            tsda.SetDampingCoefficient(0.02 * c_crit)
            system.Add(tsda)
            tsda_list.append((tsda, tension_k[tension_idx]))
            tension_idx += 1

        prev_body, prev_theta, prev_pos = child, THETA0[j], POS0[j]

    torsion_specs = [
        (bodies[5], bodies[6], torsion_theta0[0], torsion_k[0]),
        (bodies[6], bodies[7], torsion_theta0[1], torsion_k[1]),
        (bodies[7], bodies[8], torsion_theta0[2], torsion_k[2]),
        (bodies[8], bodies[9], torsion_theta0[3], torsion_k[3]),
    ]

    return system, bodies, torsion_specs, tsda_list


def apply_torsion_springs(torsion_specs):
    for parent, child, theta0_spring, k in torsion_specs:
        theta_rel = fsm_angle_of(child) - fsm_angle_of(parent)
        torque_fsm = -k * (theta_rel - theta0_spring)
        # Chrono's angle = -FSM's angle, so a torque that increases theta_fsm
        # corresponds to a NEGATIVE Chrono Z-torque on the child (and the
        # equal/opposite reaction on the parent).
        child.Accumulate_torque(chrono.ChVectorD(0, 0, -torque_fsm), False)
        parent.Accumulate_torque(chrono.ChVectorD(0, 0, torque_fsm), False)


def compute_energy(bodies, torsion_specs, tsda_list):
    ke = 0.0
    for b in bodies:
        v = b.GetPos_dt()
        w = b.GetWvel_par().z
        ke += 0.5 * b.GetMass() * (v.x ** 2 + v.y ** 2) + 0.5 * b.GetInertiaXX().z * w ** 2
    pe_axial = 0.0
    for tsda, k in tsda_list:
        stretch = tsda.GetLength() - tsda.GetRestLength()
        pe_axial += 0.5 * k * stretch ** 2
    pe_bend = 0.0
    for parent, child, theta0_spring, k in torsion_specs:
        theta_rel = fsm_angle_of(child) - fsm_angle_of(parent)
        pe_bend += 0.5 * k * (theta_rel - theta0_spring) ** 2
    return ke, pe_axial, pe_bend


if __name__ == "__main__":
    TSPAN, DT = 5.0, 2e-6

    system, bodies, torsion_specs, tsda_list = build_system()

    print("Static (t=0) check -- Chrono vs FSM position/angle per body:", flush=True)
    max_pos_err = 0.0
    max_ang_err = 0.0
    for i, b in enumerate(bodies):
        p = b.GetPos()
        pos_err = math.hypot(p.x - POS0[i, 0], p.y - POS0[i, 1])
        ang_err = abs(fsm_angle_of(b) - THETA0[i])
        max_pos_err = max(max_pos_err, pos_err)
        max_ang_err = max(max_ang_err, ang_err)
    print(f"  max |pos error| over all bodies: {max_pos_err:.3e} m", flush=True)
    print(f"  max |angle error| over all bodies: {max_ang_err:.3e} rad", flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "truth",
                             "_pychrono_mixed_chain_traj.npz")
    n_steps = int(round(TSPAN / DT))
    report_every = max(1, n_steps // 1000)
    checkpoint_every = max(1, n_steps // 20)  # ~20 checkpoints over the full run

    t_list, hub_theta, payload_x, payload_z = [], [], [], []
    ke_list, pe_axial_list, pe_bend_list = [], [], []
    t_start = time.time()
    for k in range(n_steps + 1):
        if k % report_every == 0 or k == n_steps:
            t_list.append(k * DT)
            hub_theta.append(fsm_angle_of(bodies[0]))
            payload_x.append(bodies[10].GetPos().x)
            payload_z.append(bodies[10].GetPos().y)
            ke, pe_ax, pe_bd = compute_energy(bodies, torsion_specs, tsda_list)
            ke_list.append(ke)
            pe_axial_list.append(pe_ax)
            pe_bend_list.append(pe_bd)
            elapsed = time.time() - t_start
            print(f"t={k*DT:.4f}  total_E={ke+pe_ax+pe_bd:.4f}  hub={hub_theta[-1]:.5f}  wall={elapsed:.1f}s",
                  flush=True)
        if k % checkpoint_every == 0 and k > 0:
            np.savez(
                out_path,
                t=np.array(t_list), hub_theta=np.array(hub_theta),
                payload_x=np.array(payload_x), payload_z=np.array(payload_z),
                ke=np.array(ke_list), pe_axial=np.array(pe_axial_list), pe_bend=np.array(pe_bend_list),
                complete=np.array(False),
            )
            print(f"CHECKPOINT at k={k}/{n_steps}", flush=True)
        if k < n_steps:
            for b in bodies:
                b.Empty_forces_accumulators()
            apply_torsion_springs(torsion_specs)
            system.DoStepDynamics(DT)

    t_arr = np.array(t_list)
    ke_arr = np.array(ke_list)
    pe_axial_arr = np.array(pe_axial_list)
    pe_bend_arr = np.array(pe_bend_list)
    total = ke_arr + pe_axial_arr + pe_bend_arr

    print("Chrono hub angle range (rad):", np.array(hub_theta).min(), np.array(hub_theta).max(), flush=True)
    print("Chrono payload x range:", min(payload_x), max(payload_x), flush=True)
    print("Chrono payload z range:", min(payload_z), max(payload_z), flush=True)
    print(f"Chrono(dt={DT:g}, {TSPAN:g}s) total energy range:", total.min(), total.max(), flush=True)
    print(f"Chrono(dt={DT:g}, {TSPAN:g}s) growth factor:", total.max() / total[0], flush=True)
    print("Total wall time (s):", time.time() - t_start, flush=True)

    np.savez(
        out_path,
        t=t_arr, hub_theta=np.array(hub_theta), payload_x=np.array(payload_x), payload_z=np.array(payload_z),
        ke=ke_arr, pe_axial=pe_axial_arr, pe_bend=pe_bend_arr, complete=np.array(True),
    )
    print("SAVED (complete)", out_path, flush=True)

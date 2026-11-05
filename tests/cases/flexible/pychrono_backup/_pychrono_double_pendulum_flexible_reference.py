# -*- coding: utf-8 -*-
r"""Independent, third-party-software cross-validation of FSM's "more
flexible" double pendulum (`Examples_flexible_FSM/flex_double_pendulum_flexible.py`,
EXAMPLE S in FSM_table_generator.py) using REAL PyChrono.

This script is NOT part of the normal pytest suite and cannot be imported
or collected by it: PyChrono is only installed in a separate conda
environment named `chrono` (see below), not in the `multibody_env`
environment the rest of this repository's tests run under. Instead, this
script is a one-off, manually-run verification tool.

Companion to `_pychrono_double_pendulum_moderate_reference.py` in this same
folder: SAME nominal EI and EI_BOOST (nominal=1e6, boost=100, absolute
boosted EI=1e8 -- the proven-stable value from the moderate case), but
density (RHO) increased 4x (10000/50000 -> 20000/100000) to match
`flex_double_pendulum_flexible.py`, driving more bending via increased
inertial loading instead of lower stiffness.

Two lower-EI attempts were tried FIRST and REJECTED (see repo memory for
full details) before landing on "raise density instead":
  - nominal EI=1e5, boost=100 (absolute=1e7): catastrophic blowup, root_theta
    swung to -64 deg (vs FSM's ~9 deg) -- same geometric-coupling instability
    as the original nominal EI=1e6/no-boost case.
  - nominal EI=3e5, boost=100 (absolute=3e7): no outright blowup, but a
    significant, opposite-skewed divergence from FSM (-11.2 to +2.5 deg vs
    FSM's -2.0 to +7.2 deg at the same EI) -- early-onset distortion, not a
    trustworthy match. Confirms the Chrono absolute-EI stability floor for
    this geometry/IC sits close to 1e8, not far below it, and that lowering
    EI is the wrong lever once a validated EI_BOOST is already in hand (see
    `build_link`/EI_BOOST below) -- prefer raising mass/inertia instead.

How to run this script
-----------------------
PyChrono lives in a separate conda env (`chrono`), and importing it
requires prepending its DLL directories to PATH first:

    $env:PATH = "C:\Users\<user>\Anaconda3\envs\chrono;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Library\bin;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Scripts;" + $env:PATH
    C:\Users\<user>\Anaconda3\envs\chrono\python.exe _pychrono_double_pendulum_flexible_reference.py
"""
import math
import os

import numpy as np
import pychrono as chrono
import pychrono.fea as fea

G_ACC = 9.81
A_STIFF = 1.0e4
EI_BOOST = 100.0


def chrono_pos(xz):
    return chrono.ChVectorD(xz[0], xz[1], 0.0)


def chrono_quat(theta_fsm):
    return chrono.Q_from_AngAxis(-theta_fsm, chrono.ChVectorD(0, 0, 1))


def make_section(E, A, I, rho):
    sec = fea.ChBeamSectionEulerAdvanced()
    sec.SetYoungModulus(E)
    sec.SetGshearModulus(E / 2.6)
    sec.SetDensity(rho)
    sec.SetArea(A)
    sec.SetIyy(I)
    sec.SetIzz(I)
    sec.SetBeamRaleyghDampingAlpha(0.0)
    sec.SetBeamRaleyghDampingBeta(0.0)
    return sec


def build_link(mesh, root_node, L, n_seg, E, rho, theta, theta_dot):
    dx = L / n_seg
    dx_half = dx / 2.0
    nodes = [root_node]
    elements_info = []
    prev_node = root_node
    prev_xz = np.array([root_node.GetPos().x, root_node.GetPos().y])
    joint_vel = np.array([root_node.GetPos_dt().x, root_node.GetPos_dt().y])
    for i in range(n_seg):
        th = theta[i]
        thd = theta_dot[i]
        next_xz = prev_xz + dx * np.array([math.cos(th), -math.sin(th)])
        seg_cg_vel = joint_vel + thd * dx_half * np.array([-math.sin(th), -math.cos(th)])
        node_vel = seg_cg_vel + thd * dx_half * np.array([-math.sin(th), -math.cos(th)])

        new_node = fea.ChNodeFEAxyzrot(chrono.ChFrameD(chrono_pos(next_xz), chrono_quat(th)))
        new_node.SetPos_dt(chrono.ChVectorD(node_vel[0], node_vel[1], 0.0))
        new_node.SetWvel_par(chrono.ChVectorD(0, 0, -thd))
        mesh.AddNode(new_node)

        ei_here = E * EI_BOOST
        elem = fea.ChElementBeamEuler()
        elem.SetNodes(prev_node, new_node)
        elem.SetSection(make_section(ei_here, A_STIFF, 1.0, rho / A_STIFF))
        mesh.AddElement(elem)
        elements_info.append((elem, ei_here, dx))

        nodes.append(new_node)
        prev_node, prev_xz, joint_vel = new_node, next_xz, node_vel
    return nodes, elements_info


def build_system(omega_root, omega_tip):
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, -G_ACC, 0))
    system.SetSolver(chrono.ChSolverSparseQR())
    system.SetTimestepperType(chrono.ChTimestepper.Type_TRAPEZOIDAL)

    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(True)
    system.Add(mesh)

    L1, L2 = 2.0, 2.0
    EI1, EI2 = 1.0e6, 1.0e6  # SAME as the validated "moderate" case -- unchanged
    RHO1, RHO2 = 20000.0, 100000.0  # 2x denser than "moderate" (was 10000/50000)
    N1, N2 = 6, 6
    n_total = N1 + N2
    theta_all = np.full(n_total, np.pi / 2.0)
    theta_dot_all = np.linspace(omega_root, omega_tip, n_total)

    ground = chrono.ChBody()
    ground.SetBodyFixed(True)
    system.Add(ground)

    root1 = fea.ChNodeFEAxyzrot(chrono.ChFrameD(chrono_pos([0.0, 0.0]), chrono_quat(np.pi / 2)))
    root1.SetPos_dt(chrono.ChVectorD(0, 0, 0))
    root1.SetWvel_par(chrono.ChVectorD(0, 0, 0))
    mesh.AddNode(root1)

    joint_ground = chrono.ChLinkMateGeneric(True, True, True, True, True, False)
    joint_ground.Initialize(ground, root1, False, chrono.ChFrameD(chrono_pos([0.0, 0.0])),
                             chrono.ChFrameD(chrono_pos([0.0, 0.0])))
    system.Add(joint_ground)

    link1_nodes, link1_elems = build_link(mesh, root1, L1, N1, EI1, RHO1,
                                           theta_all[:N1], theta_dot_all[:N1])
    link1_tip = link1_nodes[-1]

    root2 = fea.ChNodeFEAxyzrot(chrono.ChFrameD(link1_tip.GetPos(), link1_tip.GetRot()))
    root2.SetPos_dt(link1_tip.GetPos_dt())
    root2.SetWvel_par(link1_tip.GetWvel_par())
    mesh.AddNode(root2)

    joint_mid = chrono.ChLinkMateGeneric(True, True, True, True, True, False)
    joint_mid.Initialize(link1_tip, root2, False, chrono.ChFrameD(), chrono.ChFrameD())
    system.Add(joint_mid)

    link2_nodes, link2_elems = build_link(mesh, root2, L2, N2, EI2, RHO2,
                                           theta_all[N1:], theta_dot_all[N1:])
    link2_tip = link2_nodes[-1]

    dx1, dx2 = L1 / N1, L2 / N2
    segments = []
    for i in range(N1):
        segments.append((link1_nodes[i], link1_nodes[i + 1], RHO1 * dx1, dx1))
    for i in range(N2):
        segments.append((link2_nodes[i], link2_nodes[i + 1], RHO2 * dx2, dx2))

    elements_info = link1_elems + link2_elems

    return system, root1, link1_tip, link2_tip, segments, elements_info


def compute_energy(segments):
    # Two-point lumped KE (removes the single-average-CG underestimation bias
    # -- see the moderate-case reference script for the derivation) + gravity
    # PE; elastic/bending strain energy tracked separately below.
    ke_total = 0.0
    pe_total = 0.0
    for node_a, node_b, m, dx in segments:
        pa, pb = node_a.GetPos(), node_b.GetPos()
        va, vb = node_a.GetPos_dt(), node_b.GetPos_dt()
        wa, wb = node_a.GetWvel_par().z, node_b.GetWvel_par().z
        cg_y = 0.5 * (pa.y + pb.y)
        J_half = (m * dx * dx / 12.0) / 2.0
        ke_total += 0.5 * (m / 2.0) * (va.x ** 2 + va.y ** 2) + 0.5 * (m / 2.0) * (vb.x ** 2 + vb.y ** 2)
        ke_total += 0.5 * J_half * wa ** 2 + 0.5 * J_half * wb ** 2
        pe_total += m * G_ACC * cg_y
    return ke_total, pe_total


def compute_elastic_energy(elements_info):
    # Bending strain energy per element via Simpson's rule on M(eta)^2/(2*EI),
    # using the TRUE internal bending moment (EvaluateSectionForceTorque's
    # torque.z) sampled at eta=-1,0,1 (exact since M is linear in eta).
    energy = 0.0
    force = chrono.ChVectorD(0, 0, 0)
    torque = chrono.ChVectorD(0, 0, 0)
    for elem, ei, dx in elements_info:
        m_vals = []
        for eta in (-1.0, 0.0, 1.0):
            elem.EvaluateSectionForceTorque(eta, force, torque)
            m_vals.append(torque.z)
        m_minus, m_mid, m_plus = m_vals
        energy += (dx / (12.0 * ei)) * (m_minus ** 2 + 4.0 * m_mid ** 2 + m_plus ** 2)
    return energy


def fsm_angle_of(body_or_node):
    v = body_or_node.TransformDirectionLocalToParent(chrono.ChVectorD(1, 0, 0))
    return -math.atan2(v.y, v.x)


if __name__ == "__main__":
    OMEGA_ROOT, OMEGA_TIP = 0.05, 0.5
    system, root1, link1_tip, link2_tip, segments, elements_info = build_system(OMEGA_ROOT, OMEGA_TIP)
    # Without this, the very first EvaluateSectionForceTorque call (at t=0,
    # before any DoStepDynamics) returns NaN -- see repo memory.
    system.Setup()
    system.Update()

    tspan, dt = 2.0, 2e-4
    n_steps = int(round(tspan / dt))
    report_every = max(1, n_steps // 400)

    t_list, tip_x, tip_z, root_theta, ke_list, pe_list, elastic_list = [], [], [], [], [], [], []
    for k in range(n_steps + 1):
        if k % report_every == 0 or k == n_steps:
            t_list.append(k * dt)
            tip_x.append(link2_tip.GetPos().x)
            tip_z.append(link2_tip.GetPos().y)
            root_theta.append(fsm_angle_of(root1))
            ke, pe = compute_energy(segments)
            ke_list.append(ke)
            pe_list.append(pe)
            elastic_list.append(compute_elastic_energy(elements_info))
        if k < n_steps:
            system.DoStepDynamics(dt)

    t_arr = np.array(t_list)
    root_theta_arr = np.array(root_theta)
    deg_from_eq = (root_theta_arr - math.pi / 2) * 180 / math.pi
    ke_arr, pe_arr, elastic_arr = np.array(ke_list), np.array(pe_list), np.array(elastic_list)
    total_e = ke_arr + pe_arr
    total_e_full = ke_arr + pe_arr + elastic_arr
    print("Chrono root_theta range (deg from equilibrium):", deg_from_eq.min(), deg_from_eq.max(), flush=True)
    print("Chrono KE+PE_grav at t=0:", total_e[0], flush=True)
    print("Chrono KE+PE_grav rel drift:", (total_e.max() - total_e.min()) / abs(total_e[0]), flush=True)
    print("Chrono elastic energy range:", elastic_arr.min(), elastic_arr.max(), flush=True)
    print("Chrono KE+PE_grav+elastic rel drift:",
          (total_e_full.max() - total_e_full.min()) / abs(total_e_full[0]), flush=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "truth",
                             "_pychrono_double_pendulum_flexible_traj.npz")
    np.savez(out_path, t=t_arr, tip_x=np.array(tip_x), tip_z=np.array(tip_z), root_theta=root_theta_arr,
             ke=ke_arr, pe=pe_arr, elastic=elastic_arr)
    print("SAVED", out_path, flush=True)

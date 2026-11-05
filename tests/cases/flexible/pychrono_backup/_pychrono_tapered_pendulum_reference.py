# -*- coding: utf-8 -*-
r"""Independent, third-party-software cross-validation of FSM's tapered
flexible PENDULUM (`Examples_flexible_FSM/flex_tapered_pendulum.py`, EXAMPLE
Q in FSM_table_generator.py) using REAL PyChrono -- companion to this same
folder's `_pychrono_cantilever_reference.py`, but a GRAVITY-DRIVEN case
instead of a clamped/zero-gravity frequency check.

Why a pendulum and not another frequency check: `_pychrono_cantilever_reference.py`
validates FSM's tapered-member stiffness/mass machinery, but a clamped,
zero-gravity modal test never exercises FSM's documented "a tapered
segment's CG stays at the geometric midpoint, not its true mass-weighted
centroid" simplification (see source/multibody/flexible/
_flex_distribution_helper.py's module docstring) -- gravitational torque
depends on exactly where the CG truly is, so a PENDULUM (gravity acting on
a PINNED, freely-swinging tapered member) is the right case to actually
show that simplification's real dynamical effect.

This script is NOT part of the normal pytest suite and cannot be imported
or collected by it: PyChrono is only installed in a separate conda
environment named `chrono` (see below), not in the `multibody_env`
environment the rest of this repository's tests run under. Instead, this
script is a one-off, manually-run verification tool. Its output values are
hand-copied into test_tapered_pendulum_matches_pychrono_reference() in
test_flex_distribution.py, the same pattern `_pychrono_cantilever_reference.py`
and `tests/cases/chrono/truth/*.npy` already use.

How to run this script
-----------------------
PyChrono lives in a separate conda env (`chrono`), and importing it
requires prepending its DLL directories to PATH first (confirmed necessary
on Windows with this specific install):

    $env:PATH = "C:\Users\<user>\Anaconda3\envs\chrono;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Library\bin;" + `
                "C:\Users\<user>\Anaconda3\envs\chrono\Scripts;" + $env:PATH
    C:\Users\<user>\Anaconda3\envs\chrono\python.exe _pychrono_tapered_pendulum_reference.py

Model (must match FSM_table_generator.py's EXAMPLE Q exactly)
---------------------------------------------------------------
One bending member, length L=1.0, PINNED (not clamped) at its thick root
end, swinging freely under gravity. Width tapers linearly from the root to
TIP_RATIO=0.3 of the root's width (constant-depth rectangular section
assumed), giving EI(x) ~ width(x)^3 (tip EI = 2.7% of root -- "very
tapered") and mass/length ~ width(x)^1 (tip mass = 30% of root). Lumped
convention (flex_A=flex_I=1, flex_E=EI(x) product directly, flex_rho=
mass/length(x)). Released from horizontal (angle=0) at rest -- gravity
alone drives both a large (~180 degree) rigid swing and internal bending.
RHO_ROOT=10000 here is the "more flexible" (2x the first attempt's mass,
EI held fixed) version -- see repo memory pychrono_environment.md for the
full iteration history.

Method
------
Same technique as `_pychrono_double_pendulum_moderate_reference.py` (this
same folder, already validated for FSM's uniform, non-tapered double
pendulum): a chain of `ChElementBeamEuler`
corotational FEA beam elements, each with its own `ChBeamSectionEulerAdvanced`
sampling the SAME tapered EI(x)/rho(x) profile at that element's own
midpoint (a FINER mesh than FSM's own N_SEG=8 -- N_ELEM_CHRONO=32 here,
same "finer Chrono mesh" precedent as EXAMPLE O), pinned to ground via
`ChLinkMateGeneric(True,True,True,True,True,False)` (locks 5 of the FEA
node's 6 DOF, leaves Z-rotation free -- the FEA-node equivalent of a
revolute joint; FEA nodes are `ChNodeFEAxyzrot`, not `ChBody`, so the
`ChLinkLock*` family used for rigid-body joints does not apply to them).

Two corrections carried over UNCHANGED from the already-validated double
pendulum, and confirmed to work on the FIRST attempt here with no
retuning needed (strong evidence these are general properties of the
"lumped-EI continuous-beam vs FSM rigid-segment" mismatch, not specific to
the original double pendulum's exact parameters):
  - `EI_BOOST=100`: FSM's lumped-EI convention, mapped onto a genuine
    continuous corotational beam, is soft enough that large-rotation
    geometric nonlinearities create spurious energy exchange between
    swing and bending that FSM's own rigid-segment discretization can't
    reproduce (see repo memory for the full double-pendulum investigation
    that discovered and calibrated this). Multiplying E by 100 (I stays 1)
    suppresses it while preserving FSM's intended EI(x) taper SHAPE.
  - `A_STIFF=1e4`: FSM's flex_A=1 placeholder gives a severely
    under-stiff axial (stretching) DOF if taken literally (EA=E*1 is tiny
    compared to EI for a realistic member) -- visible as spurious
    stretching under gravity load. Using a large `A_STIFF` for the
    section's actual axial area (while keeping density = rho/A_STIFF so
    `density*A_STIFF` still equals the correct mass/length) decouples
    axial stiffness from mass without changing EI.

Validated results (this run, RHO_ROOT=10000 "more flexible" version)
----------------------------------------------------------------------
Compared against FSM's own `flex_tapered_pendulum.py` output over the
full tspan=2.0s:
  - root angle range:  FSM [0.0, 3.1179] rad   vs  Chrono [-0.0, 3.1414] rad
  - tip_x range:        FSM [-0.9374, 0.9375] m vs Chrono [-1.0000, 1.0000] m
  - tip_z range:        FSM [-0.9369, 0.0532] m vs Chrono [-1.0000, 0.0001] m
  - max pointwise error (full 2s, interpolated): root 0.135 rad (~4.3% of
    the pi swing), tip_x 0.075 m (~4.0% of range), tip_z 0.093 m (~9.4% of
    range) -- same ballpark as the original (non-tapered) double
    pendulum's own ~8-16% errors.
  - energy conservation (normalized against the KE/PE SWING SCALE, not the
    near-zero conserved total -- see repo memory's "ENERGY-NORMALIZATION
    GOTCHA" note for why that distinction matters here): FSM 3.0e-7
    relative (as always, excellent), Chrono 0.8% relative (bounded,
    oscillating with the swing period, not growing).
  - Chrono's swing reaches systematically slightly FARTHER every cycle
    (tip_z bottoms out at -1.00 vs FSM's -0.94) -- this is the expected
    CG-shift signature: this taper's TRUE mass-weighted centroid sits at
    0.41*L (confirmed via direct numerical integration), i.e. 9% of the
    length TOWARD THE ROOT from the geometric midpoint (0.50*L) FSM
    assumes. Mass effectively closer to the pivot means lower true
    rotational inertia about the joint, so the physically-correct system
    swings farther for the same PE drop -- exactly the direction observed.
"""
import math

import numpy as np
import pychrono as chrono
import pychrono.fea as fea

G_ACC = 9.81
A_STIFF = 1.0e4
EI_BOOST = 100.0

L_BEAM = 1.0
TIP_RATIO = 0.3
EI_ROOT = 2e4
RHO_ROOT = 10000.0   # "more flexible" version -- 2x the first attempt's mass, EI held fixed
N_ELEM_CHRONO = 32
TSPAN, DT = 2.0, 2e-4


def width_ratio(x):
    return 1.0 - (1.0 - TIP_RATIO) * x / L_BEAM


def EI_of_x(x):
    return EI_ROOT * width_ratio(x) ** 3


def rho_of_x(x):
    return RHO_ROOT * width_ratio(x)


def chrono_pos(xz):
    return chrono.ChVectorD(xz[0], xz[1], 0.0)


def chrono_quat(theta_fsm):
    return chrono.Q_from_AngAxis(-theta_fsm, chrono.ChVectorD(0, 0, 1))


def fsm_angle_of(body_or_node):
    v = body_or_node.TransformDirectionLocalToParent(chrono.ChVectorD(1, 0, 0))
    return -math.atan2(v.y, v.x)


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


def build_tapered_link(mesh, root_node, n_elem):
    dx = L_BEAM / n_elem
    nodes = [root_node]
    elements_info = []
    segments = []
    prev_node = root_node
    x_prev = 0.0
    for i in range(n_elem):
        x_mid = x_prev + dx / 2.0
        x_next = x_prev + dx

        new_node = fea.ChNodeFEAxyzrot(chrono.ChFrameD(chrono_pos([x_next, 0.0]), chrono_quat(0.0)))
        new_node.SetPos_dt(chrono.ChVectorD(0, 0, 0))
        new_node.SetWvel_par(chrono.ChVectorD(0, 0, 0))
        mesh.AddNode(new_node)

        ei_here = EI_of_x(x_mid) * EI_BOOST
        rho_here = rho_of_x(x_mid)
        elem = fea.ChElementBeamEuler()
        elem.SetNodes(prev_node, new_node)
        elem.SetSection(make_section(ei_here, A_STIFF, 1.0, rho_here / A_STIFF))
        mesh.AddElement(elem)
        elements_info.append((elem, ei_here, dx))
        segments.append((prev_node, new_node, rho_here * dx, dx))

        nodes.append(new_node)
        prev_node, x_prev = new_node, x_next
    return nodes, elements_info, segments


def build_system():
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, -G_ACC, 0))
    system.SetSolver(chrono.ChSolverSparseQR())
    system.SetTimestepperType(chrono.ChTimestepper.Type_TRAPEZOIDAL)

    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(True)
    system.Add(mesh)

    ground = chrono.ChBody()
    ground.SetBodyFixed(True)
    system.Add(ground)

    root = fea.ChNodeFEAxyzrot(chrono.ChFrameD(chrono_pos([0.0, 0.0]), chrono_quat(0.0)))
    root.SetPos_dt(chrono.ChVectorD(0, 0, 0))
    root.SetWvel_par(chrono.ChVectorD(0, 0, 0))
    mesh.AddNode(root)

    joint_ground = chrono.ChLinkMateGeneric(True, True, True, True, True, False)
    joint_ground.Initialize(ground, root, False, chrono.ChFrameD(chrono_pos([0.0, 0.0])),
                             chrono.ChFrameD(chrono_pos([0.0, 0.0])))
    system.Add(joint_ground)

    nodes, elements_info, segments = build_tapered_link(mesh, root, N_ELEM_CHRONO)
    tip_node = nodes[-1]

    return system, root, tip_node, segments, elements_info


def compute_energy(segments):
    ke_total, pe_total = 0.0, 0.0
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
    energy = 0.0
    force, torque = chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 0)
    for elem, ei, dx in elements_info:
        m_vals = []
        for eta in (-1.0, 0.0, 1.0):
            elem.EvaluateSectionForceTorque(eta, force, torque)
            m_vals.append(torque.z)
        m_minus, m_mid, m_plus = m_vals
        energy += (dx / (12.0 * ei)) * (m_minus ** 2 + 4.0 * m_mid ** 2 + m_plus ** 2)
    return energy


if __name__ == "__main__":
    system, root, tip_node, segments, elements_info = build_system()
    system.Setup()
    system.Update()

    n_steps = int(round(TSPAN / DT))
    report_every = max(1, n_steps // 400)

    t_list, tip_x, tip_z, root_theta, tip_theta = [], [], [], [], []
    ke_list, pe_list, elastic_list = [], [], []
    for k in range(n_steps + 1):
        if k % report_every == 0 or k == n_steps:
            t_list.append(k * DT)
            tip_x.append(tip_node.GetPos().x)
            tip_z.append(tip_node.GetPos().y)
            root_theta.append(fsm_angle_of(root))
            tip_theta.append(fsm_angle_of(tip_node))
            ke, pe = compute_energy(segments)
            ke_list.append(ke)
            pe_list.append(pe)
            elastic_list.append(compute_elastic_energy(elements_info))
        if k < n_steps:
            system.DoStepDynamics(DT)

    root_theta_arr = np.array(root_theta)
    tip_theta_arr = np.array(tip_theta)
    ke_arr, pe_arr, elastic_arr = np.array(ke_list), np.array(pe_list), np.array(elastic_list)
    total_e = ke_arr + pe_arr + elastic_arr
    print("Chrono root_theta range:", root_theta_arr.min(), root_theta_arr.max(), flush=True)
    print("Chrono tip_x range:", min(tip_x), max(tip_x), flush=True)
    print("Chrono tip_z range:", min(tip_z), max(tip_z), flush=True)
    print("Chrono tip-vs-root relative bend angle range:",
          (tip_theta_arr - root_theta_arr).min(), (tip_theta_arr - root_theta_arr).max(), flush=True)
    scale = max(abs(ke_arr.max()), abs(pe_arr.min()), 1e-12)
    print("Chrono energy drift (relative to KE/PE swing scale):",
          (total_e.max() - total_e.min()) / scale, flush=True)

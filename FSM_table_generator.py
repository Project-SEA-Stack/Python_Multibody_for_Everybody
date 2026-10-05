# main_FSM_table_generator.py
"""
@author: Sahand Sabet

Define a flexible pendulum with the Table 1 / Table 2 interface, then
generate a standalone model file from it.

Run this script directly -- it will:
  1) build Table 1 (bodies/joints) and Table 2 (flexible-member properties),
  2) call mb.generate_flex_model_file(...) to write a real, standalone .py
     model file (printed path below), under Examples_flexible_FSM/.

The generated file itself is an ordinary example script, in the SAME
flat-list convention as any hand-written Examples_mbd/ script -- open it to
see the fully expanded bodies/joints/springs, or add your own forces
directly in it past the "USER ADDITIONS" marker. To actually run it (build
the MbdSystem, integrate, plot, animate), point main.py's "Example to
import" section at it, exactly like main.py already does for any
hand-written example:

    from Examples_flexible_FSM import _generated_flex_pendulum as ex

then just run `python main.py`.
"""
import sys
import os

source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'source'))
sys.path.insert(0, source_dir)

import numpy as np

import multibody as mb

# ---------------------------------------------------------------------------
# EXAMPLE A: single flexible pendulum (one flex member, pinned at ground).
# Commented out for now -- uncomment this block (and comment out EXAMPLE B
# below) to switch back to it.
# ---------------------------------------------------------------------------
# N_SEG = 6
# L_BEAM = 1.0
# dx = L_BEAM / N_SEG
#
# joints = [[0, 1]]
# types = ['R']
# parent_cg_to_joint = [[0.0, 0.0]]        # root joint sits exactly at the origin
# joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]  # joint -> the WHOLE member's own CG (its
#                                             # midpoint), not any one segment's CG --
#                                             # unused for this free-tip row, but kept
#                                             # geometrically honest (L comes from flex_L)
# prismatic_direction = [[np.nan, np.nan]]
# flex_bd = [1]            # nan would mean rigid; flexible rows use their own body index
# m0 = [None]              # ignored for a flex row (mass comes from Table 2)
# J0 = [None]              # ignored for a flex row (inertia comes from Table 2)
#
# flex_bds = [1]   # Table 2 row 0 belongs to body 1
# flex_n_seg = [N_SEG]
# flex_L = [L_BEAM]
# flex_E = [200e9]
# flex_A = [1e-4]
# flex_I = [1e-8]
# flex_rho = [7800.0]
# flex_deformation_mode = ["bending"]
# flex_clamp_on = [False]   # pinned root -> swings freely, still bends internally
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_pendulum.py")
#
# n_bodies = N_SEG
# ic = np.zeros(2 * n_bodies)
# ic[:n_bodies] = np.pi / 2 + 0.5   # all segments start ~0.5 rad past vertical
#
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.ones(n_bodies), tspan=5.0, TimeStep=0.02,
#     animation_on=1, SaveMovieOn=None, plotTstep=5,
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE C: single flexible member, CLAMPED root (same topology as
# cantilever_FSM.py; illustrative "soft" stiffness scale here, released from
# a bent shape, rather than cantilever_FSM.py's own real-steel/eigenvalue
# setup -- see fsm_table_generator_validate.py for the exact steel-parameter
# reproduction). Commented out -- see Examples_flexible_FSM/comparison_bending/
# generate_all_cases.py (case_C) for the full, runnable, plotted version.
# ---------------------------------------------------------------------------
# N_SEG, L_BEAM, EI = 6, 1.0, 12.0
# joints = [[0, 1]]
# types = ['R']
# parent_cg_to_joint = [[0.0, 0.0]]
# joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]]
# flex_bd, m0, J0 = [1], [None], [None]
# flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
# flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [1.0]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]   # clamped, not pinned
#
# n_bodies = N_SEG
# ic = np.zeros(2 * n_bodies)
# ic[:n_bodies] = np.linspace(0.3 / N_SEG, 0.3, N_SEG)   # bent shape, released from rest
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_cantilever.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.zeros(n_bodies), tspan=2.0, TimeStep=0.005,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE D: clamped cantilever + a rigid TIP PAYLOAD (subedi_link3_compare.py/
# deluca_link1_compare.py's pattern: a flex row followed by one extra RIGID
# row for the payload mass/inertia -- the alternative to the cancelled
# automatic-payload-merge "Phase 2", per this session's earlier design
# decision). Commented out -- see generate_all_cases.py (case_D).
# ---------------------------------------------------------------------------
# N_SEG, L_BEAM, EI = 6, 1.0, 12.0
# joints = [[0, 1], [1, 2]]      # row 1 = beam (flex), row 2 = payload (rigid, tip)
# types = ['R', 'R']
# parent_cg_to_joint = [[0.0, 0.0], [L_BEAM / 2.0, 0.0]]
# joint_to_child_cg = [[L_BEAM / 2.0, 0.0], [0.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]] * 2
# flex_bd = [1, np.nan]           # row 2 is rigid (the payload)
# m0 = [None, 0.3]               # payload mass
# J0 = [None, 0.01]              # payload inertia
# flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
# flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [1.0]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]
#
# n_bodies = N_SEG + 1   # + the rigid payload body
# ic = np.zeros(2 * n_bodies)
# ic[:N_SEG] = np.linspace(0.3 / N_SEG, 0.3, N_SEG)
# ic[N_SEG] = ic[N_SEG - 1]   # payload starts aligned with the beam tip
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_cantilever_payload.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.zeros(n_bodies), tspan=2.0, TimeStep=0.005,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE E: two flex links chained via REAL (non-negligible) rigid "hub"
# bodies, each carrying its own genuine mass/inertia, plus a tip payload
# (deluca_twolink_fig5.py's pattern). Unlike the EARLIER near-massless "hub
# trick" (only needed to decouple swing/bend for matching an external
# assumed-mode model -- see EXAMPLE B's own history), these hubs represent
# REAL lumped inertia (e.g. a motor/gearbox at a joint). Commented out --
# see generate_all_cases.py (case_E).
# ---------------------------------------------------------------------------
L1, L2, N1, N2, EI = 1.0, 1.0, 4, 4, 12.0
# rows: 1=hub1(rigid), 2=link1(flex), 3=hub2(rigid), 4=link2(flex), 5=payload(rigid)
joints = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
types = ['R', 'R', 'R', 'R', 'R']
parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0], [L1 / 2.0, 0.0], [0.0, 0.0], [L2 / 2.0, 0.0]]
joint_to_child_cg = [[0.0, 0.0], [L1 / 2.0, 0.0], [0.0, 0.0], [L2 / 2.0, 0.0], [0.0, 0.0]]
prismatic_direction = [[np.nan, np.nan]] * 5
flex_bd = [np.nan, 2, np.nan, 4, np.nan]
m0 = [0.05, None, 0.1, None, 0.2]     # hub1, --, hub2, --, payload
J0 = [0.01, None, 0.02, None, 0.02]
flex_bds, flex_n_seg, flex_L = [2, 4], [N1, N2], [L1, L2]
flex_E, flex_A, flex_I, flex_rho = [EI, EI], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]
flex_deformation_mode, flex_clamp_on = ["bending", "bending"], [True, True]

n_bodies = 1 + N1 + 1 + N2 + 1
ic = np.zeros(2 * n_bodies)
ic[2 + N1:2 + N1 + N2] = np.linspace(0.3 / N2, 0.3, N2)   # link2 released from a bent shape
ic[2 + N1 + N2] = ic[1 + N1 + N2]                          # payload aligned with link2's tip

output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_hub_chain.py")
mb.generate_flex_model_file(
    joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
    flex_bd, m0, J0,
    flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
    flex_deformation_mode, flex_clamp_on,
    output_path,
    ic=ic, gVec=np.zeros(n_bodies), tspan=2.0, TimeStep=0.005,
    integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
    overwrite=True,
)

# ---------------------------------------------------------------------------
# EXAMPLE F: translating RIGID CART ('P' joint) carrying a CLAMPED flexible
# beam (franco_pendulum_cart_coupled_compare.py's pattern). Demonstrates a
# 'P'-type parent row + a flex row mounted via 'R' -- the flex row's OWN
# mount type is unaffected by its parent's joint type. Commented out -- see
# generate_all_cases.py (case_F).
# ---------------------------------------------------------------------------
# N_SEG, L_BEAM, EI = 6, 1.0, 12.0
# joints = [[0, 1], [1, 2]]      # row 1 = cart (rigid, prismatic), row 2 = beam (flex)
# types = ['P', 'R']
# parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0]]
# joint_to_child_cg = [[0.0, 0.0], [L_BEAM / 2.0, 0.0]]
# prismatic_direction = [[1.0, 0.0], [np.nan, np.nan]]   # cart translates horizontally
# flex_bd = [np.nan, 2]
# m0 = [0.1, None]     # cart mass
# J0 = [1e-4, None]
# flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L_BEAM]
# flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [1.0]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]
#
# n_bodies = 1 + N_SEG
# ic = np.zeros(2 * n_bodies)
# ic[1:n_bodies] = np.linspace(0.3 / N_SEG, 0.3, N_SEG)   # beam released from a bent shape
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_cart_beam.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.zeros(n_bodies), tspan=2.0, TimeStep=0.005,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE G: clamped cantilever pointing UP under gravity -- gravity and
# bending strongly coupled (pai_beam_FSM_validate.py's pattern; parameters
# below match that paper's actual beam). Commented out -- see
# generate_all_cases.py (case_G).
# ---------------------------------------------------------------------------
# N_SEG, L_BEAM = 6, 0.479
# joints = [[0, 1]]
# types = ['R']
# parent_cg_to_joint = [[0.0, 0.0]]
# joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]]
# flex_bd, m0, J0 = [1], [None], [None]
# flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
# flex_E, flex_A, flex_I, flex_rho = [0.048918], [1.0], [1.0], [0.101270]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]
#
# n_bodies = N_SEG
# ic = np.zeros(2 * n_bodies)
# ic[:n_bodies] = -np.pi / 2 + np.linspace(0.015 / N_SEG, 0.015, N_SEG)   # pointing up, tiny bend
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_cantilever_up.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.ones(n_bodies), tspan=1.0, TimeStep=0.002,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE H: VALIDATION -- subedi_link3_compare.py exact reproduction (a
# clamped flex link + a rigid tip body with REAL mass/inertia, mp/Jp -- the
# has-a-child + rigid-child-of-a-flex-parent construction that exposed the
# seam-offset bug, see summary below). Commented out -- see
# Examples_flexible_FSM/comparison_bending/validate_against_papers.py
# (case_D_subedi) for the actual runnable/linearized validation.
#
# Reference: Subedi, Tyapin & Hovland (2021), "Dynamic Modeling of Planar
# Multi-Link Flexible Manipulators," Robotics 10(2):70 (MDPI, open access).
# Link 3 (outermost link, Table 3 targets qr2=qr3=0deg) reproduced here.
#
# VALIDATION SUMMARY (2026-09-21, against subedi_link3_compare.py's
# published targets, via linearization + eigh -> natural frequencies):
#   mp=0 kg (payload has Jp but zero mass): f1=14.10 Hz (-0.48%), f2=86.94 Hz
#     (-2.12%) vs target 14.17/88.82 Hz -- UNCHANGED by the geometry fix
#     below (a massless body's position doesn't affect the dynamics, only
#     Jp does, which is position-independent).
#   mp=2 kg (real payload mass): BEFORE the fix, f1/f2 were off by -39%/-41%
#     (this large error is what surfaced the bug). AFTER the fix: f1=4.88 Hz
#     (-0.32%), f2=63.15 Hz (-1.28%) vs target 4.90/63.97 Hz.
# Root cause (fixed in flex_expand.py's _mounting_offset/_expand_bending_row):
# a rigid body (or another flex row) mounted on a flex row's TIP was using
# the row's raw Table 1 parent_cg_to_joint value (L_parent/2, the HALF-MEMBER
# distance) as its real offset from the last segment's own CG -- wrong,
# since the last segment is only dx_parent=L_parent/n_seg long. Fixed to
# auto-derive dx_parent/2 from the parent row's own generated segment
# instead, along whatever direction that row's own axis was declared in
# (not hardcoded to local +x).
# ---------------------------------------------------------------------------
# L3, EI3, RHO3, JP, MP, N_SEG = 1.5, 2.4114e3, 0.7425, 3.2e-4, 2.0, 10
# joints = [[0, 1], [1, 2]]      # row 1 = link3 (flex, clamped), row 2 = tip payload (rigid)
# types = ['R', 'R']
# parent_cg_to_joint = [[0.0, 0.0], [L3 / 2.0, 0.0]]
# joint_to_child_cg = [[L3 / 2.0, 0.0], [0.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]] * 2
# flex_bd = [1, np.nan]           # row 2 is rigid (the payload)
# m0 = [None, MP]                # payload mass (set to 0.0 for the mp=0 case)
# J0 = [None, JP]                # payload's own rotational inertia
# flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L3]
# flex_E, flex_A, flex_I, flex_rho = [EI3], [1.0], [1.0], [RHO3]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]
#
# n_bodies = N_SEG + 1
# ic = np.zeros(2 * n_bodies)
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_subedi_link3.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.zeros(n_bodies), tspan=1.0, TimeStep=0.01,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )
# # The payload's attachment is a very stiff (not literally rigid) torsional
# # spring -- M4E has no true "weld" joint -- added AFTER generation, in the
# # generated file's own Force['TorsionSpring'] list (past the "USER
# # ADDITIONS" marker), same convention every hand-built reference script uses:
# #   dx = L3 / N_SEG
# #   kr = 2.0 * EI3 * 1.0 / dx
# #   ex.Force["TorsionSpring"].append(([N_SEG, N_SEG + 1], [0.0, 1000.0 * kr]))

# ---------------------------------------------------------------------------
# EXAMPLE I: VALIDATION -- franco_pendulum_cart_coupled_compare.py exact
# reproduction (translating rigid cart + clamped flex beam + rigid tip mass,
# hanging down -- a THREE-row chain: rigid 'P' cart -> flex beam -> rigid
# tip mass, mixing both the cart-mounted-flex-row and rigid-child-of-a-
# flex-row cases at once). Commented out -- see Examples_flexible_FSM/
# comparison_bending/validate_against_papers.py (case_F_franco) for the
# actual runnable/linearized validation.
#
# Reference: Franco, Astolfi & Rodriguez y Baena (2018), "Robust balancing
# control of flexible inverted-pendulum systems," Mechanism and Machine
# Theory 130:539-551. The paper doesn't tabulate a frequency for this exact
# coupled, open-loop cart+beam+tip-mass system -- validate_against_papers.py
# derives one directly from the paper's own Eq. 17-18/Appendix A formulas
# (linearized about the "beam hanging straight down" equilibrium), then
# compares an M4E/FSM model of the same physical system against it.
#
# VALIDATION SUMMARY (2026-09-21): target f1=1.479 Hz (derived from the
# paper's own equations, not directly tabulated). BEFORE the geometry fix:
# -23.03% error. AFTER the fix: n_seg=10 -> +5.43%, n_seg=20 -> +5.60%,
# n_seg=30 -> +5.63% -- a huge improvement, but NOT fully closed, and the
# error does NOT shrink with finer mesh (a stable ~5.5% plateau, not
# converging toward 0) -- this points to a separate modeling/parameter
# mismatch vs. the paper (not investigated further yet), not a leftover
# geometry bug (which would show the large, mesh-independent errors seen
# in the subedi case above BEFORE its own fix, not a small stable plateau).
# ---------------------------------------------------------------------------
# N_SEG = 10
# L, E, I_A, RHO, A0, M0_TIP, MC = 0.305, 9e10, 1.066e-13, 8400.0, 8e-6, 2.75e-2, 0.1
# EI = E * I_A
# RHO_LIN = RHO * A0
# dx = L / N_SEG
# kr = 2.0 * EI * 1.0 / dx   # k_root = 2EI/dx (clamped half-cell)

# joints = [[0, 1], [1, 2], [2, 3]]   # row1=cart(rigid,P), row2=beam(flex), row3=tip mass(rigid)
# types = ['P', 'R', 'R']
# parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0], [L / 2.0, 0.0]]
# joint_to_child_cg = [[0.0, 0.0], [L / 2.0, 0.0], [0.0, 0.0]]
# prismatic_direction = [[1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
# flex_bd = [np.nan, 2, np.nan]
# m0 = [MC, None, M0_TIP]
# J0 = [1e-6, None, 1e-9]
# flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L]
# flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [RHO_LIN]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]

# n_bodies = 1 + N_SEG + 1   # cart + beam segments + tip mass
# ic = np.zeros(2 * n_bodies)
# ic[1:1 + N_SEG] = np.pi / 2   # beam hanging straight down (stable equilibrium)
# ic[1 + N_SEG] = np.pi / 2     # tip mass aligned with the beam's tip

# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_cart_beam_tipmass.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.ones(n_bodies), tspan=1.0, TimeStep=0.01,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )
# # Tip mass attachment -- same "very stiff torsional spring" convention as
# # EXAMPLE H, added AFTER generation:
# #   ex.Force["TorsionSpring"].append(([1 + N_SEG, 2 + N_SEG], [0.0, 1000.0 * kr]))

# ---------------------------------------------------------------------------
# EXAMPLE J: COMPLEX MIXED-DEFORMATION-MODE chain (Phase 3 demo). Ground ->
# rigid hub ('R') -> AXIAL member ('P', clamped) -> BENDING member ('R',
# clamped) -> rigid tip payload ('R'). Exercises all THREE cross-mode
# mounting cases in one chain: rigid->axial (hub's own physical extent),
# axial->bending (a bending row mounted directly on an axial row's phantom
# tip anchor -- correctly uses ZERO extra offset, since the anchor already
# sits exactly at the axial member's true end), and bending->rigid (the
# payload, using the same dx_parent/2 seam fix validated earlier against
# subedi). Commented out -- see this file's own repo memory entry
# (fsm-core-table-design.md, "PHASE 3 IMPLEMENTED") for the full
# verification: every body's position matched hand calculation exactly,
# and the axial segment count reserves n_seg+1 real bodies (one phantom
# tip anchor beyond the n_seg real segments -- a genuine physics
# requirement, not a bookkeeping quirk, since a free axial tip has no
# equivalent to bending's "naturally zero moment" free end).
# ---------------------------------------------------------------------------
# HUB_HALF, L_AX, L_BEND, N_AX, N_BEND = 0.05, 0.8, 0.6, 4, 4
# EA, RHO_AX = 1.0e6, 2.0      # soft, illustrative axial stiffness (not real steel)
# EI, RHO_BEND = 50.0, 1.0     # soft, illustrative bending stiffness (lumped EI convention)
#
# joints = [[0, 1], [1, 2], [2, 3], [3, 4]]
# types  = ['R', 'P', 'R', 'R']
# parent_cg_to_joint  = [[0.1, 0.0], [HUB_HALF, 0.0], [L_AX / 2.0, 0.0], [L_BEND / 2.0, 0.0]]
# joint_to_child_cg   = [[HUB_HALF, 0.0], [L_AX / 2.0, 0.0], [L_BEND / 2.0, 0.0], [0.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]]
# flex_bd = [np.nan, 2, 3, np.nan]  # hub (rigid), axial member, bending member, payload (rigid)
# m0 = [0.05, None, None, 0.2]     # hub mass, --, --, payload mass
# J0 = [0.001, None, None, 0.001]
#
# flex_bds              = [2, 3]
# flex_n_seg            = [N_AX, N_BEND]
# flex_L                = [L_AX, L_BEND]
# flex_E                = [EA, EI]
# flex_A                = [1.0, 1.0]
# flex_I                = [1.0, 1.0]
# flex_rho              = [RHO_AX, RHO_BEND]
# flex_deformation_mode = ["axial", "bending"]   # row 1 = axial, row 2 = bending
# flex_clamp_on         = [True, True]
#
# n_bodies = 1 + (N_AX + 1) + N_BEND + 1   # hub + (axial segs + tip anchor) + bending segs + payload
# ic = np.zeros(2 * n_bodies)
# ic[1 + N_AX + 1:1 + N_AX + 1 + N_BEND] = np.linspace(0.2 / N_BEND, 0.2, N_BEND)  # bending released from a bent shape
#
# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_mixed_system.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.zeros(n_bodies), tspan=1.0, TimeStep=0.01,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE K: FLOATING body ('F', 3 DOF: X,Z,Theta) + AXIAL flexible member +
# a rigid tip mass welded to the member's far end. Pure mechanics (no
# gravity/hydrodynamics) -- demonstrates a floating body connected
# (indirectly, through the axial member) to a flexible body, with an EXACT
# analytical target: the classic two-mass-and-a-spring free oscillator,
# omega = sqrt(k_eff * (1/M1 + 1/M2)), k_eff = E*A/L (the member's own
# total series axial stiffness, root to tip). Commented out -- see
# tests/cases/flexible/test_fsm_table_generator.py for the automated
# regression check (verified: -0.004% error against this exact formula,
# using a near-massless beam so its own distributed inertia doesn't
# perturb the idealized massless-spring assumption).
# The tip mass is WELDED (not a free hinge) via a manually-added, very
# stiff, zero-rest-length TensionSpring -- M4E has no literal "weld" joint,
# same convention as the bending tip-payload cases (EXAMPLE H/I).
# ---------------------------------------------------------------------------
# M1, M2 = 5.0, 2.0
# L, E, A, RHO = 1.0, 1e6, 1.0, 1e-6   # RHO tiny -- beam's own mass negligible vs M1/M2
# N_SEG = 6
# dx = L / N_SEG

# joints = [[0, 1], [1, 2], [2, 3]]
# types = ['F', 'P', 'P']    # row1=floating body, row2=axial member, row3=tip mass (welded)
# parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0], [L / 2.0, 0.0]]
# joint_to_child_cg = [[np.nan, np.nan], [L / 2.0, 0.0], [0.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan], [1.0, 0.0], [1.0, 0.0]]
# flex_bd = [np.nan, 2, np.nan]
# m0 = [M1, None, M2]
# J0 = [1.0, None, 1e-6]
# flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L]
# flex_E, flex_A, flex_I, flex_rho = [E], [A], [1.0], [RHO]
# flex_deformation_mode, flex_clamp_on = ["axial"], [True]

# n_total_dof = 3 + (N_SEG + 1) + 1   # float=3 DOF + axial (n_seg+1 real joints) + tip mass
# ic = np.zeros(2 * n_total_dof)
# ic[n_total_dof] = 0.5   # kick the floating body's own X-velocity -- with the tip
#                         # mass initially at rest, this excites the axial (spring)
#                         # mode between the two masses (zero IC alone shows NO motion
#                         # at all -- there's nothing to excite it without this).

# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_float_axial.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on, output_path,
#     ic=ic, gVec=np.zeros(1 + (N_SEG + 1) + 1), tspan=1.0, TimeStep=0.01,
#     overwrite=True,
# )
# # Weld the tip mass rigidly to the axial member's own phantom tip anchor
# # (added AFTER generation, real body numbers):
#   tip_anchor_real, payload_real = 1 + N_SEG + 1, 1 + N_SEG + 2
#   k_weld = 1000.0 * (2.0 * E * A / dx)
#   ex.Force["TensionSpring"].append(
#       ((f"CG{tip_anchor_real}_{tip_anchor_real}", f"CG{payload_real}_{payload_real}"), [0.0, k_weld]))

# ---------------------------------------------------------------------------
# EXAMPLE L: FLOATING body with a SMALL mass/inertia (approximating a
# nearly-free end) + a clamped BENDING member, free tip. With the floating
# body light relative to the beam, the member behaves close to an ordinary
# uniform FREE-FREE beam -- analytical target (matched almost exactly with
# an even smaller M_FLOAT, see tests/cases/flexible/test_fsm_table_generator.py):
# the classic free-free beam eigenvalues (beta_n*L = 4.73004, 7.85320,
# 10.99561), omega_n = (beta_n*L)^2 * sqrt(EI/(rho*A*L^4)).
# NOTE: the pytest/analytical-match version uses M_FLOAT=1e-6 -- fine for
# the LINEARIZED eigenvalue check (no time-stepping involved), but a
# genuinely bad choice for actual nonlinear time integration (a near-zero
# mass reacting to any force needs near-infinite acceleration, making the
# ODE numerically pathological -- confirmed this hangs Radau in practice).
# M_FLOAT here is deliberately larger (still light, not negligible) so this
# version actually integrates in a reasonable time when run.
# ---------------------------------------------------------------------------
# L, EI, RHO_A = 1.0, 12.0, 1.0   # lumped-EI convention (A=I=1)
# N_SEG = 10
# M_FLOAT = 0.05   # light but not pathologically tiny -- see note above

# joints = [[0, 1], [1, 2]]
# types = ['F', 'R']
# parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0]]
# joint_to_child_cg = [[np.nan, np.nan], [L / 2.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]] * 2
# flex_bd, m0, J0 = [np.nan, 2], [M_FLOAT, None], [1e-4, None]
# flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L]
# flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [RHO_A]
# flex_deformation_mode, flex_clamp_on = ["bending"], [True]

# n_total_dof = 3 + N_SEG   # float=3 DOF + N_SEG bending segments
# ic = np.zeros(2 * n_total_dof)
# OMEGA_ROOT, OMEGA_TIP = 0.3, 1.5   # rad/s, root -> tip velocity ramp
# ic[n_total_dof + 3: n_total_dof + 3 + N_SEG] = np.linspace(OMEGA_ROOT, OMEGA_TIP, N_SEG)
# # A UNIFORM velocity (same value for every segment) is a pure rigid-body
# # spin with ZERO relative velocity between segments -- excites no bending
# # at all (same lesson as EXAMPLE B). The ramp above ensures every joint
# # has a genuine nonzero relative velocity from t=0.

# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_float_bending.py")
# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on, output_path,
#     ic=ic, gVec=np.zeros(1 + N_SEG), tspan=1.0, TimeStep=0.01,
#     overwrite=True,
# )

# ---------------------------------------------------------------------------
# EXAMPLE B (ACTIVE): chained two-flexible-link double pendulum.
#   Row 1: link1 (flex, ground -> link1, PINNED -- free rotation at ground,
#          same as pendulum_FSM.py's already-validated single-link pattern.
#          HAS A CHILD (row 2), so its length is DERIVED, not given via flex_L)
#   Row 2: link2 (flex, link1 -> link2, PINNED -- free rotation relative to
#          link1's own tip orientation, free tip)
# No separate rigid "hub" bodies are needed: a pinned (clamp_on=False) mount
# ALREADY gives free rotation at that joint (k=0), exactly like the ground
# pivot in pendulum_FSM.py -- an extra hub body would only be needed to
# artificially separate "rigid swing" from "bending" as independent DOFs
# (useful for matching an external assumed-mode analytical model, NOT needed
# here), and their near-zero mass made the system numerically stiff to
# integrate for no benefit.
# Same physical parameters as Examples_flexible_FSM/comparison/
# flexible_double_pendulum_analytical_vs_m4e.py (L, "EI"/"rho" lumped the
# same way -- I=A=1 so flex_E directly carries E*I and flex_rho directly
# carries mass-per-unit-length).
# ---------------------------------------------------------------------------
# Tuned so the bending PERIOD (~2*pi/sqrt(k_int/J_seg)) is comparable to
# tspan below (~0.4s here, vs tspan=2.0s) -- otherwise the elastic response
# simply doesn't have TIME to develop within the simulated window, no matter
# how "floppy" EI looks on paper (a heavier/longer beam has a SLOWER bending
# response, not faster -- see chat explanation). flex_A=flex_I=1 here (the
# lumped-EI trick), so flex_E directly IS the E*I product -- don't change
# flex_I away from 1 without re-deriving flex_E, or "EI1" silently stops
# meaning what its name says.
# L1, L2 = 2.0, 2.0
# EI1, EI2 = 1e6, 1e6
# RHO1, RHO2 = 10000.0, 50000.0
# N_SEG1, N_SEG2 = 6, 6

# joints              = [[0, 1], [1, 2]]
# types               = ['R', 'R']
# parent_cg_to_joint  = [[0.0, 0.0], [L1 / 2.0, 0.0]]
# joint_to_child_cg   = [[L1 / 2.0, 0.0], [L2 / 2.0, 0.0]]
# prismatic_direction = [[np.nan, np.nan]] * 2
# flex_bd             = [1, 2]
# m0                  = [None, None]
# J0                  = [None, None]

# flex_bds              = [1, 2]
# flex_n_seg            = [N_SEG1, N_SEG2]
# flex_L                = [L1, L2]          # link1's is cross-checked against the derived value (has a child); link2's is required (free tip)
# flex_E                = [EI1, EI2]
# flex_A                = [1.0, 1.0]
# flex_I                = [1.0, 1.0]
# flex_rho              = [RHO1, RHO2]
# flex_deformation_mode = ["bending", "bending"]
# flex_clamp_on         = [False, False]    # pinned -- free rotation at both joints, matching pendulum_FSM.py's pattern

# output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_double_pendulum.py")

# # Released from rest position (straight, hanging down -- M4E's hanging-down
# # equilibrium is +pi/2, same convention as pendulum_FSM.py) but NOT at rest
# # velocity. IMPORTANT: a UNIFORM velocity (same value for every segment) is
# # a pure rigid-body spin with ZERO relative velocity between segments --
# # it excites no bending at all, exactly like a uniform initial ANGLE excites
# # no bending (same mistake, applied to velocity instead of position).
# # Use a velocity RAMP across the whole chain instead: root starts slow, tip
# # starts fast, so every joint has a genuine nonzero relative velocity from
# # t=0 -- this actually injects bending-mode kinetic energy.
# OMEGA_ROOT, OMEGA_TIP = 2.0, 20.0   # rad/s, root -> tip velocity ramp
# n_bodies = N_SEG1 + N_SEG2
# ic = np.zeros(2 * n_bodies)
# ic[:n_bodies] = np.pi / 2                                  # straight down, unbent
# ic[n_bodies:] = np.linspace(OMEGA_ROOT, OMEGA_TIP, n_bodies)  # velocity ramp root -> tip

# mb.generate_flex_model_file(
#     joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
#     flex_bd, m0, J0,
#     flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
#     flex_deformation_mode, flex_clamp_on,
#     output_path,
#     ic=ic, gVec=np.ones(n_bodies), tspan=2.0, TimeStep=0.005,
#     animation_on=1, SaveMovieOn=None, plotTstep=2,
#     integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
#     overwrite=True,
# )
# print(f"Generated model file: {output_path}")
# print("Open it to see the fully expanded bodies/joints/springs.")
# print("Run it via main.py (see this file's module docstring) to simulate/plot/animate it.")

# ---------------------------------------------------------------------------
# EXAMPLE M: VALIDATION -- clamped cantilever with TAPERED (per-segment,
# non-uniform) E and I, exercising Table 2's flex_E/flex_I "scalar-or-
# length-n_seg-list" broadcasting path (_broadcast in flex_expand.py) with
# an actual varying array for the first time (every earlier example used a
# uniform scalar). flex_A/flex_rho stay uniform scalars here -- isolating
# variable E and I specifically, as requested.
# Validated by hand-computing each TorsionSpring's stiffness directly from
# the same half-cell-compliance formula (k_root = 2*E0*I0/dx; internal
# k_i = 1/(dx/(2*E_{i-1}*I_{i-1}) + dx/(2*E_i*I_i))) using the SAME tapered
# E/I arrays, independent of flex_expand.py's own implementation, then
# comparing against the generated file's actual Force["TorsionSpring"]
# stiffness values.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

N_SEG = 6
L_BEAM = 1.0
dx = L_BEAM / N_SEG
E_vals = np.linspace(20.0, 8.0, N_SEG).tolist()   # stiffer at root, softer at tip
I_vals = np.linspace(2.0, 0.5, N_SEG).tolist()    # larger cross-section at root

joints = [[0, 1]]
types = ['R']
parent_cg_to_joint = [[0.0, 0.0]]
joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
prismatic_direction = [[np.nan, np.nan]]
flex_bd, m0, J0 = [1], [None], [None]
flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
flex_E, flex_A, flex_I, flex_rho = [E_vals], [1.0], [I_vals], [1.0]   # E, I: per-segment lists
flex_deformation_mode, flex_clamp_on = ["bending"], [True]   # clamped root, free tip

n_bodies = N_SEG
ic = np.zeros(2 * n_bodies)
ic[:n_bodies] = np.linspace(0.3 / N_SEG, 0.3, N_SEG)   # bent shape, released from rest

output_path = os.path.join(os.path.dirname(__file__), "Examples_flexible_FSM", "_generated_flex_tapered_cantilever.py")
mb.generate_flex_model_file(
    joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
    flex_bd, m0, J0,
    flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
    flex_deformation_mode, flex_clamp_on,
    output_path,
    ic=ic, gVec=np.zeros(n_bodies), tspan=2.0, TimeStep=0.005,
    integrator_kwargs={"algorithm": "Radau", "rtol": 1e-4, "atol": 1e-6},
    overwrite=True,
)

# --- independent hand-computed stiffness check (see docstring above) ---
k_expected = [2.0 * E_vals[0] * I_vals[0] / dx]
for i in range(1, N_SEG):
    compliance = dx / (2.0 * E_vals[i - 1] * I_vals[i - 1]) + dx / (2.0 * E_vals[i] * I_vals[i])
    k_expected.append(1.0 / compliance)

_spec = _ilu.spec_from_file_location("_generated_flex_tapered_cantilever", output_path)
_gen = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

_springs_sorted = sorted(_gen.Force["TorsionSpring"], key=lambda e: tuple(e[0]))
k_actual = [p[1] for _, p in _springs_sorted]
if np.allclose(k_actual, k_expected, rtol=1e-10):
    print(f"EXAMPLE M tapered-stiffness check: PASS ({N_SEG} springs match hand-computed values)")
else:
    print("EXAMPLE M tapered-stiffness check: FAIL")
    print("  expected:", k_expected)
    print("  actual:  ", k_actual)

# ---------------------------------------------------------------------------
# EXAMPLE N: 1 rigid hub body sandwiched between TWO flexible bending
# members -- link1 (uniform E, I) clamped to ground, then a rigid hub, then
# link2 (TAPERED: E AND I both vary per-segment) clamped to the hub. Not
# validated against any paper -- purely a Table 1/Table 2 structure demo for
# a multi-flex-body system where one member has non-uniform E/I. No file is
# generated and no simulation is run -- just the two tables, printed below.
# ---------------------------------------------------------------------------
L1, N1, E1, I1 = 0.6, 4, 70e9, 2e-6              # link1: uniform E, I
L2, N2 = 0.5, 5
E2_vals = np.linspace(200e9, 80e9, N2).tolist()  # link2: tapered E (stiffer at root)
I2_vals = np.linspace(5e-6, 1e-6, N2).tolist()   # link2: tapered I (larger cross-section at root)

# rows: 1=link1 (flex), 2=hub (rigid), 3=link2 (flex)
joints = [[0, 1], [1, 2], [2, 3]]
types = ['R', 'R', 'R']
parent_cg_to_joint = [[0.0, 0.0], [L1 / 2.0, 0.0], [0.0, 0.0]]
joint_to_child_cg = [[L1 / 2.0, 0.0], [0.0, 0.0], [L2 / 2.0, 0.0]]
prismatic_direction = [[np.nan, np.nan]] * 3
flex_bd = [1, np.nan, 3]
m0 = [None, 0.15, None]
J0 = [None, 0.0008, None]

flex_bds, flex_n_seg, flex_L = [1, 3], [N1, N2], [L1, L2]
flex_E, flex_A, flex_I, flex_rho = [E1, E2_vals], [1.0, 1.0], [I1, I2_vals], [2700.0, 2700.0]
flex_deformation_mode, flex_clamp_on = ["bending", "bending"], [True, True]

mb.tables.bodies_table(joints, types, parent_cg_to_joint, joint_to_child_cg,
                        prismatic_direction, m0, J0, flex_bd,
                        title="\n===== EXAMPLE N -- Table 1 (bodies/joints, pre-expansion) =====")
mb.tables.flex_properties_table(flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
                                 flex_deformation_mode, flex_clamp_on,
                                 title="\n===== EXAMPLE N -- Table 2 (flexible-member properties) =====")


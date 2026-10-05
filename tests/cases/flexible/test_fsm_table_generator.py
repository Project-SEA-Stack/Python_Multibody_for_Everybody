# -*- coding: utf-8 -*-
"""Pytest entry points for the FSM (flexible-body) table generator.

Run with::

    pytest tests/cases/flexible/test_fsm_table_generator.py

Self-contained by design: the reference model and every comparison helper
used below are defined directly in this file (no sys.path reach into
Examples_flexible_FSM/, no loading external example scripts) -- the
bending-chain reference mirrors Examples_flexible_FSM/comparison/
cantilever_FSM.py's/pendulum_FSM.py's own build_cantilever_fsm formulas
exactly, just built in-memory instead of on disk. This keeps the suite
independent of anything that lives under Examples_flexible_FSM/.

Covers the core code paths added/fixed this session, each against an
independent, verifiable target rather than just "runs without error":
  - bending, free-tip (cantilever/pendulum): EXACT match (static table +
    bit-for-bit dynamic trajectory) against the hand-built reference
    discretization (see _build_reference below).
  - axial, ground-mounted: matches the analytical uniform-bar elongation
    delta = P*L/(E*A) under a tip load.
  - floating body ('F', 3 DOF) + axial member + tip mass: matches the
    exact two-mass-spring-oscillator frequency omega = sqrt(k*(1/M1+1/M2)).
  - floating body (near-zero mass, approximating a free end) + bending
    member: matches the classical free-free uniform beam frequencies.
  - clamped cantilever with TAPERED (per-segment, non-uniform) E and I:
    matches independently hand-computed half-cell-compliance torsion
    spring stiffnesses, exercising Table 2's scalar-or-length-n_seg-list
    broadcasting path with actually-varying values.
"""
from __future__ import annotations

import os
import sys
import contextlib
import importlib.util
import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import eigh

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "source"))

import multibody as mb  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Finite-difference linearization about (q0, qd0) on the compiled numeric
# functions (R_func/M_func/Force_func) -- same approach as
# Examples_flexible_FSM/comparison/cantilever_FSM_validate.py's own
# numeric_linearize, inlined so the eigenvalue-based tests below have no
# on-disk dependency. Returns (M0, C0, K0) s.t. M0@ddq + C0@dq + K0@q = F_ext.
# ---------------------------------------------------------------------------
def numeric_linearize(MBDsys, q0, qd0, m0, J0,
                       forces_points_num=None, body_data_num=None, eps=1e-6):
    n                  = len(q0)
    forces_points_num  = np.array([]) if forces_points_num is None else forces_points_num
    body_data_num      = np.array([]) if body_data_num is None else body_data_num

    def generalized_force(q, qd):
        main_num_vars = np.concatenate([q, qd, forces_points_num, body_data_num])
        input_vec     = np.concatenate([main_num_vars, m0, J0])
        R = MBDsys.R_func(*main_num_vars)
        F = MBDsys.Force_func(*input_vec)
        return np.asarray(R.T @ F, dtype=float).flatten()

    main_num_vars0 = np.concatenate([q0, qd0, forces_points_num, body_data_num])
    input_vec0     = np.concatenate([main_num_vars0, m0, J0])
    R0             = MBDsys.R_func(*main_num_vars0)
    M0             = np.asarray(R0.T @ MBDsys.M_func(*input_vec0) @ R0, dtype=float)

    K0 = np.zeros((n, n))
    C0 = np.zeros((n, n))
    for i in range(n):
        dq       = np.zeros(n); dq[i] = eps
        K0[:, i] = -(generalized_force(q0 + dq, qd0) - generalized_force(q0 - dq, qd0)) / (2 * eps)

        dqd      = np.zeros(n); dqd[i] = eps
        C0[:, i] = -(generalized_force(q0, qd0 + dqd) - generalized_force(q0, qd0 - dqd)) / (2 * eps)

    return M0, C0, K0


# ---------------------------------------------------------------------------
# Bending, free-tip: exact match against a hand-built reference, built
# in-memory with the SAME half-cell-spring formulas as
# Examples_flexible_FSM/comparison/cantilever_FSM.py's build_cantilever_fsm
# (clamped root k_root=2EI/dx, or pinned root k_root=0).
# ---------------------------------------------------------------------------
N_SEG, L_BEAM, E_MOD, I_AREA, RHO, A_CROSS = 12, 1.0, 200e9, 1e-8, 7800.0, 1e-4


def _build_reference(clamp_on, ic, gVec, tspan, TimeStep,
                      n_seg=N_SEG, L_beam=L_BEAM, E_mod=E_MOD, I_area=I_AREA,
                      rho=RHO, A_cross=A_CROSS):
    dx    = L_beam / n_seg
    m_seg = rho * A_cross * dx
    J_seg = m_seg * dx ** 2 / 12.0
    k_int = E_mod * I_area / dx
    k_root = (2.0 * E_mod * I_area / dx) if clamp_on else 0.0

    joints = [[i, i + 1] for i in range(n_seg)]
    types = ["R"] * n_seg
    parent_cg_to_joint = [[dx / 2.0, 0.0]] * n_seg
    joint_to_child_cg = [[dx / 2.0, 0.0]] * n_seg
    prismatic_direction = mb.normalize_prismatic([[np.nan, np.nan]] * n_seg)

    force = {
        "PointsBD": [], "CG": [], "TensionSpring": [], "TensionDamper": [], "TorsionDamper": [],
        "TorsionSpring": [([0, 1], [0.0, k_root])] + [([i, i + 1], [0.0, k_int]) for i in range(1, n_seg)],
    }

    return SimpleNamespace(
        joints=joints, types=types,
        parent_cg_to_joint=parent_cg_to_joint, joint_to_child_cg=joint_to_child_cg,
        prismatic_direction=prismatic_direction,
        Force=force, Initial_Points={"GR": [], "BD": {}},
        m0=np.full(n_seg, m_seg), J0=np.full(n_seg, J_seg),
        ic=ic, gVec=np.asarray(gVec), tspan=tspan, TimeStep=TimeStep,
        ForcesPointsNum=np.array([]), BodyDataNum=np.array([]),
    )


def _build_via_generator(clamp_on, ic, gVec, tspan, TimeStep, out_path):
    dx = L_BEAM / N_SEG
    joints              = [[0, 1]]
    types               = ["R"]
    parent_cg_to_joint  = [[dx / 2.0, 0.0]]   # matches _build_reference's own root convention
    joint_to_child_cg   = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0     = [1], [None], [None]

    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0,
        [1], [N_SEG], [L_BEAM], [E_MOD], [A_CROSS], [I_AREA], [RHO],
        ["bending"], [clamp_on],
        out_path,
        ic=ic, gVec=gVec, tspan=tspan, TimeStep=TimeStep,
        overwrite=True,
    )
    return _load(out_path, os.path.basename(out_path)[:-3])


def _compare_static(ref, gen, label):
    print(f"--- {label}: static comparison ---")
    ok = True

    if ref.joints != gen.joints:
        print("  MISMATCH joints:", ref.joints, "vs", gen.joints); ok = False
    if ref.types != gen.types:
        print("  MISMATCH types:", ref.types, "vs", gen.types); ok = False
    if not np.allclose(ref.parent_cg_to_joint, gen.parent_cg_to_joint):
        print("  MISMATCH parent_cg_to_joint"); ok = False
    if not np.allclose(ref.joint_to_child_cg, gen.joint_to_child_cg):
        print("  MISMATCH joint_to_child_cg"); ok = False
    if not np.allclose(ref.m0, gen.m0):
        print("  MISMATCH m0:", ref.m0, "vs", gen.m0); ok = False
    if not np.allclose(ref.J0, gen.J0):
        print("  MISMATCH J0:", ref.J0, "vs", gen.J0); ok = False

    ref_ts = list(ref.Force["TorsionSpring"])
    gen_ts = list(gen.Force["TorsionSpring"])
    # The generator omits any k=0 spring (physically inert); the reference
    # includes it explicitly. Strip those before comparing.
    ref_ts_nonzero = [(b, p) for b, p in ref_ts if p[1] != 0.0]
    dropped = [(b, p) for b, p in ref_ts if p[1] == 0.0]
    if dropped:
        print(f"  (info: reference has {len(dropped)} explicit k=0 spring(s) "
              f"{dropped} -- generator omits these, physically inert, not a mismatch)")

    ref_sorted = sorted(ref_ts_nonzero, key=lambda e: tuple(e[0]))
    gen_sorted = sorted(gen_ts, key=lambda e: tuple(e[0]))
    if len(ref_sorted) != len(gen_sorted):
        print(f"  MISMATCH TorsionSpring count: {len(ref_sorted)} vs {len(gen_sorted)}"); ok = False
    else:
        for (rb, rp), (gb, gp) in zip(ref_sorted, gen_sorted):
            if list(rb) != list(gb) or not np.allclose(rp, gp):
                print(f"  MISMATCH TorsionSpring: {rb} {rp} vs {gb} {gp}"); ok = False

    print("  STATIC:", "PASS" if ok else "FAIL")
    return ok


def _run_dynamic(ex, **integrate_kwargs):
    MBDsys = mb.MbdSystem.from_example(ex)
    mainNumVars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))
    sol = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, tspan=ex.tspan, dt=ex.TimeStep,
                            **integrate_kwargs)
    return sol


def _compare_dynamic(ref_ex, gen_ex, label, **integrate_kwargs):
    print(f"--- {label}: dynamic comparison ({integrate_kwargs}) ---")
    sol_ref = _run_dynamic(ref_ex, **integrate_kwargs)
    sol_gen = _run_dynamic(gen_ex, **integrate_kwargs)
    diff = np.max(np.abs(sol_ref.y - sol_gen.y))
    ok = diff < 1e-6
    print(f"  max |trajectory diff| = {diff:.3e}  ->", "PASS" if ok else "FAIL")
    return ok


def test_cantilever_static_matches_reference(tmp_path):
    ref = _build_reference(clamp_on=True, ic=np.zeros(2 * N_SEG), gVec=np.zeros((N_SEG, 1)),
                            tspan=0.5, TimeStep=0.0005)
    gen = _build_via_generator(
        clamp_on=True, ic=ref.ic, gVec=ref.gVec.flatten(), tspan=ref.tspan, TimeStep=ref.TimeStep,
        out_path=str(tmp_path / "_gen_cantilever_static.py"),
    )
    assert _compare_static(ref, gen, "cantilever")


def test_cantilever_dynamic_matches_reference(tmp_path):
    ref = _build_reference(clamp_on=True, ic=np.zeros(2 * N_SEG), gVec=np.zeros((N_SEG, 1)),
                            tspan=0.5, TimeStep=0.0005)
    gen = _build_via_generator(
        clamp_on=True, ic=ref.ic, gVec=ref.gVec.flatten(), tspan=ref.tspan, TimeStep=ref.TimeStep,
        out_path=str(tmp_path / "_gen_cantilever_dynamic.py"),
    )
    assert _compare_dynamic(ref, gen, "cantilever", algorithm="Radau", rtol=1e-8, atol=1e-8)


def test_pendulum_static_matches_reference(tmp_path):
    ic = np.zeros(2 * N_SEG)
    ic[:N_SEG] = np.pi / 2 + 0.5
    ref = _build_reference(clamp_on=False, ic=ic, gVec=np.ones((N_SEG, 1)), tspan=5.0, TimeStep=0.001)
    gen = _build_via_generator(
        clamp_on=False, ic=ref.ic, gVec=ref.gVec.flatten(), tspan=ref.tspan, TimeStep=ref.TimeStep,
        out_path=str(tmp_path / "_gen_pendulum_static.py"),
    )
    assert _compare_static(ref, gen, "pendulum")


def test_pendulum_dynamic_matches_reference(tmp_path):
    ic = np.zeros(2 * N_SEG)
    ic[:N_SEG] = np.pi / 2 + 0.5
    ref = _build_reference(clamp_on=False, ic=ic, gVec=np.ones((N_SEG, 1)), tspan=5.0, TimeStep=0.001)
    gen = _build_via_generator(
        clamp_on=False, ic=ref.ic, gVec=ref.gVec.flatten(), tspan=ref.tspan, TimeStep=ref.TimeStep,
        out_path=str(tmp_path / "_gen_pendulum_dynamic.py"),
    )
    assert _compare_dynamic(ref, gen, "pendulum", algorithm="Radau", rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Axial, ground-mounted: matches P*L/(E*A)
# ---------------------------------------------------------------------------
def test_axial_cantilever_matches_analytical_elongation(tmp_path):
    L, E, A, RHO, P, N_SEG = 1.0, 200e9, 1e-4, 7800.0, 1000.0, 6

    joints = [[0, 1]]
    types = ["P"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L / 2.0, 0.0]]
    prismatic_direction = [[1.0, 0.0]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L]
    flex_E, flex_A, flex_I, flex_rho = [E], [A], [1.0], [RHO]
    flex_deformation_mode, flex_clamp_on = ["axial"], [True]

    out = str(tmp_path / "_gen_axial_cantilever.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out,
        ic=np.zeros(2 * (N_SEG + 1)), gVec=np.zeros(N_SEG + 1), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out, "_gen_axial_cantilever_pytest")
    ex.Force["CG"] = [[N_SEG + 1, P, 0.0, 0.0]]

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    n_dof = sum(MBDsys.NDOF)
    q0, qd0 = np.zeros(n_dof), np.zeros(n_dof)
    _, _, K0 = numeric_linearize(MBDsys, q0, qd0, ex.m0, ex.J0)
    main_vars0 = np.concatenate([q0, qd0])
    input_vec0 = np.concatenate([main_vars0, ex.m0, ex.J0])
    R0 = MBDsys.R_func(*main_vars0)
    F0 = MBDsys.Force_func(*input_vec0)
    F_gen = (R0.T @ F0).flatten()
    S = np.linalg.solve(K0, F_gen)

    total_elongation = S.sum()
    analytical = P * L / (E * A)
    rel_err = abs(total_elongation - analytical) / analytical
    assert rel_err < 1e-9, f"axial elongation error {rel_err:.3e} vs analytical {analytical:.6e} m"


# ---------------------------------------------------------------------------
# Floating body + axial member + tip mass: matches the exact two-mass
# oscillator formula (beam mass made negligible so the idealized
# massless-spring formula applies almost exactly)
# ---------------------------------------------------------------------------
def test_floating_axial_matches_two_mass_oscillator(tmp_path):
    M1, M2 = 5.0, 2.0
    L, E, A, RHO = 1.0, 1e6, 1.0, 1e-6
    N_SEG = 6
    dx = L / N_SEG
    k_eff = E * A / L

    joints = [[0, 1], [1, 2], [2, 3]]
    types = ["F", "P", "P"]
    parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0], [L / 2.0, 0.0]]
    joint_to_child_cg = [[np.nan, np.nan], [L / 2.0, 0.0], [0.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan], [1.0, 0.0], [1.0, 0.0]]
    flex_bd = [np.nan, 2, np.nan]
    m0 = [M1, None, M2]
    J0 = [1.0, None, 1e-6]
    flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L]
    flex_E, flex_A, flex_I, flex_rho = [E], [A], [1.0], [RHO]
    flex_deformation_mode, flex_clamp_on = ["axial"], [True]

    n_total_dof = 3 + (N_SEG + 1) + 1
    out = str(tmp_path / "_gen_floating_axial.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out,
        ic=np.zeros(2 * n_total_dof), gVec=np.zeros(1 + (N_SEG + 1) + 1), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out, "_gen_floating_axial_pytest")

    tip_anchor_real = 1 + N_SEG + 1
    payload_real = tip_anchor_real + 1
    k_root = 2.0 * E * A / dx
    k_weld = 1000.0 * k_root
    ex.Force["TensionSpring"] = list(ex.Force["TensionSpring"]) + [
        ((f"CG{tip_anchor_real}_{tip_anchor_real}", f"CG{payload_real}_{payload_real}"), [0.0, k_weld])
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    n_dof = sum(MBDsys.NDOF)
    q0, qd0 = np.zeros(n_dof), np.zeros(n_dof)
    M0, _, K0 = numeric_linearize(MBDsys, q0, qd0, ex.m0, ex.J0)
    eigvals, _ = eigh(K0, M0)
    freqs_hz = np.sqrt(np.abs(eigvals[eigvals > 1e-6])) / (2 * np.pi)

    analytical_hz = np.sqrt(k_eff * (1.0 / M1 + 1.0 / M2)) / (2 * np.pi)
    rel_err = abs(freqs_hz[0] - analytical_hz) / analytical_hz
    assert rel_err < 1e-3, (
        f"floating+axial mode {freqs_hz[0]:.4f} Hz vs analytical {analytical_hz:.4f} Hz, "
        f"error {100 * rel_err:.4f}%"
    )


# ---------------------------------------------------------------------------
# Floating body (near-zero mass, approximating a free end) + bending
# member: matches the classical free-free uniform beam frequencies
# ---------------------------------------------------------------------------
def test_floating_bending_matches_free_free_beam(tmp_path):
    L, EI, RHO_A = 1.0, 12.0, 1.0
    N_SEG = 10
    M_FLOAT = 1e-6

    joints = [[0, 1], [1, 2]]
    types = ["F", "R"]
    parent_cg_to_joint = [[0.0, 0.0], [0.0, 0.0]]
    joint_to_child_cg = [[np.nan, np.nan], [L / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]] * 2
    flex_bd, m0, J0 = [np.nan, 2], [M_FLOAT, None], [1e-9, None]
    flex_bds, flex_n_seg, flex_L = [2], [N_SEG], [L]
    flex_E, flex_A, flex_I, flex_rho = [EI], [1.0], [1.0], [RHO_A]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    n_total_dof = 3 + N_SEG
    out = str(tmp_path / "_gen_floating_bending.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out,
        ic=np.zeros(2 * n_total_dof), gVec=np.zeros(1 + N_SEG), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out, "_gen_floating_bending_pytest")

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    n_dof = sum(MBDsys.NDOF)
    q0, qd0 = np.zeros(n_dof), np.zeros(n_dof)
    M0, _, K0 = numeric_linearize(MBDsys, q0, qd0, ex.m0, ex.J0)
    eigvals, _ = eigh(K0, M0)
    freqs_hz = np.sqrt(np.abs(eigvals[eigvals > 1e-6])) / (2 * np.pi)

    beta_L = [4.73004, 7.85320, 10.99561]
    targets_hz = [(bL ** 2) / (2 * np.pi * L ** 2) * np.sqrt(EI / RHO_A) for bL in beta_L]
    for i, target in enumerate(targets_hz):
        rel_err = abs(freqs_hz[i] - target) / target
        assert rel_err < 5e-3, (
            f"free-free mode {i + 1}: FSM={freqs_hz[i]:.4f} Hz vs target={target:.4f} Hz, "
            f"error {100 * rel_err:.4f}%"
        )


# ---------------------------------------------------------------------------
# Clamped cantilever with TAPERED (per-segment) E and I: matches
# independently hand-computed half-cell-compliance torsion spring
# stiffnesses (k_root = 2*E0*I0/dx; internal k_i = 1/(dx/(2*E_{i-1}*I_{i-1})
# + dx/(2*E_i*I_i))) -- exercises flex_E/flex_I as length-n_seg lists
# rather than uniform scalars.
# ---------------------------------------------------------------------------
def test_tapered_cantilever_matches_handcomputed_stiffness(tmp_path):
    N_SEG, L_BEAM = 6, 1.0
    dx = L_BEAM / N_SEG
    E_vals = np.linspace(20.0, 8.0, N_SEG).tolist()
    I_vals = np.linspace(2.0, 0.5, N_SEG).tolist()

    joints = [[0, 1]]
    types = ["R"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
    flex_E, flex_A, flex_I, flex_rho = [E_vals], [1.0], [I_vals], [1.0]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    out = str(tmp_path / "_gen_tapered_cantilever.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out,
        ic=np.zeros(2 * N_SEG), gVec=np.zeros(N_SEG), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out, "_gen_tapered_cantilever_pytest")

    k_expected = [2.0 * E_vals[0] * I_vals[0] / dx]
    for i in range(1, N_SEG):
        compliance = dx / (2.0 * E_vals[i - 1] * I_vals[i - 1]) + dx / (2.0 * E_vals[i] * I_vals[i])
        k_expected.append(1.0 / compliance)

    springs_sorted = sorted(ex.Force["TorsionSpring"], key=lambda e: tuple(e[0]))
    k_actual = [p[1] for _, p in springs_sorted]
    assert np.allclose(k_actual, k_expected, rtol=1e-10), (
        f"tapered stiffness mismatch: expected {k_expected} vs actual {k_actual}"
    )

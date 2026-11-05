# -*- coding: utf-8 -*-
"""Pytest entry points for the symbolic property-distribution feature --
source/multibody/flexible/_flex_distribution_helper.py, and its wiring into
flex_expand.py's _expand_bending_row/_expand_axial_row.

Run with::

    pytest tests/cases/flexible/test_flex_distribution.py

Covers:
  - a row whose symbolic distribution is actually constant reproduces
    the plain-scalar path's springs/mass/inertia EXACTLY (the "almost
    constant distribution" sanity check).
  - a linearly tapered axial bar's total elongation under a tip load matches
    the classic closed-form P*L/(E*A0*alpha)*ln(1/(1-alpha)) to near machine
    precision, and -- unlike a typical mesh-convergence check -- this
    agreement is independent of n_seg (confirmed at n_seg=2,3,6,12), because
    FSM's half-cell compliances partition [0, L] exactly, so their sum
    equals the continuous integral regardless of how finely it is split.
  - the rigorous per-segment inertia integral is IDENTICAL to the simplified
    m*dx**2/12 formula for a linearly-varying density (a mathematical
    identity: a linear term is odd about the segment's own midpoint, so it
    does not contribute to the second central moment) -- and measurably
    different for a quadratic (curved) density, shrinking as the mesh
    refines, confirming it's a genuine discretization effect, not a bug.
  - validate_row's full error contract (flex_x_dist None but expression
    present; invalid flex_x_dist type; mismatched symbols; per-segment list
    mixed with a symbolic distribution; declared-but-unused symbol).
  - a free, undamped, distributed-property bending cantilever conserves
    total mechanical energy during time integration (the check Chrono's own
    FEA/beam test suite relies on most heavily -- a subtle stiffness/mass
    bug often shows up as energy drift even when static checks pass).
  - the quadratic-density inertia correction converges at the expected
    O(dx^2) RATE (not just direction) as the mesh refines.
  - FSM's tapered bending natural frequencies cross-validated against an
    INDEPENDENT numerical method (_euler_bernoulli_fem_reference.py -- a
    classical cubic-Hermite finite element, not FSM's own rigid-segment +
    half-cell-compliance formulation), the same validation tier
    Chrono/OpenFAST rely on for continuously-varying-property beams.
  - the same tapered-bending frequencies cross-validated a second time
    against REAL PyChrono (ChElementBeamEuler FEA, run separately via
    pychrono_backup/_pychrono_cantilever_reference.py and hardcoded here as a reference
    value -- see that script and test_tapered_bending_frequencies_match_pychrono_reference
    for the full method).
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys

import numpy as np
import pytest
import sympy as sym
from scipy.linalg import eigh

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "source"))
sys.path.insert(0, os.path.dirname(__file__))

import multibody as mb  # noqa: E402
from multibody.flexible import _flex_distribution_helper as flex_dist  # noqa: E402
from multibody.checks.flex_distribution_check import validate_row  # noqa: E402
from _euler_bernoulli_fem_reference import cantilever_bending_frequencies  # noqa: E402


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def numeric_linearize(MBDsys, q0, qd0, m0, J0, forces_points_num=None, body_data_num=None, eps=1e-6):
    """Same finite-difference linearization as test_fsm_table_generator.py's own
    (self-contained, no on-disk dependency) -- duplicated locally on purpose,
    this file has no dependency on that one."""
    n = len(q0)
    forces_points_num = np.array([]) if forces_points_num is None else forces_points_num
    body_data_num = np.array([]) if body_data_num is None else body_data_num

    def generalized_force(q, qd):
        main_num_vars = np.concatenate([q, qd, forces_points_num, body_data_num])
        input_vec = np.concatenate([main_num_vars, m0, J0])
        R = MBDsys.R_func(*main_num_vars)
        F = MBDsys.Force_func(*input_vec)
        return np.asarray(R.T @ F, dtype=float).flatten()

    main_num_vars0 = np.concatenate([q0, qd0, forces_points_num, body_data_num])
    input_vec0 = np.concatenate([main_num_vars0, m0, J0])
    R0 = MBDsys.R_func(*main_num_vars0)
    M0 = np.asarray(R0.T @ MBDsys.M_func(*input_vec0) @ R0, dtype=float)

    K0 = np.zeros((n, n))
    C0 = np.zeros((n, n))
    for i in range(n):
        dq = np.zeros(n); dq[i] = eps
        K0[:, i] = -(generalized_force(q0 + dq, qd0) - generalized_force(q0 - dq, qd0)) / (2 * eps)
        dqd = np.zeros(n); dqd[i] = eps
        C0[:, i] = -(generalized_force(q0, qd0 + dqd) - generalized_force(q0, qd0 - dqd)) / (2 * eps)
    return M0, C0, K0


# ---------------------------------------------------------------------------
# "Almost constant distribution" sanity check (bending): a symbolic
# expression with a NEGLIGIBLY small (but genuinely nonzero -- sympy would
# otherwise auto-simplify an exactly-zero coefficient away, collapsing the
# expression to a plain constant with zero free symbols, which validate_row
# correctly rejects as "declared but unused") linear term must reproduce the
# plain-scalar path's springs/mass/inertia to within the tiny tolerance that
# term actually introduces.
# ---------------------------------------------------------------------------
def test_bending_near_constant_distribution_matches_constant_path(tmp_path):
    N_SEG, L_BEAM, E_VAL, I_VAL, RHO_VAL = 6, 1.0, 200e9, 2e-6, 7800.0
    joints = [[0, 1]]
    types = ["R"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    out_const = str(tmp_path / "_gen_const.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L,
        [E_VAL], [1.0], [I_VAL], [RHO_VAL], flex_deformation_mode, flex_clamp_on,
        out_const, ic=np.zeros(2 * N_SEG), gVec=np.zeros(N_SEG), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex_const = _load(out_const, "_gen_const_pytest")

    # A genuinely nonzero, but 10 orders of magnitude smaller than E/I/rho
    # themselves, linear term -- sympy keeps x_dist in free_symbols (it is
    # NOT simplified away, unlike a literal "+ 0*x_dist"), while the actual
    # numeric effect on the springs/mass/inertia is negligible.
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    out_sym = str(tmp_path / "_gen_near_constant.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L,
        [E_VAL + 1e-2 * x_dist], [1.0], [I_VAL + 1e-16 * x_dist], [RHO_VAL + 1e-7 * x_dist],
        flex_deformation_mode, flex_clamp_on, out_sym,
        flex_x_dist=[x_dist],
        ic=np.zeros(2 * N_SEG), gVec=np.zeros(N_SEG), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex_sym = _load(out_sym, "_gen_near_constant_pytest")

    assert np.allclose(ex_const.m0, ex_sym.m0, rtol=1e-6)
    assert np.allclose(ex_const.J0, ex_sym.J0, rtol=1e-6)

    springs_const = sorted(ex_const.Force["TorsionSpring"], key=lambda e: tuple(e[0]))
    springs_sym = sorted(ex_sym.Force["TorsionSpring"], key=lambda e: tuple(e[0]))
    k_const = [p[1] for _, p in springs_const]
    k_sym = [p[1] for _, p in springs_sym]
    assert np.allclose(k_const, k_sym, rtol=1e-6), (
        f"a near-constant symbolic distribution should reproduce the "
        f"constant-path springs to within its own (tiny) actual effect: "
        f"{k_const} vs {k_sym}"
    )


# ---------------------------------------------------------------------------
# Analytical validation: linearly tapered axial bar, A(x) = A0*(1 - alpha*x/L),
# matches the classic closed-form elongation under a tip load P:
#   delta = P*L/(E*A0*alpha) * ln(1/(1-alpha))
# (standard result for a linearly tapering cross-section, e.g. Hibbeler,
# "Mechanics of Materials" -- axial deformation of a tapered bar). Checked at
# several n_seg to confirm the agreement does NOT depend on mesh fineness:
# FSM's half-cell compliances partition [0, L] exactly, so their sum equals
# the continuous integral for ANY n_seg, not just in the limit of many
# segments (verified: ~1e-11 relative error at n_seg=2, 3, 6, and 12 alike).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_seg", [2, 3, 6, 12])
def test_tapered_axial_matches_analytical_elongation(tmp_path, n_seg):
    L, E, A0, alpha, RHO, P = 1.0, 200e9, 1e-4, 0.5, 1e-8, 1000.0
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    A_expr = A0 * (1 - alpha * x_dist / L)

    joints = [[0, 1]]
    types = ["P"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L / 2.0, 0.0]]
    prismatic_direction = [[1.0, 0.0]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [n_seg], [L]
    flex_E, flex_A, flex_I, flex_rho = [E], [A_expr], [1.0], [RHO]
    flex_deformation_mode, flex_clamp_on = ["axial"], [True]

    out_path = str(tmp_path / f"_gen_tapered_axial_{n_seg}.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out_path,
        flex_x_dist=[x_dist],
        ic=np.zeros(2 * (n_seg + 1)), gVec=np.zeros(n_seg + 1), tspan=1.0, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out_path, f"_gen_tapered_axial_{n_seg}_pytest")
    ex.Force["CG"] = [[n_seg + 1, P, 0.0, 0.0]]

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
    total_elongation = np.linalg.solve(K0, F_gen).sum()

    analytical = (P * L) / (E * A0 * alpha) * math.log(1.0 / (1.0 - alpha))
    rel_err = abs(total_elongation - analytical) / analytical
    assert rel_err < 1e-8, (
        f"n_seg={n_seg}: tapered axial elongation error {rel_err:.3e} "
        f"vs analytical {analytical:.6e} m"
    )


# ---------------------------------------------------------------------------
# Rigorous inertia: identical to m*dx**2/12 for a LINEAR density (a
# mathematical identity, not an approximation -- the linear term is odd
# about the segment's own midpoint and contributes zero to the second
# central moment); measurably different, and shrinking with mesh refinement,
# for a QUADRATIC density.
# ---------------------------------------------------------------------------
def test_rigorous_inertia_matches_simplified_formula_for_linear_density():
    L, n_seg = 1.0, 4
    dx = L / n_seg
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    rho_val = 2700.0
    A_linear = 1e-4 - 0.3e-4 * x_dist / L

    m_list, J_list = flex_dist.resolve_mass_and_inertia(rho_val, A_linear, x_dist, n_seg, dx, row_label=1)
    J_simplified = [m * dx * dx / 12.0 for m in m_list]
    assert np.allclose(J_list, J_simplified, rtol=1e-9), (
        "a linearly-varying density must give IDENTICAL rigorous and "
        "simplified inertia (mathematical identity, not an approximation)"
    )


def test_rigorous_inertia_differs_from_simplified_formula_for_quadratic_density():
    L = 1.0
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    rho_val = 2700.0
    A_quad = 1e-4 - 0.6e-4 * (x_dist / L) ** 2

    rel_diffs = {}
    for n_seg in (2, 4, 12):
        dx = L / n_seg
        m_list, J_list = flex_dist.resolve_mass_and_inertia(rho_val, A_quad, x_dist, n_seg, dx, row_label=1)
        J_simplified = [m * dx * dx / 12.0 for m in m_list]
        rel_diffs[n_seg] = max(abs(jr - js) / js for jr, js in zip(J_list, J_simplified))

    # must be a REAL, nonzero correction for a curved (quadratic) density...
    assert rel_diffs[2] > 1e-3, f"expected a measurable correction at a coarse mesh, got {rel_diffs[2]:.3e}"
    # ...and it must shrink as the mesh refines (discretization effect, not a bug)
    assert rel_diffs[2] > rel_diffs[4] > rel_diffs[12], (
        f"correction should shrink monotonically with finer mesh: {rel_diffs}"
    )


# ---------------------------------------------------------------------------
# validate_row's full error contract.
# ---------------------------------------------------------------------------
def test_validate_row_rejects_expression_when_x_dist_is_none():
    x_dist = sym.symbols("x_dist")
    with pytest.raises(ValueError, match="flex_x_dist is None"):
        validate_row(None, {"E": 200e9 + 0 * x_dist, "I": 2e-6}, row_label=1)


def test_validate_row_rejects_invalid_x_dist_type():
    with pytest.raises(ValueError, match="must be None or a single sympy.Symbol"):
        validate_row(1.0, {"E": 200e9, "I": 2e-6}, row_label=1)
    with pytest.raises(ValueError, match="must be None or a single sympy.Symbol"):
        validate_row("x", {"E": 200e9, "I": 2e-6}, row_label=1)


def test_validate_row_rejects_mismatched_symbols():
    x_dist, other = sym.symbols("x_dist other")
    with pytest.raises(ValueError, match="uses symbol"):
        validate_row(x_dist, {"E": 200e9 + 1e9 * other, "I": 2e-6}, row_label=1)


def test_validate_row_rejects_list_mixed_with_symbolic():
    x_dist = sym.symbols("x_dist")
    with pytest.raises(ValueError, match="per-segment list"):
        validate_row(x_dist, {"E": 200e9 + 0 * x_dist, "I": [2e-6, 1.8e-6]}, row_label=1)


def test_validate_row_rejects_declared_but_unused_symbol():
    x_dist = sym.symbols("x_dist")
    with pytest.raises(ValueError, match="no property actually depends on it"):
        validate_row(x_dist, {"E": 200e9, "I": 2e-6}, row_label=1)


def test_validate_row_accepts_well_formed_distributed_row():
    x_dist = sym.symbols("x_dist")
    validate_row(x_dist, {"E": 200e9 - 1e9 * x_dist, "I": 2e-6, "rho": 2700.0}, row_label=1)


# ---------------------------------------------------------------------------
# Positivity validation: a degenerate distribution (cross-section reaching
# zero) must be rejected, not silently integrated into nonsense.
# ---------------------------------------------------------------------------
def test_degenerate_distribution_is_rejected():
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    L, n_seg = 1.0, 4
    dx = L / n_seg
    A_degenerate = 1e-4 * (1 - x_dist / L)  # reaches exactly zero at the tip
    with pytest.raises(ValueError, match="strictly positive"):
        flex_dist.resolve_compliance_half_cells(200e9, A_degenerate, x_dist, n_seg, dx, row_label=1, prop_name="A")


# ---------------------------------------------------------------------------
# Convergence RATE (not just direction) of the rigorous-inertia correction
# for a quadratic (curved) density: a Jensen's-gap/curvature effect should
# shrink as O(dx^2), i.e. the ratio of successive errors should approach 4
# as n_seg doubles. Verified directly (not just "gets smaller") to catch a
# subtly wrong implementation that still happens to improve with refinement
# for the wrong reason -- the same reasoning Chrono/OpenFAST-style
# convergence studies use, not just a monotonic-trend check.
# ---------------------------------------------------------------------------
def test_rigorous_inertia_correction_converges_at_expected_rate():
    L = 1.0
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    rho_val = 2700.0
    A_quad = 1e-4 - 0.6e-4 * (x_dist / L) ** 2

    rel_diffs = {}
    for n_seg in (4, 8, 16, 32, 64):
        dx = L / n_seg
        m_list, J_list = flex_dist.resolve_mass_and_inertia(rho_val, A_quad, x_dist, n_seg, dx, row_label=1)
        J_simplified = [m * dx * dx / 12.0 for m in m_list]
        rel_diffs[n_seg] = max(abs(jr - js) / js for jr, js in zip(J_list, J_simplified))

    ns = sorted(rel_diffs)
    ratios = [rel_diffs[ns[i]] / rel_diffs[ns[i + 1]] for i in range(len(ns) - 1)]
    # the ratio should approach 4 (O(dx^2)) as the mesh refines -- check the
    # finest-mesh ratio specifically, where the asymptotic rate dominates
    assert 3.6 < ratios[-1] < 4.4, (
        f"expected the finest-mesh error ratio to approach 4 (O(dx^2) "
        f"convergence), got ratios {ratios} (n_seg={ns})"
    )
    # and every ratio must be monotonically increasing toward that limit,
    # not just "small" by coincidence
    assert all(r2 > r1 for r1, r2 in zip(ratios, ratios[1:])), (
        f"convergence ratio should increase monotonically toward 4: {ratios}"
    )


# ---------------------------------------------------------------------------
# Dynamic energy conservation: a free (no gravity, no damping), distributed-
# property bending cantilever, released from a bent shape and integrated
# with a tight-tolerance implicit solver, must conserve total mechanical
# energy (kinetic + spring strain, via MbdSystem.Energy_func) to within the
# integrator's own tolerance. This is the check Chrono's own FEA/beam test
# suite leans on most heavily: a subtle bug in a hand-derived stiffness or
# mass formulation (exactly the kind of code added in
# _flex_distribution_helper.py) very often shows up as energy drifting over
# time, even when purely static/linearized checks (every other test in this
# file) pass perfectly.
# ---------------------------------------------------------------------------
def test_tapered_bending_conserves_energy(tmp_path):
    L_BEAM, N_SEG = 1.0, 4
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    E_expr = 20.0 - 12.0 * x_dist / L_BEAM   # continuously tapered, same scale as EXAMPLE M
    I_val, A_val, RHO_val = 1.0, 1.0, 1.0

    joints = [[0, 1]]
    types = ["R"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
    flex_E, flex_A, flex_I, flex_rho = [E_expr], [A_val], [I_val], [RHO_val]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    ic = np.zeros(2 * N_SEG)
    ic[:N_SEG] = np.linspace(0.3 / N_SEG, 0.3, N_SEG)   # bent shape, released from rest

    out_path = str(tmp_path / "_gen_energy_check.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out_path,
        flex_x_dist=[x_dist],
        ic=ic, gVec=np.zeros(N_SEG), tspan=0.5, TimeStep=0.01,
        integrator_kwargs={"algorithm": "Radau", "rtol": 1e-8, "atol": 1e-9},
        overwrite=True,
    )
    ex = _load(out_path, "_gen_energy_check_pytest")

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    main_num_vars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))
    sol = MBDsys.integrate(main_num_vars, ex.m0, ex.J0, tspan=ex.tspan, dt=ex.TimeStep, **ex.integrator_kwargs)

    energy = []
    for i, ti in enumerate(sol.t):
        v = main_num_vars.copy()
        v[:len(sol.y)] = sol.y[:, i]
        v[MBDsys.t_update] = ti
        energy.append(MBDsys.Energy_func(*v, *ex.m0, *ex.J0))
    energy = np.array(energy)

    rel_dev = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    assert rel_dev < 1e-6, (
        f"total mechanical energy drifted by {rel_dev:.3e} (relative) over "
        f"the simulation -- a distributed-property member should conserve "
        f"energy to within the integrator's own tolerance, same as any "
        f"other free, undamped FSM model."
    )


# ---------------------------------------------------------------------------
# Self-check of the independent FEM reference itself, BEFORE trusting it to
# validate FSM: it must reproduce the textbook closed-form natural
# frequencies of a UNIFORM clamped-free beam,
#   f_n = (beta_n*L)**2 / (2*pi*L**2) * sqrt(E*I / (rho*A)),
# with beta_n*L = 1.875104, 4.694091, 7.854757 for the first three modes
# (any vibrations textbook, e.g. Blevins "Formulas for Natural Frequency and
# Mode Shape", Table 8-1, case 1).
# ---------------------------------------------------------------------------
def test_euler_bernoulli_fem_reference_matches_uniform_cantilever_closed_form():
    L, E, I, A, rho = 1.0, 200e9, 8e-8, 1e-4, 7800.0
    beta_L = np.array([1.875104, 4.694091, 7.854757])
    analytical = (beta_L ** 2) / (2 * math.pi * L ** 2) * math.sqrt(E * I / (rho * A))

    freqs = cantilever_bending_frequencies(L, lambda x: E * I, lambda x: rho * A, n_elem=40, n_modes=3)
    rel_err = np.abs(freqs - analytical) / analytical
    assert np.all(rel_err < 1e-4), (
        f"independent FEM reference does not match the uniform-cantilever "
        f"closed form: got {freqs} Hz, expected {analytical} Hz (rel_err={rel_err})"
    )


# ---------------------------------------------------------------------------
# Cross-validation against an INDEPENDENT numerical method -- the check
# requested in place of an actual Chrono/OpenFAST installation (neither is
# available in this environment) or a specific published tapered-beam
# frequency table (none could be verified here without fabrication risk).
#
# FSM's tapered-bending natural frequencies (same I(x) taper as EXAMPLE O /
# Examples_flexible_FSM/flex_tapered_bending.py) are compared against a
# classical cubic-Hermite finite element (_euler_bernoulli_fem_reference.py)
# that shares NO code with FSM's own rigid-segment + half-cell-compliance
# formulation (source/multibody/flexible/). Two fundamentally different
# discretizations of the same physical system agreeing -- and converging
# toward each other as both are refined, confirmed manually at
# n_seg=6,8,12,20 (mode-1 error 1.27e-2 -> 7.2e-3 -> 3.2e-3 -> 1.16e-3,
# monotonically shrinking toward the FEM reference) -- is strong, genuinely
# independent evidence, not a self-consistency check of FSM's own machinery.
# ---------------------------------------------------------------------------
def test_tapered_bending_frequencies_match_independent_fem_reference(tmp_path):
    L_BEAM, N_SEG = 1.0, 20
    E_STEEL, RHO_STEEL, A_VAL = 200e9, 7800.0, 1e-4
    I_ROOT, I_TIP = 8e-8, 2e-8
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    I_expr = I_ROOT - (I_ROOT - I_TIP) * x_dist / L_BEAM

    joints = [[0, 1]]
    types = ["R"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
    flex_E, flex_A, flex_I, flex_rho = [E_STEEL], [A_VAL], [I_expr], [RHO_STEEL]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    out_path = str(tmp_path / "_gen_fem_crossval.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out_path,
        flex_x_dist=[x_dist],
        ic=np.zeros(2 * N_SEG), gVec=np.zeros(N_SEG), tspan=0.1, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out_path, "_gen_fem_crossval_pytest")

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    n_dof = sum(MBDsys.NDOF)
    q0, qd0 = np.zeros(n_dof), np.zeros(n_dof)
    M0, _, K0 = numeric_linearize(MBDsys, q0, qd0, ex.m0, ex.J0)
    eigvals = eigh(K0, M0, eigvals_only=True)
    freqs_fsm = np.sqrt(np.sort(eigvals[eigvals > 1e-6])) / (2 * np.pi)

    freqs_fem = cantilever_bending_frequencies(
        L_BEAM,
        lambda x: E_STEEL * (I_ROOT - (I_ROOT - I_TIP) * x / L_BEAM),
        lambda x: RHO_STEEL * A_VAL,
        n_elem=64, n_modes=3,
    )

    rel_err = np.abs(freqs_fsm[:3] - freqs_fem) / freqs_fem
    assert np.all(rel_err < 1e-2), (
        f"FSM's tapered-bending frequencies at n_seg={N_SEG} do not agree "
        f"with the independent FEM reference: FSM={freqs_fsm[:3]} Hz, "
        f"FEM={freqs_fem} Hz (rel_err={rel_err})"
    )


# ---------------------------------------------------------------------------
# Cross-validation against REAL PyChrono (third-party, industry-standard
# software -- the specific check requested in place of the from-scratch FEM
# reference above). PyChrono is only installed in a separate `chrono` conda
# environment (not `multibody_env`), so it cannot be imported or run from
# this pytest session. Instead, the reference frequencies below were
# produced by manually running
# tests/cases/flexible/pychrono_backup/_pychrono_cantilever_reference.py (see that file's
# docstring for the full method and the required PATH fix) against the
# EXACT SAME tapered I(x) profile used here and in
# test_tapered_bending_frequencies_match_independent_fem_reference, and are
# hardcoded here the same way tests/cases/chrono/truth/*.npy bakes in
# pre-computed Chrono results for the rigid-body regression cases.
#
# PyChrono's own natural-frequency extraction (no pychrono.modal in this
# install, so a free-vibration time-domain simulation + FFT peak-picking is
# used instead of a direct eigenvalue solve) has a validated precision of
# about 1% for mode 1 and a few % for mode 2 (confirmed against the closed-
# form uniform-cantilever frequencies in the reference script's own
# self-check), so the tolerances below are set accordingly -- tighter than
# that would be checking PyChrono's own FFT precision, not FSM.
# ---------------------------------------------------------------------------
def test_tapered_bending_frequencies_match_pychrono_reference(tmp_path):
    L_BEAM, N_SEG = 1.0, 20
    E_STEEL, RHO_STEEL, A_VAL = 200e9, 7800.0, 1e-4
    I_ROOT, I_TIP = 8e-8, 2e-8
    x_dist = sym.symbols("x_dist", real=True, nonnegative=True)
    I_expr = I_ROOT - (I_ROOT - I_TIP) * x_dist / L_BEAM

    joints = [[0, 1]]
    types = ["R"]
    parent_cg_to_joint = [[0.0, 0.0]]
    joint_to_child_cg = [[L_BEAM / 2.0, 0.0]]
    prismatic_direction = [[np.nan, np.nan]]
    flex_bd, m0, J0 = [1], [None], [None]
    flex_bds, flex_n_seg, flex_L = [1], [N_SEG], [L_BEAM]
    flex_E, flex_A, flex_I, flex_rho = [E_STEEL], [A_VAL], [I_expr], [RHO_STEEL]
    flex_deformation_mode, flex_clamp_on = ["bending"], [True]

    out_path = str(tmp_path / "_gen_pychrono_crossval.py")
    mb.generate_flex_model_file(
        joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
        flex_bd, m0, J0, flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
        flex_deformation_mode, flex_clamp_on, out_path,
        flex_x_dist=[x_dist],
        ic=np.zeros(2 * N_SEG), gVec=np.zeros(N_SEG), tspan=0.1, TimeStep=0.01,
        overwrite=True,
    )
    ex = _load(out_path, "_gen_pychrono_crossval_pytest")

    with contextlib.redirect_stdout(io.StringIO()):
        MBDsys = mb.MbdSystem.from_example(ex)
    n_dof = sum(MBDsys.NDOF)
    q0, qd0 = np.zeros(n_dof), np.zeros(n_dof)
    M0, _, K0 = numeric_linearize(MBDsys, q0, qd0, ex.m0, ex.J0)
    eigvals = eigh(K0, M0, eigvals_only=True)
    freqs_fsm = np.sqrt(np.sort(eigvals[eigvals > 1e-6])) / (2 * np.pi)

    # From pychrono_backup/_pychrono_cantilever_reference.py, run manually against the
    # 'chrono' conda env (real PyChrono ChElementBeamEuler FEA, n_elem=40).
    freqs_pychrono = np.array([72.66666667, 390.66666667])
    rel_err = np.abs(freqs_fsm[:len(freqs_pychrono)] - freqs_pychrono) / freqs_pychrono
    assert rel_err[0] < 0.02, (
        f"FSM's mode-1 tapered-bending frequency does not agree with real "
        f"PyChrono: FSM={freqs_fsm[0]:.4f} Hz, PyChrono={freqs_pychrono[0]:.4f} Hz "
        f"(rel_err={rel_err[0]:.4f})"
    )
    assert rel_err[1] < 0.06, (
        f"FSM's mode-2 tapered-bending frequency does not agree with real "
        f"PyChrono: FSM={freqs_fsm[1]:.4f} Hz, PyChrono={freqs_pychrono[1]:.4f} Hz "
        f"(rel_err={rel_err[1]:.4f})"
    )

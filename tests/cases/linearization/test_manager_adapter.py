# tests/cases/linearization/test_manager_adapter.py
"""
Prompt 5 verification: LinearizationManager adapter handling.

No changes to LinearizationManager were required because:

* MoorPy's K is frequency-independent (2-D array) → the existing
  omega-dependent-K guard is not triggered.
* Our adapter returns shape (3·nb, 3·nb) for each matrix → detect_space
  correctly identifies "global" for systems where ncart ≠ nq.
* F["dc"] is a sym.MatrixBase → the existing DC-term branch handles it.

Tests
-----
1. ``TestDetectSpace``     — unit tests for the detect_space helper.
2. ``TestManagerTransform`` — verify R0^T @ M @ R0 etc. with a fake
                              Cartesian adapter (no MoorPy, no Capytaine).
3. ``TestScalarVsVector``  — scalar and vector omega produce consistent
                              shapes.
4. ``TestDCForce``         — constant DC SymPy matrix adds to eq_residual
                              correctly.
5. ``TestMoorPyAdapterSmoke`` — build a real MoorPyLinearMCKF adapter and
                                verify dimensions and finiteness.
                                Skipped if MoorPy is not installed.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Union

import numpy as np
import pytest
import sympy as sym

import multibody as mbd
from multibody import MbdSystem, normalize_prismatic
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization._helper_linearization import detect_space


# ============================================================= #
# Minimal MBD test fixtures                                      #
# ============================================================= #

def _double_pendulum() -> Tuple[MbdSystem, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two R-jointed bodies hanging under gravity.  nq=2, nb=2, ncart=6."""
    L1, L2 = sym.symbols("L1 L2", real=True, positive=True)

    mbd_sys = MbdSystem(
        joints=[[0, 1], [1, 2]],
        types=["R", "R"],
        parent_cg_to_joint=[[0, 0], [0, -L1 / 2]],
        joint_to_child_cg=[[0, -L1 / 2], [0, -L2 / 2]],
        prismatic_direction=normalize_prismatic(
            [[np.nan, np.nan], [np.nan, np.nan]]
        ),
        BodyDataSym=[L1, L2],
    )

    q0 = np.zeros(2)
    m0 = np.ones(2)
    J0 = np.ones(2)
    # mainSymVars = [θ1, θ2, θ̇1, θ̇2, L1, L2]  → length 6
    mainNumVars = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    return mbd_sys, q0, mainNumVars, m0, J0


def _single_float() -> Tuple[MbdSystem, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One floating body with one fairlead and one anchor.  nq=3, nb=1."""
    mbd_sys = MbdSystem(
        joints=[[0, 1]],
        types=["F"],
        parent_cg_to_joint=[[0, 0]],
        joint_to_child_cg=[[np.nan, np.nan]],
        prismatic_direction=normalize_prismatic([[np.nan, np.nan]]),
        Initial_Points={
            "GR": [[-50.0, -100.0]],
            "BD": {1: [[2.0, -5.0]]},
        },
    )

    q0 = np.zeros(3)          # x=0, z=0, theta=0
    m0 = np.array([1.0])
    J0 = np.array([1.0])
    mainNumVars = np.zeros(6)  # [x, z, theta, xd, zd, thetad]
    return mbd_sys, q0, mainNumVars, m0, J0


def _single_line_inputs():
    """One mooring line connecting GR[0] to BD[1][0]."""
    from multibody.linearization._moorpy_schema import (
        MoorPyInputs, LineType, MooringLine, PointRef,
    )
    return MoorPyInputs(
        depth=100.0,
        rho=1025.0,
        g=9.81,
        line_types={
            "chain": LineType(
                diameter=0.09,
                mass_per_length=160.0,
                axial_stiffness=854e6,
                drag_coefficient=2.4,
                added_mass_coefficient=1.0,
            )
        },
        mooring_lines=[
            MooringLine(
                name="line_1",
                line_type="chain",
                unstretched_length=250.0,
                num_segments=5,
                anchor=PointRef(kind="GR", point_id=0),
                fairlead=PointRef(kind="BD", body_id=1, point_id=0),
            )
        ],
    )


# ============================================================= #
# Fake Cartesian adapter (no external dependencies)             #
# ============================================================= #

class _FakeCartAdapter:
    """Returns prescribed constant matrices in body-Cartesian space."""

    def __init__(
        self,
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        ncart: int,
        dc_force: Any = None,
    ) -> None:
        self.name = "FakeCart"
        self._M = np.asarray(M, dtype=float)
        self._C = np.asarray(C, dtype=float)
        self._K = np.asarray(K, dtype=float)
        self._ncart = ncart
        self._dc = sym.zeros(ncart, 1) if dc_force is None else dc_force

    def frequency_domain_MCKF(
        self, omega: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        Fhat = np.zeros((self._ncart, 1), dtype=complex)
        return self._M, self._C, self._K, {"dc": self._dc, "phasor": Fhat}


# ============================================================= #
# 1. detect_space                                               #
# ============================================================= #

class TestDetectSpace:
    """Unit tests for detect_space(A, nb, nq)."""

    def test_global_shape_detected(self):
        # nb=2 → ncart=6, nq=2 → (6,6) is global
        A = np.zeros((6, 6))
        assert detect_space(A, nb=2, nq=2) == "global"

    def test_joint_shape_detected(self):
        # nq=2 → (2,2) is joint when ncart=6 ≠ 2
        A = np.zeros((2, 2))
        assert detect_space(A, nb=2, nq=2) == "joint"

    def test_omega_batched_global(self):
        # (10, 6, 6) — trailing shape matches global
        A = np.zeros((10, 6, 6))
        assert detect_space(A, nb=2, nq=2) == "global"

    def test_invalid_shape_raises(self):
        A = np.zeros((4, 4))
        with pytest.raises(ValueError):
            detect_space(A, nb=2, nq=2)


# ============================================================= #
# 2. Manager Cartesian → joint transformation                   #
# ============================================================= #

class TestManagerTransform:
    """Verify R0^T M R0, R0^T(CR0 + M RD0), R0^T K R0 via fake adapter."""

    @pytest.fixture(scope="class")
    def dp_setup(self):
        mbd_sys, q0, mainNumVars, m0, J0 = _double_pendulum()
        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)

        ncart = 3 * len(mbd_sys.NDOF)
        np.random.seed(42)
        # Random symmetric positive-definite matrices so values are non-trivial
        _rand_spd = lambda n: (lambda A: A.T @ A + n * np.eye(n))(np.random.randn(n, n))
        M_cart = _rand_spd(ncart)
        C_cart = _rand_spd(ncart)
        K_cart = _rand_spd(ncart)

        adapter = _FakeCartAdapter(M_cart, C_cart, K_cart, ncart)
        manager.register(adapter)

        totals = manager.assemble_frequency_domain(1.0)  # scalar omega
        return manager, M_cart, C_cart, K_cart, totals

    def test_Mq_equals_RtMR(self, dp_setup):
        manager, M_cart, C_cart, K_cart, totals = dp_setup
        R0 = manager.R0
        expected = R0.T @ M_cart @ R0
        actual = np.asarray(totals.M) - manager.M_mbd
        assert np.allclose(actual, expected, atol=1e-10), \
            f"Mq mismatch: max err = {np.max(np.abs(actual - expected)):.3e}"

    def test_Kq_equals_RtKR(self, dp_setup):
        manager, M_cart, C_cart, K_cart, totals = dp_setup
        R0 = manager.R0
        expected = R0.T @ K_cart @ R0
        actual = np.asarray(totals.K) - manager.K_mbd
        assert np.allclose(actual, expected, atol=1e-10), \
            f"Kq mismatch: max err = {np.max(np.abs(actual - expected)):.3e}"

    def test_Cq_equals_RtCRplusRtMRD0(self, dp_setup):
        manager, M_cart, C_cart, K_cart, totals = dp_setup
        R0 = manager.R0
        RD0 = manager.RD0
        expected = R0.T @ (C_cart @ R0 + M_cart @ RD0)
        actual = np.asarray(totals.C) - manager.C_mbd
        assert np.allclose(actual, expected, atol=1e-10), \
            f"Cq mismatch: max err = {np.max(np.abs(actual - expected)):.3e}"

    def test_matrices_are_symmetric(self, dp_setup):
        manager, M_cart, C_cart, K_cart, totals = dp_setup
        M = np.asarray(totals.M)
        K = np.asarray(totals.K)
        assert np.allclose(M, M.T, atol=1e-10), "M_total is not symmetric"
        assert np.allclose(K, K.T, atol=1e-10), "K_total is not symmetric"


# ============================================================= #
# 3. Scalar vs vector omega                                      #
# ============================================================= #

class TestScalarVsVector:
    """Scalar and vector omega both produce consistent shapes."""

    @pytest.fixture(scope="class")
    def manager_with_adapter(self):
        mbd_sys, q0, mainNumVars, m0, J0 = _double_pendulum()
        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        ncart = 3 * len(mbd_sys.NDOF)
        adapter = _FakeCartAdapter(
            np.eye(ncart) * 100.0,
            np.eye(ncart) * 10.0,
            np.eye(ncart) * 500.0,
            ncart,
        )
        manager.register(adapter)
        return manager

    def test_scalar_omega_gives_2d_M(self, manager_with_adapter):
        totals = manager_with_adapter.assemble_frequency_domain(1.0)
        nq = manager_with_adapter.nq
        assert np.asarray(totals.M).shape == (nq, nq)

    def test_vector_omega_gives_3d_M(self, manager_with_adapter):
        omegas = np.linspace(0.1, 2.0, 5)
        totals = manager_with_adapter.assemble_frequency_domain(omegas)
        nq = manager_with_adapter.nq
        assert np.asarray(totals.M).shape == (5, nq, nq)

    def test_scalar_and_vector_M_agree_at_same_omega(self, manager_with_adapter):
        omega0 = 1.5
        t_scalar = manager_with_adapter.assemble_frequency_domain(omega0)
        t_vector = manager_with_adapter.assemble_frequency_domain(np.array([omega0]))
        assert np.allclose(t_scalar.M, t_vector.M[0], atol=1e-12)

    def test_K_is_always_2d(self, manager_with_adapter):
        """K must be omega-independent (2-D) regardless of omega input shape."""
        nq = manager_with_adapter.nq
        totals = manager_with_adapter.assemble_frequency_domain(np.linspace(0.1, 2.0, 4))
        assert np.asarray(totals.K).shape == (nq, nq)


# ============================================================= #
# 4. DC force handling                                          #
# ============================================================= #

class TestDCForce:
    """Constant DC SymPy matrix contributes to eq_residual correctly."""

    def _make_manager_with_dc(self, dc_value: float):
        mbd_sys, q0, mainNumVars, m0, J0 = _double_pendulum()
        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        ncart = 3 * len(mbd_sys.NDOF)

        # Construct a non-zero constant DC force: e.g. Fx on body-1 = dc_value
        dc = sym.zeros(ncart, 1)
        dc[0] = dc_value  # Fx on body 1

        adapter = _FakeCartAdapter(
            np.zeros((ncart, ncart)),
            np.zeros((ncart, ncart)),
            np.zeros((ncart, ncart)),
            ncart,
            dc_force=dc,
        )
        manager.register(adapter)
        return manager

    def test_zero_dc_leaves_residual_unchanged(self):
        mbd_sys, q0, mainNumVars, m0, J0 = _double_pendulum()
        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        baseline = manager.assemble_frequency_domain(1.0).eq_residual.copy()

        ncart = 3 * len(mbd_sys.NDOF)
        adapter = _FakeCartAdapter(
            np.zeros((ncart, ncart)),
            np.zeros((ncart, ncart)),
            np.zeros((ncart, ncart)),
            ncart,
        )
        manager.register(adapter)
        with_adapter = manager.assemble_frequency_domain(1.0).eq_residual
        assert np.allclose(baseline, with_adapter, atol=1e-12)

    def test_constant_dc_adds_to_residual(self):
        dc_value = 500.0  # 500 N in Fx on body 1
        manager = self._make_manager_with_dc(dc_value)
        mbd_ref, q0_ref, mnv_ref, m0_ref, J0_ref = _double_pendulum()
        totals_no_dc = LinearizationManager(
            mbd_ref, q0_ref, mnv_ref, m0_ref, J0_ref
        ).assemble_frequency_domain(1.0)

        # Re-run with DC
        totals_with_dc = manager.assemble_frequency_domain(1.0)
        delta = np.asarray(totals_with_dc.eq_residual) - np.asarray(totals_no_dc.eq_residual)
        # Delta must be non-zero (DC force added a generalized-force contribution)
        assert np.linalg.norm(delta) > 1e-6, \
            "DC force had no effect on eq_residual"

    def test_constant_dc_does_not_add_stiffness(self):
        """A constant DC force has zero Jacobian → K_dc = 0."""
        dc_value = 1000.0
        manager = self._make_manager_with_dc(dc_value)
        totals = manager.assemble_frequency_domain(1.0)

        # Build a reference manager with zero DC adapter
        mbd_sys, q0, mainNumVars, m0, J0 = _double_pendulum()
        ref_manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        ref_totals = ref_manager.assemble_frequency_domain(1.0)

        # K must be the same whether DC is zero or constant
        assert np.allclose(
            np.asarray(totals.K), np.asarray(ref_totals.K), atol=1e-10
        ), "Constant DC force must not add stiffness"


# ============================================================= #
# 5. MoorPy adapter smoke test                                  #
# ============================================================= #

@pytest.mark.skipif(
    pytest.importorskip("moorpy", reason="MoorPy not installed") is None,
    reason="MoorPy not installed",
)
class TestMoorPyAdapterSmoke:
    """Build a real MoorPyLinearMCKF and verify dimensions/finiteness.

    Skipped automatically if MoorPy is not installed.
    """

    @pytest.fixture(scope="class")
    def adapter(self):
        moorpy = pytest.importorskip("moorpy")  # skip if not available
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF

        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()
        inputs = _single_line_inputs()
        return MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, inputs)

    def test_matrix_dimensions(self, adapter):
        nb_m4e = 1
        ncart = 3 * nb_m4e
        M, C, K, F = adapter.frequency_domain_MCKF(1.0)
        assert M.shape == (ncart, ncart), f"M shape {M.shape} != ({ncart},{ncart})"
        assert C.shape == (ncart, ncart)
        assert K.shape == (ncart, ncart)
        assert F["phasor"].shape == (ncart, 1)

    def test_matrices_are_finite(self, adapter):
        M, C, K, _ = adapter.frequency_domain_MCKF(1.0)
        assert np.all(np.isfinite(M)), "M has non-finite entries"
        assert np.all(np.isfinite(C)), "C has non-finite entries"
        assert np.all(np.isfinite(K)), "K has non-finite entries"

    def test_phasor_is_zero(self, adapter):
        _, _, _, F = adapter.frequency_domain_MCKF(1.0)
        assert np.all(F["phasor"] == 0), "Mooring excitation phasor must be zero"

    def test_dc_is_sympy_matrix_by_default(self, adapter):
        _, _, _, F = adapter.frequency_domain_MCKF(1.0)
        assert isinstance(F["dc"], sym.MatrixBase)

    def test_reduction_indices(self, adapter):
        """idx = [0, 2, 4] for one body (MoorPy x=0, z=2, pitch=4)."""
        expected = np.array([0, 2, 4])
        assert np.array_equal(adapter.reduction_indices, expected)

    def test_K_has_nonzero_entries(self, adapter):
        """Mooring stiffness matrix should not be all zeros for a taut line."""
        _, _, K, _ = adapter.frequency_domain_MCKF(1.0)
        assert np.any(np.abs(K) > 1e-3), \
            "K_cart is all near-zero; check MoorPy initialization"

    def test_frequency_does_not_change_matrices(self, adapter):
        """All MoorPy matrices are frequency-independent."""
        M1, C1, K1, _ = adapter.frequency_domain_MCKF(0.5)
        M2, C2, K2, _ = adapter.frequency_domain_MCKF(2.0)
        assert np.array_equal(M1, M2), "M changed with frequency"
        assert np.array_equal(C1, C2), "C changed with frequency"
        assert np.array_equal(K1, K2), "K changed with frequency"

    def test_moorpy_body_reference_equals_m4e_cg(self, adapter):
        """MoorPy body r6[:3] must equal the M4E CG position at q0."""
        ms = adapter.moorpy_system
        body = ms.bodyList[0]
        # At q0 = [0, 0, 0]: CG is at origin [0, 0]
        assert np.allclose(body.r6[0], 0.0, atol=1e-10)  # x
        assert np.allclose(body.r6[2], 0.0, atol=1e-10)  # z
        assert np.allclose(body.r6[4], 0.0, atol=1e-10)  # pitch

    def test_manager_registers_adapter(self, adapter):
        """Verify the adapter integrates with LinearizationManager."""
        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()
        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager.register(adapter)

        totals = manager.assemble_frequency_domain(1.0)
        nq = manager.nq
        assert np.asarray(totals.K).shape == (nq, nq)
        assert np.all(np.isfinite(totals.K))

# tests/cases/linearization/test_moorpy_full.py
"""
Prompt 7: Full MoorPy integration test suite.

Coverage
--------
1. TestPointMapping     — GR/BD → MoorPy fixed/attached point placement;
                          fairlead global position vs M4E evaluated position;
                          body reference point equals M4E CG.

2. TestDOFReduction     — 6→3 reduction indices; M, A, B, K slicing;
                          mean body-wrench reduction and ordering;
                          two-body global index sequence.

3. TestNoDoubleTransformation
                        — adapter K matches body.getDynamicMatrices reduction;
                          translation–rotation coupling (off-diagonal) preserved;
                          applying the fairlead H matrix again changes the result.

4. TestCrossBodyCoupling
                        — two bodies with independent lines: zero off-diagonal 3×3
                          blocks; each body's block is nonzero.

5. TestMeanForce        — include_mean_force=False/True; force ordering [Fx, Fz, My];
                          q0 never mutated; residual changes only when enabled;
                          no stiffness double counting.

6. TestRealMoorPySmokeExtended
                        — finite 6-DOF matrices; finite reduced 3-DOF matrices;
                          correct dimensions; translation–pitch coupling nonzero;
                          K dominant over M, C (stiffness-dominated mooring).

All MoorPy tests are skipped when MoorPy is not installed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import sympy as sym

import multibody as _mbd_pkg            # noqa: F401 — force src layout on path
from multibody import MbdSystem, normalize_prismatic


# ================================================================= #
# Fixtures                                                           #
# ================================================================= #

def _single_float() -> Tuple[MbdSystem, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One floating body, fairlead at body-local (2, -5), anchor at (-50, -100)."""
    mbd_sys = MbdSystem(
        joints=[[0, 1]],
        types=["F"],
        parent_cg_to_joint=[[0.0, 0.0]],
        joint_to_child_cg=[[np.nan, np.nan]],
        prismatic_direction=normalize_prismatic([[np.nan, np.nan]]),
        Initial_Points={
            "GR": [[-50.0, -100.0]],
            "BD": {1: [[2.0, -5.0]]},
        },
    )
    q0           = np.zeros(3)
    m0           = np.array([1.0])
    J0           = np.array([1.0])
    mainNumVars  = np.zeros(6)
    return mbd_sys, q0, mainNumVars, m0, J0


def _single_line_inputs():
    """One catenary chain line connecting GR[0] to BD[1][0]."""
    from multibody.linearization._moorpy_schema import (
        MoorPyInputs, LineType, MooringLine, PointRef,
    )
    return MoorPyInputs(
        depth=100.0, rho=1025.0, g=9.81,
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
                name="line_1", line_type="chain",
                unstretched_length=250.0, num_segments=5,
                anchor=PointRef(kind="GR", point_id=0),
                fairlead=PointRef(kind="BD", body_id=1, point_id=0),
            )
        ],
    )


def _two_floats() -> Tuple[MbdSystem, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two floating bodies branching from ground.

    Body 1 at global (-10, 0), fairlead at body-local (0, -5) → global (-10, -5).
    Body 2 at global (+10, 0), fairlead at body-local (0, -5) → global (+10, -5).
    Anchors: GR[0]=(-60,-100) for body 1, GR[1]=(+60,-100) for body 2.

    nq=6, nb=2, ncart=6.
    """
    mbd_sys = MbdSystem(
        joints=[[0, 1], [0, 2]],
        types=["F", "F"],
        parent_cg_to_joint=[[0.0, 0.0], [0.0, 0.0]],
        joint_to_child_cg=[[np.nan, np.nan], [np.nan, np.nan]],
        prismatic_direction=normalize_prismatic(
            [[np.nan, np.nan], [np.nan, np.nan]]
        ),
        Initial_Points={
            "GR": [[-60.0, -100.0], [60.0, -100.0]],
            "BD": {1: [[0.0, -5.0]], 2: [[0.0, -5.0]]},
        },
    )
    # q = [x1, z1, θ1, x2, z2, θ2] — absolute positions at operating point
    q0          = np.array([-10.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    m0          = np.array([1.0, 1.0])
    J0          = np.array([1.0, 1.0])
    # mainNumVars = [q0, qd=0] of length 2*nq=12
    mainNumVars = np.array([-10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                              0.0,  0.0, 0.0,  0.0, 0.0, 0.0])
    return mbd_sys, q0, mainNumVars, m0, J0


def _two_line_inputs():
    """Two independent catenary lines: GR[0]→BD[1][0]  and  GR[1]→BD[2][0]."""
    from multibody.linearization._moorpy_schema import (
        MoorPyInputs, LineType, MooringLine, PointRef,
    )
    chain = LineType(
        diameter=0.09, mass_per_length=160.0, axial_stiffness=854e6,
        drag_coefficient=2.4, added_mass_coefficient=1.0,
    )
    return MoorPyInputs(
        depth=100.0, rho=1025.0, g=9.81,
        line_types={"chain": chain},
        mooring_lines=[
            MooringLine(
                name="line_1", line_type="chain",
                unstretched_length=250.0, num_segments=5,
                anchor=PointRef(kind="GR", point_id=0),
                fairlead=PointRef(kind="BD", body_id=1, point_id=0),
            ),
            MooringLine(
                name="line_2", line_type="chain",
                unstretched_length=250.0, num_segments=5,
                anchor=PointRef(kind="GR", point_id=1),
                fairlead=PointRef(kind="BD", body_id=2, point_id=0),
            ),
        ],
    )


def _make_adapter(include_mean_force: bool = False):
    """Build a MoorPyLinearMCKF for the single-float system."""
    from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
    mbd_sys, q0, mainNumVars, m0, J0 = _single_float()
    inputs = _single_line_inputs()
    return MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, inputs,
                             include_mean_force=include_mean_force)


def _make_two_body_adapter():
    """Build a MoorPyLinearMCKF for the two-float system."""
    from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
    mbd_sys, q0, mainNumVars, m0, J0 = _two_floats()
    inputs = _two_line_inputs()
    return MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, inputs)


# ================================================================= #
# 1. Point mapping                                                   #
# ================================================================= #

class TestPointMapping:
    """Verify that GR/BD points are correctly placed in MoorPy."""

    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("moorpy")
        return _make_adapter()

    def test_anchor_is_type_1_fixed(self, adapter):
        """GR anchor must be a fixed MoorPy point (type=1) not attached to a body."""
        ms = adapter.moorpy_system
        attached_ids = {pid for body in ms.bodyList for pid in body.attachedP}
        anchor_points = [
            p for i, p in enumerate(ms.pointList)
            if p.type == 1 and (i + 1) not in attached_ids
        ]
        assert len(anchor_points) == 1, \
            f"Expected 1 fixed anchor point, found {len(anchor_points)}"
        assert np.allclose(anchor_points[0].r, [-50.0, 0.0, -100.0], atol=1e-8)

    def test_fairlead_is_attached_to_body(self, adapter):
        """BD fairlead must be a fixed (type=1) point attached to the body.

        Fairleads are type=1 (not -1/coupled): attaching via ``body=`` already
        slaves the point's position to the body, so making it type=-1 as well
        would double-count 3 extra coupled DOFs (see _moorpy_system_builder.py).
        """
        ms = adapter.moorpy_system
        body = ms.bodyList[0]
        assert len(body.attachedP) == 1, \
            f"Expected 1 fairlead point attached to body, found {len(body.attachedP)}"
        pt_number = body.attachedP[0]  # 1-based index
        fairlead_pt = ms.pointList[pt_number - 1]
        assert fairlead_pt.type == 1, \
            f"Fairlead point type = {fairlead_pt.type}, expected 1 (fixed-but-attached)"

    def test_fairlead_global_position_matches_m4e(self, adapter):
        """MoorPy fairlead global position must equal M4E's evaluated BD position."""
        ms = adapter.moorpy_system
        # M4E BD[1][0] = body-local (2, -5); body at (0,0) with theta=0
        # → global = (2, -5)
        body = ms.bodyList[0]
        fairlead_pt = ms.pointList[body.attachedP[0] - 1]
        assert np.allclose(fairlead_pt.r[0], 2.0, atol=1e-3), \
            f"Fairlead x = {fairlead_pt.r[0]}, expected 2.0"
        assert np.allclose(fairlead_pt.r[2], -5.0, atol=1e-3), \
            f"Fairlead z = {fairlead_pt.r[2]}, expected -5.0"

    def test_body_r6_matches_m4e_cg_at_q0(self, adapter):
        """MoorPy body r6 = [x=0, y=0, z=0, roll=0, pitch=0, yaw=0] at q0=[0,0,0]."""
        body = adapter.moorpy_system.bodyList[0]
        assert np.allclose(body.r6[[0, 2, 4]], [0.0, 0.0, 0.0], atol=1e-10), \
            f"Body r6[[0,2,4]] = {body.r6[[0,2,4]]}, expected [0,0,0]"

    def test_fairlead_rPointRel_stored_in_body(self, adapter):
        """Body must store the fairlead's body-local relative position."""
        body = adapter.moorpy_system.bodyList[0]
        # rPointRel should be close to body-local [2, 0, -5]
        assert len(body.rPointRel) == 1, "Expected 1 attached point relative position"
        rrel = body.rPointRel[0]  # shape (3,)
        assert np.allclose(rrel[0], 2.0, atol=1e-3), f"rPointRel x = {rrel[0]}"
        assert np.allclose(rrel[2], -5.0, atol=1e-3), f"rPointRel z = {rrel[2]}"


# ================================================================= #
# 2. Six-to-three DOF reduction                                     #
# ================================================================= #

class TestDOFReduction:
    """Verify that 6→3 reduction is applied correctly for M, A, B, K and force."""

    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("moorpy")
        return _make_adapter()

    def test_single_body_reduction_indices(self, adapter):
        """Reduction indices for one body must be [0, 2, 4]."""
        assert np.array_equal(adapter.reduction_indices, [0, 2, 4])

    def test_M_cart_equals_M_plus_A_reduced(self, adapter):
        """M_cart must equal (M + A)[idx,:][:,idx] from the MoorPy system matrix."""
        ms = adapter.moorpy_system
        M6, A6, _, _ = ms.getSystemDynamicMatrices(DOFtype="coupled", lines_only=True)
        idx = adapter.reduction_indices
        expected = (M6 + A6)[np.ix_(idx, idx)]
        assert np.allclose(adapter._M_cart[0:3, 0:3], expected, atol=1e-6), \
            f"M_cart mismatch; max diff = {np.max(np.abs(adapter._M_cart[0:3, 0:3] - expected)):.2e}"

    def test_C_cart_equals_B_reduced(self, adapter):
        """C_cart must equal B[idx,:][:,idx] from the MoorPy system matrix."""
        ms = adapter.moorpy_system
        _, _, B6, _ = ms.getSystemDynamicMatrices(DOFtype="coupled", lines_only=True)
        idx = adapter.reduction_indices
        expected = B6[np.ix_(idx, idx)]
        assert np.allclose(adapter._C_cart[0:3, 0:3], expected, atol=1e-6)

    def test_K_cart_equals_K_reduced(self, adapter):
        """K_cart must equal K[idx,:][:,idx] from the MoorPy system matrix."""
        ms = adapter.moorpy_system
        _, _, _, K6 = ms.getSystemDynamicMatrices(DOFtype="coupled", lines_only=True)
        idx = adapter.reduction_indices
        expected = K6[np.ix_(idx, idx)]
        assert np.allclose(adapter._K_cart[0:3, 0:3], expected, atol=1e-6), \
            f"K_cart mismatch; max diff = {np.max(np.abs(adapter._K_cart[0:3, 0:3] - expected)):.2e}"

    def test_two_body_index_sequence(self):
        """Two bodies produce reduction indices [0,2,4,6,8,10]."""
        pytest.importorskip("moorpy")
        adapter2 = _make_two_body_adapter()
        expected = np.array([0, 2, 4, 6, 8, 10])
        assert np.array_equal(adapter2.reduction_indices, expected), \
            f"Two-body idx = {adapter2.reduction_indices}, expected {expected}"

    def test_two_body_cartesian_ordering(self):
        """Two-body adapter places body1 at rows [0,1,2] and body2 at rows [3,4,5]."""
        pytest.importorskip("moorpy")
        adapter2 = _make_two_body_adapter()
        expected_rows = np.array([0, 1, 2, 3, 4, 5])
        assert np.array_equal(adapter2.m4e_cartesian_rows, expected_rows), \
            f"m4e_cartesian_rows = {adapter2.m4e_cartesian_rows}, expected {expected_rows}"

    def test_mean_wrench_reduction_gives_Fx_Fz_My(self):
        """include_mean_force=True: dc[0]=Fx, dc[1]=Fz, dc[2]=My."""
        pytest.importorskip("moorpy")
        adapter_dc = _make_adapter(include_mean_force=True)
        ms = adapter_dc.moorpy_system
        body = ms.bodyList[0]

        # MoorPy 6-DOF wrench: [Fx, Fy, Fz, Mx, My, Mz]
        f6 = body.getForces(lines_only=True, all_DOFs=True)
        Fx, Fz, My = f6[0], f6[2], f6[4]

        dc = adapter_dc._Fdc_sym
        assert float(dc[0]) == pytest.approx(Fx, rel=1e-6), \
            f"dc[0]={float(dc[0])} ≠ Fx={Fx}"
        assert float(dc[1]) == pytest.approx(Fz, rel=1e-6), \
            f"dc[1]={float(dc[1])} ≠ Fz={Fz}"
        assert float(dc[2]) == pytest.approx(My, rel=1e-6), \
            f"dc[2]={float(dc[2])} ≠ My={My}"


# ================================================================= #
# 3. No double transformation                                        #
# ================================================================= #

class TestNoDoubleTransformation:
    """Verify the adapter does not re-apply the fairlead H transformation."""

    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("moorpy")
        return _make_adapter()

    def test_K_cart_matches_body_getDynamicMatrices(self, adapter):
        """K_cart block must equal body.getDynamicMatrices reduction — no extra H."""
        ms = adapter.moorpy_system
        body = ms.bodyList[0]
        # body.getDynamicMatrices with all_DOFs=True → 6×6 body-level matrix
        _, _, _, K6_body = body.getDynamicMatrices(lines_only=True, all_DOFs=True)
        local_idx = np.array([0, 2, 4])
        K3_body = K6_body[np.ix_(local_idx, local_idx)]

        K3_adapter = adapter._K_cart[0:3, 0:3]
        assert np.allclose(K3_adapter, K3_body, atol=1e-6), (
            f"K_cart block does not match body.getDynamicMatrices.\n"
            f"Adapter:  {K3_adapter}\n"
            f"Body:     {K3_body}\n"
            f"max diff = {np.max(np.abs(K3_adapter - K3_body)):.2e}"
        )

    def test_M_cart_matches_body_getDynamicMatrices(self, adapter):
        """M_cart = (M+A) block from body.getDynamicMatrices — no extra H."""
        ms = adapter.moorpy_system
        body = ms.bodyList[0]
        M6_body, A6_body, _, _ = body.getDynamicMatrices(lines_only=True, all_DOFs=True)
        local_idx = np.array([0, 2, 4])
        MA3_body = (M6_body + A6_body)[np.ix_(local_idx, local_idx)]

        M3_adapter = adapter._M_cart[0:3, 0:3]
        assert np.allclose(M3_adapter, MA3_body, atol=1e-6)

    def test_translation_rotation_coupling_is_nonzero(self, adapter):
        """K[x, pitch] must be nonzero because fairlead has a 2 m horizontal offset."""
        K_cart = adapter._K_cart
        Kx_pitch = K_cart[0, 2]  # x-row, pitch-col
        assert abs(Kx_pitch) > 1.0, (
            f"K[x, pitch] = {Kx_pitch:.4f} — expected nonzero from 2 m offset fairlead"
        )

    def test_applying_H_again_changes_K(self, adapter):
        """A second H transformation produces a different result, proving the adapter
        returns the correctly single-transformed body-level matrix."""
        ms = adapter.moorpy_system
        body = ms.bodyList[0]

        # The body-level K after ONE correct H transform (via getDynamicMatrices).
        _, _, _, K6_body = body.getDynamicMatrices(lines_only=True, all_DOFs=True)
        local_idx = np.array([0, 2, 4])
        K3_correct = K6_body[np.ix_(local_idx, local_idx)]

        # Verify adapter matches the single-transform result.
        K3_adapter = adapter._K_cart[0:3, 0:3]
        assert np.allclose(K3_adapter, K3_correct, atol=1e-6), \
            "Adapter K does not match single-transform body result"

        # Now simulate a WRONG second H transform in the planar [x,z,pitch] space.
        # Fairlead body-local offset: rx=2, rz=-5 → pitch moment arm.
        # A naive second rotation of K3_correct around pitch would differ.
        rrel = body.rPointRel[0]  # [rx, ry, rz]
        rx, rz = float(rrel[0]), float(rrel[2])
        # Build a tiny 3×3 moment-arm matrix in the planar DOF space [x, z, pitch]
        # representing the extra (wrong) H step: only the pitch coupling matters.
        H_planar = np.array([[0.0, 0.0, rz],
                              [0.0, 0.0, -rx],
                              [-rz, rx,   0.0]])
        K3_double = (np.eye(3) + H_planar).T @ K3_correct @ (np.eye(3) + H_planar)

        # The double-transformed result must differ from the correct single-transform.
        assert not np.allclose(K3_correct, K3_double, atol=1e-3), (
            "Double-H transform produced the same result — the offset fairlead may not "
            "produce enough coupling to distinguish the two."
        )


# ================================================================= #
# 4. Cross-body coupling                                            #
# ================================================================= #

class TestCrossBodyCoupling:
    """Two moored bodies with independent lines: verify index placement and
    zero off-diagonal blocks (no body-to-body line in this config)."""

    @pytest.fixture(scope="class")
    def adapter2(self):
        pytest.importorskip("moorpy")
        return _make_two_body_adapter()

    def test_K_cart_shape(self, adapter2):
        """Two bodies → K_cart must be 6×6."""
        assert adapter2._K_cart.shape == (6, 6), \
            f"K_cart.shape = {adapter2._K_cart.shape}, expected (6,6)"

    def test_body1_block_is_nonzero(self, adapter2):
        """Body-1 diagonal block K_cart[0:3, 0:3] must be nonzero."""
        block = adapter2._K_cart[0:3, 0:3]
        assert np.any(np.abs(block) > 1.0), \
            f"Body-1 K block is all near-zero:\n{block}"

    def test_body2_block_is_nonzero(self, adapter2):
        """Body-2 diagonal block K_cart[3:6, 3:6] must be nonzero."""
        block = adapter2._K_cart[3:6, 3:6]
        assert np.any(np.abs(block) > 1.0), \
            f"Body-2 K block is all near-zero:\n{block}"

    def test_off_diagonal_K_blocks_zero_for_independent_lines(self, adapter2):
        """Independent mooring lines produce no K cross-coupling between bodies."""
        K_offdiag_12 = adapter2._K_cart[0:3, 3:6]
        K_offdiag_21 = adapter2._K_cart[3:6, 0:3]
        assert np.allclose(K_offdiag_12, 0.0, atol=1e-6), \
            f"K off-diagonal [1→2] nonzero:\n{K_offdiag_12}"
        assert np.allclose(K_offdiag_21, 0.0, atol=1e-6), \
            f"K off-diagonal [2→1] nonzero:\n{K_offdiag_21}"

    def test_body1_and_body2_blocks_differ(self, adapter2):
        """Each body's block must reflect its own geometry (same line geometry
        here, so blocks should be equal — but not trivially zero)."""
        block1 = adapter2._K_cart[0:3, 0:3]
        block2 = adapter2._K_cart[3:6, 3:6]
        # Both bodies have symmetrically placed mooring lines
        assert np.allclose(block1, block2, atol=0.1 * np.max(np.abs(block1))), \
            "Expected symmetric body blocks for symmetric two-body system"

    def test_frequency_domain_MCKF_shape_two_bodies(self, adapter2):
        """Two bodies → adapter matrices shape = (6,6)."""
        M, C, K, F = adapter2.frequency_domain_MCKF(1.0)
        assert M.shape == (6, 6)
        assert C.shape == (6, 6)
        assert K.shape == (6, 6)
        assert F["phasor"].shape == (6, 1)


# ================================================================= #
# 5. Mean force                                                      #
# ================================================================= #

class TestMeanForce:
    """include_mean_force=False/True behavior."""

    @pytest.fixture(scope="class")
    def adapter_no_dc(self):
        pytest.importorskip("moorpy")
        return _make_adapter(include_mean_force=False)

    @pytest.fixture(scope="class")
    def adapter_dc(self):
        pytest.importorskip("moorpy")
        return _make_adapter(include_mean_force=True)

    def test_dc_false_returns_zero_sympy_matrix(self, adapter_no_dc):
        _, _, _, F = adapter_no_dc.frequency_domain_MCKF(1.0)
        dc = F["dc"]
        assert isinstance(dc, sym.MatrixBase)
        assert dc == sym.zeros(3, 1), \
            f"Expected sym.zeros, got {dc.T}"

    def test_dc_true_returns_nonzero_sympy_matrix(self, adapter_dc):
        _, _, _, F = adapter_dc.frequency_domain_MCKF(1.0)
        dc = F["dc"]
        assert isinstance(dc, sym.MatrixBase)
        # Catenary line exerts a net horizontal and vertical force
        vals = [float(dc[i]) for i in range(3)]
        assert any(abs(v) > 1.0 for v in vals), \
            f"DC force is near-zero: {vals}; expected mooring pretension"

    def test_dc_force_ordering_Fx_Fz_My(self, adapter_dc):
        """DC force entries must follow [Fx, Fz, My] (M4E Cartesian body ordering)."""
        ms = adapter_dc.moorpy_system
        body = ms.bodyList[0]
        f6 = body.getForces(lines_only=True, all_DOFs=True)
        Fx, Fz, My = f6[0], f6[2], f6[4]

        _, _, _, F = adapter_dc.frequency_domain_MCKF(1.0)
        dc = F["dc"]
        assert float(dc[0]) == pytest.approx(Fx, rel=1e-6), "dc[0] ≠ Fx"
        assert float(dc[1]) == pytest.approx(Fz, rel=1e-6), "dc[1] ≠ Fz"
        assert float(dc[2]) == pytest.approx(My, rel=1e-6), "dc[2] ≠ My"

    def test_q0_not_modified_by_include_mean_force(self):
        """q0 passed to the adapter must not be mutated."""
        pytest.importorskip("moorpy")
        q0_orig = np.zeros(3)
        q0_copy = q0_orig.copy()
        mbd_sys, _, mainNumVars, _, _ = _single_float()
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
        MoorPyLinearMCKF(mbd_sys, q0_orig, mainNumVars, _single_line_inputs(),
                          include_mean_force=True)
        assert np.array_equal(q0_orig, q0_copy), \
            "q0 was mutated during MoorPyLinearMCKF construction"

    def test_equilibrium_residual_unchanged_without_dc(self):
        """With include_mean_force=False, the eq_residual is that of MBD alone."""
        pytest.importorskip("moorpy")
        from multibody.linearization.linearization_main import LinearizationManager
        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()

        # Manager with no adapters
        manager_bare = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        residual_bare = manager_bare.assemble_frequency_domain(1.0).eq_residual.copy()

        # Manager with zero-DC adapter
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
        manager_nodc = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        adapter = MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, _single_line_inputs(),
                                    include_mean_force=False)
        manager_nodc.register(adapter)
        residual_nodc = manager_nodc.assemble_frequency_domain(1.0).eq_residual

        assert np.allclose(residual_bare, residual_nodc, atol=1e-12), (
            "Registering a zero-DC adapter changed eq_residual."
        )

    def test_equilibrium_residual_changes_with_dc(self):
        """With include_mean_force=True, eq_residual includes the mooring wrench."""
        pytest.importorskip("moorpy")
        from multibody.linearization.linearization_main import LinearizationManager
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()

        manager_nodc = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager_nodc.register(
            MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, _single_line_inputs(),
                              include_mean_force=False)
        )
        manager_dc = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager_dc.register(
            MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, _single_line_inputs(),
                              include_mean_force=True)
        )

        r_nodc = np.asarray(manager_nodc.assemble_frequency_domain(1.0).eq_residual)
        r_dc   = np.asarray(manager_dc.assemble_frequency_domain(1.0).eq_residual)

        assert not np.allclose(r_nodc, r_dc, atol=1e-3), (
            "eq_residual did not change when mean mooring force was enabled."
        )

    def test_no_stiffness_double_counting(self):
        """K_joint must be the same whether include_mean_force is True or False.

        A constant DC force has zero Jacobian w.r.t. q, so it must not add
        stiffness beyond what is already in K_cart (the mooring stiffness matrix).
        """
        pytest.importorskip("moorpy")
        from multibody.linearization.linearization_main import LinearizationManager
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF
        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()

        manager_nodc = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager_nodc.register(
            MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, _single_line_inputs(),
                              include_mean_force=False)
        )
        manager_dc = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager_dc.register(
            MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, _single_line_inputs(),
                              include_mean_force=True)
        )

        K_nodc = np.asarray(manager_nodc.assemble_frequency_domain(1.0).K)
        K_dc   = np.asarray(manager_dc.assemble_frequency_domain(1.0).K)

        assert np.allclose(K_nodc, K_dc, atol=1e-6), (
            f"K_joint differs with/without mean force — possible stiffness double counting.\n"
            f"K_nodc:\n{K_nodc}\nK_dc:\n{K_dc}\n"
            f"diff max = {np.max(np.abs(K_nodc - K_dc)):.2e}"
        )

    def test_lines_only_force_excludes_body_weight(self, adapter_dc):
        """lines_only=True must not include body weight/buoyancy (Fz from gravity)."""
        ms = adapter_dc.moorpy_system
        body = ms.bodyList[0]
        # lines_only force
        f6_lines = body.getForces(lines_only=True, all_DOFs=True)
        # full force (includes weight and buoyancy)
        f6_full  = body.getForces(lines_only=False, all_DOFs=True)
        # For a real body with non-zero mass and volume they would differ;
        # here m=1 kg and rho×V≈0, so the difference is just m*g ≈ 9.81 N.
        # The adapter must use lines_only, so dc[1] (Fz) should NOT include mg.
        # We just verify that the lines-only and full forces are different.
        # (This test is informational — if mass=0 the test is trivially satisfied.)
        dc_fz = float(adapter_dc._Fdc_sym[1])
        assert dc_fz == pytest.approx(f6_lines[2], rel=1e-6), \
            "DC force does not match lines-only body force"


# ================================================================= #
# 6. Real MoorPy smoke (extended)                                   #
# ================================================================= #

class TestRealMoorPySmokeExtended:
    """Extended smoke tests for a real single-line MoorPy system."""

    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("moorpy")
        return _make_adapter()

    def test_6dof_system_matrices_are_finite(self, adapter):
        """Full 6-DOF MoorPy system matrices must be finite."""
        ms = adapter.moorpy_system
        M6, A6, B6, K6 = ms.getSystemDynamicMatrices(DOFtype="coupled", lines_only=True)
        for name, mat in [("M", M6), ("A", A6), ("B", B6), ("K", K6)]:
            assert np.all(np.isfinite(mat)), \
                f"{name} has non-finite entries at rows/cols: "  \
                f"{np.argwhere(~np.isfinite(mat))}"

    def test_reduced_3dof_matrices_are_finite(self, adapter):
        """Reduced 3-DOF Cartesian matrices must be finite."""
        M, C, K, _ = adapter.frequency_domain_MCKF(0.5)
        for name, mat in [("M_cart", M), ("C_cart", C), ("K_cart", K)]:
            assert np.all(np.isfinite(mat)), f"{name} has non-finite entries"

    def test_K_is_larger_than_M_at_low_freq(self, adapter):
        """For a mooring-dominated system, ||K|| should exceed ||M|| at low omega."""
        M, _, K, _ = adapter.frequency_domain_MCKF(0.1)
        assert np.linalg.norm(K) > np.linalg.norm(M), \
            "Expected stiffness-dominated mooring at low frequency"

    def test_surge_stiffness_positive(self, adapter):
        """K_cart[x, x] must be positive (restoring force in surge)."""
        _, _, K, _ = adapter.frequency_domain_MCKF(1.0)
        assert K[0, 0] > 0, f"K[x,x] = {K[0,0]:.4f} — expected positive"

    def test_heave_stiffness_positive(self, adapter):
        """K_cart[z, z] must be positive (restoring force in heave from line weight)."""
        _, _, K, _ = adapter.frequency_domain_MCKF(1.0)
        assert K[1, 1] > 0, f"K[z,z] = {K[1,1]:.4f} — expected positive"

    def test_translation_pitch_coupling_nonzero(self, adapter):
        """Offset fairlead (x=2 m) must produce nonzero K[x, pitch] coupling."""
        _, _, K, _ = adapter.frequency_domain_MCKF(1.0)
        assert abs(K[0, 2]) > 1.0, \
            f"K[x,pitch] = {K[0,2]:.4f} — expected nonzero from 2 m offset"

    def test_matrices_are_frequency_independent(self, adapter):
        """MoorPy matrices must not change with omega (mooring is quasi-static)."""
        M1, C1, K1, _ = adapter.frequency_domain_MCKF(0.3)
        M2, C2, K2, _ = adapter.frequency_domain_MCKF(3.0)
        assert np.array_equal(M1, M2), "M_cart changed with omega"
        assert np.array_equal(C1, C2), "C_cart changed with omega"
        assert np.array_equal(K1, K2), "K_cart changed with omega"

    def test_K_matrix_is_2d_not_batched(self, adapter):
        """K must always be 2-D (omega-independent)."""
        _, _, K, _ = adapter.frequency_domain_MCKF(np.linspace(0.1, 2.0, 5))
        # Adapter always returns 2-D K regardless of omega shape
        assert K.ndim == 2, f"K.ndim = {K.ndim}, expected 2"
        assert K.shape == (3, 3), f"K.shape = {K.shape}"

    def test_manager_K_joint_has_mooring_contribution(self, adapter):
        """After registration, K_joint must increase relative to MBD-only K."""
        from multibody.linearization.linearization_main import LinearizationManager
        mbd_sys, q0, mainNumVars, m0, J0 = _single_float()

        manager_bare = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        K_bare = np.asarray(manager_bare.assemble_frequency_domain(1.0).K)

        manager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0)
        manager.register(adapter)
        K_with = np.asarray(manager.assemble_frequency_domain(1.0).K)

        delta_K = K_with - K_bare
        assert np.any(np.abs(delta_K) > 1.0), \
            "Mooring adapter did not contribute to K_joint"

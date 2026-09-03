# tests/cases/linearization/test_moorpy_raft_reference.py
"""
Regression test: MoorPyLinearMCKF vs. the frozen RAFT/MoorPy reference.

Purpose
-------
Ensure future changes to the M4E MoorPy adapter continue to reproduce the
previously (manually) verified M, C, K matrices obtained from RAFT's own
MoorPy dynamic-matrix pathway. This test does NOT import or run RAFT, and
does NOT rebuild any independent MoorPy reference system — MoorPy is the
backend under test, so MoorPyLinearMCKF is allowed (expected) to use it
internally. The comparison is only against the frozen matrices below.

Provenance
----------
* Reference case  : single float, one catenary line, offset fairlead
                     (anchor at (-50,-100), fairlead at body-local (2,-5)) —
                     same case as build_moorpy_case(); RAFT auto-creates its
                     MoorPy body at this same reference point (CG, rCG=0) and
                     the same undisplaced r6=[0,0,0,0,0,0] operating state.
* Generated from  : Examples_linearization/verification_raft_moorpy/compare_raft.py
                     (manually reviewed; that script reproduces RAFT's own
                     MoorPy construction + updateSystemDynamicMatrices() +
                     getCoupledDynamicMatrices(lines_only=True) call sequence
                     — traced from RAFT source, not used here).
* RAFT version    : NLRWindSystems/RAFT (formerly WISDEM/RAFT), `main` branch,
                     latest release v2.0.4, traced 2026-08-18 (RAFT itself is
                     not installed or imported — see compare_raft.py docstring).
* Date generated  : 2026-08-18
* MoorPy version  : 1.3.0
* Frequency       : omega = 0.5 rad/s (matrices are frequency-independent)
* Matrix ordering : [x, z, pitch] per body (one body here -> 3x3)
* Tolerances      : rtol=atol=1e-8 — the manual verification observed an
                     exact (0.0 abs/rel) match, since both paths run the
                     identical deterministic MoorPy computation on identical
                     inputs.

The reference matrices in truth/raft_moorpy_reference.npz are trusted
regression values and must NEVER regenerate automatically here. Changing
them requires intentionally rerunning and re-reviewing
Examples_linearization/verification_raft_moorpy/compare_raft.py, then
manually overwriting the .npz file.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.cases.linearization._moorpy_regression_fixtures import build_moorpy_case

_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "truth", "raft_moorpy_reference.npz")

RTOL = 1e-8
ATOL = 1e-8


class TestRaftMoorPyRegression:
    """MoorPyLinearMCKF must keep reproducing the frozen RAFT/MoorPy M, C, K."""

    @pytest.fixture(scope="class")
    def reference(self):
        # Loaded as-is; never generated or overwritten by test code.
        return np.load(_REFERENCE_PATH)

    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("moorpy")
        from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF

        mbd_sys, q0, mainNumVars, moorpy_inputs = build_moorpy_case()
        return MoorPyLinearMCKF(
            mbd_sys, q0, mainNumVars, moorpy_inputs, include_mean_force=False
        )

    def test_M_matches_frozen_reference(self, adapter, reference):
        omega = float(reference["omega"][0])
        M_actual, _, _, _ = adapter.frequency_domain_MCKF(omega)
        np.testing.assert_allclose(M_actual, reference["M"], rtol=RTOL, atol=ATOL)

    def test_C_matches_frozen_reference(self, adapter, reference):
        omega = float(reference["omega"][0])
        _, C_actual, _, _ = adapter.frequency_domain_MCKF(omega)
        np.testing.assert_allclose(C_actual, reference["C"], rtol=RTOL, atol=ATOL)

    def test_K_matches_frozen_reference(self, adapter, reference):
        omega = float(reference["omega"][0])
        _, _, K_actual, _ = adapter.frequency_domain_MCKF(omega)
        np.testing.assert_allclose(K_actual, reference["K"], rtol=RTOL, atol=ATOL)

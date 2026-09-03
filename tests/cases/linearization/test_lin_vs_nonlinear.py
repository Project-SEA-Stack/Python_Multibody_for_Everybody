# -*- coding: utf-8 -*-
"""
Regression tests: linearized M4E solution vs full nonlinear M4E solution.

For each scenario in :data:`tests.cases.linearization.catalog.LIN_VS_NL_CASES`
the model is integrated twice (full equations of motion and equations
linearized about the rest configuration) and the two body-CG trajectories
are compared with :func:`tests.metrics.nrmse_per_body`.

These tests guard against regressions in :mod:`multibody.linearization` and
in the building blocks it depends on (Jacobians, mass matrices, generalized
force assembly).
"""

from __future__ import annotations

import pytest

from tests.cases.linearization.catalog import LIN_VS_NL_CASES, LinVsNLCaseEntry
from tests.cases.linearization.runner import run_lin_vs_nl
from tests.metrics import nrmse_per_body


POSITION_NRMSE_TOL = 5.0e-3
VELOCITY_NRMSE_TOL = 1.0e-2


@pytest.mark.parametrize(
    "entry",
    LIN_VS_NL_CASES,
    ids=[c.case_id for c in LIN_VS_NL_CASES],
)
def test_linear_matches_nonlinear(entry: LinVsNLCaseEntry) -> None:
    """Linearized trajectory must match the nonlinear one in the linear regime."""
    res = run_lin_vs_nl(entry.module)

    pos = nrmse_per_body(res.r_lin, res.r_nl, res.t)
    vel = nrmse_per_body(res.v_lin, res.v_nl, res.t)

    assert (pos["NRMSE_per_body"] < POSITION_NRMSE_TOL).all(), (
        f"[{entry.case_id}] position NRMSE per body = "
        f"{pos['NRMSE_per_body']}, tol = {POSITION_NRMSE_TOL}"
    )
    assert (vel["NRMSE_per_body"] < VELOCITY_NRMSE_TOL).all(), (
        f"[{entry.case_id}] velocity NRMSE per body = "
        f"{vel['NRMSE_per_body']}, tol = {VELOCITY_NRMSE_TOL}"
    )

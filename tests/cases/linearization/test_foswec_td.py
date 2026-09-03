# -*- coding: utf-8 -*-
"""
Regression test: M4E foswec linearized time-domain trajectory vs WEC-Sim.

Integrates the linearized foswec model at the principal wave frequency
(``waves.omega.values[0]``) for a short duration and compares body-CG
positions against the WEC-Sim reference trajectories in
``Linearization_valid/foswec/results_wecSim.mat``.

Only positions are asserted: previous comparison data on disk shows that
body pitch angles disagree by ~70× between M4E linear and WEC-Sim — a
known issue tracked separately.  This test is also marked ``slow`` because
each run spends a few seconds compiling the operating point and integrating.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.cases.linearization.case_foswec import WECSIM_MAT
from tests.cases.linearization.foswec_runner import run_foswec_time_domain
from tests.cases.linearization.wecsim_truth import load_wecsim
from tests.metrics import nrmse_per_body


TSPAN_S = 60.0
DT_S = 0.05
RAMP_S = 30.0
POSITION_NRMSE_TOL = 1.0e-4


def _resample_along_time(t_target, t_source, y_source):
    """Interpolate a (nt, ...) array along axis 0 onto ``t_target``."""
    y = np.asarray(y_source)
    flat = y.reshape(y.shape[0], -1)
    out = np.empty((t_target.size, flat.shape[1]), dtype=flat.dtype)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(t_target, t_source, flat[:, k])
    return out.reshape((t_target.size,) + y.shape[1:])


@pytest.mark.slow
def test_foswec_td_position_matches_wecsim():
    """M4E linearized positions must match WEC-Sim positions in steady state."""
    m4e = run_foswec_time_domain(tspan=TSPAN_S, dt=DT_S, ramp_T=RAMP_S)
    truth = load_wecsim(WECSIM_MAT)

    # Truth is sampled at dt=0.1 over 1000 s; restrict to the test window
    # past the ramp so we compare steady-state behavior.
    t0 = RAMP_S
    t1 = float(m4e.t[-1])
    mask_m4e = (m4e.t >= t0) & (m4e.t <= t1)
    mask_ref = (truth.t >= t0) & (truth.t <= t1)

    t_grid = m4e.t[mask_m4e]
    r_m4e = m4e.r[mask_m4e]
    r_ref = _resample_along_time(t_grid, truth.t[mask_ref], truth.r[mask_ref])

    metrics = nrmse_per_body(r_m4e, r_ref, t_grid)
    assert (metrics["NRMSE_per_body"] < POSITION_NRMSE_TOL).all(), (
        f"foswec position NRMSE per body = {metrics['NRMSE_per_body']}, "
        f"tol = {POSITION_NRMSE_TOL}"
    )

# -*- coding: utf-8 -*-
"""
Regression test: M4E foswec time-domain RAO vs MATLAB/WEC-Sim RAO.

Methodology matches the MATLAB reference generation script:

  1. Integrate the linearized M4E system for ``tspan`` seconds at each
     wave frequency with a smooth amplitude ramp of ``ramp_T`` seconds.
  2. Extract the steady-state window ``t >= ss_start``.
  3. Compute the half-peak-to-peak amplitude per joint DOF::

         A_resp = 0.5 * (max(x_ss) - min(x_ss))
         RAO    = A_resp / A_wave

  4. Compare against ``tests/cases/linearization/truth/RAO_matlab.txt``.

DOF column order (both M4E and reference file):
  omega | platform_surge | platform_heave | platform_pitch
        | flap1_pitch | flap2_pitch

Observed maximum relative errors (1000 s run, last 100 s steady state):
  surge          < 0.2 %
  heave          < 0.5 %
  platform_pitch < 0.3 %
  flap1_pitch    < 3.4 %
  flap2_pitch    < 3.0 %

All tests are marked ``slow`` because each of the five wave-frequency
integrations runs for 1000 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.cases.linearization.foswec_runner import run_foswec_rao_td
from tests.cases.linearization.wecsim_truth import load_rao
from tests.cases.linearization.case_foswec import RAO_MATLAB_TXT


# --- simulation parameters (must match MATLAB generation script) ---
TSPAN_S  = 1000.0
DT_S     = 0.05
RAMP_S   = 30.0
SS_START = 900.0     # analyse last 100 s of the 1000 s run

# --- per-DOF relative tolerances (5× observed max error) ---
# RAO_matlab.txt columns: omega(0), surge(1), heave(2), pitch(3), flap1(4), flap2(5)
DOF_TOLERANCES = {
    "platform_surge":  1.0e-2,
    "platform_heave":  3.0e-2,
    "platform_pitch":  1.5e-2,
    "flap1_pitch":     2.0e-1,
    "flap2_pitch":     2.0e-1,
}


@pytest.fixture(scope="module")
def foswec_rao_td():
    """Compute M4E time-domain RAO once for the whole module."""
    return run_foswec_rao_td(
        tspan=TSPAN_S, dt=DT_S, ramp_T=RAMP_S, ss_start=SS_START
    )


@pytest.fixture(scope="module")
def matlab_rao():
    """Load MATLAB/WEC-Sim RAO once for the whole module."""
    return load_rao(RAO_MATLAB_TXT)


@pytest.mark.slow
def test_foswec_rao_frequencies(foswec_rao_td, matlab_rao):
    """Wave frequencies in the M4E sweep must match the reference grid."""
    np.testing.assert_allclose(
        foswec_rao_td.omega, matlab_rao[:, 0], rtol=1e-6
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "m4e_col, matlab_col, name",
    [
        (0, 1, "platform_surge"),
        (1, 2, "platform_heave"),
        (2, 3, "platform_pitch"),
        (3, 4, "flap1_pitch"),
        (4, 5, "flap2_pitch"),
    ],
)
def test_foswec_rao_dofs(foswec_rao_td, matlab_rao, m4e_col, matlab_col, name):
    """M4E time-domain RAO must match MATLAB within per-DOF tolerance."""
    m4e    = foswec_rao_td.rao[:, m4e_col]
    ref    = matlab_rao[:, matlab_col]
    tol    = DOF_TOLERANCES[name]
    rel    = np.abs(m4e - ref) / np.maximum(np.abs(ref), 1e-12)
    assert (rel < tol).all(), (
        f"{name}: relative error per omega = {rel}, tol = {tol}"
    )

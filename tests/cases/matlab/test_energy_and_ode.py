# -*- coding: utf-8 -*-
"""
Energy conservation validation against MATLAB ODE reference.

For each of the eight canonical multibody examples the test:
  1. Loads the MATLAB energy trace from
     ``tests/cases/matlab/truth/energy/Em_example{i}.mat``
  2. Integrates the same system with :meth:`~multibody.MbdSystem.integrate`
  3. Evaluates total mechanical energy at every timestep via
     ``MbdSystem.Energy_func``
  4. Asserts that the RMSE between the two energy traces is below
     :data:`ENERGY_RMSE_TOL`.

Typical RMSE values (from the reference run):
  examples 1–3, 5, 7 → 0.0 (exact)
  example 4 → ~1.1e-8,  example 6 → ~9.1e-6,  example 8 → ~1.5e-5
"""

import os

import numpy as np
import pytest
from scipy.io import loadmat

from multibody import MbdSystem
from tests.cases.catalog import MATLAB_CASES, load_case

_MAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "truth", "energy")
)

ENERGY_RMSE_TOL = 1.0e-4


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry",
    MATLAB_CASES,
    ids=[e.case_id for e in MATLAB_CASES],
)
def test_energy_conservation(entry):
    """Python energy trace must match the MATLAB reference within ENERGY_RMSE_TOL."""
    mat_path = os.path.join(_MAT_DIR, f"Em_example{entry.matlab_index}.mat")
    m = loadmat(mat_path)
    t_mat = m["time"].ravel()
    E_mat = m["Em_num"].astype("float32").ravel()

    ex = load_case(entry)
    mbdsys = MbdSystem.from_example(ex)
    mnv = np.hstack((ex.ic, ex.ForcesPointsNum, ex.BodyDataNum))
    sol = mbdsys.integrate(mnv, ex.m0, ex.J0, tspan=ex.tspan, dt=ex.TimeStep)

    t_py = np.array(sol.t)
    E_py = []
    for k, ti in enumerate(t_py):
        mnv_k = mnv.copy()
        mnv_k[: len(sol.y)] = sol.y[:, k]
        mnv_k[mbdsys.t_update] = ti
        E_py.append(mbdsys.Energy_func(*mnv_k, *ex.m0, *ex.J0))
    E_py = np.array(E_py)

    tol_grid = 0.001 * ex.TimeStep
    if t_mat.shape == t_py.shape and np.allclose(t_mat, t_py, atol=tol_grid, rtol=0.0):
        E_py_cmp = np.interp(t_mat, t_py, E_py).astype("float32")
    else:
        E_py_cmp = E_py

    rmse = float(np.sqrt(((E_py_cmp - E_mat) ** 2).mean()))
    assert rmse < ENERGY_RMSE_TOL, (
        f"Example {entry.matlab_index}: energy RMSE = {rmse:.3e} >= {ENERGY_RMSE_TOL}"
    )

# -*- coding: utf-8 -*-
"""
Runner helpers for linearization regression tests.

Builds an :class:`~multibody.MbdSystem`, integrates it twice (nonlinear and
linearized about the rest configuration) and returns both trajectories
sampled on the same time grid so callers can compare them with
:func:`tests.metrics.nrmse_per_body`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

import numpy as np

import multibody as mbd
from multibody import MbdSystem
from multibody.linearization.linearization_main import LinearizationManager


@dataclass
class LinearizationResult:
    """Per-case nonlinear vs linear trajectories.

    Attributes
    ----------
    t:
        Common time vector ``(N,)`` produced by the nonlinear integration.
    r_nl, v_nl:
        Body-CG positions and velocities from the nonlinear run, shape
        ``(N, nbody, 2)``.
    r_lin, v_lin:
        Same quantities from the linearized run, resampled onto ``t`` if
        the linear solver landed on a different grid.
    """

    t: np.ndarray
    r_nl: np.ndarray
    v_nl: np.ndarray
    r_lin: np.ndarray
    v_lin: np.ndarray


def _load_module(dotted: str) -> ModuleType:
    """Import (and force-reload) a model module so repeated test runs are isolated."""
    mod = importlib.import_module(dotted)
    return importlib.reload(mod)


def _resample(t_target: np.ndarray, t_source: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resample ``y`` (time on axis 0) from ``t_source`` to ``t_target``."""
    if t_source.shape == t_target.shape and np.allclose(t_source, t_target):
        return y
    flat = y.reshape(y.shape[0], -1)
    out = np.empty((t_target.shape[0], flat.shape[1]), dtype=flat.dtype)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(t_target, t_source, flat[:, k])
    return out.reshape((t_target.shape[0],) + y.shape[1:])


def run_lin_vs_nl(module: str) -> LinearizationResult:
    """Integrate the model nonlinearly and around its rest equilibrium.

    Parameters
    ----------
    module:
        Dotted path of a model module that defines the standard M4E
        scenario attributes (``joints``, ``types``, ``ic``, ``tspan``,
        ``TimeStep``, ``m0``, ``J0`` and friends).

    Returns
    -------
    LinearizationResult
        Nonlinear and linearized trajectories on the same time grid.
    """
    ex = _load_module(module)
    MBDsys = MbdSystem.from_example(ex)

    mnv = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))

    # --- nonlinear (integrate mutates the mainNumVars buffer in place) ---
    sol_nl = MBDsys.integrate(
        mnv.copy(), ex.m0, ex.J0, tspan=ex.tspan, dt=ex.TimeStep
    )
    r_nl, v_nl, _, _ = mbd.evaluate_trajectories(MBDsys, sol_nl, mnv.copy())

    # --- linearized about q=0, qd=0 ---
    q0 = np.zeros(len(MBDsys.Q))
    LM = LinearizationManager(MBDsys, q0, mnv.copy(), ex.m0, ex.J0)
    LM.compile_operating_point(
        0.0, eq_tol=1e-6, eq_mode="zero_if_small", ramp_T=0.0
    )
    sol_lin = LM.integrate_linear_system(
        tspan=ex.tspan,
        dt=ex.TimeStep,
        method="RK45",
        rtol=1e-8,
        atol=1e-11,
    )
    # LinearizationManager returns the perturbation; add the operating point back.
    offset = np.hstack((q0, np.zeros_like(q0))).reshape(-1, 1)
    sol_lin.y = sol_lin.y + offset

    r_lin, v_lin, _, _ = mbd.evaluate_trajectories(MBDsys, sol_lin, mnv.copy())

    t_nl = np.asarray(sol_nl.t)
    t_lin = np.asarray(sol_lin.t)
    r_nl = np.asarray(r_nl)
    v_nl = np.asarray(v_nl)
    r_lin = _resample(t_nl, t_lin, np.asarray(r_lin))
    v_lin = _resample(t_nl, t_lin, np.asarray(v_lin))

    return LinearizationResult(
        t=t_nl, r_nl=r_nl, v_nl=v_nl, r_lin=r_lin, v_lin=v_lin
    )

# -*- coding: utf-8 -*-
"""
Runner helpers for the foswec WEC-Sim comparison tests.

Three entry points are exposed:

* :func:`run_foswec_rao_td` integrates the linearized system at each wave
  frequency and extracts the half-peak-to-peak amplitude in the steady-state
  window, matching the MATLAB/WEC-Sim RAO generation methodology.
* :func:`run_foswec_rao` solves the frequency-domain impedance problem at
  each wave frequency (kept for reference; not used by the main RAO test).
* :func:`run_foswec_time_domain` integrates the linearized equations of
  motion at the principal wave frequency and returns body-CG positions
  and pitch angles vs time.

All helpers load the pre-computed BEM dataset
``tests/cases/linearization/foswec/hydroData/bem_1025.nc`` so the heavy
Capytaine solve is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import multibody as mbd
from multibody import MbdSystem
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization.hydro_linear_mckf import HydroLinearMCKF

from tests.cases.linearization.case_foswec import (
    ex,
    waves,
    body_inputs,
    m0,
    J0,
    BEM_FOLDER,
    BEM_FILE,
)


@dataclass
class FoswecRAOResult:
    """Per-frequency magnitude RAO."""

    omega: np.ndarray            # (nfreq,)
    rao: np.ndarray              # (nfreq, ndof) — |q_hat| / wave_amplitude


@dataclass
class FoswecTDResult:
    """Time-domain linear response."""

    t: np.ndarray                # (nt,)
    r: np.ndarray                # (nt, nbody, 2)  CG positions
    angle: np.ndarray            # (nt, nbody)     body pitch angles
    omega0: float                # primary wave frequency [rad/s]
    ramp_T: float                # ramp duration [s]


def _build_system() -> Tuple[MbdSystem, LinearizationManager, np.ndarray, np.ndarray]:
    """Assemble :class:`MbdSystem`, register hydro, return (system, manager, q0, mnv)."""
    MBDsys = MbdSystem.from_example(ex)
    q0 = (MBDsys.ic - ex.ic)[: len(MBDsys.Q)]
    mnv = ex.ic.copy()

    LM = LinearizationManager(
        MBDsys, q0, mnv.copy(), m0, J0, print_sym_matrices=False
    )
    hyd = HydroLinearMCKF(
        MBDsys,
        (mnv.copy(), m0, J0),
        is_2D=False,
        omega_r=waves.omega.values,
        body_inputs=body_inputs,
        wave_amplitude=waves.attrs["Amplitude (m)"],
        equilibrium_pos=q0,
        load_dir=BEM_FOLDER,
        file_name=BEM_FILE,
        rho=1025,
    )
    LM.register(hyd)
    return MBDsys, LM, q0, mnv


def run_foswec_rao_td(
    tspan: float = 1000.0,
    dt: float = 0.05,
    ramp_T: float = 30.0,
    ss_start: float = 900.0,
) -> FoswecRAOResult:
    """Time-domain RAO by half-peak-to-peak amplitude extraction in steady state.

    For each wave frequency in the BEM sweep the linearized system is
    integrated for ``tspan`` seconds with an amplitude ramp of ``ramp_T``
    seconds.  The response amplitude per joint DOF is estimated as half
    the peak-to-peak value in the steady-state window ``[ss_start, tspan]``
    and divided by the wave amplitude to give the RAO.

    This matches the MATLAB/WEC-Sim RAO methodology::

        A_resp = 0.5 * (max(x_ss) - min(x_ss))
        RAO    = A_resp / A_wave

    DOF ordering in the returned :class:`FoswecRAOResult` matches the
    ``RAO_matlab.txt`` columns:
    platform surge, platform heave, platform pitch, flap1 pitch, flap2 pitch.

    Parameters
    ----------
    tspan:
        Total simulation time [s].  Default 1000 s matches the MATLAB run.
    dt:
        Maximum output time step [s].
    ramp_T:
        Duration of the smooth excitation ramp [s].
    ss_start:
        Start of the steady-state window [s].  Default 900 s (last 10 %
        of the default 1000 s run, matching the MATLAB ``time > 900`` mask).
    """
    MBDsys, LM, q0, mnv = _build_system()
    nq = len(MBDsys.Q)
    amp = waves.attrs["Amplitude (m)"]
    omega_arr = waves.omega.values

    rao = []
    for omega in omega_arr:
        LM.compile_operating_point(
            float(omega), eq_tol=1e-6, eq_mode="warn", ramp_T=float(ramp_T)
        )
        sol = LM.integrate_linear_system(
            tspan=float(tspan),
            dt=float(dt),
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        t = np.asarray(sol.t)
        # Joint displacement perturbations from equilibrium.
        # The equilibrium offset q0 is constant and cancels in max-min,
        # so amplitude extraction does not require adding it back.
        q_pert = sol.y[:nq, :]
        mask = t >= float(ss_start)
        q_ss = q_pert[:, mask]
        A = 0.5 * (q_ss.max(axis=1) - q_ss.min(axis=1))
        rao.append(A / amp)

    return FoswecRAOResult(omega=np.asarray(omega_arr), rao=np.asarray(rao))


def run_foswec_rao() -> FoswecRAOResult:
    """Frequency-domain RAO via impedance matrix inversion (reference only).

    Kept for diagnostic comparison against :func:`run_foswec_rao_td`.
    The main RAO test uses the time-domain approach to match the MATLAB
    methodology.
    """
    _, LM, _, _ = _build_system()
    omega_arr = waves.omega.values
    amp = waves.attrs["Amplitude (m)"]

    rao = []
    for omega in omega_arr:
        fd = LM.assemble_frequency_domain(omega)
        Z = -omega**2 * fd.M + 1j * omega * fd.C + fd.K
        qhat = np.linalg.solve(Z, fd.Fhat)
        rao.append(np.abs(qhat).squeeze() / amp)

    return FoswecRAOResult(omega=np.asarray(omega_arr), rao=np.asarray(rao))


def run_foswec_time_domain(
    tspan: Optional[float] = None,
    dt: Optional[float] = None,
    ramp_T: float = 30.0,
) -> FoswecTDResult:
    """Integrate the linearized foswec system in time at the lowest wave frequency.

    Parameters
    ----------
    tspan, dt:
        Simulation duration and time step.  Default to the values declared
        in the M4E scenario (typically 1000 s / 0.01 s); pass smaller values
        for fast tests.
    ramp_T:
        Smooth ramp duration applied to the excitation.
    """
    MBDsys, LM, q0, mnv = _build_system()

    omega0 = float(waves.omega.values[0])
    LM.compile_operating_point(
        omega0, eq_tol=1e-6, eq_mode="warn", ramp_T=float(ramp_T)
    )

    sol = LM.integrate_linear_system(
        tspan=float(tspan if tspan is not None else ex.tspan),
        dt=float(dt if dt is not None else ex.TimeStep),
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    offset = np.hstack((q0, np.zeros_like(q0))).reshape(-1, 1)
    sol.y = sol.y + offset
    r, _, angle, _ = mbd.evaluate_trajectories(MBDsys, sol, mnv.copy())

    return FoswecTDResult(
        t=np.asarray(sol.t),
        r=np.asarray(r),
        angle=np.asarray(angle),
        omega0=omega0,
        ramp_T=float(ramp_T),
    )

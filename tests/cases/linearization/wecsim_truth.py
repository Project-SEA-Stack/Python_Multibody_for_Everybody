# -*- coding: utf-8 -*-
"""
WEC-Sim reference data loader for the foswec validation test.

Loads ``Linearization_valid/foswec/results_wecSim.mat`` directly so no
``.npy`` cache is required.  The .mat file contains the variables produced
by the WEC-Sim simulation:

* ``t_wec``     — time vector, shape ``(nt, 1)`` [s]
* ``r_wec``     — body-CG positions, shape ``(nt, nbody, 2)`` [m]
* ``angle_wec`` — body pitch angles, shape ``(nt, nbody)`` [rad]
* ``Fexc``     — excitation forces (not used here)
* ``Ftot``     — total forces (not used here)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class WecSimResult:
    """Bundle of WEC-Sim reference trajectories."""

    t: np.ndarray        # (nt,)
    r: np.ndarray        # (nt, nbody, 2)
    angle: np.ndarray    # (nt, nbody)


def load_wecsim(mat_path: str) -> WecSimResult:
    """Read a WEC-Sim results .mat file and return the trajectories."""
    data = sio.loadmat(mat_path)
    t = np.asarray(data["t_wec"]).reshape(-1)
    r = np.asarray(data["r_wec"])
    angle = np.asarray(data["angle_wec"])
    return WecSimResult(t=t, r=r, angle=angle)


def load_rao(txt_path: str) -> np.ndarray:
    """Read a RAO comparison .txt file (header + omega + DOF columns).

    Returns
    -------
    ndarray of shape ``(nfreq, ncols)``
        First column is omega, remaining columns are the magnitude RAO
        for each DOF in the order declared in the header.
    """
    return np.loadtxt(txt_path, delimiter=",", skiprows=1)

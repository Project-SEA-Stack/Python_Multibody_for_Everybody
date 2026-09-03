# -*- coding: utf-8 -*-
"""Parse a PyChrono ``trajectory.csv`` and return ``(t, r, v)`` arrays.

The CSV is the ground-truth artifact stored under
``tests/cases/yaml/truth/<case_id>/trajectory.csv``.

:func:`read_trajectory_csv` is the single place where the CSV format is
understood; :mod:`tests.cases.yaml.truth` calls it directly.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


def read_trajectory_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a tidy ``trajectory.csv`` exported by the C++ Chrono loop.

    Returns
    -------
    t : ndarray, shape (T,)
        Time vector (seconds).
    r : ndarray, shape (T, NB, 2)
        Body center-of-mass positions, columns ``(x, z)``.
    v : ndarray, shape (T, NB, 2)
        Body center-of-mass velocities, columns ``(x, z)``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the pivoted matrices contain NaNs (indicating uneven sampling).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values(["time", "body_index"]).reset_index(drop=True)

    t = df.pivot(index="time", columns="body_index", values="time").to_numpy()[:, 0]
    rx = df.pivot(index="time", columns="body_index", values="pos_x").to_numpy()
    rz = df.pivot(index="time", columns="body_index", values="pos_z").to_numpy()
    vx = df.pivot(index="time", columns="body_index", values="vel_x").to_numpy()
    vz = df.pivot(index="time", columns="body_index", values="vel_z").to_numpy()

    if np.any(np.isnan(rx) | np.isnan(rz) | np.isnan(vx) | np.isnan(vz)):
        raise ValueError(
            f"NaNs in pivoted trajectory data from {csv_path!r}; "
            "check that all bodies share the same sample times."
        )

    r = np.stack([rx, rz], axis=-1)
    v = np.stack([vx, vz], axis=-1)
    return t, r, v

# -*- coding: utf-8 -*-
"""Load PyChrono reference (truth) trajectories for YAML validation cases.

The ground truth for every YAML case is the ``trajectory.csv`` stored in
``tests/cases/yaml/truth/<case_id>/``.  The loader rebuilds the
``(t, r, v)`` arrays from CSV on every call via
:func:`tests.cases.yaml.read_csv.read_trajectory_csv`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .catalog import YamlCaseEntry, get_entry
from .read_csv import read_trajectory_csv


@dataclass(frozen=True)
class ChronoTruth:
    """Container for a single PyChrono reference trajectory.

    Shapes:
      * ``t`` — ``(T,)``
      * ``r`` — ``(T, NB, 2)`` body CG positions, columns ``(x, z)``
      * ``v`` — ``(T, NB, 2)`` body CG velocities, columns ``(x, z)``
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray


def load_chrono_truth(case: YamlCaseEntry | str) -> ChronoTruth:
    """Build the PyChrono reference for ``case`` from ``trajectory.csv``."""
    entry = get_entry(case)
    folder = os.path.join(os.path.dirname(__file__), "truth", entry.case_id)
    csv = os.path.join(folder, "trajectory.csv")

    if not os.path.isfile(csv):
        raise FileNotFoundError(
            f"{entry.case_id}: trajectory.csv not found in {folder}"
        )

    t, r, v = read_trajectory_csv(csv)
    return ChronoTruth(t=t, r=r, v=v)


def _select_bodies(arr: np.ndarray, bodies: Optional[tuple]) -> np.ndarray:
    """Return the slice of ``arr`` along axis 1 selected by ``bodies``.

    Passing ``None`` returns ``arr`` unchanged.
    """
    if bodies is None:
        return arr
    return arr[:, list(bodies), :]

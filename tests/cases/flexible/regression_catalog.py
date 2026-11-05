# -*- coding: utf-8 -*-
"""
Catalog of FSM self-consistency (regression) cases.

Unlike ``tests/cases/chrono/catalog.py`` (which freezes an INDEPENDENT
PyChrono trajectory as ground truth, to validate FSM's physics against
third-party software) or ``test_flex_distribution.py``'s PyChrono/FEM
cross-validation tests, this catalog freezes FSM's OWN previously-computed
trajectory as a baseline, to catch any unintended change in FSM's output
(a solver change, a refactor, a dependency upgrade) between runs -- a
classic "golden master"/characterization test, not an accuracy check.

Each entry points at a frozen model in ``tests/cases/flexible/models/`` that
also has an independent PyChrono cross-validation on record (see the
``_pychrono_*_reference.py`` scripts in ``pychrono_backup/``),
so the frozen baseline is known-good, not just "whatever FSM happened to output".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FlexRegressionEntry:
    """One FSM self-consistency regression scenario.

    Parameters
    ----------
    case_id:
        Unique short identifier used as the pytest node ID.
    example_path:
        Path (relative to the repository root) of the frozen FSM model to
        load and integrate, e.g.
        ``tests/cases/flexible/models/flex_tapered_pendulum.py``.
    truth_dir:
        Path (relative to the repository root) to the local truth folder
        holding the stored FSM baseline arrays (``t.npy``, ``r.npy``,
        ``v.npy``, ``angles.npy``).
    description:
        Human-readable summary of the scenario.
    """

    case_id: str
    example_path: str
    truth_dir: str
    description: str


NRMSE_TOL = 1e-4


FLEX_REGRESSION_CASES: Tuple[FlexRegressionEntry, ...] = (
    FlexRegressionEntry(
        case_id="tapered_pendulum",
        example_path="tests/cases/flexible/models/flex_tapered_pendulum.py",
        truth_dir="tests/cases/flexible/truth/tapered_pendulum",
        description=(
            "Gravity-driven tapered bending pendulum (EXAMPLE Q), "
            "cross-validated against real PyChrono: max errors root "
            "0.073 rad, tip_x 3.4%, tip_z 7.9% of range."
        ),
    ),
    FlexRegressionEntry(
        case_id="double_pendulum_moderate",
        example_path="tests/cases/flexible/models/flex_double_pendulum_moderate.py",
        truth_dir="tests/cases/flexible/truth/double_pendulum_moderate",
        description=(
            "Gravity-driven flexible double pendulum (EXAMPLE R), "
            "cross-validated against real PyChrono: KE+PE_grav+elastic "
            "relative energy drift ~2e-5, tip position errors ~8-16% of range."
        ),
    ),
)


def get_entry(case_id) -> FlexRegressionEntry:
    if isinstance(case_id, FlexRegressionEntry):
        return case_id
    for entry in FLEX_REGRESSION_CASES:
        if entry.case_id == case_id:
            return entry
    raise KeyError(f"Unknown flex regression case_id: {case_id!r}")

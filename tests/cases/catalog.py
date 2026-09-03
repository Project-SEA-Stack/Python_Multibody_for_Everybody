# -*- coding: utf-8 -*-
"""
Shared catalog of multibody regression cases.

The eight cases previously stored under ``Pytests/Example{1..8}.py`` are
the canonical inputs for the symbolic-kinematics check
(:mod:`tests.cases.matlab.test_automated_validation`) and the energy
conservation check (:mod:`tests.cases.matlab.test_energy_and_ode`).
Their definitions now live in :mod:`tests.cases.matlab.case_0{1..8}` and
are registered here.

Use :func:`load_case` from any script or future pytest fixture to obtain
the case module by integer index (1..8) or by string id ("matlab_01" ...).
The MATLAB reference data keyed by ``matlab_index`` now lives under
``tests/cases/matlab/truth/symbolic/Example{i}.json`` and
``tests/cases/matlab/truth/energy/Em_example{i}.mat``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CaseEntry:
    """A reusable multibody case backed by a module that mirrors the
    historical ``Pytests/Example{i}.py`` surface used by
    ``MbdSystem.from_example``."""

    case_id: str
    module: str
    matlab_index: int
    description: str


MATLAB_CASES: Tuple[CaseEntry, ...] = (
    CaseEntry(
        case_id="matlab_01",
        module="tests.cases.matlab.case_01",
        matlab_index=1,
        description=(
            "10-body tree, R/P/F joint mix, full force suite "
            "(point forces, CG forces, tension and torsion springs/dampers, "
            "lambda-defined nonlinearities, BodyData symbols)."
        ),
    ),
    CaseEntry(
        case_id="matlab_02",
        module="tests.cases.matlab.case_02",
        matlab_index=2,
        description=(
            "12-body tree with branching, R/P/F joint mix, no time-varying "
            "forcing, gravity disabled (g=0)."
        ),
    ),
    CaseEntry(
        case_id="matlab_03",
        module="tests.cases.matlab.case_03",
        matlab_index=3,
        description="Pure 2-body double pendulum (R+R), no forces, gravity on.",
    ),
    CaseEntry(
        case_id="matlab_04",
        module="tests.cases.matlab.case_04",
        matlab_index=4,
        description="3-body P+R+R with exponential time-varying CG force, g=0.",
    ),
    CaseEntry(
        case_id="matlab_05",
        module="tests.cases.matlab.case_05",
        matlab_index=5,
        description="3-body R+P+P, no forces, non-zero initial revolute angle.",
    ),
    CaseEntry(
        case_id="matlab_06",
        module="tests.cases.matlab.case_06",
        matlab_index=6,
        description="4-body R+P+P+R, no forces, non-zero initial prismatic offset.",
    ),
    CaseEntry(
        case_id="matlab_07",
        module="tests.cases.matlab.case_07",
        matlab_index=7,
        description="5-body F+P+P+R+R, floating base coordinates.",
    ),
    CaseEntry(
        case_id="matlab_08",
        module="tests.cases.matlab.case_08",
        matlab_index=8,
        description="6-body branched R+P+P+P+R+R, diagonal prismatic direction.",
    ),
)


# Small subset that covers distinct multibody workflows without running
# the full eight-case battery. Selected to exercise:
#   * matlab_01 - full force/lambda surface and BodyData symbols
#   * matlab_03 - pure revolute baseline (gravity only)
#   * matlab_04 - prismatic joints with explicit time-varying CG force
#   * matlab_07 - floating-base coordinates
SMOKE_CASES: Tuple[CaseEntry, ...] = (
    MATLAB_CASES[0],
    MATLAB_CASES[2],
    MATLAB_CASES[3],
    MATLAB_CASES[6],
)


_BY_ID = {c.case_id: c for c in MATLAB_CASES}
_BY_INDEX = {c.matlab_index: c for c in MATLAB_CASES}


def get_entry(case):
    """Return the :class:`CaseEntry` for a string id or 1-based MATLAB index."""
    if isinstance(case, CaseEntry):
        return case
    if isinstance(case, int):
        return _BY_INDEX[case]
    return _BY_ID[case]


def load_case(case):
    """Import and return the case module backing ``case``.

    Accepts a :class:`CaseEntry`, an integer in ``1..8`` (the historical
    ``Example{i}`` index), or a case-id string like ``"matlab_03"``.
    The returned module exposes the same attribute surface that
    ``MbdSystem.from_example`` consumes today (``joints``, ``types``,
    ``ic``, ``tspan``, ``TimeStep``, ``Force``, ``Initial_Points``,
    ``m0``, ``J0``, ``g``, ``gVec``, ``ForcesPointsNum``,
    ``BodyDataNum`` and the symbolic helpers).
    """
    return importlib.import_module(get_entry(case).module)


__all__ = [
    "CaseEntry",
    "MATLAB_CASES",
    "SMOKE_CASES",
    "get_entry",
    "load_case",
]

# -*- coding: utf-8 -*-
"""
Catalog of linearization-validation cases.

Two test families live under ``tests/cases/linearization``:

1. **Linear vs nonlinear (self-consistency)** — :data:`LIN_VS_NL_CASES`
   The same M4E model is integrated twice: once with the full nonlinear
   equations of motion (``MbdSystem.integrate``) and once with the linearized
   equations around the equilibrium point (``LinearizationManager``).
   In the linear regime (small initial perturbation, no large excitation)
   the two trajectories must agree closely.

2. **wecSim time-domain and RAO comparison** — :data:`FOSWEC_TD_CASES`,
   :data:`FOSWEC_RAO_CASES`
   The foswec FOWT model is compared against WEC-Sim reference data stored
   in ``Linearization_valid/foswec/results_wecSim.mat`` (time-domain) and
   ``Linearization_valid/foswec/compare/RAO_matlab.txt`` (RAO).  Truth data
   is loaded on-the-fly; no ``.npy`` caches are stored.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Family 1: linear-vs-nonlinear regression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinVsNLCaseEntry:
    """One M4E linear-vs-nonlinear scenario.

    Parameters
    ----------
    case_id:
        Unique short identifier used as the pytest node ID.
    module:
        Dotted import path of the model module inside
        ``tests.cases.linearization``.
    description:
        Human-readable summary.
    """

    case_id: str
    module: str
    description: str


LIN_VS_NL_CASES: Tuple[LinVsNLCaseEntry, ...] = (
    LinVsNLCaseEntry(
        case_id="lin_dp",
        module="tests.cases.linearization.case_dp",
        description="Double pendulum, small IC, gravity only.",
    ),
    LinVsNLCaseEntry(
        case_id="lin_dp_advanced",
        module="tests.cases.linearization.case_dp_advanced",
        description="Double pendulum with linear torsion spring/damper, small IC.",
    ),
    LinVsNLCaseEntry(
        case_id="lin_slider_w_dpend",
        module="tests.cases.linearization.case_slider_w_dpend",
        description="Slider + double pendulum chain, small IC, no excitation.",
    ),
)


# ---------------------------------------------------------------------------
# Family 2: foswec vs WEC-Sim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoswecCaseEntry:
    """One foswec wecSim comparison.

    Parameters
    ----------
    case_id:
        Unique short identifier used as the pytest node ID.
    description:
        Human-readable summary.
    """

    case_id: str
    description: str


FOSWEC_TD_CASES: Tuple[FoswecCaseEntry, ...] = (
    FoswecCaseEntry(
        case_id="foswec_td",
        description=(
            "FOSWEC platform + 2 flaps under regular wave, M4E linear "
            "time-domain solution vs WEC-Sim reference."
        ),
    ),
)


FOSWEC_RAO_CASES: Tuple[FoswecCaseEntry, ...] = (
    FoswecCaseEntry(
        case_id="foswec_rao",
        description=(
            "FOSWEC platform + 2 flaps frequency-sweep, M4E linear RAO "
            "vs WEC-Sim/MATLAB RAO."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_lin_vs_nl_entry(case: Union[str, LinVsNLCaseEntry]) -> LinVsNLCaseEntry:
    """Return a :class:`LinVsNLCaseEntry` by ``case_id`` string or pass-through."""
    if isinstance(case, LinVsNLCaseEntry):
        return case
    for entry in LIN_VS_NL_CASES:
        if entry.case_id == case:
            return entry
    raise KeyError(f"Unknown linearization case: {case!r}")

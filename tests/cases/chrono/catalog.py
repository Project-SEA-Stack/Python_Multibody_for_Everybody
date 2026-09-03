# -*- coding: utf-8 -*-
"""
Catalog of Chrono-validated multibody regression cases.

Each entry pairs a Python model module (under ``tests.cases.chrono``) with a
folder inside ``tests/cases/chrono/truth/`` that contains the Chrono
ground-truth arrays ``t_chrono.npy``, ``r_chrono.npy``, and ``v_chrono.npy``.

The catalog drives :mod:`tests.cases.chrono.runner` and the pytest suite
``test_chrono_cases.py``.

Model modules are self-contained copies of the corresponding
``Validation2externalSoft/MBD_examples/`` scripts with the path adjusted to
resolve relative to the repository root.  The IC and tspan for each test
variant are stored here; :func:`run_chrono_case` applies them at runtime so
the same module covers multiple Chrono runs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ChronoCaseEntry:
    """One Chrono validation scenario.

    Parameters
    ----------
    case_id:
        Unique short identifier used as the pytest node ID.
    module:
        Dotted import path of the model module inside ``tests.cases.chrono``.
    ic:
        Initial conditions to override the module default, or ``None`` to use
        the module's own ``ic`` array.
    tspan:
        Simulation end time (seconds).  The runner calls
        ``MbdSystem.integrate(..., tspan=entry.tspan, ...)``.
    chrono_dir:
        Path (relative to the repository root) to the local truth folder
        ``tests/cases/chrono/truth/<case_id>/`` that holds the ``.npy``
        ground-truth files (``t_chrono.npy``, ``r_chrono.npy``, ``v_chrono.npy``).
    description:
        Human-readable summary of the scenario.
    """

    case_id: str
    module: str
    ic: Optional[Tuple[float, ...]]
    tspan: float
    chrono_dir: str
    description: str


CHRONO_CASES: Tuple[ChronoCaseEntry, ...] = (
    ChronoCaseEntry(
        case_id="dp_1s",
        module="tests.cases.chrono.case_dp",
        ic=(0., 0., -0.4, 0.),
        tspan=0.99,
        chrono_dir="tests/cases/chrono/truth/dp_1s",
        description="Double pendulum, small oscillation, ~1 s",
    ),
    ChronoCaseEntry(
        case_id="dp_10s",
        module="tests.cases.chrono.case_dp",
        ic=(0., 0., -0.4, 0.),
        tspan=10.,
        chrono_dir="tests/cases/chrono/truth/dp_10s",
        description="Double pendulum, small oscillation, 10 s",
    ),
    ChronoCaseEntry(
        case_id="dp_30s",
        module="tests.cases.chrono.case_dp",
        ic=(0., 0., -0.4, 0.),
        tspan=30.,
        chrono_dir="tests/cases/chrono/truth/dp_30s",
        description="Double pendulum, small oscillation, 30 s",
    ),
    ChronoCaseEntry(
        case_id="dp_large_30s",
        module="tests.cases.chrono.case_dp",
        ic=(0., 0., -4., 0.),
        tspan=30.,
        chrono_dir="tests/cases/chrono/truth/dp_large_30s",
        description="Double pendulum, large oscillation, 30 s",
    ),
    ChronoCaseEntry(
        case_id="sp_1s",
        module="tests.cases.chrono.case_sp",
        ic=(0., -0.4),
        tspan=0.99,
        chrono_dir="tests/cases/chrono/truth/sp_1s",
        description="Single gravity pendulum, ~1 s",
    ),
    ChronoCaseEntry(
        case_id="sp_10s",
        module="tests.cases.chrono.case_sp",
        ic=(0., -0.4),
        tspan=10.,
        chrono_dir="tests/cases/chrono/truth/sp_10s",
        description="Single gravity pendulum, 10 s",
    ),
    ChronoCaseEntry(
        case_id="ps_1s",
        module="tests.cases.chrono.case_ps",
        ic=None,
        tspan=0.99,
        chrono_dir="tests/cases/chrono/truth/ps_1s",
        description="Pendulum + prismatic slider, ~1 s",
    ),
    ChronoCaseEntry(
        case_id="sw_10s",
        module="tests.cases.chrono.case_sw",
        ic=None,
        tspan=10.,
        chrono_dir="tests/cases/chrono/truth/sw_10s",
        description="Sliding body with double pendulum, cosine drive, 10 s",
    ),
    ChronoCaseEntry(
        case_id="sw_300s",
        module="tests.cases.chrono.case_sw",
        ic=None,
        tspan=300.,
        chrono_dir="tests/cases/chrono/truth/sw_300s",
        description="Sliding body with double pendulum, cosine drive, 300 s",
    ),
)


def get_entry(case: Union[str, ChronoCaseEntry]) -> ChronoCaseEntry:
    """Return a :class:`ChronoCaseEntry` by ``case_id`` string or pass-through."""
    if isinstance(case, ChronoCaseEntry):
        return case
    for entry in CHRONO_CASES:
        if entry.case_id == case:
            return entry
    raise KeyError(f"Unknown chrono case: {case!r}")

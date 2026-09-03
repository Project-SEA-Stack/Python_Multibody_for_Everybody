# -*- coding: utf-8 -*-
"""
Foswec floating WEC scenario for linearization tests.

This module is a thin wrapper around
``tests.cases.linearization.foswec.M4E_inputs`` and
``tests.cases.linearization.foswec.hydro_inputs``: it re-exports the model
attributes that the runner needs while keeping tspan/TimeStep adjustable
so the test suite runs in reasonable time.

The heavy mesh generation in ``hydro_inputs`` runs once at import (~3 s);
the BEM is *not* recomputed because the runner asks
:class:`HydroLinearMCKF` to load the pre-computed dataset
``tests/cases/linearization/foswec/hydroData/bem_1025.nc``.
"""
from __future__ import annotations

import os

# Re-export the canonical M4E scenario as ``ex``.
from tests.cases.linearization.foswec import M4E_inputs as ex  # noqa: F401
from tests.cases.linearization.foswec.hydro_inputs import (  # noqa: F401
    waves,
    body_inputs,
    m0,
    J0,
)

#: Absolute path to the pre-computed BEM file.
BEM_FOLDER = os.path.join(os.path.dirname(__file__), "foswec", "hydroData")
BEM_FILE = "bem_1025.nc"

#: Path to the WEC-Sim reference (local truth copy).
WECSIM_MAT = os.path.join(os.path.dirname(__file__), "truth", "results_wecSim.mat")

#: Path to the MATLAB RAO reference (local truth copy).
RAO_MATLAB_TXT = os.path.join(os.path.dirname(__file__), "truth", "RAO_matlab.txt")

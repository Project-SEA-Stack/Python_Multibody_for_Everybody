# -*- coding: utf-8 -*-
"""
pytest suite for Chrono-validated multibody regression cases.

Each case in :data:`~tests.cases.chrono.catalog.CHRONO_CASES` is run and
its M4E trajectory is compared against the ``.npy`` Chrono reference stored
in ``Validation2externalSoft/results/<folder>/``.

Tolerances
----------
POSITION_NRMSE_TOL : 1e-2
    Maximum allowed NRMSE across bodies for position.
VELOCITY_NRMSE_TOL : 1.5e-2
    Maximum allowed NRMSE across bodies for velocity.
    Slightly looser than position to accommodate the large-oscillation case
    (``dp_large_30s``) where the velocity NRMSE reaches ~1.1e-2.
"""

import pytest

from tests.cases.chrono.catalog import CHRONO_CASES
from tests.cases.chrono.runner import compare_against_chrono

POSITION_NRMSE_TOL = 1.0e-2
VELOCITY_NRMSE_TOL = 1.5e-2


@pytest.mark.parametrize(
    "entry",
    CHRONO_CASES,
    ids=[c.case_id for c in CHRONO_CASES],
)
def test_chrono_case(entry):
    metrics = compare_against_chrono(entry)
    pos_nrmse = metrics["pos_metrics"]["NRMSE_per_body"].max()
    vel_nrmse = metrics["vel_metrics"]["NRMSE_per_body"].max()
    assert pos_nrmse < POSITION_NRMSE_TOL, (
        f"{entry.case_id}: pos NRMSE {pos_nrmse:.2e} >= {POSITION_NRMSE_TOL}"
    )
    assert vel_nrmse < VELOCITY_NRMSE_TOL, (
        f"{entry.case_id}: vel NRMSE {vel_nrmse:.2e} >= {VELOCITY_NRMSE_TOL}"
    )

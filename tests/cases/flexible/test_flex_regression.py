# -*- coding: utf-8 -*-
"""
pytest suite for FSM flexible-model self-consistency (regression) cases.

Unlike ``test_flex_distribution.py``'s PyChrono/FEM cross-validation tests
(which check FSM's physics against an INDEPENDENT reference), these tests
check FSM's OWN freshly-computed trajectory against a FROZEN baseline that
was captured at a moment when that same case was ALSO confirmed to match
its independent PyChrono reference (see the ``_pychrono_*_reference.py``
scripts in ``pychrono_backup/`` and ``regression_catalog.py`` for each case's
recorded PyChrono agreement). This catches regressions --
an unintended change in FSM's output from a refactor, dependency upgrade,
or solver change -- cheaply, deterministically, and without needing
PyChrono at all.

Run with::

    pytest tests/cases/flexible/test_flex_regression.py

To deliberately refresh a baseline after a verified, intentional change
(re-confirm against PyChrono first)::

    python -m tests.cases.flexible.regression_runner --update <case_id>
"""
import pytest

from tests.cases.flexible.regression_catalog import FLEX_REGRESSION_CASES, NRMSE_TOL
from tests.cases.flexible.regression_runner import compare_against_truth

@pytest.mark.parametrize(
    "entry",
    FLEX_REGRESSION_CASES,
    ids=[c.case_id for c in FLEX_REGRESSION_CASES],
)
def test_flex_regression_case(entry):
    metrics = compare_against_truth(entry)
    pos_nrmse = metrics["pos_metrics"]["NRMSE_per_body"].max()
    vel_nrmse = metrics["vel_metrics"]["NRMSE_per_body"].max()
    ang_nrmse = metrics["angle_metrics"]["NRMSE_per_body"].max()
    assert pos_nrmse < NRMSE_TOL, (
        f"{entry.case_id}: position NRMSE {pos_nrmse:.2e} >= {NRMSE_TOL} "
        f"vs the stored FSM baseline -- FSM's own output has drifted"
    )
    assert vel_nrmse < NRMSE_TOL, (
        f"{entry.case_id}: velocity NRMSE {vel_nrmse:.2e} >= {NRMSE_TOL} "
        f"vs the stored FSM baseline -- FSM's own output has drifted"
    )
    assert ang_nrmse < NRMSE_TOL, (
        f"{entry.case_id}: angle NRMSE {ang_nrmse:.2e} >= {NRMSE_TOL} "
        f"vs the stored FSM baseline -- FSM's own output has drifted"
    )

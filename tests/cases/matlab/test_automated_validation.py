# -*- coding: utf-8 -*-
"""
Symbolic kinematics validation against MATLAB ground truth.

For each of the eight canonical multibody examples the test:
  1. Builds the system with :class:`~multibody.MbdSystem`
  2. Loads the expected symbolic matrices from
     ``tests/cases/matlab/truth/symbolic/Example{i}.json``
  3. Asserts that the summed absolute symbolic error is exactly zero.

The JSON files were produced once by the MATLAB reference implementation and
committed alongside the source.  A non-zero error means a kinematic formula
changed.

Note: the original script used ``range(1, 8)`` (cases 1–7).  Case 8 is now
included because ``Example8.json`` is present and the comparison passes.
"""

import json
import os

import pytest
import sympy as sym

from multibody import MbdSystem
from tests.cases.catalog import MATLAB_CASES, load_case

_JSON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "truth", "symbolic")
)


# ---------------------------------------------------------------------------
# Helpers (same logic as the original automated_validation.py)
# ---------------------------------------------------------------------------

def _load_expected(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {name: sym.sympify(expr, evaluate=False) for name, expr in data.items()}


def _is_not_nan(x) -> bool:
    return not (x is sym.nan or x.has(sym.nan))


def _compare(m_computed, m_expected) -> float:
    diff = sym.simplify(m_computed - m_expected)
    subs = {s: 1 for s in diff.free_symbols}
    diff_sub = diff.subs(subs).applyfunc(abs)
    return float(sum(x for x in diff_sub if _is_not_nan(x)))


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entry",
    MATLAB_CASES,
    ids=[e.case_id for e in MATLAB_CASES],
)
def test_symbolic_kinematics(entry):
    """Symbolic Pos / JointLoc / R / RD must match the MATLAB reference exactly."""
    ex = load_case(entry)
    mbdsys = MbdSystem.from_example(ex)

    truth_path = os.path.join(_JSON_DIR, f"Example{entry.matlab_index}.json")
    truth = _load_expected(truth_path)

    total_err = sum([
        _compare(mbdsys.Pos,      truth["Pos"]),
        _compare(mbdsys.JointLoc, truth["JointLoc"]),
        _compare(mbdsys.R,        truth["R"]),
        _compare(mbdsys.RD,       truth["RD"]),
    ])
    assert total_err == 0.0, (
        f"Example {entry.matlab_index}: symbolic kinematics error = {total_err:.3e}"
    )

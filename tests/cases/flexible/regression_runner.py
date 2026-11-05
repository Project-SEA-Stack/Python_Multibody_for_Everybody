# -*- coding: utf-8 -*-
"""
Runner for FSM flexible-model self-consistency (regression) cases.

Usage (standalone)::

    python -m tests.cases.flexible.regression_runner [case_id ...]
    python -m tests.cases.flexible.regression_runner --update [case_id ...]

Omitting case IDs runs/checks all cases in
:data:`~tests.cases.flexible.regression_catalog.FLEX_REGRESSION_CASES`.
``--update`` (re)generates the stored baseline instead of checking against
it -- only do this deliberately, after confirming the example still matches
its independent PyChrono reference (see each example's entry in
``regression_catalog.py`` and the ``_pychrono_*_reference.py`` scripts in
``pychrono_backup/``), since it overwrites the regression baseline.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "source"))

import multibody as mb  # noqa: E402
from multibody.multibody_core.post_process import evaluate_trajectories  # noqa: E402
from tests.metrics import nrmse_per_body, resample_to  # noqa: E402
from tests.cases.flexible.regression_catalog import (  # noqa: E402
    FLEX_REGRESSION_CASES, NRMSE_TOL, FlexRegressionEntry, get_entry,
)


@dataclass
class FSMResult:
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    angles: np.ndarray


def _load_example(example_path):
    full_path = os.path.join(REPO_ROOT, example_path)
    spec = importlib.util.spec_from_file_location("_flex_regression_example", full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_fsm_example(entry) -> FSMResult:
    """Integrate the frozen model defined by *entry* and return its trajectory."""
    entry = get_entry(entry)
    ex = _load_example(entry.example_path)
    MBDsys = mb.MbdSystem.from_example(ex)
    main_num_vars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))
    kwargs = getattr(ex, "integrator_kwargs", {})
    sol = MBDsys.integrate(main_num_vars, ex.m0, ex.J0, tspan=ex.tspan, dt=ex.TimeStep, **kwargs)
    r, v, angles, _ = evaluate_trajectories(MBDsys, sol, main_num_vars)
    return FSMResult(t=np.array(sol.t), r=np.array(r), v=np.array(v), angles=np.array(angles))


def _truth_dir(entry: FlexRegressionEntry) -> str:
    return os.path.join(REPO_ROOT, entry.truth_dir)


def save_truth(case_id) -> None:
    """(Re)generate the stored FSM baseline trajectory for *case_id*."""
    entry = get_entry(case_id)
    result = run_fsm_example(entry)
    d = _truth_dir(entry)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "t.npy"), result.t)
    np.save(os.path.join(d, "r.npy"), result.r)
    np.save(os.path.join(d, "v.npy"), result.v)
    np.save(os.path.join(d, "angles.npy"), result.angles)


def compare_against_truth(case_id) -> dict:
    """Run *case_id* fresh and return NRMSE metrics against its stored FSM baseline."""
    entry = get_entry(case_id)
    result = run_fsm_example(entry)
    d = _truth_dir(entry)
    t_ref = np.load(os.path.join(d, "t.npy"))
    r_ref = np.load(os.path.join(d, "r.npy"))
    v_ref = np.load(os.path.join(d, "v.npy"))
    angles_ref = np.load(os.path.join(d, "angles.npy"))
    r_rs = resample_to(t_ref, result.t, result.r)
    v_rs = resample_to(t_ref, result.t, result.v)
    angles_rs = resample_to(t_ref, result.t, result.angles[:, :, None])[:, :, 0]
    return {
        "pos_metrics": nrmse_per_body(r_rs, r_ref, t_ref),
        "vel_metrics": nrmse_per_body(v_rs, v_ref, t_ref),
        "angle_metrics": nrmse_per_body(angles_rs[:, :, None], angles_ref[:, :, None], t_ref),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    update = "--update" in args
    targets = [a for a in args if a != "--update"] or [e.case_id for e in FLEX_REGRESSION_CASES]
    if update:
        for case_id in targets:
            save_truth(case_id)
            print(f"{case_id}: baseline (re)saved.")
    else:
        print(f"{'case_id':<28}  {'pos_nrmse':>10}  {'vel_nrmse':>10}  {'ang_nrmse':>10}  status")
        print("-" * 77)
        for case_id in targets:
            try:
                metrics = compare_against_truth(case_id)
                pn = metrics["pos_metrics"]["NRMSE_per_body"].max()
                vn = metrics["vel_metrics"]["NRMSE_per_body"].max()
                an = metrics["angle_metrics"]["NRMSE_per_body"].max()
                ok = pn < NRMSE_TOL and vn < NRMSE_TOL and an < NRMSE_TOL
                print(f"{case_id:<28}  {pn:10.2e}  {vn:10.2e}  {an:10.2e}  {'OK' if ok else 'FAIL'}")
            except Exception as exc:
                print(f"{case_id:<28}  {'':>10}  {'':>10}  {'':>10}  ERROR: {exc}")

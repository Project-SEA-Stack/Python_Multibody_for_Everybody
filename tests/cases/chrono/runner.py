# -*- coding: utf-8 -*-
"""
Runner for Chrono-validated multibody regression cases.

Usage (standalone)::

    python -m tests.cases.chrono.runner [case_id ...]

Omitting case IDs runs all cases in :data:`~tests.cases.chrono.catalog.CHRONO_CASES`.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Union

import numpy as np

from multibody import MbdSystem, evaluate_trajectories
from tests.metrics import nrmse_per_body, resample_to
from tests.cases.chrono.catalog import CHRONO_CASES, ChronoCaseEntry, get_entry


@dataclass
class M4EResult:
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    angles: np.ndarray


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def run_chrono_case(entry: Union[str, ChronoCaseEntry]) -> M4EResult:
    """Integrate the model defined by *entry* and return the trajectory."""
    entry = get_entry(entry)
    mod = importlib.import_module(entry.module)
    importlib.reload(mod)  # reset module globals to file defaults
    if entry.ic is not None:
        mod.ic = np.array(entry.ic, dtype=float)
    sys_ = MbdSystem.from_example(mod)
    mnv = np.hstack((sys_.ic, mod.ForcesPointsNum, mod.BodyDataNum))
    sol = sys_.integrate(mnv, mod.m0, mod.J0, tspan=entry.tspan, dt=mod.TimeStep)
    r, v, angles, _ = evaluate_trajectories(sys_, sol, mnv)
    return M4EResult(
        t=np.array(sol.t),
        r=np.array(r),
        v=np.array(v),
        angles=np.array(angles),
    )


def _truth_dir(entry: ChronoCaseEntry) -> str:
    """Absolute path of the local truth folder for *entry*."""
    return os.path.join(
        os.path.dirname(__file__), "truth", entry.case_id
    )


def compare_against_chrono(entry: Union[str, ChronoCaseEntry]) -> dict:
    """Run *entry* and return position/velocity NRMSE metrics against Chrono truth."""
    entry = get_entry(entry)
    result = run_chrono_case(entry)
    d = _truth_dir(entry)
    t_ref = np.load(os.path.join(d, "t_chrono.npy"))
    r_ref = np.load(os.path.join(d, "r_chrono.npy"))
    v_ref = np.load(os.path.join(d, "v_chrono.npy"))
    r_rs = resample_to(t_ref, result.t, result.r)
    v_rs = resample_to(t_ref, result.t, result.v)
    return {
        "pos_metrics": nrmse_per_body(r_rs, r_ref, t_ref),
        "vel_metrics": nrmse_per_body(v_rs, v_ref, t_ref),
    }


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [e.case_id for e in CHRONO_CASES]
    print(f"{'case_id':<18}  {'pos_nrmse':>10}  {'vel_nrmse':>10}  status")
    print("-" * 55)
    for case_id in targets:
        try:
            metrics = compare_against_chrono(case_id)
            pn = metrics["pos_metrics"]["NRMSE_per_body"].max()
            vn = metrics["vel_metrics"]["NRMSE_per_body"].max()
            ok = pn < 1e-2 and vn < 1.5e-2
            print(f"{case_id:<18}  {pn:10.2e}  {vn:10.2e}  {'OK' if ok else 'FAIL'}")
        except Exception as exc:
            print(f"{case_id:<18}  {'':>10}  {'':>10}  ERROR: {exc}")

# -*- coding: utf-8 -*-
"""Run a YAML validation case and compare against PyChrono truth.

Workflow per case (see :func:`compare_against_chrono`):

  1. Load the YAML pair via :func:`multibody.load_yaml_as_example`.
  2. Build the :class:`multibody.MbdSystem` and integrate the equations of
     motion over ``ex.tspan``.
  3. Evaluate body-CG positions and velocities via
     :func:`multibody.evaluate_trajectories`.
  4. Linearly interpolate the M4E trajectories onto the Chrono time grid.
  5. Compute NRMSE per body / per DOF against the Chrono reference.

The runner never writes ``r_M4E.npy`` / ``v_M4E.npy`` — those files in
``Validation2externalSoft/results/yaml_*/`` are legacy artifacts and are
ignored. Every test invocation regenerates the M4E trajectory in-memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from multibody import MbdSystem, evaluate_trajectories, load_yaml_as_example

from ...metrics import nrmse_per_body, resample_to
from .catalog import YamlCaseEntry, get_entry
from .truth import ChronoTruth, _select_bodies, load_chrono_truth


def _models_dir() -> str:
    """Absolute path to ``tests/cases/yaml/models/``."""
    return os.path.join(os.path.dirname(__file__), "models")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass(frozen=True)
class M4EResult:
    """In-memory M4E trajectory produced by :func:`run_yaml_case`.

    Shapes:
      * ``t`` — ``(T,)``
      * ``r`` — ``(T, NB, 2)`` body CG positions, columns ``(x, z)``
      * ``v`` — ``(T, NB, 2)`` body CG velocities, columns ``(x, z)``
      * ``angles`` — ``(T, NB)`` planar pitch angles
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    angles: np.ndarray


def run_yaml_case(case: YamlCaseEntry | str) -> M4EResult:
    """Integrate the YAML case and return its M4E trajectory."""
    entry = get_entry(case)

    yaml_path = os.path.join(_models_dir(), entry.yaml_basename)

    ex = load_yaml_as_example(yaml_path)
    mbd_system = MbdSystem.from_example(ex)

    main_num_vars = np.hstack((mbd_system.ic, ex.ForcesPointsNum, ex.BodyDataNum))

    sol = mbd_system.integrate(
        main_num_vars,
        ex.m0,
        ex.J0,
        tspan=ex.tspan,
        dt=ex.TimeStep,
    )

    com_positions, com_velocities, angle_positions, _ = evaluate_trajectories(
        mbd_system, sol, main_num_vars
    )

    return M4EResult(
        t=np.asarray(sol.t),
        r=np.asarray(com_positions),
        v=np.asarray(com_velocities),
        angles=np.asarray(angle_positions),
    )


def compare_against_chrono(case: YamlCaseEntry | str) -> Mapping[str, object]:
    """Run a YAML case and report NRMSE against the Chrono reference.

    The returned dict contains:
      * ``case_id``                — string identifier
      * ``bodies_compared``        — tuple of 0-based body indices used
      * ``t_truth``                — Chrono time vector
      * ``r_truth`` / ``v_truth``  — sliced Chrono arrays
      * ``r_m4e``   / ``v_m4e``    — M4E arrays resampled onto ``t_truth``
      * ``position_metrics``       — output of :func:`tests.metrics.nrmse_per_body`
      * ``velocity_metrics``       — output of :func:`tests.metrics.nrmse_per_body`
    """
    entry = get_entry(case)
    truth: ChronoTruth = load_chrono_truth(entry)
    m4e = run_yaml_case(entry)

    # Decide which bodies participate in the comparison. If the catalog
    # doesn't restrict the set, default to "every body the truth has".
    nb_truth = truth.r.shape[1]
    if entry.bodies_to_compare is None:
        bodies = tuple(range(nb_truth))
    else:
        bodies = entry.bodies_to_compare
        max_idx = max(bodies)
        if max_idx >= nb_truth:
            raise IndexError(
                f"{entry.case_id}: requested body index {max_idx} but truth "
                f"only has {nb_truth} bodies."
            )

    r_truth = _select_bodies(truth.r, bodies)
    v_truth = _select_bodies(truth.v, bodies)

    if m4e.r.shape[1] < nb_truth:
        raise IndexError(
            f"{entry.case_id}: M4E produced {m4e.r.shape[1]} bodies, "
            f"truth has {nb_truth}."
        )

    r_m4e_full = _select_bodies(m4e.r, bodies)
    v_m4e_full = _select_bodies(m4e.v, bodies)

    r_m4e = resample_to(truth.t, m4e.t, r_m4e_full)
    v_m4e = resample_to(truth.t, m4e.t, v_m4e_full)

    return {
        "case_id": entry.case_id,
        "bodies_compared": bodies,
        "t_truth": truth.t,
        "r_truth": r_truth,
        "v_truth": v_truth,
        "r_m4e": r_m4e,
        "v_m4e": v_m4e,
        "position_metrics": nrmse_per_body(r_m4e, r_truth, truth.t),
        "velocity_metrics": nrmse_per_body(v_m4e, v_truth, truth.t),
    }


__all__ = ["M4EResult", "run_yaml_case", "compare_against_chrono"]

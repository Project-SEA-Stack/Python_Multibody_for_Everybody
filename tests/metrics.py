# -*- coding: utf-8 -*-
"""Numerical comparison helpers shared by validation tests.

The metrics here mirror the ones in
``Validation2externalSoft/compare.py`` (modified Okada-2022 NRMSE) so that
results obtained from the standalone scripts and from the test runners are
directly comparable.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def nrmse_per_body(
    arr_pred: np.ndarray,
    arr_truth: np.ndarray,
    time: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Compute per-body / per-DOF NRMSE between two trajectory arrays.

    Both inputs must have shape ``(T, NB, ND)`` and share the same shape.
    The truth array is used as the normalization reference (Gonzalez 2006).

    Returns a dict with:
      * ``NRMSE_per_body``        — shape ``(NB,)``
      * ``NRMSE_per_body_per_DOF`` — shape ``(NB, ND)``
      * ``L2_per_DOF``             — shape ``(T, NB, ND)`` (per-sample relative L2)
    """
    if arr_pred.shape != arr_truth.shape:
        raise ValueError(
            f"Shape mismatch: pred {arr_pred.shape} vs truth {arr_truth.shape}"
        )

    N = len(time)
    max_ref = np.max(np.abs(arr_truth), axis=0)
    # avoid divide-by-zero when a DOF is identically zero in truth
    safe_ref = np.where(max_ref == 0.0, 1.0, max_ref)
    rel_L2 = np.abs(arr_pred - arr_truth) / safe_ref
    nrmse_dof = np.linalg.norm(rel_L2 / N, axis=0)
    nrmse_body = np.linalg.norm(nrmse_dof, axis=-1)

    return {
        "NRMSE_per_body": np.atleast_1d(nrmse_body),
        "NRMSE_per_body_per_DOF": np.atleast_2d(nrmse_dof),
        "L2_per_DOF": np.atleast_3d(rel_L2),
    }


def resample_to(
    t_target: np.ndarray,
    t_source: np.ndarray,
    arr_source: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``arr_source`` (shape ``(T_s, NB, ND)``) onto
    ``t_target``. Returns an array of shape ``(len(t_target), NB, ND)``.

    Samples in ``t_target`` outside the range of ``t_source`` are clipped
    to the boundary values of ``arr_source``.
    """
    if arr_source.ndim != 3:
        raise ValueError(f"Expected (T, NB, ND); got shape {arr_source.shape}")
    T_t = len(t_target)
    _, NB, ND = arr_source.shape
    out = np.empty((T_t, NB, ND), dtype=arr_source.dtype)
    for b in range(NB):
        for d in range(ND):
            out[:, b, d] = np.interp(t_target, t_source, arr_source[:, b, d])
    return out

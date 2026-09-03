# multibody/linearization/linear_integrator.py
from __future__ import annotations
import numpy as np


def linear_integrator(LinManager, t: float, y: np.ndarray, mainNumVars: np.ndarray, t_update: list[int]):
    """
    Compiled linear time integration:
        M qdd + C qd + K q = F(t)
    with y = [q, qd].
    """
    y = np.asarray(y).reshape(-1)
    nq = y.size // 2
    q = y[:nq]
    qd = y[nq:]

    if not LinManager.has_compiled():
        raise RuntimeError("Call compile_operating_point(...) before integrating.")

    M = LinManager.compiled.M
    C = LinManager.compiled.C
    K = LinManager.compiled.K
    F = LinManager.force_time(t).reshape(-1)

    qdd = np.linalg.solve(M, F - C @ qd - K @ q)

    dydt = np.zeros_like(y)
    dydt[:nq] = qd
    dydt[nq:] = qdd
    return dydt

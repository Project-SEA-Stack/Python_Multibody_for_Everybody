# -*- coding: utf-8 -*-
"""
@author: Sahand Sabet

Independent reference implementation for cross-validating FSM's bending
natural frequencies: a classical Euler-Bernoulli beam finite element
(cubic Hermite shape functions, consistent mass matrix, textbook formulas --
e.g. Przemieniecki, "Theory of Matrix Structural Analysis", or any standard
FEM text), built completely independently of FSM's own rigid-segment +
half-cell-compliance formulation (source/multibody/flexible/).

This is the same validation tier Chrono/OpenFAST rely on most heavily for a
continuously-varying-property beam element: comparing against a SECOND,
differently-formulated numerical method, not just an analytical closed form
or a self-consistency check of FSM's own machinery. Two fundamentally
different discretizations (rigid links + point springs vs. continuous
cubic polynomials + consistent mass, Gauss-quadrature integrated) agreeing
as both are refined is strong, independent evidence neither is wrong in a
way that happens to cancel out within itself.

Supports both constant and continuously-varying EI(x)/rho(x)*A(x) (the
latter integrated per-element via Gauss-Legendre quadrature -- 5 points,
exact for ancient polynomials up to degree 9 and very accurate for smooth
non-polynomial tapers too), for a clamped-free (cantilever) boundary
condition only (the one every FSM bending example in this repo also uses).
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh


def _hermite_shape_and_curvature(xi):
    """Cubic Hermite shape functions N(xi) and their second derivative
    w.r.t. the NORMALIZED coordinate xi in [0, 1] (caller divides by Le**2
    to get the real curvature). Node ordering: [w1, theta1, w2, theta2]."""
    N1 = 1 - 3 * xi**2 + 2 * xi**3
    N2 = xi - 2 * xi**2 + xi**3
    N3 = 3 * xi**2 - 2 * xi**3
    N4 = -xi**2 + xi**3
    d2N1 = -6 + 12 * xi
    d2N2 = -4 + 6 * xi
    d2N3 = 6 - 12 * xi
    d2N4 = -2 + 6 * xi
    return (N1, N2, N3, N4), (d2N1, d2N2, d2N3, d2N4)


def cantilever_bending_frequencies(L, EI_of_x, rhoA_of_x, n_elem, n_modes=3, n_gauss=5):
    """
    Natural frequencies [Hz] of a clamped-free (cantilever) Euler-Bernoulli
    beam, via a standard 2-node/4-DOF cubic Hermite finite element.

    Parameters
    ----------
    L : float -- total beam length.
    EI_of_x, rhoA_of_x : callable(x) -> float -- bending stiffness and mass
        per unit length at position x in [0, L] (0 at the clamped root).
    n_elem : int -- number of elements (uniform length).
    n_modes : int -- number of lowest modes to return.
    n_gauss : int -- Gauss-Legendre points per element (5 is already exact
        for any polynomial EI(x)*rhoA(x) up to degree 9).

    Returns
    -------
    numpy.ndarray, length n_modes, ascending.
    """
    Le = L / n_elem
    gauss_pts, gauss_wts = leggauss(n_gauss)   # on [-1, 1]
    xi_pts = 0.5 * (gauss_pts + 1.0)             # mapped to [0, 1]
    xi_wts = 0.5 * gauss_wts

    n_dof = 2 * (n_elem + 1)
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    for e in range(n_elem):
        x0 = e * Le
        Ke = np.zeros((4, 4))
        Me = np.zeros((4, 4))
        for xi, w in zip(xi_pts, xi_wts):
            x = x0 + xi * Le
            (N1, N2, N3, N4), (d2N1, d2N2, d2N3, d2N4) = _hermite_shape_and_curvature(xi)
            N_vec = np.array([N1, Le * N2, N3, Le * N4])
            B_vec = np.array([d2N1, Le * d2N2, d2N3, Le * d2N4]) / Le**2
            Ke += EI_of_x(x) * np.outer(B_vec, B_vec) * w * Le
            Me += rhoA_of_x(x) * np.outer(N_vec, N_vec) * w * Le
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += Ke[a, b]
                M[dofs[a], dofs[b]] += Me[a, b]

    # clamped root: remove the first 2 DOF (w1, theta1); everything else free
    K_reduced, M_reduced = K[2:, 2:], M[2:, 2:]
    eigvals, _ = eigh(K_reduced, M_reduced)
    eigvals = np.sort(eigvals[eigvals > 1e-9])
    return (np.sqrt(eigvals) / (2 * np.pi))[:n_modes]

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:32:52 2025

@author: adiazfl
"""

import sympy as sym

def systems_energy(NBodies, Q, QD, R, CGpos, g, gVec):
    """
    Computes the total mechanical energy of the multibody system. Correct
    only for linear springs and dampers for now.

    Parameters
    ----------
    NBodies : int
        Number of bodies in the system.
    Q, QD : list of sympy.Symbol
        Generalized coordinates and velocities.
    R : sympy.Matrix
        Velocity transformation matrix.
    CGpos : sympy.Matrix
        Center-of-gravity positions for each body (rows: bodies, columns: x, z).
    g : sympy.Symbol or float
        Gravity constant.
    gVec : list-like or sympy.Matrix
        Per-body gravity-direction factors (e.g. 1 or 0).

    Returns
    -------
    Em : sympy.Expr
        Total mechanical energy (kinetic + potential).
    M : sympy.Matrix
        Block-diagonal mass/inertia matrix.
    m : sympy.Matrix
        Mass vector for each body.
    J : sympy.Matrix
        Moment-of-inertia vector for each body.
    """

    # Define symbolic mass and moment of inertia vectors
    m = sym.Matrix(sym.symbols(f"m1:{NBodies+1}", real=True))
    J = sym.Matrix(sym.symbols(f"J1:{NBodies+1}", real=True))

    # Construct full diagonal mass matrix: [m1 m1 J1 ... mN mN JN]
    M_values = sym.Matrix([item for i in range(NBodies) for item in [m[i], m[i], J[i]]])

    M = sym.diag(*M_values)

    # Velocity vector
    V = R * sym.Matrix(QD)

    # Kinetic energy
    K = sym.Rational(1, 2) * V.T * M * V

    # Potential energy: U = sum(m_i * g * gVec[i] * height_i)
    h = CGpos[:, 1]  # z-component of CG position (assuming 2D)
    gVec = sym.Matrix(gVec)
    U = m.dot(g * sym.Matrix([gVec[i] * h[i] for i in range(NBodies)]))

    # Total mechanical energy
    Em = K[0] + U  # K is a 1x1 matrix → extract scalar

    return Em, M, m, J

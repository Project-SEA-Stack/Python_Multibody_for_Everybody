# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:30:44 2025

@author: adiazfl
"""

import numpy as np

def integrate_dynamics(t, y, mainNumVars, m0, J0, MBDsys, external_manager=None):
    """
    Python version of RigidBIntegrator from MATLAB.
    
    Parameters
    ----------
    t : float
        Current time
    y : np.ndarray
        State vector: [Q; QD]
    mainNumVars : np.ndarray (mutable)
        Preallocated symbolic input vector (will be updated in-place)
    m0, J0 : np.ndarray or list
        Constant body parameters
    t_update : list of int
        Indices in mainNumVars where `t` should be inserted
    
    Returns
    -------
    dydt : np.ndarray
        Time derivative of y = [QD; QDD]
    """

    Q_length = len(y) // 2
    if len(y) % 2 != 0:
        raise ValueError("Length of y should be divisible by 2")
    
    Q  = y[:Q_length]
    QD = y[Q_length:]

    # Update symbolic input vector
    mainNumVars[:len(y)] = np.hstack((Q, QD))
  
    mainNumVars[MBDsys.t_update] = t

    # print(f"Time: {t:.4f}")

    # Prepare inputs for lambdified symbolic functions
    input_vec = np.hstack((mainNumVars, m0, J0))

    # Evaluate symbolic EOM functions
    ReducedM    = MBDsys.R_func(*mainNumVars).T @ MBDsys.M_func(*input_vec) @ MBDsys.R_func(*mainNumVars)
    Right_side  = MBDsys.R_func(*mainNumVars).T @ (MBDsys.Force_func(*input_vec) \
                                                  - MBDsys.M_func(*input_vec) @ MBDsys.RD_func(*mainNumVars) @ QD.reshape((-1,1)))

    # Add external modules forces. TODO: substitute by mask to avoid if statements
    if external_manager is not None:
        Q_ext, Madd = external_manager.generalized_forces(t,mainNumVars)
    else:
        Q_ext = 0*Right_side
        Madd  = 0*ReducedM

    Right_side  += Q_ext
    ReducedM    += Madd
    

    # Compute acceleration: QDD = M \ F
    QDD = np.linalg.solve(ReducedM, Right_side).flatten()

    # Return time derivatives
    dydt = np.zeros_like(y)
    dydt[:Q_length] = QD
    dydt[Q_length:] = QDD

    return dydt

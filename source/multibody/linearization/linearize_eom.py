import sympy as sym
from dataclasses import dataclass
import numpy as np
from typing import Callable

@dataclass
class LinearizedMBD:
    Mbar: Callable[..., np.ndarray]
    Cbar: Callable[..., np.ndarray]
    Kbar: Callable[..., np.ndarray]
    Fbar: Callable[..., np.ndarray]
    # Optional evaluators (NumPy-ready) can be added by lambdify later

def linearize_mbd(MBD, q0, m0, J0, print_sym_matrices=False):
    """
    Linearize B(q)^T M B(q) qdd = B(q)^T ( f − M dotB(q,qd) qd )
    about (q0, qd0) with qdd0 = 0, using your class' symbolic fields.

    Expected attributes on `MBD` (names follow your codebase):
        MBD.Q   -> list[Sym]     : q symbols
        MBD.QD  -> list[Sym]     : qd symbols
        MBD.R   -> sym.Matrix    : B(q) (cartesian-velocity map)
        MBD.RD  -> sym.Matrix    : dotB(q,qd)
        MBD.M   -> sym.Matrix    : Cartesian mass/inertia (block-diag)
        MBD.ForceAllCombined -> sym.Matrix (Cartesian forces f). If absent and assume_f_constant=True,
                                we set f as a symbol vector (treated as constant).

    Returns:
        LinearizedEOM with M0, Kq, Kqd so that:
            M0 * δqdd + (C + dF/dqd)* δqd + K * δq = F_ext
    """
    # --- Pull symbols/expressions
    q_syms  = MBD.Q
    qd_syms = MBD.QD
    f       = MBD.ForceAllCombined

    # --- Evaluate base objects at (q0, qd0)
    # [Assumption] make qd0 = 0 for all variables
    qd0                 = [0] * len(qd_syms)
    subs_base           = dict(zip(q_syms, q0)) | dict(zip(qd_syms, qd0))
    subs_base_extended  = subs_base | dict(zip(MBD.m,m0)) | dict(zip(MBD.J,J0))

    # Precompute auxiliary matrices and substitute all joint coordinates DOFs and lambdify as a function of mainNumVars, m, J
    M_sym       = MBD.M.subs(subs_base_extended)
    R_lin_sym   = MBD.R.subs(subs_base)
    RD_lin_sym  = MBD.RD.subs(subs_base)
    dFdQ        = (MBD.R.T * f).jacobian(q_syms).subs(subs_base_extended) 
    dFdQD       = R_lin_sym.T * np.array(f.jacobian(qd_syms).subs(subs_base))

    # Compute symbolic linearized M, C, K matrices
    M_lin_sym = R_lin_sym.T * M_sym * R_lin_sym
    C_lin_sym = R_lin_sym.T * M_sym * RD_lin_sym - dFdQD
    K_lin_sym = - dFdQ
    F_lin_sym = R_lin_sym.T * f.subs(subs_base_extended)
    if print_sym_matrices:
        print('Linearized Mass matrix Mbar:')
        sym.pprint(M_lin_sym)
        print('Linearized Damping matrix Cbar:')
        sym.pprint(C_lin_sym)
        print('Linearized Stiffness matrix Kbar:')
        sym.pprint(K_lin_sym)
        print('Linearized Force vector Fbar:')
        sym.pprint(F_lin_sym)
    # Lambdify M, C, K. [Note]: I am not including m,J as arguments cos they have been substituted already
    # [Note] these will be a function of symbolic variables, but will not change with q, qd, m, J
    Mbar_func = sym.lambdify(MBD.mainSymVars, M_lin_sym, modules="numpy")
    Cbar_func = sym.lambdify(MBD.mainSymVars, C_lin_sym, modules="numpy")
    Kbar_func = sym.lambdify(MBD.mainSymVars, K_lin_sym, modules="numpy")
    Fbar_func = sym.lambdify(MBD.mainSymVars, F_lin_sym, modules="numpy")

    return LinearizedMBD(Mbar=Mbar_func, Cbar=Cbar_func, Kbar=Kbar_func, Fbar=Fbar_func)

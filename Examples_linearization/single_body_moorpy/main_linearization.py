# Examples_linearization/single_body_moorpy/main_linearization.py
"""
Minimal MoorPy–M4E linearization example.

This example demonstrates:

1. A single floating body whose mooring dynamics are modelled by MoorPy.
2. How MoorPy's full 6×6 body-level matrices are reduced to the M4E planar
   Cartesian space ``[x, z, pitch]``.
3. How ``LinearizationManager`` transforms Cartesian matrices into the M4E
   joint-coordinate space.
4. Nonzero surge–pitch coupling arising from the offset fairlead.
5. That the MoorPy body reference point equals the M4E CG at ``q0``.

Files
-----
M4E_inputs.py      — MBD topology, GR/BD point coordinates, mass/inertia.
moorpy_inputs.py   — line-type properties and point references (no coordinates).
main_linearization.py — this file; buildable and runnable via ``run()``.

Call ``run()`` from the REPL or from tests.  The function returns
``(adapter, manager, totals)`` for further inspection.
"""

from __future__ import annotations

import os
import sys
import importlib

import numpy as np

# Ensure the package root and source directory are on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_ROOT, "source")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run(print_output: bool = True):
    """Build the moored single-float linearization and return results.

    Parameters
    ----------
    print_output:
        When ``True`` (default), prints the MoorPy and M4E matrices to stdout.

    Returns
    -------
    adapter : MoorPyLinearMCKF
        The instantiated mooring adapter.
    manager : LinearizationManager
        The linearization manager with the adapter registered.
    totals : FrequencyDomainTotals
        Joint-coordinate M, C, K and force at ``omega = 0.5 rad/s``.
    """
    # ------------------------------------------------------------------ #
    # 1. Import M4E and mooring input modules                             #
    # ------------------------------------------------------------------ #
    from Examples_linearization.single_body_moorpy import M4E_inputs as ex
    from Examples_linearization.single_body_moorpy import moorpy_inputs as mp_mod

    from multibody import MbdSystem
    from multibody.linearization.linearization_main import LinearizationManager
    from multibody.linearization._moorpy_schema import from_module, validate_moorpy_inputs
    from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF

    # ------------------------------------------------------------------ #
    # 2. Parse and validate the mooring inputs schema                     #
    # ------------------------------------------------------------------ #
    moorpy_inputs = from_module(mp_mod)
    validate_moorpy_inputs(moorpy_inputs, ex.Initial_Points)

    # ------------------------------------------------------------------ #
    # 3. Build the M4E multibody system                                   #
    # ------------------------------------------------------------------ #
    mbd_sys = MbdSystem.from_example(ex)

    # ------------------------------------------------------------------ #
    # 4. Define the linearization operating point                         #
    # ------------------------------------------------------------------ #
    # q0: joint-space equilibrium (all zeros — buoy at rest at origin).
    q0 = (mbd_sys.ic - ex.ic)[: len(mbd_sys.Q)]  # shape (nq,) = (3,)

    # mainNumVars: full [q, qd] vector evaluated at the operating point.
    mainNumVars = ex.ic.copy()   # shape (6,) = [0,0,0,0,0,0]

    if print_output:
        print("\n" + "=" * 60)
        print("Single-body MoorPy–M4E linearization example")
        print("=" * 60)
        print(f"  q0           = {q0}")
        print(f"  nq (joint DOFs) = {len(mbd_sys.Q)}")
        print(f"  nb (M4E bodies) = {len(mbd_sys.NDOF)}")

    # ------------------------------------------------------------------ #
    # 5. Build the MoorPy adapter                                         #
    # ------------------------------------------------------------------ #
    adapter = MoorPyLinearMCKF(
        mbd_sys,
        q0,
        mainNumVars,
        moorpy_inputs,
        include_mean_force=False,
    )

    # ------------------------------------------------------------------ #
    # 6. Show the original MoorPy 6×6 matrices                           #
    # ------------------------------------------------------------------ #
    ms = adapter.moorpy_system
    M6, A6, B6, K6 = ms.getSystemDynamicMatrices(
        DOFtype="coupled", lines_only=True
    )

    if print_output:
        _pp = lambda name, mat: (
            print(f"\n--- {name} (6×6 MoorPy body matrix) ---"),
            print(np.array2string(mat, precision=4, suppress_small=True)),
        )
        _pp("M_mooring  (line structural inertia)", M6)
        _pp("A_mooring  (line added mass)",          A6)
        _pp("B_mooring  (line damping)",              B6)
        _pp("K_mooring  (mooring stiffness, incl. geometric)", K6)

    # ------------------------------------------------------------------ #
    # 7. Show the reduction indices and the 3×3 Cartesian matrices        #
    # ------------------------------------------------------------------ #
    idx = adapter.reduction_indices       # [0, 2, 4]
    M_cart, C_cart, K_cart, _ = adapter.frequency_domain_MCKF(0.5)

    # Independent verification: reduce 6×6 manually.
    M3_check = (M6 + A6)[np.ix_(idx, idx)]
    C3_check = B6[np.ix_(idx, idx)]
    K3_check = K6[np.ix_(idx, idx)]

    if print_output:
        print(f"\n--- Reduction indices (x=0, z=2, pitch=4) ---")
        print(f"  idx = {idx}")

        _pc = lambda name, mat: (
            print(f"\n--- {name} (3×3 M4E body-Cartesian) ---"),
            print(np.array2string(mat, precision=4, suppress_small=True)),
        )
        _pc("M_cart  [x, z, pitch]", M_cart)
        _pc("C_cart  [x, z, pitch]", C_cart)
        _pc("K_cart  [x, z, pitch]", K_cart)

        assert np.allclose(M_cart, M3_check), \
            "Adapter M_cart does not match manual 6→3 reduction"
        assert np.allclose(C_cart, C3_check), \
            "Adapter C_cart does not match manual 6→3 reduction"
        assert np.allclose(K_cart, K3_check), \
            "Adapter K_cart does not match manual 6→3 reduction"
        print("\n[OK] Adapter matrices match independent manual reduction.")

        # Surge–pitch coupling term: K_cart[0, 2] (x → pitch)
        K_xp = K_cart[0, 2]
        print(f"\n--- Surge–pitch coupling K[x, pitch] = {K_xp:.4f} N·m/m ---")
        if abs(K_xp) > 1.0:
            print("     (nonzero: offset fairlead produces translation–rotation coupling ✓)")
        else:
            print("     (near zero — check fairlead offset geometry)")

        # Body reference point verification
        body = ms.bodyList[0]
        print(f"\n--- MoorPy body r6 at q0 ---")
        print(f"  x     = {body.r6[0]:.6f} m  (expected 0.0)")
        print(f"  z     = {body.r6[2]:.6f} m  (expected 0.0)")
        print(f"  pitch = {body.r6[4]:.6f} rad (expected 0.0)")
        assert abs(body.r6[0]) < 1e-9, "MoorPy body x ≠ M4E CG x"
        assert abs(body.r6[2]) < 1e-9, "MoorPy body z ≠ M4E CG z"
        assert abs(body.r6[4]) < 1e-9, "MoorPy body pitch ≠ M4E CG pitch"
        print("[OK] MoorPy body reference point equals M4E CG.")

    # ------------------------------------------------------------------ #
    # 8. Build LinearizationManager and register the adapter              #
    # ------------------------------------------------------------------ #
    manager = LinearizationManager(
        mbd_sys, q0, mainNumVars, ex.m0, ex.J0, print_sym_matrices=False
    )
    manager.register(adapter)

    # ------------------------------------------------------------------ #
    # 9. Assemble joint-coordinate matrices                               #
    # ------------------------------------------------------------------ #
    omega_demo = 0.5   # rad/s
    totals = manager.assemble_frequency_domain(omega_demo)

    nq = manager.nq
    M_joint = np.asarray(totals.M).reshape(nq, nq)
    C_joint = np.asarray(totals.C).reshape(nq, nq)
    K_joint = np.asarray(totals.K).reshape(nq, nq)

    if print_output:
        R0 = manager.R0  # (ncart, nq) = (3, 3) for single float
        _pj = lambda name, mat: (
            print(f"\n--- {name} (joint-coordinate, {nq}×{nq}) ---"),
            print(np.array2string(mat, precision=4, suppress_small=True)),
        )
        _pj("M_joint  [x, z, theta]", M_joint)
        _pj("C_joint  [x, z, theta]", C_joint)
        _pj("K_joint  [x, z, theta]", K_joint)

        # -------------------------------------------------------------- #
        # Five-step chain verification                                   #
        # -------------------------------------------------------------- #
        # Step 1: full 6-DOF body matrices from MoorPy — already shown.
        # Step 2: independently reduced 3-DOF submatrices — M3_check etc.
        # Step 3: adapter Cartesian output — M_cart, C_cart, K_cart.
        # Step 4: manager joint-coordinate output — M_joint, C_joint, K_joint.
        # Step 5: independent R0^T @ X_cart @ R0 for all three matrices.
        print(f"\n--- Five-step chain verification ---")

        M_cart_contrib = R0.T @ M_cart @ R0
        C_cart_contrib = R0.T @ C_cart @ R0
        K_cart_contrib = R0.T @ K_cart @ R0

        M_joint_contrib = M_joint - manager.M_mbd
        C_joint_contrib = C_joint - manager.C_mbd
        K_joint_contrib = K_joint - manager.K_mbd

        diff_M = np.max(np.abs(M_cart_contrib - M_joint_contrib))
        diff_C = np.max(np.abs(C_cart_contrib - C_joint_contrib))
        diff_K = np.max(np.abs(K_cart_contrib - K_joint_contrib))

        print(f"  Step 5 M: max |R0^T M_cart R0  -  dM_joint| = {diff_M:.2e}")
        print(f"  Step 5 C: max |R0^T C_cart R0  -  dC_joint| = {diff_C:.2e}")
        print(f"  Step 5 K: max |R0^T K_cart R0  -  dK_joint| = {diff_K:.2e}")

        assert np.allclose(M_cart_contrib, M_joint_contrib, atol=1e-8), \
            "R0^T M_cart R0 ≠ mooring contribution to M_joint"
        assert np.allclose(C_cart_contrib, C_joint_contrib, atol=1e-8), \
            "R0^T C_cart R0 ≠ mooring contribution to C_joint"
        assert np.allclose(K_cart_contrib, K_joint_contrib, atol=1e-8), \
            "R0^T K_cart R0 ≠ mooring contribution to K_joint"
        print("[OK] Steps 1→2→3→4→5 all consistent.")

        print("\n" + "=" * 60)
        print("Example complete — all consistency checks passed.")
        print("=" * 60)

    return adapter, manager, totals


if __name__ == "__main__":
    run(print_output=True)

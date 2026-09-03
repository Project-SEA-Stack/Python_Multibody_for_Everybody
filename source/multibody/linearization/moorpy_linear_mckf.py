# multibody/linearization/moorpy_linear_mckf.py
"""
MoorPy mooring dynamics adapter for M4E frequency-domain linearization.

This module provides :class:`MoorPyLinearMCKF`, which implements the
:class:`~multibody.linearization.linearization_main.LinearizationAdapter`
protocol using MoorPy body-level dynamic matrices.

Architecture overview
---------------------
::

    M4E Initial_Points
        ├── GR anchors       → MoorPy fixed points
        └── BD fairleads     → MoorPy points attached to coupled bodies

    MoorPy (via MoorPySystemBuilder)
        ├── construct MoorPy bodies at M4E CGs (rCG = 0)
        ├── attach fairleads with body-local BD coordinates
        ├── initialize catenary line state
        ├── obtain full 6 × N matrices  M, A, B, K
        └── reduce [x, y, z, roll, pitch, yaw] → [x, z, pitch]

    frequency_domain_MCKF(ω) returns
        M_cart = M + A   (3·nb_m4e × 3·nb_m4e)
        C_cart = B        (3·nb_m4e × 3·nb_m4e)
        K_cart = K        (3·nb_m4e × 3·nb_m4e)
        F["dc"]           constant SymPy Matrix
        F["phasor"]       complex ndarray, zeros

    LinearizationManager
        └── body Cartesian → M4E joint coordinates

Matrix units
------------
* M_cart : kg (structural + added mass of line segments)
* C_cart : kg/s (viscous drag, approximate — see limitations)
* K_cart : N/m or N·m/rad as appropriate (includes geometric stiffness)
* F_dc   : N or N·m (mean line-induced body forces, optional)

DOF reduction
-------------
MoorPy returns 6-DOF body matrices in the ordering
``[x, y, z, roll, pitch, yaw]``.
M4E planar DOFs correspond to local body indices ``[0, 2, 4]`` (x, z, pitch).

For ``nb_mp`` mooring bodies::

    idx_6dof = [6*i + d for i in range(nb_mp) for d in (0, 2, 4)]

Fairlead transformation
-----------------------
MoorPy's ``Body.getDynamicMatrices`` already transforms line-end forces and
moments to the body reference point via its internal ``translateMatrix3to6``
helper, using the fairlead offset stored in ``body.rPointRel``.  No
additional ``H`` transformation is applied here.

Body reference point
--------------------
The MoorPy body reference point is set equal to the M4E body CG by passing
``rCG=np.zeros(3)`` to ``System.addBody``.  No reference-point translation
is required in the adapter.

MoorPy methods used
-------------------
* ``System.initialize()``
* ``System.updateSystemDynamicMatrices(omegas, S_zeta)``
* ``System.getSystemDynamicMatrices(DOFtype="coupled", lines_only=True)``
* ``System.getForces(DOFtype="coupled", lines_only=True)``

Frequency dependence
--------------------
All four MoorPy matrices (M, A, B, K) are frequency-independent.  The
adapter returns 2-D (not ω-batched) arrays.

Limitations
-----------
* The damping matrix ``B`` (and hence ``C_cart``) uses unit-amplitude motions
  of line nodes by default.  For physically accurate damping, RAOs must be
  provided to ``line.updateLumpedMass``.  This limitation is inherited from
  MoorPy's ``updateSystemDynamicMatrices`` API.
* Off-diagonal M/A/B cross-body blocks are *not* assembled by
  ``getSystemDynamicMatrices`` (confirmed MoorPy source issue).  Only K
  cross-body coupling is preserved.
* MoorPy is an optional dependency.  Importing this module does not require
  MoorPy.  The adapter raises ``ImportError`` at instantiation time if MoorPy
  is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import sympy as sym


class MoorPyLinearMCKF:
    """Mooring dynamics adapter for :class:`~multibody.linearization.linearization_main.LinearizationManager`.

    Constructs a MoorPy system at the M4E operating point ``q0``, extracts
    body-level dynamic matrices, reduces them to M4E planar Cartesian
    coordinates ``[x, z, pitch]``, and assembles the global body-Cartesian
    matrices.

    Parameters
    ----------
    mbd_sys:
        Compiled :class:`~multibody.multibody_core.mbd_system.MbdSystem`.
    q0:
        Joint-coordinate equilibrium vector, shape ``(nq,)``.
    mainNumVars:
        Full numeric parameter vector for the M4E model.  ``q`` and ``qd``
        slots are overwritten internally.
    moorpy_inputs:
        Validated
        :class:`~multibody.linearization._moorpy_schema.MoorPyInputs`.
    include_mean_force:
        When ``True``, populate ``F["dc"]`` with the constant mean body
        wrench due to mooring lines at ``q0``.  When ``False`` (default),
        ``F["dc"]`` is a zero SymPy matrix.

    Raises
    ------
    ImportError
        If MoorPy is not installed (raised at instantiation, not at import).

    Notes
    -----
    ``q0`` is read once during construction and is never modified.  The
    MoorPy equilibrium solver is not invoked — bodies remain fixed at the
    prescribed ``q0``.
    """

    name: str = "MoorPyLinearMCKF"

    def __init__(
        self,
        mbd_sys: Any,
        q0: npt.ArrayLike,
        mainNumVars: npt.ArrayLike,
        moorpy_inputs: Any,
        *,
        include_mean_force: bool = False,
    ) -> None:

        # Build the MoorPy system at q0.
        from ._moorpy_system_builder import MoorPySystemBuilder

        self._builder = MoorPySystemBuilder(
            mbd_sys,
            np.asarray(q0, dtype=float),
            np.asarray(mainNumVars, dtype=float),
            moorpy_inputs,
        )

        # Sizes.
        self._nb_m4e: int = len(mbd_sys.NDOF)   # total M4E bodies
        self._nb_mp: int = len(self._builder.involved_bodies)  # MoorPy (mooring) bodies
        self._ncart: int = 3 * self._nb_m4e
        self._include_mean_force = include_mean_force

        # ------------------------------------------------------------------ #
        # Reduction indices                                                   #
        # ------------------------------------------------------------------ #
        # MoorPy 6-DOF body ordering: [x, y, z, roll, pitch, yaw]
        # M4E planar DOFs: x→0, z→2, pitch→4
        self._idx_mp: np.ndarray = np.array(
            [6 * i + d for i in range(self._nb_mp) for d in (0, 2, 4)],
            dtype=int,
        )

        # Global M4E Cartesian rows/cols for the mooring bodies.
        # involved_bodies[i] is a 1-based M4E body ID.
        # M4E Cartesian ordering: body k (1-based) occupies rows 3*(k-1) : 3*(k-1)+3.
        self._m4e_rows: np.ndarray = np.array(
            [3 * (b_id - 1) + d
             for b_id in self._builder.involved_bodies
             for d in range(3)],
            dtype=int,
        )

        # ------------------------------------------------------------------ #
        # Pre-compute frequency-independent matrices                         #
        # ------------------------------------------------------------------ #
        self._M_cart, self._C_cart, self._K_cart = self._build_cart_matrices()
        self._Fdc_sym = self._build_dc_force()

    # ---------------------------------------------------------------------- #
    # Public adapter API                                                      #
    # ---------------------------------------------------------------------- #

    def frequency_domain_MCKF(
        self,
        omega: Union[float, npt.ArrayLike],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return mooring matrices in M4E body-Cartesian coordinates.

        All MoorPy matrices are frequency-independent.  ``omega`` is accepted
        for API compatibility but does not change the returned values.

        Parameters
        ----------
        omega:
            Frequency [rad/s].  Scalar or 1-D array.

        Returns
        -------
        M_cart : ndarray, shape ``(3·nb_m4e, 3·nb_m4e)``
            Mooring inertia = line structural mass + line added mass.
        C_cart : ndarray, shape ``(3·nb_m4e, 3·nb_m4e)``
            Mooring damping (unit-amplitude approximation, see limitations).
        K_cart : ndarray, shape ``(3·nb_m4e, 3·nb_m4e)``
            Mooring stiffness including geometric stiffness from pretension.
        F : dict
            ``"dc"``     — constant SymPy Matrix ``(3·nb_m4e, 1)``.
            ``"phasor"`` — complex ndarray ``(3·nb_m4e, 1)`` of zeros.
            Mooring lines produce no harmonic external excitation.

        Notes
        -----
        M4E Cartesian coordinate ordering per body: ``[Fx, Fz, My]``.
        Non-mooring M4E bodies receive zero blocks.
        Cross-body mooring K terms are preserved; M/A/B cross-body terms
        are zero due to a limitation in MoorPy's assembly (see module docs).
        """
        Fhat = np.zeros((self._ncart, 1), dtype=complex)
        F: Dict[str, Any] = {"dc": self._Fdc_sym, "phasor": Fhat}
        return self._M_cart, self._C_cart, self._K_cart, F

    # ---------------------------------------------------------------------- #
    # Properties exposing builder state for diagnostics / tests              #
    # ---------------------------------------------------------------------- #

    @property
    def moorpy_system(self) -> Any:
        """The underlying ``moorpy.System`` object."""
        return self._builder.sys

    @property
    def reduction_indices(self) -> np.ndarray:
        """6-DOF indices used to reduce MoorPy matrices to M4E planar DOFs."""
        return self._idx_mp.copy()

    @property
    def m4e_cartesian_rows(self) -> np.ndarray:
        """Global Cartesian row indices for the mooring bodies."""
        return self._m4e_rows.copy()

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _build_cart_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and reduce MoorPy M, A, B, K to M4E Cartesian form.

        Returns
        -------
        M_cart, C_cart, K_cart : ndarray, each ``(3·nb_m4e, 3·nb_m4e)``
            ``M_cart = M + A``, ``C_cart = B``, ``K_cart = K``.
        """
        ms = self._builder.sys
        M_mp, A_mp, B_mp, K_mp = ms.getSystemDynamicMatrices(
            DOFtype="coupled", lines_only=True
        )
        # M_mp, A_mp, B_mp, K_mp each have shape (6·nb_mp, 6·nb_mp)

        idx = self._idx_mp
        M3 = M_mp[np.ix_(idx, idx)]
        A3 = A_mp[np.ix_(idx, idx)]
        B3 = B_mp[np.ix_(idx, idx)]
        K3 = K_mp[np.ix_(idx, idx)]
        # M3, A3, B3, K3 each have shape (3·nb_mp, 3·nb_mp)

        rows = self._m4e_rows  # shape (3·nb_mp,) — rows in global (ncart, ncart) matrix

        M_cart = np.zeros((self._ncart, self._ncart))
        C_cart = np.zeros((self._ncart, self._ncart))
        K_cart = np.zeros((self._ncart, self._ncart))

        M_cart[np.ix_(rows, rows)] = M3 + A3  # mooring inertia = structural + added mass
        C_cart[np.ix_(rows, rows)] = B3       # mooring damping
        K_cart[np.ix_(rows, rows)] = K3       # mooring stiffness

        return M_cart, C_cart, K_cart

    def _build_dc_force(self) -> sym.Matrix:
        """Build the mean mooring wrench as a constant SymPy column vector.

        When :attr:`_include_mean_force` is ``False``, returns a zero matrix
        so the LinearizationManager's Jacobian computation produces zero
        stiffness and damping contributions from this adapter's DC term.

        When ``True``, returns the line-only body wrench evaluated at ``q0``
        reduced to M4E planar DOFs ``[Fx, Fz, My]`` per body.

        Returns
        -------
        sym.Matrix
            Shape ``(3·nb_m4e, 1)``.  Constant entries — no symbolic
            dependence on ``q`` or ``qd``.
        """
        if not self._include_mean_force:
            return sym.zeros(self._ncart, 1)

        ms = self._builder.sys
        # getForces(DOFtype="coupled") assembles forces for all coupled bodies.
        # lines_only=True excludes body weight, buoyancy, and f6Ext.
        # Shape: (6·nb_mp,) ordered as coupled bodies in insertion order.
        f_mp = ms.getForces(DOFtype="coupled", lines_only=True)

        # Reduce to planar DOFs: index [0,2,4] picks [Fx, Fz, My] per body.
        f3 = f_mp[self._idx_mp]  # (3·nb_mp,)

        # Embed into the full global Cartesian vector.
        f_dc = np.zeros(self._ncart)
        f_dc[self._m4e_rows] = f3

        # Return as a constant SymPy matrix (Jacobian w.r.t. q will be zero).
        return sym.Matrix(f_dc.reshape(-1, 1).tolist())

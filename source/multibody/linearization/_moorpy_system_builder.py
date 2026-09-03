# multibody/linearization/_moorpy_system_builder.py
"""
Constructs and caches a MoorPy System at the M4E operating point.

This module is the boundary between M4E kinematics and MoorPy mooring
dynamics.  It evaluates all body and point positions at ``q0``, builds the
MoorPy system, runs the catenary initialization, and prepares the system for
dynamic-matrix extraction.

**No MoorPy equilibrium solve is run** that would move the M4E bodies.
Bodies are fixed at the prescribed ``q0`` by using MoorPy body type ``-1``
(coupled), which means MoorPy treats them as externally controlled.

Coordinate mapping
------------------
M4E operates in the x–z plane (y = 0 out-of-plane).  MoorPy uses a full
3-D coordinate system.  The mapping is:

==================  =============================================
M4E quantity        MoorPy body ``r6`` slot
==================  =============================================
body CG x           ``r6[0]``  (x)
0 (planar)          ``r6[1]``  (y = 0)
body CG z           ``r6[2]``  (z)
0 (planar)          ``r6[3]``  (roll = 0)
body pitch θ        ``r6[4]``  (pitch about y-axis)
0 (planar)          ``r6[5]``  (yaw = 0)
==================  =============================================

Fairlead local coordinates ``BD[body_id][point_id] = [x_local, z_local]``
are passed to MoorPy as ``[x_local, 0.0, z_local]``.  MoorPy's
``Body.getDynamicMatrices`` already transforms the fairlead moment arm into
the body reference-point frame via its internal ``translateMatrix3to6``
helper.  No additional ``H`` matrix multiplication is needed in the adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np


class MoorPySystemBuilder:
    """Build and cache a MoorPy System at the M4E operating point ``q0``.

    Parameters
    ----------
    mbd_sys:
        Compiled :class:`~multibody.multibody_core.mbd_system.MbdSystem`.
    q0:
        Joint-coordinate equilibrium vector, shape ``(nq,)``.
    mainNumVars:
        Full numeric parameter vector for the M4E model, shape
        ``(len(mbd_sys.mainSymVars),)``.  The ``q`` and ``qd`` slots are
        overwritten internally with ``q0`` and zeros — the caller's array is
        not modified.
    moorpy_inputs:
        Validated
        :class:`~multibody.linearization._moorpy_schema.MoorPyInputs`
        describing line types and line topology.

    Raises
    ------
    ImportError
        If MoorPy is not installed.
    AssertionError
        If any fairlead global position does not match M4E within
        :attr:`FAIRLEAD_TOL`.

    Attributes
    ----------
    sys : moorpy.System
        The constructed and initialized MoorPy system.
    involved_bodies : list[int]
        Sorted list of 1-based M4E body IDs that have MoorPy counterparts.
    body_mp_number : dict[int, int]
        Maps each M4E body ID (1-based) to its 1-based position in
        ``sys.bodyList``.  MoorPy body ``sys.bodyList[body_mp_number[bid]-1]``
        corresponds to M4E body ``bid``.
    gr_refs : dict[int, int]
        Maps GR anchor ``point_id`` (0-based) to the MoorPy point number
        (1-based) of the corresponding fixed anchor.
    fl_refs : dict[tuple[int, int], int]
        Maps ``(m4e_body_id, point_id)`` to the MoorPy point number (1-based)
        of the corresponding coupled fairlead.
    """

    #: Tolerance used for fairlead-position assertions [m].
    FAIRLEAD_TOL: float = 1e-4

    def __init__(
        self,
        mbd_sys: Any,
        q0: "np.ndarray",
        mainNumVars: "np.ndarray",
        moorpy_inputs: Any,
    ) -> None:
        try:
            import moorpy  # noqa: F401 (used below via the module reference)
        except ImportError as exc:
            raise ImportError(
                "MoorPy is required for MoorPySystemBuilder. "
                "Install it with:  pip install moorpy"
            ) from exc

        # ------------------------------------------------------------------ #
        # Build the evaluation vector at equilibrium (q = q0, qd = 0)        #
        # ------------------------------------------------------------------ #
        nq = len(mbd_sys.Q)
        eq_vars = np.asarray(mainNumVars, dtype=float).copy()
        eq_vars[:nq] = np.asarray(q0, dtype=float)
        eq_vars[nq : nq + len(mbd_sys.QD)] = 0.0

        # ------------------------------------------------------------------ #
        # Evaluate M4E positions at equilibrium                               #
        # ------------------------------------------------------------------ #
        # Body CG global positions — (NB, 2) array of [x_cg, z_cg].
        cg_pos = np.asarray(mbd_sys.CGpoints_func(*eq_vars), dtype=float)

        # Full body-state vector [x1, z1, θ1, x2, z2, θ2, ...] — (3·NB,).
        pos_vec = np.asarray(mbd_sys.Pos_func(*eq_vars), dtype=float).ravel()

        # GR anchor global positions — list of 1×2 arrays → (n_gr, 2).
        gr_raw = mbd_sys.GRpoints_func(*eq_vars)
        if len(gr_raw) == 0:
            gr_pts: np.ndarray = np.zeros((0, 2), dtype=float)
        else:
            gr_pts = np.asarray(gr_raw, dtype=float).reshape(-1, 2)

        # ------------------------------------------------------------------ #
        # Build the MoorPy System                                             #
        # ------------------------------------------------------------------ #
        ms = moorpy.System(
            depth=moorpy_inputs.depth,
            rho=moorpy_inputs.rho,
            g=moorpy_inputs.g, # TODO: this should come from mbd_sys.g
        )

        # --- Line types --------------------------------------------------- #
        # MoorPy >=1.2 removed addLineType(); setLineType(..., lineType={...})
        # with an explicit dict is the replacement (passing d_vol/mass/EA as
        # direct kwargs hits an upstream UnboundLocalError on d_nom).
        for lt_name, lt in moorpy_inputs.line_types.items():
            ms.setLineType(
                name=lt_name,
                lineType=dict(
                    name=lt_name,
                    d_vol=lt.diameter,
                    m=lt.mass_per_length,
                    EA=lt.axial_stiffness,
                    material="user_defined",
                ),
            )
            # Hydrodynamic coefficients not accepted by setLineType — update dict.
            ms.lineTypes[lt_name]["Cd"] = lt.drag_coefficient
            ms.lineTypes[lt_name]["Ca"] = lt.added_mass_coefficient
            ms.lineTypes[lt_name]["CdAx"] = lt.drag_coefficient_axial
            ms.lineTypes[lt_name]["CaAx"] = lt.added_mass_coefficient_axial

        # --- Bodies -------------------------------------------------------- #
        # Add in sorted M4E body-ID order so that the MoorPy body list index
        # is predictable and the coupled-matrix column/row ordering is
        # determined by the M4E topology, not insertion order.
        involved_bodies: List[int] = sorted(
            {ml.fairlead.body_id for ml in moorpy_inputs.mooring_lines}
        )
        body_mp_number: Dict[int, int] = {}

        for mp_idx, m4e_id in enumerate(involved_bodies, start=1):
            x_cg = float(cg_pos[m4e_id - 1, 0])
            z_cg = float(cg_pos[m4e_id - 1, 1])
            pitch = float(pos_vec[3 * (m4e_id - 1) + 2])
            r6 = np.array([x_cg, 0.0, z_cg, 0.0, pitch, 0.0])
            # type=-1: coupled (externally controlled) — body stays at r6.
            # rCG=zeros: MoorPy reference point coincides with M4E CG.
            ms.addBody(-1, r6, rCG=np.zeros(3))
            body_mp_number[m4e_id] = mp_idx

        # --- Anchor points (GR, type=1 fixed) ----------------------------- #
        # Use a dict so each unique anchor is added only once even when
        # multiple lines share the same anchor.
        gr_refs: Dict[int, int] = {}

        for ml in moorpy_inputs.mooring_lines:
            gid = ml.anchor.point_id
            if gid not in gr_refs:
                x_gr = float(gr_pts[gid, 0])
                z_gr = float(gr_pts[gid, 1])
                ms.addPoint(1, np.array([x_gr, 0.0, z_gr]))
                gr_refs[gid] = len(ms.pointList)  # 1-based number

        # --- Fairlead points (BD, type=1, rigidly attached to body) ------- #
        # r is the body-local position [x_local, 0.0, z_local].
        # type=1 ("fixed") is required here: `body=mp_body` already makes
        # MoorPy's Body.attachPoint slave this point's position to the body,
        # so the point must NOT also be type=-1 ("coupled"), or MoorPy's
        # "coupled" DOF bookkeeping double-counts it as 3 extra independent
        # DOFs on top of the body's 6, producing 6+3=9-sized system matrices
        # instead of 6 for a single moored body (matches MoorPy's own
        # examples, which always use type=1 for body-attached fairleads).
        # MoorPy's Body.setPosition will convert this to global during
        # initialize().  Multiple lines sharing the same fairlead point
        # reuse the existing MoorPy point.
        fl_refs: Dict[Tuple[int, int], int] = {}

        for ml in moorpy_inputs.mooring_lines:
            key = (ml.fairlead.body_id, ml.fairlead.point_id)
            if key not in fl_refs:
                b_id = ml.fairlead.body_id
                p_id = ml.fairlead.point_id
                bd_local = mbd_sys.Initial_Points["BD"][b_id][p_id]  # [x_local, z_local]
                x_loc = float(bd_local[0])
                z_loc = float(bd_local[1])
                mp_body = body_mp_number[b_id]
                ms.addPoint(1, np.array([x_loc, 0.0, z_loc]), body=mp_body)
                fl_refs[key] = len(ms.pointList)  # 1-based number

        # --- Lines -------------------------------------------------------- #
        for ml in moorpy_inputs.mooring_lines:
            a_pt = gr_refs[ml.anchor.point_id]
            b_pt = fl_refs[(ml.fairlead.body_id, ml.fairlead.point_id)]
            ms.addLine(
                ml.unstretched_length,
                ml.line_type,
                nSegs=ml.num_segments,
                pointA=a_pt,
                pointB=b_pt,
            )

        # ------------------------------------------------------------------ #
        # Initialize: solve catenary geometry, hold bodies at q0             #
        # ------------------------------------------------------------------ #
        # initialize() calls:
        #   body.setPosition(body.r6)  → updates attached-point global positions
        #   point.setPosition(point.r) → sets free-point positions
        #   line.staticSolve()         → finds catenary shape
        #   point/body.getForces()     → computes net loads
        # Bodies are type=-1 so they are NOT displaced by solveEquilibrium.
        # We do not call solveEquilibrium here.
        ms.initialize()

        # ------------------------------------------------------------------ #
        # Assert fairlead positions match M4E within tolerance               #
        # ------------------------------------------------------------------ #
        self._check_fairlead_positions(ms, fl_refs, mbd_sys, eq_vars, moorpy_inputs)

        # ------------------------------------------------------------------ #
        # Update line dynamic matrices                                        #
        # ------------------------------------------------------------------ #
        # Required before getSystemDynamicMatrices() can return correct M/A/B/K.
        # omegas=[0.0], S_zeta=[0.0] → frequency-independent operating point;
        # B matrix uses unit-amplitude motions (known limitation).
        ms.updateSystemDynamicMatrices(
            omegas=np.array([0.0]),
            S_zeta=np.array([0.0]),
        )

        # ------------------------------------------------------------------ #
        # Cache public state                                                  #
        # ------------------------------------------------------------------ #
        self.sys = ms
        self.involved_bodies = involved_bodies
        self.body_mp_number = body_mp_number
        self.gr_refs = gr_refs
        self.fl_refs = fl_refs
        self._eq_vars = eq_vars

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _check_fairlead_positions(
        self,
        ms: Any,
        fl_refs: Dict[Tuple[int, int], int],
        mbd_sys: Any,
        eq_vars: "np.ndarray",
        moorpy_inputs: Any,
    ) -> None:
        """Assert each MoorPy fairlead global position matches M4E's BD evaluation.

        This check verifies that:
        * The body was placed at the correct r6.
        * MoorPy's ``rotatePosition`` and M4E's ``rot_2D_wave`` produce
          identical global fairlead positions for the given ``q0``.

        Parameters
        ----------
        ms:
            The MoorPy System after ``initialize()``.
        fl_refs:
            ``(m4e_body_id, point_id)`` → MoorPy point number mapping.
        mbd_sys:
            The M4E system (provides ``BDpoints_func``).
        eq_vars:
            Evaluation vector at equilibrium.
        moorpy_inputs:
            Input definitions (iterated for unique fairlead references).

        Raises
        ------
        AssertionError
            On any mismatch exceeding :attr:`FAIRLEAD_TOL`.
        """
        seen: Set[Tuple[int, int]] = set()
        for ml in moorpy_inputs.mooring_lines:
            key = (ml.fairlead.body_id, ml.fairlead.point_id)
            if key in seen:
                continue
            seen.add(key)

            b_id, p_id = key
            mp_r = ms.pointList[fl_refs[key] - 1].r  # [x, y, z] global after setPosition

            # M4E evaluated global position of this BD point: (n_pts, 2)
            bd_global = np.asarray(mbd_sys.BDpoints_func[b_id](*eq_vars), dtype=float)
            expected = bd_global[p_id]  # [x_global, z_global]

            # MoorPy x and z components (y = 0 in planar configuration)
            actual = np.array([float(mp_r[0]), float(mp_r[2])])

            if not np.allclose(actual, expected, atol=self.FAIRLEAD_TOL):
                raise AssertionError(
                    f"Fairlead position mismatch for body={b_id}, point={p_id}:\n"
                    f"  M4E evaluated  : [x={expected[0]:.6f} m, z={expected[1]:.6f} m]\n"
                    f"  MoorPy computed: [x={actual[0]:.6f} m, z={actual[1]:.6f} m]\n"
                    f"  tolerance = {self.FAIRLEAD_TOL} m"
                )

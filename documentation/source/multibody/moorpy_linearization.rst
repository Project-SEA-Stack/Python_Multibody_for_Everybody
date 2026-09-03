MoorPy mooring dynamics adapter
================================

The MoorPy adapter adds quasi-static and linearised mooring dynamics to the
:class:`~multibody.linearization.linearization_main.LinearizationManager`
frequency-domain assembly.  It implements the
:class:`~multibody.linearization.linearization_main.LinearizationAdapter`
protocol and is entirely optional — the base ``multibody`` package does not
depend on MoorPy.

Architecture
------------

::

    M4E Initial_Points
        ├── GR anchors       → MoorPy fixed points (type = 1)
        └── BD fairleads     → MoorPy coupled points (type = −1, attached to body)

    MoorPy input module (moorpy_inputs.py)
        ├── line_types       (diameter, mass, stiffness, drag/added-mass coefficients)
        ├── mooring_lines    (name, type, length, segments, anchor ref, fairlead ref)
        └── point references (by M4E GR/BD IDs — no duplicated coordinates)

    MoorPySystemBuilder
        ├── evaluates M4E body CG positions at q0
        ├── evaluates M4E BD/GR positions at q0
        ├── constructs MoorPy bodies, fixed points, coupled fairleads
        ├── calls System.initialize()  (catenary solve, internal DOFs only)
        └── calls System.updateSystemDynamicMatrices()

    MoorPyLinearMCKF.frequency_domain_MCKF(ω)
        ├── calls System.getSystemDynamicMatrices("coupled", lines_only=True)
        │       → M, A, B, K  each (nCpldDOF × nCpldDOF)
        │         nCpldDOF = 6·nb_bodies + 3·nb_attached_points
        ├── extracts body block with idx = [6i+d  for i in range(nb)  for d in (0,2,4)]
        │       → M3, A3, B3, K3  each (3·nb × 3·nb)
        ├── maps into M_cart = M3 + A3,  C_cart = B3,  K_cart = K3
        │       embedded in (3·nb_m4e × 3·nb_m4e) with zero rows for non-mooring bodies
        └── returns M_cart, C_cart, K_cart, F{"dc": SymPy, "phasor": zeros}

    LinearizationManager
        └── body Cartesian → M4E joint coordinates
              Mq = R0ᵀ M_cart R0
              Cq = R0ᵀ (C_cart R0 + M_cart RD0)
              Kq = R0ᵀ K_cart R0

Architectural guarantees
------------------------

The following statements are invariants of the implementation.

1. **MoorPy performs the fairlead-to-body transformation.**
   ``Body.getDynamicMatrices`` calls MoorPy's internal
   ``translateMatrix3to6`` using the body-local fairlead offset stored in
   ``body.rPointRel``.  The resulting body matrix already accounts for the
   moment arm.

2. **MoorPy includes fairlead moment arms in body forces and moments.**
   ``Body.getForces`` similarly reduces line end-forces and torques to the
   body reference point.  The mean wrench returned by the adapter therefore
   includes translation–rotation coupling due to the fairlead offset.

3. **MoorPy body-level stiffness includes geometric contributions.**
   ``K`` returned by ``getSystemDynamicMatrices`` accounts for the change in
   catenary geometry under small body displacements, including the
   pretension contribution.

4. **The M4E adapter does not reapply a fairlead transformation.**
   No additional ``H`` matrix is applied.  Doing so would double-count the
   fairlead offset that MoorPy already applied internally.

5. **Reduction uses local DOF indices** ``[0, 2, 4]``.
   MoorPy's 6-DOF body ordering is
   ``[x, y, z, roll, pitch, yaw]``.
   M4E planar DOFs correspond to indices 0 (surge), 2 (heave), 4 (pitch).
   For ``nb`` mooring bodies::

       idx = [6*i + d  for i in range(nb)  for d in (0, 2, 4)]

6. **MoorPy body reference points coincide with M4E CGs.**
   ``System.addBody`` is called with ``rCG = np.zeros(3)`` so the MoorPy
   body reference frame origin coincides with the CG.  No reference-point
   translation is required in the adapter.

7. **q0 is externally supplied and remains unchanged.**
   The adapter reads ``q0`` once at construction time.  MoorPy's
   equilibrium solver is not invoked; body positions are held fixed at the
   prescribed ``q0``.

8. **Mean mooring force is optional.**
   ``include_mean_force=False`` (default) returns a zero SymPy DC force.
   When ``True``, the mean line-induced body wrench at ``q0`` is returned
   as a constant symbolic vector for equilibrium-residual checking.  It
   does not modify ``q0`` and does not double-count stiffness already
   present in ``K``.

9. **MoorPy is an optional dependency.**
   ``import multibody`` never triggers a MoorPy import.  Only instantiating
   :class:`~multibody.linearization.moorpy_linear_mckf.MoorPyLinearMCKF`
   raises an ``ImportError`` if MoorPy is not installed.

10. **Experimental limitations of MoorPy's dynamic matrices are identified.**
    See the :ref:`moorpy-limitations` section below.

.. _moorpy-matrix-semantics:

Matrix semantics
----------------

.. list-table::
   :header-rows: 1
   :widths: 12 14 40 20

   * - Symbol
     - Units
     - Description
     - MoorPy source
   * - ``M_cart``
     - kg
     - Line structural inertia + line added mass (``M + A``)
     - ``M``, ``A`` from ``getSystemDynamicMatrices``
   * - ``C_cart``
     - kg/s
     - Mooring line viscous drag damping
     - ``B`` from ``getSystemDynamicMatrices``
   * - ``K_cart``
     - N/m or N·m/rad
     - Mooring stiffness including geometric contribution
     - ``K`` from ``getSystemDynamicMatrices``
   * - ``F["dc"]``
     - N or N·m
     - Mean line-induced body wrench at ``q0`` (optional)
     - ``getForces(DOFtype="coupled", lines_only=True)``
   * - ``F["phasor"]``
     - N or N·m
     - Harmonic mooring excitation — zero (no external wave load)
     - n/a

Cartesian coordinate ordering per body: ``[Fx, Fz, My]``
(corresponding to DOFs ``[x, z, pitch]``).

.. _moorpy-limitations:

Known limitations
-----------------

* **Damping approximation.**  ``B`` (and hence ``C_cart``) is computed
  assuming unit-amplitude motion of all line nodes.  For physically accurate
  mooring damping, RAOs must be provided to
  ``line.updateLumpedMass`` before calling
  ``updateSystemDynamicMatrices``.  This limitation is inherited from
  MoorPy's ``updateSystemDynamicMatrices`` API.

* **Cross-body M/A/B not assembled.**  ``getSystemDynamicMatrices``
  assembles cross-body coupling only for ``K``.  Off-diagonal
  ``M``, ``A``, and ``B`` blocks between different mooring bodies are zero.
  Cross-body ``K`` coupling is preserved.

* **nCpldDOF includes attached fairlead points.**  For a system with one
  coupled body and one attached fairlead point, ``getSystemDynamicMatrices``
  returns a 9×9 matrix (6 body DOFs + 3 attached-point DOFs).  The
  reduction indices ``[0, 2, 4]`` correctly address the body block only.
  The attached-point block is not used.

* **BD-to-BD mooring lines not supported.**  The current schema only allows
  lines from a GR anchor to a BD fairlead.  Body-to-body lines (both
  endpoints on M4E bodies) are not supported and will fail schema
  validation.

* **MoorPy dev branch required.**  ``System.updateSystemDynamicMatrices``
  and ``System.getSystemDynamicMatrices`` are not present in MoorPy 1.1.0
  (PyPI).  The adapter requires MoorPy installed from the ``dev`` branch::

      pip install git+https://github.com/NatLabRockies/MoorPy.git@dev

  or the local editable install used during development::

      pip install -e /path/to/MoorPy

Numerical validation procedure
--------------------------------

The following five-step chain verifies the adapter end-to-end.  It is
implemented in
``Examples_linearization/single_body_moorpy/main_linearization.py``
and executed automatically when the example is run.

.. code-block:: text

    Step 1  MoorPy full body matrix
            ms.getSystemDynamicMatrices("coupled", lines_only=True)
            → M_mp, A_mp, B_mp, K_mp   shape (nCpldDOF × nCpldDOF)

    Step 2  Independent manual reduction
            idx = [0, 2, 4]
            M3 = (M_mp + A_mp)[np.ix_(idx, idx)]
            C3 = B_mp[np.ix_(idx, idx)]
            K3 = K_mp[np.ix_(idx, idx)]

    Step 3  Adapter Cartesian output
            adapter.frequency_domain_MCKF(ω) → M_cart, C_cart, K_cart
            assert np.allclose(M_cart, M3)   # embedded in ncart×ncart zeros
            assert np.allclose(C_cart, C3)
            assert np.allclose(K_cart, K3)

    Step 4  Manager joint-coordinate output
            totals = manager.assemble_frequency_domain(ω)
            M_joint = totals.M,  C_joint = totals.C,  K_joint = totals.K

    Step 5  Independent R0ᵀ X_cart R0
            assert np.allclose(R0.T @ M_cart @ R0,  M_joint - M_mbd)
            assert np.allclose(R0.T @ C_cart @ R0,  C_joint - C_mbd)
            assert np.allclose(R0.T @ K_cart @ R0,  K_joint - K_mbd)

Usage example
-------------

.. code-block:: python

    import numpy as np
    from multibody import MbdSystem
    from multibody.linearization.linearization_main import LinearizationManager
    from multibody.linearization._moorpy_schema import from_module, validate_moorpy_inputs
    from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF

    from Examples_linearization.single_body_moorpy import M4E_inputs as ex
    from Examples_linearization.single_body_moorpy import moorpy_inputs as mp_mod

    # Parse and validate mooring topology
    moorpy_inputs = from_module(mp_mod)
    validate_moorpy_inputs(moorpy_inputs, ex.Initial_Points)

    # Build MBD system and operating point
    mbd_sys = MbdSystem.from_example(ex)
    q0 = (mbd_sys.ic - ex.ic)[: len(mbd_sys.Q)]
    mainNumVars = ex.ic.copy()

    # Instantiate adapter and register with manager
    adapter = MoorPyLinearMCKF(mbd_sys, q0, mainNumVars, moorpy_inputs)
    manager = LinearizationManager(mbd_sys, q0, mainNumVars, ex.m0, ex.J0)
    manager.register(adapter)

    # Assemble frequency-domain totals
    totals = manager.assemble_frequency_domain(omega=0.5)

    # Inspect joint-coordinate matrices
    nq = manager.nq
    K_joint = np.asarray(totals.K).reshape(nq, nq)
    print("K_joint =", K_joint)

Minimum working example: ``Examples_linearization/single_body_moorpy/``

New modules added
-----------------

.. code-block:: bash

   source/multibody/linearization/
   ├── _moorpy_schema.py          # Input schema dataclasses and validation
   ├── _moorpy_system_builder.py  # Constructs MoorPy System at q0
   └── moorpy_linear_mckf.py      # LinearizationAdapter implementation

   Examples_linearization/single_body_moorpy/
   ├── M4E_inputs.py              # MBD topology (one floating body)
   ├── moorpy_inputs.py           # Mooring line type and topology
   └── main_linearization.py      # Runnable example with validation checks

   tests/cases/linearization/
   ├── test_moorpy_schema.py      # 33 schema + validation tests
   ├── test_manager_adapter.py    # 24 manager/adapter protocol tests
   └── test_moorpy_full.py        # 39 point mapping, reduction, coupling tests

API reference
-------------

.. autosummary::
   :toctree: ../_autosummaries
   :nosignatures:

   multibody.linearization.moorpy_linear_mckf.MoorPyLinearMCKF
   multibody.linearization._moorpy_schema.MoorPyInputs
   multibody.linearization._moorpy_schema.LineType
   multibody.linearization._moorpy_schema.MooringLine
   multibody.linearization._moorpy_schema.PointRef
   multibody.linearization._moorpy_schema.from_module
   multibody.linearization._moorpy_schema.validate_moorpy_inputs

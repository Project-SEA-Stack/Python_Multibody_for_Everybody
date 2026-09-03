Linearization package (M4E v2.0)
================================

The ``linearization`` package linearizes the nonlinear equations of motion about
an equilibrium point, assembles frequency-domain system matrices when external
force adapters are registered, and provides a compiled time integrator for the
linearized system.

Two usage levels are provided:

Level 1 — symbolic linearization only
--------------------------------------

:func:`~multibody.linearize_mbd` takes an assembled :class:`~multibody.MbdSystem`
and an equilibrium configuration and returns a
:class:`~multibody.linearization.linearize_eom.LinearizedMBD` dataclass whose four
fields (``Mbar``, ``Cbar``, ``Kbar``, ``Fbar``) are NumPy-ready callables.
The linearization solves

.. math::

   \bar{M}(q_0)\,\delta\ddot{q}
   + \bar{C}(q_0)\,\delta\dot{q}
   + \bar{K}(q_0)\,\delta q
   = \bar{F}

where :math:`\bar{M} = B^T M B` and :math:`B = R\lvert_{q_0}`.

Level 2 — frequency-domain assembly with external adapters
----------------------------------------------------------

:class:`~multibody.linearization.linearization_main.LinearizationManager` wraps
Level 1 and coordinates one or more
:class:`~multibody.linearization.linearization_main.LinearizationAdapter` objects
that supply frequency-dependent added mass, radiation damping, hydrostatic
stiffness and wave excitation.

The typical workflow is:

.. code-block:: python

   from multibody import MbdSystem
   from multibody.linearization.linearization_main import LinearizationManager
   from multibody.linearization.hydro_linear_mckf import HydroLinearMCKF

   MBD = MbdSystem.from_example(ex)
   q0  = MBD.ic[:len(MBD.Q)]

   lm = LinearizationManager(MBD, q0, mainNumVars, m0, J0)
   lm.register(HydroLinearMCKF(MBD, (mainNumVars, m0, J0), omega_r=omega, body_inputs=body_inputs, ...))

   fd  = lm.assemble_frequency_domain(omega)   # FrequencyDomainTotals
   lm.compile_operating_point(omega0)
   sol = lm.integrate_linear_system(tspan=200.0, dt=0.05)

``integrate_linear_system`` returns a ``scipy.integrate.OdeSolution`` object.

Custom adapters
---------------

Any object satisfying the
:class:`~multibody.linearization.linearization_main.LinearizationAdapter` protocol
can be registered.  The protocol requires:

- a ``name`` attribute (unique string),
- a ``frequency_domain_MCKF(omega)`` method returning ``(M, C, K, F)`` where
  ``F`` is a dict with keys ``"dc"`` (symbolic or numeric DC force) and
  ``"phasor"`` (complex excitation).

Hydrodynamic coupling — regular waves
--------------------------------------

:class:`~multibody.linearization.hydro_linear_mckf.HydroLinearMCKF` implements
the adapter protocol for **linear potential-flow hydrodynamics under regular
(monochromatic) waves**.  It:

- runs or loads a BEM solution via
  `Capytaine <https://github.com/capytaine/capytaine>`_,
- provides frequency-dependent added mass :math:`A(\omega)` and radiation
  damping :math:`B(\omega)` matrices,
- computes hydrostatic stiffness :math:`K_{hs}` and buoyancy forces,
- returns complex wave excitation phasors (Froude-Krylov + diffraction) scaled
  by wave amplitude.

Problem statement
^^^^^^^^^^^^^^^^^

The module solves the linearized equations of motion for a floating multibody
system subject to a **single regular wave** of angular frequency
:math:`\omega_0` and amplitude :math:`A`.  Working in the Cartesian space of
all bodies and projecting onto joint coordinates through the velocity
transformation matrix :math:`R_0 = B\lvert_{q_0}`, the assembled
frequency-domain system is

.. math::

   \bigl[\bar{M} + \bar{A}(\omega)\bigr]\,\delta\ddot{q}
   + \bigl[\bar{C} + \bar{B}(\omega)\bigr]\,\delta\dot{q}
   + \bigl[\bar{K} + \bar{K}_{hs}\bigr]\,\delta q
   = \operatorname{Re}\!\left(R_0^T\,\hat{F}(\omega)\,e^{-i\omega t}\right)

where:

- :math:`\bar{M} = R_0^T M_{\text{rigid}} R_0` — rigid-body inertia in joint
  coordinates,
- :math:`\bar{A}(\omega) = R_0^T A(\omega) R_0` — frequency-dependent added
  mass projected onto joint coordinates,
- :math:`\bar{B}(\omega) = R_0^T B(\omega) R_0` — radiation damping projected
  onto joint coordinates,
- :math:`\bar{K}` and :math:`\bar{K}_{hs}` — structural and hydrostatic
  stiffness in joint coordinates,
- :math:`\hat{F}(\omega)` — complex excitation phasor (Froude-Krylov +
  diffraction) at wave amplitude :math:`A`, wave direction 0 rad (head-on).

The Cartesian state per body uses the **planar ordering**
:math:`[F_x,\, F_z,\, M_y]`; 6-DOF BEM results are reduced to this 3-DOF
subset before projection.

The time-domain signal reconstructed by
:meth:`~multibody.linearization.hydro_linear_mckf.HydroLinearMCKF.assemble_time_forces_from_linear_solution`
is

.. math::

   F(t) = \operatorname{Re}\!\left(\hat{F}(\omega_0)\,e^{-i\omega_0 t}\right)\,r(t)
          - B(\omega_0)\,\dot{x}(t)
          - K_{hs}\,x(t)
          + F_b

where :math:`r(t)` is an optional cosine ramp and :math:`F_b` is the static
buoyancy force.

Hydrodynamic coefficient convention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All dimensional BEM quantities (added mass, radiation damping, excitation
force, hydrostatic stiffness) are stored with the units produced by Capytaine
and are **proportional to the fluid density** :math:`\rho` used in the BEM
solve.  When the simulation density differs from the density stored in the
``.nc`` file, every coefficient is rescaled by :math:`\rho_\text{user} /
\rho_\text{nc}`.  This is a **linear proportional rescaling** — it is not the
same non-dimensionalization that tools such as WEC-Sim apply.

Limitations
^^^^^^^^^^^

- **Linear regime only.** The theory assumes small-amplitude waves and small
  motions about the equilibrium :math:`q_0`.  Results degrade for large wave
  amplitudes or large excursions from equilibrium.
- **Regular (monochromatic) waves.** Only a single wave frequency and direction
  (0 rad, head-on) are supported.  Irregular sea states require superposition
  outside this module.
- **Dimensional coefficients.** Added mass, damping, stiffness, and excitation
  forces are kept dimensional.  Non-dimensionalization (e.g. by
  :math:`\rho g A`) is not performed automatically.
- **2-D planar embedding.** The full 6-DOF BEM solution is reduced to the
  in-plane degrees of freedom :math:`[F_x, F_z, M_y]`.  Out-of-plane forces
  are discarded.
- **Potential flow only.** Viscous drag (Morison equation) is not included.
- **Equilibrium-point linearization.** The velocity transformation matrix
  :math:`R_0` is evaluated once at :math:`q_0`.  It does not update as the
  system evolves.

.. autosummary::
   :toctree: ../_autosummaries
   :nosignatures:

   multibody.linearize_mbd
   multibody.linearization.linearize_eom.LinearizedMBD
   multibody.linearization.linearization_main.LinearizationManager
   multibody.linearization.linearization_main.LinearizationAdapter
   multibody.linearization.hydro_linear_mckf.HydroLinearMCKF

MoorPy mooring adapter
-----------------------

The :class:`~multibody.linearization.moorpy_linear_mckf.MoorPyLinearMCKF`
adapter couples MoorPy quasi-static / linearised mooring dynamics with the
``LinearizationManager``.  MoorPy is an **optional** dependency.

.. toctree::
   :maxdepth: 1

   moorpy_linearization

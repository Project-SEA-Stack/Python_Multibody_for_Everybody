.. _Developer's documentation:

Developer's documentation
=========================

The developer’s documentation provides guidance for anyone who wants to extend or maintain the Multibody4Everybody codebase.
It describes the internal architecture, public API layout, and validation routines that ensure numerical and symbolic correctness.
Developers can use this section to understand how the core modules interact, where to implement new functionality, and how to run the built-in validation and energy comparison tests to verify changes.

For a detailed overview of the v1.0 capabilities please refer to `M4E publication <https://www.mdpi.com/2075-1702/14/2/145>`_.

Project Layout
**************

The following shows the structure of the ``multibody`` package:

.. code-block:: bash

   multibody/
   ├── checks/
   │   ├── body_check.py
   │   └── points_forces_check.py
   ├── ext_forces_manager/
   │   ├── coupling_main.py
   │   ├── moordyn_adapter.py
   │   ├── hydro_adapter.py
   │   ├── external_class_template.py
   │   └── moordyn_writter.py
   ├── multibody_core/
   │   ├── _joints_helpers.py
   │   ├── joints_system.py
   │   ├── kinematics_transformation.py
   │   ├── mbd_system.py
   │   ├── points_forces_definition.py
   │   ├── rigid_body_integrator.py
   │   ├── _springs_dampers_helpers.py
   │   ├── symvars_py.py
   │   ├── post_process.py
   │   └── systems_energy.py
   ├── plotting/
   │   ├── plotting_main.py
   │   └── plotting_helper.py
   ├── linearization/
   │   ├── linearize_eom.py
   │   ├── linearization_main.py
   │   ├── linear_integrator.py
   │   ├── hydro_linear_mckf.py
   │   ├── _moorpy_schema.py          # MoorPy input schema and validation (no MoorPy import)
   │   ├── _moorpy_system_builder.py  # Constructs MoorPy System at q0
   │   └── moorpy_linear_mckf.py     # MoorPy LinearizationAdapter (optional dep.)
   ├── tables/
   │   └── points_forces_table.py
   ├── yaml_parser/
   │   ├── yaml_adapter.py
   │   └── simulation_parser.py
   └── validation/
      ├── automated_validation.py
      └── energy_and_ode_comparison.py

Public API summary
******************

The following symbols are exposed by default when importing the ``multibody`` package:

.. code-block:: python

    from multibody import (
      # High-level interface
      MbdSystem,              # MultiBody dynamics class 
                              # with all symbolic information
      integrate_dynamics,     # Integration in time
      ExternalForcesManager,  # Enables coupling external libraries 
                              # providing forces to the MBD code

      # Core classes & routines
      JointSystem,            # Creates a class containing all bodies 
                              # information, connectivity and variables
      normalize_prismatic,    # Ensures the vectors of prismatic joints
                              # are unit length
      VelocityTransformation, # Computes Velocity transformation matrix and its
                              # derivative, CG positions and joint positions
      points_force_finder,    # Assembles the generalized force vector 
                              # and points-forces mapping
      systems_energy,         # Computes system energy, assembles mass matrix
      symvars_definition,     # Creates the symbolic variables defined in the example
      evaluate_trajectories,  # Post-processes integration results

      # Linearization & YAML
      linearize_mbd,          # Linearizes EOM about equilibrium
      load_yaml_as_example,   # Loads *.model.yaml + *.simulation.yaml
      
      # Submodules (namespaces)
      plot,                   # Quick access to plotting tools            
      checks,                  # Quick access to checking tools
      tables,                 # Quick access to table-display tools
   )


Multibody system creation workflow
**********************************

The following workflow outlines the main steps followed by the multibody code. This
is not necessary

.. image:: /_static/figs/main_workflow.png
   :alt: MBD workflow
   :width: 70%
   :align: center

Tests
*****

The test suite lives entirely under ``tests/`` and is driven by `pytest <https://docs.pytest.org>`_.
All configuration is in ``pyproject.toml`` (``testpaths = ["tests"]``).
Truth data (reference outputs) are committed alongside the source so no
external tools or network access are required.

Test layout
-----------

.. code-block:: bash

   tests/
   ├── metrics.py                      # Shared NRMSE helpers
   ├── cases/
   │   ├── catalog.py                  # Shared MATLAB case registry (cases 1–8)
   │   ├── matlab/                     # Symbolic kinematics + energy tests
   │   │   ├── test_automated_validation.py
   │   │   ├── test_energy_and_ode.py
   │   │   └── truth/
   │   │       ├── symbolic/           # Example{1-8}.json  (MATLAB reference)
   │   │       └── energy/             # Em_example{1-8}.mat
   │   ├── chrono/                     # Chrono regression tests (9 cases)
   │   │   ├── test_chrono_cases.py
   │   │   ├── catalog.py
   │   │   ├── runner.py
   │   │   └── truth/<case_id>/        # t_chrono.npy, r_chrono.npy, v_chrono.npy
   │   ├── yaml/                       # YAML-parser regression tests (7 cases)
   │   │   ├── test_yaml_cases.py
   │   │   ├── catalog.py
   │   │   ├── runner.py
   │   │   ├── models/                 # *.model.yaml + *.simulation.yaml
   │   │   └── truth/<case_id>/        # trajectory.csv (PyChrono reference)
   │   └── linearization/             # Linearization tests
   │       ├── test_lin_vs_nonlinear.py   # 3 cases: lin ≈ NL for small IC
   │       ├── test_foswec_rao.py         # 6 cases: RAO vs WEC-Sim (slow)
   │       ├── test_foswec_td.py          # 1 case:  time-domain vs WEC-Sim (slow)
   │       ├── test_moorpy_schema.py      # 33 cases: schema validation, no MoorPy needed
   │       ├── test_manager_adapter.py    # 24 cases: manager protocol + MoorPy smoke
   │       ├── test_moorpy_full.py        # 39 cases: point mapping, DOF reduction, coupling
   │       ├── catalog.py
   │       ├── runner.py
   │       ├── foswec_runner.py
   │       ├── foswec/                    # FOSWEC example inputs + BEM data
   │       └── truth/                     # RAO_matlab.txt, results_wecSim.mat

Suite descriptions
------------------

``tests/cases/matlab/``
   Validates symbolic kinematics and energy conservation for eight canonical
   examples against MATLAB-generated reference data.

   * **test_automated_validation** — checks CG positions, joint locations,
     and the :math:`R` / :math:`\dot{R}` matrices for Examples 1–8 against
     ``.json`` files in ``truth/symbolic/``.  A non-zero error means a
     kinematic formula has changed.
   * **test_energy_and_ode** — integrates each system and compares the total
     mechanical energy trace against ``.mat`` files in ``truth/energy/``.
     Typical RMSE values are zero for conservative cases and below
     :math:`10^{-5}` for the flexible/slider cases.  The tolerance is
     :math:`10^{-4}`.

``tests/cases/chrono/``
   Runs nine M4E multibody models and compares position and velocity
   trajectories against PyChrono simulation results stored as ``.npy`` files.
   Tolerances: NRMSE ≤ 1 % (position) and ≤ 1.5 % (velocity).

``tests/cases/yaml/``
   Verifies the YAML parser by loading seven models from ``.model.yaml`` +
   ``.simulation.yaml`` pairs, integrating them, and comparing against
   PyChrono reference trajectories stored as ``trajectory.csv`` files.
   Tolerances: NRMSE ≤ 1 × 10⁻⁴ (position) and ≤ 1 × 10⁻³ (velocity).

``tests/cases/linearization/``
   Three test files covering the linearization workflow:

   * **test_lin_vs_nonlinear** — integrates each model twice (full nonlinear
     and linearized about the rest configuration) and asserts that position
     NRMSE ≤ 5 × 10⁻³ and velocity NRMSE ≤ 10⁻².
   * **test_foswec_rao** *(slow)* — computes the time-domain RAO at five wave
     frequencies via the ``LinearizationManager`` and compares against a
     WEC-Sim MATLAB reference stored in ``truth/RAO_matlab.txt``.  Platform
     DOF tolerances are ≤ 3 %; flap DOFs ≤ 20 %.
   * **test_foswec_td** *(slow)* — integrates the full FOSWEC system in the
     time domain and checks position NRMSE against the WEC-Sim reference
     trajectory in ``truth/results_wecSim.mat``.
   * **test_moorpy_schema** — 33 tests covering schema dataclasses, ID
     validation, and reference checking.  Does not require MoorPy.
   * **test_manager_adapter** — 24 tests covering ``LinearizationManager``
     matrix transforms (Mq, Cq, Kq), ``detect_space``, DC-force handling, and
     a MoorPy smoke test verifying adapter instantiation and output shapes.
   * **test_moorpy_full** — 39 tests covering point mapping, 6→3 DOF
     reduction, absence of double H-transform, cross-body coupling, mean
     force ordering, and end-to-end real-MoorPy smoke tests.

Running the tests
-----------------

From the repository root:

.. code-block:: bash

   # Fast suite only (excludes slow hydrodynamic tests, ~40 s)
   pytest -m "not slow"

   # Full suite including slow tests
   pytest

   # A single sub-suite
   pytest tests/cases/matlab/
   pytest tests/cases/chrono/
   pytest tests/cases/yaml/
   pytest tests/cases/linearization/

   # Verbose output
   pytest -v

The ``slow`` marker is applied to any test that takes more than a few seconds
(currently the FOSWEC RAO and time-domain tests).  Deselect them during
routine development with ``-m "not slow"``.

API
***

.. toctree::
   :maxdepth: 2

   multibody_core
   plotting
   checks
   tables
   ext_forces_manager
   yaml_parsers
   linearization
   validation

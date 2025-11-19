.. _Developer's documentation:

Developer's documentation
=========================

The developer’s documentation provides guidance for anyone who wants to extend or maintain the Multibody4Everybody codebase.
It describes the internal architecture, public API layout, and validation routines that ensure numerical and symbolic correctness.
Developers can use this section to understand how the core modules interact, where to implement new functionality, and how to run the built-in validation and energy comparison tests to verify changes.

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
   ├── tables/
   │   └── points_forces_table.py
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

PyTests
*******

In case of contributing to this project, there are two files that must be run to check everything
is working as it should. These files are under ``multibody.validation``. The first file, 
``automated_validation``, checks that the position of the CGs for each body is correct as well as 
the joint location and the :math:`R` and :math:`\dot{R}` matrices for Examples 1-8. The comparison is done to
the True values stored in ``.json`` files under **TrueMBDvars** folder in the same location. 
The second set of checks, compares the energy of the system to a set of true values saved in **TrueEnergy**.
In this case the variables are saved as ``.mat`` files. This file is called ``energy_and_ode`` and it
should provide values around or below :math:`10^{-5}`

Both these files can be run directly from this location. 

In summary, from the root do the following:

.. code-block:: bash

   cd source/multibody/validation
   python automated_validation.py
   python energy_and_ode.py

Whether you use forward or backward slash will depend on the terminal and OS you use. 

API
***

.. toctree::
   :maxdepth: 2

   multibody_core
   plotting
   checks
   tables
   ext_forces_manager
   validation

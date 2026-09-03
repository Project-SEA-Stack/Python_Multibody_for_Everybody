YAML parser package (M4E v2.0)
==============================

The ``yaml_parser`` package translates ``*.model.yaml`` and ``*.simulation.yaml``
input files into the ``SimpleNamespace`` expected by ``MbdSystem.from_example()``.
This allows users to define multibody systems declaratively without writing Python
example modules.

Both files are required and must share the same base path (e.g.
``my_model.model.yaml`` and ``my_model.simulation.yaml``).

The entry point is :func:`~multibody.load_yaml_as_example`.  Internally it
instantiates :class:`~multibody.yaml_parser.yaml_adapter.YAML2Example`, which:

1. anchors the system to a fixed body or synthesises a world anchor,
2. builds a directed body graph and assigns topological indices,
3. emits joints in parent–child order,
4. assembles the kinematic lists expected by :class:`~multibody.JointSystem`,
5. reads mass and out-of-plane inertia :math:`I_y` per body,
6. builds initial conditions from declared positions and velocities,
7. parses spring–dampers (``rsdas`` / ``tsdas``), body loads and motors,
8. validates the model with the :mod:`multibody.checks` package.

The simulation file (``*.simulation.yaml``) provides ``time_step``, ``end_time``
and optional ``gravity``; these are appended to the namespace as ``TimeStep``,
``tspan`` and ``g``.

.. note::

   The parser enforces a **2-D constraint**: all body CG Y-coordinates must be
   zero and orientations must be aligned with the out-of-plane axis. Alternatively, `helper.py` 
   can be modified to specify what are the working axis, updating the variable `AXIS_2D`. In the 
   future, this should be an input parameter.  Models imported from 3-D CAD tools 
   may need adjustment before parsing.

Minimum model file structure
----------------------------

.. code-block:: yaml

   model:
     bodies:
       - name: ground
         fixed: true
         location: [0, 0, 0]
       - name: link
         location: [0, 0, -0.5]
         mass: 1.0
         inertia:
           moments: [0.083, 0.083, 0.001]
     joints:
       - name: hinge
         type: revolute
         body1: ground
         body2: link
         location: [0, 0, 0]

Minimum simulation file structure
---------------------------------

.. code-block:: yaml

   simulation:
     time_step: 0.01
     end_time: 10.0
     gravity: [0, 0, -9.81]

Usage
-----

.. code-block:: python

   from multibody import load_yaml_as_example, MbdSystem

   ex  = load_yaml_as_example("yaml_examples/double_pendulum")
   MBD = MbdSystem.from_example(ex)

.. autosummary::
   :toctree: ../_autosummaries
   :nosignatures:

   multibody.load_yaml_as_example
   multibody.yaml_parser.yaml_adapter.YAML2Example

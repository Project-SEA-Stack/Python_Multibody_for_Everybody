MultibodyCore package
=====================

This package contains the core multibody-dynamics engine: system setup, kinematics  
and symbolic equation generation. It also creates lambdified expressions for performance.
The core elemtent of this package is the MBDsystem class. The class is initialized:

- from_example [method]: reads an example where all the joints and connectivity is defined

.. autosummary::
   :toctree: ../_autosummaries
   :nosignatures:

   multibody.MbdSystem
   multibody.JointSystem
   multibody.normalize_prismatic
   multibody.VelocityTransformation
   multibody.points_force_finder
   multibody.systems_energy
   multibody.symvars_definition
# source/multibody/__init__.py

# — Core multibody API now lives in multibody_core/ --------------
from .multibody_core.joints_system              import JointSystem, normalize_prismatic
from .multibody_core.symvars                    import symvars_definition
from .multibody_core.kinematics_transformation  import VelocityTransformation
from .multibody_core.points_forces_definition   import points_force_finder
from .multibody_core.systems_energy             import systems_energy
from .multibody_core.mbd_system                 import MbdSystem
from .multibody_core.rigid_body_integrator      import integrate_dynamics
from .multibody_core.post_process               import evaluate_trajectories
from .ext_forces_manager.coupling_main          import ExternalForcesManager
from .yaml_parser.yaml_adapter                  import load_yaml_as_example

# — Plotting namespace -------------------------------
from . import plotting as plot

# — Verification/checks ----------------------
from . import checks 

# — Tables (e.g., inertia) ----------------------
from . import tables 

# — Public API ---------------------------------------------------
__all__ = [
  # high-level
  "MbdSystem",
  "integrate_dynamics",
  "ExternalForcesManager",
  # low-level
  "JointSystem",
  "normalize_prismatic",
  "VelocityTransformation",
  "points_force_finder",
  "systems_energy",
  "symvars_definition",
  "evaluate_trajectories",
  "load_yaml_as_example",
  # namespaces
  "plot",
  "checks",
  "tables",
]

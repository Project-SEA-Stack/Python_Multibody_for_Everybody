# Examples_linearization/single_body_moorpy/M4E_inputs.py
"""
M4E system definition for the single_body_moorpy example.

System
------
A single floating body moored by one catenary line.

Bodies
------
Body 1 — buoy, CG at the mean free surface (x=0, z=0 at equilibrium).
  mass:            2500 kg
  pitch inertia:   2500 kg·m²  (radius of gyration ≈ 1 m)

Joints
------
One 'F' (floating) joint connecting the inertial ground (body 0) to the buoy.

Degrees of freedom
------------------
q = [x, z, theta]   (surge, heave, pitch — planar 2D)

Initial_Points
--------------
GR[0]:    Anchor at global (x=-50.0 m, z=-100.0 m).
          This is the seabed attachment point; z=-100 m ≡ 100 m water depth.

BD[1][0]: Fairlead at body-1-local (x_local=2.0 m, z_local=-5.0 m).
          The horizontal offset of 2 m creates a nonzero surge–pitch
          coupling in the MoorPy reduced stiffness matrix.

Point ownership
---------------
Anchor and fairlead coordinates are defined HERE and referenced by ID from
moorpy_inputs.py.  No coordinates are duplicated there.
"""

import sys
import os

# Add the workspace root so 'multibody' is importable when this module is
# executed directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_ROOT, "source")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from multibody import normalize_prismatic, symvars_definition

# ---------------------------------------------------------------------------
# Symbolic variable sets (empty for this minimal example)
# ---------------------------------------------------------------------------

dataNames   = []   # no body-data symbolic parameters
BodyDataSym = symvars_definition(dataNames, globals())

PointNames  = []   # no additional force points
PointsSym   = symvars_definition(PointNames, globals())

ForceNames  = []   # no symbolic force magnitudes
ForceSym    = symvars_definition(ForceNames, globals())

ForcesPointsSym = PointsSym + ForceSym   # empty list

# ---------------------------------------------------------------------------
# Multibody topology
# ---------------------------------------------------------------------------

Reference_frame_Origin = np.array([0.0, 0.0])

joints             = [[0, 1]]                   # ground → buoy
types              = ["F"]                      # floating joint
parent_cg_to_joint = [[0.0, 0.0]]              # ground CG coincides with origin
joint_to_child_cg  = [[np.nan, np.nan]]         # required for 'F' joints
prismatic_direction = normalize_prismatic([[np.nan, np.nan]])

# ---------------------------------------------------------------------------
# Point definitions
# ---------------------------------------------------------------------------

# Do not modify the structure — only modify the *values* inside.
Initial_Points = {}

# GR: fixed ground points (global Cartesian [x, z]).
# GR[0] is the mooring anchor at x=-50 m, z=-100 m (seabed, 100 m water depth).
Initial_Points["GR"] = [
    [-50.0, -100.0],   # index 0: anchor
]

# BD: body-defined points (body-local [x_local, z_local]).
# BD[1][0] is the fairlead offset on body 1 relative to its CG.
# A 2 m horizontal offset produces nonzero surge–pitch coupling in K.
Initial_Points["BD"] = {
    1: [
        [2.0, -5.0],   # index 0: fairlead (2 m starboard, 5 m below CG)
    ],
}

# ---------------------------------------------------------------------------
# External forces (none for this minimal MoorPy example)
# ---------------------------------------------------------------------------

Force = {}
Force["PointsBD"]     = []
Force["CG"]           = []
Force["TensionSpring"]= []
Force["TensionDamper"]= []
Force["TorsionSpring"]= []
Force["TorsionDamper"]= []

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

g    = 9.81                           # gravitational acceleration [m/s²]
gVec = np.ones((len(types), 1))       # gravity active on body 1

m0   = np.array([2500.0])             # buoy mass [kg]
J0   = np.array([2500.0])             # buoy pitch inertia [kg·m²]

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
# ic = [x, z, theta, xd, zd, thetad]  (position then velocity)
# Buoy starts at rest at the origin — all zeros.

from multibody import JointSystem     # noqa: E402 (import after path setup)

_joint_system = JointSystem.from_data(
    joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction
)
_Q, _QD, _, NDOF, _ = _joint_system.coordinate_finder()

ic = np.zeros(2 * sum(NDOF))   # [x, z, theta, xd, zd, thetad] = [0, …, 0]

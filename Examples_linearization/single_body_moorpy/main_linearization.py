# Examples_linearization/single_body_moorpy/main_linearization.py
"""
Minimal MoorPy–M4E linearization example.

This example demonstrates:

1. A single floating body whose mooring dynamics are modelled by MoorPy.
2. How MoorPy's full 6×6 body-level matrices are reduced to the M4E planar
   Cartesian space ``[x, z, pitch]``.
3. How ``LinearizationManager`` transforms Cartesian matrices into the M4E
   joint-coordinate space.
4. Nonzero surge–pitch coupling arising from the offset fairlead.
5. That the MoorPy body reference point equals the M4E CG at ``q0``.

Files
-----
M4E_inputs.py      — MBD topology, GR/BD point coordinates, mass/inertia.
moorpy_inputs.py   — line-type properties and point references (no coordinates).
main_linearization.py — this file; buildable and runnable via ``run()``.

Call ``run()`` from the REPL or from tests.  The function returns
``(adapter, manager, totals)`` for further inspection.
"""

from __future__ import annotations

import os
import sys
import importlib

import numpy as np

# Ensure the package root and source directory are on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_ROOT, "source")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------------------ #
# 1. Import M4E and mooring input modules                             #
# ------------------------------------------------------------------ #
from Examples_linearization.single_body_moorpy import M4E_inputs as ex
from Examples_linearization.single_body_moorpy import moorpy_inputs as mp_mod
from Examples_linearization.single_flap.hydro_inputs import waves, body_inputs, freq, amplitude, m0, J0

# Linearization
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization.hydro_linear_mckf import HydroLinearMCKF

from multibody import MbdSystem
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization._moorpy_schema import from_module, validate_moorpy_inputs
from multibody.linearization.moorpy_linear_mckf import MoorPyLinearMCKF

# ------------------------------------------------------------------ #
# 2. Parse and validate the mooring inputs schema                     #
# ------------------------------------------------------------------ #
moorpy_inputs = from_module(mp_mod)
validate_moorpy_inputs(moorpy_inputs, ex.Initial_Points)

# ------------------------------------------------------------------ #
# 3. Build the M4E multibody system                                   #
# ------------------------------------------------------------------ #
mbd_sys = MbdSystem.from_example(ex)

# ------------------------------------------------------------------ #
# 4. Define the linearization operating point                         #
# ------------------------------------------------------------------ #
# q0: joint-space equilibrium (all zeros — buoy at rest at origin).
q0 = mbd_sys.ic[: len(mbd_sys.Q)]  # shape (nq,) = (3,)

# Define initial conditions w.r.t equilibrium
mainNumVars = np.hstack((ex.ic, ex.ForcesPointsNum, ex.BodyDataNum)) # delta_q0 shape (6,) = [0,0,0,0,0,0]

# 3 - Linearization manager
LinManager = LinearizationManager(mbd_sys, q0, mainNumVars, m0, J0, print_sym_matrices=False)

# 4 - Instanciate and register hydrodynamics adapter
hydroAdapter = HydroLinearMCKF(
        mbd_sys,
        (mainNumVars, m0, J0),
        is_2D=False,
        omega_r=waves.omega.values,
        body_inputs=body_inputs,
        wave_amplitude=waves.attrs['Amplitude (m)'],
        equilibrium_pos=q0,
        ## kwargs
        save_dir=os.getcwd() + '\\single_flap\\',
        file_name="bem.nc", 
        rho=1025,
        )

# 5 - Build the MoorPy adapter
adapter = MoorPyLinearMCKF(
    mbd_sys,
    q0,
    mainNumVars,
    moorpy_inputs,
    include_mean_force=False,
)

# 6 - Register adapters
LinManager.register(hydroAdapter)
LinManager.register(adapter)

# 7 - ########### Frequency-domain analysis ############
# --- Frequency-domain matrices for all omegas ---
fd = LinManager.assemble_frequency_domain(waves.omega.values)

print('Frequency-domain total mass matrix:')
print(fd.M)
print('Frequency-domain total damping matrix:')
print(fd.C)
print('Frequency-domain total stiffness matrix:')
print(fd.K)
print('Frequency-domain total force vector:')
print(fd.Fhat)



   

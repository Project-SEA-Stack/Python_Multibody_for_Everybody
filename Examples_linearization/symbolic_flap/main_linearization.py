# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""

import os 
import sys

# Make the repo root + source dir importable regardless of the cwd this is run from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_ROOT, "source")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Python libraries
import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from multibody import MbdSystem            # new public class
import multibody as mbd

# Linearization
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization.hydro_linear_mckf import HydroLinearMCKF

############ Example to import ############
# from Examples import spiderfloat as ex
import Examples_linearization.symbolic_flap.M4E_inputs as ex
from Examples_linearization.symbolic_flap.hydro_inputs import waves, body_inputs, freq, amplitude, m0, J0
# from Pytests import Example1 as ex

# Target folder 
folder = os.path.join(_HERE, 'hydroData/')

############### Beginning of the multibody simulation ###############
# 1- Initialize the Multibody system
MBDsys = MbdSystem.from_example(ex)

# 2 - Define initial numerical values
# Define the equilibrium position
q0 = (MBDsys.ic - ex.ic)[:len(MBDsys.Q)] # This may need revision but avoids user thinking too much

# Define initial conditions w.r.t equilibrium
mainNumVars = np.hstack((ex.ic, ex.ForcesPointsNum, ex.BodyDataNum)) # delta_q0

print('Finished initialization')

# 3 - Linearization manager
LinManager = LinearizationManager(MBDsys, q0, mainNumVars, m0, J0, print_sym_matrices=True)

# Instanciate and register hydrodynamics adapter
hydroAdapter = HydroLinearMCKF(
        MBDsys,
        (mainNumVars, m0, J0),
        is_2D=False,
        omega_r=waves.omega.values,
        body_inputs=body_inputs,
        wave_amplitude=waves.attrs['Amplitude (m)'],
        equilibrium_pos=q0,
        ## kwargs
        # save_dir=folder,
        load_dir=folder,
        file_name="bem.nc", 
        rho=1025,
        )

LinManager.register(hydroAdapter)

########### Frequency-domain analysis ############
# --- Frequency-domain matrices for all omegas ---
fd = LinManager.assemble_frequency_domain(waves.omega.values)

# RAO for each frequency
RAO = []

for omega in waves.omega.values:
    fd0  = LinManager.assemble_frequency_domain(omega)
    Z    = -omega**2 * fd0.M + 1j*omega*fd0.C + fd0.K
    qhat = np.linalg.solve(Z, fd0.Fhat)
    RAO.append(np.abs(qhat).squeeze() / waves.attrs['Amplitude (m)'])

RAO = np.array(RAO)

print('RAO values:\n', RAO)


########### Time-domain analysis ############
# --- Compile and integrate ---
# NOTE: first provide an operating point i.e. operating omega and ramp time (Default ramp is 0.0). 
omega0 = waves.omega.values[0]
LinManager.compile_operating_point(omega0, eq_tol=1e-6, eq_mode="warn", ramp_T=30.0)

result = LinManager.integrate_linear_system(
    tspan=ex.tspan,
    dt=ex.TimeStep,
    method="RK45",
    solver_opts={'rtol': 1e-6, 'atol': 1e-9},
)

# Evaluate trajectories
offset          = np.hstack((q0, np.zeros_like(q0))).reshape(-1,1) # Add the equilibrium position to the integrated solution.
result.y        += offset
com_positions, com_velocities, angle_positions, _ = mbd.evaluate_trajectories(MBDsys, result, mainNumVars.copy())

RHS = []

for i, ti in enumerate(result.t):
    mainNumVars_copy                   = mainNumVars.copy()
    mainNumVars_copy[:len(result.y)]   = result.y[:,i]
    mainNumVars_copy[MBDsys.t_update]  = ti

    RHS.append(MBDsys.Force_func(*mainNumVars_copy,
                                    *m0, *J0))
RHS = np.hstack(RHS).T

# Create figure with 3 subplots for x, z, and pitch DOF
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot x position for all bodies
for body in range(len(com_positions[1])):
        axes[0].plot(result.t, com_positions[:, body, 0], label=f'Body {body+1}')
        axes[1].plot(result.t, com_positions[:, body, 1], label=f'Body {body+1}')
        axes[2].plot(result.t, angle_positions[:, body]*180/np.pi, label=f'Body {body+1}')

axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('x [m]')
axes[0].set_title('X Position')
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('z [m]')
axes[1].set_title('Z Position')
axes[1].legend()
axes[1].grid(True)

axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Pitch [deg]')
axes[2].set_title('Pitch Angle')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

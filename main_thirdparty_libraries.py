# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""

# Python libraries
import numpy as np
# import sympy as sym
import matplotlib.pyplot as plt

# Custom modules
from multibody import MbdSystem            # new public class
import multibody as mbd

# temporary
from multibody import ExternalForcesManager
from multibody.ext_forces_manager.hydro_adapter import HydroInterface
from multibody.ext_forces_manager.moordyn_adapter import MoorDynInterface

############ Example to import ############
# from Examples import Moordyn_flex_pendulum as ex
from Examples_linearization.foswec import M4E_inputs as ex
from Examples_linearization.foswec.hydro_inputs import freq, amplitude, body_inputs, m0, J0

# from hydroInputs import omega_r, body_inputs, waveheight
# from hydroInputs import data

############### Beginning of the multibody simulation ###############
# 1- Initialize the Multibody system 
MBDsys  = MbdSystem.from_example(ex)          

# 3 - Define initial numerical values
mainNumVars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))

# 2) set up the manager
# xfm = None
xfm = ExternalForcesManager(MBDsys)

if xfm is not None:
    # 4) create & register the MoorDyn adapter
    # md_adapter = MoorDynInterface(
    #         MBDsys,
    #         (mainNumVars, ex.m0, ex.J0),
    #         dt=ex.TimeStep,
    #         MDline_types=ex.MoorLineTypes,
    #         MDoptions=ex.MoorOpts,
    #         MDoutputs=ex.MoorOutputs,
    #         MDoutfile="MD_sm2.txt"
    #         # MD_infile="teehee.txt"
    #         )
    
    # 5) Register a list of adapters
    # xfm.register(md_adapter)
    # Target folder 
    folder = 'Examples_linearization/foswec/hydroData/'

    md_adapter = HydroInterface(
            MBDsys,
            (mainNumVars, m0, J0),
            is_2D=False,
            body_inputs=body_inputs,
            wave_amplitude=amplitude,
            omega_r=np.array([2*np.pi*freq]),
            operating_omega=2*np.pi*freq[0],
            equilibrium_pos=np.zeros_like(MBDsys.Q),
            T_ramp = 30.0,
            ## kwargs
            # save_dir=folder,
            load_dir=folder,
            file_name="bem_1025.nc", 
            rho=1025,
            )
    
    xfm.register(md_adapter)

print('Finished initialization')
# 3 - Integrate over time
sol     = MBDsys.integrate(mainNumVars, m0, J0, 
                           tspan=ex.tspan, dt=ex.TimeStep,  external_manager=xfm,
                           )  # ❷ run

if xfm is not None:
    xfm.end()  # Need to close the MoorDyn adapter. Cleans up files.

# Post-processing: obtain the expanded DOFs trajectories
com_positions, com_velocities, angle_positions, joint = mbd.evaluate_trajectories(MBDsys, sol, mainNumVars)


# Plot external forces
RHS = []

for i, ti in enumerate(sol.t):
    mainNumVars_copy                   = mainNumVars.copy()
    mainNumVars_copy[:len(sol.y)]      = sol.y[:,i]
    mainNumVars_copy[MBDsys.t_update]  = ti
    
    RHS.append(MBDsys.Force_func(*mainNumVars_copy,
                                    *m0, *J0))
RHS = np.hstack(RHS).T

Fexc = np.hstack(md_adapter.F_excitation).T
Ftot = np.hstack(md_adapter.F_total).T

# Plot the forces above in two subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(Fexc[:,0], lw=1.5,label='Surge')
axs[0].plot(Fexc[:,1], lw=1.5,label='Heave')
axs[0].plot(Fexc[:,2], lw=1.5,label='Pitch')
axs[0].legend()
axs[0].set_title('Excitation Forces over Time')
axs[0].set_xlabel('Time Steps')
axs[0].set_ylabel('Excitation Force [N]')
axs[0].grid(True)
axs[1].plot(Ftot[:,1] + RHS[0,1], lw=1.5, color='blue')
axs[1].set_title('Total Forces in Heave over Time')  
axs[1].set_xlabel('Time Steps')
axs[1].set_ylabel('Total Force [N]')
axs[1].grid(True)
axs[2].plot(Ftot[:,0] + RHS[0,0], lw=1.5, color='orange')
axs[2].plot(Ftot[:,2] + RHS[0,2], lw=1.5, color='green', linestyle='dashed')
axs[2].set_title('Total Forces in Surge and pitch over Time')
axs[2].set_xlabel('Time Steps')
axs[2].set_ylabel('Total Force [N]')
axs[2].grid(True)
plt.tight_layout()

# Plot position for all bodies
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for body in range(len(com_positions[1])):
        axes[0].plot(sol.t, com_positions[:, body, 0], label=f'Body {body+1}')
        axes[1].plot(sol.t, com_positions[:, body, 1], label=f'Body {body+1}')
        axes[2].plot(sol.t, angle_positions[:, body]*180/np.pi, label=f'Body {body+1}')

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

# Initial time plot
frame_tol       = 0.2
frame_bounds    = mbd.plot.bound_finder(frame_tol, sol.t, sol.y ,mainNumVars, MBDsys)

# Optional time value
current_time                 = 0.0
current_vars                 = mainNumVars.copy()
current_vars[:len(sol.y)]    = sol.y[:,0]
current_vars[MBDsys.t_update]= current_time  # update time-dependent variables

# Call the plot function
fig, ax = mbd.plot.plot_multibody_system(
    main_num_vars   = current_vars,
    MBD             = MBDsys,
    xfm             = xfm,
    t_val           = current_time,
    frame_bounds    = frame_bounds,
    dark_mode       = False  # Or True for dark background
)
plt.show()  # or fig.savefig('snapshot.png')

#%% Animation
# Frames bounds
ex.animation_on = True
ex.SaveMovieOn = None
ex.plotTstep = 1
if ex.animation_on:
    anim = mbd.plot.animate_multibody(
        tvec            = sol.t[::ex.plotTstep],
        y               = sol.y[:,::ex.plotTstep],
        MBD             = MBDsys,
        mainNumVars     = mainNumVars.copy(),
        frame_bounds    = frame_bounds,
        save_path       = ex.SaveMovieOn,   # ".gif" ➜ GIF  ·  None ➜ just show
        fps             = 100,
        loop            = False,
        xfm             = xfm
    )
plt.close()

print('\nReached end')
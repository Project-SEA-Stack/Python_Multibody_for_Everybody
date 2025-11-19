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
from multibody.ext_forces_manager.moordyn_adapter import MoorDynInterface

############ Example to import ############
from Examples_native import Moordyn_flex_pendulum as ex

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
    md_adapter = MoorDynInterface(
            MBDsys,
            (mainNumVars, ex.m0, ex.J0),
            dt=ex.TimeStep,
            MDline_types=ex.MoorLineTypes,
            MDoptions=ex.MoorOpts,
            MDoutputs=ex.MoorOutputs,
            MDoutfile="MD_sm2.txt"
            # MD_infile="your_file.txt"
            )
    
    # 5) Register a list of adapters
    xfm.register(md_adapter)

print('Finished initialization')
# 3 - Integrate over time
sol     = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, 
                           tspan=ex.tspan, dt=ex.TimeStep,  external_manager=xfm,
                           )  # ❷ run

if xfm is not None:
    xfm.end()  # Need to close the MoorDyn adapter. Cleans up files.

# Access results
com_positions, com_velocities, joint_positions, omega = mbd.evaluate_trajectories(MBDsys, sol, mainNumVars)

#%% Plot energy
Em_num = []

for i, ti in enumerate(sol.t):
    mainNumVars_copy                   = mainNumVars.copy()
    mainNumVars_copy[:len(sol.y)]      = sol.y[:,i]
    mainNumVars_copy[MBDsys.t_update]  = ti
    
    Em_num.append(MBDsys.Energy_func(*mainNumVars_copy,
                                    *ex.m0, *ex.J0))
Em_num = np.array(Em_num)

fig, ax = plt.subplots(figsize=(8,4))
mbd.plot.plt_template(ax, dark=False)
ax.plot(sol.t, Em_num, lw=1.8)
ax.set_xlabel("t [s]"); ax.set_ylabel("E [J]")
ax.set_title("Total mechanical energy")
plt.tight_layout(); plt.show()

# Initial time plot
frame_tol       = 10.
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
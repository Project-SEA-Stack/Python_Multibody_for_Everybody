# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""

# Python libraries
import io
import contextlib
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from time import time

# Custom modules
from multibody import MbdSystem    
import multibody as mbd
from multibody import load_yaml_as_example

############ Example to import ############
# from Examples_mbd import _4_Flexible_pendulum as ex
# Some hand-written examples print their own (redundant) Points/Forces
# tables at import time -- suppress that noise, we only want Table 1/2/3
# (joints/bodies) below, whichever kind of example this is.
with contextlib.redirect_stdout(io.StringIO()):
    from Examples_flexible_FSM  import flex_double_pendulum as ex

############ Example to import ############
# folder  = 'Examples_yaml/'
# ex_file = 'slider_w_dpend'  # Example file to import
# # Load the example from the YAML file
# ex      = load_yaml_as_example(folder + ex_file)

# 1- Initialize the Multibody system 
MBDsys  = MbdSystem.from_example(ex)          

# Print the body/joint table(s) to the terminal. Flex-generated examples
# (from generate_flex_model_file) embed the ORIGINAL pre-expansion Table 1/
# Table 2 as flex_table1/flex_table2 -- ordinary rigid examples don't have
# these attributes at all, so only the one table (joints/types/etc. as
# declared) is shown for them.
if hasattr(ex, 'flex_table1'):
    mbd.tables.bodies_table(**ex.flex_table1, title="\n===== Table 1 (bodies/joints, pre-expansion) =====")
    mbd.tables.flex_properties_table(**ex.flex_table2, title="\n===== Table 2 (flexible-member properties) =====")
    mbd.tables.bodies_table(ex.joints, ex.types, ex.parent_cg_to_joint, ex.joint_to_child_cg,
                             ex.prismatic_direction, ex.m0, ex.J0,
                             title="\n===== Table 3 (expanded, what MbdSystem actually uses) =====")
else:
    mbd.tables.bodies_table(ex.joints, ex.types, ex.parent_cg_to_joint, ex.joint_to_child_cg,
                             ex.prismatic_direction, ex.m0, ex.J0,
                             title="\n===== Bodies/joints table =====")
print()

# 3 - Define initial numerical values
mainNumVars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))

print('Finished initialization')
# 3 - Integrate over time
# integrator_kwargs is an OPTIONAL example attribute (e.g. FSM/flexible models
# set it to force a stiff-compatible implicit solver) -- absent for ordinary
# rigid examples, so their integrate() call behaves exactly as before.
integrator_kwargs = getattr(ex, 'integrator_kwargs', {})
sol     = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, 
                           tspan=ex.tspan, dt=ex.TimeStep, **integrator_kwargs)  # ❷ run


com_positions, com_velocities, angle_positions, joint = mbd.evaluate_trajectories(MBDsys, sol, mainNumVars)

############### Plotting ###############
# Multibody connections graph plot
# MBDsys.graph.show()

# Create figure with 3 subplots for x, z, and pitch DOF
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot x position for all bodies
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
plt.show()

# Plot energy
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
    t_val           = current_time,
    frame_bounds    = frame_bounds,
    dark_mode       = False  # Or True for dark background
)

# plt.tight_layout()
plt.show()  # or fig.savefig('snapshot.png')

#%% Animation
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
    )
plt.close()

print('\nReached end')
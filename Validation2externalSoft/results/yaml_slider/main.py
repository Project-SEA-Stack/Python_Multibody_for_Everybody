# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""

# Python libraries
import numpy as np
import sympy
import matplotlib.pyplot as plt

# Custom modules
import multibody as mbd
from multibody import MbdSystem            # new public class
from yaml_parser.yaml_adapter import load_yaml_as_example

# sAVING CAPABILITIES
from multibody import evaluate_trajectories 

############ Example to import ############
ex_file = 'slider_w_dpend'  # Example file to import

# # Load the example from the YAML file
ex = load_yaml_as_example(ex_file)

# import spiderfloat as ex

############### Beginning of the multibody simulation ###############
# 1- Initialize the Multibody system 
MBDsys  = MbdSystem.from_example(ex)          

print('Finished initialization')

# 2 - Define initial numerical values
mainNumVars = np.hstack((MBDsys.ic, ex.ForcesPointsNum, ex.BodyDataNum))

# 4 - Integrate over time
sol     = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, 
                           tspan=ex.tspan+ex.TimeStep, dt=ex.TimeStep)  # ❷ run

com_positions, com_velocities, joint_positions, angle_positions = evaluate_trajectories(MBDsys, sol, mainNumVars)
# Save trajectories to .npy files
target_t = 0.01
skip = int(target_t / ex.TimeStep)
np.save('M4E_export\\r_M4E.npy', com_positions[::skip])
np.save('M4E_export\\v_M4E.npy', com_velocities[::skip])
# np.save('joint_positions.npy', joint_positions)
np.save('M4E_export\\angle_positions.npy', angle_positions[::skip])
print(skip)
print(com_positions[::skip].shape)
quit()
# print joints, types, parent cg to joint vectors and joint to children cg
print("\nSummary of the Multibody System:")
print(f"  Joint: {ex.joints}")
print(f"  Type: {ex.types}")
print(f"  Parent CG to Joint: {ex.parent_cg_to_joint}")
print(f"  Joint to Children CG: {ex.joint_to_child_cg}")
print(f"  Prismatic direction: {ex.prismatic_direction}")

print("\nInitial Points:")
print(f"  Ground Points: {ex.Initial_Points['GR']}")   
print(f"  Body Points: {ex.Initial_Points['BD']}")
print("\nForces:")
print(f"  PointsBD: {ex.Force['PointsBD']}")
print(f"  CG: {ex.Force['CG']}")
print(f"  TensionSpring: {ex.Force['TensionSpring']}")
print(f"  TensionDamper: {ex.Force['TensionDamper']}")
print(f"  TorsionSpring: {ex.Force['TorsionSpring']}")
print(f"  TorsionDamper: {ex.Force['TorsionDamper']}")

# Questions to clarify:
print("\n"+30*"#")
print("\nTODO: is offset the reference from origin?")
print("TODO: add motor cases to the YAML file if the function type is force. [info] need to wait for example")

# Info
print("\n[Info] The parser assumes no symbolic variables are defined in the YAML file.")
print("\n[Info] Initial conditions are not considered in YAML file.")
# warnings
print("\n[Warning] Body points are defined in the body frame, not in the global frame.")
print("\n[Warning] I am defining gravity in negative y-direction as in pychrono.")

# Initial time plot
frame_tol       = 1.
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
plt.show()  # or fig.savefig('snapshot.png')

#%% Animation
# Frames bounds
# Animation
animation_on    = 1 
SaveMovieOn     = None 
plotTstep       = 100

if animation_on:
    anim = mbd.plot.animate_multibody(
        tvec            = sol.t[::plotTstep],
        y               = sol.y[:,::plotTstep],
        MBD             = MBDsys,
        mainNumVars     = mainNumVars.copy(),
        frame_bounds    = frame_bounds,
        save_path       = SaveMovieOn,   # ".gif" ➜ GIF  ·  None ➜ just show
        fps             = plotTstep,
        loop            = False,
    )
plt.close()

print('\nReached end')


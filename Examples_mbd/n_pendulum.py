# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""
import sys
import os

# Add the source and examples directory to the path to run examples
source_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, source_dir)

# External libraries
import numpy as np
import sympy as sym

# This module libraries
import multibody as mb
from multibody import JointSystem, normalize_prismatic
from multibody import symvars_definition

# Do not modify
Initial_Points  = {}
Force           = {}
# End of Do not Modify

# =============================================================================
# USER VARIABLE: number of links
n = 60

# Origin offset
Reference_frame_Origin = np.array([0, 0])

# Bodies definition
dataNames   = []  # Define symbolic variables for force points (for example, b1, b2, b3).
BodyDataSym = symvars_definition(dataNames, globals())

# =============================================================================
# n-link pendulum joint definitions
# =============================================================================
# Joint connectivity: [parent, child]
# For an n-link chain: 0-1, 1-2, ..., (n-1)-n
joints = [[i, i + 1] for i in range(n)]

# Joint types: all revolute for a simple n-link pendulum
types = ['R'] * n

# Geometry (example uses unit-length links split as 0.5 + 0.5)
parent_cg_to_joint  = [[0., -1.]] * n
joint_to_child_cg   = [[0., -1.]] * n

# No prismatic joints here
prismatic_direction = [[np.nan, np.nan]] * n
prismatic_direction = normalize_prismatic(prismatic_direction)

# Points definition
PointNames  = []  # Define symbolic variables for force points (for example, b1, b2, b3).
PointsSym   = symvars_definition(PointNames, globals())

# Define the structure for initial points.
Initial_Points["GR"]    = []   # Ground points
Initial_Points["BD"]    = {}   # Body-defined points dict

# Forces definition
ForceNames  = []  # Define symbolic variables for force points (for example, b1, b2, b3).
ForceSym    = symvars_definition(ForceNames, globals())

Force["PointsBD"]       = []
Force["CG"]             = []
Force["TensionSpring"]  = []
Force["TensionDamper"]  = []
Force["TorsionSpring"]  = []
Force["TorsionDamper"]  = []

ForcesPointsSym = PointsSym + ForceSym

# Initial conditions
joint_system        = JointSystem.from_data(
    joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction
)  # DO NOT MODIFY
Q, QD, _, NDOF, _   = joint_system.coordinate_finder()  # DO NOT MODIFY

# Initial conditions vector: [q1, q1d, q2, q2d, ...] (based on your existing pattern)
ic = np.zeros(2 * sum(NDOF))

# Set all joint angles to pi/2 initially
ic[:n] = -np.pi / 2

# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))

# Simulation
TimeStep    = 0.01
tspan       = 10.
g           = 9.81                     # gravity
gVec        = np.ones((len(types), 1))  # Percentage of gravity acting on each body
m0          = 10*np.ones(len(types))
J0          = m0 * 2.**2 / 12

# Animation
animation_on    = 1     # Display animation:1, 0-otherwise
SaveMovieOn     = None  # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 5     # how many times faster are we plotting w.r.t to simulation time

###########################################################################
######################### END OF USER DEFINITION ##########################
###########################################################################

#%% Checks section: ensures everything is defined correctly
# mb.checks.bodies(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction)
# mb.checks.points_forces(joints, types, Initial_Points, Force)

# NBodies = len(joints)

# if len(m0) != NBodies or len(J0) != NBodies or not np.all(m0) or not np.all(J0):
#     raise ValueError("The mass matrix has zero-valued entries check m0 and J0")

# #%% Display tables
# print('\nBodies and joints table:')
# joint_system.display_table()

# print('\nPoints table:')
# mb.tables.points_table(Initial_Points)
# print('\nForces table:')
# mb.tables.force_table(Force)

#%% Problem variables and required initial conditions
# ic_vars = Q + QD
# print('\nThe variables that require initial conditions, in this order, are:\n' + str(ic_vars) + '\n')
print(f'number of links: {n}')

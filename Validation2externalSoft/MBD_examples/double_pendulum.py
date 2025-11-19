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

# Example 3
Reference_frame_Origin = np.array([0,0])

# Bodies definition
dataNames   = [] # Define symbolic variables for force points (for example, b1, b2, b3).
BodyDataSym = symvars_definition(dataNames,globals())


joints              = [[0, 1],[1, 2]] # Joint connectivity: [parent, child]
types               = ['R', 'R']  # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[0, 0],[0, -0.5]] # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[0, -0.5],[0, -0.5]] # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
prismatic_direction = normalize_prismatic(prismatic_direction)

# Points definition
PointNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
PointsSym   = symvars_definition(PointNames,globals())

# Define the structure for initial points.
Initial_Points          = {}
# Ground points: these are fixed and given as [x, z] coordinates.
Initial_Points["GR"]    = []

# Body-defined points ("BD"): we use a dictionary where each key (an integer)
# corresponds to a body and the value is a list of points.
Initial_Points["BD"]    = {}

# Forces definition
ForceNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
ForceSym    = symvars_definition(ForceNames,globals())

# Define the structure for Force.
Force = {}
# Forces applied on body-defined points (PointsBD):
Force["PointsBD"] = []

# Forces applied at the center of gravity (CG):
Force["CG"] = []

# Tension springs: stored as a list of tuples: (connection, [l0, stiffness]).
# Here "BD41" means (for example) the 1st point on body 4.
Force["TensionSpring"] = []

# Tension dampers: here we have two pairs.
# The first pair uses a constant damping coefficient.
# The second uses a lambda function to represent a nonlinear damping coefficient.
Force["TensionDamper"] = []

# Torsion springs: here the first element is a list of parameters and the second element is a lambda.
Force["TorsionSpring"] = []

# Torsion dampers:
Force["TorsionDamper"] = []

ForcesPointsSym = PointsSym + ForceSym 

# Initial conditions
ic = np.array([0,0,-0.4,0.]) # Multiplied by 2 because is position and velocity


# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))

# Simulation
TimeStep    = 0.001
tspan       = 10.
g           = 9.81                      # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = np.ones(len(types))

# Animation
animation_on    = 1 # Display animation:1, 0-otherwise
SaveMovieOn     = None # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 50 # how many times faster are we plotting w.r.t to simulation time

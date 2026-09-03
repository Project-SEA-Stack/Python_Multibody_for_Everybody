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

flapHeight = 0.56
flapCG = np.array([0,0,flapHeight/2])

joints              = [[0, 1]] # Joint connectivity: [parent, child]
types               = ['R']  # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[0, -flapHeight - 0.01]] # [[3, 0],[0,1]] # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[0,flapCG[2]]] # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
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

# Gravity
g           = 9.81  # Gravity acceleration (m/s^2)
TimeStep    = 0.005  # Time step for the simulation (s)
tspan       = 100+TimeStep  # Time span for the simulation (s)
ic          = np.array([0.0, 0.0])  # Initial conditions

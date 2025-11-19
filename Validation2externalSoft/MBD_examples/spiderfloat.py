# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:18:55 2025

@author: adiazfl
Validated
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# External libraries
import numpy as np
import sympy as sym

# This module libraries
from multibody import JointSystem, normalize_prismatic
from multibody import symvars_definition

# Do not modify 
Initial_Points  = {} 
Force           = {} 
# End of Do not Modify

# Spiderfloat example
Reference_frame_Origin = np.array([0,0])

# Bodies definition
dataNames   = [] # Define symbolic variables for force points (for example, b1, b2, b3).
BodyDataSym = symvars_definition(dataNames,globals())

joints              = [[0, 1],[1, 2],[1, 3],[2, 4],[3, 5],[4, 6],[5, 7]]              # Joint connectivity: [parent, child]
types               = ['F', 'R', 'R', 'P', 'P', 'R', 'R']                         # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[10,6],[0.5,-2],[-0.5,-2],[4.5,0],[-4.5,0],[0,0],[0,0]]     # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[np.nan, np.nan],[5,0],[-5,0],[0.5,0],[-0.5,0],[0,1.5],[0,1.5]]   # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan],[1,0],[1,0],[np.nan, np.nan],[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
prismatic_direction = normalize_prismatic(prismatic_direction)

# Points definition
PointNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
PointsSym   = symvars_definition(PointNames,globals())

# Define the structure for initial points.
# Ground points: these are fixed and given as [x, z] coordinates.
Initial_Points["GR"]    = [[20,0], [0,0]]

# Body-defined points ("BD"): we use a dictionary where each key (an integer)
# corresponds to a body and the value is a list of points.
Initial_Points["BD"]    = {}
Initial_Points["BD"][1] = [[0.5,-2.5],[-0.5,-2.5],[0.5,5],[-0.5,5]]
Initial_Points["BD"][2] = [[4,0],[4.5,0]]
Initial_Points["BD"][3] = [[-4,0],[-4.5,0]]

# Forces definition
ForceNames  = ["t"] # Define symbolic variables for force points (for example, b1, b2, b3).
ForceSym    = symvars_definition(ForceNames,globals())

# Define the structure for Force.
# Forces applied on body-defined points (PointsBD):
Force["PointsBD"] = []

# Forces applied at the center of gravity (CG):
Force["CG"] = [[1,0, 1.*sym.cos(t),0],
                [4, 1.*sym.cos(t),0, 0],
                [5,-1.*sym.cos(t),0, 0]
              ]

# Tension springs: stored as a list of tuples: (connection, [l0, stiffness]).
# Here "BD41" means (for example) the 1st point on body 4.
Force["TensionSpring"] = [
    (("BD10","BD20"), [9.0139,1e-1]),
    (("BD11","BD30"),[9.0139,1e-1]),
    (("BD12","BD21"),[11.8004,1e-1]),
    (("BD13","BD31"),[11.8004,1e-1]),
    (("BD20","GR00"),[3.0414,1e-1]),
    (("GR01","BD30"),[3.0414,1e-1])
]

# Tension dampers: here we have two pairs.
# The first pair uses a constant damping coefficient.
# The second uses a lambda function to represent a nonlinear damping coefficient.
Force["TensionDamper"] = [
    (("GR00", "CG22"), 3),
    (("GR01", "CG33"), 3)
]

# Torsion springs: here the first element is a list of parameters and the second element is a lambda.
Force["TorsionSpring"] = [
    ((1,2), [0, 1.e2]),
    ((1,3), [0, 1.e2])
]

# Torsion dampers:
Force["TorsionDamper"] = [
    ((1,2), 1.e1),
    ((1,3), 1.e1)
]

ForcesPointsSym = PointsSym + ForceSym # TODO: move this outside of user definition

# Initial conditions
# Create the JointSystem using the from_data class method.
joint_system        = JointSystem.from_data(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction) # DO NOT MODIFY
Q, QD, _, NDOF, _   = joint_system.coordinate_finder() # DO NOT MODIFY

# If you are unsure what are the systems DOF run the example and you will see on screen
ic                  = np.zeros(2*sum(NDOF)) # Multiplied by 2 because is position and velocity

# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))

# Simulation
TimeStep    = 0.01
tspan       = 10
g           = 0.                       # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = 2.*np.ones(len(types))

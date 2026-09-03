# -*- coding: utf-8 -*-
"""
Double pendulum with torsion spring/damper for linearization regression testing.

Two rigid bodies in an R-R chain hanging under gravity, with a linear torsion
spring (stiffness 10 Nm/rad, zero rest angle) and damper (2 Nm·s/rad) acting
between ground and body 1.

Mirrors ``Validation2externalSoft/MBD_examples/double_pendulum_advanced.py``.

Physical parameters:  L1 = L2 = 1 m,  m1 = m2 = 1 kg,  J1 = J2 = 1 kg·m²,
                      g = 9.81 m/s².
IC:                   theta1 = 0.1, theta2 = 0.1, velocities = 0.
"""
import sys
import os

source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, source_dir)

import numpy as np

from multibody import normalize_prismatic, symvars_definition

Initial_Points = {}
Force = {}

Reference_frame_Origin = np.array([0, 0])

dataNames = ["L1", "L2"]
BodyDataSym = symvars_definition(dataNames, globals())

joints = [[0, 1], [1, 2]]
types = ["R", "R"]
parent_cg_to_joint = [[0, 0], [0, -L1 / 2]]
joint_to_child_cg = [[0, -L1 / 2], [0, -L2 / 2]]
prismatic_direction = [[np.nan, np.nan], [np.nan, np.nan]]
prismatic_direction = normalize_prismatic(prismatic_direction)

PointNames = []
PointsSym = symvars_definition(PointNames, globals())

Initial_Points["GR"] = []
Initial_Points["BD"] = {}

ForceNames = []
ForceSym = symvars_definition(ForceNames, globals())

Force["PointsBD"] = []
Force["CG"] = []
Force["TensionSpring"] = []
Force["TensionDamper"] = []
Force["TorsionSpring"] = [((0, 1), [0, 10])]
Force["TorsionDamper"] = [((0, 1), 2)]

ForcesPointsSym = PointsSym + ForceSym

ic = np.array([0.1, 0.1, 0., 0.])

ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum = np.array([1., 1.])

TimeStep = 0.01
tspan = 10.
g = 9.81
gVec = np.ones((len(types), 1))
m0 = np.ones(len(types))
J0 = np.ones(len(types))

animation_on = 0
SaveMovieOn = None
plotTstep = 50

# -*- coding: utf-8 -*-
"""
Single gravity pendulum model for Chrono regression testing.

One rigid body on a revolute joint hanging from ground under gravity.
No applied forces.

Physical parameters:  L = 1 m (joint_to_child_cg = [0, -0.5]),
                      m = 1 kg,  J = 1 kg·m²,  g = 9.81 m/s².

Default IC:  θ = 0, θ̇ = -0.4 rad/s  (matches Chrono reference).
"""
import sys
import os

source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, source_dir)

import numpy as np

from multibody import normalize_prismatic
from multibody import symvars_definition

Initial_Points = {}
Force = {}

Reference_frame_Origin = np.array([0, 0])

dataNames = []
BodyDataSym = symvars_definition(dataNames, globals())

joints = [[0, 1]]
types = ["R"]
parent_cg_to_joint = [[0, 0]]
joint_to_child_cg = [[0., -0.5]]
prismatic_direction = [[np.nan, np.nan]]
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
Force["TorsionSpring"] = []
Force["TorsionDamper"] = []

ForcesPointsSym = PointsSym + ForceSym

ic = np.array([0., -0.4])

ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum = np.ones(len(BodyDataSym))

TimeStep = 0.001
tspan = 10.
g = 9.81
gVec = np.ones((len(types), 1))
m0 = np.ones(len(types))
J0 = np.ones(len(types))

animation_on = 0
SaveMovieOn = None
plotTstep = 100

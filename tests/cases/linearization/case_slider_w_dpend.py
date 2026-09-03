# -*- coding: utf-8 -*-
"""
Slider + double pendulum model for linearization regression testing.

A prismatic slider on ground carries a two-link revolute chain under gravity.
No external excitation — the system responds to a small initial perturbation
on the slider so that the linearized and nonlinear M4E solutions match.

Topology: ground →[P]→ slider →[R]→ pendulum1 →[R]→ pendulum2.

Geometry matches ``yaml_examples/slider_w_dpend.model.yaml``.

Physical parameters:  all masses 1 kg, all inertias 1 kg·m²,  g = 9.81 m/s².
IC:                   slider x = 0.05, all other coordinates and velocities = 0.
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

dataNames = []
BodyDataSym = symvars_definition(dataNames, globals())

joints = [[0, 1], [1, 2], [2, 3]]
types = ["P", "R", "R"]
parent_cg_to_joint = [[0, 0], [0, 0], [0, -0.5]]
joint_to_child_cg = [[0, 0.125], [0, -0.625], [0, -0.5]]
prismatic_direction = [[1, 0], [np.nan, np.nan], [np.nan, np.nan]]
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

ic = np.array([0.05, 0.05, 0.05, 0., 0., 0.])

ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum = np.ones(len(BodyDataSym))

TimeStep = 0.01
tspan = 10.
g = 9.81
gVec = np.ones((len(types), 1))
m0 = 1. * np.ones(len(types))
J0 = 1. * np.ones(len(types))

animation_on = 0
SaveMovieOn = None
plotTstep = 10

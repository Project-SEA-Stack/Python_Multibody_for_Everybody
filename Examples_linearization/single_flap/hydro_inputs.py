import numpy as np
import capytaine as cpt
import pygmsh
from capytaine.io import mesh_writers
from capytaine.io.meshio import load_from_meshio
import wecopttool as wot
import xarray as xr
import os
import matplotlib.pyplot as plt

#%% Waves and frequencies definitions
# Frequency object
wavefreq    = 1/8 # Hz
f1          = wavefreq
nfreq       = 10
freq        = wot.frequency(f1, nfreq, False) # False -> no zero frequency

# Define waves object
amplitude   = 0.01/2 # m
phase       = 30 # degrees
wavedir     = 0 # degrees
waves       = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# --- flap dimensions ---
flapThickness   = 0.05
outboardWidth   = 0.1
platformWidth   = 0.76 + 2*outboardWidth
flapWidth       = platformWidth - 2*outboardWidth
flapHeight      = 0.56

# --- generate mesh (meshio) ---
with pygmsh.geo.Geometry() as geom:
    geom.add_box(
        -flapThickness/2, flapThickness/2,
        -flapWidth/2,     flapWidth/2,
        -flapHeight/2,    flapHeight/2,
        mesh_size=0.15
    )
    meshio_mesh = geom.generate_mesh()

# --- convert meshio -> capytaine Mesh ---
capy_mesh = load_from_meshio(meshio_mesh)

# (optional but recommended sanity checks)
capy_mesh.heal_mesh()          # remove duplicates / fix orientation if needed

# --- export STL ---
# folder = os.getcwd() + '\\single_flap\\'
# mesh_writers.write_STL(folder + "flap.stl", capy_mesh.vertices, capy_mesh.faces)

# Define dictionary for body inputs
m = 0.5*(.05*.76*.56)*1000
J = 1.19

body_inputs = {1: {
    "name": "flap",
    "mesh": capy_mesh,
    "mesh_reference": "body_cg",
    "mass": m,
    "inertia_diag": [m,m,m, J, J, J]
}}

m0 = [m]
J0 = [J]
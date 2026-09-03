# Imports
import wecopttool as wot
import pygmsh
import gmsh
import numpy as np
import capytaine as cpt

# Waves and frequencies definitions
# Frequency object
wavefreq    = 0.3 # Hz
f1          = wavefreq
nfreq       = 2
freq        = wot.frequency(f1, nfreq, False) # False -> no zero frequency

# Define waves object
amplitude   = 0.0625 # m
phase       = 0 # degrees
wavedir     = 0 # degrees
waves       = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# Mesh 
mesh_size_factor = 0.2
r1 = 0.88
r2 = 0.35
h1 = 0.17
h2 = 0.37
scale_factor = 1
freeboard = 0.01
with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
    cyl = geom.add_cylinder([0, 0, 0],
                            [0, 0, -h1],
                            r1)
    cone = geom.add_cone([0, 0, -h1],
                            [0, 0, -h2],
                            r1, r2)
    geom.translate(cyl, [0, 0, freeboard])
    geom.translate(cone, [0, 0, freeboard])
    geom.boolean_union([cyl, cone])
    mesh = geom.generate_mesh()

# Mass and inertia
m_wb = 876.9698045922379
J_wb = 0

# Pre-clip meshes at the waterplane (z=0) so compute_hydrostatics() uses only the submerged volume.
# User is responsible for clipping (not done inside HydroLinearMCKF).
fb = cpt.FloatingBody(mesh=mesh, name="WaveBot")
fb = fb.immersed_part()
mesh_clipped = fb.mesh  # clipped Capytaine mesh for hydrostatics/BEM


# body_inputs: spar first (key=1) then torus (key=2) to match MBD Cartesian ordering [spar, float]
body_inputs = {
    1: {
        "name": "WaveBot",
        "mesh": mesh_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_wb, m_wb, m_wb, J_wb, J_wb, J_wb],
    },
}

water_depth = np.inf # depth of wave flume

m0 = [m_wb]
J0 = [J_wb]

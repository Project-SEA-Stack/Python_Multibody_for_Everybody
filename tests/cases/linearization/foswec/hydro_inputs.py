# Imports
import wecopttool as wot
import pygmsh
import gmsh

#%% Waves and frequencies definitions
# Frequency object
wavefreq    = 1/8 # Hz
f1          = wavefreq
nfreq       = 5
freq        = wot.frequency(f1, nfreq, False) # False -> no zero frequency

# Define waves object
amplitude   = 0.1 # m
phase       = 0 # degrees
wavedir     = 0 # degrees
waves       = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

#%% Mesh and geometry definitions
# [NOTE]: the CGs should be defined inside M4E example file

# platform is a rectangle with cg at -0.46 and 1.44 m long, 0.76 m wide
platformCG_z_from_waterline = 0.0
flapThicknessBottom = 0.04
flapThicknessTop    = 0.1

platformLength      = 1.44 
outboardWidth       = 0.1
platformWidth       = 0.76+2*outboardWidth
platformHeight      = 0.05
platformCenter      = [0, 0, -0.46-platformCG_z_from_waterline]
columnsRadius       = 0.05
columnsDraft        = 0.6
columnsXY           = [platformLength/2, platformWidth/2]
cutoutWidth         = 0.76
cutoutLength        = 1.44-2*outboardWidth

# floats attached to side of platform
floatWidth  = 0.2
floatHeight = 0.35

flapWidth   = platformWidth-2*outboardWidth
flapHeight  = 0.45
flap1CG     = [-platformLength/2, 0, -0.36-platformCG_z_from_waterline] # from water surface/origin
flap2CG     = [platformLength/2, 0, -0.36-platformCG_z_from_waterline] # from water surface/origin

#%% Define the meshes
with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.3)
    platform            = geom.add_box([-platformLength/2, -platformWidth/2, platformCenter[2] - platformHeight/2], [platformLength,platformWidth,platformHeight])
    cutout              = geom.add_box([-cutoutLength/2, -cutoutWidth/2, platformCenter[2] - platformHeight/2], [cutoutLength,cutoutWidth,platformHeight])
    cyl1                = geom.add_cylinder([-columnsXY[0], -columnsXY[1], -columnsDraft-platformCG_z_from_waterline],[0, 0, columnsDraft+.01], columnsRadius)
    cyl2                = geom.add_cylinder([-columnsXY[0], columnsXY[1], -columnsDraft-platformCG_z_from_waterline],[0, 0, columnsDraft+.01], columnsRadius)
    cyl3                = geom.add_cylinder([columnsXY[0], -columnsXY[1], -columnsDraft-platformCG_z_from_waterline],[0, 0, columnsDraft+.01], columnsRadius)
    cyl4                = geom.add_cylinder([columnsXY[0], columnsXY[1], -columnsDraft-platformCG_z_from_waterline],[0, 0, columnsDraft+.01], columnsRadius)
    float1              = geom.add_box([-platformLength/2, -platformWidth/2-0.2, -floatHeight-platformCG_z_from_waterline], [platformLength,floatWidth,floatHeight])
    float2              = geom.add_box([-platformLength/2, platformWidth/2, -floatHeight-platformCG_z_from_waterline], [platformLength,floatWidth,floatHeight])
    platformCutout      = geom.boolean_difference(platform,cutout)
    geom.boolean_union([platformCutout,cyl1,cyl2,cyl3,cyl4,float1,float2])
    platformMesh        = geom.generate_mesh()

with pygmsh.geo.Geometry() as geom:
    flap1 = geom.add_polygon(
            [[-platformLength/2 - flapThicknessBottom/2, -flapWidth/2, platformCenter[2]],
            [-platformLength/2 + flapThicknessBottom/2, -flapWidth/2, platformCenter[2]],
            [-platformLength/2 + flapThicknessTop/2, -flapWidth/2, -.01-platformCG_z_from_waterline],
            [-platformLength/2 - flapThicknessTop/2, -flapWidth/2, -.01-platformCG_z_from_waterline]],mesh_size=0.1)
    geom.extrude(flap1,[0,flapWidth,0])
    flap1Mesh = geom.generate_mesh()

with pygmsh.geo.Geometry() as geom:
    flap2 = geom.add_polygon(
            [[platformLength/2 - flapThicknessBottom/2, -flapWidth/2, platformCenter[2]],
            [platformLength/2 + flapThicknessBottom/2, -flapWidth/2, platformCenter[2]],
            [platformLength/2 + flapThicknessTop/2, -flapWidth/2, -.01-platformCG_z_from_waterline],
            [platformLength/2 - flapThicknessTop/2, -flapWidth/2, -.01-platformCG_z_from_waterline]],mesh_size=0.1)
    geom.extrude(flap2,[0,flapWidth,0])
    flap2Mesh = geom.generate_mesh()

#%% Body descriptors
# NOTE: you can import the mesh if you already have it

m1 = 350
J1 = 30 * 1000
m2 = 0.5*(.05*.76*.56)*1000
J2 = 1.19 * 1000

body_inputs = {
    1: {
        "name": "platform",
        "mesh": platformMesh,
        "mesh_reference": "absolute",
        "inertia_diag": [m1, m1, m1, J1, J1, J1],
    },
    2: {
        "name": "flap1",
        "mesh": flap1Mesh,
        "mesh_reference": "absolute",
        "inertia_diag": [m2, m2, m2, J2, J2, J2],
    },
    3: {
        "name": "flap2",
        "mesh": flap2Mesh,
        "mesh_reference": "absolute",
        "inertia_diag": [m2, m2, m2, J2, J2, J2],
    },
}

m0 = [m1, m2, m2]
J0 = [J1, J2, J2]

# mesh export and loading 
# import pyvista as pv

# folder = 'foswec/geometry/'

# platform = pv.read(folder + "platform.stl")
# flap1 = pv.read(folder + "flap.stl")
# flap2 = pv.read(folder + "flap.stl")

# plotter = pv.Plotter()
# plotter.add_mesh(platform, opacity=0.5, show_edges=True)
# plotter.add_mesh(flap1, show_edges=True)
# plotter.add_mesh(flap2, show_edges=True)
# plotter.add_axes()
# plotter.show()

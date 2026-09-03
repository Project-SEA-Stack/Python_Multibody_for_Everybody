# Imports
import wecopttool as wot
import pygmsh
import gmsh
import capytaine as cpt

# NOTE: these physical parameters differ from the WEC-Sim-validated FOSWEC
# case in tests/cases/linearization/foswec/ (see test_foswec_rao.py and
# test_foswec_td.py); use that copy if you need validated numbers.

# Waves and frequencies definitions
# Frequency object
wavefreq    = 1/2.61 # Hz
f1          = wavefreq
nfreq       = 2
freq        = wot.frequency(f1, nfreq, False) # False -> no zero frequency

# Define waves object
amplitude   = 0.0005 # m
phase       = 0 # degrees
wavedir     = 0 # degrees
waves       = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# Mesh 
# Meshes consist of flaps and platform components
flap_thickness_bottom = 0.04
flap_thickness_top    = 0.1
flap_center_distance_apart = 1.44
flap_width   = 0.76
flap_height = 0.58
flap_draft  = 0.59 # 0.53 is based on matching impedance, 0.59 for fully submerged
cg_height_above_hinge = 0.17
flap1_cg     = [-flap_center_distance_apart/2, 0, -flap_draft+cg_height_above_hinge] # from water surface/origin
flap2_cg     = [flap_center_distance_apart/2, 0, -flap_draft+cg_height_above_hinge] # from water surface/origin

platform_frame_length      = 1.44 
platform_frame_thickness = 0.05
platform_frame_width = 1.06
platform_frame_top_depth = 0.53 + 0.1
platform_cg      = [0, 0, -0.8]
cutout_width         = platform_frame_width - 2*platform_frame_thickness
cutout_length        = platform_frame_length - 2*platform_frame_thickness

full_width = 1.63
columns_radius       = 0.1
columns_draft        = 0.53 + 0.1 + 0.4
columns_xy           = [flap_center_distance_apart/2, full_width/2-columns_radius]

daq_length = 0.72
daq_width = 0.6
daq_height = 0.2
daq_top_depth = 0.53 + 0.1 + 0.05

# Define the meshes
with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', 0.3)
    platform = geom.add_box([-platform_frame_length/2, -platform_frame_width/2, -platform_frame_top_depth-platform_frame_thickness], [platform_frame_length,platform_frame_width,platform_frame_thickness])
    cutout = geom.add_box([-cutout_length/2, -cutout_width/2, -platform_frame_top_depth-platform_frame_thickness], [cutout_length,cutout_width,platform_frame_thickness])
    cyl1 = geom.add_cylinder([-columns_xy[0], -columns_xy[1], -columns_draft],[0, 0, columns_draft+.01], columns_radius)
    cyl2 = geom.add_cylinder([-columns_xy[0], columns_xy[1], -columns_draft],[0, 0, columns_draft+.01], columns_radius)
    cyl3 = geom.add_cylinder([columns_xy[0], -columns_xy[1], -columns_draft],[0, 0, columns_draft+.01], columns_radius)
    cyl4 = geom.add_cylinder([columns_xy[0], columns_xy[1], -columns_draft],[0, 0, columns_draft+.01], columns_radius)
    DAQ = geom.add_box([-daq_length/2, -daq_width/2, -daq_top_depth-daq_height], [daq_length,daq_width,daq_height])
    platformFrame = geom.boolean_difference(platform,cutout)
    geom.boolean_union([platformFrame,DAQ,cyl1,cyl2,cyl3,cyl4])
    mesh_platform = geom.generate_mesh()

with pygmsh.geo.Geometry() as geom:
    flap1 = geom.add_polygon(
            [[-flap_center_distance_apart/2 - flap_thickness_bottom/2, -flap_width/2, -flap_draft],
            [-flap_center_distance_apart/2 + flap_thickness_bottom/2, -flap_width/2, -flap_draft],
            [-flap_center_distance_apart/2 + flap_thickness_top/2, -flap_width/2, flap_height-flap_draft],
            [-flap_center_distance_apart/2 - flap_thickness_top/2, -flap_width/2, flap_height-flap_draft]],mesh_size=0.1)
    geom.extrude(flap1,[0,flap_width,0])
    mesh_flap1 = geom.generate_mesh()

with pygmsh.geo.Geometry() as geom:
    flap2 = geom.add_polygon(
            [[flap_center_distance_apart/2 - flap_thickness_bottom/2, -flap_width/2, -flap_draft],
            [flap_center_distance_apart/2 + flap_thickness_bottom/2, -flap_width/2, -flap_draft],
            [flap_center_distance_apart/2 + flap_thickness_top/2, -flap_width/2, flap_height-flap_draft],
            [flap_center_distance_apart/2 - flap_thickness_top/2, -flap_width/2, flap_height-flap_draft]],mesh_size=0.1)
    geom.extrude(flap2,[0,flap_width,0])
    mesh_flap2 = geom.generate_mesh()

# Mass and inertia
m_platform = 189.8
J_platform = 30
m_flaps = 23.1
J_flaps = 1.19

# Pre-clip meshes at the waterplane (z=0) so compute_hydrostatics() uses only the submerged volume.
# User is responsible for clipping (not done inside HydroLinearMCKF).
fb_platform = cpt.FloatingBody(mesh=mesh_platform, name="platform")
fb_platform = fb_platform.keep_immersed_part()
mesh_platform_clipped = fb_platform.mesh  # clipped Capytaine mesh for hydrostatics/BEM

# Flaps don't need to be clipped because they are fully submerged but still good practice in case geometry changes in the future.
fb_flap1 = cpt.FloatingBody(mesh=mesh_flap1, name="flap1")
fb_flap1 = fb_flap1.keep_immersed_part()
mesh_flap1_clipped = fb_flap1.mesh  # clipped Capytaine mesh for hydrostatics/BEM

fb_flap2 = cpt.FloatingBody(mesh=mesh_flap2, name="flap2")
fb_flap2 = fb_flap2.keep_immersed_part()
mesh_flap2_clipped = fb_flap2.mesh  # clipped Capytaine mesh for hydrostatics/BEM

# body_inputs: spar first (key=1) then torus (key=2) to match MBD Cartesian ordering [spar, float]
body_inputs = {
    1: {
        "name": "platform",
        "mesh": mesh_platform_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_platform, m_platform, m_platform, J_platform, J_platform, J_platform],
    },
    2: {
        "name": "flap1",
        "mesh": mesh_flap1_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_flaps, m_flaps, m_flaps, J_flaps, J_flaps, J_flaps],
    },
    3: {
        "name": "flap2",
        "mesh": mesh_flap2_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_flaps, m_flaps, m_flaps, J_flaps, J_flaps, J_flaps],
    },
}

water_depth = 2 # depth of wave tank

m0 = [m_platform, m_flaps, m_flaps]
J0 = [J_platform, J_flaps, J_flaps]

# Examples_linearization/single_body_moorpy/moorpy_inputs.py
"""
MoorPy mooring-system inputs for the single_body_moorpy example.

Coordinates are NOT repeated here.  This file contains only:
* Line-type physical properties.
* Mooring line topology expressed as references to M4E point IDs.

Anchor and fairlead coordinates are owned exclusively by M4E_inputs.py:
  * Anchor  → Initial_Points["GR"][0]  (index 0 in the GR list)
  * Fairlead → Initial_Points["BD"][1][0]  (body 1, point index 0)

Line-type property names map to MoorPy System.addLineType as follows:
  diameter            → d      (volume-equivalent diameter [m])
  mass_per_length     → mass   (linear mass density [kg/m])
  axial_stiffness     → EA     (extensional stiffness [N])
  drag_coefficient    → Cd     (transverse Morison drag [-])
  added_mass_coefficient → Ca  (transverse added mass [-])

Water environment
-----------------
depth : float
    Water depth below the mean free surface [m].
rho : float
    Water density [kg/m³].
g : float
    Gravitational acceleration [m/s²].
"""

# ---------------------------------------------------------------------------
# Water environment (forwarded to moorpy.System)
# ---------------------------------------------------------------------------
depth: float = 100.0   # [m]
rho:   float = 1025.0  # [kg/m³]
g:     float = 9.81    # [m/s²]

# ---------------------------------------------------------------------------
# Line types
# ---------------------------------------------------------------------------
line_types = {
    "studless_chain": {
        "diameter":               0.09,     # volume-equivalent diameter [m]
        "mass_per_length":        160.0,    # [kg/m]
        "axial_stiffness":        854.0e6,  # EA [N]
        "drag_coefficient":       2.4,      # Cd transverse [-]
        "added_mass_coefficient": 1.0,      # Ca transverse [-]
    },
}

# ---------------------------------------------------------------------------
# Mooring lines
# ---------------------------------------------------------------------------
# Topology expressed entirely via M4E point IDs.
# Coordinates are resolved at adapter construction time from Initial_Points.
mooring_lines = [
    {
        "name":               "line_1",
        "line_type":          "studless_chain",
        "unstretched_length": 250.0,   # [m]  — set so line hangs in catenary at q0
        "num_segments":       20,
        "anchor": {
            "kind":     "GR",
            "point_id": 0,             # Initial_Points["GR"][0]
        },
        "fairlead": {
            "kind":     "BD",
            "body_id":  1,             # body number 1 (1-based)
            "point_id": 0,             # Initial_Points["BD"][1][0]
        },
    },
]

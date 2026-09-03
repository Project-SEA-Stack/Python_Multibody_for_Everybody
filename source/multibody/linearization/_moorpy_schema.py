# multibody/linearization/_moorpy_schema.py
"""
MoorPy input schema — dataclasses and validation.

This module owns the data model for the mooring-system inputs that are
consumed by the MoorPy adapter.  It intentionally does **not** import MoorPy
so that the schema can be validated, tested, and imported without the optional
MoorPy dependency being present.

Point-ownership rules
---------------------
* Anchor coordinates live exclusively in ``Initial_Points["GR"]`` (a list of
  ``[x, z]`` pairs, 0-based index).
* Fairlead coordinates live exclusively in ``Initial_Points["BD"]`` (a dict
  keyed by 1-based body number, each value a 0-based list of ``[x_local,
  z_local]`` body-frame vectors).
* The schema stores only integer IDs that reference those M4E structures —
  never duplicated coordinates.

MoorPy line-type property mapping
----------------------------------
When the adapter constructs the MoorPy ``System`` it will call::

    system.addLineType(
        type_string = name,
        d           = lt.diameter,           # volume-equivalent diameter [m]
        mass        = lt.mass_per_length,    # line mass density [kg/m]
        EA          = lt.axial_stiffness,    # axial stiffness [N]
    )

and then update the resulting dict with the optional hydrodynamic coefficients.
Keeping the schema forward-compatible with 3D extension: field names are
generic rather than explicitly 2D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LineType:
    """Properties for one mooring line type.

    All fields map directly to a MoorPy ``System.addLineType`` call followed
    by updating the resulting dict.

    Parameters
    ----------
    diameter:
        Volume-equivalent diameter [m].  Maps to MoorPy ``d``.
    mass_per_length:
        Linear mass density [kg/m].  Maps to MoorPy ``mass``.
    axial_stiffness:
        Axial (extensional) stiffness EA [N].  Maps to MoorPy ``EA``.
    drag_coefficient:
        Transverse Morison drag coefficient Cd [-].  Maps to MoorPy ``Cd``.
    added_mass_coefficient:
        Transverse added-mass coefficient Ca [-].  Maps to MoorPy ``Ca``.
    drag_coefficient_axial:
        Tangential (axial) drag coefficient CdAx [-].  Maps to MoorPy ``CdAx``.
    added_mass_coefficient_axial:
        Tangential (axial) added-mass coefficient CaAx [-].  Maps to MoorPy ``CaAx``.
    """

    diameter: float
    mass_per_length: float
    axial_stiffness: float
    drag_coefficient: float = 1.2
    added_mass_coefficient: float = 1.0
    drag_coefficient_axial: float = 0.4
    added_mass_coefficient_axial: float = 0.0


@dataclass
class PointRef:
    """A reference to one point in M4E ``Initial_Points``.

    Parameters
    ----------
    kind:
        ``"GR"`` for a ground/anchor point or ``"BD"`` for a body-defined
        (fairlead) point.
    point_id:
        0-based index into ``Initial_Points["GR"]`` (when ``kind == "GR"``)
        or into ``Initial_Points["BD"][body_id]`` (when ``kind == "BD"``).
    body_id:
        1-based body number.  Required (and validated) when ``kind == "BD"``;
        ignored when ``kind == "GR"``.
    """

    kind: str
    point_id: int
    body_id: int = 0


@dataclass
class MooringLine:
    """Definition of a single mooring line.

    Parameters
    ----------
    name:
        Unique identifier for this line.
    line_type:
        Key into the ``line_types`` dict of the parent :class:`MoorPyInputs`.
    unstretched_length:
        Unstretched (natural) length of the line [m].
    num_segments:
        Number of finite-element segments to discretise the line into.
    anchor:
        Reference to the anchor point (must be ``kind == "GR"``).
    fairlead:
        Reference to the fairlead point (must be ``kind == "BD"``).
    """

    name: str
    line_type: str
    unstretched_length: float
    num_segments: int
    anchor: PointRef
    fairlead: PointRef


@dataclass
class MoorPyInputs:
    """Complete MoorPy mooring-system inputs for one scenario.

    Typically constructed via :func:`from_module` from a
    ``moorpy_inputs.py`` example file.

    Parameters
    ----------
    line_types:
        Dict mapping line-type name to its :class:`LineType` properties.
    mooring_lines:
        Ordered list of :class:`MooringLine` definitions.
    depth:
        Water depth [m].  Passed to ``moorpy.System(depth=...)``.
    rho:
        Water density [kg/m³].
    g:
        Gravitational acceleration [m/s²].
    """

    line_types: Dict[str, LineType]
    mooring_lines: List[MooringLine]
    depth: float = 200.0
    rho: float = 1025.0
    g: float = 9.81


# ---------------------------------------------------------------------------
# Factory: build from a moorpy_inputs module
# ---------------------------------------------------------------------------

def from_module(module: ModuleType) -> MoorPyInputs:
    """Build a :class:`MoorPyInputs` from a ``moorpy_inputs.py`` module.

    The module must expose two module-level variables:

    * ``line_types``: ``dict[str, dict]`` — each inner dict has keys matching
      :class:`LineType` field names.
    * ``mooring_lines``: ``list[dict]`` — each dict has keys matching
      :class:`MooringLine` field names, with ``anchor`` and ``fairlead``
      values that are dicts matching :class:`PointRef` field names.

    Optional module-level floats ``depth``, ``rho``, and ``g`` are forwarded
    to :class:`MoorPyInputs`.

    Parameters
    ----------
    module:
        An imported ``moorpy_inputs`` module.

    Returns
    -------
    MoorPyInputs
    """
    raw_types = getattr(module, "line_types", {})
    raw_lines = getattr(module, "mooring_lines", [])

    lt_objects: Dict[str, LineType] = {
        name: LineType(**props) for name, props in raw_types.items()
    }

    ml_objects: List[MooringLine] = []
    for raw in raw_lines:
        raw = dict(raw)  # shallow copy — do not mutate caller's data
        raw["anchor"]   = PointRef(**raw["anchor"])
        raw["fairlead"] = PointRef(**raw["fairlead"])
        ml_objects.append(MooringLine(**raw))

    return MoorPyInputs(
        line_types    = lt_objects,
        mooring_lines = ml_objects,
        depth         = float(getattr(module, "depth", 200.0)),
        rho           = float(getattr(module, "rho",   1025.0)),
        g             = float(getattr(module, "g",     9.81)),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_moorpy_inputs(
    inputs: MoorPyInputs,
    initial_points: dict,
) -> None:
    """Validate *inputs* against the M4E ``Initial_Points`` structure.

    Checks performed
    ----------------
    * All line-type property values are positive where required.
    * Every line name is unique.
    * Every line references a known line type.
    * ``unstretched_length > 0`` and ``num_segments >= 1``.
    * Anchors reference ``"GR"`` points at valid indices.
    * Fairleads reference ``"BD"`` points at valid body and point indices.

    Parameters
    ----------
    inputs:
        The :class:`MoorPyInputs` object to check.
    initial_points:
        The M4E ``Initial_Points`` dict (must contain ``"GR"`` list and
        ``"BD"`` dict).

    Raises
    ------
    ValueError
        On the first validation failure encountered, with a descriptive
        message.
    """
    gr_list: list = initial_points.get("GR", [])
    bd_dict: dict = initial_points.get("BD", {})

    # --- validate line types ---
    for name, lt in inputs.line_types.items():
        if lt.diameter <= 0:
            raise ValueError(
                f"LineType '{name}': diameter must be positive, got {lt.diameter}"
            )
        if lt.mass_per_length <= 0:
            raise ValueError(
                f"LineType '{name}': mass_per_length must be positive, "
                f"got {lt.mass_per_length}"
            )
        if lt.axial_stiffness <= 0:
            raise ValueError(
                f"LineType '{name}': axial_stiffness must be positive, "
                f"got {lt.axial_stiffness}"
            )

    # --- validate mooring lines ---
    seen_names: set = set()
    for line in inputs.mooring_lines:

        # unique names
        if line.name in seen_names:
            raise ValueError(f"Duplicate mooring line name: '{line.name}'")
        seen_names.add(line.name)

        # known line type
        if line.line_type not in inputs.line_types:
            raise ValueError(
                f"Line '{line.name}' references unknown line type "
                f"'{line.line_type}' (defined types: "
                f"{sorted(inputs.line_types)})"
            )

        # positive length and valid segment count
        if line.unstretched_length <= 0:
            raise ValueError(
                f"Line '{line.name}': unstretched_length must be positive, "
                f"got {line.unstretched_length}"
            )
        if line.num_segments < 1:
            raise ValueError(
                f"Line '{line.name}': num_segments must be >= 1, "
                f"got {line.num_segments}"
            )

        # anchor must be GR
        anch = line.anchor
        if anch.kind != "GR":
            raise ValueError(
                f"Line '{line.name}': anchor must have kind='GR', "
                f"got '{anch.kind}'"
            )
        if anch.point_id < 0 or anch.point_id >= len(gr_list):
            raise ValueError(
                f"Line '{line.name}': anchor point_id={anch.point_id} is out "
                f"of range — Initial_Points['GR'] has {len(gr_list)} entries "
                f"(valid indices 0..{len(gr_list)-1})"
            )

        # fairlead must be BD
        fair = line.fairlead
        if fair.kind != "BD":
            raise ValueError(
                f"Line '{line.name}': fairlead must have kind='BD', "
                f"got '{fair.kind}'"
            )
        if fair.body_id not in bd_dict:
            raise ValueError(
                f"Line '{line.name}': fairlead body_id={fair.body_id} not "
                f"found in Initial_Points['BD'] "
                f"(available bodies: {sorted(bd_dict)})"
            )
        body_pt_list = bd_dict[fair.body_id]
        if fair.point_id < 0 or fair.point_id >= len(body_pt_list):
            raise ValueError(
                f"Line '{line.name}': fairlead point_id={fair.point_id} is "
                f"out of range — Initial_Points['BD'][{fair.body_id}] has "
                f"{len(body_pt_list)} entries "
                f"(valid indices 0..{len(body_pt_list)-1})"
            )

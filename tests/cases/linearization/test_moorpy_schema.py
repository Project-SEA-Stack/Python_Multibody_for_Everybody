# tests/cases/linearization/test_moorpy_schema.py
"""
Unit tests for the MoorPy input schema and reference validation.

These tests cover:
* :class:`multibody.linearization._moorpy_schema.LineType`
* :class:`multibody.linearization._moorpy_schema.PointRef`
* :class:`multibody.linearization._moorpy_schema.MooringLine`
* :class:`multibody.linearization._moorpy_schema.MoorPyInputs`
* :func:`multibody.linearization._moorpy_schema.from_module`
* :func:`multibody.linearization._moorpy_schema.validate_moorpy_inputs`

MoorPy is NOT imported anywhere in this file.
"""

from __future__ import annotations

import sys
import types

import pytest

from multibody.linearization._moorpy_schema import (
    LineType,
    MooringLine,
    MoorPyInputs,
    PointRef,
    from_module,
    validate_moorpy_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_line_type() -> LineType:
    return LineType(
        diameter=0.09,
        mass_per_length=160.0,
        axial_stiffness=854e6,
    )


@pytest.fixture()
def simple_initial_points() -> dict:
    """M4E Initial_Points with one GR anchor and one BD fairlead."""
    return {
        "GR": [[-50.0, -100.0]],        # GR[0]
        "BD": {1: [[2.0, -5.0]]},       # BD[body=1][point=0]
    }


@pytest.fixture()
def simple_inputs(simple_line_type) -> MoorPyInputs:
    return MoorPyInputs(
        line_types={"chain": simple_line_type},
        mooring_lines=[
            MooringLine(
                name="line_1",
                line_type="chain",
                unstretched_length=250.0,
                num_segments=20,
                anchor=PointRef(kind="GR", point_id=0),
                fairlead=PointRef(kind="BD", body_id=1, point_id=0),
            )
        ],
    )


# ---------------------------------------------------------------------------
# LineType dataclass
# ---------------------------------------------------------------------------

class TestLineType:
    def test_required_fields(self):
        lt = LineType(diameter=0.1, mass_per_length=50.0, axial_stiffness=1e8)
        assert lt.diameter == 0.1
        assert lt.mass_per_length == 50.0
        assert lt.axial_stiffness == 1e8

    def test_optional_defaults(self):
        lt = LineType(diameter=0.1, mass_per_length=50.0, axial_stiffness=1e8)
        assert lt.drag_coefficient == pytest.approx(1.2)
        assert lt.added_mass_coefficient == pytest.approx(1.0)

    def test_override_optional(self):
        lt = LineType(
            diameter=0.1,
            mass_per_length=50.0,
            axial_stiffness=1e8,
            drag_coefficient=2.4,
            added_mass_coefficient=0.8,
        )
        assert lt.drag_coefficient == pytest.approx(2.4)
        assert lt.added_mass_coefficient == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# PointRef dataclass
# ---------------------------------------------------------------------------

class TestPointRef:
    def test_gr_ref(self):
        ref = PointRef(kind="GR", point_id=2)
        assert ref.kind == "GR"
        assert ref.point_id == 2
        assert ref.body_id == 0  # default sentinel

    def test_bd_ref(self):
        ref = PointRef(kind="BD", body_id=1, point_id=0)
        assert ref.kind == "BD"
        assert ref.body_id == 1
        assert ref.point_id == 0


# ---------------------------------------------------------------------------
# from_module
# ---------------------------------------------------------------------------

def _make_module(name: str, **attrs) -> types.ModuleType:
    """Create a synthetic module object with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class TestFromModule:
    def test_round_trip(self):
        mod = _make_module(
            "fake_moorpy_inputs",
            line_types={
                "chain": {
                    "diameter": 0.09,
                    "mass_per_length": 160.0,
                    "axial_stiffness": 854e6,
                }
            },
            mooring_lines=[
                {
                    "name": "line_1",
                    "line_type": "chain",
                    "unstretched_length": 250.0,
                    "num_segments": 20,
                    "anchor":   {"kind": "GR", "point_id": 0},
                    "fairlead": {"kind": "BD", "body_id": 1, "point_id": 0},
                }
            ],
            depth=100.0,
            rho=1025.0,
            g=9.81,
        )
        inputs = from_module(mod)

        assert isinstance(inputs.line_types["chain"], LineType)
        assert inputs.line_types["chain"].diameter == pytest.approx(0.09)

        assert len(inputs.mooring_lines) == 1
        line = inputs.mooring_lines[0]
        assert isinstance(line, MooringLine)
        assert isinstance(line.anchor, PointRef)
        assert isinstance(line.fairlead, PointRef)
        assert line.anchor.kind == "GR"
        assert line.fairlead.kind == "BD"
        assert line.fairlead.body_id == 1

        assert inputs.depth == pytest.approx(100.0)
        assert inputs.rho == pytest.approx(1025.0)

    def test_defaults_when_missing(self):
        mod = _make_module(
            "fake_minimal_inputs",
            line_types={
                "chain": {
                    "diameter": 0.09,
                    "mass_per_length": 160.0,
                    "axial_stiffness": 854e6,
                }
            },
            mooring_lines=[],
        )
        inputs = from_module(mod)
        assert inputs.depth == pytest.approx(200.0)
        assert inputs.rho == pytest.approx(1025.0)
        assert inputs.g == pytest.approx(9.81)

    def test_does_not_mutate_source_dicts(self):
        """from_module must not modify the caller's raw dict objects."""
        raw_line = {
            "name": "line_1",
            "line_type": "chain",
            "unstretched_length": 100.0,
            "num_segments": 10,
            "anchor":   {"kind": "GR", "point_id": 0},
            "fairlead": {"kind": "BD", "body_id": 1, "point_id": 0},
        }
        mod = _make_module(
            "fake_immutable_inputs",
            line_types={
                "chain": {"diameter": 0.09, "mass_per_length": 50.0, "axial_stiffness": 1e8}
            },
            mooring_lines=[raw_line],
        )
        from_module(mod)
        # anchor and fairlead should still be dicts in the original list
        assert isinstance(raw_line["anchor"], dict)
        assert isinstance(raw_line["fairlead"], dict)


# ---------------------------------------------------------------------------
# validate_moorpy_inputs — happy-path
# ---------------------------------------------------------------------------

class TestValidateHappyPath:
    def test_single_line_passes(self, simple_inputs, simple_initial_points):
        # Should not raise
        validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_multiple_lines_different_anchors(self, simple_line_type):
        initial_points = {
            "GR": [[-50.0, -100.0], [50.0, -100.0]],
            "BD": {1: [[2.0, -5.0], [-2.0, -5.0]]},
        }
        inputs = MoorPyInputs(
            line_types={"chain": simple_line_type},
            mooring_lines=[
                MooringLine(
                    name="line_A",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=0),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=0),
                ),
                MooringLine(
                    name="line_B",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=1),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=1),
                ),
            ],
        )
        validate_moorpy_inputs(inputs, initial_points)

    def test_shared_anchor_two_lines(self, simple_line_type):
        """Two lines may share the same GR anchor."""
        initial_points = {
            "GR": [[-50.0, -100.0]],
            "BD": {1: [[2.0, -5.0], [-2.0, -5.0]]},
        }
        inputs = MoorPyInputs(
            line_types={"chain": simple_line_type},
            mooring_lines=[
                MooringLine(
                    name="line_A",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=0),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=0),
                ),
                MooringLine(
                    name="line_B",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=0),   # same anchor
                    fairlead=PointRef(kind="BD", body_id=1, point_id=1),
                ),
            ],
        )
        validate_moorpy_inputs(inputs, initial_points)

    def test_fairleads_on_multiple_bodies(self, simple_line_type):
        initial_points = {
            "GR": [[-50.0, -100.0], [50.0, -100.0]],
            "BD": {
                1: [[2.0, -5.0]],
                2: [[-2.0, -5.0]],
            },
        }
        inputs = MoorPyInputs(
            line_types={"chain": simple_line_type},
            mooring_lines=[
                MooringLine(
                    name="line_1",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=0),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=0),
                ),
                MooringLine(
                    name="line_2",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=1),
                    fairlead=PointRef(kind="BD", body_id=2, point_id=0),
                ),
            ],
        )
        validate_moorpy_inputs(inputs, initial_points)


# ---------------------------------------------------------------------------
# validate_moorpy_inputs — failure cases
# ---------------------------------------------------------------------------

class TestValidateLineTypeErrors:
    def test_zero_diameter(self, simple_inputs, simple_initial_points):
        simple_inputs.line_types["chain"].diameter = 0.0
        with pytest.raises(ValueError, match="diameter"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_negative_mass(self, simple_inputs, simple_initial_points):
        simple_inputs.line_types["chain"].mass_per_length = -1.0
        with pytest.raises(ValueError, match="mass_per_length"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_zero_axial_stiffness(self, simple_inputs, simple_initial_points):
        simple_inputs.line_types["chain"].axial_stiffness = 0.0
        with pytest.raises(ValueError, match="axial_stiffness"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)


class TestValidateLineErrors:
    def test_duplicate_line_name(self, simple_line_type, simple_initial_points):
        initial_points = {
            "GR": [[-50.0, -100.0], [50.0, -100.0]],
            "BD": {1: [[2.0, -5.0], [-2.0, -5.0]]},
        }
        inputs = MoorPyInputs(
            line_types={"chain": simple_line_type},
            mooring_lines=[
                MooringLine(
                    name="line_1",
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=0),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=0),
                ),
                MooringLine(
                    name="line_1",          # duplicate
                    line_type="chain",
                    unstretched_length=250.0,
                    num_segments=20,
                    anchor=PointRef(kind="GR", point_id=1),
                    fairlead=PointRef(kind="BD", body_id=1, point_id=1),
                ),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            validate_moorpy_inputs(inputs, initial_points)

    def test_unknown_line_type(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].line_type = "unknown_type"
        with pytest.raises(ValueError, match="unknown line type"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_zero_unstretched_length(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].unstretched_length = 0.0
        with pytest.raises(ValueError, match="unstretched_length"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_negative_unstretched_length(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].unstretched_length = -10.0
        with pytest.raises(ValueError, match="unstretched_length"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_zero_num_segments(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].num_segments = 0
        with pytest.raises(ValueError, match="num_segments"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)


class TestValidateAnchorErrors:
    def test_anchor_wrong_kind(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].anchor.kind = "BD"
        with pytest.raises(ValueError, match="anchor must have kind='GR'"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_anchor_point_id_out_of_range(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].anchor.point_id = 5  # GR only has 1 entry
        with pytest.raises(ValueError, match="anchor point_id"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_anchor_negative_point_id(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].anchor.point_id = -1
        with pytest.raises(ValueError, match="anchor point_id"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_empty_gr_list(self, simple_inputs):
        initial_points = {"GR": [], "BD": {1: [[2.0, -5.0]]}}
        with pytest.raises(ValueError, match="anchor point_id"):
            validate_moorpy_inputs(simple_inputs, initial_points)


class TestValidateFairleadErrors:
    def test_fairlead_wrong_kind(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].fairlead.kind = "GR"
        with pytest.raises(ValueError, match="fairlead must have kind='BD'"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_fairlead_body_id_not_in_bd(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].fairlead.body_id = 99
        with pytest.raises(ValueError, match="fairlead body_id"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_fairlead_point_id_out_of_range(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].fairlead.point_id = 5  # body 1 has only 1 point
        with pytest.raises(ValueError, match="fairlead point_id"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)

    def test_fairlead_negative_point_id(self, simple_inputs, simple_initial_points):
        simple_inputs.mooring_lines[0].fairlead.point_id = -1
        with pytest.raises(ValueError, match="fairlead point_id"):
            validate_moorpy_inputs(simple_inputs, simple_initial_points)


# ---------------------------------------------------------------------------
# Load the real example module and validate it
# ---------------------------------------------------------------------------

class TestRealExampleModule:
    """
    Load the installed single_body_moorpy/moorpy_inputs.py and validate it
    against a matching Initial_Points structure.  MoorPy is not required.
    """

    def test_load_and_validate_example(self):
        import importlib
        mod = importlib.import_module(
            "Examples_linearization.single_body_moorpy.moorpy_inputs"
        )
        inputs = from_module(mod)

        # Build a matching Initial_Points for the single-body example
        # (one GR anchor, one BD fairlead on body 1)
        initial_points = {
            "GR": [[-50.0, -100.0]],
            "BD": {1: [[2.0, -5.0]]},
        }

        # Must not raise
        validate_moorpy_inputs(inputs, initial_points)

    def test_example_has_one_line(self):
        import importlib
        mod = importlib.import_module(
            "Examples_linearization.single_body_moorpy.moorpy_inputs"
        )
        inputs = from_module(mod)
        assert len(inputs.mooring_lines) == 1

    def test_example_anchor_is_gr(self):
        import importlib
        mod = importlib.import_module(
            "Examples_linearization.single_body_moorpy.moorpy_inputs"
        )
        inputs = from_module(mod)
        assert inputs.mooring_lines[0].anchor.kind == "GR"

    def test_example_fairlead_is_bd(self):
        import importlib
        mod = importlib.import_module(
            "Examples_linearization.single_body_moorpy.moorpy_inputs"
        )
        inputs = from_module(mod)
        assert inputs.mooring_lines[0].fairlead.kind == "BD"

    def test_example_no_coordinate_duplication(self):
        """The moorpy_inputs module must not define any coordinate lists."""
        import importlib
        mod = importlib.import_module(
            "Examples_linearization.single_body_moorpy.moorpy_inputs"
        )
        # The module must NOT have keys like 'GR', 'BD', or 'Initial_Points'
        assert not hasattr(mod, "Initial_Points"), (
            "moorpy_inputs.py must not define Initial_Points — "
            "coordinates are owned by M4E_inputs.py"
        )
        assert not hasattr(mod, "GR"), (
            "moorpy_inputs.py must not define GR coordinates"
        )
        assert not hasattr(mod, "BD"), (
            "moorpy_inputs.py must not define BD coordinates"
        )

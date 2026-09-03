# tests/cases/linearization/_moorpy_regression_fixtures.py
"""
Shared M4E/MoorPy system definition for the frozen MoorPy regression tests.

Used by ``test_moorpy_standalone_reference.py`` and
``test_moorpy_raft_reference.py`` so the same physical mooring case is not
duplicated across both files. This is the identical single-body,
offset-fairlead mooring case manually verified in
``Examples_linearization/verification_moorpy/`` and
``Examples_linearization/verification_raft_moorpy/`` — see
``tests/cases/linearization/truth/standalone_moorpy_reference.npz`` and
``raft_moorpy_reference.npz`` for the frozen reference matrices generated
from those two examples.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from multibody import MbdSystem, normalize_prismatic
from multibody.linearization._moorpy_schema import (
    LineType,
    MooringLine,
    MoorPyInputs,
    PointRef,
)


def build_moorpy_case() -> Tuple[MbdSystem, np.ndarray, np.ndarray, MoorPyInputs]:
    """Build the frozen regression case: one float, one line, offset fairlead.

    Physical values match Examples_linearization/verification_moorpy and
    Examples_linearization/verification_raft_moorpy: anchor at (-50,-100),
    fairlead at body-local (2,-5), q0 = [0,0,0] (undisplaced).

    Returns
    -------
    mbd_sys, q0, mainNumVars, moorpy_inputs
    """
    mbd_sys = MbdSystem(
        joints=[[0, 1]],
        types=["F"],
        parent_cg_to_joint=[[0.0, 0.0]],
        joint_to_child_cg=[[np.nan, np.nan]],
        prismatic_direction=normalize_prismatic([[np.nan, np.nan]]),
        Initial_Points={
            "GR": [[-50.0, -100.0]],
            "BD": {1: [[2.0, -5.0]]},
        },
    )
    nq = len(mbd_sys.Q)
    q0 = np.zeros(nq)
    mainNumVars = np.zeros(2 * nq)  # [q0, qd=0]; no extra body-data parameters

    moorpy_inputs = MoorPyInputs(
        depth=100.0, rho=1025.0, g=9.81,
        line_types={
            "studless_chain": LineType(
                diameter=0.09,
                mass_per_length=160.0,
                axial_stiffness=854e6,
                drag_coefficient=2.4,
                added_mass_coefficient=1.0,
            )
        },
        mooring_lines=[
            MooringLine(
                name="line_1", line_type="studless_chain",
                unstretched_length=250.0, num_segments=20,
                anchor=PointRef(kind="GR", point_id=0),
                fairlead=PointRef(kind="BD", body_id=1, point_id=0),
            )
        ],
    )
    return mbd_sys, q0, mainNumVars, moorpy_inputs

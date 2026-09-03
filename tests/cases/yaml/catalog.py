# -*- coding: utf-8 -*-
"""Catalog of YAML validation cases.

Each :class:`YamlCaseEntry` pairs a YAML model/simulation file (loaded via
:func:`multibody.load_yaml_as_example`) with the PyChrono reference trajectory
stored under ``tests/cases/yaml/truth/<case_id>/``.

The catalog is the single source of truth for the YAML validation test suite:
any new case is added here and is automatically picked up by the runner in
``tests.cases.yaml.runner`` and the standalone script in
``tests.cases.yaml.run_validation``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class YamlCaseEntry:
    """Static description of a single YAML validation case.

    Attributes
    ----------
    case_id:
        Short identifier (used in CLI output and ``get_entry`` lookups).
    yaml_basename:
        Stem of the YAML pair (without ``.model.yaml`` / ``.simulation.yaml``)
        relative to the repository root. Both files must exist at
        ``tests/cases/yaml/models/<stem>.model.yaml`` and
        ``tests/cases/yaml/models/<stem>.simulation.yaml``.
    chrono_dir:
        Path (relative to the repository root) to the local truth folder
        ``tests/cases/yaml/truth/<case_id>/`` that contains the PyChrono
        reference trajectory ``trajectory.csv``.
    bodies_to_compare:
        Tuple of 0-based body indices to use in the comparison. ``None``
        means "compare every body present in the truth array".
    description:
        Human-readable summary.
    """

    case_id: str
    yaml_basename: str
    chrono_dir: str
    bodies_to_compare: Optional[Tuple[int, ...]] = None
    description: str = ""


YAML_CASES: Tuple[YamlCaseEntry, ...] = (
    YamlCaseEntry(
        case_id="yaml_double_pendulum",
        yaml_basename="double_pendulum",
        chrono_dir="tests/cases/yaml/truth/yaml_double_pendulum",
        description="Planar double pendulum, two revolute joints.",
    ),
    YamlCaseEntry(
        case_id="yaml_float",
        yaml_basename="float",
        chrono_dir="tests/cases/yaml/truth/yaml_float",
        description="Single floating body (no joints).",
    ),
    YamlCaseEntry(
        case_id="yaml_pendulum",
        yaml_basename="pendulum",
        chrono_dir="tests/cases/yaml/truth/yaml_pendulum",
        description="Single revolute pendulum.",
    ),
    YamlCaseEntry(
        case_id="yaml_pendulum_slider",
        yaml_basename="pendulum_slider",
        chrono_dir="tests/cases/yaml/truth/yaml_pendulum_slider",
        description="Pendulum carrying a prismatic slider.",
    ),
    YamlCaseEntry(
        case_id="yaml_pendulum_rotor",
        yaml_basename="pendulum_rotor",
        chrono_dir="tests/cases/yaml/truth/yaml_pendulum_rotor",
        description="Pendulum with a driven rotor body.",
    ),
    YamlCaseEntry(
        case_id="yaml_slider_w_dpend",
        yaml_basename="slider_w_dpend",
        chrono_dir="tests/cases/yaml/truth/yaml_slider_w_dpend",
        description="Slider on a track carrying a double pendulum.",
    ),
    YamlCaseEntry(
        case_id="yaml_spiderfloat",
        yaml_basename="spiderfloat",
        chrono_dir="tests/cases/yaml/truth/yaml_spiderfloat",
        # Original compare.py uses r2plot=[3,4,5,6] for this case only.
        bodies_to_compare=(3, 4, 5, 6),
        description="Floating spider/WEC platform; only legs (bodies 3-6) compared.",
    ),
)


# A smoke subset that exercises every joint family in the YAML suite at low
# cost: pure-revolute baseline, floating base, prismatic+revolute, and the
# multi-body floating spider with its body-subset comparison.
SMOKE_YAML_CASES: Tuple[YamlCaseEntry, ...] = (
    YAML_CASES[0],  # yaml_double_pendulum
    YAML_CASES[1],  # yaml_float
    YAML_CASES[3],  # yaml_pendulum_slider
    YAML_CASES[6],  # yaml_spiderfloat
)


def get_entry(case: Union[YamlCaseEntry, str]) -> YamlCaseEntry:
    """Resolve ``case`` (a :class:`YamlCaseEntry` or a ``case_id`` string)
    into a catalog entry.
    """
    if isinstance(case, YamlCaseEntry):
        return case
    if isinstance(case, str):
        for entry in YAML_CASES:
            if entry.case_id == case:
                return entry
        raise KeyError(f"Unknown YAML case id: {case!r}")
    raise TypeError(
        f"case must be YamlCaseEntry or str, got {type(case).__name__}"
    )


__all__ = [
    "YamlCaseEntry",
    "YAML_CASES",
    "SMOKE_YAML_CASES",
    "get_entry",
]

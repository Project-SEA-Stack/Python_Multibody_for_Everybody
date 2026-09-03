# -*- coding: utf-8 -*-
"""Standalone YAML validation driver.

Loops over every :data:`tests.cases.yaml.catalog.YAML_CASES` entry, runs the
M4E integration, and compares the result against the PyChrono reference
trajectory. Prints a per-case NRMSE summary at the end so the script
output is easy to diff between branches.

Usage (from the repository root):

    python -m tests.cases.yaml.run_validation
    python -m tests.cases.yaml.run_validation yaml_double_pendulum yaml_pendulum

When no positional argument is given, the full catalog runs. Otherwise
only the named ``case_id``s execute.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Iterable, List, Tuple

import numpy as np

# Allow ``python source/multibody/...`` style invocations from any CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.cases.yaml.catalog import YAML_CASES, YamlCaseEntry, get_entry
from tests.cases.yaml.runner import compare_against_chrono


def _select_cases(names: Iterable[str]) -> Tuple[YamlCaseEntry, ...]:
    names = list(names)
    if not names:
        return YAML_CASES
    return tuple(get_entry(n) for n in names)


def _format_row(case_id: str, pos_max: float, vel_max: float, status: str) -> str:
    return f"  {case_id:<24s}  pos NRMSE={pos_max:.3e}   vel NRMSE={vel_max:.3e}   [{status}]"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        help="Optional case_id filters (default: run the whole catalog).",
    )
    args = parser.parse_args(argv)

    cases = _select_cases(args.cases)
    summary: List[Tuple[str, float, float, str]] = []

    for entry in cases:
        print("\n" + "#" * 60)
        print(f"#  Running YAML case: {entry.case_id}")
        print(f"#  YAML:   {entry.yaml_basename}")
        print(f"#  Chrono: {entry.chrono_dir}")
        print("#" * 60)
        try:
            result = compare_against_chrono(entry)
        except Exception as exc:  # surface failures without aborting the run
            print(f"[FAIL] {entry.case_id}: {exc}")
            traceback.print_exc()
            summary.append((entry.case_id, float("nan"), float("nan"), "FAIL"))
            continue

        pos_nrmse = result["position_metrics"]["NRMSE_per_body"]
        vel_nrmse = result["velocity_metrics"]["NRMSE_per_body"]
        bodies = result["bodies_compared"]

        print(f"Bodies compared: {list(bodies)}")
        print("Position NRMSE per body:")
        for b, val in zip(bodies, pos_nrmse):
            print(f"  body {b}: {val:.3e}")
        print("Velocity NRMSE per body:")
        for b, val in zip(bodies, vel_nrmse):
            print(f"  body {b}: {val:.3e}")

        summary.append(
            (
                entry.case_id,
                float(np.max(pos_nrmse)),
                float(np.max(vel_nrmse)),
                "OK",
            )
        )

    print("\n" + "=" * 60)
    print("Summary (max NRMSE across compared bodies)")
    print("=" * 60)
    for row in summary:
        print(_format_row(*row))
    print()

    return 0 if all(status == "OK" for *_, status in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

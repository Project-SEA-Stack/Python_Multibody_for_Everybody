"""
generate_moordyn_file.py
----------------------------------------------------------------
General-purpose writer for MoorDyn *.txt* input files.

Sections handled
----------------
LINE TYPES
BODIES              
POINT PROPERTIES
LINES
OPTIONS
OUTPUTS             (free-form list)

The file headers themselves are **immutable** to avoid formatting drift.
"""

from pathlib import Path
from typing import Iterable, Mapping, Sequence

# For more information on MoorDyn format, see https://moordyn.readthedocs.io/

# -------------------------------------------------------------------------
# Template header (do **not** edit unless MoorDyn format itself changes)
# -------------------------------------------------------------------------
_HEADER = """\
--------------------- MoorDyn Input File ------------------------------------
MoorDyn input file automatically generated
"""

_LINE_TYPES_HDR = """\
----------------------- LINE TYPES ------------------------------------------
TypeName   Diam    Mass/m     EA         BA/-zeta    EI         Cd     Ca     CdAx    CaAx
(name)     (m)     (kg/m)     (N)        (N-s/-)     (N-m^2)    (-)    (-)    (-)     (-)
"""

# _BODIES_HDR = """\
# ----------------------- BODIES ------------------------------------------------------
# ID   Attachment  X0   Y0   Z0   r0    p0    y0   Mass  CG*   I*     Volume  CdA*   Ca*
# (#)     (-)      (m)  (m)  (m) (deg) (deg) (deg) (kg)  (m) (kg-m^2) (m^3)   (m^2)  (-)
# """

_POINTS_HDR = """\
---------------------- POINT PROPERTIES --------------------------------
ID    Type      X       Y       Z       Mass   Volume  CdA    Ca
(#)   (-)       (m)     (m)     (m)     (kg)   (m^3)   (m^2)  (-)
"""

_LINES_HDR = """\
---------------------- LINES ----------------------------------------
ID   LineType   AttachA  AttachB  UnstrLen  NumSegs  LineOutputs
(#)   (name)     (#)      (#)       (m)       (-)     (-)
"""

_OPTIONS_HDR = "---------------------- OPTIONS -----------------------------------------\n"
_OUTPUTS_HDR = "---------------------- OUTPUTS -----------------------------------------\n"
_NEED_THIS_LINE = "------------------------- need this line -------------------------------------- \n"


# -------------------------------------------------------------------------
# Section-specific format strings (edit widths only if really needed)
# -------------------------------------------------------------------------
_LINE_TYPE_FMT = "{TypeName:<10} {Diam:<7.3f} {MassPerM:<9.4f} {EA:<11} {BA:<11} {EI:<11} " \
                 "{Cd:<6} {Ca:<6} {CdAx:<7} {CaAx:<6}\n"

# _BODY_FMT = "{ID:<4} {Attachment:<10} {X0:<5} {Y0:<5} {Z0:<6} {r0:<5} {p0:<5} {y0:<5} " \
#             "{Mass:<6} {CG:<4} {I:<8} {Volume:<7} {CdA:<6} {Ca:<4}\n"

_POINT_FMT = "{ID:<5} {Type:<10} {X:<8.2f} {Y:<8.2f} {Z:<8.2f} {Mass:<6} {Volume:<7} {CdA:<6} {Ca:<4}\n"

_LINE_FMT = "{ID:<5} {LineType:<10} {AttachA:<8} {AttachB:<8} {UnstrLen:<9.1f} {NumSegs:<8} {LineOutputs}\n"

_OPTION_FMT = "{value:<13} {name}\n"


# -------------------------------------------------------------------------
# Helper to coerce both dict-of-dicts **and** list-of-dicts to iterable
# -------------------------------------------------------------------------
def _iter_records(section: Sequence | Mapping):
    """Yield record dicts in ID order."""
    if isinstance(section, Mapping):
        for k in sorted(section):
            rec = section[k]
            rec.setdefault("ID", k)
            yield rec
    else:  # assume list/tuple of dict
        yield from sorted(section, key=lambda d: d.get("ID", 0))


# -------------------------------------------------------------------------
# Main generation routine
# -------------------------------------------------------------------------
def write_moordyn_file(
    out_path: str | Path,
    *,
    line_types: Sequence[dict] | Mapping[int, dict],
    points: Sequence[dict] | Mapping[int, dict],
    lines: Sequence[dict] | Mapping[int, dict],
    # bodies: Sequence[dict] | Mapping[int, dict] | None = None,
    options: Mapping[str, float | int | str] | None = None,
    outputs: Iterable[str] | None = None,
):
    """
    Parameters
    ----------
    out_path : str | Path
        Destination *.txt*.
    line_types / points / lines 
        Either list-of-dicts **or** dict keyed by ID → record-dict.
        Missing numeric fields default to zero.
    options :
        Mapping {name: value}.  Preserves insertion order (Python ≥3.7).
    outputs :
        Iterable of output channel names (e.g. ["FairTen1", "FairTen2"]).
    """

    out_path = Path(out_path)

    # ------------------ write file ------------------
    with out_path.open("w", encoding="utf-8") as f:
        # Headers
        f.write(_HEADER)

        # 1) --- LINE TYPES ---
        f.write(_LINE_TYPES_HDR)
        for rec in _iter_records(line_types):
            defaults = dict(Diam=0, MassPerM=0, EA=0, BA=0, EI=0, Cd=0,
                            Ca=0, CdAx=0, CaAx=0) # Cl   dF   cF   are other optional fields here for modeling VIV. 
            defaults.update(rec)
            f.write(_LINE_TYPE_FMT.format(**defaults))

        #         # 2) --- BODIES ---
        # f.write(_BODIES_HDR)
        # # Provide defaults then override with user-specified values
        # if bodies:
        #     for rec in _iter_records(bodies):
        #         # default stub for each body
        #         defaults = dict(
        #             ID=rec.get("ID", 1),
        #             Attachment="Coupled", X0=0, Y0=0, Z0=0,
        #             r0=0, p0=0, y0=0,
        #             Mass=0, CG=0, I=0,
        #             Volume=0, CdA=0, Ca=0
        #         )
        #         # override defaults with provided values
        #         defaults.update(rec)
        #         f.write(_BODY_FMT.format(**defaults))
        # else:
        #     # single stub body if none provided
        #     stub = dict(
        #         ID=1,
        #         Attachment="Coupled", X0=0, Y0=0, Z0=0,
        #         r0=0, p0=0, y0=0,
        #         Mass=0, CG=0, I=0,
        #         Volume=0, CdA=0, Ca=0
        #     )
        #     f.write(_BODY_FMT.format(**stub))

        # 3) --- POINT PROPERTIES ---
        f.write(_POINTS_HDR)
        for rec in _iter_records(points):
            d = dict(Mass=0, Volume=0, CdA=0, Ca=0, **rec)
            f.write(_POINT_FMT.format(**d))

        # 4) --- LINES ---
        f.write(_LINES_HDR)
        for rec in _iter_records(lines):
            defaults = dict(UnstrLen=0.0, NumSegs=0, LineOutputs="")
            defaults.update(rec)
            f.write(_LINE_FMT.format(**defaults))

        # 5) --- OPTIONS ---
        f.write(_OPTIONS_HDR)
        if options:
            for name, value in options.items():
                f.write(_OPTION_FMT.format(value=value, name=name))

        # 6) --- OUTPUTS ---
        f.write(_OUTPUTS_HDR)
        for ch in (outputs or []):
            f.write(f"{ch}\n")

        # 7) --- terminal line (required) ---
        f.write(_NEED_THIS_LINE)

    print(f"[MoorDyn] File written → {out_path.resolve()}")
    


# source/multibody/flexible/flex_expand.py
"""
@author: Sahand Sabet

FSM-in-core preprocessor: expands a two-table (joints + flex properties)
model description into the flat joints/types/offset lists MbdSystem expects.

Table 1 -- the SAME parallel lists as any rigid-case script, plus one more:
    joints                list[[parent, child]]      # unchanged
    types                 list[str]                  # 'R'|'P'|'F', unchanged
    parent_cg_to_joint    list[[x, z]]                # unchanged
    joint_to_child_cg     list[[x, z]]                # unchanged
    prismatic_direction   list[[x, z] or nan-pair]     # unchanged
    flex_bd               list[int or nan/None]       # NEW: nan/None=rigid (default), otherwise MUST
                                                       # equal this row's own (1-based) body index --
                                                       # e.g. row 4 is flexible -> flex_bd[3] = 4.
    m0                    list[float or None/nan]     # unchanged container; ignored for flex rows
    J0                    list[float or None/nan]     # unchanged container; ignored for flex rows

Both offset fields describe THIS row's own incoming joint (matching M4E's
core convention exactly: parent_cg_to_joint = parent's own CG -> this joint;
joint_to_child_cg = this joint -> THIS row's own CG) -- confirmed directly
against `_joints_helpers.py` (`rev_joint`/`pris_joint`). Neither field is ever
optional; a free-tip flex row still provides both, same as any row.

IMPORTANT for a flex row: "THIS row's own CG" means the CG of the WHOLE
flexible member (this row stands in for the entire member, not any one of
its internal segments) -- for a uniform straight beam of length L mounted
at one end, that's L/2, NOT dx/2 (one segment's own half-length). This
value is only ever consumed as an opaque distance in the has-a-child
length-derivation sum (see _derive_or_validate_length) -- it is never used
to place any individual real segment (each segment's own true dx/2 offset
is computed independently in _expand_bending_row from the derived/given L).
For a free-tip row it is not read at all (L comes from flex_L directly),
but should still be set to L/2 by convention for a physically honest table.

Table 2 -- parallel lists, indexed by POSITION (flex_bds gives each
position's real body index):
    flex_bds               list[int]                  # NEW: the actual (1-based) body index of
                                                       # each Table 2 row, e.g. [2, 4] -- no
                                                       # relation to row order required, but every
                                                       # value here must match a flex_bd in Table 1
                                                       # (Table 1 and Table 2 are cross-validated).
    flex_n_seg             list[int]
    flex_L                 list[float or None]        # REQUIRED (non-None) only for a free-tip member
    flex_E, flex_A, flex_I, flex_rho   list[scalar or list[n_seg]]
    flex_deformation_mode  list[str]                  # "bending" (only mode supported so far)
    flex_clamp_on          list[bool]
    flex_m, flex_J         list[scalar or list[n_seg] or None]   # optional mass/inertia override
"""
import math
import warnings
from ..multibody_core._springs_dampers_helpers import parse_point_str

FORCE_ENTRY_TYPES = (
    "PointsBD", "CG", "TensionSpring", "TensionDamper",
    "TorsionSpring", "TorsionDamper",
)


def _is_unset(value):
    """True for None or NaN -- the two ways a placeholder slot may be spelled."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _translate_flex_bd(flex_bd, flex_bds, n_bodies):
    """
    Translate the user-facing flex_bd/flex_bds schema (Table 1: nan/None for
    rigid, this row's own 1-based body index for flexible; Table 2: flex_bds
    lists those same body indices, one per Table 2 row) into the internal
    flex_id/flex_ids representation the rest of this module already uses
    (Table 1: 0 for rigid, sequential k=1,2,... for flexible, in flex_bds'
    order; Table 2: flex_ids=[1,2,...,n_flex]) -- a pure translation +
    validation layer, so the actual expansion logic (index map, root-joint
    resolution, bending/axial expansion) never needs to change.
    """
    flexible_rows = set()
    for i, val in enumerate(flex_bd, start=1):
        if _is_unset(val):
            continue
        if val != i:
            raise ValueError(
                f"row {i}: flex_bd={val!r} must equal this row's own body "
                f"index ({i}), or be nan/None if rigid."
            )
        flexible_rows.add(i)

    flex_id = [0] * n_bodies
    seen = set()
    for k, body in enumerate(flex_bds, start=1):
        if not (1 <= body <= n_bodies):
            raise ValueError(f"flex_bds[{k - 1}]={body!r} is not a valid body index (1..{n_bodies}).")
        if body in seen:
            raise ValueError(f"flex_bds has body {body} listed more than once.")
        seen.add(body)
        if body not in flexible_rows:
            raise ValueError(
                f"flex_bds lists body {body} as flexible, but Table 1's "
                f"flex_bd[{body - 1}] is nan (rigid) -- Table 1 and Table 2 disagree."
            )
        flex_id[body - 1] = k

    missing = flexible_rows - seen
    if missing:
        raise ValueError(
            f"row(s) {sorted(missing)}: flex_bd marks them flexible, but "
            f"they never appear in Table 2's flex_bds -- add them there."
        )

    flex_ids = list(range(1, len(flex_bds) + 1))
    return flex_id, flex_ids


def _build_index_map(joints, types, flex_id, flex_n_seg, flex_deformation_mode):
    """
    Single pass over Table 1: assign real (expanded) body indices to each
    logical row. A rigid row consumes 1 real index. A flex row consumes
    flex_n_seg[flex_id-1] real indices for "bending" (the segments
    themselves), or flex_n_seg[flex_id-1]+1 for "axial" (the segments PLUS
    one phantom tip-anchor body -- a free axial tip needs an explicit body
    to carry the last half-segment's own stretch, unlike a free bending
    tip, which needs nothing extra there). Works identically regardless of
    how rigid/flex rows are interleaved -- purely sequential bookkeeping.

    Also tracks each real body's own DOF count and its starting offset in
    the flat ic/state array (position, then velocity, halves). Every
    'R'/'P' body (including every flex-generated segment -- a flex row can
    never itself be 'F', enforced in _resolve_root_joint_spec) has exactly
    1 DOF, so its offset is simply real-1, same as always. A RIGID row
    declared 'F' (floating) has 3 DOF (X, Z, Theta) instead -- every real
    body after it is shifted accordingly, which is why a flat real-1
    lookup silently breaks the moment any 'F' row exists anywhere earlier
    in the table.

    Returns
    -------
    index_map : dict[int, dict], keyed by 1-based LOGICAL row index:
        {"kind": "rigid"|"flex", "flex_id": int, "mode": "bending"|"axial"|None,
         "real": int (rigid) or list[int] (flex),
         "root": int, "tip": int}
        For a rigid row, root == tip == real. For a flex row, "tip" is the
        LAST reserved real index -- the last segment for bending, or the
        phantom tip anchor for axial (the only sensible continuation point
        for a chain either way).
    n_real_bodies : int, total real bodies across the whole table.
    dof_start : dict[int, int], keyed by REAL body number -> the index
        where that body's own generalized coordinate(s) begin in the flat
        ic/state array (position half; add n_total_dof for the matching
        velocity entry).
    n_total_dof : int, total DOF across the whole table (== n_real_bodies
        unless at least one rigid row is 'F').
    """
    index_map = {}
    dof_start = {}
    next_real = 1
    next_dof = 0
    for logical_idx in range(1, len(joints) + 1):
        fid = flex_id[logical_idx - 1]
        if fid == 0:
            real = next_real
            dof_count = 3 if types[logical_idx - 1] == "F" else 1
            index_map[logical_idx] = {"kind": "rigid", "flex_id": 0, "mode": None,
                                       "real": real, "root": real, "tip": real}
            dof_start[real] = next_dof
            next_real += 1
            next_dof += dof_count
        else:
            if fid < 1 or fid > len(flex_n_seg):
                raise ValueError(
                    f"row {logical_idx} declares flex_id={fid}, but flex_n_seg "
                    f"only has {len(flex_n_seg)} entries (Table 2)."
                )
            n_seg = flex_n_seg[fid - 1]
            if not isinstance(n_seg, int) or n_seg < 1:
                raise ValueError(f"flex_id={fid}: flex_n_seg[{fid - 1}]={n_seg!r} must be a positive integer.")
            mode = flex_deformation_mode[fid - 1]
            if mode == "bending":
                n_reserve = n_seg
            elif mode == "axial":
                n_reserve = n_seg + 1  # + phantom tip anchor
            else:
                raise NotImplementedError(
                    f"flex_id={fid}: deformation_mode={mode!r} not supported "
                    f"yet (must be 'bending' or 'axial')."
                )
            reals = list(range(next_real, next_real + n_reserve))
            index_map[logical_idx] = {"kind": "flex", "flex_id": fid, "mode": mode,
                                       "real": reals, "root": reals[0], "tip": reals[-1]}
            for r in reals:
                dof_start[r] = next_dof
                next_dof += 1  # every flex-generated segment is always 1 DOF ('R' or 'P', never 'F')
            next_real += n_reserve
    return index_map, next_real - 1, dof_start, next_dof


def _resolve_parent_ref(logical_parent_idx, index_map):
    """
    Translate a logical parent-row reference into the real body index a new
    joint should actually attach to: a rigid parent resolves to its single
    real index; a flex parent resolves to its TIP (the only sensible
    continuation point for a chain -- needs no user-facing syntax, purely
    internal bookkeeping). logical_parent_idx == 0 (ground) passes through
    unchanged.
    """
    if logical_parent_idx == 0:
        return 0
    if logical_parent_idx not in index_map:
        raise ValueError(f"parent index {logical_parent_idx} does not exist in Table 1.")
    return index_map[logical_parent_idx]["tip"]


def _mounting_offset(row_parent, index_map, declared_offset):
    """
    Offset from the parent's own CG to THIS row's mounting joint.

    If the parent is a flex row, its Table 1 offset (declared_offset) is
    the HALF-MEMBER distance (L_parent/2) -- a bookkeeping value for the
    whole flexible member, not the real body this joint actually attaches
    to. For a BENDING parent, the real (last-segment) body is only
    dx_parent long, and it points along the parent row's own declared axis
    (not necessarily local +x), so the correct offset is dx_parent/2
    (stashed as "dx_vec" on index_map once that row was expanded). For an
    AXIAL parent, the child mounts directly on the phantom tip anchor,
    which already sits exactly at the true physical end -- zero further
    offset needed. Ground/rigid parents have no such row to derive
    anything from, so their declared Table 1 value is used directly,
    unchanged.
    """
    if row_parent != 0 and index_map[row_parent]["kind"] == "flex":
        if index_map[row_parent]["mode"] == "axial":
            return [0.0, 0.0]
        dx_vec = index_map[row_parent]["dx_vec"]
        return [dx_vec[0] / 2.0, dx_vec[1] / 2.0]
    return list(declared_offset)


def _classify_child_topology(joints):
    """
    For every logical row, determine whether some OTHER row lists it as
    parent (has a real child) or not (free tip). Checked per row, not by
    table position -- a branching tree can have several independent
    terminal flex rows at once.

    Returns
    -------
    has_child : dict[int, bool] keyed by 1-based logical row index.
    """
    has_child = {i: False for i in range(1, len(joints) + 1)}
    for logical_idx, (parent, _child) in enumerate(joints, start=1):
        if parent != 0:
            if parent not in has_child:
                raise ValueError(f"row {logical_idx} references parent={parent}, which does not exist.")
            has_child[parent] = True
    return has_child


def _point_str_body(pt_str):
    """Body index referenced by a point-string, or None for a ground (GR) point."""
    point_type, body, _cell, *_ = parse_point_str(pt_str)
    if point_type == "GR":
        return None
    return body


def _validate_no_flex_forces(Force, index_map):
    """
    Reject any Force-dict entry (any of the 6 types) that references a
    flex-id row, by any means. Flexible-body forces are added directly in
    the generated file, never in the pre-expansion input -- only the
    member's own internal FSM springs are auto-generated for a flex row.
    """
    def _is_flex(logical_body):
        info = index_map.get(logical_body)
        return info is not None and info["kind"] == "flex"

    def _fail(entry_type, logical_body):
        info = index_map[logical_body]
        raise ValueError(
            f"Force['{entry_type}'] references body {logical_body}, which is "
            f"flex_id={info['flex_id']} -- forces on flexible bodies must be "
            f"added directly in the generated file, not in the input Force dict."
        )

    for entry in Force.get("PointsBD", []):
        body = entry[0]
        if _is_flex(body):
            _fail("PointsBD", body)

    for entry in Force.get("CG", []):
        body = entry[0]
        if _is_flex(body):
            _fail("CG", body)

    for entry_type in ("TensionSpring", "TensionDamper"):
        for pair, _params in Force.get(entry_type, []):
            for pt_str in pair:
                body = _point_str_body(pt_str)
                if body is not None and _is_flex(body):
                    _fail(entry_type, body)

    for entry_type in ("TorsionSpring", "TorsionDamper"):
        for bodies, _params in Force.get(entry_type, []):
            for body in bodies:
                if _is_flex(body):
                    _fail(entry_type, body)


def _vec_norm(v):
    return math.hypot(v[0], v[1])


def _is_parallel(v1, v2, tol=1e-6):
    """
    True if v1, v2 point in the SAME direction (parallel, not anti-parallel)
    within tolerance -- cross product near zero => same line, dot product
    > 0 => same direction (not opposite).
    """
    n1, n2 = _vec_norm(v1), _vec_norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("Cannot check collinearity against a near-zero-length offset vector.")
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return dot > 0 and abs(cross) <= tol * n1 * n2


def _find_children(logical_idx, joints):
    """Logical indices of every row whose parent == logical_idx."""
    return [j for j, (parent, _child) in enumerate(joints, start=1) if parent == logical_idx]


def _check_collinearity_and_length(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, has_child, tol=1e-6):
    """
    For a row that HAS a child (skipped for free-tip rows -- there's no
    child data to check against at all in that case):
      1. EVERY child's parent_cg_to_joint must be parallel to this row's own
         joint_to_child_cg (confirms this row is a straight FSM segment;
         magnitude may legitimately differ per child -- an off-center CG is
         normal and not itself a collinearity violation).
      2. If MORE THAN ONE child exists, all children's IMPLIED LENGTHS
         (own joint_to_child_cg magnitude + that child's own
         parent_cg_to_joint magnitude) must match each other within
         tolerance -- multiple children of one flex row must all attach at
         the SAME shared tip point, not just each individually be collinear.

    Returns the list of per-child derived lengths (all equal, once check 2
    passes).
    """
    if not has_child[logical_idx]:
        return None  # free tip: nothing to check against, see module docs

    own_vec = joint_to_child_cg[logical_idx - 1]
    own_len = _vec_norm(own_vec)

    children = _find_children(logical_idx, joints)
    derived_lengths = []
    for child_idx in children:
        child_vec = parent_cg_to_joint[child_idx - 1]
        if not _is_parallel(own_vec, child_vec, tol):
            raise ValueError(
                f"row {logical_idx}'s child (row {child_idx}) is not collinear "
                f"with row {logical_idx}'s own axis -- parent_cg_to_joint="
                f"{child_vec} is not parallel to joint_to_child_cg={own_vec}. "
                f"FSM requires a straight member with all attachments along "
                f"one line through the body."
            )
        derived_lengths.append(own_len + _vec_norm(child_vec))

    if len(derived_lengths) > 1:
        ref = derived_lengths[0]
        for other, child_idx in zip(derived_lengths[1:], children[1:]):
            if abs(other - ref) > tol * max(ref, other, 1.0):
                raise ValueError(
                    f"row {logical_idx} has multiple children with inconsistent "
                    f"implied lengths ({ref} vs {other}, from child rows "
                    f"{children[0]} and {child_idx}) -- multiple children of a "
                    f"flex row must all attach at the SAME shared tip point."
                )

    return derived_lengths


def _derive_or_validate_length(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg,
                                has_child, flex_id, flex_L, tol=1e-6):
    """
    Derive L for a has-child flex row from
    joint_to_child_cg[row] + parent_cg_to_joint[child] (validated for
    collinearity/multi-child consistency by _check_collinearity_and_length
    first); require an explicit, finite, positive L in Table 2 for a
    free-tip row instead (nothing to derive it from).
    """
    fid = flex_id[logical_idx - 1]
    explicit_L = flex_L[fid - 1]
    if _is_unset(explicit_L):
        explicit_L = None

    if has_child[logical_idx]:
        derived_lengths = _check_collinearity_and_length(
            logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, has_child, tol)
        L_derived = derived_lengths[0]
        if explicit_L is not None and abs(explicit_L - L_derived) > tol * max(L_derived, explicit_L, 1.0):
            warnings.warn(
                f"row {logical_idx} (flex_id={fid}): explicit flex_L[{fid - 1}]="
                f"{explicit_L} disagrees with the geometrically-derived "
                f"L={L_derived}; using the derived value.",
                stacklevel=2,
            )
        return L_derived
    else:
        if explicit_L is None:
            raise ValueError(
                f"row {logical_idx} (flex_id={fid}) has no continuing joint "
                f"(free tip) -- flex_L[{fid - 1}] must be supplied explicitly "
                f"since it can't be derived from a child that doesn't exist."
            )
        if not (math.isfinite(explicit_L) and explicit_L > 0):
            raise ValueError(
                f"row {logical_idx} (flex_id={fid}): flex_L[{fid - 1}] must be "
                f"finite and positive, got {explicit_L!r}."
            )
        return explicit_L


_REQUIRED_ELASTIC_TYPE = {"bending": "R", "axial": "P"}


def _resolve_root_joint_spec(logical_idx, types, prismatic_direction, flex_id,
                              flex_deformation_mode, flex_clamp_on):
    """
    Determines the joint(s) to build for a flex row's ROOT (mounting)
    connection: either ONE merged joint, or the declared mounting joint
    PLUS a separate elastic joint in series (mounting is NEVER replaced or
    ignored). Each deformation_mode requires a specific elastic joint type
    for its own root/internal behavior ('R'+TorsionSpring for bending,
    'P'+TensionSpring for axial) -- if the declared mounting type matches,
    they merge into one joint; otherwise the mounting stays separate.

    Returns a list of 1 or 2 dicts, in parent-to-child order, each:
        {"type": "R"|"P", "role": "merged"|"mount"|"elastic",
         "stiffness": "half_cell"|"free"}
    "half_cell" means this joint gets the real root formula (2EI/dx or
    2EA/dx); "free" means k=0. A 2-entry list means an intermediate
    massless body is needed between the mount and elastic joints (built
    later, in expansion).
    """
    fid = flex_id[logical_idx - 1]
    deformation_mode = flex_deformation_mode[fid - 1]
    clamp_on = flex_clamp_on[fid - 1]
    if deformation_mode not in _REQUIRED_ELASTIC_TYPE:
        raise NotImplementedError(
            f"row {logical_idx}: deformation_mode={deformation_mode!r} "
            f"not supported yet (must be 'bending' or 'axial')."
        )
    required_type = _REQUIRED_ELASTIC_TYPE[deformation_mode]

    mount_type = types[logical_idx - 1]
    if mount_type not in ("R", "P"):
        raise ValueError(
            f"row {logical_idx}: mounting type {mount_type!r} is not supported "
            f"for a flexible row -- must be 'R' or 'P'."
        )

    if mount_type == required_type:
        # Matches this deformation_mode's own required elastic type -> merge into one joint.
        return [{"type": required_type, "role": "merged",
                 "stiffness": "half_cell" if clamp_on else "free"}]

    # Mismatched vs this deformation_mode's required elastic type.
    if clamp_on:
        # Only meaningful when pinned (free) -- clamped+mismatched has no
        # legitimate use here, since it just reconstructs a rigid joint
        # less directly than declaring the matching type. Auto-correct+warn.
        if required_type == "P":
            # Correcting 'R' -> 'P' needs a real sliding axis, which an
            # 'R'-typed row never collects by convention -- silently
            # guessing one would be worse than an explicit error.
            pd = prismatic_direction[logical_idx - 1]
            if pd is None or any(math.isnan(v) for v in pd):
                raise ValueError(
                    f"row {logical_idx}: mounting type 'R' with deformation_mode="
                    f"'axial' and clamp_on=True has no legitimate use as declared -- "
                    f"auto-correcting to 'P' needs a real prismatic_direction, which "
                    f"is missing/NaN. Supply prismatic_direction[{logical_idx - 1}] "
                    f"explicitly (or just declare this row's mounting type as 'P')."
                )
        warnings.warn(
            f"row {logical_idx}: mounting type {mount_type!r} with deformation_mode="
            f"{deformation_mode!r} and clamp_on=True has no legitimate use -- "
            f"auto-correcting mounting type to {required_type!r}.",
            stacklevel=2,
        )
        return [{"type": required_type, "role": "merged", "stiffness": "half_cell"}]

    # Genuinely useful mismatch: a "pin-in-slot" joint (M4E's own term for a
    # sliding mount that ALSO deforms independently at the same point) --
    # the declared mount is kept separate, in series with the member's own
    # real elastic stiffness. The elastic joint ALWAYS uses the real
    # half-cell formula here, regardless of clamp_on (clamp_on only ever
    # applies to the mounting joint once separate). Deferred -- see
    # _expand_bending_row/_expand_axial_row.
    return [
        {"type": mount_type, "role": "mount", "stiffness": "free"},
        {"type": required_type, "role": "elastic", "stiffness": "half_cell"},
    ]


def _broadcast(value, n, name):
    """Scalar-or-length-n-list -> always a length-n list."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Table 2 field {name!r} has length {len(value)}, expected {n} (n_seg).")
        return list(value)
    return [value] * n


def _half_cell_compliance(E, I, dx):
    """Compliance (1/k) of ONE half-cell: dx / (2*E*I)."""
    return dx / (2.0 * E * I)


def _joint_stiffness(compliances):
    """
    Unified series-compliance rule: k = 1 / sum(compliances). A rigid side
    contributes compliance 0.0 -- this alone reproduces the exact half-cell
    root/tip formula automatically, with no separate root/internal/tip
    branching logic needed.
    """
    total_compliance = sum(compliances)
    if total_compliance <= 0:
        raise ValueError("a joint needs at least one side with nonzero compliance.")
    return 1.0 / total_compliance


def _expand_bending_row(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, index_map, flex_id,
                         flex_n_seg, flex_E, flex_A, flex_I, flex_rho, flex_m, flex_J,
                         L, root_spec, ic_lookup=None):
    """
    Build the n_seg segment bodies + internal 'R'+TorsionSpring joints for
    ONE bending-mode (deformation_mode="bending") flex row.

    MERGED root case only for now (root_spec of length 1). The "pin-in-slot"
    case (root_spec of length 2 -- a sliding mount that ALSO bends
    independently, M4E's own term for this) needs an extra placeholder body
    between the sliding and bending joints that the index map doesn't
    reserve yet. Deferred by user decision (2026-09-17) -- raises
    NotImplementedError rather than silently building wrong body numbering.

    ic_lookup : optional callable(real_body_index) -> initial angle, used to
    set the root TorsionSpring's theta_0 to the member's actual baseline
    orientation. Defaults to 0.0 if not supplied.

    Returns
    -------
    dict with keys: joints, types, parent_cg_to_joint, joint_to_child_cg,
    prismatic_direction (all length n_seg, in root-to-tip order), torsion_springs
    (list of ([parent_real, child_real], [theta0, k]) -- omitted entirely for
    any joint whose stiffness is "free", i.e. k=0), m0, J0 (length n_seg).
    """
    if len(root_spec) != 1:
        raise NotImplementedError(
            f"row {logical_idx}: a 'pin-in-slot' root (sliding mount that "
            f"also bends independently) needs an extra placeholder body "
            f"between the two joints that isn't supported yet -- deferred, "
            f"not currently available."
        )
    root = root_spec[0]

    fid = flex_id[logical_idx - 1]
    n_seg = flex_n_seg[fid - 1]
    info = index_map[logical_idx]
    reals = info["real"]
    if len(reals) != n_seg:
        raise ValueError(f"row {logical_idx}: index map reserved {len(reals)} bodies, expected n_seg={n_seg}.")

    E   = _broadcast(flex_E[fid - 1], n_seg, "flex_E")
    I   = _broadcast(flex_I[fid - 1], n_seg, "flex_I")
    A   = _broadcast(flex_A[fid - 1], n_seg, "flex_A")
    rho = _broadcast(flex_rho[fid - 1], n_seg, "flex_rho")
    m_val = flex_m[fid - 1] if flex_m is not None else None
    J_val = flex_J[fid - 1] if flex_J is not None else None
    m_override = _broadcast(m_val, n_seg, "flex_m") if not _is_unset(m_val) else None
    J_override = _broadcast(J_val, n_seg, "flex_J") if not _is_unset(J_val) else None

    dx = L / n_seg
    own_vec = joint_to_child_cg[logical_idx - 1]
    own_len = _vec_norm(own_vec)
    if own_len < 1e-12:
        raise ValueError(
            f"row {logical_idx}: joint_to_child_cg is near-zero length -- "
            f"no direction to build this member's segments along."
        )
    ux, uy = own_vec[0] / own_len, own_vec[1] / own_len
    dx_vec = [dx * ux, dx * uy]
    index_map[logical_idx]["dx_vec"] = dx_vec
    row_parent = joints[logical_idx - 1][0]
    parent_real = _resolve_parent_ref(row_parent, index_map)
    row_parent_cg_to_joint = _mounting_offset(row_parent, index_map, parent_cg_to_joint[logical_idx - 1])

    out_joints, out_types = [], []
    out_parent_cg_to_joint, out_joint_to_child_cg, out_prismatic_direction = [], [], []
    torsion_springs = []
    m0, J0 = [], []

    for seg_i in range(n_seg):
        child_real = reals[seg_i]
        this_parent_real = parent_real if seg_i == 0 else reals[seg_i - 1]
        out_joints.append([this_parent_real, child_real])
        out_types.append("R")
        out_prismatic_direction.append([float("nan"), float("nan")])

        if seg_i == 0:
            # Mounting-side offset: whatever the user declared on this row
            # (describes the geometry of the PARENT/mount, untouched by dx).
            out_parent_cg_to_joint.append(list(row_parent_cg_to_joint))
        else:
            out_parent_cg_to_joint.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])
        out_joint_to_child_cg.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])

        m_i = m_override[seg_i] if m_override is not None else rho[seg_i] * A[seg_i] * dx
        J_i = J_override[seg_i] if J_override is not None else m_i * dx * dx / 12.0
        m0.append(m_i)
        J0.append(J_i)

        if seg_i == 0:
            k = (_joint_stiffness([0.0, _half_cell_compliance(E[0], I[0], dx)])
                 if root["stiffness"] == "half_cell" else 0.0)
            theta0 = ic_lookup(child_real) if (k > 0 and ic_lookup is not None) else 0.0
        else:
            k = _joint_stiffness([
                _half_cell_compliance(E[seg_i - 1], I[seg_i - 1], dx),
                _half_cell_compliance(E[seg_i], I[seg_i], dx),
            ])
            theta0 = 0.0  # internal joints measure a genuine relative angle -> always 0

        if k > 0:
            torsion_springs.append(([this_parent_real, child_real], [theta0, k]))

    return {
        "joints": out_joints, "types": out_types,
        "parent_cg_to_joint": out_parent_cg_to_joint,
        "joint_to_child_cg": out_joint_to_child_cg,
        "prismatic_direction": out_prismatic_direction,
        "torsion_springs": torsion_springs,
        "m0": m0, "J0": J0,
    }


def _expand_axial_row(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, index_map, flex_id,
                       flex_n_seg, flex_E, flex_A, flex_I, flex_rho, flex_m, flex_J,
                       L, root_spec, gr_points):
    """
    Build the n_seg segment bodies PLUS one phantom TIP-ANCHOR body
    (n_seg+1 real bodies total) + internal 'P'+TensionSpring joints, for
    ONE axial (deformation_mode="axial") flex row.

    Unlike bending, a free tip is NOT naturally zero-force for axial
    stretch: the last half-segment genuinely stretches under load, so a
    tip half-cell TensionSpring is ALWAYS added (regardless of clamp_on --
    that only ever controls the ROOT), connecting the last real segment to
    a near-massless phantom anchor body representing the true free end.
    This is why an axial row reserves n_seg+1 real bodies, not n_seg like
    bending (verified against the hand-built cantilever_axial_FSM.py,
    where omitting this tip spring under-predicted total elongation by
    exactly 1/n_seg, e.g. -8.33% at n_seg=6).

    No anchor is needed at the ROOT: a direct 'P' mount to ground/parent
    works correctly (the historical root-anchor workaround in
    cantilever_axial_FSM.py was for a since-fixed framework bug -- a bare
    'P'-chain directly off ground -- confirmed fixed by a direct symbolic
    test, see fsm-axial-flexibility-design.md). The root TensionSpring's
    "parent side" point is the parent's own real CG, except when the
    parent is ground (which has no CG of its own) -- in that case a new
    ground point is appended to gr_points (a small mutable accumulator
    shared across the whole table-expansion pass, extending whatever
    Initial_Points["GR"] the user already supplied) and referenced via a
    GR point-string instead.

    Returns
    -------
    dict with keys: joints, types, parent_cg_to_joint, joint_to_child_cg,
    prismatic_direction (all length n_seg+1), tension_springs (list of
    ((pt1_str, pt2_str), [l0, k])), m0, J0 (length n_seg+1).
    """
    if len(root_spec) != 1:
        raise NotImplementedError(
            f"row {logical_idx}: a 'pin-in-slot' root (sliding mount that "
            f"also stretches independently) needs an extra placeholder "
            f"body between the two joints that isn't supported yet -- "
            f"deferred, not currently available."
        )
    root = root_spec[0]

    fid = flex_id[logical_idx - 1]
    n_seg = flex_n_seg[fid - 1]
    info = index_map[logical_idx]
    reals = info["real"]
    if len(reals) != n_seg + 1:
        raise ValueError(f"row {logical_idx}: index map reserved {len(reals)} bodies, expected n_seg+1={n_seg + 1}.")
    seg_reals, anchor_real = reals[:-1], reals[-1]

    E   = _broadcast(flex_E[fid - 1], n_seg, "flex_E")
    A   = _broadcast(flex_A[fid - 1], n_seg, "flex_A")
    rho = _broadcast(flex_rho[fid - 1], n_seg, "flex_rho")
    m_val = flex_m[fid - 1] if flex_m is not None else None
    J_val = flex_J[fid - 1] if flex_J is not None else None
    m_override = _broadcast(m_val, n_seg, "flex_m") if not _is_unset(m_val) else None
    J_override = _broadcast(J_val, n_seg, "flex_J") if not _is_unset(J_val) else None

    dx = L / n_seg
    own_vec = joint_to_child_cg[logical_idx - 1]
    own_len = _vec_norm(own_vec)
    if own_len < 1e-12:
        raise ValueError(
            f"row {logical_idx}: joint_to_child_cg is near-zero length -- "
            f"no direction to build this member's segments along."
        )
    ux, uy = own_vec[0] / own_len, own_vec[1] / own_len
    dx_vec = [dx * ux, dx * uy]
    index_map[logical_idx]["dx_vec"] = dx_vec
    axis_dir = [ux, uy]

    row_parent = joints[logical_idx - 1][0]
    parent_real = _resolve_parent_ref(row_parent, index_map)
    row_parent_cg_to_joint = _mounting_offset(row_parent, index_map, parent_cg_to_joint[logical_idx - 1])
    root_offset_len = _vec_norm(row_parent_cg_to_joint)

    def cg_point(real):
        return f"CG{real}_{real}"

    out_joints, out_types = [], []
    out_parent_cg_to_joint, out_joint_to_child_cg, out_prismatic_direction = [], [], []
    tension_springs = []
    m0, J0 = [], []

    # --- root joint: ground/parent -> first segment, DIRECT (no anchor) ---
    out_joints.append([parent_real, seg_reals[0]])
    out_types.append("P")
    out_prismatic_direction.append(list(axis_dir))
    out_parent_cg_to_joint.append(list(row_parent_cg_to_joint))
    out_joint_to_child_cg.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])

    m_i = m_override[0] if m_override is not None else rho[0] * A[0] * dx
    J_i = J_override[0] if J_override is not None else m_i * dx * dx / 12.0
    m0.append(m_i)
    J0.append(J_i)

    k_root = (_joint_stiffness([0.0, _half_cell_compliance(E[0], A[0], dx)])
              if root["stiffness"] == "half_cell" else 0.0)
    if k_root > 0:
        l0_root = root_offset_len + dx / 2.0
        if row_parent == 0:
            gr_points.append(list(row_parent_cg_to_joint))
            pt1 = f"GR0_{len(gr_points) - 1}"
        else:
            pt1 = cg_point(parent_real)
        tension_springs.append(((pt1, cg_point(seg_reals[0])), [l0_root, k_root]))

    # --- internal joints: segment i -> segment i+1 ---
    for seg_i in range(1, n_seg):
        this_real, next_real = seg_reals[seg_i - 1], seg_reals[seg_i]
        out_joints.append([this_real, next_real])
        out_types.append("P")
        out_prismatic_direction.append(list(axis_dir))
        out_parent_cg_to_joint.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])
        out_joint_to_child_cg.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])

        m_i = m_override[seg_i] if m_override is not None else rho[seg_i] * A[seg_i] * dx
        J_i = J_override[seg_i] if J_override is not None else m_i * dx * dx / 12.0
        m0.append(m_i)
        J0.append(J_i)

        k_int = _joint_stiffness([
            _half_cell_compliance(E[seg_i - 1], A[seg_i - 1], dx),
            _half_cell_compliance(E[seg_i], A[seg_i], dx),
        ])
        tension_springs.append(((cg_point(this_real), cg_point(next_real)), [dx, k_int]))

    # --- tip joint: last segment -> phantom tip anchor (ALWAYS present,
    # regardless of clamp_on -- a free axial tip is not naturally zero-force) ---
    out_joints.append([seg_reals[-1], anchor_real])
    out_types.append("P")
    out_prismatic_direction.append(list(axis_dir))
    out_parent_cg_to_joint.append([dx_vec[0] / 2.0, dx_vec[1] / 2.0])
    out_joint_to_child_cg.append([0.0, 0.0])
    m0.append(1e-6)
    J0.append(1e-6)

    k_tip = _joint_stiffness([0.0, _half_cell_compliance(E[-1], A[-1], dx)])
    tension_springs.append(((cg_point(seg_reals[-1]), cg_point(anchor_real)), [dx / 2.0, k_tip]))

    return {
        "joints": out_joints, "types": out_types,
        "parent_cg_to_joint": out_parent_cg_to_joint,
        "joint_to_child_cg": out_joint_to_child_cg,
        "prismatic_direction": out_prismatic_direction,
        "tension_springs": tension_springs,
        "m0": m0, "J0": J0,
    }



def _remap_point_str(pt_str, index_map):
    """Translate the body number inside a point string from logical to real."""
    point_type, body, cell, *_ = parse_point_str(pt_str)
    if point_type == "GR":
        return pt_str  # ground points don't reference a real body at all
    new_body = index_map[body]["real"]
    return f"{point_type}{new_body}_{cell}"


def _remap_force_dict(Force, index_map):
    """
    Translate every logical body reference in a Force dict (already
    validated by _validate_no_flex_forces to reference only rigid rows)
    into its real body index. Returns a NEW dict; does not mutate the input.
    """
    remapped = {k: [] for k in FORCE_ENTRY_TYPES}

    for body, ptid, fx, fy, mz in Force.get("PointsBD", []):
        remapped["PointsBD"].append([index_map[body]["real"], ptid, fx, fy, mz])

    for body, fx, fz, my in Force.get("CG", []):
        remapped["CG"].append([index_map[body]["real"], fx, fz, my])

    for entry_type in ("TensionSpring", "TensionDamper"):
        for (pt1, pt2), params in Force.get(entry_type, []):
            remapped[entry_type].append((
                (_remap_point_str(pt1, index_map), _remap_point_str(pt2, index_map)),
                params,
            ))

    for entry_type in ("TorsionSpring", "TorsionDamper"):
        for (b1, b2), params in Force.get(entry_type, []):
            remapped[entry_type].append(([index_map[b1]["real"], index_map[b2]["real"]], params))

    return remapped


def _remap_initial_points(Initial_Points, index_map):
    """
    Translate Initial_Points['BD']'s logical body keys to real indices (GR
    entries are ground points, not body-indexed, so they pass through
    unchanged). Rejects a BD key that belongs to a flex row -- exactly like
    Force entries, points on flexible bodies are added directly in the
    generated file, not the pre-expansion input.
    """
    Initial_Points = Initial_Points or {}
    bd_in = Initial_Points.get("BD", {})
    bd_out = {}
    for body, pts in bd_in.items():
        info = index_map.get(body)
        if info is None:
            raise ValueError(f"Initial_Points['BD'] references body {body}, which does not exist.")
        if info["kind"] == "flex":
            raise ValueError(
                f"Initial_Points['BD'] references body {body}, which is "
                f"flex_id={info['flex_id']} -- points on flexible bodies must be "
                f"added directly in the generated file, not in the input."
            )
        bd_out[info["real"]] = list(pts)
    return {"GR": list(Initial_Points.get("GR", [])), "BD": bd_out}


def expand_flex_table(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction,
                      flex_bd, m0, J0,
                      flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
                      flex_deformation_mode, flex_clamp_on, flex_m=None, flex_J=None,
                      Force=None, Initial_Points=None, ic=None):
    """
    The main orchestrator. Operates on the SAME parallel lists any rigid-case
    script already uses (joints, types, parent_cg_to_joint, joint_to_child_cg,
    prismatic_direction, m0, J0), plus flex_bd and Table 2's own parallel
    lists (flex_bds, flex_n_seg, flex_L, flex_E, flex_A, flex_I, flex_rho,
    flex_deformation_mode, flex_clamp_on, and the optional flex_m/flex_J).

    ic : optional array-like (length 2*n_real_bodies: positions then
    velocities), the SAME initial-condition array eventually written into
    the generated file. If given, a CLAMPED bending row's root
    TorsionSpring reference angle (theta_0) is set to match this row's own
    declared baseline orientation (ic[real_body-1]) instead of always
    defaulting to 0.0 -- otherwise a clamped root at any baseline other
    than horizontal bakes in a spurious constant restoring torque (invisible
    to linearized frequency checks, since centered finite differences cancel
    a constant bias, but wrong for genuine nonlinear/time-domain use). If
    omitted, behavior is unchanged (theta_0=0.0, as before).

    Returns a dict of fully expanded flat lists ready to build an
    ``MbdSystem``: joints, types, parent_cg_to_joint, joint_to_child_cg,
    prismatic_direction, Force, Initial_Points, m0, J0.
    """
    flex_id, flex_ids = _translate_flex_bd(flex_bd, flex_bds, len(joints))

    n_flex = max(flex_id) if flex_id else 0
    if flex_m is None:
        flex_m = [None] * n_flex
    if flex_J is None:
        flex_J = [None] * n_flex

    Force = Force or {}
    index_map, n_real, dof_start, n_total_dof = _build_index_map(joints, types, flex_id, flex_n_seg, flex_deformation_mode)
    if ic is not None and len(ic) != 2 * n_total_dof:
        raise ValueError(
            f"ic has length {len(ic)}, expected 2*n_total_dof={2 * n_total_dof} "
            f"({n_real} real bodies, {n_total_dof} total DOF -- a rigid 'F' "
            f"(floating) row contributes 3 DOF, not 1; every other row is 1 DOF, "
            f"and an axial row reserves n_seg+1 real bodies, one more than n_seg "
            f"for a bending row)."
        )
    ic_lookup = (lambda real: ic[dof_start[real]]) if ic is not None else None
    _validate_no_flex_forces(Force, index_map)
    has_child = _classify_child_topology(joints)
    gr_points = list((Initial_Points or {}).get("GR", []))

    out_joints, out_types = [], []
    out_parent_cg_to_joint, out_joint_to_child_cg, out_prismatic_direction = [], [], []
    torsion_springs, tension_springs = [], []
    out_m0 = [None] * n_real
    out_J0 = [None] * n_real

    for logical_idx in range(1, len(joints) + 1):
        fid = flex_id[logical_idx - 1]
        if fid == 0:
            real = index_map[logical_idx]["real"]
            row_parent = joints[logical_idx - 1][0]
            parent_real = _resolve_parent_ref(row_parent, index_map)
            out_joints.append([parent_real, real])
            out_types.append(types[logical_idx - 1])
            out_parent_cg_to_joint.append(_mounting_offset(row_parent, index_map, parent_cg_to_joint[logical_idx - 1]))
            out_joint_to_child_cg.append(list(joint_to_child_cg[logical_idx - 1]))
            pd = prismatic_direction[logical_idx - 1]
            out_prismatic_direction.append(list(pd) if pd is not None else [float("nan"), float("nan")])
            out_m0[real - 1] = m0[logical_idx - 1]
            out_J0[real - 1] = J0[logical_idx - 1]
        else:
            if not _is_unset(m0[logical_idx - 1]) or not _is_unset(J0[logical_idx - 1]):
                warnings.warn(
                    f"row {logical_idx}: m0/J0 were given, but this row is "
                    f"flex_id={fid} -- these only apply to rigid rows and are "
                    f"IGNORED here. Use flex_m/flex_J (Table 2, index {fid - 1}) "
                    f"instead if you want to override the per-segment mass/inertia.",
                    stacklevel=2,
                )
            deformation_mode = flex_deformation_mode[fid - 1]
            L = _derive_or_validate_length(logical_idx, joints, parent_cg_to_joint,
                                            joint_to_child_cg, has_child, flex_id, flex_L)
            root_spec = _resolve_root_joint_spec(logical_idx, types, prismatic_direction, flex_id,
                                                  flex_deformation_mode, flex_clamp_on)
            if deformation_mode == "bending":
                expanded = _expand_bending_row(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, index_map,
                                                flex_id, flex_n_seg, flex_E, flex_A, flex_I, flex_rho,
                                                flex_m, flex_J, L, root_spec, ic_lookup=ic_lookup)
                torsion_springs.extend(expanded["torsion_springs"])
            elif deformation_mode == "axial":
                expanded = _expand_axial_row(logical_idx, joints, parent_cg_to_joint, joint_to_child_cg, index_map,
                                              flex_id, flex_n_seg, flex_E, flex_A, flex_I, flex_rho,
                                              flex_m, flex_J, L, root_spec, gr_points)
                tension_springs.extend(expanded["tension_springs"])
            else:
                raise NotImplementedError(
                    f"row {logical_idx}: deformation_mode={deformation_mode!r} "
                    f"not supported yet (must be 'bending' or 'axial')."
                )

            out_joints.extend(expanded["joints"])
            out_types.extend(expanded["types"])
            out_parent_cg_to_joint.extend(expanded["parent_cg_to_joint"])
            out_joint_to_child_cg.extend(expanded["joint_to_child_cg"])
            out_prismatic_direction.extend(expanded["prismatic_direction"])

            reals = index_map[logical_idx]["real"]
            for offset, real in enumerate(reals):
                out_m0[real - 1] = expanded["m0"][offset]
                out_J0[real - 1] = expanded["J0"][offset]

    if any(_is_unset(v) for v in out_m0) or any(_is_unset(v) for v in out_J0):
        raise ValueError("internal error: not every real body received a mass/inertia value.")

    final_force = _remap_force_dict(Force, index_map)
    final_force["TorsionSpring"] = list(final_force.get("TorsionSpring", [])) + torsion_springs
    final_force["TensionSpring"] = list(final_force.get("TensionSpring", [])) + tension_springs
    points_for_remap = dict(Initial_Points or {})
    points_for_remap["GR"] = gr_points
    final_points = _remap_initial_points(points_for_remap, index_map)

    return {
        "joints": out_joints, "types": out_types,
        "parent_cg_to_joint": out_parent_cg_to_joint,
        "joint_to_child_cg": out_joint_to_child_cg,
        "prismatic_direction": out_prismatic_direction,
        "Force": final_force,
        "Initial_Points": final_points,
        "m0": out_m0, "J0": out_J0,
        "n_real_bodies": n_real,
        "n_total_dof": n_total_dof,
        "index_map": index_map,
    }






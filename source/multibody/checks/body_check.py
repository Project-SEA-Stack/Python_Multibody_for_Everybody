import numpy as np
import sympy as sym

def bodies(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction):
    """
    Validate a planar multibody joint table for structural and kinematic consistency.

    This function mirrors MATLAB’s `table_check`, performing the following checks:
    
    1. **Ground Connection**  

        - The first joint must connect to ground (parent index 0).
    
    2. **Table Completeness and Correctness**  

        - All input lists/arrays must have the same length.  
        - Child body indices must form a continuous sequence (no gaps).  
        - Each joint’s parent must already appear as a body (no parentless children).
    
    3. **Ground-Related Checks**  
        - Every floating (‘F’) joint must attach directly to ground.  
        - For each joint type:  

            * **F**: `joint_to_child_cg` must be `[nan, nan]`.  
            * **R**: `parent_cg_to_joint` must be fully defined (no `nan`).  
            * **P**: `joint_to_child_cg` must be fully defined (no `nan`).
    
    4. **Prismatic-Direction Consistency**  

        - For prismatic (‘P’) joints: direction vector must be normalized (unit length) and not contain `nan`.  
        - For non-prismatic joints: prismatic direction must be `[nan, nan]`.

    Parameters
    ----------
    joints : array-like of shape (n_joints, 2)
        Parent–child index pairs for each joint (ground = 0).
    types : sequence of str
        Joint type codes for each joint:  
        `'R'` = revolute, `'P'` = prismatic, `'F'` = floating.
    parent_cg_to_joint : sequence of array-like
        Vectors from each parent body’s center of gravity to its joint location.
    joint_to_child_cg : sequence of array-like
        Vectors from each joint to its child body’s center of gravity.
    prismatic_direction : sequence of array-like
        Direction vectors for prismatic joints; other entries should be `[nan, nan]`.

    Returns
    -------
    None
        If all checks pass, prints a success message:  
        `"Table validation successful:  No errors found."`

    Raises
    ------
    ValueError
        If any validation step fails, with a specific error message indicating:

        - First joint is not grounded.
        - Inconsistent input lengths.
        - Missing or non‐continuous body indices.
        - Parentless child definition.
        - Floating joint not grounded.
        - Incorrect `nan` usage in CG vectors for a given joint type.
        - Prismatic direction not normalized or missing when required, or erroneously provided for non-prismatic joints.
    """

    # Convert all inputs to numpy arrays
    joints              = np.array(joints, dtype=int)
    types               = np.array(types, dtype=str)
    parent_cg_to_joint  = [sym.Matrix(v) for v in parent_cg_to_joint]
    joint_to_child_cg   = [sym.Matrix(v) for v in joint_to_child_cg]
    prismatic_direction = [sym.Matrix(v) for v in prismatic_direction]

    num_joints = len(joints)

    # ===========================
    # CHECK 2.1: First defined joint must be connected to ground
    # ===========================
    if joints[0, 0] != 0:
        raise ValueError("Error: First defined joint must have ground (0) as its parent.")

    # ===========================
    # CHECK 1: Table Completeness and Correctness
    # ===========================

    # 1.1: Ensure all rows have the same number of elements
    if not (len(types) == num_joints == len(parent_cg_to_joint) == len(joint_to_child_cg) == len(prismatic_direction)):
        raise ValueError("Error: Inconsistent number of rows across input arrays. Check your inputs length.")

    # 1.2: Ensure no missing body indices
    child_bodies = np.sort(joints[:, 1])
    if np.any(np.diff(child_bodies) > 1):
        raise ValueError("Error: Missing body index detected. Joints should be continuous.")

    # 1.3: Ensure parent is defined before child
    current_parent_list = [0]  # Ground (0) is the only initially defined parent
    for i in range(num_joints):
        parent = joints[i, 0]
        child  = joints[i, 1]
        if parent in current_parent_list:
            current_parent_list.append(child)
        else:
            raise ValueError(f"Error: Body {child} is a parentless child. Ensure proper definition.")

    # ===========================
    # CHECK 2: Ground Checks
    # ===========================

    # 2.2.1: Ensure all floating joints are connected to ground
    for i in range(num_joints):
        if types[i] == 'F' and joints[i, 0] != 0:
            raise ValueError(f"Error: Floating joint ({i+1}) must have ground (0) as its parent.")

    # 2.2.2: Ensure ParentCGtoJoint and JointtoChildCG are correctly assigned
    for i in range(num_joints):
        t = types[i]
        parent_vec = parent_cg_to_joint[i]
        child_vec  = joint_to_child_cg[i]

        if t == 'F' and not child_vec.has(sym.nan):
            raise ValueError(f"Error: Floating joint ({i+1}) must have NaN in joint_to_child_cg.")
        elif t == 'R' and parent_vec.has(sym.nan):
            raise ValueError(f"Error: Revolute joint ({i+1}) must have defined parent_cg_to_joint.")
        elif t == 'P' and child_vec.has(sym.nan):
            raise ValueError(f"Error: Prismatic joint ({i+1}) must have defined joint_to_child_cg.")

    # ===========================
    # CHECK 3: Prismatic Joint Direction
    # ===========================

    for i in range(num_joints):
        t = types[i]
        direction_vec = prismatic_direction[i]

        if t == 'P':
            # Must be normalized and not NaN
            if direction_vec.has(sym.nan):
                raise ValueError(f"Error: Prismatic joint ({i+1}) must have a valid direction vector.")
            norm_val = sym.sqrt(direction_vec.dot(direction_vec))
            if not (abs(norm_val - 1) < 1e-8):
                raise ValueError(f"Error: Prismatic joint ({i+1}) must have a normalized direction vector.")
        else:
            # For non-prismatic, direction should be NaN
            if not direction_vec.has(sym.nan):
                raise ValueError(f"Error: Non-prismatic joint ({i+1}) should have [NaN, NaN] in prismatic_direction.")

    print("Table validation successful:\t No errors found.")

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:22:49 2025

@author: adiazfl
"""

import numpy as np
from multibody.checks import bodies

# 1) Check 2.1: First defined joint must be connected to ground
example_1 = {
    "joints":              [[1, 2],[2, 3]],  # First joint's parent != 0
    "types":               ['R','P'],
    "parent_cg_to_joint":  [[0,0],[1,0]],
    "joint_to_child_cg":   [[0,1],[1,1]],
    "prismatic_direction": [[np.nan,np.nan],[1,0]]
}

# 2) Check 1.1: Inconsistent row counts
example_2 = {
    "joints":              [[0,1],[1,2],[2,3]],
    "types":               ['R','P'],  # Fewer rows than joints
    "parent_cg_to_joint":  [[0,0],[1,0],[0,1]],
    "joint_to_child_cg":   [[0,1],[1,1],[0,1]],
    "prismatic_direction": [[np.nan,np.nan],[1,0],[np.nan,np.nan]]
}

# 3) Check 1.2: Missing body index
example_3 = {
    "joints":              [[0,1],[1,2],[2,4]], # Missing body 3
    "types":               ['R','P','F'],
    "parent_cg_to_joint":  [[0,0],[1,0],[np.nan,np.nan]],
    "joint_to_child_cg":   [[0,1],[1,1],[np.nan,np.nan]],
    "prismatic_direction": [[np.nan,np.nan],[1,0],[np.nan,np.nan]]
}

# 4) Check 1.3: Parentless child
example_4 = {
    "joints":              [[0,1],[2,3],[1,4]], # Body 2 used as parent but never defined
    "types":               ['R','P','R'],
    "parent_cg_to_joint":  [[0,0],[1,0],[0,1]],
    "joint_to_child_cg":   [[0,1],[1,1],[0,1]],
    "prismatic_direction": [[np.nan,np.nan],[1,0],[np.nan,np.nan]]
}

# 5) Check 2.2.1: Floating joint must be connected to ground
example_5 = {
    "joints":              [[0,1],[1,2],[2,3]],  # Joint 3 is floating but parent=2
    "types":               ['R','P','F'],
    "parent_cg_to_joint":  [[0,0],[1,0],[np.nan,np.nan]],
    "joint_to_child_cg":   [[0,1],[1,1],[np.nan,np.nan]],
    "prismatic_direction": [[np.nan,np.nan],[1,0],[np.nan,np.nan]]
}

# 6) Check 2.2.2: ParentCGtoJoint or JointtoChildCG incorrect
example_6 = {
    "joints":              [[0,1],[1,2],[0,3]],
    "types":               ['F','R','P'],
    # Revolute joint (index=1) has NaN in parent vec, Prismatic joint (index=2) has NaN in child vec
    "parent_cg_to_joint":  [[0,0],[np.nan,np.nan],[0,1]],
    "joint_to_child_cg":   [[np.nan,np.nan],[1,1],[np.nan,np.nan]],
    "prismatic_direction": [[np.nan,np.nan],[np.nan,np.nan],[1,0]]
}

# 7) Check 3: Prismatic direction must be normalized
example_7 = {
    "joints":              [[0,1],[1,2],[2,3]],
    "types":               ['R','P','F'],
    "parent_cg_to_joint":  [[0,0],[ 1,0],[np.nan,np.nan]],
    "joint_to_child_cg":   [[0,1],[ 1,1],[np.nan,np.nan]],
    "prismatic_direction": [[np.nan,np.nan],[1,0.5],[np.nan,np.nan]]  # Not normalized
}


def run_test(example_data, example_id):
    """Helper function to run a single test example."""
    try:
        bodies(
            example_data["joints"],
            example_data["types"],
            example_data["parent_cg_to_joint"],
            example_data["joint_to_child_cg"],
            example_data["prismatic_direction"]
        )
        print(f"Example {example_id} passed (NO ERROR).")
    except ValueError as e:
        print(f"Example {example_id} triggered error: {e}")

if __name__ == '__main__':

    print("\n--- Example 1: Check 2.1 ---")
    run_test(example_1, 1)

    print("\n--- Example 2: Check 1.1 ---")
    run_test(example_2, 2)

    print("\n--- Example 3: Check 1.2 ---")
    run_test(example_3, 3)

    print("\n--- Example 4: Check 1.3 ---")
    run_test(example_4, 4)

    print("\n--- Example 5: Check 2.2.1 ---")
    run_test(example_5, 5)

    print("\n--- Example 6: Check 2.2.2 ---")
    run_test(example_6, 6)

    print("\n--- Example 7: Check 3 ---")
    run_test(example_7, 7)

#### Helper class for joints equations ####
import sympy as sym
from sympy import Matrix, zeros
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

##############################################################################
# 1) rot_2D_wave and tilde_2D_wave (your custom rotation/tilde for y-axis)
##############################################################################
def rot_2D_wave(angle):
    """
    Returns a 2x2 rotation matrix for a rotation about the y-axis
    with the sign convention changed (per your specification).
    """
    c = sym.cos(angle)
    s = sym.sin(angle)
    return Matrix([[c, s],
                [-s, c]])

def tilde_2D_wave(vec2):
    """
    Compute the 2D skew‐symmetric (tilde) mapping of a vector.

    This operation corresponds to a 90° rotation in the plane:
    
      [x, z]^T  →  [[ 0,  1],
                    [-1,  0]] @ [x, z]^T  =  [ z, -x ]^T

    Parameters
    ----------
    vec2 : sympy.Matrix or array_like, shape (2,) or (2,1)
        The 2D column vector [x, z]^T to be transformed.

    Returns
    -------
    skewed : sympy.Matrix, shape (2,1)
        The result of applying the tilde operator:
        ``[[0, 1], [-1, 0]] * vec2``.

    Examples
    --------
    >>> from sympy import Matrix
    >>> tilde_2D_wave(Matrix([1, 2]))
    Matrix([[2], [-1]])
    >>> tilde_2D_wave(Matrix([x, z]))
    Matrix([[z], [-x]])
    """
    
    M = Matrix([[0, 1],
                [-1, 0]])
    return M * vec2

##############################################################################
# 2) Standalone tree_finder function
##############################################################################
def tree_finder(tree_struc):
    """
    Python equivalent of your MATLAB tree_finder.
    Given a 2D NumPy array (n x 2) of [parent, child] relationships,
    create a directed graph, highlight branches, and return a list
    of all root-to-leaf paths.
    """
    # Create a directed graph from (parent -> child) edges
    G       = nx.DiGraph()
    edges   = [tuple(row) for row in tree_struc]
    G.add_edges_from(edges)

    # Attempt a layered layout (Graphviz), else fallback
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G)

    graph = plt.figure()
    nx.draw(G, pos, arrows=True,
            node_color='b', edge_color=[0.6, 0.6, 0.2],
            with_labels=True, arrowsize=20)
    plt.title('Graph with Highlighted Branches')

    # Potential root nodes: those with no incoming edges
    allNodes    = list(G.nodes())
    parentNodes = set([child for _, child in edges])
    rootNodes   = [node for node in allNodes if node not in parentNodes]
    if not rootNodes:
        rootNodes = allNodes

    # Leaf nodes: no outgoing edges
    leafNodes   = [node for node in G.nodes() if G.out_degree(node) == 0]

    print('All branches in the graph:')
    branchCounter   = 1
    path_all        = []
    num_colors      = len(leafNodes)*len(rootNodes)
    cmap            = plt.get_cmap('hsv', num_colors)
    colorIdx        = 0

    for rootNode in rootNodes:
        for leaf in leafNodes:
            try:
                path = nx.shortest_path(G, source=rootNode, target=leaf)
            except nx.NetworkXNoPath:
                path = []
            if path:
                print(f'Branch {branchCounter}: {path}')
                path_all.append(path)
                # highlight edges & nodes
                branch_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(G, pos,
                                       edgelist=branch_edges,
                                       edge_color=[cmap(colorIdx)],
                                       width=2.5)
                nx.draw_networkx_nodes(G, pos,
                                       nodelist=path,
                                       node_color=[cmap(colorIdx)])
                branchCounter += 1
                colorIdx      += 1

    plt.gca().set_facecolor([0.12, 0.12, 0.12])
    plt.close()
    return path_all, graph

############################################################################
# 3) Helper methods for VelocityTransformation class
                 
############################################################################
# 3A) GROUND-RELATED METHODS (unchanged from your original code)
############################################################################
def rev_joint_to_ground(i, QD, Body_col, JointtoChildCG, ParentCGtoJoint,
                        Reference_frame_Origin, Child_Joint, Pos, R, RD,
                        JointLoc, block_ind):
    """
    Helper for a revolute joint with parent == ground.
    """
    # Fill JointLoc
    # JointLoc[i, 0] = Reference_frame_Origin[0] + ParentCGtoJoint[i][0]
    # JointLoc[i, 1] = Reference_frame_Origin[1] + ParentCGtoJoint[i][1]
    
    JointLoc[i, :] =  (Reference_frame_Origin + ParentCGtoJoint[i]).T

    # The child's offset in Pos
    S_vec              = rot_2D_wave(Child_Joint) @ JointtoChildCG[i]
    body_offset        = 3*block_ind
    Pos[body_offset]   = JointLoc[i, 0] + S_vec[0]
    Pos[body_offset+1] = JointLoc[i, 1] + S_vec[1]
    Pos[body_offset+2] = Child_Joint

    # Fill in R & RD
    S_tilde = tilde_2D_wave(S_vec)
    col_idx = Body_col[block_ind][-1]

    R[body_offset,   col_idx] = S_tilde[0]
    R[body_offset+1, col_idx] = S_tilde[1]
    R[body_offset+2, col_idx] = 1

    S_tilde2    = tilde_2D_wave(S_tilde)
    body_thetaD = QD[ Body_col[block_ind][-1] ]
    tmp         = S_tilde2 * body_thetaD

    RD[body_offset,   col_idx] = tmp[0]
    RD[body_offset+1, col_idx] = tmp[1]
    RD[body_offset+2, col_idx] = 0
    return Pos, JointLoc, R, RD
    
def pris_joint_to_ground(i, Body_col, JointtoChildCG, ParentCGtoJoint,
                        Reference_frame_Origin, Child_Joint, Pos, R, RD,
                        JointLoc, block_ind, prismatic_direction):
    """
    Helper for a prismatic joint with parent == ground.
    """
    JointLoc[i, 0] = Reference_frame_Origin[0] + ParentCGtoJoint[i][0]
    JointLoc[i, 1] = Reference_frame_Origin[1] + ParentCGtoJoint[i][1]

    body_offset        = 3 * block_ind
    Pos[body_offset]   = JointLoc[i, 0] + JointtoChildCG[i][0] + Child_Joint * prismatic_direction[i][0]
    Pos[body_offset+1] = JointLoc[i, 1] + JointtoChildCG[i][1] + Child_Joint * prismatic_direction[i][1]
    Pos[body_offset+2] = 0  # no rotation

    col_idx                   = Body_col[block_ind][-1]
    R[body_offset,   col_idx] = prismatic_direction[i][0]
    R[body_offset+1, col_idx] = prismatic_direction[i][1]
    R[body_offset+2, col_idx] = 0

    return Pos, JointLoc, R, RD

############################################################################
# 3B) NEW: Non-Ground Bodies
############################################################################
def rev_joint(QD, Body_col, JointtoChildCG, ParentCGtoJoint, Child_Joint, Parent_Joint,
            Pos, R, RD, JointLoc, joint_ind, child_ind, Child_Body_ind, Parent_Body_ind):
    """
    Computes the updated values for Pos, JointLoc, R, and RD for a revolute joint.
    
    Parameters
    ----------
    QD : list of sympy expressions (velocity variables)
    Body_col : list of lists containing DOF column indices (indexed by body number-1)
    JointtoChildCG : list (or array) of vectors for joint-to-child CG
    ParentCGtoJoint : list of vectors for parent CG-to-joint
    Child_Joint : sympy symbol for the child joint variable (e.g. Theta)
    Parent_Joint : sympy symbol for the parent's joint variable
    Pos : sympy Matrix (global position vector)
    R, RD : sympy Matrices (velocity transformation and its derivative)
    JointLoc : sympy Matrix (joint location, row vector per joint)
    joint_ind : integer index for the joint (0-based)
    child_ind : integer body number (1-based)
    Child_Body_ind : list of global indices for the child body ([x_index, z_index, theta_index])
    Parent_Body_ind : list of global indices for the parent body
    
    Returns
    -------
    Updated Pos, JointLoc, R, RD.
    """
    # Compute JointLoc row as parent's (x,z) plus rotated ParentCGtoJoint vector.
    parent_pos      = Matrix([Pos[i, 0] for i in Parent_Body_ind[:-1]])
    rot_parent      = rot_2D_wave(Parent_Joint)
    parent_cg       = Matrix(ParentCGtoJoint[joint_ind])  # column vector
    jointloc_row    = (parent_pos + rot_parent @ parent_cg).T  # row vector

    # Update JointLoc (set row 'joint_ind')
    for idx, val in enumerate(jointloc_row.tolist()[0]):
        JointLoc[joint_ind, idx] = val

    # Retrieve the angular velocity of the child body.
    body_thetaD = QD[ Body_col[child_ind-1][-1] ]
    col_idx     = Body_col[child_ind-1][-1]

    # Compute S vector for the joint and its derivatives.
    S        = rot_2D_wave(Child_Joint) @ Matrix(JointtoChildCG[joint_ind])
    S_tilde  = tilde_2D_wave(S)
    S_tilde2 = tilde_2D_wave(S_tilde)
    tmp      = S_tilde2 * body_thetaD

    for idx in range(len(Child_Body_ind)-1):
        Pos[Child_Body_ind[idx], 0]      = JointLoc[joint_ind, idx] + S[idx] # Update the child's position (x,z) and orientation.
        R[Child_Body_ind[idx], col_idx]  = S_tilde[idx]                      # Fill in R: first two rows get tilde_2D_wave(S), third row gets 1.
        RD[Child_Body_ind[idx], col_idx] = tmp[idx]                          # Fill in RD: use second derivative of S.

    Pos[Child_Body_ind[-1], 0]      = Child_Joint
    R[Child_Body_ind[-1], col_idx]  = 1
    RD[Child_Body_ind[-1], col_idx] = 0

    return Pos, JointLoc, R, RD
    
def pris_joint(Q, QD, Body_col, JointtoChildCG, ParentCGtoJoint, Child_Joint,
            Pos, R, RD, JointLoc, joint_ind, child_ind, Child_Body_ind, Parent_Body_ind,
            prismatic_direction, prism_joint_to_omega_col_ind, prism_parent_body):
    """
    Computes the updated values for Pos, JointLoc, R, and RD for a prismatic joint.
    
    Parameters
    ----------
    Q, QD : lists of sympy symbols (positions and velocities)
    Body_col : list of lists of DOF column indices
    JointtoChildCG, ParentCGtoJoint : lists of vectors
    Child_Joint : sympy symbol (prismatic coordinate)
    Pos, R, RD, JointLoc : sympy Matrices to update
    joint_ind : index of the joint (0-based)
    child_ind : body number (1-based)
    Child_Body_ind, Parent_Body_ind : lists of global indices for the child and parent bodies
    prismatic_direction : list/array of prismatic direction vectors
    prism_joint_to_omega_col_ind, prism_parent_body : arrays/dicts (from prsimmatic_omega_finder)
    
    Returns
    -------
    Updated Pos, JointLoc, R, RD.
    """
    prismatic_dir               = Matrix(prismatic_direction[joint_ind])
    real_parent_omega_column    = prism_joint_to_omega_col_ind[joint_ind]
    parent_body_flag            = prism_parent_body[joint_ind]
    child_body_coordinateD      = QD[ Body_col[child_ind-1][-1] ]

    # No rotational ancestor between this joint and ground (pure 'P' chain) -> theta=0.
    # Mirrors the same check already done in pris_joint_children().
    if parent_body_flag == 0:
        real_parent_theta   = 0
        real_parent_thetaD  = 0
    else:
        real_parent_theta   = Q[real_parent_omega_column]
        real_parent_thetaD  = QD[real_parent_omega_column]

    # Compute JointLoc as parent's (x,z) plus rotated sum of ParentCGtoJoint and (Child_Joint * prismatic_dir)
    parent_pos      = Matrix([Pos[i, 0] for i in Parent_Body_ind[:-1]])
    CG2JointChild   = Matrix(ParentCGtoJoint[joint_ind]) + Child_Joint * prismatic_dir
    JointLoc_row    = (parent_pos + rot_2D_wave(real_parent_theta) @ CG2JointChild).T
    col_idx         = Body_col[child_ind-1][-1]

    for idx, val in enumerate(JointLoc_row.tolist()[0]):
        JointLoc[joint_ind, idx] = val

    # Connecting vectors and their rotations to go from parent to child
    S       = rot_2D_wave(real_parent_theta) @ Matrix(JointtoChildCG[joint_ind])
    U       = rot_2D_wave(real_parent_theta) * prismatic_dir
    U_tilde = tilde_2D_wave(U)

    for idx in range(len(Child_Body_ind)-1):
        Pos[Child_Body_ind[idx], 0] = JointLoc[joint_ind, idx] + S[idx]
    Pos[Child_Body_ind[-1], 0] = real_parent_theta

    R[Child_Body_ind[0], col_idx]   = U[0]
    R[Child_Body_ind[1], col_idx]   = U[1]
    R[Child_Body_ind[-1], col_idx]  = 0

    RD[Child_Body_ind[0], col_idx]  = U_tilde[0] * real_parent_thetaD
    RD[Child_Body_ind[1], col_idx]  = U_tilde[1] * real_parent_thetaD
    RD[Child_Body_ind[-1], col_idx] = 0

    return Pos, JointLoc, R, RD

def pris_joint_children(Q, QD, joint_ind, prismatic_direction, 
                        prism_joint_to_omega_col_ind,prism_parent_body):
    """
    Computes the R_child and RD_child for the children of a prismatic joint.
    
    Parameters
    ----------
    Q, QD : lists of sympy symbols
    joint_ind : joint index (0-based)
    prismatic_direction : list/array of prismatic direction vectors
    prism_joint_to_omega_col_ind : array/dict with angular velocity column indices
    
    Returns
    -------
    R_child, RD_child : sympy Matrices (2 x n)
    """
    prismatic_dir               = Matrix(prismatic_direction[joint_ind])
    real_parent_omega_column    = prism_joint_to_omega_col_ind[joint_ind]
    prism_parent_body           = prism_parent_body[joint_ind]

    # Tracking parent's omega
    if prism_parent_body == 0: #TODO: conflict between body number and body column (solved)
        real_parent_theta   = 0
        real_parent_thetaD  = 0
    else:
        real_parent_theta   = Q[real_parent_omega_column]
        real_parent_thetaD  = QD[real_parent_omega_column]

    U        = rot_2D_wave(real_parent_theta) @ prismatic_dir
    R_child  = U
    RD_child = tilde_2D_wave(U) * real_parent_thetaD

    return R_child, RD_child

def rev_joint_children(Body_col, body_thetaD, JointtoChildCG, ParentCGtoJoint,
                    current_coordinate, Type, connecting_parent_joint, connecting_child_joint,
                    prismatic_direction, child_coordinate, j, current_path, Joints, Q, QD):
    """
    Computes R_child and RD_child for the children of a revolute joint.
    
    If the connecting joint type is 'P' (prismatic), sets a flag RevPrisChain and calls
    find_prismatic_chain.
    
    Parameters
    ----------
    Body_col, JointtoChildCG, ParentCGtoJoint : as before.
    current_coordinate : sympy symbol (current joint coordinate)
    Type : list of joint type strings
    connecting_parent_joint, connecting_child_joint : indices (or values) for the connected joints
    prismatic_direction : list/array
    child_coordinate : sympy symbol for child joint coordinate
    j : integer index into current_path
    current_path : list of body numbers representing a path in the tree
    Joints : array (n x 2) of [parent, child] relationships
    Q, QD : lists of sympy symbols
    
    Returns
    -------
    R_child, RD_child : sympy Matrices; RevPrisChain : flag (1 if prismatic chain encountered, else 0)
    """
    if Type[connecting_child_joint] == 'P':
        RevPrisChain        = 1
        parent_type         = 'R'
        R_child, RD_child   = find_prismatic_chain(Body_col, JointtoChildCG, ParentCGtoJoint,
                                                current_coordinate, Type, prismatic_direction,
                                                j, current_path, Joints, Q, QD, body_thetaD, parent_type)
    else:
        S               = rot_2D_wave(current_coordinate) @ (Matrix(JointtoChildCG[connecting_parent_joint]) + Matrix(ParentCGtoJoint[connecting_child_joint]))
        R_child         = tilde_2D_wave(S)
        RD_child        = tilde_2D_wave(tilde_2D_wave(S)) * body_thetaD
        RevPrisChain    = 0

    return R_child, RD_child, RevPrisChain

def float_joint_children(Body_col, body_thetaD, JointtoChildCG, ParentCGtoJoint,
                        current_coordinate, Type, connecting_parent_joint, connecting_child_joint,
                        prismatic_direction, child_coordinate, j, current_path, Joints, Q, QD):
    """
    Computes R_child and RD_child for the children of a floating joint.
    
    If the connecting joint type is 'P', sets a flag FloatPrisChain and calls
    find_prismatic_chain.
    
    Returns R_child, RD_child, FloatPrisChain.
    """
    if Type[connecting_child_joint] == 'P':
        FloatPrisChain      = 1
        parent_type         = 'F'
        R_child, RD_child   = find_prismatic_chain(Body_col, JointtoChildCG, ParentCGtoJoint,
                                                current_coordinate, Type, prismatic_direction,
                                                j, current_path, Joints, Q, QD, body_thetaD, parent_type)
    else:
        S               = rot_2D_wave(current_coordinate) * Matrix(ParentCGtoJoint[connecting_child_joint])
        R_child         = tilde_2D_wave(S)
        RD_child        = tilde_2D_wave(tilde_2D_wave(S)) * body_thetaD
        FloatPrisChain  = 0

    return R_child, RD_child, FloatPrisChain

def prismatic_omega_finder(Joints, Type, non_zero_elem, Body_col, prism_joint_to_omega_col_ind, R):
    """
    For each prismatic joint (with nonzero parent) finds the column index associated with its
    angular velocity. Also builds mappings for prismatic parent bodies.
    
    Parameters
    ----------
    Joints : NumPy array of shape (n,2) with [parent, child]
    Type : list of joint type strings
    non_zero_elem : indices where parent is not 0
    Body_col : list of lists (indexed by body number-1)
    prism_joint_to_omega_col_ind : list (to be updated)
    R : sympy Matrix (to be updated)
    
    Returns
    -------
    Updated prism_joint_to_omega_col_ind, prism_parent_body, pris_body_to_parent_map, and R.
    """
    n                       = Joints.shape[0]
    prism_parent_body       = np.full((n,), np.nan)
    pris_body_to_parent_map = np.full((n,), np.nan)
    zero_elem               = np.where(Joints[:,0] == 0)[0]

    for i in zero_elem:
        if Type[i] == 'P':
            current_body                    = Joints[i,1]
            prism_joint_to_omega_col_ind[i] = 0
            prism_parent_body[i]            = 0
            # Assuming body numbers are 1-based:
            pris_body_to_parent_map[current_body-1] = 0

    for i in non_zero_elem:
        if Type[i] == 'P':
            prism_father                    = Joints[i,0]
            indices                         = np.where(Joints[:,1] == prism_father)[0]

            if indices.size > 0 and Type[indices[0]] != 'P':
                prism_joint_to_omega_col_ind[i]         = Body_col[prism_father-1][-1]
                prism_parent_body[i]                    = prism_father
                current_body                            = Joints[i,1]
                pris_body_to_parent_map[current_body-1] = prism_parent_body[i]

            else:
                parent_joint_rows = np.where(Joints[:,1] == prism_father)[0]

                if parent_joint_rows.size > 0:
                    real_parent             = Joints[parent_joint_rows[0], 0]
                    real_parent_joint_rows  = np.where(Joints[:,1] == real_parent)[0]

                    while real_parent_joint_rows.size > 0 and Type[real_parent_joint_rows[0]] == 'P':
                        parent_joint_rows   = np.where(Joints[:,1] == real_parent)[0]

                        if parent_joint_rows.size == 0:
                            break

                        real_parent                 = Joints[parent_joint_rows[0], 0]
                        real_parent_joint_rows      = np.where(Joints[:,1] == real_parent)[0]

                    prism_parent_body[i]                    = real_parent
                    current_body                            = Joints[i,1]
                    pris_body_to_parent_map[current_body-1] = prism_parent_body[i]
                    # real_parent==0 means the chain traces back to ground with no
                    # rotational ancestor -- leave the omega column unset (NaN); any
                    # consumer must check prism_parent_body==0 first (theta=0), same
                    # as pris_joint_children() already does, instead of indexing here
                    # (Body_col[-1] would silently wrap to the wrong body).
                    if real_parent != 0:
                        prism_joint_to_omega_col_ind[i] = Body_col[real_parent-1][-1]

    pris_ind = [i for i in range(len(prism_joint_to_omega_col_ind)) if not math.isnan(prism_joint_to_omega_col_ind[i])]

    for i in pris_ind:
        prismatic_body          = Joints[i,1]
        prismatic_real_parent   = np.int64(prism_parent_body[i])
        Child_Body_ind          = [3*(prismatic_body-1), 3*(prismatic_body-1)+1, 3*(prismatic_body-1)+2]

        if prismatic_real_parent != 0:
            col = Body_col[prismatic_real_parent-1][-1]
            R[Child_Body_ind[-1], col] = 1  # RD is already zero

    return prism_joint_to_omega_col_ind, prism_parent_body, pris_body_to_parent_map, R

def find_prismatic_chain(Body_col, JointtoChildCG, ParentCGtoJoint, current_coordinate, Type,
                        prismatic_direction, j, current_path, Joints, Q, QD, body_thetaD, parent_type):
    """
    Traverses through a chain of prismatic joints and computes the corresponding R_child and RD_child.
    
    Parameters
    ----------
    Body_col : list of lists of DOF indices
    JointtoChildCG, ParentCGtoJoint : lists of vectors
    current_coordinate : sympy symbol (current joint coordinate)
    Type : list of joint type strings
    prismatic_direction : list/array of prismatic direction vectors
    j : current index in current_path
    current_path : list of body numbers (1-based) representing the path
    Joints : NumPy array of [parent, child] relationships
    Q, QD : lists of sympy symbols
    body_thetaD : sympy expression (angular velocity of current body)
    parent_type : string ('R' or 'F')
    
    Returns
    -------
    R_child, RD_child : sympy Matrices (2 x n), computed for the prismatic chain.
    """
    # Initialization
    NBodies         = len(JointtoChildCG)
    R_child         = zeros(2, NBodies)
    RD_child        = zeros(2, NBodies)
    S               = Matrix([0, 0])
    current_body    = current_path[j]
    parent_body     = current_path[j-1]
    child_body      = current_path[j+1]
    Joints          = np.array(Joints)

    # Tracking the prismatic joint chain parents and children
    connecting_parent_joint = np.where((Joints[:,0]==parent_body) & (Joints[:,1]==current_body))[0][0]
    connecting_child_joint  = np.where((Joints[:,0]==current_body) & (Joints[:,1]==child_body))[0][0]

    First_iteration = True
    RD_Sdot         = sym.zeros(2, 1)

    while Type[connecting_child_joint] == 'P':
        child_coordinate    = Q[ Body_col[child_body-1][-1] ]
        child_coordinateD   = QD[ Body_col[child_body-1][-1] ]
        prismatic_dir       = Matrix(prismatic_direction[connecting_child_joint])

        if First_iteration:
            if parent_type == 'R':
                S += rot_2D_wave(current_coordinate) @ (Matrix(JointtoChildCG[connecting_parent_joint]) +
                    Matrix(ParentCGtoJoint[connecting_child_joint]) +
                    child_coordinate * prismatic_dir +
                    Matrix(JointtoChildCG[connecting_child_joint]))
                
            elif parent_type == 'F':
                S += rot_2D_wave(current_coordinate) @ (Matrix(ParentCGtoJoint[connecting_child_joint]) +
                    child_coordinate * prismatic_dir +
                    Matrix(JointtoChildCG[connecting_child_joint]))
                
        else:
            if parent_type == 'R':
                S += rot_2D_wave(current_coordinate) @ (Matrix(ParentCGtoJoint[connecting_child_joint]) +
                    child_coordinate * prismatic_dir +
                    Matrix(JointtoChildCG[connecting_child_joint]))
                
            elif parent_type == 'F':
                S += rot_2D_wave(current_coordinate) @ (Matrix(ParentCGtoJoint[connecting_child_joint]) +
                    child_coordinate * prismatic_dir +
                    Matrix(JointtoChildCG[connecting_child_joint]))
                
        R_child[:, child_body-1]    = tilde_2D_wave(S)
        RD_Sdot                     +=  child_coordinateD * tilde_2D_wave(rot_2D_wave(current_coordinate) @ prismatic_dir)
        RD_child[:, child_body-1]   = tilde_2D_wave(tilde_2D_wave(S)) * body_thetaD + RD_Sdot

        First_iteration = False
        j               = j + 1
        if j >= len(current_path)-1:
            break
        else:
            current_body            = current_path[j]
            parent_body             = current_path[j-1]
            child_body              = current_path[j+1]
            connecting_parent_joint = np.where((Joints[:,0]==parent_body) & (Joints[:,1]==current_body))[0][0]
            connecting_child_joint  = np.where((Joints[:,0]==current_body) & (Joints[:,1]==child_body))[0][0]

            if Type[connecting_child_joint] != 'P':
                S = S + rot_2D_wave(current_coordinate) @ Matrix(ParentCGtoJoint[connecting_child_joint])

                while True:
                    R_child[:, child_body-1]    = tilde_2D_wave(S)
                    RD_child[:, child_body-1]   = tilde_2D_wave(tilde_2D_wave(S)) * body_thetaD + RD_Sdot

                    j = j + 1
                    if j < len(current_path)-1:
                        child_body = current_path[j+1]
                    else:
                        break
    return R_child, RD_child

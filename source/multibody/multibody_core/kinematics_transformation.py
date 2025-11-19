import sympy as sym
from sympy import Matrix
import numpy as np
from ._joints_helpers import *



##############################################################################
# 3) VelTransformation Class:
#    - Keeps your original compute_grounded() (so it won't break)
#    - Adds compute_non_grounded() & static methods for the rest.
##############################################################################
class VelocityTransformation:
    """
    Compute kinematic position and velocity transformations for a planar multibody system.

    The VelocityTransformation class takes a populated :class:`JointSystem` and a 2×1
    reference‐frame origin, then builds the global position vector ``Pos``,
    the velocity‐transformation matrix ``R``, its time derivative ``RD``, and
    a tracking matrix ``R_track`` for every body in the mechanism. It provides
    separate methods to handle joints grounded to the base (parent==0) and to
    propagate kinematics through non-grounded bodies.

    Parameters
    ----------
    joint_system : JointSystem
        Instance that supplies joint topology, symbolic coordinates, velocities,
        accelerations, DOF counts, and body-to-global index mappings.
    Reference_frame_Origin : array‐like or sympy.Matrix
        2×1 vector locating the global frame origin. If not already a
        ``sympy.Matrix``, it will be converted internally.

    Attributes
    ----------
    Joints : list[list[int]]
        List of `[parent, child]` index pairs for each joint.
    Type : list[str]
        Joint type codes (``'R'``, ``'P'``, ``'F'``) in the same order as ``Joints``.
    ParentCGtoJoint : list[sympy.Matrix]
        Vectors from each parent’s center of gravity to its joint.
    JointtoChildCG : list[sympy.Matrix]
        Vectors from each joint to its child’s center of gravity.
    prismatic_direction : list[sympy.Matrix]
        Unit-length axis vectors for prismatic joints, or ``[nan, nan]`` otherwise.
    NBodies : int
        Number of moving bodies (ground is index 0, so maximum child index).
    Q, QD, QDD : list[sympy.Symbol]
        Symbolic position, velocity, and acceleration variables extracted from `joint_system`.
    NDoF : list[int]
        Degrees of freedom contributed by each joint (1 for revolute/prismatic, 3 for floating).
    Body_col : list[list[int]]
        Maps each body (by index–1) to its global coordinate indices in ``Q``, ``QD``, ``QDD``.
    Reference_frame_Origin : sympy.Matrix
        2×1 sympy vector representing the global frame origin after conversion.

    Examples
    --------
    >>> from sympy import nan
    >>> from multibody.MultibodyCore.joints_system import JointSystem
    >>> from multibody.MultibodyCore.kinematicsTransformation import VelTransformation
    >>> # assume `jsys` is a prebuilt JointSystem and `origin` a 2×1 vector
    >>> vt = VelTransformation(jsys, origin)
    >>> Pos, JointLoc, R, RD, R_track = vt.compute_grounded()
    >>> Pos, JointLoc, R, RD, R_track = vt.compute_non_grounded(Pos, JointLoc, R, RD, R_track)
    """

    def __init__(self, joint_system, Reference_frame_Origin):
        self.Joints                 = []
        self.Type                   = []
        self.ParentCGtoJoint        = []
        self.JointtoChildCG         = []
        self.prismatic_direction    = []

        for joint in joint_system.joints:
            self.Joints.append([joint.parent, joint.child])
            self.Type.append(joint.joint_type.value)
            self.ParentCGtoJoint.append(joint.parent_cg_to_joint)
            self.JointtoChildCG.append(joint.joint_to_child_cg)

            if joint.prismatic_direction is None:
                self.prismatic_direction.append([sym.nan, sym.nan])
            else:
                self.prismatic_direction.append(joint.prismatic_direction)

        # Number of bodies: max child index (assuming ground=0)
        self.NBodies = max(j[1] for j in self.Joints)

        # Symbolic coordinates from the joint system
        Q, QD, QDD, NDoF, Body_col = joint_system.coordinate_finder()

        self.Q          = Q
        self.QD         = QD
        self.QDD        = QDD
        self.NDoF       = NDoF
        self.Body_col   = Body_col

        # Convert reference origin to sympy Matrix
        if not isinstance(Reference_frame_Origin, Matrix):
            self.Reference_frame_Origin = Matrix(Reference_frame_Origin)
        else:
            self.Reference_frame_Origin = Reference_frame_Origin

    ############################################################################
    # 3A) GROUND-RELATED METHODS (unchanged from your original code)
    ############################################################################
    def compute_grounded(self):
        """
        Original method: returns (``Pos, JointLoc, R, RD, R_track``)
        for joints/bodies grounded to 0.  This remains unchanged,
        so your call to 'compute_grounded()' still works.
        """
        return VelocityTransformation._pos_vel_acc_finder_grounded_joint(
            self.Joints,
            self.Type,
            self.ParentCGtoJoint,
            self.JointtoChildCG,
            self.prismatic_direction,
            self.Body_col,
            self.NDoF,
            self.Reference_frame_Origin,
            self.Q, self.QD, self.QDD,
            self.NBodies
        )

    @staticmethod
    def _pos_vel_acc_finder_grounded_joint(Joints,
                                          Type,
                                          ParentCGtoJoint,
                                          JointtoChildCG,
                                          prismatic_direction,
                                          Body_col,
                                          NDoF,
                                          Reference_frame_Origin,
                                          Q, QD, QDD,
                                          NBodies):
        """
        Your original ground-handling code. 
        (No changes here to avoid breaking existing calls.)
        Returns: (Pos, JointLoc, R, RD, R_track)
        """

        # Initialization
        NGlobalCoordinates  = NBodies * 3
        Pos                 = sym.zeros(NGlobalCoordinates, 1)
        JointLoc            = sym.zeros(len(Joints), 2)
        total_dofs          = sum(NDoF)
        R                   = sym.zeros(NGlobalCoordinates, total_dofs)
        RD                  = sym.zeros(NGlobalCoordinates, total_dofs)
        R_track             = sym.zeros(NBodies, NBodies)

        # Go through each joint that has parent==0 (ground)...
        for i, (parent_ind, child_ind) in enumerate(Joints):
            if parent_ind == 0:
                # Mark that body as “tracked”
                R_track[child_ind-1, child_ind-1]   = 1
                joint_type                          = Type[i]

                if joint_type == 'R':
                    # Use your existing _rev_joint_to_ground helper
                    Child_Joint = Q[Body_col[child_ind-1][-1]]

                    Pos, JointLoc, R, RD = rev_joint_to_ground(
                        i, QD, Body_col, JointtoChildCG, ParentCGtoJoint,
                        Reference_frame_Origin, Child_Joint, Pos, R, RD,
                        JointLoc, (child_ind - 1)
                    )
                elif joint_type == 'P':
                    # Use your existing _PrisJointToGround helper
                    Child_Joint = Q[Body_col[child_ind-1][-1]] #TODO: rename to child coordinate

                    Pos, JointLoc, R, RD = pris_joint_to_ground(
                        i, Body_col, JointtoChildCG, ParentCGtoJoint,
                        Reference_frame_Origin, Child_Joint, Pos, R, RD,
                        JointLoc, (child_ind - 1), prismatic_direction
                    )
                elif joint_type == 'F':
                    # Floating joint to ground
                    dof_indices = Body_col[child_ind-1]
                    X_          = Q[dof_indices[0]]
                    Z_          = Q[dof_indices[1]]
                    Theta_      = Q[dof_indices[2]]

                    body_offset        = 3*(child_ind-1)
                    Pos[body_offset]   = Reference_frame_Origin[0] + X_
                    Pos[body_offset+1] = Reference_frame_Origin[1] + Z_
                    Pos[body_offset+2] = Theta_

                    JointLoc[i, 0]     = sym.nan
                    JointLoc[i, 1]     = sym.nan

                    for j, col in enumerate(dof_indices):
                        R[body_offset+j, col] = 1
                    # RD remains zero
                else:
                    print("Joint type not defined or recognized.")

        return Pos, JointLoc, R, RD, R_track

    ############################################################################
    # 3B) NEW: Non-Ground Bodies
    ############################################################################
    def compute_non_grounded(self, Pos, JointLoc, R, RD, R_track):
        """
        After calling compute_grounded() to get initial
        (``Pos, JointLoc, R, RD, R_track``), call this method to
        fill in the rest for bodies that have a non-zero parent.
        
        Returns updated (``Pos, JointLoc, R, RD, R_track``).
        """
        return VelocityTransformation._pos_vel_acc_finder(
            self.Joints,
            self.Type,
            self.ParentCGtoJoint,
            self.JointtoChildCG,
            self.prismatic_direction,
            self.Body_col,
            self.NDoF,
            self.Reference_frame_Origin,
            self.Q, self.QD, self.QDD,
            self.NBodies,
            Pos, JointLoc, R, RD, R_track
        )

    @staticmethod
    def _pos_vel_acc_finder(Joints, Type, ParentCGtoJoint, JointtoChildCG, prismatic_direction,
                            Body_col, NDoF, Reference_frame_Origin, Q, QD, QDD, NBodies,
                            Pos, JointLoc, R, RD, R_track):
        """
        Python analogue of your MATLAB pos_vel_acc_finder.m for non‐ground bodies.
        It traverses all paths (from ground to leaves) and, for each joint along each path,
        computes the velocity of the current body’s CG and then propagates the velocity
        transformation (R matrix) to its children, including prismatic chain effects.
        
        Parameters
        ----------
        Joints : list of [parent, child] pairs (1-based body numbers)
        Type : list of joint type strings (e.g., 'R', 'P', 'F')
        ParentCGtoJoint, JointtoChildCG : lists of vectors
        prismatic_direction : list/array of prismatic direction vectors
        Body_col : list of lists of global DOF indices per body (indexed by body number-1)
        NDoF : list of DOF counts per joint
        Reference_frame_Origin : sympy Matrix for the origin
        Q, QD, QDD : lists of sympy symbols (positions, velocities, accelerations)
        NBodies : number of bodies
        Pos, JointLoc, R, RD, R_track : sympy Matrices (global position vector,
            joint locations, velocity transformation matrix, its derivative, and a tracking matrix)
        
        Returns
        -------
        Updated Pos, JointLoc, R, RD, R_track.
        """
        import math  # for math.isnan

        # Step 1. Identify joints whose parent is not ground (i.e. parent != 0)
        array_joints    = np.array(Joints)
        non_zero_elem   = np.where(array_joints[:, 0] != 0)[0]
        
        # Step 2. Build paths using tree_finder on the submatrix of non-ground joints.
        submatrix   = array_joints[non_zero_elem, :]
        paths,graph = tree_finder(submatrix.tolist())

        # Prepend 0 to each path (to indicate the ground)
        for i in range(len(paths)):
            paths[i] = [0] + paths[i]
        
        # Step 3. Initialize prismatic joint omega information.
        prism_joint_to_omega_col_ind = [math.nan] * len(Type)
        (prism_joint_to_omega_col_ind,
        prism_parent_body,
        pris_body_to_parent_map,
        R) = prismatic_omega_finder(array_joints, Type, non_zero_elem,
                                            Body_col, prism_joint_to_omega_col_ind, R)
        
        # Step 4. Traverse each path in paths.
        for path in paths:
            # For each path, traverse j from 1 to len(path)-1 (MATLAB: j=2:length(path))
            # TODO: check what happens with the last body bc range() doesn't evaluate the last number
            for j in range(1, len(path)):
                RevPrisChain    = 0
                FloatPrisChain  = 0
                parent_body     = path[j-1]
                current_body    = path[j]

                # Find joint index that connects parent_body to current_body.
                joint_indices   = [idx for idx, row in enumerate(Joints) 
                                   if row[0] == parent_body and row[1] == current_body]
                
                if len(joint_indices) == 0:
                    continue

                joint_ind   = joint_indices[0]
                joint_type  = Type[joint_ind]

                # Compute global indices in Position vector,for the current body.
                # TODO: I think this variable is redundant as Child_Body_ind does the job
                Current_Body_ind = [3*(current_body-1), 3*(current_body-1)+1, 3*(current_body-1)+2]

                # If the CG velocity for current_body is not yet computed...
                if R_track[current_body-1, current_body-1] != 1:
                    R_track[current_body-1, current_body-1] = 1

                    child_ind       = current_body
                    Child_Body_ind  = Current_Body_ind
                    Parent_Body_ind = [3*(parent_body-1), 3*(parent_body-1)+1, 3*(parent_body-1)+2]

                    if joint_type == 'R':
                        Child_Coordinate = Q[ Body_col[current_body-1][-1] ]
                        # Determine Parent_Coordinate: if the parent's prismatic mapping is nan, then use parent's own coordinate.

                        if math.isnan(pris_body_to_parent_map[parent_body-1]):
                            Parent_Coordinate = Q[Body_col[parent_body-1][-1]]

                        else:
                            if pris_body_to_parent_map[parent_body-1] == 0:
                                Parent_Coordinate = 0
                            else:
                                # Adjust index: pris_body_to_parent_map stores 1-based numbers.
                                real_parent         = int(pris_body_to_parent_map[parent_body-1])
                                Parent_Coordinate   = Q[Body_col[real_parent-1][-1]]

                        (Pos, JointLoc, R, RD) = rev_joint(
                            QD, Body_col, JointtoChildCG, ParentCGtoJoint,
                            Child_Coordinate, Parent_Coordinate,
                            Pos, R, RD, JointLoc,
                            joint_ind, child_ind, Child_Body_ind, Parent_Body_ind
                        )
                    elif joint_type == 'P':
                        Child_Coordinate        = Q[Body_col[current_body-1][-1]]

                        (Pos, JointLoc, R, RD)  = pris_joint(
                            Q, QD, Body_col, JointtoChildCG, ParentCGtoJoint,
                            Child_Coordinate,
                            Pos, R, RD, JointLoc,
                            joint_ind, child_ind, Child_Body_ind, Parent_Body_ind,
                            prismatic_direction, prism_joint_to_omega_col_ind, prism_parent_body
                        )
                    else:
                        print("Floating joint father is ground; its values should have been computed in compute_grounded().")
                        break  # or continue, as appropriate

                # Step 2: For propagation along the path (if current body has children).
                if j < len(path)-1:
                    R_child     = None
                    RD_child    = None

                    if joint_type == 'R':
                        # For revolute joints, the next child in the path is used.
                        parent_body = path[j-1]
                        child_body  = path[j+1]

                        connecting_parent_joint = [idx for idx, row in enumerate(Joints) 
                                                    if row[0]==parent_body and row[1]==current_body]
                        connecting_child_joint  = [idx for idx, row in enumerate(Joints) 
                                                    if row[0]==current_body and row[1]==child_body]
                        
                        current_Coordinate  = Q[Body_col[current_body-1][-1]]
                        body_thetaD         = QD[Body_col[current_body-1][-1]]
                        child_coordinate    = Q[Body_col[child_body-1][-1]]

                        (R_child, RD_child, RevPrisChain) = rev_joint_children(
                            Body_col, body_thetaD, JointtoChildCG, ParentCGtoJoint,
                            current_Coordinate, Type,
                            connecting_parent_joint[0], connecting_child_joint[0],
                            prismatic_direction, child_coordinate,
                            j, path, Joints, Q, QD
                        )
                    elif joint_type == 'P':
                        (R_child, RD_child) = pris_joint_children(
                            Q, QD, joint_ind, prismatic_direction,
                            prism_joint_to_omega_col_ind,prism_parent_body
                        )
                    elif joint_type == 'F':
                        parent_body = path[j-1]
                        child_body  = path[j+1]

                        connecting_parent_joint = [idx for idx, row in enumerate(Joints)
                                                    if row[0]==parent_body and row[1]==current_body]
                        connecting_child_joint  = [idx for idx, row in enumerate(Joints)
                                                    if row[0]==current_body and row[1]==child_body]
                        
                        current_Coordinate  = Q[Body_col[current_body-1][-1]]
                        body_thetaD         = QD[Body_col[current_body-1][-1]]
                        child_coordinate    = Q[Body_col[child_body-1][-1]]

                        (R_child, RD_child, FloatPrisChain) = float_joint_children(
                            Body_col, body_thetaD, JointtoChildCG, ParentCGtoJoint,
                            current_Coordinate, Type,
                            connecting_parent_joint[0], connecting_child_joint[0],
                            prismatic_direction, child_coordinate,
                            j, path, Joints, Q, QD
                        )
                        
                        NDOF                = 3
                        translationalDOF    = NDOF - 1 
                        children_all        = path[2:] # For all the children in the chain
                        row_indices         = []
                        
                        for child in children_all:
                            start_idx = NDOF * child - 3  # Equivalent to 3*(child-1)
                            row_indices.extend([start_idx, start_idx + 1])
                        
                        # Get the column indices for the translation part of the current body's DOFs.
                        col_indices         = Body_col[current_body - 1][:translationalDOF]
                        
                        # Create a block by vertically repeating a 2x2 identity matrix 
                        identity_block      = np.tile(np.eye(translationalDOF), (len(children_all), 1))
                        
                        # Finally, assign the identity block to the R matrix at the specified rows and columns.
                        # Simpy does not allow submatrix assignment
                        for m_idx, m in enumerate(row_indices):
                            for n_idx, n in enumerate(col_indices):
                                R[m, n] = identity_block[m_idx, n_idx]

                    else:
                        print("The joint type is not defined")
                        break

                # Step 3: Propagate R_child and RD_child to the subsequent bodies along the path.
                for k in range(j+1, len(path)):
                    child_ind           = path[k]
                    Child_Body_ind_k    = [3*(child_ind-1), 3*(child_ind-1)+1, 3*(child_ind-1)+2]

                    if RevPrisChain == 0 and FloatPrisChain == 0:
                        R_track[child_ind-1, current_body-1] = 1

                        # For x and z rows:
                        for idx in range(2):
                            R[Child_Body_ind_k[idx], Body_col[current_body-1][-1]]  = R_child[idx]
                            RD[Child_Body_ind_k[idx], Body_col[current_body-1][-1]] = RD_child[idx]

                    elif  FloatPrisChain == 1:
                        R_track[child_ind-1, current_body-1] = 1

                        # For x and z rows, assign identity:
                        R[Child_Body_ind_k[0], Body_col[current_body-1][0]] = 1
                        R[Child_Body_ind_k[1], Body_col[current_body-1][1]] = 1

                        # For theta row, assign from R_child
                        for idx in range(2):
                            R[Child_Body_ind_k[idx], Body_col[current_body-1][-1]]  = R_child[idx,child_ind-1]
                            RD[Child_Body_ind_k[idx], Body_col[current_body-1][-1]] = RD_child[idx,child_ind-1]

                    elif RevPrisChain == 1:
                        R_track[child_ind-1, current_body-1] = 1

                        R[Child_Body_ind_k[0], Body_col[current_body-1][-1]]    = R_child[0, child_ind-1]
                        R[Child_Body_ind_k[1], Body_col[current_body-1][-1]]    = R_child[1, child_ind-1]
                        RD[Child_Body_ind_k[0], Body_col[current_body-1][-1]]   = RD_child[0, child_ind-1]
                        RD[Child_Body_ind_k[1], Body_col[current_body-1][-1]]   = RD_child[1, child_ind-1]
            # End for each path.
        return Pos, JointLoc, R, RD, R_track, pris_body_to_parent_map, paths, graph

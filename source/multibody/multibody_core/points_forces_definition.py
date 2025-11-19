"""
Created on Thu Mar 20 10:13:33 2025

@author: adiazfl
"""
import sympy as sym
from ._joints_helpers import rot_2D_wave, tilde_2D_wave
from ._springs_dampers_helpers import cross2D, SpringsAndDampersHelpers as SDH, parse_point_str
import copy

def points_force_finder(Joints, NBodies, Pos, Q, QD, Initial_Points, Force, JointLoc, 
                       pris_body_to_parent_map, Body_col, Vel, Reference_frame_Origin):
    """
    Reads the points and forces defined in the dictionaries Initial_Points and Forces and parses them
    into the MBDsystem class.
    
    Parameters
    ----------
    data : list
        Contains fields 'joints_system.Joint' (list of [parent, child] pairs) and 'Type' (list of joint types).
    NBodies : int
        Number of bodies in the system (1-based indexing for bodies).
    Pos : sympy.Matrix
        Global positions of all bodies stacked as [x1, z1, theta1, x2, z2, theta2, ...].
    Q : list of sympy.Symbol
        The symbolic position coordinates (e.g. [Theta_1, S_2, ...]).
    QD : list of sympy.Symbol
        The symbolic velocity coordinates (e.g. [ThetaD_1, SD_2, ...]).
    Initial_Points : dict
        Must contain:
          - "GR": list of ground points (each [x, z]).
          - "BD": dict keyed by body number -> list of points [xRel, zRel], possibly symbolic.
    Force : dict
        Dictionary of forces. May contain:
          - "PointsBD", "CG", "TensionSpring", "TensionDamper", "TorsionSpring", "TorsionDamper".
    JointLoc : list of 2D coordinates
        The location of each joint, consistent with data['Joints'] ordering.
    pris_body_to_parent_map : list or dict
        Info for prismatic joints' actual parent angles, as returned by the velocity routines.
    Body_col : list of lists
        For each body (1-based), the column indices in Q/QD that belong to it.
        E.g. Body_col[i-1] is a list of DOF indices for body i.
    Vel : sympy.Matrix
        Global velocities, parallel to Pos (stacked as [xD1, zD1, thetaD1, xD2, zD2, thetaD2, ...]).
    
    Returns
    -------
    Points_All : dict
        Dictionary storing the computed positions for all relevant points (BD, GR, CG, JO).
    Force_All : dict
        Dictionary storing the computed force contributions for each body in each force category.
    SpringPotentialEnergy : sympy.Expr
        Total spring potential energy from tension/torsion springs.
    """

    # Initialize the output dictionary Points_All
    # We'll store:
    #   - BD (body-defined points)
    #   - GR (ground points)
    #   - CG (centers of gravity)
    #   - JO (joint points)
    Points_All = {}
    Vel_All = {}
    
    # 1) Initialize placeholders for JO (joint) info. 
    #    Typically, you'd store each joint location associated with its body.
    #    We'll do a minimal version: a list of size NBodies+1 
    #    (since ground is considered 'body 0' at index NBodies).
    Points_All['JO'] = [[] for _ in range(NBodies+1)]
    Points_All['num_Joint_onBody_tracker'] = [0]*(NBodies+1)
    Points_All['type_joint_tracker'] = [[] for _ in range(NBodies+1)]
    
    # 2) Fill in JO (joint) data using JointLoc and Type
    # for i, (parent_body, child_body) in enumerate(Joints):
    for i,jointID in enumerate(Joints):
        parent_body = jointID.parent
        child_body  = jointID.child
        jType       = jointID.joint_type.value
        # If parent is ground, store joint info at index NBodies
        if parent_body == 0:
            # Update ground
            Points_All['num_Joint_onBody_tracker'][NBodies] += 1
            Points_All['type_joint_tracker'][NBodies].append(jType)
            Points_All['JO'][NBodies].append(JointLoc[i,:])
            
            # Update child
            child_idx = child_body - 1
            Points_All['num_Joint_onBody_tracker'][child_idx] += 1
            Points_All['type_joint_tracker'][child_idx].append(jType)
            Points_All['JO'][child_idx].append(JointLoc[i,:])
        else:
            # Both parent and child are > 0
            parent_idx = parent_body - 1
            child_idx  = child_body - 1
            
            Points_All['num_Joint_onBody_tracker'][parent_idx] += 1
            Points_All['type_joint_tracker'][parent_idx].append(jType)
            Points_All['JO'][parent_idx].append(JointLoc[i,:])

            Points_All['num_Joint_onBody_tracker'][child_idx] += 1
            Points_All['type_joint_tracker'][child_idx].append(jType)
            Points_All['JO'][child_idx].append(JointLoc[i,:])
    
    Points_All['JO'] = copy.deepcopy(JointLoc) # TODO: see if the above can be reomoved
    # 3) Centers of gravity (CG) for each body
    #    For a 2D system, CG is at indices [3*(i-1), 3*(i-1)+1] in Pos.
    CG_positions    = []
    CG_velocities   = []
    
    for body_i in range(1, NBodies+1):
        offset  = 3*(body_i-1)
        xz      = Pos[offset:offset+2, 0]  # x,z the zero keeps it as array
        CG_positions.append(xz)
        
        xz_vel  = Vel[offset:offset+2, 0]  # xD, zD
        CG_velocities.append(xz_vel)
        
    Points_All['CG']    = sym.Matrix.vstack(*CG_positions).reshape(len(CG_positions),2)
    Vel_All['CG']       = sym.Matrix.vstack(*CG_velocities).reshape(len(CG_velocities),2)
    
    # 4) Body-defined points (BD)
    #    For each (body -> list of [xRel, zRel]) in Initial_Points['BD']:
    #    transform them by the body's rotation and CG.
    #    NOTE: the points in each body are accessed through the body index,
    #          i.e, BD points for body is ...['BD'][1] 
    Points_All['BD']    = {}
    Vel_All['BD']       = {}
    
    for body, pt_list in Initial_Points["BD"].items():
        Points_All['BD'][body]      = []
        Vel_All['BD'][body]         = []
        
        # The angle for this body is Q at the last DOF in Body_col[body-1]
        # We store CG, CG_vel, etc.
        offset = 3*(body-1)
        theta  = Pos[offset+2, 0]
        thetaD = Vel[offset+2, 0]
        rotM   = rot_2D_wave(theta)
        cg_pos = CG_positions[body-1]
        cg_vel = CG_velocities[body-1]
        
        for pt_local in pt_list:
            pt_vec = sym.Matrix(pt_local)
            # Absolute position
            rel_pos = rotM * pt_vec
            pt_abs  = cg_pos + rel_pos
            # Velocity
            pt_vel  = cg_vel + tilde_2D_wave(rel_pos)*thetaD
            
            Points_All['BD'][body].append(pt_abs)
            Vel_All['BD'][body].append(pt_vel)
         
        # Convert each entry in the dictionary to a sym.Matrix
        Points_All['BD'][body]  = sym.Matrix(Points_All['BD'][body]).reshape(len(pt_list),2)
        Vel_All['BD'][body]     = sym.Matrix(Vel_All['BD'][body]).reshape(len(pt_list),2)
    
    # 5) Ground points (GR)
    #    Just store them as absolute coords, with zero velocity.
    Points_All['GR']    = []
    Vel_All['GR']      = []
    if "GR" in Initial_Points:
        for pt in Initial_Points["GR"]:
            GR_pos = sym.Matrix(pt) + Reference_frame_Origin
            Points_All['GR'].append(sym.Matrix(GR_pos).T)
            Vel_All['GR'].append(sym.zeros(1,2))  # 2D zero
    
    # 6) Prepare to accumulate forces in Force_All
    #    We'll store a 3D vector [Fx, Fz, My] for each body in each category.
    Force_All = {}
    categories = ["PointsBD", "CG", "TensionSpring", "TensionDamper", "TorsionSpring", "TorsionDamper"]
    for cat in categories:
        Force_All[cat] = [sym.zeros(3,1) for _ in range(NBodies)]
    
    # Also track the total spring potential energy
    SpringPotentialEnergy = sym.Integer(0)
    
    # 7) PointsBD forces
    #    Each entry is [body, pointID, Fx, Fy, Mz], but in 2D we treat Fy as Fz.
    if "PointsBD" in Force:
        for entry in Force["PointsBD"]:
            # entry is e.g. [body, ptID, Fx, Fz, My]
            # In MATLAB, user might pass a function handle; here we treat them as sympify or lambda.
            # Convert to sym
            row_sym = [sym.sympify(e) if not callable(e) else e(0) for e in entry]
            b_    = int(row_sym[0])
            ptID_ = int(row_sym[1])
            Fx_   = row_sym[2]
            Fz_   = row_sym[3]
            My_   = row_sym[4]
            
            # Locate the actual point in Points_All['BD'][b_][ptID_-1].
            # Then compute the moment about the CG if needed.
            pt_abs  = Points_All['BD'][b_][ptID_,:]
            cg_abs  = Points_All['CG'][b_-1,:]
            rel_pos = pt_abs - cg_abs  # 2D vector
            M_force = cross2D(rel_pos,sym.Matrix([Fx_,Fz_])) # TODO: The sign folllows y-roations
            # net moment about CG = Mz from the user + the cross product
            net_Mz  = My_ + M_force
            Force_All["PointsBD"][b_-1] += sym.Matrix([Fx_, Fz_, net_Mz])
    
    # 8) CG forces
    if "CG" in Force:
        for entry in Force["CG"]:
            row_sym = [sym.sympify(e) if not callable(e) else e(0) for e in entry]
            b_    = int(row_sym[0])
            Fx_   = row_sym[1]
            Fz_   = row_sym[2]
            My_   = row_sym[3]
            Force_All["CG"][b_-1] += sym.Matrix([Fx_, Fz_, My_])
    
    # 9) TensionSpring
    #    Each item: (("BD12","BD23"), [l0, k]) or (("CG11","BD32"), [...]) etc.
    if "TensionSpring" in Force:
        '''
        Force_TensionSpring : list
            List of tension spring definitions. Each element is a tuple:
              ((point_str1, point_str2), [l0, k])
            where l0 is the undeformed length and k is the spring stiffness (which
            may be a constant or a lambda function of DeltaL).
        '''
    
        for (points, spring_params) in Force["TensionSpring"]:
        # for (points, spring_params) in Force_TensionSpring:
            point_str1, point_str2 = points
            # Parse the point strings
            p1_type, p1_body, p1_cell, _, _, _ = parse_point_str(point_str1)
            p2_type, p2_body, p2_cell, _, _, _ = parse_point_str(point_str2)
    
            # Use the helper to compute the connecting vectors, direction, and length.
            RelPosFirst, RelPosSecond, UnitVector, Length, RelPos = \
                SDH.connecting_point_finder(
                    p1_type, p1_body, p1_cell,
                    p2_type, p2_body, p2_cell,
                    Points_All
                )
            # Compute the spring force and potential energy.
            F1, F2, springPE = SDH.spring_force(
                RelPosFirst, RelPosSecond, UnitVector, Length, spring_params
            )
            # Add the computed forces to the corresponding bodies (if not ground).
            if p1_type != "GR":
                Force_All["TensionSpring"][p1_body - 1] += F1.T
            if p2_type != "GR":
                Force_All["TensionSpring"][p2_body - 1] += F2.T
            # Accumulate the spring potential energy.
            SpringPotentialEnergy += springPE
    
    
        # 10) TensionDamper
    if "TensionDamper" in Force:
        for (points, damper_val) in Force["TensionDamper"]:
            point_str1, point_str2 = points
            # Parse the point strings
            p1_type, p1_body, p1_cell, _, _, _ = parse_point_str(point_str1)
            p2_type, p2_body, p2_cell, _, _, _ = parse_point_str(point_str2)
            
            # Use the helper to compute the connecting vectors, direction, and length.
            RelPosFirst, RelPosSecond, UnitVector, Length, RelPos = \
                SDH.connecting_point_finder(
                    p1_type, p1_body, p1_cell,
                    p2_type, p2_body, p2_cell,
                    Points_All
                )
                
            (FirstPointVel, SecondPointVel) = \
                SDH.vel_finder(
                    p1_type, p1_body, p1_cell,
                    p2_type, p2_body, p2_cell,
                    Vel_All
                )

            F1,F2 = SDH.damper_force(RelPosFirst, RelPosSecond, UnitVector, Length,
                                  damper_val, FirstPointVel, SecondPointVel, RelPos)
            
            # Add the computed forces to the corresponding bodies (if not ground).
            if p1_type != "GR":
                Force_All["TensionDamper"][p1_body - 1] += F1.T
            if p2_type != "GR":
                Force_All["TensionDamper"][p2_body - 1] += F2.T
            
        # 11) TorsionSpring
        if "TorsionSpring" in Force:
            for torsion_entry in Force["TorsionSpring"]:
                # Assume each entry is a tuple: (params, optional lambda)
                # If a lambda is provided, it is ignored in this basic call
                if isinstance(torsion_entry, (list, tuple)) and len(torsion_entry) == 2:
                    bodies, params = torsion_entry
                else:
                    bodies = torsion_entry
                    params = None
                # The params list is expected to be [theta0, k]
                first_body = int(bodies[0])
                second_body = int(bodies[1])
                # Call the torsional spring helper:
                F1, F2, torsionPE = SDH.torsional_spring(
                    first_body, second_body, pris_body_to_parent_map, Body_col, Q, params
                )
                # Add the computed torques to the appropriate bodies.
                if first_body != 0:
                    Force_All["TorsionSpring"][first_body - 1] += sym.Matrix([0,0,F1])
                if second_body != 0:
                    Force_All["TorsionSpring"][second_body - 1] +=  sym.Matrix([0,0,F2])
                # Accumulate the potential energy for the torsion spring.
                SpringPotentialEnergy += torsionPE
        
        # 12) TorsionDamper
        if "TorsionDamper" in Force:
            for torsion_entry in Force["TorsionDamper"]:
                # Assume each entry is a tuple: (params, optional lambda)
                if isinstance(torsion_entry, (list, tuple)) and len(torsion_entry) == 2:
                    bodies, params = torsion_entry
                else:
                    bodies = torsion_entry
                    params = None
                # The params float is expected to be C_0
                first_body = int(bodies[0])
                second_body = int(bodies[1])
                # Call the torsional damper helper:
                F1, F2 = SDH.torsional_damper(
                    first_body, second_body, pris_body_to_parent_map, Body_col, QD, params
                )
                if first_body != 0:
                    Force_All["TorsionDamper"][first_body - 1] += sym.Matrix([0,0,F1])
                if second_body != 0:
                    Force_All["TorsionDamper"][second_body - 1] += sym.Matrix([0,0,F2])

    
    return Points_All, Force_All, SpringPotentialEnergy, Vel_All







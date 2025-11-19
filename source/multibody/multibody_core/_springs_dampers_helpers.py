"""
Created on Thu Mar 20 16:32:55 2025

@author: adiazfl
"""
import sympy as sym

def cross2D(MomentArm, Force):
    """
    2D cross product scalar: cross2D((Ax,Az), (Bx,Bz)) = Ax*Bz - Az*Bx
    Returns a Sympy expression (scalar).
    """
    return MomentArm[1]*Force[0] - MomentArm[0]*Force[1]

def parse_point_str(point_str):
    """
    Parses a point string used in tension spring definitions.
    
    Expected formats:
      - If length is 4 (e.g., "BD41"): 
            point type = first two characters (e.g., "BD"),
            body = third character as integer,
            cell = fourth character as integer.
            
      - If length is 6 or 8, similar logic applies (the extra characters are ignored).
    
    Returns:
      (point_type, body, cell, None, None, None)
      (Only the first three values are used for tension spring calculations.)
    """
    if len(point_str) == 4:
        point_type  = point_str[0:2].upper()
        body        = int(point_str[2])
        cell        = int(point_str[3])
        return point_type, body, cell, None, None, None
    elif len(point_str) in [6, 8]:
        point_type  = point_str[0:2].upper()
        body        = int(point_str[2])
        cell        = int(point_str[3])
        return point_type, body, cell, None, None, None
    else:
        raise ValueError("Unexpected format for tension spring point string: " + point_str)

class SpringsAndDampersHelpers:
    """
    A helper class collecting the spring/damper routines from your MATLAB code.
    """

    @staticmethod
    def connecting_point_finder(FristPointType, FristPointBody, FristPointCell,
                              SecondPointType, SecondPointBody, SecondPointCell,
                              Points_All):
        """
        Python translation of the MATLAB ConnectingPointFinder function.

        Returns:
          RelPosFirst (1x2 sympy.Matrix) : The point's position relative to the CG of its body (first).
          RelPosSecond (1x2 sympy.Matrix): The point's position relative to the CG of its body (second).
          UnitVector (1x2 sympy.Matrix)  : The direction (unit vector) from the first point to the second.
          Length (sympy.Expr)            : The scalar distance between the two points.
          RelPos (1x2 sympy.Matrix)      : The vector (second - first).
        """
        # We'll store the absolute positions in FirstPointPos, SecondPointPos
        # Then compute relative positions, etc.
        # Note: We assume "Points_All" is a dictionary with keys "BD","CG","GR", etc.

        # For Python, we do a 0x2 or 1x2 approach using sympy.Matrix(1,2, [...]) or just a 2-vector.
        # We'll store row vectors as dimension (2,1) or (1,2). Let's pick (1,2).
        zero_1x2 = sym.Matrix([[0,0]])  # 1 row, 2 cols

        # --- 1) Figure out absolute positions
        if FristPointType == 'GR':
            # "Points_All['GR']" is presumably a list, so we do Points_All['GR'][FristPointBody]
            # if body indexing for ground is used that way. The code below might need an offset if not 0-based.
            # We'll assume 1-based: e.g. if FristPointBody=1, we do Points_All['GR'][0].
            FirstPointPos   = Points_All['GR'][FristPointCell]  # a 2D vector
            RelPosFirst     = zero_1x2
        elif FristPointType == 'BD':
            FirstPointPos = Points_All['BD'][FristPointBody][FristPointCell,:]  # also a 2D vector
            # Relative to CG: Points_All["CG"][FristPointBody - 1] is the CG for body FristPointBody
            cg_first        = Points_All['CG'][FristPointBody - 1,:]
            RelPosFirst     = FirstPointPos - cg_first  # (2,1) => (1,2)
        elif FristPointType == 'CG':
            # The CG is stored in Points_All["CG"][bodyIndex - 1]
            FirstPointPos   = Points_All['CG'][FristPointBody - 1,:]
            RelPosFirst     = zero_1x2
        elif FristPointType == 'JO':
            raise ValueError("Cannot define a tension-based spring/damper on a joint ('JO').")
        else:
            raise ValueError(f"Unknown first point type: {FristPointType}")

        if SecondPointType == 'GR':
            SecondPointPos = Points_All['GR'][SecondPointCell]
            RelPosSecond   = zero_1x2
        elif SecondPointType == 'BD':
            SecondPointPos = Points_All['BD'][SecondPointBody][SecondPointCell,:]
            cg_second      = Points_All['CG'][SecondPointBody - 1,:]
            RelPosSecond   = SecondPointPos - cg_second
        elif SecondPointType == 'CG':
            SecondPointPos = Points_All['CG'][SecondPointBody - 1,:]
            RelPosSecond   = zero_1x2
        elif SecondPointType == 'JO':
            raise ValueError("Cannot define a tension-based spring/damper on a joint ('JO').")
        else:
            raise ValueError(f"Unknown second point type: {SecondPointType}")

        # RelPos is 1x2
        RelPos      = (SecondPointPos - FirstPointPos)
        Length      = sym.sqrt(RelPos[0]**2 + RelPos[1]**2)
        UnitVector  = RelPos / Length
        
        return RelPosFirst, RelPosSecond, UnitVector, Length, RelPos


    @staticmethod
    def vel_finder(FristPointType, FristPointBody, FristPointCell,
                  SecondPointType, SecondPointBody, SecondPointCell,
                  Vel_All):
        """
        Python translation of the MATLAB VelFinder function.

        Returns:
          FirstPointVel (1x2 sympy.Matrix)
          SecondPointVel (1x2 sympy.Matrix)
        """
        zero_1x2 = sym.Matrix([[0,0]])

        def getVelBD(body, cell):
            # Return Vel_All["BD"][body][cell - 1] as a row (1x2).
            # If it's nan, we raise an error or warning.
            vel_val = Vel_All["BD"][body][cell,:]
            # If it's a valid 2D vector, convert to row:
            return vel_val.T if vel_val.shape == (2,1) else vel_val

        # 1) First point velocity
        if FristPointType == 'GR':
            FirstPointVel = zero_1x2
        elif FristPointType == 'BD':
            # check if it's NaN
            candidate = Vel_All["BD"][FristPointBody][FristPointCell,:]
            # if it is all nan?
            if (candidate[0]== sym.nan or candidate[1]== sym.nan):
                raise ValueError('This point is not defined in "points_force_def.m"')
            FirstPointVel = getVelBD(FristPointBody, FristPointCell)
        elif FristPointType == 'CG':
            # CG is stored as row vector maybe
            FirstPointVel = Vel_All["CG"][FristPointBody - 1,:]
            if FirstPointVel.shape == (2,1):
                FirstPointVel = FirstPointVel.T
        elif FristPointType == 'JO':
            raise ValueError("Cannot define a tension-based spring/damper on a joint ('JO').")
        else:
            raise ValueError(f"Unknown first point type: {FristPointType}")

        # 2) Second point velocity
        if SecondPointType == 'GR':
            SecondPointVel = zero_1x2
        elif SecondPointType == 'BD':
            candidate = Vel_All["BD"][SecondPointBody][SecondPointCell,:]
            if (candidate[0]== sym.nan or candidate[1]== sym.nan):
                raise ValueError('This point is not defined in "points_force_def.m"')
            SecondPointVel = getVelBD(SecondPointBody, SecondPointCell)
        elif SecondPointType == 'CG':
            SecondPointVel = Vel_All["CG"][SecondPointBody - 1,:]
            if SecondPointVel.shape == (2,1):
                SecondPointVel = SecondPointVel.T
        elif SecondPointType == 'JO':
            raise ValueError("Cannot define a tension-based spring/damper on a joint ('JO').")
        else:
            raise ValueError(f"Unknown second point type: {SecondPointType}")

        # Simplify
        FirstPointVel = sym.simplify(FirstPointVel)
        SecondPointVel = sym.simplify(SecondPointVel)

        return FirstPointVel, SecondPointVel


    @staticmethod
    def spring_force(RelPosFirst, RelPosSecond, UnitVector, Length, Currentl0_Stifness):
        """
        Python translation of the MATLAB SpringForceFinder function.

        Returns:
          FirstPointForce (1x3 sympy.Matrix): [Fx, Fz, My]
          SecondPointForce (1x3 sympy.Matrix): [Fx, Fz, My]
          SpringPotentialEnergy_i (sympy.Expr)
        """
        # Currentl0_Stifness = [l0, K] or possibly symbolic
        UndeformedLength = Currentl0_Stifness[0]
        Stiffness        = Currentl0_Stifness[1]
        DeltaL           = Length - UndeformedLength

        # Check if Stiffness is a function of 'l'
        Stiffness           = sym.sympify(Stiffness)
        vars_in_stiffness   = list(Stiffness.free_symbols)

        if  any(s.name == 'l' for s in vars_in_stiffness):
            # Nonlinear spring where Stiffness is the norm of the force
            Stiffness           = Stiffness.subs('l', DeltaL)
            FirstPointForceXZ   = UnitVector * Stiffness
        else:
            # Linear spring (linear in DeltaL)
            FirstPointForceXZ   = UnitVector * Stiffness * DeltaL
        
        # Potential energy: 1/2 * K * (DeltaL^2)
        # If K is no longer constant, we might do an integral. The code does a direct expression:
        SpringPotentialEnergy_i = sym.Rational(1,2) * Stiffness * (DeltaL**2)

        # 2D cross product for moments:
        # cross2D(RelPosFirst, Force)
        # RelPosFirst, RelPosSecond are row vectors or 1x2 -> convert to col for cross:
        rF_col = RelPosFirst
        if rF_col.shape == (1,2):
            rF_col = rF_col.T
        forceXZ_col = FirstPointForceXZ
        if forceXZ_col.shape == (1,2):
            forceXZ_col = forceXZ_col.T
            
        moment_first = cross2D(rF_col, forceXZ_col)

        # FirstPointForce: [Fx, Fz, M]
        # We'll keep it as 1x3
        FxF             = FirstPointForceXZ[0]
        FzF             = FirstPointForceXZ[1]
        FirstPointForce = sym.Matrix([[FxF, FzF, moment_first]])

        # second point:
        SecondPointForceXZ = -FirstPointForceXZ
        rS_col = RelPosSecond
        if rS_col.shape == (1,2):
            rS_col = rS_col.T
            
        moment_second       = cross2D(rS_col, SecondPointForceXZ.T)
        FxS                 = SecondPointForceXZ[0]
        FzS                 = SecondPointForceXZ[1]
        SecondPointForce    = sym.Matrix([[FxS, FzS, moment_second]])

        return FirstPointForce, SecondPointForce, SpringPotentialEnergy_i


    @staticmethod
    def damper_force(RelPosFirst, RelPosSecond, UnitVector, Length,
                          CurrentDamping, FirstPointVel, SecondPointVel, RelPos):
        """
        Python translation of the MATLAB DamperForceFinder function.

        Returns:
          FirstPointForce (1x3 sympy.Matrix): [Fx, Fz, My]
          SecondPointForce (1x3 sympy.Matrix): [Fx, Fz, My]
        """
        # DeltaLDot = (1/Length)*RelPos*(SecondPointVel - FirstPointVel)'
        # We'll treat RelPos, FirstPointVel, SecondPointVel as row vectors (1x2).
        # Convert them to col if needed:
        R_col = RelPos
        if R_col.shape == (1,2):
            R_col = R_col.T
        # velocity difference:
        Vdiff = (SecondPointVel - FirstPointVel)
        if Vdiff.shape == (1,2):
            Vdiff = Vdiff.T
        # scalar:
        DeltaLDot = (R_col.T * Vdiff)[0] / Length  # R dot (v2 - v1) / |R|

        # Check if CurrentDamping is a function of 'ld'4
        CurrentDamping  = sym.simplify(CurrentDamping)
        vars_in_damping = list(CurrentDamping.free_symbols)
        
        if any(s.name == 'ld' for s in vars_in_damping):
            # Nonlinear damper where CurrentDamping is the norm of the force
            c_damp              = CurrentDamping.subs('ld', DeltaLDot)
            FirstPointForceXZ   = UnitVector * c_damp
            
        else:
            # Linear damper (linear on velocity)
            FirstPointForceXZ   = UnitVector * CurrentDamping * DeltaLDot
        
        # Cross product for moment
        rF_col = RelPosFirst
        if rF_col.shape == (1,2):
            rF_col = rF_col.T
        forceXZ_col = FirstPointForceXZ
        if forceXZ_col.shape == (1,2):
            forceXZ_col = forceXZ_col.T
        moment_first = cross2D(rF_col, forceXZ_col)

        FxF             = FirstPointForceXZ[0]
        FzF             = FirstPointForceXZ[1]
        FirstPointForce = sym.Matrix([[FxF, FzF, moment_first]])

        SecondPointForceXZ = -FirstPointForceXZ
        rS_col = RelPosSecond
        if rS_col.shape == (1,2):
            rS_col = rS_col.T
            
        moment_second       = cross2D(rS_col, SecondPointForceXZ.T)
        FxS                 = SecondPointForceXZ[0]
        FzS                 = SecondPointForceXZ[1]
        SecondPointForce    = sym.Matrix([[FxS, FzS, moment_second]])

        return FirstPointForce, SecondPointForce


    @staticmethod
    def torsional_spring(FristPointBody, SecondPointBody,
                            pris_body_to_parent_map, Body_col, Q,
                            params_TorsionSprings):
        """
        Python translation of TorsionalSpringInfo.
        
        Returns:
          FirstBodyTorque, SecondBodyTorque (scalars or symbolic),
          SpringTorPotentialEnergy_i (sympy.Expr)
        """
        # current_TorsionSprings = [b1, b2, theta_0, k]
        # We assume 1-based body indexing. We fetch angle for each body, 
        #   or 0 if pris_body_to_parent_map says 0, or from parent's angle if not None.

        def get_angle(body):
            if body == 0:
                return sym.Integer(0)
            mapval = sym.Float(pris_body_to_parent_map[body-1])
            if mapval is sym.nan:
                # no prismatic joint
                # use Q at Body_col[body-1][-1]
                return Q[Body_col[body-1][-1]]
            elif (mapval < 1e-8 and mapval > -1e-8): 
                return sym.Float(0)
            else:
                # real parent
                real_parent = int(mapval)
                return Q[Body_col[real_parent-1][-1]]

        first_angle     = get_angle(FristPointBody)
        second_angle    = get_angle(SecondPointBody)

        theta_0     = params_TorsionSprings[0]
        DeltaTheta  = (second_angle - first_angle - theta_0)
        Stiffness   = params_TorsionSprings[1]
        
        # Check if Stiffness depends on 'theta'
        Stiffness           = sym.sympify(Stiffness)
        vars_in_stiffness   = list(Stiffness.free_symbols)

        if any(s.name == 'theta' for s in vars_in_stiffness):
            # Nonlinear spring where Stiffness is the norm of the force
            Stiffness       = Stiffness.subs('theta', DeltaTheta)
            FirstBodyTorque = Stiffness
        else:
            # Linear spring (Linear on DeltaTheta)
            FirstBodyTorque = Stiffness * DeltaTheta
        
        SecondBodyTorque            = -FirstBodyTorque
        SpringTorPotentialEnergy_i  = sym.Rational(1,2) * Stiffness * (DeltaTheta**2)
        
        return FirstBodyTorque, SecondBodyTorque, SpringTorPotentialEnergy_i


    @staticmethod
    def torsional_damper(FristPointBody, SecondPointBody,
                            pris_body_to_parent_map, Body_col, QD,
                            param_TorsionDamper):
        """
        Python translation of TorsionalDamperInfo.

        Returns:
          FirstBodyTorque, SecondBodyTorque
        """
        # current_TorsionDamper = [b1, b2, c]
        def get_omega(body):
            if body == 0:
                return sym.Integer(0)
            mapval = sym.Float(pris_body_to_parent_map[body-1])
            if mapval is sym.nan:
                return QD[Body_col[body-1][-1]]
            elif (mapval < 1e-8 and mapval > -1e-8): 
                return sym.Integer(0)
            else:
                real_parent = int(mapval)
                return QD[Body_col[real_parent-1][-1]]

        first_omega     = get_omega(FristPointBody)
        second_omega    = get_omega(SecondPointBody)
        DeltaThetaD     = second_omega - first_omega
        CurrentDamping  = param_TorsionDamper

        # Check if CurrentDamping is a function of 'ld'4
        CurrentDamping  = sym.simplify(CurrentDamping)
        vars_in_damping = list(CurrentDamping.free_symbols)
        
        if any(s.name == 'thetad' for s in vars_in_damping):
            # Nonlinear damper where CurrentDamping is the norm of the force
            c_damp          = CurrentDamping.subs('thetad', DeltaThetaD)
            FirstBodyTorque = c_damp
        else:
            # Linear damper (Linear on DeltaThetaD)
            FirstBodyTorque = CurrentDamping * DeltaThetaD

        SecondBodyTorque = -FirstBodyTorque
        return FirstBodyTorque, SecondBodyTorque



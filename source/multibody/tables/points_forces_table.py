# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:21:10 2025

@author: adiazfl
"""

import pandas as pd

# def to_float(v):
#     import sympy as sym
#     if isinstance(v, sym.Expr):
#         return float(v.evalf(3))
#     return v

def points_table(Initial_Points):
    """
    Build a pandas DataFrame listing all defined points in the system.
    
    Parameters
    ----------
    Initial_Points : dict
        Dictionary with two entries:
        
        - "GR" : list of array-like, shape (2,)
            Ground points, each given as [x, z].
        - "BD" : dict of {int: list of array-like, shape (2,)}
            Body-defined points. Keys are body indices (>0), values are lists of
            points [x_rel, z_rel] relative to that body’s center of gravity.
    
    Returns
    -------
    df_points : pandas.DataFrame
        Table with columns:
    
        - Body : int  
            Body index (0 for ground points, >0 for body-defined points).
        - PointType : str  
            Either "GR" (ground) or "BD" (body-defined).
        - PointID : int  
            Index of the point within its category list.
        - Rel coordinates : array-like, shape (2,)  
            The [x, z] coordinates (absolute for ground, relative for body points).
    """
    # Create a list to store table rows.
    rows = []
    
    # Process ground points (Body 0, PointType "GR").
    for i, pt in enumerate(Initial_Points["GR"]):
        rows.append({
            "Body": 0,
            "PointType": "GR",
            "PointID": i,
            "Rel coordinates": pt
        })
    
    # Process body-defined points (PointType "BD").
    for body, pts in Initial_Points["BD"].items():
        for i, pt in enumerate(pts):
            rows.append({
                "Body": body,
                "PointType": "BD",
                "PointID": i,
                "Rel coordinates": pt
            })
    
    # Convert the list of dictionaries into a pandas DataFrame.
    df_points = pd.DataFrame(rows, columns=[
        "Body",
        "PointType",
        "PointID",
        "Rel coordinates"
        ])
    
    # Display the DataFrame.
    print(df_points.to_string(index=False))
    
    
def parse_point_str(pt_str):
    """
    Simple parser for a string like 'BD11' or 'GR11'.
    Returns (pointType, bodyChar, pointIDStr).
    
    Example:
      'BD11' -> ('BD', '1', '1')
      'GR22' -> ('GR', '2', '2')
      
    Adjust if your naming scheme differs!
    """
    if len(pt_str) < 4:
        raise ValueError(f"Point string '{pt_str}' is unexpectedly short.")
    pointType = pt_str[:2]        # e.g. 'BD', 'GR', 'CG', etc.
    bodyChar  = pt_str[2]         # third character is body index
    pIDStr    = pt_str[3:]        # everything after that is point ID
    return pointType, bodyChar, pIDStr

def force_table(Force):
    """
    Build a consolidated pandas DataFrame of all force entries.

    Parameters
    ----------
    Force : dict of lists
        Dictionary mapping force categories to their definitions:
        
        - PointsBD : list of lists
            Each entry is [body, ptID, Fx, Fy, Mz].
        - CG : list of lists
            Each entry is [body, Fx, Fy, Mz].
        - TensionSpring : list of tuples
            Each entry is ((ptStr1, ptStr2), [l0, stiffness]).
        - TensionDamper : list of tuples
            Each entry is ((ptStr1, ptStr2), damping_value).
        - TorsionSpring : list of tuples
            Each entry is ([body1, body2], [theta0, stiffness]).
        - TorsionDamper : list of tuples
            Each entry is ([body1, body2], damping_value).

    Returns
    -------
    df_forces : pandas.DataFrame
        Table with columns:
        
        - ForceType : str
            One of "PointsBD", "CG", "TensionSpring", "TensionDamper",
            "TorsionSpring", "TorsionDamper".
        - BodiesConnected : str
            Comma-separated body IDs (e.g. "2" or "1,3").
        - PointType : str
            Point category code (e.g. "BD", "CG", "BD,BD", "GR,BD", "JO").
        - PointID : str
            Numeric point IDs (or "NA" if none).
        - ForceComponents : list or None
            [Fx, Fy, Mz] for point-based forces, else None.
        - SpringParameters : list or None
            [l0, stiffness] for springs, else None.
        - DamperParameters : scalar or None
            Damping coefficient for dampers, else None.
    """
    rows = []

    # 1) PointsBD: each entry is [body, ptID, Fx, Fy, Mz]
    for entry in Force["PointsBD"]:
        body = entry[0]
        ptID = entry[1]
        Fx   = entry[2]
        Fy   = entry[3]
        Mz   = entry[4]

        # For a single-body force, BodiesConnected is just e.g. "2".
        bodies_str  = str(body)
        point_type  = "BD"
        point_id    = str(ptID)

        # Force components = [Fx, Fy, Mz]
        force_cmp   = [Fx, Fy, Mz]
        spring_par  = None
        damper_par  = None
        
        # Force type
        force_id = 'PointsBD'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })
        

    # 2) CG: each entry is [body, Fx, Fy, Mz]
    for entry in Force["CG"]:
        body = entry[0]
        Fx   = entry[1]
        Fy   = entry[2]
        Mz   = entry[3]

        bodies_str  = str(body)
        point_type  = "CG"
        point_id    = "NA"  # CG has no numeric point ID
        force_cmp   = [Fx, Fy, Mz]
        spring_par  = None
        damper_par  = None
        
        # Force type
        force_id = 'CG'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })

    # 3) TensionSpring: list of tuples: ( (ptStr1, ptStr2), [l0, stiffness] )
    for entry in Force["TensionSpring"]:
        points_tuple, spring_vals = entry
        ptStr1, ptStr2 = points_tuple
        l0, kVal = spring_vals

        # Parse each point string
        pType1, bodyC1, pID1 = parse_point_str(ptStr1)
        pType2, bodyC2, pID2 = parse_point_str(ptStr2)

        # E.g. "1,2" for bodies
        bodies_str  = f"{bodyC1},{bodyC2}"
        point_type  = f"{pType1},{pType2}"   # e.g. "BD,BD" or "GR,BD" etc.
        point_id    = f"{pID1},{pID2}"

        force_cmp   = None  # Springs do not have direct force vectors here
        spring_par  = [l0, kVal]
        damper_par  = None
        
        # Force type
        force_id = 'TensionSpring'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })

    # 4) TensionDamper: list of tuples: ( (ptStr1, ptStr2), dampingVal )
    for entry in Force["TensionDamper"]:
        points_tuple, damp_val = entry
        ptStr1, ptStr2 = points_tuple

        pType1, bodyC1, pID1 = parse_point_str(ptStr1)
        pType2, bodyC2, pID2 = parse_point_str(ptStr2)

        bodies_str  = f"{bodyC1},{bodyC2}"
        point_type  = f"{pType1},{pType2}"
        point_id    = f"{pID1},{pID2}"

        force_cmp   = None
        spring_par  = None
        damper_par  = damp_val  # e.g. 3, or a symbolic expression
        
        # Force type
        force_id = 'TensionDamper'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })

    # 5) TorsionSpring: e.g. each entry might be [b1, b2, theta0, kVal], or similar
    for entry in Force["TorsionSpring"]:
        bodies,params = entry

        b1      = bodies[0]
        b2      = bodies[1]
        t0      = params[0]
        kVal    = params[1]
        
        bodies_str  = f"{b1},{b2}"
        point_type  = "JO"   # Torsion springs act at a joint
        point_id    = "NA"   # no numeric point ID

        force_cmp   = None
        spring_par  = [t0, kVal]  # or [theta0, stiffness]
        damper_par  = None
        
        # Force type
        force_id = 'TorsionSpring'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })

    # 6) TorsionDamper: e.g. each entry is [b1, b2, dampingVal]
    for (entry,param) in Force["TorsionDamper"]:
        b1    = entry[0]
        b2    = entry[1]
        dVal  = param

        bodies_str  = f"{b1},{b2}"
        point_type  = "JO"
        point_id    = "NA"

        force_cmp   = None
        spring_par  = None
        damper_par  = dVal
        
        # Force type
        force_id = 'TorsionDamper'

        rows.append({
            "ForceType": force_id,
            "BodiesConnected": bodies_str,
            "PointType": point_type,
            "PointID": point_id,
            "ForceComponents": force_cmp,
            "SpringParameters": spring_par,
            "DamperParameters": damper_par
        })

    # Convert to a pandas DataFrame
    df_forces = pd.DataFrame(rows, columns=[
        "ForceType",
        "BodiesConnected",
        "PointType",
        "PointID",
        "ForceComponents",
        "SpringParameters",
        "DamperParameters"
    ])
    
    # df_forces = df_forces.applymap(to_float)
    pd.set_option('display.precision', 3)
    print(df_forces.to_string(index=False))



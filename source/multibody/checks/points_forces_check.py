import warnings
import sympy as sym
import re
import inspect

def check_point_string(pointStr, nBodies, Initial_Points):
    """
    Parse and validate a sequence of point‐definition strings.

    Each string must start with one of:
      - 'BD' (body‐defined point)
      - 'GR' (ground reference point)
      - 'CG' (center of gravity)

    The next two characters are interpreted as:
      - body index (0‐based for BD/CG, 1‐based for GR)
      - point index (0‐based)

    Parameters
    ----------
    pointStr : sequence of str
        Iterable of point codes (e.g. ['BD4_1', 'GR0_2', ...]).
    nBodies : int
        Total number of moving bodies (ground = 0).
    Initial_Points : mapping or object
        Must provide:
          - Initial_Points['BD']: sequence of lists for body‐defined points
          - Initial_Points['GR']: sequence of ground reference points

    Raises
    ------
    ValueError
        - If any element of `pointStr` is not a string.
        - If a prefix is not one of 'BD', 'GR', or 'CG'.
        - If body or point indices cannot be parsed or lie outside valid ranges.
        - If a referenced BD point is undefined or empty.
    """
    
    for i,pt in enumerate(pointStr):
        if not isinstance(pt, str):
            raise ValueError(f"Force array entry({i}) expected a string for the point definition. Found {type(pointStr)}.")
        
        # Extract prefix (first two characters) and the rest of the string
        prefix = pt[:2].upper()  # e.g., 'BD', 'GR' or 'CG'
        
        
        if prefix not in {'BD', 'GR', 'CG'}:
            raise ValueError(f"Unknown prefix '{prefix}' in string '{pointStr}' at cell iCell={i}. Use BD, GR or CG")
        
        
        # Delimited format required, e.g. "BD12_3" -> body=12, point=3 (any width).
        rest = pt[2:]
        if "_" not in rest:
            raise ValueError(f"'{pointStr}' at iCell={i} uses the legacy point-string format. Use the delimited form instead, e.g. 'BD4_1' (body=4, point=1).")
        try:
            body_str, ptIdx_str = rest.split("_")
            bodyIdx, ptIdx = int(body_str), int(ptIdx_str)
        except ValueError:
            raise ValueError(f"Failed to parse '{pointStr}' at iCell={i} into numeric body/pt indices.")
        
        if bodyIdx < 0 or (bodyIdx > nBodies and prefix != 'GR'):
            raise ValueError(f"String '{pointStr}' references body={bodyIdx} out of range (1..{nBodies}).")
        
        # If prefix is BD, check that the BD cell exists and is not empty.
        if prefix == 'BD':
            # bdSize is (num_rows, num_cols) so bdSize[1] is the maximum valid point index.
            bdSize = len(Initial_Points[prefix][bodyIdx])
            if ptIdx < 0 or ptIdx >= bdSize:
                raise ValueError(f"'{pointStr}' references point={ptIdx} out of range (1..{bdSize[1]}) in BD array.")
            # Assuming Initial_Points.BD is 0-based.
            point = Initial_Points['BD'][bodyIdx][ptIdx]
            # Here, an empty point is defined as None or an empty list.
            if point is None or (hasattr(point, '__len__') and len(point) == 0):
                raise ValueError(f"'{pointStr}' references an empty BD point. The user never defined Initial_Points['BD']{{{bodyIdx},{ptIdx}}}.")
        elif prefix == 'GR':
            if ptIdx >= len(Initial_Points['GR']):
                raise ValueError(f"'{pointStr}' references an empty GR point. The user never defined Initial_Points['GR']{{{bodyIdx},{ptIdx}}}.")

            
                    
            
def references_time_in_handle(fn, tSym):
    """
    Determine whether a callable’s source code references the time symbol.

    Attempts to retrieve the function’s source; if unavailable, falls back
    to inspecting its `__code__.co_names`. Searches for `t` as a standalone
    token.

    Parameters
    ----------
    fn : callable
        The function or lambda to inspect.
    tSym : sympy.Symbol
        The symbolic time variable (e.g. `sym.symbols('t', real=True)`).

    Returns
    -------
    bool
        True if `t` appears in the source or code object names; False otherwise.
    """
    try:
        source = inspect.getsource(fn)
    except Exception:
        try:
            names = fn.__code__.co_names
            return 't' in names
        except Exception:
            return False
    # Use regular expression to detect "t" as a whole word.
    return re.search(r'\bt\b', source) is not None
            
def check_parameter_has_no_function_of_t(param, forceType, tSym): 
    """
    Ensure a force parameter does not depend on time via a Python function.

    If `param` is callable and its source (or code object) references `tSym`,
    this is disallowed: force parameter expressions must be fully symbolic
    in time.

    Parameters
    ----------
    param : any
        The force parameter to validate (callable, numeric, or symbolic).
    forceType : str
        Descriptor of the force category (e.g. 'TensionSpring').
    tSym : sympy.Symbol
        The symbolic time variable to check against.

    Raises
    ------
    ValueError
        If `param` is callable and references `tSym` in its implementation.
    """
    if callable(param):
        if references_time_in_handle(param, tSym):
            raise ValueError(f"{forceType} entry uses a callable that references t. Must use a symbolic expression for t instead.")

def points_forces(joints, types, Initial_Points, Force, t=sym.symbols('t', real=True)):
    """
    Validate point and force definitions for a planar multibody mechanism.

    Performs these checks in order:

      1. Adds missing keys in `Force` → ensures lists for
         'PointsBD', 'CG', 'TensionSpring', 'TensionDamper',
         'TorsionSpring', 'TorsionDamper'.

      2. Adds missing keys in `Initial_Points` → ensures lists for 'BD', 'GR'.
      
      3. Validates each entry in `Force['PointsBD']`:

         - Must be list‐like with ≥ 2 elements.
         - Body and point indices are integers within valid ranges.
         - Referenced BD point exists and is not empty.

      4. Validates each entry in `Force['CG']`:

         - Must be list‐like with ≥ 2 elements.
         - Body indices are within [1, nBodies].

      5. Validates `TensionSpring` and `TensionDamper` entries:

         - Entries come in (pointStr, param) pairs.
         - `pointStr` parsed via `check_point_string`.
         - `param` passes `check_parameter_has_no_function_of_t`.
         - Callable `param` is converted to symbolic form.

      6. Validates `TorsionSpring` and `TorsionDamper` entries:

         - Same pairing and time‐dependence rules.
         - Checks that any numeric/symbolic iterable has valid body indices.

    Parameters
    ----------
    joints : array‐like of shape (n_joints, 2)
        List of [parent, child] index pairs.
    types : sequence of str
        Joint type codes ('R', 'P', 'F') matching `joints`.
    Initial_Points : mapping or object
        Must supply:
          - Initial_Points['BD']: body‐defined point lists
          - Initial_Points['GR']: ground reference point list
    Force : dict‐like
        Force definitions with keys:
          'PointsBD', 'CG', 'TensionSpring', 'TensionDamper',
          'TorsionSpring', 'TorsionDamper'
    t : sympy.Symbol
        The symbolic time variable for detecting unwanted time‐dependence.

    Returns
    -------
    None
        Prints success messages if all validations pass.

    Raises
    ------
    ValueError
        If any of the above checks fail, with an explanatory message.
    """
    # ---------------------------------------------
    # A) Ensure each Force type field exists
    # ---------------------------------------------
    expectedForceFields = ['PointsBD', 'CG', 'TensionSpring', 'TensionDamper', 'TorsionSpring', 'TorsionDamper']
    for field in expectedForceFields:
        if not field in Force:
            warnings.warn(f"Force.{field} does not exist. Creating Force.{field} = [].")
            setattr(Force, field, [])
    
    # ---------------------------------------------
    # B) Ensure each Initial_Points field exists
    # ---------------------------------------------
    expectedPointsFields = ['BD', 'GR']
    for field in expectedPointsFields:
        if not field in Initial_Points:
            warnings.warn(f"Initial_Points.{field} does not exist. Creating Initial_Points.{field} = [].")
            setattr(Initial_Points, field, [])
    

    nBodies = len(joints)
    
    # ---------------------------------------------
    # C) Check Force.PointsBD entries
    # ---------------------------------------------
    for i, entry in enumerate(Force['PointsBD'], start=1):
        # TODO: check value is numeric or symbolic
        # Check that entry is list-like and has at least 2 elements.
        try:
            if len(entry) < 2:
                raise ValueError(f"Force.PointsBD{{{i}}} must have at least [bodyIdx, pointIdx, Fx, Fy, ...].")
        except TypeError:
            raise ValueError(f"Force.PointsBD{{{i}}} must be list-like; got type {type(entry)}.")
        bodyIdx = entry[0]
        pointIdx = entry[1]
        bd_cols = len(Initial_Points['BD'][bodyIdx])
        # Check that bodyIdx is an integer.
        if not (isinstance(bodyIdx, int) or (isinstance(bodyIdx, sym.Basic) and bodyIdx.is_integer)):
            raise ValueError(f"Force.PointsBD{{{i}}} body index must be an integer. Found {bodyIdx}.")
        if bodyIdx < 1 or bodyIdx > nBodies:
            raise ValueError(f"In Force.PointsBD{{{i}}}, body index = {bodyIdx} exceeds the number of bodies ({nBodies}).")
        if pointIdx < 0 or pointIdx > bd_cols:
            raise ValueError(f"In Force.PointsBD{{{i}}}, point index = {pointIdx} not valid (should be between 1 and {bd_cols}).")
        point = Initial_Points['BD'][bodyIdx][pointIdx-1]
        if point is None or (hasattr(point, '__len__') and len(point) == 0):
            raise ValueError(f"In Force.PointsBD{{{i}}}, the referenced point is not defined: Initial_Points.BD{{{bodyIdx},{pointIdx}}} is empty.")
    
    # ---------------------------------------------
    # E) Check Force.CG entries
    # ---------------------------------------------
    for i, entry in enumerate(Force['CG']):
        # TODO: check value is numeric or symbolic
        try:
            if len(entry) < 2:
                raise ValueError(f"Force.CG{{{i}}} should have at least [bodyIdx, Fx, Fy].")
        except TypeError:
            raise ValueError(f"Force.CG{{{i}}} must be list-like; got type {type(entry)}.")
        bodyIdx = entry[0]
        if bodyIdx < 1 or bodyIdx > nBodies:
            raise ValueError(f"In Force.CG{{{i}}}, body index = {bodyIdx} exceeds the number of bodies ({nBodies}).")
    
    # ---------------------------------------------
    # F1) Check TensionSpring (pairs)
    # ---------------------------------------------
    for i,force_type in enumerate(Force['TensionSpring']):
        if len(force_type) % 2 != 0:
            raise ValueError(f"{force_type} entries should come in pairs, but an odd number was found. Missing points or parameters")

        pointStr,param = force_type
        check_point_string(pointStr, nBodies, Initial_Points)
        check_parameter_has_no_function_of_t(param, force_type, t)
        
        if callable(param):
            # Convert to sympy symbolic
            Symvar                       = sym.Symbol('l')
            param = param(Symvar)
            Force['TensionDamper'][i]   = (pointStr,param)
      
     # ---------------------------------------------
     # F2) Check TensionDamper (pairs)
     # ---------------------------------------------           
    for i,force_type in enumerate(Force['TensionDamper']):
        if len(force_type) % 2 != 0:
            raise ValueError(f"{force_type} entries should come in pairs, but an odd number was found. Missing points or parameters")

        pointStr,param = force_type
        check_point_string(pointStr, nBodies, Initial_Points)
        check_parameter_has_no_function_of_t(param, force_type, t)
        
        if callable(param):
            # Convert to sympy symbolic
            Symvar                       = sym.Symbol('ld')
            param = param(Symvar)
            Force['TensionDamper'][i]   = (pointStr,param)
        
    
    # ---------------------------------------------
    # G) Check TorsionSpring entries
    # ---------------------------------------------
    for i,force_type in enumerate(Force['TorsionSpring']):
        if len(force_type) % 2 != 0:
            raise ValueError(f"{force_type} entries should come in pairs, but an odd number was found. Missing points or parameters")
        
        pointStr,param = force_type
        check_parameter_has_no_function_of_t(param, force_type, t)
        
        if callable(param):
            # Convert to sympy symbolic
            Symvar                      = sym.Symbol('theta')
            param                       = param(Symvar)
            Force['TorsionSpring'][i]   = (pointStr,param)
            
        if (isinstance(force_type, (int, float, sym.Expr)) or hasattr(force_type, '__iter__')):
            try:
                if len(force_type) >= 2:
                    bodyA = float(pointStr[0])
                    bodyB = float(pointStr[1])
                    if bodyA > nBodies or bodyB > nBodies:
                        raise ValueError(f"TorsionSpring: Body index out of range in entry {i}.")
            except TypeError:
                raise ValueError(f"Force.TorsionSpring{{{i}}} must be list-like or callable; got type {type(force_type)}.")
        else:
            raise ValueError(f"Force.TorsionSpring{{{i}}} must be numeric, symbolic, or callable; got type {type(force_type)}.")
    
    # ---------------------------------------------
    # H) Check TorsionDamper entries (similar to TorsionSpring)
    # ---------------------------------------------
    for i,force_type in enumerate(Force['TorsionDamper']):
        if len(force_type) % 2 != 0:
           raise ValueError(f"{force_type} entries should come in pairs, but an odd number was found. Missing points or parameters")
       
        pointStr,param = force_type
        check_parameter_has_no_function_of_t(param, force_type, t)
       
        if callable(param):
           # Convert to sympy symbolic
           Symvar                       = sym.Symbol('thetad')
           param = param(Symvar)
           Force['TorsionDamper'][i]   = (pointStr,param)
            
        if (isinstance(force_type, (int, float, sym.Expr)) or hasattr(force_type, '__iter__')):
            try:
                if len(force_type) >= 2:
                    bodyA = float(pointStr[0])
                    bodyB = float(pointStr[1])
                    if bodyA > nBodies or bodyB > nBodies:
                        raise ValueError(f"TorsionDamper: Body index out of range in entry {i}.")
            except TypeError:
                raise ValueError(f"Force.TorsionDamper{{{i}}} must be list-like or callable; got type {type(force_type)}.")
        else:
            raise ValueError(f"Force.TorsionDamper{{{i}}} must be numeric, symbolic, or callable; got type {type(force_type)}.")
    
    print("Forces validation successful:\t No errors found.")
    print("Points validation successful:\t No errors found.\n")

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:35:32 2025

@author: adiazfl
"""

import sympy as sym

def symvars_definition(varNames,scope_globals):
    """
    Converts each string in the list to ``sympy.Symbol`` and saves it in the global workspace

    Parameters:
    - varNames: list of strings, each a variable name (e.g., ['x', 'y', 'z'])
    
    Returns:
    - symbolicVars: sympy.Matrix column of symbolic variables
    - Also assigns each symbol into the global namespace (like assignin('base', ...))
    """
    if varNames:
        symbolicVars = [sym.Symbol(name, real=True) for name in varNames]
        for name, var in zip(varNames, symbolicVars):
            scope_globals[name] = var
    else:
        symbolicVars = []
    return symbolicVars

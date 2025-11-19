# -*- coding: utf-8 -*-
"""
Automated Validation of Multibody Examples
==========================================

This script runs through a suite of predefined multibody system examples,
computes key symbolic outputs via the public `MBDSystem` class, and then
compares them against “ground-truth” expressions stored as JSON.

For each Example 1–8, it:

  1. Imports the example via `Examples.Example{i}`  
  2. Constructs the multibody system with `MbdSystem.from_example(...)`  
  3. Loads the expected results (Pos, JointLoc, R, RD) from `TrueMBDvars/Example{i}.json`  
  4. Symbolically computes the difference, substitutes unit values, and sums absolute errors  
  5. Prints per-example error and a final summary list  

Usage
-----
Run on the command line (from the project root):

    python automated_validation.py

Or import and call individual functions (`load_expected`, `compare`) from another script.

"""

import importlib, json, sympy as sym
import sys, os

# add parent directory (project root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the multibody system class
from multibody import MbdSystem            # new public class


############ Beginning of the file ############
def load_expected(path):
    with open(path) as f:
        data = json.load(f)
    return { name: sym.sympify(expr, evaluate=False) for name, expr in data.items() }

def is_not_nan(x):
    return not (x is sym.nan or x.has(sym.nan))

def compare(m_computed, m_expected):
    diffVar     = sym.simplify(m_computed - m_expected)
    subs        = { s: 1 for s in diffVar.free_symbols }
    diffVar_sub = diffVar.subs(subs).applyfunc(abs)

    # flatten and filter out NaNs
    flat_entries = [x for x in diffVar_sub if is_not_nan(x)]

    return float(sum(flat_entries))


if __name__=="__main__":
    summary = []
    for i in range(1,8):
        # 1) import the example data
        print(f"\n##### For Example {i} #####")
        
        # 1- Example to import 
        ex = importlib.import_module(f"Pytests.Example{i}")
        
        # 2- Initialize the Multibody system 
        MBDsys  = MbdSystem.from_example(ex) 
        
        # 3) compare
        Truth = load_expected(f"TrueMBDvars/Example{i}.json")
        errs = [
            compare(MBDsys.Pos,      Truth["Pos"]),
            compare(MBDsys.JointLoc, Truth["JointLoc"]),
            compare(MBDsys.R,        Truth["R"]),
            compare(MBDsys.RD,       Truth["RD"])
        ]
        total = sum(errs)
        summary.append(total)
        print(f"Example {i}: error = {total:.3e}")
        print(20*'#')
        print('End of example')
        print( 20*'#'+'\n')
        
        
    print("\nSummary:", summary)
    
    
    
# Write to matlab: the following lines allow you to export the variables both to .mat and .json to recreate the True values
# Convert SymPy matrices to DataFrames for better visualization
# df_pos = pd.DataFrame(Pos.tolist(), columns=["Pos"])
# df_jointloc = pd.DataFrame(JointLoc.tolist(), columns=[f"JointLoc_{i+1}" for i in range(JointLoc.shape[1])])
# df_R = pd.DataFrame(R.tolist(), columns=[f"R_{i+1}" for i in range(R.shape[1])])
# df_RD = pd.DataFrame(RD.tolist(), columns=[f"RD_{i+1}" for i in range(RD.shape[1])])

# data_dict = {
#     'Pos_python': np.array([[str(val) for val in row] for row in Pos.tolist()], dtype=object),
#     'JointLoc_python': np.array([[str(val) for val in row] for row in JointLoc.tolist()], dtype=object),
#     'R_python': np.array([[str(val) for val in row] for row in R.tolist()], dtype=object),
#     'RD_python': np.array([[str(val) for val in row] for row in RD.tolist()], dtype=object)
# }

# # Save files as .mat for validation
# savemat('Validation/' + file2save+'.mat', data_dict)
# print("Data successfully saved as 'output_data.mat'")
# data = {
#     "Pos":   sym.srepr(Pos),
#     "JointLoc": sym.srepr(JointLoc),
#     "R":     sym.srepr(R),
#     "RD":    sym.srepr(RD),
# }

# with open("Validation/" + file2save + ".json","w") as f:
#     json.dump(data, f, indent=2)
    
# exit()

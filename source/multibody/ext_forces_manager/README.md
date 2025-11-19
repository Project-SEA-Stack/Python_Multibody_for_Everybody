This wrapper code provides an interface between the multibody solver and MoorDyn-C. 

It is currently only set up to couple via MoorDyn point objects, with rigid body motion handled by the multibody code. 

Other MoorDyn objects can be used in the MoorDyn simulation (i.e. bodies and rods), but must not be the type `coupled` or `vessel`.

The MoorDyn-C source code can be found at: https://github.com/FloatingArrayDesign/MoorDyn

The documentation can be found at: https://moordyn.readthedocs.io/en/latest/ 
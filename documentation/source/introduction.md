<a id="introduction"></a>

# Executive Summary

Welcome to **Multibody4Everybody**, a symbolic multibody dynamics engine designed to be accessible, efficient, and extensible. 
Multibody dynamics (MBD) allows us to model mechanical systems composed of interconnected rigid (or flexible) bodies.

These systems are common in robotics, biomechanics, aerospace, marine engineering, and many other fields. However, many existing 
tools for MBD are either closed-source, difficult to extend, or require advanced knowledge to use effectively.

This project was developed to provide a **user-friendly and symbolic multibody framework** that simplifies the modeling process 
while retaining full flexibility and mathematical transparency. Additionally, this code is an efficient tool for design exploration, 
controls, and optimization.

At a glance, this is what is possible with Multibody4Everybody design optimization:

```{figure} _static/figs/MBD_capabilities_workflow.jpg
:width: 100%
:align: center
```

## Why a New Multibody Tool?

Several open-source MBD libraries already exist:

- **Chrono** is a powerful C++-based library with many add-on modules, but it does not provide symbolic equations of motion or use reduced-coordinate formulations.
- **PyDy** (Python Dynamics) uses Kane’s method and offers symbolic equation generation—but it often requires deep familiarity with multibody theory and SymPy mechanics.

In contrast, **Multibody4Everybody** is built for:

- **Ease of use**, even for those new to MBD  
- **Symbolic generation** of the equations of motion (EOMs) for design and optimization  
- **Efficient simulation** using minimal (joint) coordinates  
- **Plug-and-play interfaces** for external forces, including libraries like MoorDyn  

## How Multibody4Everybody Works

To use the code, all you need is a single input file. In short, you define the kinematics of the system (a snapshot at its initial condition) 
and the forces and points where forces are applied.

This file contains structured information about your mechanical system:

1. **Define the reference frame and bodies**  
   - Specify the location of the global origin (the y-axis is perpendicular to the working plane).  
   - Define the bodies and how they are connected (parent–child relationships).  
   - Define the center of mass (CG) of each body and the location of each joint relative to the CG (in the body-fixed frame).  
   - Describe the type of joint connecting each body (e.g., revolute, prismatic, floating).

2. **Body Points (Optional)**  
   - Add **body points**, defined in the body-fixed frame and relative to the CG of the corresponding body, which move with the body.  
   - Add **ground points**, in the global frame, which remain fixed. These can later be used for applying forces or constraints.

3. **External Forces (Optional)**  
   - Apply forces or torques to the CG or to body points defined above.  
   - Add **springs**, **dampers**, or **mooring lines** using a consistent format. The springs and dampers can be linear or nonlinear.  
   - Mooring systems can be defined via **MoorDyn**, using the same interface style as springs and dampers.

4. **Numerical Parameters**  
   - Provide initial conditions (positions and velocities).  
   - Set body properties like mass and inertia.  
   - Define simulation parameters: gravity, total simulation time, and time step.

The numerical parameters step is needed to simulate the system over time. The parameters defined above could have been symbolic or numeric—once 
everything is defined, the code can generate and simulate the system, animate its motion, and optionally save it as a GIF.

---

## Validation and Help for New Users

Each example included in the package contains the structure described above, along with:

- **Automatic checks** to validate that joints, points, and forces are defined consistently.  
- **Table displays** for joints, points, and forces—allowing visual inspection of each section.  
- **Initial condition guidance**, showing which variables require starting values, even if you're unfamiliar with the joint coordinate method.

For clarity, each example ends with a short reference section explaining the expected format for each field, including valid values and data types. 
This documentation is designed to help both new and advanced users define systems correctly.

## Symbolic or Numeric—Your Choice

A key strength of this package is flexibility: any quantity you define can be symbolic (for symbolic derivations) or numeric (for direct simulation). 
You can start simple and build complexity as needed.

---

This introduction is intended to help you get started quickly and confidently. If you're new to multibody modeling, don't worry—the examples and built-in checks will guide you.

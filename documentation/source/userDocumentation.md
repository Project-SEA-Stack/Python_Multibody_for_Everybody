# User documentation
This chapter aims to help understand the M4E 2D code usage. 
This is not the documentation for developers, but rather teaches how to define
new examples to use the tool without knowledge of the math or what happens in the
backend. What you can expect from this chapter is to learn how to define the 
multi-body system either using numeric values or symbolic values and how to loop
over multiple symbolic parameters. 

You will learn how to define simple bodies connected by *Float*, *Revolut*, or
*Prismatic* joints. You will learn how to understand the visualization of your
defined example, what are the degrees of freedom associated with each joint briefly
and how to animate your simulation (the cool part).

## How to Write an Example
The goal of the examples section is to build knowledge in a dynamic fashion. For that reason,
we have decided to increase the explanations step-by-step and the amount of information each 
example contains increases as you progress in this tutorial. 

This section is organized into three main steps:

1. **Quick examples overview** explain how to set your example file without going into detailed explanation of 
what each variable means. 
2. **Detailed explanation examples**, where bodies, points and forces definition are tackled in detailed. 
3. **Validation** of the equations of motion (EOM) comparing results to a benchmark example.

## Quick examples overview:

### Example 1: Double pendulum
The double-pendulum example is a multibody dynamics benchamark, which we use to validate the framework. 
For the analytical solution refer to <span style="color: red;">Reference to theoretical paper</span>. 
The following image represents the sketch of a double pendulum following the notation from the publication
above. This is example number 1 from the default examples. 

<!-- ![Double pendulum Representation](/_static/figs/DoublePendulum_sketch.jpg) -->

```{figure} _static/figs/DoublePendulum_sketch.jpg
:width: 30%
:align: center
```

#### **1 - Define origin of inertial frame of reference (f.o.r)**
This section allows for origin offset, but how to use it is tackled in the next example. 

```python
# Example 3
Reference_frame_Origin = np.array([0,0])
```

#### **2 - Bodies definition**

***a. Symbols for bodies (optional)***

If you’ll define symbolic body variables, initialize them first. 
Otherwise, you can leave this lists empty :

```python
# Bodies definition
dataNames   = []    # Define symbolic variables 
BodyDataSym = symvars_definition(dataNames,globals())
```

***b. Joint connectivity***

Define lists for all the fields below:

```python
joints              = [[0, 1],[1, 2]]   # Joint connectivity: [parent, child]
types               = ['R', 'R']        # Joint types: 'R' for revolute, 'P' for prismatic, 'F' for floating
parent_cg_to_joint  = [[3, 0],[0,1]]    # Vectors from parent's center-of-gravity (CG) to the joint location.
joint_to_child_cg   = [[0, 2],[0,4]]    # Vectors from the joint to the child's CG.
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan]] # For prismatic joints, the direction vector; for others, [nan, nan] is used.
prismatic_direction = normalize_prismatic(prismatic_direction)
```

#### **3 - Points definition**
This section allows to define additional points on top of the CG points and joints. 
For now we leave it blank. Even if not used, these fields must be declared empty.

```python
# Points definition
PointNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
PointsSym   = symvars_definition(PointNames,globals())

# Define the structure for initial points.
Initial_Points          = {}
# Ground points: these are fixed and given as [x, z] coordinates.
Initial_Points["GR"]    = []

# Body-defined points ("BD"): we use a dictionary where each key (an integer)
# corresponds to a body and the value is a list of points.
Initial_Points["BD"]    = {}
```

#### **4 - Forces definition**
This section allows to define forces at specific location (CG, joints, additional points) and between bodies. 
For now we leave it blank. Even if not used, these fields must be declared empty.

```python
# Forces definition
ForceNames  = [] # Define symbolic variables for force points (for example, b1, b2, b3).
ForceSym    = symvars_definition(ForceNames,globals())

# Define the structure for Force.
Force = {}
# Forces applied on body-defined points (PointsBD):
Force["PointsBD"] = []

# Forces applied at the center of gravity (CG):
Force["CG"] = []

# Tension springs: stored as a list of tuples: (connection, [l0, stiffness]).
# Here "BD41" means (for example) the 1st point on body 4.
Force["TensionSpring"] = []

# Tension dampers: here we have two pairs.
# The first pair uses a constant damping coefficient.
# The second uses a lambda function to represent a nonlinear damping coefficient.
Force["TensionDamper"] = []

# Torsion springs: here the first element is a list of parameters and the second element is a lambda.
Force["TorsionSpring"] = []

# Torsion dampers:
Force["TorsionDamper"] = []

ForcesPointsSym = PointsSym + ForceSym 
```

#### **5 - Define numerical values**

Now that joints, points, and forces are defined, you need to provide numerical values for:

- Initial conditions (positions/velocities)
- Mass & inertia of each body
- Gravity value (default 9.81)
- Gravity vector (percentage of “g” per body)
- Simulation time & timestep
- Animation options

***a. Initial Conditions (ICs)***

When bodies are initially defined, each body is given a body frame of reference which is **parallel**
to the global frame of reference. The joint-coordinates (explained later) are initialized to zero
at this initial configuration. This means that if we define our pendulum vertically it is not going 
to move, but otherwise it will and the initial conditions are defined as zero at that initial snapshot.

The second set of numerical values to initialized is ``ForcesPointsNum`` and ``BodyDataNum``. These variables 
give a numerical quantity during time integration to quantities originally defined as symbolic. 
By default we always give it a value of 1 and they must always exist, even if unused. 

```python
# Initial conditions
# Create the JointSystem using the from_data class method.
joint_system        = JointSystem.from_data(joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction) # DO NOT MODIFY
Q, QD, _, NDOF, _   = joint_system.coordinate_finder() # DO NOT MODIFY

# If you are unsure what are the systems DOF run the example and you will see on screen
ic                  = np.zeros(2*sum(NDOF)) # Multiplied by 2 because is position and velocity

# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))
```

***b. Time & Simulation parameters***

Here we define values such as the mass and inertia of our bodies, the value of gravity (optional)
and the percentage of gravity applied to each body (yes, not all bodies need to be subject to the
same gravity value, imagine a body is floating in water and others are not). We also define for how
long we want to run our simulation and how big the time between integration steps will be.  

```python
# Simulation
TimeStep    = 0.05
tspan       = 5.
g           = 9.81                      # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = np.ones(len(types))
```

***c. Animation parameters***
Very straight forward, do you want to visualize the animation? Yes (1) or no(0)?. Do you want to save
it as gif? None for no and string name as 'name.gif' if you do. Lastly, is the animation too slow? Just
speed it up by specifying how many time steps to skip (natural numbers only) with the variable ``plotTstep``.

```python
# Animation
animation_on    = 1     # Display animation:1, 0-otherwise
SaveMovieOn     = None  # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 5     # how many times faster are we plotting w.r.t to simulation time
```

#### **6 - call from main**

Having defined the example it only remains importing it in the main file a running it.
To import the double flap example make sure the line in main where the Examples folders 
is, looks like this:

```python
from Examples import _1_Double_pendulum as ex
```

Then just run the program! You should see a plot for the mechanical energy, which should be constant,
and a plot with the initial snapshot. The initial snapshot should look like this:

<!-- ![Double pendulum t0](/_static/figs/DoublePend_t0.png) -->

```{figure} _static/figs/DoublePend_t0.png
:width: 30%
:align: center
```

### Example 2: Spiderfloat
The Spiderfloat device (developed under ARPA-E ATLANTIS) is an offshore-wind 
platform designed to produce 1,200 GW of green energy. To see more, visit the 
[NREL Spiderfloat page](https://www.nrel.gov/news/detail/program/2019/spiderfloat-innovation). 
This example has reduced explanation on what each element to be defined is and focuses more on
a general overview on how to set it up.

<!-- ![Spiderfloat Representation](/_static/figs/spiderfloat.jpg) -->

```{figure} _static/figs/spiderfloat.jpg
:width: 100%
:align: center
```

#### **1 - Define origin of inertial frame of reference (f.o.r)**

The user can choose the value of gravity and the origin of f.o.r which will 
influence how the bodies appear in the global frame of reference. 
If ``Reference_frame_Origin`` is provided other than ``[0,0]``, then the 
origin will be located at ``Reference_frame_Origin`` from the point `[0,0]`.
Then the coordinates `parent_cg_to_joint` for the float are measured from
this ``Reference_frame_Origin`` point. 

```python
# Spiderfloat example
Reference_frame_Origin = np.array([10,5])
```

#### **2 - Bodies definition**

***a. Symbols for bodies (optional)***

If you’ll define symbolic body variables, initialize them first. 
Otherwise, you can leave this lists empty :

```python
# Bodies definition
dataNames   = []    # Define symbolic variables 
                    # for force points (for example, b1, b2, b3).
BodyDataSym = symvars_definition(dataNames,globals())
```

***b. Joint connectivity***

Define lists for:

- `joints`: each entry is [parent_body, child_body]

- `types`: corresponding joint type for each pair (`'R'`, `'P'`, or `'F'`)

- `parent_cg_to_joint`: vector from parent’s CG to the joint location

- `joint_to_child_cg`: vector from joint to the child’s CG

- `prismatic_direction`: direction of translation for prismatic joints (set to `[np.nan, np.nan]` for `'R'` or `'F'` joints).

```python
# (a) Joint connectivity: [parent, child]
joints = [
    [0, 1],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5]
]

# (b) Joint types: 'R' = Revolute, 'P' = Prismatic, 'F' = Floating
types = ['F', 'R', 'R', 'R', 'R']

# (c) Parent_CG_to_Joint (meters)
parent_cg_to_joint = [
    [0, 1],     # Parent = 0 to Joint 0
    [0.5, -2],    # Parent = 1 to Joint 1
    [-0.5, -2],   # Parent = 1 to Joint 2
    [4.5, 0],     # Parent = 2 to Joint 3
    [-4.5, 0]     # Parent = 3 to Joint 4
]

# (d) Joint_to_Child_CG (meters)
joint_to_child_cg = [
    [np.nan, np.nan],  # For 'F' joint, use [nan, nan]
    [5, 0],            # Joint 1 to Child 2’s CG
    [-5, 0],           # Joint 2 to Child 3’s CG
    [0, 1.5],          # Joint 3 to Child 4’s CG
    [0, 1.5]           # Joint 4 to Child 5’s CG
]

# (e) Prismatic direction vectors (only for 'P' joints; else use [nan, nan])
prismatic_direction = [
    [np.nan, np.nan],
    [np.nan, np.nan],
    [np.nan, np.nan],
    [np.nan, np.nan],
    [np.nan, np.nan]
]

# Normalize the prismatic direction (no effect on [nan, nan])
prismatic_direction = normalize_prismatic(prismatic_direction)
```

*Notes:*

- *Order matters. Parent bodies should be already defined for every new joint you create. 
For example, if you haven't defined body 3 before, you can't say body 3 is parent of body 4
- *For floating joints ('F'), set their `joint_to_child_cg` vectors to [np.nan, np.nan].*


#### **3 - Points definition**

Points let you apply forces at specific locations. There are two categories:

- **Ground points** (`"GR"`): fixed in model coordinates. The coordinates are relative
to ``Reference_frame_Origin`` to be consistent with the rest of the framework and other
tools such as Project Chrono.
- **Body-defined points** (`"BD"`): move with their respective body

Initialize the point dictionaries `Initial_Points` (even if you don’t use them):

```python
# Symbolic point names (if needed):
PointNames = []             
PointsSym  = symvars_definition(PointNames, globals())

# Initialize containers
Initial_Points = {}
Initial_Points["GR"] = []   # Ground points (list of [x, y] coordinates)
Initial_Points["BD"] = {}   # Body points (dict: body_index → list of [x, y] for that body)
```

***a. Define ground points***

```python
# Example: two ground points at (20, 0) and (0, 0)
Initial_Points["GR"] = [
    [20, 0],
    [0, 0]
]
```

***b. Define body points***

```python
# Example: Body 1 has four points; Body 2 has two; Body 3 has two
Initial_Points["BD"][1] = [
    [0.5, -2.5],
    [-0.5, -2.5],
    [0.5, 5],
    [-0.5, 5]
]
Initial_Points["BD"][2] = [
    [4, 0],
    [4.5, 0]
]
Initial_Points["BD"][3] = [
    [-4, 0],
    [-4.5, 0]
]
```

*Remember: Even if you don’t need ground points or body points, you must still initialize:*

```python
Initial_Points["GR"] = []
Initial_Points["BD"] = {}
```

#### **4 - Forces definition**

Initialize symbolic force names (if needed):

```python
ForceNames = []
ForceSym   = symvars_definition(ForceNames, globals())

Force = {}
```
Then define each force category. Even if a category has no entries, assign an empty list so 
that the “checks” run consistently (especially if you’re converting from MATLAB, which might 
implicitly remove empty variables).

***a. Point forces applied to the body-defined points***

```python
   # Each entry: [body_index, point_index, Fx, Fz, My]
Force["PointsBD"] = [
    # Example: apply a moment of 1 at Body 4’s 1st point [4, 0, 0, 0, 1] (no forces here for Spiderfloat)
]
```

***b. Forces applied to the body centers of gravity (CG)***

```python
# Each entry: [body_index, Fx, Fz, My]
Force["CG"] = [
    # Example: [2, 5, 0, 0] applies Fx=5 N on Body 2’s CG
]
```

***c. Tension springs***

```python
# Each entry is ((pointA, pointB), [l0, stiffness])
# pointA/pointB are strings like "BD10" = Body 1 Point 0, "GR01" = Ground Point 1
Force["TensionSpring"] = [
    (("BD10","BD20"), [7.0139, 1e0]),
    (("BD11","BD30"), [7.0139, 1e0]),
    (("BD12","BD21"), [11.8004, 1e-1]),
    (("BD13","BD31"), [11.8004, 1e-1]),
    (("BD20","GR00"), [3.0414, 1e-1]),
    (("GR01","BD30"), [3.0414, 1e-1])
]
```

***d. Tension dampers***

```python
# Each entry: ((pointA, pointB), damping_coefficient)
Force["TensionDamper"] = [
    (("GR00","CG22"), 3),
    (("GR01","CG33"), 3)
]
```

***e. Torsion springs & dampers***

If you have torsion springs or torsion dampers, add them—even if empty—to keep the structure consistent:

```python
Force["TorsionSpring"] = [
    # ((bodyA, bodyB), [theta0, k_theta])
]

Force["TorsionDamper"] = [
    # ((bodyA, bodyB), c)
]
```

Key point:

- `PointsBD` and `CG` use lists of lists (each inner list = one force point).
- Springs and dampers use lists of tuples.


#### **5 - Define numerical values**

Now that joints, points, and forces are defined, you need to provide numerical values for:

- Initial conditions (positions/velocities)
- Mass & inertia of each body
- Gravity value
- Gravity vector (percentage of “g” per body)
- Simulation time & timestep
- Animation options

***a. Initial Conditions (ICs)***

You can either manually specify every initial position and velocity if you are already familiar
with joint coordinates, or use `coordinate_finder()` to find all the positions and velocity that 
must be initialized.

We will explain later in more detail, but for now just know that when we have a float joint, the 
position coordinates `(X,Z)`, must match `parent_cg_to_joint` corresponding to the float, and if 
it doesn't, the code will automatically adjust those initial conditions while displaying a 
warning. 

```python
# Build the JointSystem (DO NOT modify)
joint_system = JointSystem.from_data(
    joints,
    types,
    parent_cg_to_joint,
    joint_to_child_cg,
    prismatic_direction
)

# Extract coordinate info (DO NOT modify)
Q, QD, _, NDOF, _ = joint_system.coordinate_finder()
```

- `NDOF` is an array that gives the number of degrees of freedom for each joint.
- The total number of position variables is `sum(NDOF)`.
- Because we need both positions and velocities, the total IC length is `2 * sum(NDOF)`.

If you’re unsure, initialize everything to zero:

```python
# Zero initial conditions (positions + velocities)
ic = np.zeros(2 * sum(NDOF))
```

You can then modify `ic[...]` if you want specific starting position/velocities.

***b. Symbolic Parameters to Loop Over (optional)***

If you want to explore multiple force or body parameters, define arrays of symbolic values: 

```python
# For example, if ForceSym and BodyDataSym are nonempty:
ForcesPointsNum = np.ones(len(ForceSym))  # Start with ones
BodyDataNum     = np.ones(len(BodyDataSym))  # (modify later if needed)
```

*At the moment, we initialize them to ones (no effect). You can expand this later to loop over parameter sets.*
<span style="color: red;">This portion still requires an example</span>

***c. Time & Simulation Parameter***

```python
TimeStep = 0.05        # [s] integration interval
tspan    = 10.0        # [s] total simulation time (final time)
```

***d. Gravity per Body***

- `g`: global gravity constant (e.g., `9.81`)
- `gVec`: an array of size `(n_bodies, 1)` specifying the percentage of gravity each body experiences. For example, if you want zero gravity on Body 2, set `gVec[1] = 0.0`.


```python
g    = 9.81                                 # standard gravity [m/s²]
gVec = np.ones((len(types), 1))            # 100% gravity on each body
```

***e. Mass & Inertia***

Provide a 1-D NumPy array for each:

- `m0`: mass of each body
- `J0`: moment of inertia of each body (about its CG)

```python
m0 = np.ones(len(types))        # Example: all masses = 1 kg
J0 = 2.0 * np.ones(len(types))  # Example: all inertias = 2 kg·m²
```

***f. Animation Options***

These flags control whether you see an on-screen animation and/or save a GIF:

- `animation_on`: 1 to display, 0 to skip
- `SaveMovieOn`: set to None (no file) or e.g. `"Spiderfloat.gif"`
- `plotTstep`: draw every `plotTstep`-th frame (for speed)


```python
animation_on = 1     # 1 = show animation; 0 = skip
SaveMovieOn  = None  # "Spiderfloat.gif" to save, or None to skip
plotTstep    = 5     # Plot 1 frame every 5 simulation frames
```

#### **6 - call from main**

Having defined the example it only remains importing it in the main file a running it.
To import the spiderfloat example make sure the line in main where the Examples folders 
is read looks like this:

```python
from Examples import spiderfloat as ex
```

Then the multibody system is created with the lines:

```python
# 1- Initialize the Multibody system 
MBDsys  = MbdSystem.from_example(ex) 
```

The next step is to integrate over time. For that we need to define our variable containing
all numeric values as:

```python
# 3 - Define initial numerical values
mainNumVars = np.hstack((ex.ic, ex.ForcesPointsNum, ex.BodyDataNum))
```

Then we can integrate the system

```python 
# 3 - Integrate over time
sol     = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, 
                           tspan=ex.tspan, dt=ex.TimeStep,  external_manager=None) 
```

If everything went correctly, your first frame should look like this:

<!-- ![Spiderfloat animation](/_static/figs/Spiderflaot.png) -->

```{figure} _static/figs/Spiderflaot.png
:width: 80%
:align: center
```

## Detailed examples

### Example 1: FOSWEC

To model a multibody system, you need to define a kinematic sketch of the system at its initial condition.
These are the steps we use to create the sketch of the system:

1. Stablish the model's origin (Default is `[0,0]`)
2. Defining the joints of the system (starting with joint 1 which always connects body 1 to ground via prismatic, revolute, or floating joint). 
3. Each joint has a parent and a child (for instance for joint 1, parent is always the ground and its child is always body 1).  Then, for each joint, 
we define two vector: one vector connects the joint location to parent CG and the other vector connects the joint to child's CG. 
If we do this for all the joints, we have defined a kinematic sketch of the system.

There are a few exceptions:
- For instance, for joint 1, the parent is body 0 (ground) and it does't have a CG. 
- CG of ground is the inertial frame. Basically, the parentCG to joint 1 shows the location of joint 1. 


This example shows how the FOSWEC dynamics model can be created using our proposed code and how correct implementation is verified. 
Additionally we show the joints DOFs associated to the example. The equations of motion are provided symbolically using joint coordinates (reduced coordinates). 

<!-- ![FOSWEC sketch](/_static/figs/Foswec_sketch.png) -->

```{figure} _static/figs/Foswec_sketch.png
:width: 75%
:align: center
```

#### **1 - Define origin of inertial frame of reference (f.o.r)**

This time we define the origin of the frame of reference 0.91 units underneath the center of mass
of body 1. The variable `Reference_frame_Origin` moves the model origin from the point `[0,0]` by
the vector `Reference_frame_Origin`. Then, when setting the `parent_cg_to_joint` variable to `[0,0]`
it is goint to be coincident with `Reference_frame_Origin` becuase `parent_cg_to_joint` is a vector 
relative to the model frame.

```python 
Reference_frame_Origin = np.array([0,0.91])
```

#### **2 - Bodies definition**

This time we are presenting the result we achieve to aim when defining the bodies in table format
before actual definition, and if the user feels comfortable it is recommended that you try it out
before seeing the solution. 

<!-- ![FOSWEC bodies definition table](/_static/figs/Foswec_bodies.png) -->

```{figure} _static/figs/Foswec_bodies.png
:width: 100%
:align: center
```

This table is automatically generated when running the example FOSWEC.py either as standalone or from
main. To obtain such table we must define the bodies as follows

```python
joints              = [[0, 1],[1, 2],[1,3]]                                 
types               = ['F','R', 'R']                                        
parent_cg_to_joint  = [[0, 0],[0.64, 0],[-0.64, 0]]                         
joint_to_child_cg   = [[np.nan, np.nan],[0, 0.17],[0, 0.17]]                
prismatic_direction = [[np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan]]  
prismatic_direction = normalize_prismatic(prismatic_direction)              
```

The key point between the bodies definition and how it is displayed as table is that one is the
transposed of the other. This means that if we read the information of the code vertically, i.e.
column by column, that information will correspond to each row of the table. 

Each row of this table represents a kinematic joint. The entire kinematic structure is built by defining the joints and their associated vectors. The types of joints that can be modeled include: Floating Joint (F), Prismatic Joint (P), and Revolute Joint (R).

Let’s go over **the first row**. Each joint has a parent body and a child body. Each row defines how the joint is connected to the child’s center of gravity (CG) and the parent’s CG. This is how the entire kinematic structure is defined.

- The **first column** indicates the **parent** of the joint, and the **second column** indicates the **child**.
- The parent of the first defined joint is always set to **zero**, which represents the ground. The child body, in this case, is **body 1**.
- The **third column** specifies the joint type. For body 1, which is connected to the ground, this is a **floating joint** (representing the platform's base).
- The vectors in the **fourth and fifth columns** represent the [x,z] components of the vector from the ground (reference frame) to the CG of the floating body.
- The **sixth and seventh columns** should be set to `[NaN,NaN]` for floating joints.
- The **last two columns** are used only for prismatic joints to define the initial direction of motion and should be set to `[NaN,NaN]` for all non-prismatic joints.

**The second row** of the table indicates that a **revolute joint** connects **body 1** (the parent) to **body 2** (the child). Note that body 1 was already defined in the first row.

- The **third and fourth columns** specify the vector from the **CG of body 1** to the **revolute joint**. This joint is located at `[0.64,0]` meters relative to the CG of body 1.
- The **fifth and sixth columns** specify the vector from the **revolute joint** to the **CG of body 2**, which is `[0,0.17]`.

In summary, this row defines the connection between the platform’s base and the **right flap**.

The **third row** indicates that a **revolute joint** connects **body 1** (the parent) to **body 3** (the child).

- The **third and fourth columns** define the vector from the **CG of body 1** to the **revolute joint**, which is located at `[−0.64,0][-0.64, 0][−0.64,0]` meters.
- The **fifth and sixth columns** define the vector from the **revolute joint** to the **CG of body 3**, which is `[0,0.17][0, 0.17][0,0.17]`.

In summary, this row adds the **left flap** to the platform’s base.

#### **3 - Joint coordinates DOFs**

The joint coordinates associated to this example are extracted at the initial conditions section, but 
for the sake of explaining a bit the coordinates associated with each type of joint we moved it here.

```python
Q, QD, _, NDOF, _   = joint_system.coordinate_finder()
```

Running this line requires creating the `joint_system` but that is already in the line above in the code
and it's not important at the moment. If we print on screen these coordinates we get a set of degrees of 
freedom with repeated subindices, sometimes. Each subindex, points to a different body. Therefore if a
subindex is repeated it means that the joint type has multiple DOFs associated to a body. For our example it looks
like this:

```python
Q'   = [  X_1,   Z_1,   Theta_1,   Theta_2,   Theta_3]
```

There are three subindices: 1, 2, 3. This means that we have three bodies. The first body was defined 
as a floating body or joint. Floating joints have in 2D three degrees of freedom, two translations:
`X,Z` and one rotation $\theta$. The second and third bodies are connected to the first one through 
revolut joints, and this joint have one rotational degree of freedom: $\theta$. So in summary, $[X_1, Z_1]$ show the position of the CG of the platform and $[\theta_1, \theta_2, \theta_3]$ show the pitch angle of the platform, the first flap, and the second flap, respectively.

#### **4 - Points definition**

To define springs and dampers associated to the the bodies, unless these points are either joints or 
the center of gravity, we must define them on top of the bodies definition. As mentioned in the previous
example, if the bodies are fixed we define them as ground points, `GR`, whereas if this points are on
a body, we define them as body points, `BD`. These points are always relative to their own frame of reference,the model frame defined by ``Reference_frame_Origin`` for `GR`, and the body frame for `BD`. The definition of the points for FOSWEC is summarized in the table below and is used to verify we have defined everything as we wanted.

<!-- ![FOSWEC points definition table](/_static/figs/Foswec_pointTable.png) -->

```{figure} _static/figs/Foswec_pointTable.png
:width: 50%
:align: center
```

- Column 1: body to which the point belongs. Ground is always associated to 0.
- Column 2: Point type
- Column 3: Order in which the point was defined in a body
- Column 4: x-coordinate relative to the CG
- Column 5: y-coordinate relative to the CG

This points are defined in using the following lines of code:

```python
# Ground or fixed points
Initial_Points["GR"]    = [[0.75, -0.91],[-0.75, -0.91]]
# Body associated points
Initial_Points["BD"]    = {}
Initial_Points["BD"][1] = [[0.75, 0],[-0.75, 0]]
```

The variable `Initial_Points` is a dictionary with two entried:

- `GR` for ground points 
- `BD` for body points

The entry `GR` is a list containing the relative coordinates of each point in an inertial frame of reference, defined by `Reference_frame_Origin`. On the other hand, the entry `BD` is another dictionary, which entries refer to the additional points each body has. Each dictionary entry, contains a list of points. In this example, only body 1 has associated points and it has two points. The coordinates entered for the `Initial_Points['BD']` are **relative to the CG**. Thus the first point in body 1 is located 0.75 units to the right of the body 1 CG.

After defining these popints, we can access them at the code. For instance point `[0.75, 0]`, the first point defined in body 1 is accessed
as `'BD10'` and point `[-0.75,0]` defined on body 1 also is accessed as `'BD11'`

#### **5 - Forces definition**

For the forces defintion section, we are goint to introduce a new concept, defining symbolic forces. To define symbolic variables in section, we must define the variables as a list of strings in their corresponding section. *Note: if a variable is repeated, define it only once*. In this example we define the time as symbolic variable, `t`, and the springs stiffness, `k`. The string name is defined under the variable `ForceNames` and the function `symvars_definition` converts those names into variables and places them in the global workspace. 

```python
ForceNames  = ['t','k'] 
ForceSym    = symvars_definition(ForceNames,globals())
```

The final result will look like this:

<!-- ![FOSWEC forces definition table](/_static/figs/Foswec_forceTable.png) -->

```{figure} _static/figs/Foswec_forceTable.png
:width: 100%
:align: center
```

The explanation for each of the entries is divided into two groups. One point forces PointsBD and CG
which are defined as lists of lists and the two points forces which are defined as list of tuples.

- PointsBD: [lists of lists] where each list corresponding to one point contains

    1. Body of application i, which corresponds to the dictionary entry in Initial_points.
    2. Element in the list j of Initial_points[i].
    3. $F_x$: force in x-direction
    4. $F_z$: force in z-direction
    5. $M_y$: moment in y-direction

In this example PointsBD is defined as empty, but must always be defined for consistency with MATLAB code.

```python
Force["PointsBD"] = []
```

- CG:  [lists of lists] where each list corresponding to one point contains
    1. Body of application i, which corresponds to the dictionary entry in Initial_points.
    2. $F_x$: force in x-direction
    3. $F_z$: force in z-direction
    4. $M_y$: moment in y-direction

With the time variable we are going to define an exponential force at the CG of body 1, whith only vertical component. This force is an exponential decay function $F_y = e^{-t}$. 

```python
Force["CG"] = [[1,0,0.1*sym.exp(-t),0]]
```

- TensionSpring: list of tuples  where each tuple contains
    1. Tuple of strings that refer to the two points connected by the spring
    2. List of parameters containing [$l_0$: undeformed length, $k$: stiffness constant *if linear spring (will be clarified later)*]

In this example we are connecting the first ground point, `'GR00'`, with the first point in body 1, `'BD10'`. These points were already defined in the dictionary `Inital_Points`. Then we define the unstreched length, $l_0$ as 5, and the stiffness, $k$, will be symbolic.

```python
Force["TensionSpring"] = [(('GR00','BD10'),[5,k]),
                          (('GR01','BD11'),[5,k])
                            ]
```

- TensionDamper: list of tuples where each tuple contains
    1. Tuple of strings that refer to the two points connected by the damper
    2. float containing $c$: damping coefficient *if linear spring (will be clarified later)*

Tension dampers are defined in a very similar manner to tension springs. The points are defined in the same manner, however, when defining the parameters, instead of passing a list we pass a float corresponding to the damping coefficient. 

```python
Force["TensionDamper"] = [(('GR00','BD10'),1),
                          (('GR01','BD11'),1)]
```

- TorsionSpring: list of tuples where each tuple contains
    1. Tuple of strings or floats that refer to the two bodies connected by the damper
    2. list containing [$\theta_0$: undeformed angle, $k_\theta$: stiffness]

Torsion springs follow the same structure as Tension Springs, however, instead of defining the points connected we defined the bodies connected as this springs are only conceived to be at the joints between bodies. These points can be defined as a `tuple` of two `floats` or two `strings`. The parameters are defined in the same fashion as tension springs. 

```python
Force["TorsionSpring"] = [((1,2),[0,10]),
                          ((1,3),[0,10])]
```

- TorsionDamper: list of tuples where each tuple contains
    1. Tuple of strings or floats that refer to the two bodies connected by the damper
    2.  float containing $c$: damping coefficient *if linear spring (will be clarified later)*

Torsion dampers follow the same structure as Torsion springs to define the points and the same structure as tension dampers to define the damping coefficient. 

```python
Force["TorsionDamper"] = [((1,2),0),
                          ((1,3),0)]
```

#### **6 - Initial Conditions**

As opposed to frameworks using redundant coordinates (not using joint coordinates or Kane's method),
the initial conditions will not be updated due to incompatibility issues. This means that you can 
insert any value and the tool will solve as you requested except for the **position** of the float joints.
The float joints require the position of their CG to be defined and the coordinates represent the CG 
position itself, as opposed to other coordinates. Therefore the initial position of the float coordinates
must be equal to the corresponding `parent_cg_to_joint`, otherwise it will be automatically updated to 
the appropriate value while displaying an `[Info]` message on the screen.

```python
ic = np.zeros(2*sum(NDOF)) # Will automatically be updated to [0,1]
```

#### **7 - Simulation parameters**

This step is needed to simulate the system over time. The parameters defined before, could have been symbolic or numeric, to simulate the system over time, you need numerical values.

In this section we must provide at least the time parameters and the mass and inertia properties. The gravity by default is set to 9.81 and applied to all bodies. 
We define the parameters in the following manner:

- `TimeStep`: specifies the integration time taken in forward integration
- `tspan`: defines the end-time of the integration
- `g`: it is the value of gravity's acceleration
- `gVec`: it is a vector that defines, for each body, a gravity multiplier, i.e.: a value of 2 in the first position would indicate that the firs body is subject to 2 times the value of `g`.
- `m0`: it is a 0-D array specifying the mass of each body
- `J0`: it is a 0-D array specifying the second moment of inertia of each body

```python
# Simulation
TimeStep    = 0.05
tspan       = 10.
g           = 0.                        # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = 2.*np.ones(len(types))
```
The animation parameters are not related to the integration but rather just visualization. They define:

- `animation_on`: whether we want to display animation or not. It is a boolean and takes values of 0 or 1.
- `SaveMovieOn`: if name is specified with .gif extension, it saves the animation in the root folder.
- `plotTstep`: it is used to speed up the animation and specifies every how many time steps to plot. **It must be an integer.**

```python
# Animation
animation_on    = 1         # Display animation:1, 0-otherwise
SaveMovieOn     = None      # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 5         # how many times faster are we plotting w.r.t to simulation time
```

#### **8 - Simulation and results**

To run this example you only need to update one line in the main file. To import the FOSWEC example it must look like:

```python
from Examples import FOSWEC as ex
```

Foswec initial configuration looks like this (set `tspan=0.05` to get this visualization):

<!-- ![FOSWEC forces definition table](/_static/figs/Foswec_IC.png) -->

```{figure} _static/figs/Foswec_IC.png
:width: 50%
:align: center
```

and the energy plot looks:

<!-- ![FOSWEC forces definition table](/_static/figs/Foswec_energy.png) -->

```{figure} _static/figs/Foswec_energy.png
:width: 75%
:align: center
```

#### **Step-by-step PPT**

This portion contains the slides presenting this example. The slides with the animations are in TODO: ref to slides.

```{only} latex

```{raw} latex
\includepdf[
    pages=-,
    nup=1x2,
    frame=true,
    scale=0.88
]{Foswec.pdf}
```

```{only} html

```{raw} html
<object data="_static/slides/Foswec.pdf"
        type="application/pdf"
        width="100%"
        height="800px">
  <p>Your browser doesn’t support embedded PDFs.
     <a href="_static/slides/Foswec.pdf">Download the slides</a>.</p>
</object>
```

### Example 2: Double pendulum on cart
In this example we want to model a well known example in multibody dynamics and now if interest in the machine learning
community, double pendulum on a cart. This example is interesting from the control theory perspective, although we will 
not focus on stabilizing the pendulum on top of the cart but rather visualize the motion. For you is interesting also 
because you will learn how to model prismatic joints and a dig deeper in the origing offset. Additionally, you will define
time dependent forces at the center of mass. This is example 4 of the default examples, except for the displace origin. 

#### **1 - Define origin of inertial frame of reference (f.o.r)**

We want to model the base of the body to be above the ground 1 unit. One way to do so without over-complicating the vectors
``parent_cg_to_joint`` and ``joint_to_child_cg`` is to offset where the bodies definition starts. The way this offset works 
is a vector that goes from the ground origin to the model origin. In the figure below is a vector going from `O1` to `O2`. This feature
may be useful when defining the prismatic direction as horizontal but we want some offset from the ground, if anything is
parametric.

<!-- ![Double pendulum on cart](/_static/figs/DoublePend_onCart_sketch.jpg) -->

```{figure} _static/figs/DoublePend_onCart_sketch.jpg
:width: 50%
:align: center
```

```python 
Reference_frame_Origin = np.array([-1,1])
```

Check the schematics to understand better what `Reference_frame_Origin` means. After the body definition, we will clarify a 
few points on what it means then to define `parent_cg_to_joint` for the bodies in contact with ground and `joint_to_child_cg`.

```python
joints              = [[0, 1],[1, 2],[2,3]]                         
types               = ['P','R', 'R']                                
parent_cg_to_joint  = [[3, 0],[0,0],[0,1]]                          
joint_to_child_cg   = [[0,1],[0, 2],[3,0]]                          
prismatic_direction = [[3, 0],[np.nan, np.nan],[np.nan, np.nan]]    
prismatic_direction = normalize_prismatic(prismatic_direction) 
```

By moving the reference frame to `[-1,1]`, then we transformed `parent_cg_to_joint`,
vector $\vec{s}_0^{O_3}$, to a horizontal line defined by vector `[3,0]`. If we want to define this vector symbolically
and then the prismatic direction for the cart as that same vector, offsetting the origin becomes handy. 

A prismatic joint contains one degree of freedom, like revolut joints, but it is a translational DOF. To define a prismatic joint, in our framework, we need three points and a sliding direction. These points are the center of mass of the two connected bodies and the point of contact, `P` between the two bodies. In fact there are two points `P`, one belonging to the first body and another that belongs to the second one. However, initially they are defined in the same location (this can be changed at initial conditions). The sliding direction is defined by the vector $\vec{u}$ which is a unit vector and the magnitude of that vector, $\gamma$ determines how far apart the two `P` points are apart as the simulation proceeds. This variable $\gamma$ is our joint coordinate.

```{figure} _static/figs/prismatic_sketch.jpg
:width: 80%
:align: center
```



In our example, we have defined the first body as prismatic joint type. This means that we must define a prismatic direction 
associated to it. We can define it with whatever vector we want but then its norm is normalized using the function 
`normalize_prismatic` which lives under multibody. 

Another interesting feature of prismatic joints is that they do not possess their own rotational degree of freedom, and 
therefore inherit the rotation of the previous body in the chain. In this case the previous body is the ground, hence
the cart has no rotation at all and will move only horizontally. 

In summary, this system has three bodies connected by prismatic and revolut joint. These joints contribute with 1 translational
DOF (prismatic joint) and 2 rotational DOFs (1 per revolut joint). We name the translational DOFs with `S` and the rotational 
DOFs with `Theta`. Remember that the subindex shows which body they belong to. We can print the DOFs on screen by running the 
example script by itself. For a closer look you can check the following lines: 

``` python
Q, QD, QDD, NDOF, _   = joint_system.coordinate_finder()
```

These variables will show you the positional DOFs, `Q`, as well as as the velocity and acceleration coordinates, `QD`, `QDD`,
respectively. Lastly, NDOF summarizes the number of DOFs per joint, so in this case it would be a list showing `[1,1,1]`.

#### **2 - Points definition**

This system does not have forces applied outside from the CG, thus no point definition is required. However, the list for `GR`
points and the dictionary for `BD` points must be initialized.

```python
Initial_Points["GR"]    = []
Initial_Points["BD"]    = {}
```

#### **3 - Forces definition**

The forces definition involves the last learning point of this example. Define a time-dependant force at the CG of body 1. 
To define a time-dependant force, we must initialize it as a symbolic variable. Since we are using it in the force section
we initialize it in this section. If it was used in multiple sections, it should only be initialized in the **FIRST** one.
To define this symbolic variable, we define a string `"t"` in the list `ForceNames`. 

```python
ForceNames  = ["t"] 
ForceSym    = symvars_definition(ForceNames,globals()) # DO NOT MODIFY
```

The second line, where `ForceSym` is defined, and which should not be touched, converts each element in the list into a 
`sympy Symbol` with the assumption of `real=True`. This is important because manipulation of this variable will involves
taking into account that the variable is real. Check `Sympy` [Gotchas](https://docs.sympy.org/latest/explanation/gotchas.html)
to dig deeper into the concept. 

Once the symbolic variables needed are defined, then we can proceed to define the force of interest. Remember all the force
types must be defined, even as empty. We are interested in a force applied on the CG of body 1. Then, in the force dictionary,
in the key: `"CG"` we can create a list for each for applied in the CG. The elements on the list will specify to which body is 
the force applied to and the components of the force as `Fx, Fz, My`. In our case, we decided to apply a horizontal force with
exponential decay. Note that this definitions must follow `SymPy` sintax.

The force definition lines look like this: 

```python 
Force["PointsBD"]       = []
Force["CG"]             = [[1,sym.exp(-t),0,0]]
Force["TensionSpring"]  = []
Force["TensionDamper"]  = []
Force["TorsionSpring"]  = []
Force["TorsionDamper"]  = []

ForcesPointsSym = PointsSym + ForceSym 
```

The last line in the code above groups all the symbolic variables defined in the points and forces section to create the numeric 
association and mapping latter on. It must exists but it is not of the users concern. It appears here cos it is necessary for the
checks. 

The remaining lines for numerical values and simulation are left for completeness but it requires no user input and they have
been already explained. 

#### **4 - Numerical simlulation and animation**

Just a quick comment, we saw that this problem has 3 DOFs, one translational and two rotationals. Since we are solving a 2nd order
ODE, it requires initial conditions for position and velocity, thus **double** of the number of DOFs. 

```python
# If you are unsure what are the systems DOF run the example and you will see on screen
ic                  = np.array([0,0,0,0,0,0]) # Multiplied by 2 because is position and velocity

# Parameters to loop over
ForcesPointsNum = np.ones(len(ForcesPointsSym))
BodyDataNum     = np.ones(len(BodyDataSym))

# Simulation
TimeStep    = 0.05
tspan       = 5.
g           = 0.                        # gravity
gVec        = np.ones((len(types),1))   # Percentage of gravity acting on each body
m0          = np.ones(len(types))
J0          = 2.*np.ones(len(types))

# Animation
animation_on    = 0         # Display animation:1, 0-otherwise
SaveMovieOn     = None      # Extension:gif Saves animation if given a string as name, otherwise set as None
plotTstep       = 5         # how many times faster are we plotting w.r.t to simulation time
```

And after all that hard effort ... this is how it looks:

<!-- ![Double pendulum on cart sim](/_static/figs/DoublePend_onCart_fig.png) -->

```{figure} _static/figs/DoublePend_onCart_fig.png
:width: 50%
:align: center
```

#### **Step-by-step PPT**

This portion contains the slides presenting this example. The slides with the animations are in TODO: ref to slides.

```{raw} latex
\includepdf[
    pages=-,
    nup=1x2,
    frame=true,
    scale=0.88
]{DoublePendulum_onCart.pdf}
```

```{only} html

```{raw} html
<object data="_static/slides/DoublePendulum_onCart.pdf"
        type="application/pdf"
        width="100%"
        height="800px">
  <p>Your browser doesn’t support embedded PDFs.
     <a href="_static/slides/DoublePendulum_onCart.pdf">Download the slides</a>.</p>
</object>
```

## Validation of EOM derivation 

### Example 1: Oyster device
This example is done all symbolically and verified against one of the examples in Ginsberg’s Advanced Dynamics Text book. 
The example introduces some simplifications that the code doesn’t, therefore for valid comparison some numerical values 
must be substituted. The main assumption is small angle variation around equilibrium position. The derivation of the kinetic and potential energy is defined below for small $\theta$ and assuming $cos^2 \beta = \frac{b^2}{(a^2 + b^2)}$

The problem statement aims to find the equations of motion assuming the springs are initially in tension. Our goal is to find both the right and left hand sides of the equation. The sketch is as follows,

<!-- ![OYSTER problem statement](/_static/figs/Oyster_problemStatement.png) -->

```{figure} _static/figs/Oyster_problemStatement.png
:width: 60%
:align: center
```

Using M4E, the equations of motion are:

:label: eq:eom_ginsberg
EOM = \frac{m L^2}{3},\ddot{\theta}

* \left[
  \frac{2,a^{2} b^{2} k l_0}{(a^{2} + b^{2})^{3/2}}

* \tfrac{1}{2} m_1 g L
  \right]\theta = 0

Consequently, the stiffness matrix K can be found in our code by taking $\partial{EOM} / \partial{\theta}$ at equilibrium as: 

:label: eq:k_ginsberg
K = \left[
\frac{2,k,l_0,(a,\cos\beta)^2}{\sqrt{a^2 + b^2}}

* \tfrac{1}{2},m,g,L
  \right]

For small oscillations, one can write

:label: eq:cos_beta
\cos\beta = \frac{b}{\sqrt{a^2 + b^2}}

Substituting equation {eq}eq:cos_beta into equation {eq}eq:k_ginsberg
yields the same stiffness expression reported in Ginsberg’s book.

This example is defined in the *Example* folder under the name “Flap_sandia”. To run it, update the main file to:

```python
from Examples import Flap_sandia as ex
```

After running the example, we can compare the EOM given by the code:

```python
sym.simplify(MBDsys.ReducedM)
```

Where we get the expression:

$$ReducedM = (m_1*L^2)/4 + J_1$$

Recall that the moment of inertia, $J_1$, is always around the CG and for a bar is $J_1 = m_1L^2/12$. So adding up both terms, the result matches the one from the book.

*Right-hand-side:*

The first step is to declare all necessary variables as symbolic:

```python
from sympy import symbols, cos, diff

# define a symbol q
Theta_1, a, b, k, l0, m1, beta, g, L = symbols('Theta_1 a b k l0 m1 beta g L', real=True)
```

The expression from the book, for the right-hand-side without including $\theta$, i.e. a stiffness, K, can be written as:

```python 
book_eq = 2 * k * l0 * (a * cos(beta))**2 / (a**2 + b**2)**0.5 - 0.5 * m1 * g * L
```

The equivalent stiffness from the code can be compute as:

```python
K = -sym.diff(MBDsys.Right_side,MBDsys.Q[0])
```

Where the minus comes from being on the other side of the equal sign. Since comparing these two equations in analytic form is not an easy taks (left to the adventourous reader) we provide some numerical values for comparison. 

$a = 0.5$,
$b = 1$,
$k = 1$,
$L = 2$,
$m_1 = 1$,
$l0 = \sqrt{(a^2 + b^2)}$,
$\beta = cos(\frac{b^2}{a^2 + b^2})^{-1}$

<span style="color:red">Double check cos the answer matched at -9.41</span>

```python
book_eq.subs({L: 2,  m1: 1, k: 1, a: 0.5, b: 1, l0: np.sqrt(1.25), beta: betaVal, g:9.81}).evalf(4) = -9.49
```

The solution in the book is valid for small values of $\theta$, as a result we substitute it by $\theta = 0.001$

```python
K.subs({L: 2,  m1: 1, k: 1, a: 0.5, b: 1, l0: np.sqrt(1.25), beta: betaVal, MBDsys.Q[0]:1e-3}).evalf(4) = -9.41
```

Since both the right-hand-side and left-hand-side match we can conclude that the code is properly writing the equations of motion for this example.

#### **Step-by-step PPT**

This portion contains the slides presenting this example. The slides with the animations are in TODO: ref to slides.

The following slides include also a torsion spring at the hinge. To fully reproduce the same example,
leave the field empty. 

```python
Force['TorsionSpring'] = []
```

```{raw} latex
\includepdf[
    pages=-,
    nup=1x2,
    frame=true,
    scale=0.88
]{Flap.pdf}
```

```{only} html

```{raw} html
<object data="_static/slides/Flap.pdf"
        type="application/pdf"
        width="100%"
        height="800px">
  <p>Your browser doesn’t support embedded PDFs.
     <a href="_static/slides/Flap.pdf">Download the slides</a>.</p>
</object>
```



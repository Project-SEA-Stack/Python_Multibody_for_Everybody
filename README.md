# Multibody4Everybody

**Multibody4Everybody** is a symbolic multibody dynamics toolkit built in Python for 2D planar systems.
It uses joint (reduced) coordinates to generate symbolic equations of motion, and supports
nonlinear time integration, frequency-domain linearization with linear potential-flow hydrodynamics,
YAML-based model input, and flexible visualization.

---

## 📦 Features

- Symbolic computation of positions, velocities, and accelerations using joint coordinates
- Automatic construction of reduced-order equations of motion (joint coordinates formulation)
- Support for revolute (`R`), prismatic (`P`), and floating (`F`) joints
- Energy tracking and validation against benchmark solutions
- Modular architecture with animation and plotting tools
- YAML-based model and simulation input parser
- External force coupling interface (MoorDyn mooring lines, hydrodynamic loads)
- Frequency-domain linearization with linear potential-flow hydrodynamics:
  - BEM solution via [Capytaine](https://github.com/capytaine/capytaine)
  - Added mass, radiation damping, hydrostatic stiffness, and wave excitation
  - Regular (monochromatic) wave forcing with cosine ramp
  - Density rescaling when loading pre-computed `.nc` BEM files
- Compatible with both `pip` and `conda`

---

## 🛠 Installation

```bash
# Clone the repo and enter the directory
git clone https://github.com/Project-SEA-Stack/Python_Multibody_for_Everybody
cd Python_Multibody_for_Everybody
```

**Option 1 — conda (recommended):**
```bash
conda env create -f multibody_env.yaml
conda activate multibody_env
```

**Option 2 — pip (existing environment):**
```bash
pip install -e .   # editable install for developers
# or
pip install .      # regular install for users
```

---

## 🚀 Getting Started

**Run a multibody dynamics example:**
```bash
# From the repository root
python -c "
from Examples_mbd import spiderfloat as ex
from multibody import MbdSystem, integrate_dynamics
MBD = MbdSystem.from_example(ex)
"
# or use main.py after editing the import at the top
python main.py
```

**Run a YAML-defined model:**
```bash
python -c "
from multibody.yaml_parser.yaml_adapter import YamlAdapter
MBD = YamlAdapter.from_yaml('Examples_yaml/spiderfloat.model.yaml',
                             'Examples_yaml/spiderfloat.simulation.yaml')
"
```

**Instantia a YAML-defined model on the main script:**

```python
from multibody import load_yaml_as_example, MbdSystem
ex = load_yaml_as_example("yaml_examples/double_pendulum")
MBD = MbdSystem.from_example(ex)
```

**Run a linearization example (requires Capytaine):**
```bash
# Edit the example at the top of main_linearization.py, then:
python main_linearization.py
```

---

## 🧭 Project Structure

```
Python_multibody_dyamics/
├── source/
│   └── multibody/              # core library package
│       ├── multibody_core/     # joints, kinematics, EOM, integration
│       ├── linearization/      # symbolic linearization + hydrodynamic adapter
│       │   ├── linearize_eom.py
│       │   ├── linearization_main.py
│       │   ├── linear_integrator.py
│       │   └── hydro_linear_mckf.py
│       ├── ext_forces_manager/ # MoorDyn and external force coupling
|       |   ├── coupling_main.py
│       │   ├── external_class_template.py
│       │   ├── hydro_adapter.py
│       │   ├── moordyn_adapter.py
│       │   └── moordyn_writter.py
│       ├── yaml_parser/        # YAML model and simulation input
│       ├── plotting/           # animation and result plots
│       ├── checks/             # joint, point, and force validation
│       └── tables/             # tabular summaries of system definition
├── Examples_mbd/               # nonlinear MBD examples (Python input)
│   ├── _1_Double_pendulum.py
│   ├── spiderfloat.py
│   └── FOSWEC.py  ...
├── Examples_linearization/     # linearization + hydrodynamics examples
│   ├── foswec/
│   └── single_flap/
├── Examples_yaml/              # YAML model and simulation input files
│   ├── spiderfloat.model.yaml
│   └── spiderfloat.simulation.yaml  ...
├── main.py                     # entry point for nonlinear MBD simulation
├── main_linearization.py       # entry point for linearization workflow
├── multibody_env.yaml          # conda environment specification
├── pyproject.toml
└── README.md
```

---

## 📚 Documentation

Install Sphinx dependencies (included in `multibody_env.yaml`), then build:

```bash
cd documentation
conda run -n multibody_env sphinx-build source build/html -b html
# open build/html/index.html in a browser
```

To build a PDF (requires a TeX distribution such as `tectonic`):
```bash
conda install -c conda-forge tectonic
conda run -n multibody_env sphinx-build source build/latex -b latex
cd documentation/build/latex
tectonic main.tex
```


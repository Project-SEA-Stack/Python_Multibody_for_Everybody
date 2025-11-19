# Multibody4Everybody

**Multibody4Everybody** is a symbolic multibody dynamics toolkit built in Python. It supports revolute, prismatic, and floating joints with an emphasis on symbolic equation generation, modular architecture, and flexible visualization.

---

## 📦 Features

- Symbolic computation of positions, velocities, and accelerations
- Construction of reduced-order equations of motion
- Support for revolute, prismatic, and floating joints
- Energy tracking and force visualization
- Modular structure with animation and plotting tools
- Compatible with both `pip` and `conda`

---

## 🛠 Installation

To install using either `conda` or `pip`, run:

```bash
# Clone the repo and enter the directory
git clone https://github.com/Project-SEA-Stack/Python_multibody_dyamics.git
cd Python_multibody_dyamics
```

CONDA
```bash
# Create and activate the Conda environment
conda env create -f multibody_env.yaml
conda activate multibody_env
```

PIP
```bash
# OR install with pip directly
pip install .
```

## 🚀 Getting Started

To verify geometry and correct implementation of an example:

```bash
python Examples/spiderfloat.py

# or
cd Examples
python spiderfloat.py
```

To run the full simulation:

```bash
# From root folder
python main.py
```

## 📚 Documentation
To build the documentation (if using Sphinx) make sure you have all dependencies installed:

```bash
pip install Sphinx
pip install sphinx-autodoc-typehints
pip install myst-parser
pip install furo
```

```bash
cd documentation
make html
```



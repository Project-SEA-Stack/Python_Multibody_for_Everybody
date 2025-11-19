# Installation
The goal of this chapter is to help the user install the python package as an
external library that can be used no matter where it is installed. This is common
practice in the open-source community. To proceed, the user must open a terminal
at the location where they want to download the package for installation, typically
the *downloads* folder. 

To install the package without prior knowledge we recommend following option number 
one: CONDA option. This requires having conda installed in the computer. For how to
install conda please visit: [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
Once conda is installed proceed with the following steps.


**Common step:** to install using either conda or pip, first run on the terminal:

```bash
# Clone the repo and enter the directory
git clone https://github.com/Project-SEA-Stack/Python_multibody_dyamics.git
cd Python_multibody_dyamics
```

**Option 1:** then for CONDA environment creation and installation run: 

    
```bash
# Create and activate the Conda environment
conda env create -f multibody_env.yaml
conda activate multibody_env
```

**Option 2:** for PIP installation in an already existing environment, run either option:

    
```bash
pip install .    # Option 1, for users
pip install -e . # Option 2, for developers
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
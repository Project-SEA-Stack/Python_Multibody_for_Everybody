# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../source'))  # adjust based on location

project = 'Python M4E'
copyright = '2025, Alvaro Diaz Flores Caminero, Sahand Sabet'
author = 'Alvaro Diaz Flores Caminero, Sahand Sabet'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx.ext.autosummary',
  'sphinx_autodoc_typehints',
  'myst_parser',
  "sphinx.ext.mathjax",
]

# generate stub files for every module/class you list below
templates_path = ['_templates']
autosummary_generate = True  
autosummary_template_dirs = ['_templates/autosummary']

# move Python 3 type hints into the description
autodoc_typehints = 'description'

# include both class docstring and __init__ docstring
autoclass_content = 'both'

# default options for all autodoc directives
autodoc_default_options = {
  'members': True,
  'undoc-members': False,
  'inherited-members': True,
  'member-order': 'bysource',
  'show-inheritance': True,
  'exclude-members': '__init__',
#   'special-members': '__init__',
}

napoleon_google_docstring = True
napoleon_numpy_docstring  = True

# -- HTML theme -------------------------------------------------
html_theme = "furo"

pygments_style = 'sphinx'   # or any other Pygments style you like

# -- stop Sphinx choking on missing third-party imports: --
autodoc_mock_imports = [
    'moordyn',
    ]

# -- ignore your Validation examples (they try to load .mat) --
exclude_patterns = [
    '_build',
]



# -- shorten headings by dropping the full module path: ---
add_module_names = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# 1) Choose minted as the code formatter
latex_engine = 'xelatex'
latex_use_xindy = False        # keep default indexing off
latex_use_minted = True        

latex_elements = {
  'classoptions': ',openany,oneside,12pt',     # combine both options + font size
  'extraclassoptions': 'titlepage=false',
  'babel':           '\\usepackage[english]{babel}',
  'figure_align':    'H',
  'preamble': r'''
     \usepackage{setspace}
     \usepackage{lmodern}
     \usepackage{helvet}
     \renewcommand{\familydefault}{\sfdefault}
     \linespread{1.2}
     \usepackage{minted}
     \setminted{fontsize=\small,baselinestretch=1}
     \usepackage{pdfpages}
     \usepackage{textcomp}
     \usepackage{sphinx}
     \usepackage{standalone}
  ''',
}

latex_additional_files = [
    "_static/slides/Flap.pdf",
    "_static/slides/Foswec.pdf",
    "_static/slides/DoublePendulum_onCart.pdf",
]

# Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "dollarmath",   # allows $…$ and $$…$$
    "amsmath",      # (optional) if you want AMS‐style environments
    "colon_fence",  # allows ::: {raw} …
    "attrs_inline",     # so {width="…"} on inline images actually works
    # "bare_raw_block", # allows raw blocks with no extra indent
]

master_doc = 'index'

latex_documents = [
  (master_doc,                                   # source start file
   'main.tex',           # target LaTeX file name
   'Python multibody4everybody',              # document title
   'Alvaro Diaz Flores Caminero, Sahand Sabet',  # author(s)
   'manual'),                                 # documentclass
]

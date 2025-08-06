# docs/source/conf.py -------------------------------------------
import pathlib, sys

# 1. Absolute path to the repository root (three levels up from this file)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, PROJECT_ROOT.as_posix())      # must happen first
# ---------------------------------------------------------------

# now it is safe to import your project or read files inside it
from finmlkit._version import __version__        # no try/except needed

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FinMLKit'
copyright = '2025, FinMLKit'
author = 'Dániel Terbe'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.graphviz']

# Autodoc / autosummary quality-of-life
autosummary_generate = False
# autodoc_typehints = 'description'     # cleaner type hints in text
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'ignore-module-all': True,
}

# Napoleon (docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Theme
html_theme = 'furo'
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']

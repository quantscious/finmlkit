import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Points to the root directory containing 'finmlkit'
print("Python Path:", sys.path)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FinMLKit'
copyright = '2024, FinMLKit Developers'
author = 'Dániel Terbe'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest']

# Autodoc / autosummary quality-of-life
autosummary_generate = True
autodoc_typehints = 'description'     # cleaner type hints in text
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# Napoleon (docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Theme
html_theme = 'furo'
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']

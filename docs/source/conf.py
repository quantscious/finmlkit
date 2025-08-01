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

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autodoc.typehints', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autosummary', 'sphinx.ext.doctest']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Points to the root directory containing 'finmlkit'
print("Python Path:", sys.path)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

def get_version():
    """Read version from _version.py without importing."""
    # Path from docs/source/conf.py to finmlkit/_version.py
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, '..', '..', 'finmlkit', '_version.py')

    with open(version_file, 'r', encoding='utf-8') as f:
        content = f.read()
        for line in content.splitlines():
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")

    raise RuntimeError('Cannot find version string')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FinMLKit'
copyright = '2025, FinMLKit Developers'
author = 'Dániel Terbe'
release = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest']

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

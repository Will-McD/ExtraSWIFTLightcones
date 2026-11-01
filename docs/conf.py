# Sphinx configuration for SWIFTLET
import os
import sys

# make the package importable without installing it (and its heavy dependencies)
sys.path.insert(0, os.path.abspath(".."))

project = "SWIFTLET"
author = "William McDonald"
copyright = "2026, William McDonald"

try:
    from importlib.metadata import version as _version
    release = _version("swiftlet-lightcones")
except Exception:
    release = "dev"

extensions = [
    "sphinx.ext.autodoc",       # API docs from docstrings
    "sphinx.ext.autosummary",   # summary tables for the API pages
    "sphinx.ext.napoleon",   # reads NumPy-style "Parameters / Returns" docstrings
    "sphinx.ext.viewcode",      # "view source" links
    "sphinx.ext.intersphinx",   # links to numpy, h5py, ... docs
    "sphinx_design",            # card grids, tabs, icons
]



# packages Read the Docs can't install: Sphinx pretends they exist
autodoc_mock_imports = ["mpi4py", "virgo", "lightcone_io", "hdfstream"]

autodoc_member_order = "bysource"
autosummary_generate = True
templates_path = ["_templates"]
autodoc_member_order = "bysource"   # keep functions in the order they appear in the file

# show each class's __init__ docstring (its parameters) on the class page, after the class docstring
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ---- Read the Docs theme ----
html_theme = "sphinx_rtd_theme"
#html_theme = "sphinx-book-theme"
html_theme_options = {
    "navigation_depth": 4,          # how deep the sidebar tree goes
    "collapse_navigation": False,   # keep sidebar sections expanded
    "sticky_navigation": True,      # sidebar scrolls with the page
    "style_external_links": True,   # mark links that leave the docs
    "prev_next_buttons_location": "bottom",
}
html_title = "SWIFTLET"

# ---- add to docs/conf.py: links for the nitpicky (-n) build --------------------------------------------------------
# with -n, every :func:/:class: and every name in a :type: line must link to its documentation. intersphinx links the
# names of other packages to their docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "unyt": ("https://unyt.readthedocs.io/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest", None),
    "swiftsimio": ("https://swiftsimio.readthedocs.io/en/latest", None),
}

# words used in :type: lines that describe a type rather than name a class, and classes with no docs to link to
nitpick_ignore_regex = [
    ("py:class", r"array[-_]like|sequence|shape|function|callable|optional|scalar"),
    ("py:class", r"units?|equivalent|deg|Mpc|two floats|a str|\d+"),
    # private module paths that the projects' own inventories don't list
    ("py:class", r"matplotlib\.axes\._axes\..*|astropy\.cosmology\.flrw\..*"),
    # lightcone_io has no online docs
    ("py:class", r"lightcone_io\..*"),
]

# Sphinx configuration for SWIFTLET
import os
import sys

# make the package importable without installing it (and its heavy dependencies)
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../tools"))

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

autodoc_member_order = "bysource"# keep functions in the order they appear in the file
autosummary_generate = True
templates_path = ["_templates"]

# functions and classes left out of the API docs, by their full name
autosummary_context = {
    "hidden": [
        "swiftlet.snapshot_lightcone.seperator_str",
        "swiftlet.snapshot_lightcone.message",
        "swiftlet.snapshot_lightcone.rank_message",
        "swiftlet.snapshot_lightcone.print_coord_ranges",
        "swiftlet.snapshot_lightcone.FileReadSpec",
        "swiftlet.snapshot_lightcone.HaloCatalogueSpec",
        "swiftlet.snapshot_lightcone.HaloReadPiece",
        "swiftlet.snapshot_lightcone.SnapshotReadRecord",
        "swiftlet.lightcone_projections.check_ptype",
        "swiftlet.snapshot_redshifts.SnapshotCoverage",
        "swiftlet.mask_haloes.message",
        "swiftlet.mask_haloes.rank_message",
        "swiftlet.mask_haloes.get_num",
        "swiftlet.mask_haloes.attr_scalar",
        "swiftlet.mask_haloes.native_endian",
        "swiftlet.soap_reader.comoving"
    ]
}

# show each class's __init__ docstring (its parameters) on the class page, after the class docstring
autoclass_content = "both"

# autoclass_content = "both" also shows the docstring Python writes for a named tuple's __new__
# ("Create new instance of X(a, b)"), and :undoc-members: gives each field of a named tuple or dataclass its own entry,
# although the class docstring already lists them with :ivar:. These hooks leave them out, so a class page shows the
# class's own summary, the parameters of its own __init__, and its methods
def _drop_named_tuple_new(app, what, name, obj, options, lines):
    """Remove the "Create new instance of ..." text that a named tuple adds to its class docstring."""
    if what != "class":
        return
    for i, line in enumerate(lines):
        if line.startswith("Create new instance of "):
            del lines[i:]
            while lines and not lines[-1].strip():
                lines.pop()
            return


# the class autodoc is documenting: its docstring is processed before its members are
_current_class = [None]


def _note_class(app, what, name, obj, options, lines):
    """Remember the class whose members autodoc documents next."""
    if what == "class":
        _current_class[0] = obj


def _skip_fields(app, what, name, obj, skip, options):
    """
    Leave out __init__, and the fields of named tuples and dataclasses, which their class docstrings list with :ivar:.

    The other methods of these classes are still documented.
    """
    #if name == "__init__":
    #    # its parameters are already shown after the class docstring (autoclass_content = "both"), so it gets no entry
    #    # of its own, even with special-members (e.g. autodoc_default_options = {"special-members": "__init__"})
    #    return True
    #Never document the constructor as a separate method
    if name == "__init__" or name.endswith(".__init__"):
        return True
    if (getattr(obj, "__doc__", None) or "").startswith("Alias for field number"):
        return True
    fields = getattr(_current_class[0], "__dataclass_fields__", {})
    if what == "class" and name in fields:
        return True
    return None


def setup(app):
    """Connect the hooks above to autodoc."""
    app.connect("autodoc-process-docstring", _note_class)
    app.connect("autodoc-process-docstring", _drop_named_tuple_new)
    app.connect("autodoc-skip-member", _skip_fields)

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

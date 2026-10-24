#!/bin/env python
import importlib
import importlib.util
import subprocess
import sys

import pytest

import swiftlet as esl

"""
Tests for the modules top level: public names are imported when first used, and importing the
module doesn't import the modules that need MPI.
"""

def test_import_does_not_load_submodules():
    # in a new interpreter, so modules imported by other tests don't count
    code = ("import sys, swiftlet; "
            "print(sorted(m for m in sys.modules if m.startswith('swiftlet.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"

# modules that need mpi4py and virgodc to import
MPI_MODULES = {"mask_haloes"}
HAVE_MPI_EXTRA = all(importlib.util.find_spec(name) is not None for name in ("mpi4py", "virgo"))

@pytest.mark.parametrize("name", esl.__all__)
def test_public_names(name):
    if esl._PUBLIC_NAMES[name] in MPI_MODULES and not HAVE_MPI_EXTRA:
        pytest.skip("needs mpi4py and virgodc")
    module = importlib.import_module(f"swiftlet.{esl._PUBLIC_NAMES[name]}")
    assert getattr(esl, name) is getattr(module, name)


def test_submodules_and_dir():
    assert esl.config is importlib.import_module("swiftlet.config")
    assert {"SnapshotBeam", "BeamProjection", "config", "__version__"} <= set(dir(esl))
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        esl.not_a_name
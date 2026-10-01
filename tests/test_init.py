#!/bin/env python
"""
Tests for the package's top level: public names are imported when first used, and importing the
package doesn't import the modules that need MPI.
"""
import importlib
import subprocess
import sys

import pytest

import extra_swift_lightcones as esl


def test_import_does_not_load_submodules():
    # in a new interpreter, so modules imported by other tests don't count
    code = ("import sys, extra_swift_lightcones; "
            "print(sorted(m for m in sys.modules if m.startswith('extra_swift_lightcones.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize("name", esl.__all__)
def test_public_names(name):
    module = importlib.import_module(f"extra_swift_lightcones.{esl._PUBLIC_NAMES[name]}")
    assert getattr(esl, name) is getattr(module, name)


def test_submodules_and_dir():
    assert esl.config is importlib.import_module("extra_swift_lightcones.config")
    assert {"SnapshotBeam", "BeamProjection", "config", "__version__"} <= set(dir(esl))
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        esl.not_a_name
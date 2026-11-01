Development tools
=================

Scripts in ``tools/`` for maintaining SWIFTLET. They are not installed with the package.


Installed orientation tables
----------------------------

.. automodule:: make_installed_orientation_tables
   :members:

Style check
-----------

Run every style check of the CI (ruff, the flake8 docstring checks, codespell, interrogate), writing the output
to ``logs/``:

.. code-block:: bash

   bash tools/style_check.sh            # the style checks
   bash tools/style_check.sh --docs     # also build the docs with -W -n

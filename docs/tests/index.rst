Tests and developer guidelines
==============================


.. toctree::
   :maxdepth: 1
   
   test_summary
   developers
   



..
   Tests
   -----

   All contributions need to pass the automated, ``pytest`` suite and ``ruff`` style checks. 

   For an overview of the pytest suite see: 




   Extra tools for developers
   --------------------------

   A collection of simple tools for development can be found at ``./tools``, these include: 

   .. list-table::
      :header-rows: 1
      :widths: 30 40

      * - Name
        - description

      * -  ``make_installed_orientation_tables.py``
        -  Python script to generate all default oritentation tables 
           (installed with SWIFTLET) and write them too their respective 
           YAML files. This script can be edited to build and write new 
           default tables. 

      * -  ``check_docstring_summaries``
        -  Python script to check the summary docstrings of all SWIFTLET 
           modules and functions is a full sentence with a clear line break 
           after it before continuing. This rule is not applied to 
           __init__ , however __init__ should contain the descript of all 
           arguments that it accepts and that all `` :type:``, ``:vartype:`` 
           and ``:rtype:`` lines correctly name types that the docs can link 
           to. 

      * -  ``style_check.sh``
        -  Runs all the style checks for all docstrings and module summaries 
           of the CI workflow. A summary and report card is written to a log 
           file ``./logs/style_check_<date>_<time>.log``. 


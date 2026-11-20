{#- The page of a class, when an autosummary with :toctree: gives classes their own pages. Unlike Sphinx's default
    template, it has no separate __init__ entry: with autoclass_content = "both" (docs/conf.py) the parameters of
    __init__ are already shown after the class docstring. The methods table lists only the class's own methods, as
    :members: documents below it, not those it inherits (e.g. from dict). -#}
{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:

   {% block methods %}
   {% set shown = methods | reject("equalto", "__init__") | reject("in", inherited_members) | list %}
   {% if shown %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in shown %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
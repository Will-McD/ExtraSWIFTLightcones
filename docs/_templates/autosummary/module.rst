{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

Full reference
--------------

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}

{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endfor %}
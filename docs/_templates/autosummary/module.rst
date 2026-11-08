{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:

{% block classes %}
{% if classes | reject('in', hidden | map('replace', fullname ~ '.', '') | list) | list %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes if fullname ~ '.' ~ item not in hidden %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions | reject('in', hidden | map('replace', fullname ~ '.', '') | list) | list %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions if fullname ~ '.' ~ item not in hidden %}
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

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions and warnings

.. autosummary::
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

Full reference
--------------

{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: dict, tuple
   :exclude-members: __init__
{% endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}

{% for item in exceptions %}
.. autoexception:: {{ item }}
   :show-inheritance:
{% endfor %}


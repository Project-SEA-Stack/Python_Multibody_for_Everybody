{%- set title = name.split('.')[-1] -%}
{{ title }}
{{ '=' * title|length }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

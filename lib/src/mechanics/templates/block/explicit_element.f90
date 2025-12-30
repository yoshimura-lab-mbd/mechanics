! Solve explicit equations ====================================================

{%- for eq in equations %}

{%- for bound_condition, error_key in eq.bound_checks.items() %}
if (.not. ({{bound_condition}})) then; error = {{error_key}}; goto 999; end if
{%- endfor %}
{%- for var, error_key in eq.dependency_checks.items() %}
if (ieee_is_nan({{var}})) then; error = {{error_key}}; goto 999; end if
{%- endfor %}
{{eq.l}} = {{eq.r}}
{%- if eq.nan_check %}
if (ieee_is_nan({{eq.l}})) then; error = {{eq.nan_check}}; goto 999; end if
{%- endif %}
{% endfor %}

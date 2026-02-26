from collections import namedtuple
from typing import Any, Mapping
import keyword
import sympy as sp

from mechanics.util import python_name


def _validate_key(raw_key: str, source_name: str) -> str:
    key = python_name(raw_key)
    if not key.isidentifier() or keyword.iskeyword(key):
        raise ValueError(
            f'python_name key "{key}" derived from name "{source_name}" '
            "is not a valid field name."
        )
    return key


def group_key(name: str) -> str:
    return _validate_key(name, name)


def group_from_mapping(entries: Mapping[str, Any], typename: str = "Group") -> Any:
    Group = namedtuple(typename, entries.keys())
    return Group(**entries)


def group_from_named(values: tuple[Any, ...], typename: str = "Group") -> Any:
    entries: dict[str, Any] = {}
    for value in values:
        if not hasattr(value, "name"):
            raise TypeError(f"group positional args must have .name, got: {type(value)}")

        source_name = value.name
        key = _validate_key(source_name, source_name)
        if key in entries:
            raise ValueError(f'Duplicate python_name key "{key}" for name "{source_name}".')
        entries[key] = value

    return group_from_mapping(entries, typename=typename)


def group(*values: Any, **entries: Any) -> Any:
    resolved: dict[str, Any] = {}

    if values:
        positional = group_from_named(tuple(values), typename="GroupPositional")
        resolved.update(positional._asdict())

    for key, value in entries.items():
        if key in resolved:
            raise ValueError(f'Duplicate key "{key}" in group().')

        if callable(value):
            # Evaluate lazily with the currently built group, so lambdas
            # can refer to earlier fields (including positional symbols).
            ctx = group_from_mapping(resolved, typename="GroupContext")
            resolved[key] = value(ctx)
        else:
            resolved[key] = value

    return group_from_mapping(resolved)


def _is_namedtuple_instance(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields") and hasattr(value, "_asdict")


def diff(expr: Any, *args: Any, **kwargs: Any) -> Any:
    if _is_namedtuple_instance(expr):
        converted = {key: diff(value, *args, **kwargs) for key, value in expr._asdict().items()}
        return expr.__class__(**converted)
    return sp.diff(expr, *args, **kwargs)

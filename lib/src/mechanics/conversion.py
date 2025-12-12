from typing import Any, overload, cast
import sympy as sp

from mechanics.symbol import Expr, ExplicitEquations, Variable, BaseSpace


class Conversion:


    @overload
    def __call__(self, arg: Expr) -> Expr: ...
    @overload
    def __call__(self, arg: tuple[Expr, ...]) -> tuple[Expr, ...]: ...
    @overload
    def __call__(self, arg: list[Expr]) -> list[Expr]: ...
    @overload
    def __call__(self, arg: ExplicitEquations) -> ExplicitEquations: ...

    def __call__(self, arg: Any) -> Any:
        if isinstance(arg, Expr):
            return self.convert_expr(arg)
        elif isinstance(arg, dict):
            return self.convert_explicit_equations(arg)
        elif isinstance(arg, tuple):
            return tuple(self(t) for t in arg)
        elif isinstance(arg, list):
            return [self(t) for t in arg]
        else:
            raise NotImplementedError('Conversion not implemented for type: ' + str(type(arg)))

    def convert_expr(self, expr: Expr) -> Expr:
        return expr

    def convert_explicit_equations(self, equations: ExplicitEquations) -> ExplicitEquations:
        converted = {}
        for f, expr in equations.items():
            new_f = self.convert_expr(f)
            new_expr = self.convert_expr(expr)
            converted[new_f] = new_expr
        return converted

    def __mul__(self, other: 'Conversion') -> 'Compose':
        return Compose(self, other)

class Replacement(Conversion):
    _subs: list[tuple[Expr, Expr]]

    def __init__(self, replacements: dict[Expr, Expr] | list[tuple[Expr, Expr]]):
        if isinstance(replacements, dict):
            self._subs = list(replacements.items())
        else:
            self._subs = replacements

    def _replace_diff(self, *args) -> Expr:
        var = args[0]
        subs = var.base_space_subs | var.index_subs
        return sp.Derivative(var.general_form(), *args[1:]).subs(self._subs).subs(subs)

    def convert_expr(self, expr: Expr) -> Expr:
        return cast(Expr, expr.subs(self._subs).replace(sp.Derivative, self._replace_diff))
class Compose(Conversion):
    _conversions: tuple[Conversion, ...]

    def __init__(self, *conversions: Conversion):
        self._conversions = conversions

    def __call__(self, arg: Any) -> Any:
        result = arg
        for conv in reversed(self._conversions):
            result = conv(result)
        return result
from typing import Any, overload, cast

from mechanics.function import Expr, ExplicitEquations


class Conversion:

    @overload
    def __call__(self, arg: Expr) -> Expr: ...

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
    _replacements: list[tuple[Expr, Expr]]

    def __init__(self, replacements: dict[Expr, Expr] | list[tuple[Expr, Expr]]):
        if isinstance(replacements, dict):
            self._replacements = list(replacements.items())
        else:
            self._replacements = list(replacements)


    def convert_expr(self, expr: Expr) -> Expr:
        return cast(Expr, expr.subs(self._replacements))

class Compose(Conversion):
    _conversions: tuple[Conversion, ...]

    def __init__(self, *conversions: Conversion):
        self._conversions = conversions

    def __call__(self, arg: Any) -> Any:
        result = arg
        for conv in reversed(self._conversions):
            result = conv(result)
        return result
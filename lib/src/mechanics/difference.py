
from typing import Callable, overload, Any

from mechanics.symbol import Variable, Expr, Index, shift_index

class FiniteDifference:

    order: int

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        raise NotImplementedError()

    @overload
    def __call__(self, arg: Variable, index: Index, step: Expr) -> Expr: ...
    @overload
    def __call__(self, arg: tuple[Variable, ...], index: Index, step: Expr) -> tuple[Expr, ...]: ...
    @overload
    def __call__(self, arg: list[Variable], index: Index, step: Expr) -> list[Expr]: ...

    def __call__(self, arg: Any, index: Index, step: Expr) -> Any:
        if isinstance(arg, Variable):
            return self.apply(lambda shift: shift_index(arg, index, shift), step)
        elif isinstance(arg, tuple):
            return tuple(self(v, index, step) for v in arg)
        elif isinstance(arg, list):
            return [self(v, index, step) for v in arg]
        else:
            raise NotImplementedError('FiniteDifference not implemented for type: ' + str(type(arg)))

class ForwardDifference(FiniteDifference):
    order: int = 1

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        return (a(1) - a(0)) / step

class BackwardDifference(FiniteDifference):
    order: int = 1

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        return (a(0) - a(-1)) / step

class CentralDifference(FiniteDifference):
    order: int = 1

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        return (a(1) - a(-1)) / (2 * step)

class SecondCentralDifference(FiniteDifference):
    order: int = 2

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        return (a(1) - 2 * a(0) + a(-1)) / (step ** 2)


forward = ForwardDifference()
central = CentralDifference()
backward = BackwardDifference()
second_central = SecondCentralDifference()
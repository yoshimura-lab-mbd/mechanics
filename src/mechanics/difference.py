
from typing import Callable

from mechanics.symbol import Variable, Expr

class FiniteDifference:

    order: int

    def apply(self, a: Callable[[int], Expr], step: Expr) -> Expr:
        raise NotImplementedError()

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
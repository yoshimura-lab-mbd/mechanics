from dataclasses import dataclass

from mechanics.conversion import Conversion
from mechanics.symbol import ExplicitEquations, ImplicitEquations, Variable


@dataclass(frozen=True)
class EulerExplicitResult:
    states: tuple[Variable, ...]
    step_equations: ExplicitEquations
    conversion: Conversion


@dataclass(frozen=True)
class ModifiedEulerExplicitResult:
    states: tuple[Variable, ...]
    stages: tuple[Variable, ...]
    step_equations: ExplicitEquations
    conversion: Conversion


@dataclass(frozen=True)
class BackwardEulerExplicitResult:
    states: tuple[Variable, ...]
    unknowns: tuple[Variable, ...]
    step_equations: ImplicitEquations
    conversion: Conversion


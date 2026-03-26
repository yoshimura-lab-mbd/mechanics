from dataclasses import dataclass

from mechanics.conversion import Conversion
from mechanics.symbol import ExplicitEquations, ImplicitEquations, Variable

@dataclass(frozen=True)
class ExplicitIntegratorResult:
    state_variables: tuple[Variable, ...]
    stage_variables: tuple[Variable, ...]
    unknown_variables: tuple[Variable, ...]
    step_equations: ExplicitEquations
    conversion: Conversion

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self.state_variables + self.stage_variables + self.unknown_variables

    @property
    def is_explicit(self) -> bool:
        return True

@dataclass(frozen=True)
class ImplicitIntegratorResult:
    state_variables: tuple[Variable, ...]
    stage_variables: tuple[Variable, ...]
    unknown_variables: tuple[Variable, ...]
    step_equations: ImplicitEquations
    conversion: Conversion

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self.state_variables + self.stage_variables

    @property
    def is_explicit(self) -> bool:
        return False
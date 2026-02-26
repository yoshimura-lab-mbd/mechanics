from mechanics import base_space, group, variable
from mechanics.solver.block import CalculationElement
from mechanics.solver.element import SolverContext
from mechanics.solver.fortran import FortranPrinter
from mechanics.solver.runner import ErrorReceiver


def test_calculation_element_accepts_namedtuple_functions():
    t = base_space("t")
    q = variable("q", t)
    funcs = group(
        f1=q + 1,
        f2=q * 2,
    )

    context = SolverContext(ErrorReceiver())
    element = CalculationElement(context, funcs)
    code = element._generate(FortranPrinter())
    compact = code.replace(" ", "")

    assert "f1=q+1" in compact
    assert "f2=2*q" in compact

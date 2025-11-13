import sympy as sp
from sympy.printing.latex import LatexPrinter
from IPython.display import display, Math
from typing import Optional

from .function import Expr, Function, BaseSpace, Basic
from .util import tuple_ish, to_tuple
from .config import diff_notations

def show(*item: str | Basic):
    latex_str = ''
    for x in item:
        if isinstance(x, str):
            latex_str += x
        else:
            latex_str += latex(x)
    display(Math(latex_str))

def show_equations(eq: tuple_ish[Expr] | dict[Expr, Expr], rhs: Optional[Expr] = None):
    equations = []
    if isinstance(eq, dict):
        if rhs is not None:
            raise ValueError('rhs should be None when eq is a dict')
        for l, r in eq.items():
            equations.append(sp.Eq(l, r))
    else:
        for eq_n in to_tuple(eq):
            if isinstance(eq_n, sp.Eq):
                if rhs is not None:
                    raise ValueError('rhs should be None when eq contains equations')
                equations.append(eq_n)
            else:
                equations.append(sp.Eq(eq_n, rhs or 0))

    if not equations:
        return
    elif len(equations) == 1:
        show(equations[0])
    else:
        latex_str = '\\begin{cases}'
        for e in equations:
            latex_str += latex(e) + '\\\\'
        latex_str += '\\end{cases}'
        show(latex_str)

def latex(expr: Basic) -> str:
    return LatexPrinterModified().doprint(expr)

class LatexPrinterModified(LatexPrinter):

    def _print_Derivative(self, expr: Expr) -> str:

        notations = []
        no_notations = []

        for s, n in expr.args[1:]: #type:ignore
            if isinstance(s, BaseSpace) and s.name in diff_notations:
                notations.append((diff_notations[s.name], n))
            else:
                no_notations.append((s, n))

        if isinstance(expr.args[0], Function):
            def notation_combined(s):
                for notation, n in notations:
                    s = notation(s, n)
                return s
            printed = expr.args[0]._latex(self, notation=notation_combined)

        else:
            printed = self.doprint(expr.args[0])
            for notation, n in notations:
                printed = notation(printed, n)

        if no_notations:
            return super()._print_Derivative(sp.Derivative(sp.Symbol(printed), no_notations)) 
        else:
            return printed

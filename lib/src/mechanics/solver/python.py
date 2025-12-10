from sympy.printing.numpy import NumPyPrinter

from mechanics.symbol import Variable, Index
from mechanics.util import python_name


class PythonPrinter(NumPyPrinter):
    def _print_Symbol(self, symbol):
        if isinstance(symbol, Index):
            return f'{python_name(symbol.name)}'
        else:
            str(symbol)
    
    def _print_Function(self, f):
        if isinstance(f, Variable):
            if f.indices:
                indices = [f'int({self.doprint(i)})' for i in f.indices]
                return f'{python_name(f.name)}({", ".join(indices)})'
            else:
                return f'{python_name(f.name)}'
        else:
            return super()._print_Function(f)

import sympy.printing.fortran

from mechanics.symbol import Variable, Index
from mechanics.util import python_name


class FortranPrinter(sympy.printing.fortran.FCodePrinter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prefixes: dict[str, str] = {}

    def fortran_name(self, name: str) -> str:
        name = python_name(name)
        name = ''.join((c.lower() + '_') if c.isupper() else c for c in name)
        return name

    def _print_Symbol(self, symbol):
        if isinstance(symbol, Index):
            return f'{self.fortran_name(symbol.name)}'
        else:
            str(symbol)
    
    def _print_Function(self, f):
        if isinstance(f, Variable):
            if f.indices:
                indices = [f'int({self.doprint(i)})' for i in f.indices]
                return f'{self.fortran_name(f.name)}({", ".join(indices)})'
            else:
                return f'{self.fortran_name(f.name)}'
        else:
            return super()._print_Function(f)
        
    
    def print_as_array_arg(self, f: Variable) -> str:
        if not f.indices:
            return f'{self.fortran_name(f.name)}'
        args = []
        for i, value in f.index_subs.items():
            if i == value:
                args.append(':')
            else:
                args.append(f'{self.doprint(value)}')

        return f'{self.fortran_name(f.name)}({",".join(args)})'


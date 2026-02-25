from __future__ import annotations

from collections import namedtuple
from typing import Any, Optional

import sympy as sp

from .printing import latex
from .symbol import BaseSpace, Expr, IndexRange, Variable
from .util import to_tuple


class Markdown:
    def __init__(self) -> None:
        self._blocks: list[str] = []

    def show(self, *item: str | Expr) -> "Markdown":
        latex_str = ""
        for x in item:
            if isinstance(x, str):
                latex_str += x
            else:
                latex_str += latex(x)
        self._blocks.append(f"$$ {latex_str} $$")
        return self

    def show_equations(
        self,
        eq: tuple[Expr, ...] | dict[Expr, Expr],
        rhs: Optional[Expr] = None,
        indices: tuple[IndexRange, ...] = (),
    ) -> "Markdown":
        equations: list[Expr] = []
        if isinstance(eq, dict):
            if rhs is not None:
                raise ValueError("rhs should be None when eq is a dict")
            for l, r in eq.items():
                equations.append(sp.Eq(l, r))
        else:
            for eq_n in to_tuple(eq):
                if isinstance(eq_n, sp.Eq):
                    if rhs is not None:
                        raise ValueError("rhs should be None when eq contains equations")
                    equations.append(eq_n)
                else:
                    equations.append(sp.Eq(eq_n, rhs or 0))

        if indices:
            indices_str = r"\quad \text{for } " + ",".join([r._latex() for r in indices])
        else:
            indices_str = ""

        if not equations:
            return self
        if len(equations) == 1:
            self.show(equations[0], indices_str)
            return self

        latex_str = r"\begin{cases}"
        for e in equations:
            latex_str += latex(e) + r"\\"
        latex_str += r"\end{cases}"
        self.show(latex_str, indices_str)
        return self

    def add_markdown(self, text: str) -> "Markdown":
        self._blocks.append(text)
        return self

    def render(self, mo: Any) -> Any:
        body = mo.md("\n\n".join(self._blocks))
        return body.style(
            {
                "overflow-x": "auto",
                "overflow-y": "hidden",
                "max-width": "100%",
                "padding-bottom": "0.25rem",
            }
        )


class Sliders:
    def __init__(
        self,
        mo: Any,
        symbols: Any,
        configs: Optional[dict[Any, dict[str, Any]]] = None,
        value: Optional[dict[Any, Any]] = None,
        defaults: Optional[dict[str, Any]] = None,
    ) -> None:
        if not hasattr(symbols, "_asdict"):
            raise TypeError("symbols must be a namedtuple-like object")
        self._mo = mo
        self._symbols = symbols
        self._widgets: dict[str, Any] = {}
        configs = configs or {}
        value = value or {}
        slider_defaults = {"start": -100.0, "stop": 100.0, "step": 0.01, "value": 0.0}
        if defaults:
            slider_defaults.update(defaults)

        widgets: dict[str, Any] = {}
        for key, symbol in symbols._asdict().items():
            cfg: dict[str, Any] = dict(slider_defaults)
            if symbol in configs:
                cfg.update(configs[symbol])
            elif key in configs:
                cfg.update(configs[key])  # type: ignore[index]

            if symbol in value:
                cfg["value"] = value[symbol]
            elif key in value:
                cfg["value"] = value[key]  # type: ignore[index]

            label = cfg.pop("label", latex(symbol))
            widgets[key] = mo.ui.slider(label=label, **cfg)

        self._ui = mo.ui.dictionary(widgets)

    @property
    def ui(self) -> Any:
        """marimo がリアクティブ追跡できる mo.ui.dictionary を返す。
        controls セルからこれを return することで反応性が有効になる。"""
        return self._ui

    def render(self) -> Any:
        return self._mo.hstack(list(self._ui.elements.values()))

    def items(self) -> dict[Variable, Any]:
        return self.items_from(self._ui.value)

    def items_from(self, value_dict: dict[str, Any]) -> dict[Variable, Any]:
        return {symbol: value_dict[key] for key, symbol in self._symbols._asdict().items()}

    def widget_values(self) -> Any:
        SliderValues = namedtuple("SliderValues", self._ui.value.keys())
        return SliderValues(**self._ui.value)

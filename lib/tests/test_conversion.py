from collections import namedtuple

import sympy as sp

from mechanics.conversion import replacement
from mechanics.symbol import base_space, variable
from mechanics.differential_equation import to_first_order


def test_conversion_supports_namedtuple_values():
    t = base_space("t")
    q = variable("q", t)
    p = variable("p", t)
    State = namedtuple("State", ["q", "nested"])
    state = State(q=q, nested=(q, p))

    conv = replacement({q: p})
    converted = conv(state)

    assert isinstance(converted, State)
    assert converted.q == p
    assert converted.nested == (p, p)


def test_to_first_order_variables_of_order_returns_namedtuple():
    t = base_space("t")
    theta = variable(r"\theta", t)
    F = {sp.Derivative(theta, (t, 2)): -theta}

    first_order = to_first_order(F)
    v = first_order.variables_of_order(1)

    assert isinstance(v, tuple)
    assert v._fields == ("theta",)
    assert v.theta.name == r"v_{\theta}"

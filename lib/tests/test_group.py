from mechanics import base_space, diff, group, variable


def test_group_from_positional_symbols():
    t = base_space("t")
    theta = variable(r"\theta", t)
    r = variable("r", t)

    q = group(r, theta)

    assert q.r == r
    assert q.theta == theta


def test_group_supports_lambda_with_self_context():
    g = group(
        a=2,
        b=lambda self: self.a + 3,
        c=lambda self: self.a * self.b,
    )

    assert g.a == 2
    assert g.b == 5
    assert g.c == 10


def test_group_lambda_can_reference_positional_entries():
    t = base_space("t")
    r = variable("r", t)
    theta = variable(r"\theta", t)

    g = group(
        r,
        theta,
        x=lambda self: self.r + self.theta,
    )

    assert g.r == r
    assert g.theta == theta
    assert g.x == r + theta


def test_diff_applies_elementwise_to_namedtuple_group():
    t = base_space("t")
    r = variable("r", t)
    theta = variable(r"\theta", t)
    q = group(r, theta)

    dq = diff(q, t)

    assert dq.r == diff(r, t)
    assert dq.theta == diff(theta, t)

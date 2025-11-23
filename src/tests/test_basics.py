import pytest

from mechanics import *
from mechanics.lagrange import euler_lagrange_equation
import mechanics.space as space
from mechanics.integrator.runge_kutta import rk4_explicit


def test_kepler():

    t, = base_spaces('t')
    def dot(f): return diff(f, t)

    r, = variables('r', t)
    theta, = variables(r'\theta', t, space=space.S)
    q = r, theta
    dq = tuple(dot(q_n) for q_n in q)
    ddq = tuple(dot(dq_n) for dq_n in dq)

    mu, m = constants(r'\mu, m')

    x = r * cos(theta)
    y = r * sin(theta)

    U = - mu / r
    T = (m / 2 * (dot(x)**2 + dot(y)**2)).simplify()
    E = T + U
    L = T - U

    EL = euler_lagrange_equation(L, q)
    F = solve(EL, ddq)


    h, T = constants('h T')
    i, = indices('i')
    X, K, rk4_step, d = rk4_explicit(F, h, i)
    r, theta = d(q)
    v_r, v_theta = d(dq)

    with Solver() as solver:
        solver.constants(mu, m, h, T)
        solver.variables(*X, *K, index=(i, 0, T/h))
        solver.functions('x', 'y', 'E', index=(i, 0, T/h))
        solver.inputs(*(x[0] for x in X))
        with solver.steps(i, 0, T/h) as step:
            step.solve_explicit(rk4_step)
            step.calculate({'x': d(x), 'y': d(y), 'E': d(E)}, i)
    
    _ = solver.run({
        mu: 1.0, m: 1.0, 
        h: 0.1, T: 100.0,
        r[0]: 1.0, theta[0]: 0.0, #type: ignore
        v_r[0]: 0.0, v_theta[0]: 1.1, #type: ignore
    })
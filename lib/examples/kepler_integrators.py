import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from sympy import cos, sin, solve

    import mechanics.space as space
    from mechanics import Markdown, diff, base_space, constants, group, index, variable
    from mechanics.differential_equation import to_first_order
    from mechanics.integrator.euler import (
        backward_euler_explicit,
        euler_explicit,
        modified_euler_explicit,
    )
    from mechanics.integrator.runge_kutta import rk4_explicit
    from mechanics.lagrange import euler_lagrange_equation
    from mechanics.solver import build_solver


@app.cell
def intro():
    mo.md("""
    # Kepler problem: Integrator comparison

    Compare different numerical integrators for the Kepler problem.
    """)
    return


@app.cell
def _():
    mo.image(src=Path(__file__).with_name("figure") / "kepler_orbit.svg")
    return


@app.cell
def derive_lagrangian():
    t = base_space("t")

    q = group(
        variable("r", t),
        variable(r"\theta", t, space=space.S),
    )

    c = constants(r"\mu m")

    f = group(
        x=q.r * cos(q.theta),
        y=q.r * sin(q.theta),
        U=-c.mu * c.m / q.r,
        K=lambda f: (c.m / 2 * (diff(f.x, t) ** 2 + diff(f.y, t) ** 2)).simplify(),
        E=lambda f: f.K + f.U,
    )
    L = f.K - f.U
    return L, c, f, q, t


@app.cell
def derive_equations(L, q, t):
    EL = euler_lagrange_equation(L, q)
    F = solve(EL, diff(q, t, 2))
    return EL, F


@app.cell(hide_code=True)
def show_symbolic_results(EL, F, L):
    md = Markdown()
    md.add_markdown("### Lagrangian")
    md.show("L = ", L)
    md.add_markdown("### Euler-Lagrange equations")
    md.show_equations(EL)
    md.add_markdown("### Solved accelerations")
    md.show_equations(F)
    md.render(mo)
    return


@app.cell
def setup_constants():
    ht = constants("h T")
    h = ht.h
    T = ht.T
    i = index("i")
    return T, h, i


@app.cell
def build_integrators(F, h, i, q):
    first_order = to_first_order(F)

    integrators = {
        "Euler": euler_explicit(first_order.equations, h, i),
        "Backward Euler": backward_euler_explicit(first_order.equations, h, i),
        "Heun's method": modified_euler_explicit(first_order.equations, h, i),
        "RK4": rk4_explicit(first_order.equations, h, i),
    }

    q_ = integrators["RK4"].conversion(q)
    v_ = integrators["RK4"].conversion(first_order.variables_of_order(1))
    return first_order, integrators, q_, v_


@app.cell
def build_solvers(T, c, f, first_order, h, i, integrators):
    solvers = {}

    for _method, _integrator in integrators.items():
        _solver = build_solver()
        _solver.constants(*c)
        _solver.constants(h, T)
        _solver.variables(*_integrator.variables, index=(i, 0, T / h))
        _solver.functions(*f._fields, index=(i, 0, T / h))
        _solver.inputs(*(x_[0] for x_ in _integrator.state_variables))

        d = _integrator.conversion * first_order.conversion

        with _solver.steps(i, 0, T / h) as step:
            if _integrator.is_explicit:
                step.explicit(_integrator.step_equations)
            else:
                with step.newton() as newton:
                    newton.unknowns(*_integrator.unknown_variables)
                    newton.initial_guess({v: v.subs(i, i - 1) for v in _integrator.unknown_variables})
                    newton.implicit(_integrator.step_equations)
            step.calculate(d(f), i)

        solvers[_method] = _solver.generate()
    return (solvers,)


@app.cell
def controls():
    sliders = mo.ui.dictionary(
        {
            "r_0": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0, label=r"$r(0)$"),
            "theta_0": mo.ui.slider(start=-3.14, stop=3.14, step=0.01, value=0.0, label=r"$\theta(0)$"),
            "v_r_0": mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=0.0, label=r"$\dot{r}(0)$"),
            "v_theta_0": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.1, label=r"$\dot{\theta}(0)$"),
            "h": mo.ui.number(start=0.001, stop=0.5, step=0.01, value=0.2, label=r"$h$"),
            "T": mo.ui.number(start=10.0, stop=200.0, step=10.0, value=100.0, label=r"$T$"),
        }
    )
    sliders
    return (sliders,)


@app.cell
def run_simulations(T, c, h, q_, sliders, solvers, v_):
    _values = sliders.value
    results = {}

    for _method, _solver in solvers.items():
        results[_method] = _solver.run(
            {
                c.mu: 1.0,
                c.m: 1.0,
                q_.r[0]: _values["r_0"],
                q_.theta[0]: _values["theta_0"],
                v_.r[0]: _values["v_r_0"],
                v_.theta[0]: _values["v_theta_0"],
                h: _values["h"],
                T: _values["T"],
            }
        )
    return (results,)


@app.cell
def plot_orbits(results):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_aspect("equal")
    _ax.set_xlim(-2, 2)
    _ax.set_ylim(-2, 2)

    for _method, _result in results.items():
        _ax.plot(_result["x"], _result["y"], label=_method, linewidth=2)

    _ax.legend()
    _ax.set_title("Orbital comparison")
    _fig
    return


@app.cell
def plot_timeseries(T, h, q_, results, sliders, v_):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    h_val = sliders.value["h"]

    _fig, axes = plt.subplots(3, 2, figsize=(10, 10), tight_layout=True)
    axes = axes.flatten()

    series = [
        (r"r", q_.r),
        (r"\theta", q_.theta),
        (r"\dot{r}", v_.r),
        (r"\dot{\theta}", v_.theta),
        ("E", "E"),
    ]

    for idx, (name, var) in enumerate(series):
        ax = axes[idx]
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${name}$")

        rk4_result = results["RK4"][var][:-1]
        if hasattr(rk4_result, "max"):
            value_range = rk4_result.max() - rk4_result.min()
            value_center = (rk4_result.max() + rk4_result.min()) / 2
            ax.set_ylim(value_center - value_range * 0.6, value_center + value_range * 0.6)

        for _method, _result in results.items():
            t_ = np.arange(0, _result[T], _result[h])
            n = min(len(t_), len(_result[var]))
            ax.plot(t_[:n], _result[var][:n], label=_method, linewidth=1.5)

        ax.legend()
        ax.grid(True, alpha=0.3)

    _fig
    return


if __name__ == "__main__":
    app.run()

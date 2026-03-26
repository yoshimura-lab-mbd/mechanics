import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sympy import diff, sin, solve, tanh

    import mechanics.config
    import mechanics.space as space
    from mechanics import (
        Markdown,
        base_spaces,
        build_solver,
        constants,
        discretization,
        indices,
        show_equations,
        to_implicit,
        variables,
    )
    from mechanics.difference import central, second_central

    mechanics.config.diff_notations = {}

    return (
        Markdown,
        base_spaces,
        build_solver,
        central,
        constants,
        diff,
        discretization,
        indices,
        mo,
        np,
        plt,
        second_central,
        show_equations,
        sin,
        solve,
        space,
        tanh,
        to_implicit,
        variables,
    )


@app.cell
def intro(mo):
    mo.md("""
    # 1D Wave equation
    """)
    return


@app.cell
def setup_equation(base_spaces, constants, diff, variables):
    BC_0 = "free"  # 'free' or 'fixed'
    BC_L = "fixed"  # 'free' or 'fixed'

    t, x = base_spaces("t x")

    (u,) = variables("u", t, x)
    c, L = constants("c L")

    eq = {diff(u, t, t): c**2 * diff(u, x, x)}

    bc = {
        (diff(u, x) if BC_0 == "free" else u).subs(x, 0): 0,
        (diff(u, x) if BC_L == "free" else u).subs(x, L): 0,
    }

    return BC_0, BC_L, L, bc, c, eq, t, u, x


@app.cell
def discretize_equation(
    BC_0, BC_L, L, bc, central, discretization, eq, indices, second_central, solve, space, t, to_implicit, u, x
):
    i, j = indices("i j")
    h, k = constants("h k")
    N, M = constants("N M", space=space.Z)

    d = (
        discretization()
        .space(t, i, h)
        .diff(t, second_central)
        .space(x, j, k)
        .diff(x, second_central)
        .diff(x, central)
    )

    u_disc = d(u)

    eq_d = d(eq)
    eq_d = solve(to_implicit(eq_d), u_disc[i + 1, j])

    bc_d = d(bc)
    bc_solve_for = (
        u_disc[:, -1] if BC_0 == "free" else u_disc[:, 0],
        u_disc[:, M + 1] if BC_L == "free" else u_disc[:, M],
    )
    bc_d = solve((f.subs(L / k, M) for f in to_implicit(bc_d)), bc_solve_for)

    j_min = -1 if BC_0 == "free" else 0
    j_max = M + 1 if BC_L == "free" else M

    return bc_d, eq_d, i, j, j_max, j_min, k, u_disc


@app.cell
def build_solver_module(M, N, bc_d, build_solver, c, eq_d, h, i, j, j_max, j_min, k, u_disc):
    solver = build_solver()
    solver.constants(c, h, k, N, M)
    solver.variables(u_disc, indices=((i, -1, N), (j, j_min, j_max)))
    solver.inputs(u_disc[0, :], u_disc[-1, :])
    with solver.steps(i, 0, N - 1) as step_time:
        step_time.explicit(bc_d)
        with step_time.steps(j, j_min + 1, j_max - 1) as step_space:
            step_space.explicit(eq_d)
    solver = solver.generate()
    return (solver,)


@app.cell
def controls(mo):
    sliders = mo.ui.dictionary(
        {
            "c": mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0, label=r"$c$ (wave speed)"),
            "h": mo.ui.slider(start=0.001, stop=0.05, step=0.001, value=0.01, label=r"$h$ (time step)"),
            "k": mo.ui.slider(start=0.001, stop=0.05, step=0.001, value=0.01, label=r"$k$ (space step)"),
            "amplitude": mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0, label="IC amplitude"),
            "width": mo.ui.slider(start=1.0, stop=10.0, step=0.5, value=5.0, label="IC width"),
            "ic_type": mo.ui.dropdown(options=["sech2", "sin"], value="sech2", label="IC type"),
        }
    )
    sliders
    return (sliders,)


@app.cell
def run_simulation(M, N, c, h, j, k, sin, sliders, solver, tanh, u_disc):
    sv = sliders.value

    # Compute N and M from h and k
    N_val = int(4.0 / sv["h"])
    M_val = int(1.0 / sv["k"])

    # Create initial condition based on type
    ic_type = sv["ic_type"]
    amplitude = sv["amplitude"]
    width = sv["width"]
    k_val = sv["k"]

    if ic_type == "sech2":
        ic = amplitude * (1 - tanh(j * k_val * width) ** 2)
    elif ic_type == "sin":
        ic = amplitude * sin(j * k_val)
    else:
        ic = amplitude * (1 - tanh(j * k_val * width) ** 2)

    result = solver.run(
        {
            c: sv["c"],
            h: sv["h"],
            k: k_val,
            N: N_val,
            M: M_val,
            u_disc[0, :]: ic,
            u_disc[-1, :]: ic,
        }
    )
    return result


@app.cell
def plot_solution(np, plt, result, sliders, u_disc):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    # Extract results and parameters
    u_result = result[u_disc]
    sv = sliders.value
    h_val = sv["h"]
    k_val = sv["k"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    T = np.arange(0, u_result.shape[0]) * h_val
    X = np.arange(0, u_result.shape[1]) * k_val
    T, X = np.meshgrid(T, X, indexing="ij")

    ax.plot_surface(T, X, u_result, cmap="coolwarm")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$u(t, x)$")
    ax.set_title("1D Wave Equation Solution")

    _fig = fig
    return


if __name__ == "__main__":
    app.run()

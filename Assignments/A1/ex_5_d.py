"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.5(d)

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from FEM.plot_style import setup_style, save_figure

setup_style()


def analytical_solution(x, psi, eps):
    # Numerically stable form derived in ex_5_stability.py:
    # u(x) = (1/psi) * [x(1 - e^{-psi/eps}) + e^{-psi/eps} - e^{psi(x-1)/eps}] / (1 - e^{-psi/eps})
    # This avoids calculating e^{psi/eps} which overflows for small eps.

    exp_neg_K = np.exp(-psi / eps)
    exp_x_minus_1_K = np.exp(psi * (x - 1) / eps)

    nom = x * (1 - exp_neg_K) + exp_neg_K - exp_x_minus_1_K
    denom = 1 - exp_neg_K

    return (1 / psi) * nom / denom


psi = 1.0
eps_values = np.array([1.0, 0.01, 0.0001])
L = 1.0
# Stable form handles boundaries fine, no need for EPS offset
x = np.linspace(0, L, 1000)

data = []
for eps in eps_values:
    u = analytical_solution(x, psi, eps)
    valid = ~np.isnan(u)
    for xi, ui in zip(x[valid], u[valid]):
        data.append({"x": xi, "u": ui, r"$\varepsilon$": f"{eps}"})

df = pd.DataFrame(data)

assert not df.empty, "DataFrame should not be empty"
assert df["u"].min() >= 0.0 and df["u"].max() <= 1.0, (
    "Analytical solution values out of expected [0, 1] range"
)

fig, ax = plt.subplots()
for eps, grp in df.groupby(r"$\varepsilon$"):
    ax.plot(grp["x"], grp["u"], label=eps)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.set_title(r"Analytical solution for different $\varepsilon$ values")
ax.legend(title=r"$\varepsilon$")

save_figure(fig, "figures/A1/ex_5/ex_5_d.pdf")

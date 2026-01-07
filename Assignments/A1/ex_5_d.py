import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from FEM.plot_style import setup_style, save_figure

setup_style()


def analytical_solution(x, psi, eps):
    nom = 1 + (np.exp(psi / eps) - 1) * x - np.exp(x * psi / eps)
    denom = np.exp(psi / eps) - 1
    return 1 / psi * nom / denom


# Parameters
psi = 1.0
eps_values = np.array([1.0, 0.1, 0.01, 0.002])
L = 1.0
EPS = 1e-10

# Create data for plotting
x = np.linspace(0 + EPS, L - EPS, 1000)

data = []
for eps in eps_values:
    u = analytical_solution(x, psi, eps)
    valid = ~np.isnan(u)
    for xi, ui in zip(x[valid], u[valid]):
        data.append({"x": xi, "u": ui, r"$\varepsilon$": f"{eps}"})

df = pd.DataFrame(data)

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
sns.lineplot(data=df, x="x", y="u", hue=r"$\varepsilon$", ax=ax)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.set_title(r"Analytical solution for different $\varepsilon$ values")
save_figure(fig, "ex_5_d.pdf")

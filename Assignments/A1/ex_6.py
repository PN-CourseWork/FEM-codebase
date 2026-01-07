import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from FEM import line_mesh, refine, refinement_error, mark_elements, project, interpolate

sns.set_theme(style="whitegrid")


def u(x):
    """Target function with sharp peak at x=0.4 and gentle bump at x=0.8."""
    return np.exp(-800 * (x - 0.4) ** 2) + 0.25 * np.exp(-40 * (x - 0.8) ** 2)


def run_amr(mesh, func, tol, alpha, use_relative):
    """Run AMR loop, return history of (n_elem, max_err) per iteration."""
    history = []

    while True:
        err = refinement_error(mesh, func)
        max_err = np.max(err)
        history.append({"n_elem": mesh.n_elem, "max_err": max_err})

        if max_err < tol:
            break

        if use_relative:
            marked = mark_elements(err, alpha=alpha)
        else:
            marked = mark_elements(err, alpha=alpha, tol=tol)

        if len(marked) == 0:
            break

        mesh = refine(mesh, marked)

    return mesh, pd.DataFrame(history)


# Parameters
tol = 1e-4
alpha_rel = 0.5
alpha_abs = 1.0

# Run AMR with both criteria
mesh_rel, hist_rel = run_amr(line_mesh(1.0, 3), u, tol, alpha_rel, use_relative=True)
mesh_abs, hist_abs = run_amr(line_mesh(1.0, 3), u, tol, alpha_abs, use_relative=False)

hist_rel["criterion"] = rf"Relative ($\alpha={alpha_rel}$)"
hist_abs["criterion"] = rf"Absolute ($\alpha={alpha_abs}$)"
hist_rel["iteration"] = range(len(hist_rel))
hist_abs["iteration"] = range(len(hist_abs))

# -----------------------------------------------------------------------------
# Plot 1: Solution on adapted mesh
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

x_fine = np.linspace(0, 1, 1000)
u_exact = u(x_fine)

for ax, mesh, title in zip(axes, [mesh_rel, mesh_abs], ["Relative", "Absolute"]):
    u_nodal = project(mesh, u)
    u_interp = interpolate(mesh, u_nodal, x_fine)

    ax.plot(x_fine, u_exact, "k-", lw=1.5, label="Exact")
    ax.plot(x_fine, u_interp, "--", lw=1.5, label="Interpolant")
    ax.plot(mesh.VX, u_nodal, "o", ms=4, label="Nodes")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(f"{title}: {mesh.n_elem} elements")
    ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("figures/A1/ex_6_solution.pdf")
plt.close()

# -----------------------------------------------------------------------------
# Plot 2: Element size distribution
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, mesh, title in zip(axes, [mesh_rel, mesh_abs], ["Relative", "Absolute"]):
    h = np.diff(mesh.VX)
    x_mid = 0.5 * (mesh.VX[:-1] + mesh.VX[1:])

    ax.bar(x_mid, h, width=h * 0.9, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Element size $h$")
    ax.set_title(f"{title}: {mesh.n_elem} elements")
    ax.set_yscale("log")

fig.tight_layout()
fig.savefig("figures/A1/ex_6_element_size.pdf")
plt.close()

# -----------------------------------------------------------------------------
# Plot 3: Performance comparison
# -----------------------------------------------------------------------------
df_perf = pd.concat([hist_rel, hist_abs], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Convergence: max error vs iteration
sns.lineplot(data=df_perf, x="iteration", y="max_err", hue="criterion",
             marker="o", ax=axes[0])
axes[0].axhline(tol, color="red", linestyle="--", label=f"tol={tol}")
axes[0].set_yscale("log")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel(r"$\max_i \Delta\mathrm{err}_i$")
axes[0].set_title("Convergence")
axes[0].legend()

# Efficiency: max error vs n_elem
sns.lineplot(data=df_perf, x="n_elem", y="max_err", hue="criterion",
             marker="o", ax=axes[1])
axes[1].axhline(tol, color="red", linestyle="--", label=f"tol={tol}")
axes[1].set_yscale("log")
axes[1].set_xlabel("Number of elements")
axes[1].set_ylabel(r"$\max_i \Delta\mathrm{err}_i$")
axes[1].set_title("Efficiency")
axes[1].legend()

fig.tight_layout()
fig.savefig("figures/A1/ex_6_performance.pdf")
plt.close()

# Print summary
print("AMR Results Summary")
print("=" * 50)
for name, mesh, hist in [("Relative", mesh_rel, hist_rel), ("Absolute", mesh_abs, hist_abs)]:
    print(f"{name}: {len(hist)} iters, {mesh.n_elem} elems, "
          f"max_err={hist['max_err'].iloc[-1]:.2e}")

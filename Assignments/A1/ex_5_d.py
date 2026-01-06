import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
x = np.linspace(0 + EPS, L-EPS, 1000)

data = []
for eps in eps_values:
    u = analytical_solution(x, psi, eps)
    valid = ~np.isnan(u)
    for xi, ui in zip(x[valid], u[valid]):
        data.append({'x': xi, 'u': ui, 'ε': f'{eps}'})

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='x', y='u', hue='ε')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Analytical Solution for Different ε Values')
plt.grid(True)
plt.savefig('figures/ex_5_d.pdf')
plt.close() 


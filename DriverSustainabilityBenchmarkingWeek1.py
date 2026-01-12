import numpy as np
import matplotlib.pyplot as plt
import time

# Import the DriverAMR17 function from the external Python script DriverAMR17.py that is written swith the function contained
from DriverAMR17 import DriverAMR17

# Define input parameters (DO NOT CHANGE THIS PART)
funu = lambda x: np.exp(-800 * (x - 0.4) ** 2) + 0.25 * np.exp(-40 * (x - 0.8) ** 2)
func = lambda x: (
    -1601 * np.exp(-800 * (x - 0.4) ** 2)
    + (-1600 * x + 640.0) ** 2 * np.exp(-800 * (x - 0.4) ** 2)
    - 20.25 * np.exp(-40 * (x - 0.8) ** 2)
    + 0.25 * (-80 * x + 64.0) ** 2 * np.exp(-40 * (x - 0.8) ** 2)
)
x = np.array([0.0, 0.5, 1.0])  # Initial mesh configuration - do not change
M = len(x)
L = 1
c = funu(x[0])
d = funu(x[-1])
tol = 1e-4
maxit = 50

# Let's call the FEM BVP 1D Solver with AMR
# time the code using time
fac = 100  # we do multiple runs to get the average time
start_time = time.time()
for i in range(fac):
    xAMR, u, iter = DriverAMR17(L, c, d, x, func, tol, maxit)
CPUtime = (time.time() - start_time) / fac

# Plot
plt.figure()
plt.plot(xAMR, funu(xAMR), linewidth=2)
plt.plot(xAMR, u, ".", markersize=15)
plt.xlabel("x")
plt.ylabel("u")
DOF = len(xAMR)
CO2eq = (
    CPUtime / 3600 * 86 / 1000 * 0.135
)  # valid for MacBook Pro (assumed power consumption 105)
plt.title(
    f"Group: 16, Iter: {iter}, Time: {CPUtime:.4f} s, DOF: {DOF}, CO2e={CO2eq:.4e} kg CO2"
)
# plt.legend(["Exact", "AMR"], loc="northeast")  # "northeast" is not valid
# See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
plt.legend(["Exact", "AMR"], loc="upper right")

# Element size distribution
h = np.diff(xAMR)
plt.figure()
plt.hist(h)
plt.xlabel("h")
plt.ylabel("# of elements")
plt.show()

import os
import time
import numpy as np
from driver28b import Driver28b  # Assuming Driver28b is a module
from driver28c import Driver28c  # Assuming Driver28c is a module
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# TODO: PUT IN YOUR GROUP NO AND STUDENT IDs FOR THE GROUP HERE
groupNo = "16"  # Replace 'X' with your group no.
groupStudentIDs = "s214960/s224409"  # Replace 'Y/Z' with your student id's.

# PATHS
dirThisScript = os.path.dirname(os.path.abspath(__file__))
dirStoreResults = "/Users/apek/02623/Handinresults/"

# PARAMETERS FOR SUSTAINABILITY CALCULATION
CO2intensity = 0.285  # [kg CO2/kWh]
PowerEstimate = 60  # [kW]

# PARAMETERS FOR THE SELECTED EXERCISES (DO NOT CHANGE)
x0, y0 = 0, 0
L1, L2 = 1, 1
noelms1, noelms2 = 40, 50
lam1, lam2 = 1, 1
fun = lambda x, y: np.cos(np.pi * x) * np.cos(np.pi * y)
qt = lambda x, y: 2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)



# EXECUTE CODE
# Call Group 30 solver
start_time = time.time()
VX, VY, EToV, U = Driver28b(x0, y0, L1, L2, noelms1, noelms2, lam1, lam2, fun, qt)
tend = time.time() - start_time
DOF1 = len(U.flatten())

x0, y0 = -1, -1
L1, L2 = 2, 2
start_time = time.time()
VX2, VY2, EToV2, U2 = Driver28c(x0, y0, L1, L2, noelms1, noelms2, lam1, lam2, fun, qt)
tend2 = time.time() - start_time
DOF2 = len(U2.flatten())

CPUtime1 = tend
CPUtime2 = tend2
CO2eq1 = CPUtime1 / 3600 * PowerEstimate / 1000 * CO2intensity
CO2eq2 = CPUtime2 / 3600 * PowerEstimate / 1000 * CO2intensity

# Print results to terminal
print(f"\n{'=' * 60}")
print(f"Group: {groupNo} ({groupStudentIDs})")
print(f"{'=' * 60}")
print(f"\nExercise 2.8b:")
print(
    f"  Time: {tend:.4e}s | DOF: {DOF1} | noelms: {noelms1}x{noelms2} | CO2e: {CO2eq1:.4e} kg"
)
print(f"\nExercise 2.8c:")
print(
    f"  Time: {tend2:.4e}s | DOF: {DOF2} | noelms: {noelms1}x{noelms2} | CO2e: {CO2eq2:.4e} kg"
)
print(f"{'=' * 60}\n")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
triang = Triangulation(VX, VY, EToV)
ax[0].tripcolor(triang, U, shading="flat")
# ax[0].set_title(f'2.8b. Group: {groupStudentIDs}, Time: {tend:.4e}, DOF: {DOF1}, noelsm1={noelms1}, noelms2={noelms2}, CO2e={CO2eq1:.4e}')
ax[0].set_title("Exercise 2.8b")

triang2 = Triangulation(VX2, VY2, EToV2)
ax[1].tripcolor(triang2, U2, shading="flat")
# ax[1].set_title(f'2.8c. Group: {groupStudentIDs}, Time: {tend2:.4e}, DOF: {DOF2}, noelsm1={noelms1}, noelms2={noelms2}, CO2e={CO2eq2:.4e}')
ax[1].set_title("Exercise 2.8c")

fig.tight_layout()

plt.show()

# STORE THE RESULTS
filename = f"Week2ResultsGroup{groupNo}.txt"
with open(os.path.join(dirStoreResults, filename), "w") as file:
    file.write(f"{groupNo}\n")
    file.write(f"{groupStudentIDs}\n")
    file.write(f"{CPUtime1:.4e} {DOF1} {CO2eq1:.4e}\n")
    file.write(f"{CPUtime2:.4e} {DOF2} {CO2eq2:.4e}\n")

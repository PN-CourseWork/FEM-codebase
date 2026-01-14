
import meshio
import numpy as np

mesh = meshio.read("results_cylinder.vtk")
u = mesh.point_data["u"]
v = mesh.point_data["v"]
vel = np.sqrt(u**2 + v**2)
idx_max = np.argmax(vel)
max_vel = vel[idx_max]
pos = mesh.points[idx_max]

print(f"Max velocity: {max_vel:.4f}")
print(f"Location: {pos}")

# Check velocity on cylinder
# How to find cylinder nodes? We need tags. VTK might not have tags unless saved.
# We can check by radius from (0.2, 0.2)
r = np.sqrt((mesh.points[:, 0] - 0.2)**2 + (mesh.points[:, 1] - 0.2)**2)
on_cyl = np.abs(r - 0.05) < 1e-3
print(f"Max velocity on cylinder surface: {np.max(vel[on_cyl]):.4f}")
print(f"Mean velocity on cylinder surface: {np.mean(vel[on_cyl]):.4f}")

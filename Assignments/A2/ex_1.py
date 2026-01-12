#from FEM.datastructures import Mesh2d
import numpy as np
from FEM.datastructures import Mesh2d 


# EXERCISE 2.1

# =========================================
# CASE 1:
# =========================================
case1_x0, case1_y0 = 0, 0
case1_L1 = 1
case1_L2 = 1
case1_noelms1 = 4
case1_noelms2 = 3

case1_expected_x = np.array([
    0, 0, 0, 0,
    0.2500, 0.2500, 0.2500, 0.2500,
    0.5000, 0.5000, 0.5000, 0.5000,
    0.7500, 0.7500, 0.7500, 0.7500,
    1.0000, 1.0000, 1.0000, 1.0000
])

case1_expected_y = np.array([
    1.0000, 0.6667, 0.3333, 0,
    1.0000, 0.6667, 0.3333, 0,
    1.0000, 0.6667, 0.3333, 0,
    1.0000, 0.6667, 0.3333, 0,
    1.0000, 0.6667, 0.3333, 0
])

case1_expected_elmtab = np.array([
    [1, 6, 5], [2, 6, 1], [2, 7, 6], [3, 7, 2], [3, 8, 7], [4, 8, 3],
    [5, 10, 9], [6, 10, 5], [6, 11, 10], [7, 11, 6], [7, 12, 11], [8, 12, 7],
    [9, 14, 13], [10, 14, 9], [10, 15, 14], [11, 15, 10], [11, 16, 15], [12, 16, 11],
    [13, 18, 17], [14, 18, 13], [14, 19, 18], [15, 19, 14], [15, 20, 19], [16, 20, 15]
])

case1_mesh = Mesh2d(x0=case1_x0,y0=case1_y0,L1=case1_L1,L2=case1_L2,noelms1=case1_noelms1,noelms2=case1_noelms2)

# Assertions for Case 1 (rtol=1e-3 due to rounded expected values)
np.testing.assert_allclose(case1_mesh.VX, case1_expected_x, rtol=1e-3, err_msg="Case 1 VX mismatch")
np.testing.assert_allclose(case1_mesh.VY, case1_expected_y, rtol=1e-3, err_msg="Case 1 VY mismatch")
np.testing.assert_array_equal(case1_mesh.EToV, case1_expected_elmtab, err_msg="Case 1 EToV mismatch")
print("Case 1: PASSED")

print(f"EToV case 1: {case1_mesh.EToV}")
print(f"VX case 1: {case1_mesh.VX}")
print(f"VY case 1: {case1_mesh.VY}")


# =========================================
# CASE 2:
# =========================================
case2_x0, case2_y0 = -2.5, -4.8
case2_L1 = 7.6
case2_L2 = 5.9
case2_noelms1 = 4
case2_noelms2 = 3

case2_expected_x = np.array([
    -2.5000, -2.5000, -2.5000, -2.5000,
    -0.6000, -0.6000, -0.6000, -0.6000,
    1.3000, 1.3000, 1.3000, 1.3000,
    3.2000, 3.2000, 3.2000, 3.2000,
    5.1000, 5.1000, 5.1000, 5.1000
])

case2_expected_y = np.array([
    1.1000, -0.8667, -2.8333, -4.8000,
    1.1000, -0.8667, -2.8333, -4.8000,
    1.1000, -0.8667, -2.8333, -4.8000,
    1.1000, -0.8667, -2.8333, -4.8000,
    1.1000, -0.8667, -2.8333, -4.8000
])

case2_expected_elmtab = case1_expected_elmtab  # Same as Case 1

case2_mesh = Mesh2d(x0=case2_x0,y0=case2_y0,L1=case2_L1,L2=case2_L2,noelms1=case2_noelms1,noelms2=case2_noelms2)

# Assertions for Case 2 (rtol=1e-3 due to rounded expected values)
np.testing.assert_allclose(case2_mesh.VX, case2_expected_x, rtol=1e-3, err_msg="Case 2 VX mismatch")
np.testing.assert_allclose(case2_mesh.VY, case2_expected_y, rtol=1e-3, err_msg="Case 2 VY mismatch")
np.testing.assert_array_equal(case2_mesh.EToV, case2_expected_elmtab, err_msg="Case 2 EToV mismatch")
print("Case 2: PASSED")


print(f"EToV case 2: {case2_mesh.EToV}")
print(f"VX case 2: {case2_mesh.VX}")
print(f"VY case 2: {case2_mesh.VY}")





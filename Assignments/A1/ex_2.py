### 1.2.a)
import numpy as np
import matplotlib.pyplot as plt

def BVP1D(L, c, d, x):
    """
    Solve u'' - u = 0 on [0,L] using linear FEM.

    Parameters
    ----------
    L : float
        Domain length
    c : float
        Left boundary condition u(0)=c
    d : float
        Right boundary condition u(L)=d
    x : array_like
        Mesh nodes (length M)

    Returns
    -------
    u : ndarray
        FEM solution at the nodes
    """

    x = np.asarray(x)
    M = len(x)

    # --------------------------------------------------
    # GLOBAL ASSEMBLY (Algorithm 1)
    # --------------------------------------------------
    A = np.zeros((M, M))
    b = np.zeros(M)

    for i in range(M - 1):
        h = x[i+1] - x[i]

        # Element matrix entries from Exercise 1.1
        k11 =  1.0/h + h/3.0
        k12 = -1.0/h + h/6.0
        k22 =  1.0/h + h/3.0

        # Assemble upper triangular part
        A[i,   i]   += k11
        A[i,   i+1] += k12
        A[i+1, i+1] += k22

    # Mirror upper triangle to lower triangle
    A = A + np.triu(A, 1).T

   # --------------------------------------------------
    # IMPOSE DIRICHLET BOUNDARY CONDITIONS (Algorithm 2)
    # --------------------------------------------------

    # Left boundary: u(0) = c
    b[0] = c
    b[1] = b[1] - A[0, 1] * c

    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[1, 0] = 0.0

    # Right boundary: u(L) = d
    b[-1] = d
    b[-2] = b[-2] - A[-2, -1] * d

    A[-1, -1] = 1.0
    A[-2, -1] = 0.0
    A[-1, -2] = 0.0

    # --------------------------------------------------
    # SOLVE SYSTEM (Cholesky)
    # --------------------------------------------------
    U = np.linalg.cholesky(A)
    u = np.linalg.solve(U.T, np.linalg.solve(U, b))

    return u

### TEST CASE 1.2.a)
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Test data
# --------------------------------------------------
L = 2.0
c = 1.0
d = np.exp(2)

# Mesh for current FEM solver (uniform, M = 11)
M = 11
x, u = BVP1D(L, c, d, M)

# Exact solution
u_exact = np.exp(x)

# --------------------------------------------------
# Reference nodal solutions from Exercise 1.1
# --------------------------------------------------
# Uniform mesh (3 nodes)
x_uniform = np.array([0.0, 1.0, 2.0])
u_hat_uniform_case = np.array([
    1.0,
    (5/16)*(1 + np.exp(2)),
    np.exp(2)
])

# Non-uniform mesh (3 nodes)
x_non_uniform = np.array([0.0, 4/3, 2.0])
u_hat_non_uniform_case = np.array([
    1.0,
    19/101 + (50/101)*np.exp(2),
    np.exp(2)
])

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8,5))

# Exact solution
plt.plot(x, u_exact, 'k-', label='Exact solution $u(x)=e^x$', linewidth=2)

# Current FEM solution
plt.plot(x, u, 'r:', label='Current FEM solution', linewidth=2)

# Uniform mesh FEM nodal solution
plt.plot(x_uniform, u_hat_uniform_case, 'b-x', label='Uniform mesh FEM (3 nodes)', linewidth=2)

# Non-uniform mesh FEM nodal solution
plt.plot(x_non_uniform, u_hat_non_uniform_case, 'g:^', label='Non-uniform mesh FEM (3 nodes)', linewidth=2)

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.legend()
plt.grid(True)
plt.title('Comparison of FEM solutions and exact solution')
plt.show()

### 1.2.b)
import numpy as np

def BVP1D(L, c, d, M):
    """
    Solve u'' - u = 0 on [0,L] using linear FEM
    on a uniform mesh with M nodes.
    """

    # --------------------------------------------------
    # CREATE UNIFORM MESH
    # --------------------------------------------------
    x = np.linspace(0.0, L, M)
    h = L / (M - 1)

    # --------------------------------------------------
    # GLOBAL ASSEMBLY (Algorithm 1, simplified)
    # --------------------------------------------------
    A = np.zeros((M, M))
    b = np.zeros(M)

    # Element matrix entries (same for all elements)
    k11 =  1.0/h + h/3.0
    k12 = -1.0/h + h/6.0
    k22 =  1.0/h + h/3.0

    for i in range(M - 1):
        A[i,   i]   += k11
        A[i,   i+1] += k12
        A[i+1, i+1] += k22

    # Mirror upper triangle
    A = A + np.triu(A, 1).T

    # --------------------------------------------------
    # IMPOSE DIRICHLET BOUNDARY CONDITIONS (Algorithm 2)
    # --------------------------------------------------

    # Left boundary
    b[0] = c
    b[1] -= A[0, 1] * c
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[1, 0] = 0.0

    # Right boundary
    b[-1] = d
    b[-2] -= A[-2, -1] * d
    A[-1, -1] = 1.0
    A[-2, -1] = 0.0
    A[-1, -2] = 0.0

    # --------------------------------------------------
    # SOLVE SYSTEM (Cholesky)
    # --------------------------------------------------
    U = np.linalg.cholesky(A)
    u = np.linalg.solve(U.T, np.linalg.solve(U, b))

    return x, u

### TEST CASE 1.2.b)
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Test data
# --------------------------------------------------
L = 2.0
c = 1.0
d = np.exp(2)
M = 11   # number of nodes (refined uniform mesh)

# Solve FEM problem (uniform mesh with M nodes)
x, u = BVP1D(L, c, d, M)

# Exact solution
u_exact = np.exp(x)

# --------------------------------------------------
# Coarse FEM solutions from Exercise 1.1
# --------------------------------------------------

# Uniform mesh (3 nodes)
x_uniform = np.array([0.0, 1.0, 2.0])
u_hat_uniform_case = np.array([
    1.0,
    (5/16)*(1 + np.exp(2)),
    np.exp(2)
])

# Non-uniform mesh (3 nodes)
x_non_uniform = np.array([0.0, 4/3, 2.0])
u_hat_non_uniform_case = np.array([
    1.0,
    19/101 + (50/101)*np.exp(2),
    np.exp(2)
])

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8,5))

# Exact solution
plt.plot(x, u_exact, 'k-', label='Exact solution $u(x)=e^x$', linewidth=2)

# Current FEM solution
plt.plot(x, u, 'r:', label='FEM solution (uniform, M=11)', linewidth=2)

# Uniform coarse FEM
plt.plot(x_uniform, u_hat_uniform_case, 'b-x',
         label='Uniform mesh FEM (3 nodes)', linewidth=2, markersize=8)

# Non-uniform coarse FEM
plt.plot(x_non_uniform, u_hat_non_uniform_case, 'g:^',
         label='Non-uniform mesh FEM (3 nodes)', linewidth=2, markersize=8)

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.legend()
plt.grid(True)
plt.title('Comparison of FEM solutions and exact solution')
plt.show()

### Convergence test 1.2.d)
import numpy as np
import matplotlib.pyplot as plt

# Problem data
L = 2.0
c = 1.0
d = np.exp(2)

# Exact solution
def u_exact(x):
    return np.exp(x)

# Mesh refinements (number of nodes)
M_values = [11, 21, 41, 81, 161]

h_values = []
error_values = []

for M in M_values:
    x, u = BVP1D(L, c, d, M)
    h = L / (M - 1)

    error = np.max(np.abs(u - u_exact(x)))

    h_values.append(h)
    error_values.append(error)

h_values = np.array(h_values)
error_values = np.array(error_values)

# --------------------------------------------------
# Estimate convergence rate p
# --------------------------------------------------
p_est = np.polyfit(np.log(h_values), np.log(error_values), 1)[0]

print("h values:     ", h_values)
print("errors:       ", error_values)
print("Estimated p ≈ ", p_est)

# --------------------------------------------------
# Plot error vs h (log-log)
# --------------------------------------------------
plt.figure(figsize=(7,5))
plt.loglog(h_values, error_values, 'o-', label='Numerical error')
plt.loglog(h_values, error_values[0]*(h_values/h_values[0])**2,
           '--', label=r'Reference slope $h^2$')

plt.xlabel('$h$')
plt.ylabel(r'$\max | \hat u(x) - u(x) |$')
plt.legend()
plt.grid(True, which='both')
plt.title('Convergence of FEM solution')
plt.show()

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix

from .elements import advection
from .datastructures import Mesh

# Pre-computed Gauss quadrature points and weights (cached at module level)
_GAUSS_QUAD = {
    1: (np.array([0.0]), np.array([2.0])),
    2: (np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]), np.array([1.0, 1.0])),
    3: (np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]), np.array([5 / 9, 8 / 9, 5 / 9])),
    5: (
        np.array([
            -np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
            -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
            0.0,
            np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
            np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
        ]),
        np.array([
            (322 - 13 * np.sqrt(70)) / 900,
            (322 + 13 * np.sqrt(70)) / 900,
            128 / 225,
            (322 + 13 * np.sqrt(70)) / 900,
            (322 - 13 * np.sqrt(70)) / 900,
        ]),
    ),
}


@njit
def _assemble_csr_1d_core(Ke_all, cells, n_nodes):
    """
    Directly assemble CSR arrays for 1D problems.

    """
    n_elem = len(Ke_all)

    indptr = np.empty(n_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    nnz = 3 * n_nodes - 2
    indices = np.empty(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float64)

    # Fill indptr
    indptr[0] = 0
    indptr[1] = 2
    for i in range(2, n_nodes):
        indptr[i] = indptr[i - 1] + 3
    indptr[n_nodes] = indptr[n_nodes - 1] + 2

    # Fill indices (assuming sorted structure)
    indices[0] = 0
    indices[1] = 1

    idx = 2
    for i in range(1, n_nodes - 1):
        indices[idx] = i - 1
        indices[idx + 1] = i
        indices[idx + 2] = i + 1
        idx += 3

    # Row N-1: N-2, N-1
    indices[idx] = n_nodes - 2
    indices[idx + 1] = n_nodes - 1

    # Now fill data by iterating over elements
    for e in range(n_elem):
        n1 = cells[e, 0]
        n2 = cells[e, 1]

        if n1 == 0:
            idx_n1_n1 = 0
            idx_n1_n2 = 1  
        elif n1 == n_nodes - 1:
            idx_n1_n1 = indptr[n1] + 1
            idx_n1_n2 = indptr[n1]  
        else:
            idx_n1_n1 = indptr[n1] + 1
            if n2 == n1 + 1:
                idx_n1_n2 = indptr[n1] + 2
            else:
                idx_n1_n2 = indptr[n1]

        data[idx_n1_n1] += Ke_all[e, 0, 0]
        data[idx_n1_n2] += Ke_all[e, 0, 1]

        # Contribution to Row n2
        if n2 == 0:
            idx_n2_n2 = 0
            idx_n2_n1 = 1
        elif n2 == n_nodes - 1:
            idx_n2_n2 = indptr[n2] + 1
            idx_n2_n1 = indptr[n2]
        else:
            idx_n2_n2 = indptr[n2] + 1
            if n1 == n2 + 1:
                idx_n2_n1 = indptr[n2] + 2
            else:
                idx_n2_n1 = indptr[n2]

        data[idx_n2_n1] += Ke_all[e, 1, 0]
        data[idx_n2_n2] += Ke_all[e, 1, 1]

    return data, indices, indptr


def assemble_matrix_csr_1d(Ke_all, cells, n_nodes):
    """Assemble global matrix from element matrices (1D sorted meshes only)."""
    data, indices, indptr = _assemble_csr_1d_core(Ke_all, cells, n_nodes)
    return csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))


@njit
def _assemble_diffusion_core(h, lam):
    n_elem = len(h)
    Ke_all = np.zeros((n_elem, 2, 2))
    for e in range(n_elem):
        c = lam / h[e]
        Ke_all[e, 0, 0] = c
        Ke_all[e, 0, 1] = -c
        Ke_all[e, 1, 0] = -c
        Ke_all[e, 1, 1] = c
    return Ke_all


def assemble_diffusion(mesh: Mesh, lam=1.0) -> csr_matrix:
    """
    Assemble global diffusion matrix for 1D P1 elements.

    """
    h = _element_lengths_1d(mesh)
    Ke_all = _assemble_diffusion_core(h, lam)
    return assemble_matrix_csr_1d(Ke_all, mesh.EToV, mesh.nonodes)


def _element_lengths_1d(mesh: Mesh) -> np.ndarray:
    """Compute element lengths for 1D meshes."""
    x_left = mesh.VX[mesh.EToV[:, 0]].ravel()
    x_right = mesh.VX[mesh.EToV[:, 1]].ravel()
    h = np.abs(x_right - x_left)
    if np.any(h <= 0.0):
        raise ValueError(
            f"Non-positive element length detected: h={h}, VX={mesh.VX}, EToV={mesh.EToV}"
        )
    return h


@njit
def _assemble_mass_core(h, coeff):
    n_elem = len(h)
    Ke_all = np.zeros((n_elem, 2, 2))
    for e in range(n_elem):
        c = coeff * h[e] / 6.0
        Ke_all[e, 0, 0] = 2.0 * c
        Ke_all[e, 0, 1] = c
        Ke_all[e, 1, 0] = c
        Ke_all[e, 1, 1] = 2.0 * c
    return Ke_all


@njit
def _assemble_diffusion_mass_core(h, lam, coeff):
    n_elem = len(h)
    Ke_all = np.zeros((n_elem, 2, 2))
    for e in range(n_elem):
        hi = h[e]

        # Diffusion part
        c_diff = lam / hi

        # Mass part
        c_mass = coeff * hi / 6.0

        # Combine
        Ke_all[e, 0, 0] = c_diff + 2.0 * c_mass
        Ke_all[e, 0, 1] = -c_diff + c_mass
        Ke_all[e, 1, 0] = -c_diff + c_mass
        Ke_all[e, 1, 1] = c_diff + 2.0 * c_mass

    return Ke_all


@njit
def _assemble_diffusion_mass_csr_direct(VX, EToV, n_nodes, lam, coeff):
    """
    Fused assembly: compute element matrices and CSR data in one pass.
    Avoids intermediate Ke_all array and separate element length computation.
    """
    n_elem = EToV.shape[0]

    # CSR structure for 1D sorted mesh
    nnz = 3 * n_nodes - 2
    indptr = np.empty(n_nodes + 1, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.float64)

    # Fill indptr
    indptr[0] = 0
    indptr[1] = 2
    for i in range(2, n_nodes):
        indptr[i] = indptr[i - 1] + 3
    indptr[n_nodes] = indptr[n_nodes - 1] + 2

    # Fill indices
    indices[0] = 0
    indices[1] = 1
    idx = 2
    for i in range(1, n_nodes - 1):
        indices[idx] = i - 1
        indices[idx + 1] = i
        indices[idx + 2] = i + 1
        idx += 3
    indices[idx] = n_nodes - 2
    indices[idx + 1] = n_nodes - 1

    # Assemble directly into CSR data
    for e in range(n_elem):
        n1 = EToV[e, 0]
        n2 = EToV[e, 1]

        # Compute element length inline
        hi = VX[n2] - VX[n1]

        # Compute element matrix entries
        c_diff = lam / hi
        c_mass = coeff * hi / 6.0
        k00 = c_diff + 2.0 * c_mass
        k01 = -c_diff + c_mass
        k11 = c_diff + 2.0 * c_mass

        # Row n1
        if n1 == 0:
            idx_n1_n1 = 0
            idx_n1_n2 = 1
        else:
            idx_n1_n1 = indptr[n1] + 1
            idx_n1_n2 = indptr[n1] + 2

        data[idx_n1_n1] += k00
        data[idx_n1_n2] += k01

        # Row n2
        if n2 == n_nodes - 1:
            idx_n2_n2 = indptr[n2] + 1
            idx_n2_n1 = indptr[n2]
        else:
            idx_n2_n2 = indptr[n2] + 1
            idx_n2_n1 = indptr[n2]

        data[idx_n2_n1] += k01  # k10 = k01 (symmetric)
        data[idx_n2_n2] += k11

    return data, indices, indptr


def assemble_diffusion_mass(
    mesh: Mesh, lam: float = 1.0, coeff: float = 1.0
) -> csr_matrix:
    """
    Assemble global stiffness matrix for -lam*u'' + coeff*u.
    Combines diffusion and mass assembly to avoid expensive sparse matrix addition.
    """
    data, indices, indptr = _assemble_diffusion_mass_csr_direct(
        mesh.VX, mesh.EToV, mesh.nonodes, lam, coeff
    )
    return csr_matrix((data, indices, indptr), shape=(mesh.nonodes, mesh.nonodes))


def assemble_mass(mesh: Mesh, coeff: float = 1.0) -> csr_matrix:
    """
    Assemble global mass matrix for 1D P1 elements.
    """
    h = _element_lengths_1d(mesh)
    Ke_all = _assemble_mass_core(h, coeff)
    return assemble_matrix_csr_1d(Ke_all, mesh.EToV, mesh.nonodes)


def assemble_advection(mesh: Mesh, psi: float) -> csr_matrix:
    """
    Assemble global advection matrix for 1D P1 elements.
    """
    h = _element_lengths_1d(mesh)
    Ke_all = np.zeros((mesh.noelms, 2, 2))
    for e, hi in enumerate(h):
        Ke_all[e] = advection(hi, psi)
    return assemble_matrix_csr_1d(Ke_all, mesh.EToV, mesh.nonodes)


@njit
def _assemble_load_core(f_vals, pts, wts, h, cells, nonodes):
    b = np.zeros(nonodes)
    n_elem, n_quad = f_vals.shape

    for e in range(n_elem):
        detJ = h[e] / 2.0
        val0 = 0.0
        val1 = 0.0

        for q in range(n_quad):
            pt = pts[q]
            wt = wts[q]
            f_val = f_vals[e, q]

            # Basis functions
            N1 = 0.5 * (1 - pt)
            N2 = 0.5 * (1 + pt)

            common = f_val * wt * detJ
            val0 += common * N1
            val1 += common * N2

        b[cells[e, 0]] += val0
        b[cells[e, 1]] += val1

    return b


def assemble_load(mesh: Mesh, f_func, n_quad: int = 5) -> np.ndarray:
    """
    Assemble global load vector using Gaussian quadrature for 1D P1 elements.
    """
    if n_quad not in _GAUSS_QUAD:
        raise ValueError(f"Unsupported n_quad={n_quad}. Use 1, 2, 3, or 5.")

    pts, wts = _GAUSS_QUAD[n_quad]

    # Get element coordinates
    x_left = mesh.VX[mesh.EToV[:, 0]][:, None]
    x_right = mesh.VX[mesh.EToV[:, 1]][:, None]

    x_phys = x_left + 0.5 * (pts[None, :] + 1) * (x_right - x_left)

    n_elem = mesh.noelms
    f_vals = f_func(x_phys.ravel()).reshape(n_elem, n_quad)

    h = (x_right - x_left).ravel()

    return _assemble_load_core(f_vals, pts, wts, h, mesh.EToV, mesh.nonodes)

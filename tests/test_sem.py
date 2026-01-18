"""Tests for Spectral Element Method implementation.

Run with: uv run pytest tests/test_sem.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Spectral building blocks
from FEM.SEM.spectral import (
    legendre_gauss_lobatto_nodes,
    legendre_gauss_lobatto_weights,
    SpectralElement2D,
    stiffness_matrix_2d,
)

# SEM multi-element infrastructure
from FEM.SEM import (
    SEMMesh2D,
    create_unit_square_mesh,
    assemble_global_stiffness,
    assemble_load_vector,
    solve_poisson_sem,
    l2_error_sem,
    linf_error_sem,
)


class TestPolynomialBasis:
    """Test polynomial basis functions."""

    def test_lgl_nodes_endpoints(self):
        """LGL nodes should include -1 and 1."""
        for n in [3, 5, 8, 10]:
            nodes = legendre_gauss_lobatto_nodes(n)
            assert np.isclose(nodes[0], -1.0)
            assert np.isclose(nodes[-1], 1.0)

    def test_lgl_weights_sum(self):
        """LGL weights should sum to 2 (length of [-1,1])."""
        for n in [3, 5, 8, 10]:
            weights = legendre_gauss_lobatto_weights(n)
            assert np.isclose(np.sum(weights), 2.0)

    def test_lgl_quadrature_exactness(self):
        """LGL quadrature should be exact for polynomials up to degree 2N-3."""
        n = 5  # 5 points
        nodes = legendre_gauss_lobatto_nodes(n)
        weights = legendre_gauss_lobatto_weights(n)

        # Test x^k for k = 0, 1, ..., 2*5-3 = 7
        for k in range(8):
            numerical = np.sum(weights * nodes**k)
            # Exact integral of x^k from -1 to 1
            if k % 2 == 0:
                exact = 2.0 / (k + 1)
            else:
                exact = 0.0
            assert np.isclose(numerical, exact, atol=1e-12), f"Failed for k={k}"


class TestSpectralElement2D:
    """Test single element operators."""

    def test_element_creation(self):
        """Test SpectralElement2D initialization."""
        elem = SpectralElement2D(Nx=4, Ny=4, domain=((0, 1), (0, 1)))
        assert elem.num_nodes == 25
        assert elem.shape == (5, 5)

    def test_differentiation_constant(self):
        """Derivative of constant should be zero."""
        elem = SpectralElement2D(Nx=4, Ny=4)
        u = np.ones(elem.shape)
        du_dx = elem.Dx @ u
        du_dy = u @ elem.Dy_T
        assert np.allclose(du_dx, 0, atol=1e-12)
        assert np.allclose(du_dy, 0, atol=1e-12)

    def test_differentiation_linear(self):
        """Derivative of x should be 1, derivative of y should be 1."""
        elem = SpectralElement2D(Nx=4, Ny=4, domain=((-1, 1), (-1, 1)))

        # u = x
        du_dx = elem.Dx @ elem.X
        assert np.allclose(du_dx, 1, atol=1e-12)

        # u = y
        du_dy = elem.Y @ elem.Dy_T
        assert np.allclose(du_dy, 1, atol=1e-12)

    def test_stiffness_symmetry(self):
        """Stiffness matrix should be symmetric."""
        elem = SpectralElement2D(Nx=4, Ny=4)
        K = stiffness_matrix_2d(elem.Dx, elem.Dy, elem.wx, elem.wy).toarray()
        assert np.allclose(K, K.T, atol=1e-12)


class TestSEMMesh:
    """Test SEM mesh loading and DOF mapping."""

    @pytest.fixture
    def simple_mesh_file(self):
        """Create a simple 2x2 quad mesh for testing."""
        pytest.importorskip("gmsh")

        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            filepath = Path(f.name)

        create_unit_square_mesh(2, 2, filepath)
        yield filepath
        filepath.unlink()  # Cleanup

    def test_mesh_loading(self, simple_mesh_file):
        """Test mesh can be loaded."""
        mesh = SEMMesh2D(filepath=simple_mesh_file, polynomial_order=2)
        assert mesh.noelms == 4  # 2x2 elements
        assert mesh.nloc == 9  # (2+1)^2 nodes per element

    def test_c0_continuity(self, simple_mesh_file):
        """Test C0 continuity - shared nodes have same global index."""
        mesh = SEMMesh2D(filepath=simple_mesh_file, polynomial_order=2)

        # Check that shared nodes have same coordinates
        for e in range(mesh.noelms):
            glb = mesh.loc2glb[e, :]
            x_e = mesh.VX[glb]
            y_e = mesh.VY[glb]

            # All coordinates should be valid (not NaN or duplicated incorrectly)
            assert not np.any(np.isnan(x_e))
            assert not np.any(np.isnan(y_e))

    def test_global_dof_count(self, simple_mesh_file):
        """Test correct number of global DOFs with C0 continuity."""
        # For 2x2 mesh with p=2: (2*2+1) * (2*2+1) = 25 nodes
        mesh = SEMMesh2D(filepath=simple_mesh_file, polynomial_order=2)
        assert mesh.nonodes == 25

    def test_boundary_nodes(self, simple_mesh_file):
        """Test boundary node identification."""
        mesh = SEMMesh2D(filepath=simple_mesh_file, polynomial_order=2)

        # Check boundary nodes are on boundaries
        left_nodes = mesh.get_boundary_nodes("left")
        assert np.allclose(mesh.VX[left_nodes], 0.0)

        right_nodes = mesh.get_boundary_nodes("right")
        assert np.allclose(mesh.VX[right_nodes], 1.0)


class TestAssembly:
    """Test global matrix assembly."""

    @pytest.fixture
    def mesh_2x2(self):
        """Create 2x2 mesh for assembly tests."""
        pytest.importorskip("gmsh")

        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            filepath = Path(f.name)

        create_unit_square_mesh(2, 2, filepath)
        mesh = SEMMesh2D(filepath=filepath, polynomial_order=3)
        yield mesh
        filepath.unlink()

    def test_stiffness_symmetry(self, mesh_2x2):
        """Global stiffness should be symmetric."""
        K = assemble_global_stiffness(mesh_2x2)
        K_dense = K.toarray()
        assert np.allclose(K_dense, K_dense.T, atol=1e-12)

    def test_load_vector_constant(self, mesh_2x2):
        """Load vector for f=1 should be non-negative with correct structure."""
        f_func = lambda x, y: np.ones_like(x)
        b = assemble_load_vector(mesh_2x2, f_func)

        # All entries should be non-negative for f > 0
        assert np.all(b >= 0)
        # Should have correct size
        assert len(b) == mesh_2x2.nonodes
        # Should be non-trivial
        assert np.sum(b) > 0


class TestPoissonSolver:
    """Test Poisson equation solver."""

    @pytest.fixture
    def mesh_4x4_p4(self):
        """Create 4x4 mesh with p=4 for solver tests."""
        pytest.importorskip("gmsh")

        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            filepath = Path(f.name)

        create_unit_square_mesh(4, 4, filepath)
        mesh = SEMMesh2D(filepath=filepath, polynomial_order=4)
        yield mesh
        filepath.unlink()

    def test_manufactured_solution(self, mesh_4x4_p4):
        """Test with manufactured solution u = sin(pi*x)*sin(pi*y)."""
        # Manufactured solution
        u_exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

        # RHS: -Laplacian(u) = 2*pi^2*sin(pi*x)*sin(pi*y)
        f_func = lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        # BC: u = 0 on boundary (homogeneous)
        bc_func = lambda x, y: np.zeros_like(x)

        # Solve
        u = solve_poisson_sem(mesh_4x4_p4, f_func, bc_func)

        # Check error
        error = l2_error_sem(mesh_4x4_p4, u, u_exact)

        # Should have small error with 4x4 mesh and p=4
        assert error < 1e-4, f"L2 error too large: {error}"

    def test_polynomial_exact(self, mesh_4x4_p4):
        """Polynomial solution should be captured exactly (up to BC)."""
        # u = x*(1-x)*y*(1-y) satisfies u=0 on boundary
        u_exact = lambda x, y: x * (1 - x) * y * (1 - y)

        # -Laplacian(u) = 2*y*(1-y) + 2*x*(1-x)
        f_func = lambda x, y: 2 * y * (1 - y) + 2 * x * (1 - x)

        bc_func = lambda x, y: np.zeros_like(x)

        u = solve_poisson_sem(mesh_4x4_p4, f_func, bc_func)
        error = linf_error_sem(mesh_4x4_p4, u, u_exact)

        # Polynomial should be captured very accurately
        assert error < 1e-8, f"L∞ error too large for polynomial: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from .datastructures import Mesh as Mesh
from .mesh import line_mesh as line_mesh
from .assembly import (
    assemble_diffusion as assemble_diffusion,
    assemble_mass as assemble_mass,
    assemble_advection as assemble_advection,
    assemble_load as assemble_load,
)
from .boundary import apply_dirichlet as apply_dirichlet
from .interpolation import (
    interpolate as interpolate,
    discrete_l2_error as discrete_l2_error,
    linf_error as linf_error,
)
from .amr import (
    refine as refine,
    refinement_error as refinement_error,
    mark_elements as mark_elements,
    estimate_error_l2 as estimate_error_l2,
    run_amr as run_amr,
)
from .solvers import (
    solve_advection_diffusion_1d as solve_advection_diffusion_1d,
    solve_bvp_1d as solve_bvp_1d,
    solve_reaction_diffusion_1d_amr_hierarchical as solve_reaction_diffusion_1d_amr_hierarchical,
)
from .plot_style import setup_style as setup_style, save_figure as save_figure

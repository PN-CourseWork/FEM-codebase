"""
Shared plot styling for FEM assignments.
Uses seaborn with LaTeX fonts for consistent, publication-quality figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Output directory for figures
FIGURES_DIR = Path("figures") / "A1"


def setup_style():
    """Configure matplotlib and seaborn for consistent LaTeX-compatible plots."""
    # Use seaborn style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })


def get_colors(n=None):
    """Get seaborn color palette."""
    return sns.color_palette("deep", n)


def save_figure(fig, filename):
    """Save figure to the standard output directory."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved: {filepath}")
    return filepath

import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path("figures")
STYLE_PATH = Path(__file__).resolve().parent / "fem.mplstyle"


def setup_style():
    """Apply shared matplotlib style."""
    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)
    plt.rcParams.setdefault("savefig.bbox", "tight")


def save_figure(fig, filename: str | Path):
    """
    Save figure to the specified path.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved: {filepath}")
    return filepath

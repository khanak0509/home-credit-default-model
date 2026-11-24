from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLORS = {
    "legacy": "#2a78d6",
    "scorecard": "#eb6834",
    "xgboost": "#1baf7a",
    "lightgbm": "#6250d6",
    "lightgbm_monotone": "#e87ba4",
}
NEUTRAL = "#8a8984"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
ACCENT = "#2a78d6"
SECOND = "#eb6834"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    })


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

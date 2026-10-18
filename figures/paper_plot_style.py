"""Shared style and data access for every figure/table script (column 3.33 in, text 7.0 in)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
RESULTS = FIG_DIR.parent / "results"
COLUMN_IN, TEXT_IN = 3.33, 7.0
N_VALUES = ("1", "2", "4", "8", "16")

matplotlib.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3dc", "grid.linewidth": 0.5, "axes.edgecolor": "#6b6a64",
    "axes.linewidth": 0.6, "xtick.color": "#4a4944", "ytick.color": "#4a4944",
    "pdf.fonttype": 42, "lines.linewidth": 1.4, "mathtext.fontset": "stix",
})

# Validated categorical slots (dataviz reference palette, light mode): fixed entity -> color, never cycled.
COLOR = {"D1": "#2a78d6", "D1p": "#eb6834", "D2": "#1baf7a", "D3o": "#eda100"}
MARKER = {"D1": "o", "D1p": "s", "D2": "^", "D3o": "D"}
LABEL = {"D1": "D1 shared scaling", "D1p": "D1′ per-field scaling", "D2": "D2 fixed-axis order-pres.",
         "D3o": "D3o Prεεmpt (released)"}
FLOOR_COLOR = "#6b6a64"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def save(fig, name: str) -> None:
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def signed(x: float) -> str:
    """One-decimal signed percentage points; -0.0 is shown as 0.0."""
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def signed_ci(v: float, ci) -> str:
    return f"{signed(v)} [{signed(ci[0])}, {signed(ci[1])}]"

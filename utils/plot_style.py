from __future__ import annotations

import matplotlib as mpl


def apply_ieee_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "figure.titlesize": 9,
            "axes.grid": False,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


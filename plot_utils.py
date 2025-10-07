from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------

PLOT_RC_PARAMS: dict[str, float | int | str] = {
    "font.family": "Arial",
    "font.weight": "bold",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "xtick.major.size": 5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 3,
    "xtick.minor.width": 1,
    "ytick.major.size": 5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 3,
    "ytick.minor.width": 1,
    "legend.fontsize": 18,
    "legend.frameon": False,
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def metrics_text(
    df: pd.DataFrame,
    metrics: list[str] = ["mae", "r2"],
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
) -> str:
    """
    Create a text string containing the metrics and their values.

    Args:
        df (pd.DataFrame): DataFrame containing the true and pred values.
        metrics (list[str]): A list of metrics to be displayed in the plot.
        metrics_precision (str): Format string for the metrics.
        unit (str | None): Unit of the property.
        unit_scale (float): Scale factor for the unit.
    Returns:
        text (str): A text string containing the metrics and their values.
    """

    values: dict[str, float] = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower == "mae":
            values["MAE"] = np.mean(np.abs(df["true"] - df["pred"])) * unit_scale
        elif m_lower == "mse":
            values["MSE"] = np.mean((df["true"] - df["pred"]) ** 2) * unit_scale
        elif m_lower == "rmse":
            values["RMSE"] = (
                np.sqrt(np.mean((df["true"] - df["pred"]) ** 2)) * unit_scale
            )
        elif m_lower == "r2":
            values["R^2"] = 1 - np.sum((df["true"] - df["pred"]) ** 2) / np.sum(
                (df["true"] - df["true"].mean()) ** 2
            )
        else:
            raise ValueError(f"Unsupported metric: {m}")

    text_lines: list[str] = []
    for name, val in values.items():
        if unit and name == "MSE":
            unit_str = rf"\,\mathrm{{{unit}}}^2"
        elif unit and name != "R^2":
            unit_str = rf"\,\mathrm{{{unit}}}"
        else:
            unit_str = ""

        if name == "R^2":
            latex_name = r"R^2"
        else:
            latex_name = rf"\mathrm{{{name}}}"

        if name == "R^2":
            text_lines.append(rf"${latex_name}: {val:.3f}{unit_str}$")
        else:
            text_lines.append(rf"${latex_name}: {val:.{metrics_precision}}{unit_str}$")
    text = "\n".join(text_lines)

    return text


def plot_scatter(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    metrics: list[str] = ["mae", "r2"],
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
    subfigure_label: str | None = None,
    out_png: str | None = None,
) -> None:
    """
    Create a hexbin plot and save it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the true and pred values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ax (plt.Axes | None): Axes object to plot the hexbin on.
        metrics (list[str]): A list of strings to be displayed in the plot.
        metrics_precision (str): Format string for the metrics.
        unit (str | None): Unit of the property.
        unit_scale (float): Scale factor for the unit.
        subfigure_label (str | None): Label for the subfigure.
        out_png (str | None): Path of the PNG file in which to save the hexbin plot.

    """

    with plt.rc_context(PLOT_RC_PARAMS):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        else:
            ax.get_figure()

        ax.scatter(
            df["true"],
            df["pred"],
            s=100,  # marker size
            edgecolors="white",
            rasterized=True,  # smaller vector PDFs
            color="#385723",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Keep axes square
        ax.set_box_aspect(1)

        # Get the current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])

        # Plot y = x reference line (grey dashed)
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="grey",
            linewidth=2,
        )

        # Restore the original limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Compute requested metrics
        text = metrics_text(df, metrics, metrics_precision, unit, unit_scale)

        if subfigure_label is not None:
            text = f"{subfigure_label}\n{text}"

        ax.text(
            0.025,
            0.975,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        if out_png is not None:
            plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")

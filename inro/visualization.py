"""Plotting helpers for experiment results."""

import os
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["LinePlot"]


class LinePlot:
    """Seaborn line plot of tabular results, such as rate-distortion curves.

    Args:
        data: Results in any form :class:`pandas.DataFrame` accepts: a DataFrame, a
            mapping from column names to sequences, or a sequence of record
            dictionaries.
        columns: Columns passed to :class:`pandas.DataFrame`; ``None`` keeps all
            columns.

    Attributes:
        data: The results as a :class:`pandas.DataFrame`.
        fig: Figure created by the last :meth:`plot` call, or ``None``.
        ax: Axes created by the last :meth:`plot` call, or ``None``.

    Examples:
        >>> records = [
        ...     {"bpp": 0.1, "psnr": 30.0, "method": "FUTON"},
        ...     {"bpp": 0.2, "psnr": 33.0, "method": "FUTON"},
        ... ]
        >>> plot = LinePlot(records)
        >>> fig, ax = plot.plot(x="bpp", y="psnr", hue="method")
        >>> plot.save("figures", "rate_distortion")  # doctest: +SKIP
    """

    def __init__(
        self,
        data: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[dict[str, Any]],
        columns: Sequence[str] | None = None,
    ) -> None:
        self.data = pd.DataFrame(data, columns=columns)
        self.fig: Figure | None = None
        self.ax: Axes | None = None

    def plot(
        self,
        x: str,
        y: str,
        hue: str,
        xlim: tuple[float | None, float | None] = (None, None),
        ylim: tuple[float | None, float | None] = (None, None),
        legend_labels: Sequence[str] | None = None,
        legend_loc: str = "lower right",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Draw lines grouped by ``hue`` and store the resulting ``fig`` and ``ax``.

        Args:
            x: Column of the x-axis.
            y: Column of the y-axis.
            hue: Column defining both line color and style.
            xlim: Left and right limits; ``None`` keeps a limit automatic.
            ylim: Bottom and top limits; ``None`` keeps a limit automatic.
            legend_labels: Existing labels to keep in the legend, in order, matching
                exactly; ``None`` uses the ``hue`` values in order of appearance, which
                must be strings.
            legend_loc: Legend location accepted by ``matplotlib.axes.Axes.legend``.
            **kwargs: Passed to ``seaborn.lineplot``, such as aggregation options.

        Returns:
            The new figure and axes.
        """
        if legend_labels is None:
            legend_labels = self.data[hue].unique()

        fig, ax = plt.subplots()
        sns.lineplot(
            ax=ax,
            data=self.data,
            x=x,
            y=y,
            hue=hue,
            style=hue,
            legend="brief",
            **kwargs,
        )
        ax.grid()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        handles, labels = ax.get_legend_handles_labels()
        selected_handles = [handles[labels.index(label)] for label in legend_labels]
        ax.legend(selected_handles, list(legend_labels), loc=legend_loc, title="")

        self.fig, self.ax = fig, ax
        return fig, ax

    def save(
        self,
        save_dir: str | os.PathLike[str] = ".",
        file_name: str = "",
        format: str = "pdf",
    ) -> None:
        """Save the last plot to ``save_dir/{file_name}.{format}`` with tight bounds.

        Call :meth:`plot` first.

        Args:
            save_dir: Output directory, created if needed.
            file_name: File name without extension; must not be empty, for matplotlib to
                recognize the extension.
            format: Extension of the file, which sets its format.
        """
        os.makedirs(save_dir, exist_ok=True)

        self.fig.savefig(
            os.path.join(save_dir, f"{file_name}.{format}"),
            bbox_inches="tight",
            pad_inches=0,
        )

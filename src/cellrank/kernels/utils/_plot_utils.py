"""Private plotting utilities formerly imported from scVelo."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

__all__: list[str] = []


def _quiver_autoscale(X_emb: np.ndarray, V_emb: np.ndarray) -> float:
    """Compute auto-scale factor for quiver arrow plots.

    Creates a temporary quiver plot to determine the appropriate scale
    factor, then removes it.

    Parameters
    ----------
    X_emb
        Embedding coordinates, shape ``(n_cells, 2)``.
    V_emb
        Embedding velocity vectors, shape ``(n_cells, 2)``.

    Returns
    -------
    The quiver scale normalized by the embedding scale factor.
    """
    scale_factor = np.abs(X_emb).max()
    fig, ax = plt.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def _default_size(adata: AnnData) -> float:
    """Compute default scatter point size based on the number of cells.

    Parameters
    ----------
    adata
        Annotated data matrix.

    Returns
    -------
    Point size scaled inversely with cell count.
    """
    return 1.2e5 / adata.n_obs


def _plot_outline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    outline_color: tuple[str, str] = ("black", "white"),
    kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
    **scatter_kwargs: Any,
) -> None:
    """Draw outlined scatter points using a double-scatter technique.

    Draws three layers: a large background scatter (``outline_color[0]``),
    a medium gap scatter (``outline_color[1]``), and optionally the caller
    adds the actual data scatter on top.

    Parameters
    ----------
    x
        X coordinates.
    y
        Y coordinates.
    outline_color
        Tuple of ``(background_color, gap_color)``.
    kwargs
        Base keyword arguments for all three scatter calls (e.g. ``s``, ``alpha``).
    ax
        Matplotlib axes to draw on. If :obj:`None`, uses current axes.
    **scatter_kwargs
        Additional keyword arguments forwarded to all scatter calls (e.g. ``zorder``).
    """
    if ax is None:
        ax = plt.gca()
    if kwargs is None:
        kwargs = {}

    bg_color, gap_color = outline_color

    # background layer (larger points)
    bg_kwargs = {**kwargs, **scatter_kwargs, "color": bg_color}
    bg_kwargs["s"] = bg_kwargs.get("s", 20) * 1.4
    ax.scatter(x, y, **bg_kwargs)

    # gap layer (medium points)
    gap_kwargs = {**kwargs, **scatter_kwargs, "color": gap_color}
    gap_kwargs["s"] = gap_kwargs.get("s", 20) * 1.1
    ax.scatter(x, y, **gap_kwargs)

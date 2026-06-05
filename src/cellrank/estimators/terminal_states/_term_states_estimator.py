import abc
import logging
import types
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from matplotlib.colors import to_hex
from pandas.api.types import infer_dtype

from cellrank._utils._colors import (
    _convert_to_hex_colors,
    _create_categorical_colors,
    _map_names_and_colors,
)
from cellrank._utils._docs import d, inject_docs
from cellrank._utils._key import Key
from cellrank._utils._lineage import Lineage
from cellrank._utils._utils import (
    RandomKeys,
    _check_proportions,
    _convert_to_categorical_series,
    _map_names_and_colors_from_proportions,
    _merge_categorical_series,
    _obsm_proportion_weights,
    _obsm_proportions,
    _resolve_composition_key,
    _unique_order_preserving,
    save_fig,
)
from cellrank.estimators._base_estimator import BaseEstimator
from cellrank.estimators.mixins._utils import (
    PlotMode,
    SafeGetter,
    StatesHolder,
    log_writer,
    shadow,
)
from cellrank.kernels._base_kernel import KernelExpression
from cellrank.pl._utils import _plot_color_gradients, _plot_time_scatter

logger = logging.getLogger(__name__)
__all__ = ["TermStatesEstimator"]


@d.dedent
class TermStatesEstimator(BaseEstimator, abc.ABC):
    """Base class for all estimators predicting the initial and terminal states.

    Parameters
    ----------
    %(base_estimator.parameters)s
    """

    def __init__(
        self,
        object: AnnData | np.ndarray | sp.spmatrix | KernelExpression,
        **kwargs: Any,
    ):
        super().__init__(object=object, **kwargs)
        self._init_states = StatesHolder()
        self._term_states = StatesHolder()

    @property
    @d.get_summary(base="tse_term_states")
    def terminal_states(self) -> pd.Series | None:
        """Categorical annotation of terminal states.

        By default, all transient cells will be labeled as `NaN`.
        """
        return self._term_states.assignment

    @property
    @d.get_summary(base="tse_term_states_probs")
    def terminal_states_probabilities(self) -> pd.Series | None:
        """Probability to be a terminal state."""
        return self._term_states.probs

    @property
    @d.get_summary(base="tse_init_states")
    def initial_states(self) -> pd.Series | None:
        """Categorical annotation of initial states.

        By default, all transient cells will be labeled as `NaN`.
        """
        return self._init_states.assignment

    @property
    @d.get_summary(base="tse_init_states_probs")
    def initial_states_probabilities(self) -> pd.Series | None:
        """Probability to be an initial state."""
        return self._init_states.probs

    @d.dedent
    def set_terminal_states(
        self,
        states: pd.Series | dict[str, Sequence[Any]],
        cluster_key: str | Mapping[str, str] | None = None,
        weight_key: str | None = None,
        allow_overlap: bool = False,
        **kwargs: Any,
    ) -> "TermStatesEstimator":
        """Set the :attr:`terminal_states`.

        Parameters
        ----------
        states
            States to select. Valid options are:

            - categorical :class:`~pandas.Series` where each category corresponds to an individual state.
              `NaN` entries denote cells that do not belong to any state, i.e., transient cells.
            - :class:`dict` where keys are states and values are lists of cell barcodes corresponding to
              annotations in :attr:`~anndata.AnnData.obs_names`.
              If only 1 key is provided, values should correspond to clusters if a categorical
              :class:`~pandas.Series` can be found in :attr:`~anndata.AnnData.obs`.
        cluster_key
            Reference annotations to associate names and colors with :attr:`terminal_states`. Either a
            categorical :attr:`~anndata.AnnData.obs` column (a :class:`str` or ``{"obs": <column>}``) or
            ``{"obsm": <key>}`` pointing to a proportion :class:`~pandas.DataFrame`; see
            :meth:`~cellrank.estimators.GPCCA.compute_macrostates`.
        weight_key
            Per-observation weights, only used when ``cluster_key`` points to
            :attr:`~anndata.AnnData.obsm`; see :meth:`~cellrank.estimators.GPCCA.compute_macrostates`.
        %(allow_overlap)s
        kwargs
            Additional keyword arguments.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`terminal_states` - %(tse_term_states.summary)s
        - :attr:`terminal_states_probabilities` - %(tse_term_states_probs.summary)s
        """
        states, colors = self._set_categorical_labels(
            categories=states,
            cluster_key=cluster_key,
            weight_key=weight_key,
            existing=None,
        )
        self._write_states(
            "terminal",
            states=states,
            colors=colors,
            allow_overlap=allow_overlap,
            **kwargs,
        )
        return self

    @d.dedent
    def set_initial_states(
        self,
        states: pd.Series | dict[str, Sequence[Any]],
        cluster_key: str | Mapping[str, str] | None = None,
        weight_key: str | None = None,
        allow_overlap: bool = False,
        **kwargs: Any,
    ) -> "TermStatesEstimator":
        """Set the :attr:`initial_states`.

        Parameters
        ----------
        states
            Which states to select. Valid options are:

            - categorical :class:`~pandas.Series` where each category corresponds to an individual state.
              `NaN` entries denote cells that do not belong to any state, i.e., transient cells.
            - :class:`dict` where keys are states and values are lists of cell barcodes corresponding to
              annotations in :attr:`~anndata.AnnData.obs_names`.
              If only 1 key is provided, values should correspond to clusters if a categorical
              :class:`~pandas.Series` can be found in :attr:`~anndata.AnnData.obs`.
        cluster_key
            Reference annotations to associate names and colors with :attr:`initial_states`. Either a
            categorical :attr:`~anndata.AnnData.obs` column (a :class:`str` or ``{"obs": <column>}``) or
            ``{"obsm": <key>}`` pointing to a proportion :class:`~pandas.DataFrame`; see
            :meth:`~cellrank.estimators.GPCCA.compute_macrostates`.
        weight_key
            Per-observation weights, only used when ``cluster_key`` points to
            :attr:`~anndata.AnnData.obsm`; see :meth:`~cellrank.estimators.GPCCA.compute_macrostates`.
        %(allow_overlap)s
        kwargs
            Additional keyword arguments.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`initial_states` - %(tse_init_states.summary)s
        - :attr:`initial_states_probabilities` - %(tse_init_states_probs.summary)s
        """
        states, colors = self._set_categorical_labels(
            categories=states,
            cluster_key=cluster_key,
            weight_key=weight_key,
            existing=None,
        )
        self._write_states(
            "initial",
            states=states,
            colors=colors,
            allow_overlap=allow_overlap,
            **kwargs,
        )
        return self

    @d.get_sections(base="tse_rename_term_states", sections=["Parameters", "Returns"])
    @d.get_full_description(base="tse_rename_term_states")
    @d.dedent
    def rename_terminal_states(self, old_new: dict[str, str]) -> "TermStatesEstimator":
        """Rename the :attr:`terminal_states`.

        Parameters
        ----------
        old_new
            Dictionary that maps old names to unique new names.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`terminal_states` - %(tse_term_states.summary)s
        """
        states = self.terminal_states
        if states is None:
            raise RuntimeError(
                "Compute terminal states first as `.predict_terminal_states()` or "
                "set them manually as `.set_terminal_states()`."
            )

        # fmt: off
        if not isinstance(old_new, dict):
            raise TypeError(f"Expected new names to be a `dict`, found `{type(old_new)}`.")
        if not len(old_new):
            return self

        old_names = states.cat.categories
        mask = np.isin(list(old_new.keys()), old_names)
        if not np.all(mask):
            invalid = sorted(np.array(list(old_new.keys()))[~mask])
            raise ValueError(f"Invalid terminal states names: `{invalid}`. Valid names are: `{sorted(old_names)}`.")

        names_after_renaming = [old_new.get(n, n) for n in old_names]
        if len(set(names_after_renaming)) != len(old_names):
            raise ValueError(f"After renaming, terminal states will no longer unique: `{names_after_renaming}`.")
        # fmt: on

        assignment = states.cat.rename_categories(old_new)
        memberships = self._term_states.memberships  # save before overwriting
        self._write_states(
            "terminal",
            states=assignment,
            colors=self._term_states.colors,
            probs=self.terminal_states_probabilities,
            log=False,
        )
        if memberships is not None:
            memberships.names = [old_new.get(n, n) for n in memberships.names]

        self._term_states = self._term_states.set(assignment=assignment, memberships=memberships)
        return self

    @d.dedent
    def rename_initial_states(self, old_new: dict[str, str]) -> "TermStatesEstimator":
        """Rename the :attr:`initial_states`.

        Parameters
        ----------
        old_new
            Dictionary that maps old names to unique new names.

        Returns
        -------
        Returns self and updates the following fields:

        - :attr:`initial_states` - %(tse_init_states.summary)s
        """
        states = self.initial_states
        if states is None:
            raise RuntimeError(
                "Compute initial states first as `.predict_initial_states()` or "
                "set them manually as `.set_initial_states()`."
            )

        # fmt: off
        if not isinstance(old_new, dict):
            raise TypeError(f"Expected new names to be a `dict`, found `{type(old_new)}`.")
        if not len(old_new):
            return self

        old_names = states.cat.categories
        mask = np.isin(list(old_new.keys()), old_names)
        if not np.all(mask):
            invalid = sorted(np.array(list(old_new.keys()))[~mask])
            raise ValueError(f"Invalid terminal states names: `{invalid}`. Valid names are: `{sorted(old_names)}`.")

        names_after_renaming = [old_new.get(n, n) for n in old_names]
        if len(set(names_after_renaming)) != len(old_names):
            raise ValueError(f"After renaming, terminal states will no longer unique: `{names_after_renaming}`.")
        # fmt: on

        assignment = states.cat.rename_categories(old_new)
        memberships = self._init_states.memberships  # save overwriting
        self._write_states(
            "initial",
            states=assignment,
            colors=self._init_states.colors,
            probs=self.initial_states_probabilities,
            log=False,
        )
        if memberships is not None:
            memberships.names = [old_new.get(n, n) for n in memberships.names]

        self._init_states = self._init_states.set(assignment=assignment, memberships=memberships)
        return self

    @d.dedent
    @inject_docs(m=PlotMode)
    def plot_macrostates(
        self,
        which: Literal["all", "initial", "terminal", "initial_and_terminal"],
        states: str | Sequence[str] | None = None,
        color: str | None = None,
        discrete: bool = True,
        mode: Literal["embedding", "time"] = PlotMode.EMBEDDING,
        time_key: str = "latent_time",
        basis: str = "umap",
        same_plot: bool = True,
        title: str | Sequence[str] | None = None,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> None:
        """Plot macrostates on an embedding or along pseudotime.

        Parameters
        ----------
        which
            Which macrostates to plot. Valid options are:

            - ``'all'`` - plot all macrostates.
            - ``'initial'`` - plot macrostates marked as :attr:`initial_states`.
            - ``'terminal'`` - plot macrostates marked as :attr:`terminal_states`.
            - ``'initial_and_terminal'`` - plot both :attr:`initial_states` and :attr:`terminal_states` in one
              plot. States are renamed to ``'initial: <name>'`` and ``'terminal: <name>'`` to tell them apart.
        states
            Subset of the macrostates to show. If :obj:`None`, plot all macrostates.
        color
            Key in :attr:`~anndata.AnnData.obs` or :attr:`~anndata.AnnData.var` used to color the observations.
        discrete
            Whether to plot the data as continuous or discrete observations.
            If the data cannot be plotted as continuous observations, it will be plotted as discrete.
        mode
            Whether to plot the probabilities in an embedding or along the pseudotime.
        time_key
            Key in :attr:`~anndata.AnnData.obs` where pseudotime is stored. Only used when ``mode = {m.TIME!r}``.
        basis
            Key in :attr:`~anndata.AnnData.obsm` for the embedding to use, e.g. ``'umap'`` or ``'tsne'``.
        title
            Title of the plot.
        same_plot
            Whether to plot the data on the same plot or not. Only use when ``mode = {m.EMBEDDING!r}``.
            If `True` and ``discrete = False``, ``color`` is ignored.
        cmap
            Colormap for continuous annotations.
        kwargs
            Keyword arguments for :func:`~scanpy.pl.embedding`.

        Returns
        -------
        %(just_plots)s
        """
        if which == "all":
            obj: StatesHolder | None = getattr(self, "_macrostates", None)
            if obj is None:
                raise RuntimeError(f"`{type(self).__name__}` cannot plot macrostates.")
        elif which == "initial":
            obj = self._init_states
        elif which == "terminal":
            obj = self._term_states
        elif which == "initial_and_terminal":
            obj = self._combine_initial_terminal(discrete=discrete)
        else:
            raise ValueError(
                f"Unable to plot `{which!r}` states. Valid options are: "
                f"`{['all', 'initial', 'terminal', 'initial_and_terminal']}`."
            )

        name = "initial and terminal states" if which == "initial_and_terminal" else f"{which} states"
        if obj.assignment is None and obj.memberships is None:
            raise RuntimeError(f"Compute {name} first.")

        if not discrete and obj.memberships is None:
            logger.warning("Unable to plot %s in continuous mode, using discrete", name)
            discrete = True

        data = obj.assignment if discrete else obj.memberships
        colors = obj.colors

        if discrete:
            return self._plot_discrete(
                _data=data,
                _colors=colors,
                _title=name,
                states=states,
                color=color,
                basis=basis,
                same_plot=same_plot,
                title=title,
                cmap=cmap,
                **kwargs,
            )
        return self._plot_continuous(
            _data=data,
            _colors=colors,
            _title=name,
            states=states,
            color=color,
            mode=mode,
            time_key=time_key,
            basis=basis,
            same_plot=same_plot,
            title=title,
            cmap=cmap,
            **kwargs,
        )

    def _combine_initial_terminal(self, discrete: bool) -> StatesHolder:
        """Merge :attr:`initial_states` and :attr:`terminal_states` into a single :class:`StatesHolder`.

        States are renamed to ``'initial: <name>'`` / ``'terminal: <name>'`` so they can be told apart in the
        legend. In the rare case where a cell is assigned to both an initial and a terminal state
        (only possible with ``allow_overlap=True``), the initial-state assignment takes precedence in discrete mode.
        """
        init, term = self._init_states, self._term_states
        for kind, holder in [("initial", init), ("terminal", term)]:
            if holder.assignment is None and holder.memberships is None:
                raise RuntimeError(f"Compute {kind} states first.")

        def _rename(prefix: str, names: Sequence[str]) -> list[str]:
            return [f"{prefix}: {name}" for name in names]

        assignment = colors = memberships = None

        # discrete representation
        if init.assignment is not None and term.assignment is not None:
            init_a, term_a = init.assignment, term.assignment
            init_cats = _rename("initial", init_a.cat.categories)
            term_cats = _rename("terminal", term_a.cat.categories)
            init_r = init_a.cat.rename_categories(dict(zip(init_a.cat.categories, init_cats)))
            term_r = term_a.cat.rename_categories(dict(zip(term_a.cat.categories, term_cats)))
            combined = init_r.astype(object).combine_first(term_r.astype(object))
            assignment = pd.Series(
                pd.Categorical(combined, categories=init_cats + term_cats),
                index=init_a.index,
            )
            if init.colors is not None and term.colors is not None:
                colors = np.concatenate([init.colors, term.colors])

        # continuous representation
        if init.memberships is not None and term.memberships is not None:
            im, tm = init.memberships, term.memberships
            memberships = Lineage(
                np.concatenate([im.X, tm.X], axis=1),
                names=_rename("initial", im.names) + _rename("terminal", tm.names),
                colors=list(im.colors) + list(tm.colors),
            )

        if discrete and assignment is None:
            raise RuntimeError("Unable to plot initial and terminal states in discrete mode, compute them first.")

        return StatesHolder(assignment=assignment, colors=colors, memberships=memberships)

    def _plot_discrete(
        self,
        _data: pd.Series,
        _colors: np.ndarray | None = None,
        _title: str | None = None,
        states: str | Sequence[str] | None = None,
        color: str | None = None,
        basis: str = "umap",
        title: str | Sequence[str] | None = None,
        same_plot: bool = True,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> None:
        if not isinstance(_data, pd.Series):
            raise TypeError(f"Expected `data` to be of type `pandas.Series`, found `{type(_data)}`.")
        if not isinstance(_data.dtype, pd.CategoricalDtype):
            raise TypeError(f"Expected `data` to be `categorical`, found `{infer_dtype(_data)}`.")

        names = list(_data.cat.categories)
        if _colors is None:
            _colors = _create_categorical_colors(len(names))
        if len(_colors) != len(names):
            raise ValueError(f"Expected `colors` to be of length `{len(names)}`, found `{len(_colors)}`.")
        color_mapper = dict(zip(names, _colors))

        states = _unique_order_preserving(states or names)
        if not len(states):
            raise ValueError("No states have been selected.")

        for name in states:
            if name not in names:
                raise ValueError(f"Invalid name `{name!r}`. Valid options are: `{sorted(names)}`.")
        _data = _data.cat.set_categories(states)

        color = [] if color is None else (color,) if isinstance(color, str) else color
        color = _unique_order_preserving(color)

        same_plot = same_plot or len(names) == 1
        kwargs.setdefault("legend_loc", "on data")
        # "right" means "right margin" in scanpy
        if kwargs.get("legend_loc") == "right":
            kwargs["legend_loc"] = "right margin"
        kwargs.pop("color_map", None)
        # `sc.pl.embedding` ignores `dpi`; forward it to `save_fig`, which honors it at save time.
        dpi = kwargs.pop("dpi", None)
        save = kwargs.pop("save", None)
        show = kwargs.pop("show", None)
        kwargs["cmap"] = cmap
        basis = kwargs.pop("basis", basis)
        size = kwargs.get("size", 120_000 / self.adata.n_obs)

        # fmt: off
        with RandomKeys(self.adata, n=1 if same_plot else len(states), where="obs") as keys:
            if same_plot:
                self.adata.obs[keys[0]] = _data
                self.adata.uns[f"{keys[0]}_colors"] = [color_mapper[name] for name in states]
                title = _title if title is None else title
            else:
                for key, cat in zip(keys, states):
                    self.adata.obs[key] = _data.cat.set_categories([cat])
                    self.adata.uns[f"{key}_colors"] = [color_mapper[cat]]
                title = [f"{_title} {name}" for name in states] if title is None else title

            if isinstance(title, str):
                title = [title]

            kwargs.setdefault("na_color", "#dedede")
            kwargs.setdefault("na_in_legend", False)
            axes = sc.pl.embedding(
                self.adata,
                basis=basis,
                color=color + keys,
                title=color + title,
                show=False,
                return_fig=False,
                **kwargs,
            )

            # Overlay state cells with outlines so they appear on top of NaN cells
            if same_plot:
                axes_list = [axes] if not isinstance(axes, list | np.ndarray) else list(np.ravel(axes))
                mask = _data.notna()
                if mask.any():
                    adata_sub = self.adata[mask].copy()
                    for ax, key in zip(axes_list[len(color):], keys):
                        ax_title = ax.get_title()
                        sc.pl.embedding(
                            adata_sub,
                            basis=basis,
                            color=key,
                            add_outline=True,
                            show=False,
                            return_fig=False,
                            ax=ax,
                            legend_loc="none",
                            size=size,
                        )
                        ax.set_title(ax_title)

        if save is not None:
            save_fig(plt.gcf(), save, dpi=dpi)
        if show is True or (show is None and save is None):
            plt.show()
        # fmt: on

    def _plot_continuous(
        self,
        _data: Lineage,
        _colors: np.ndarray | None = None,
        _title: str | None = None,
        states: str | Sequence[str] | None = None,
        color: str | None = None,
        mode: Literal["embedding", "time"] = PlotMode.EMBEDDING,
        time_key: str = "latent_time",
        basis: str = "umap",
        title: str | Sequence[str] | None = None,
        same_plot: bool = True,
        cmap: str = "viridis",
        **kwargs: Any,
    ) -> None:
        mode = PlotMode(mode)
        if not isinstance(_data, Lineage):
            raise TypeError(f"Expected data to be of type `Lineage`, found `{type(_data)}`.")

        if states is None:
            states = _data.names
        if not len(states):
            raise ValueError("No lineages have been selected.")
        _data = _data[states].copy()

        if mode == "time" and same_plot:
            logger.warning("Invalid combination `mode='time'` and `same_plot=True`. Using `same_plot=False`")
            same_plot = False

        _data_X = _data.X  # list(_data.T) behaves differently than a numpy.array
        if _data_X.shape[1] == 1:
            same_plot = False
            if np.allclose(_data_X, 1.0):
                # matplotlib shows even tiny perturbations in the colormap
                _data_X = np.ones_like(_data_X)

        for col in _data_X.T:
            mask = ~np.isclose(col, 1.0)
            # change the maximum value - the 1 is artificial and obscures the color scaling
            if np.any(mask):
                col[~mask] = np.nanmax(col[mask])

        # fmt: off
        color = [] if color is None else (color,) if isinstance(color, str) else color
        color = _unique_order_preserving(color)
        basis = kwargs.pop("basis", basis)
        kwargs.pop("color_map", None)
        save = kwargs.pop("save", None)
        show = kwargs.pop("show", None)

        if mode == PlotMode.TIME:
            kwargs.setdefault("legend_loc", "best")
            if time_key is None:
                raise KeyError(
                    "The name of the column in `adata.obs` defining the pseudotime needs to be defined via the "
                    "`time_key` argument."
                )
            if title is None:
                title = [f"{_title} {state}" for state in states]
            if time_key not in self.adata.obs:
                raise KeyError(f"Unable to find pseudotime in `adata.obs[{time_key!r}]`.")
            if len(color) and len(color) not in (1, _data_X.shape[1]):
                raise ValueError(f"Expected `color` to be of length `1` or `{_data_X.shape[1]}`, "
                                 f"found `{len(color)}`.")
            _plot_time_scatter(
                self.adata, self.adata.obs[time_key].values, list(_data_X.T),
                color=color if len(color) else None,
                title=title, xlabel=time_key, ylabel="probability", cmap=cmap,
                save=save, show=show, **kwargs,
            )
        elif mode == PlotMode.EMBEDDING:
            kwargs.setdefault("legend_loc", "on data")
            # "right" means "right margin" in scanpy
            if kwargs.get("legend_loc") == "right":
                kwargs["legend_loc"] = "right margin"

            if same_plot:
                if color:
                    logger.warning("Ignoring `color` when `mode='embedding'` and `same_plot=True`")
                title = [_title] if title is None else title
                _plot_color_gradients(self.adata, _data, basis=basis, title=title,
                                      save=save, show=show, **kwargs)
            else:
                # `sc.pl.embedding` ignores `dpi`; forward it to `save_fig`, which honors it at save time.
                dpi = kwargs.pop("dpi", None)
                title = [f"{_title} {state}" for state in states] if title is None else title
                if isinstance(title, str):
                    title = [title]
                title = color + title
                # Store probability arrays as temp obs columns (scanpy requires column names)
                with RandomKeys(self.adata, n=_data_X.shape[1], where="obs") as prob_keys:
                    for key, col in zip(prob_keys, _data_X.T):
                        self.adata.obs[key] = col
                    sc.pl.embedding(
                        self.adata, basis=basis, color=color + list(prob_keys),
                        title=title, cmap=cmap, show=False, **kwargs,
                    )
                if save is not None:
                    save_fig(plt.gcf(), save, dpi=dpi)
                if show is True or (show is None and save is None):
                    plt.show()
        else:
            raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")
        # fmt: on

    def _set_categorical_labels(
        self,
        categories: pd.Series | dict[str, Any],
        cluster_key: str | Mapping[str, str] | None = None,
        weight_key: str | None = None,
        existing: pd.Series | None = None,
    ) -> tuple[pd.Series, np.ndarray]:
        # fmt: off
        if isinstance(categories, dict):
            key = next(iter(categories.keys()))
            data = self.adata.obs.get(key, None)
            is_categorical = data is not None and isinstance(data.dtype, pd.CategoricalDtype)
            if len(categories) == 1 and is_categorical:
                vals = categories[key]
                if isinstance(vals, str) or not isinstance(vals, Sequence):
                    vals = (categories[key],)

                clusters = self.adata.obs[key]
                clusters = clusters.cat.rename_categories({c: str(c) for c in clusters.cat.categories})
                vals = tuple(str(v) for v in vals)
                categories = {cat: self.adata[clusters == cat].obs_names for cat in vals}

            categories = _convert_to_categorical_series(categories, list(self.adata.obs_names))
        if not isinstance(categories.dtype, pd.CategoricalDtype):
            raise TypeError(f"Expected object to be `categorical`, found `{infer_dtype(categories)}`.")

        if existing is not None:
            categories = _merge_categorical_series(old=existing, new=categories)

        if cluster_key is not None:
            source, name = _resolve_composition_key(cluster_key)
            if weight_key is not None and source != "obsm":
                raise ValueError("`weight_key` is only supported when `cluster_key` points to `adata.obsm`.")
            if source == "obs":
                names, colors = self._names_and_colors_from_obs(name, categories)
            else:
                names, colors = self._names_and_colors_from_obsm(name, weight_key, categories)
            cats = categories.cat.categories
            categories = categories.cat.rename_categories(dict(zip(cats, names)))
        else:
            colors = _create_categorical_colors(len(categories.cat.categories))

        return categories, colors
        # fmt: on

    def _names_and_colors_from_obs(self, key: str, series_query: pd.Series) -> tuple[pd.Series, list[str]]:
        """Name and color the query categories by overlap with a categorical :attr:`~anndata.AnnData.obs` column."""
        if key not in self.adata.obs:
            raise KeyError(f"Unable to find clusters in `adata.obs[{key!r}]`.")
        series_reference = self.adata.obs[key]
        series_reference = series_reference.cat.rename_categories({c: str(c) for c in series_reference.cat.categories})

        # load the reference colors if they exist
        if Key.uns.colors(key) in self.adata.uns:
            colors_reference = _convert_to_hex_colors(self.adata.uns[Key.uns.colors(key)])
        else:
            colors_reference = _create_categorical_colors(len(series_reference.cat.categories))

        return _map_names_and_colors(
            series_reference=series_reference,
            series_query=series_query,
            colors_reference=colors_reference,
        )

    def _names_and_colors_from_obsm(
        self, key: str, weight_key: str | None, series_query: pd.Series
    ) -> tuple[pd.Series, list[str]]:
        """Name and color the query categories by the dominant covariate of an :attr:`~anndata.AnnData.obsm` frame.

        Each query category (e.g. macrostate) is named after the category with the largest (weighted) summed
        proportion among its assigned observations -- the soft-assignment analog of :meth:`_names_and_colors_from_obs`.
        """
        proportions = _obsm_proportions(self.adata, key)
        assigned = series_query[~series_query.isnull()]
        _check_proportions(proportions.loc[assigned.index], key)
        weights = _obsm_proportion_weights(self.adata, weight_key, series_query.index)

        # load the reference colors if they exist, padding if there are fewer colors than categories
        n_categories = len(proportions.columns)
        if Key.uns.colors(key) in self.adata.uns:
            colors_reference = _convert_to_hex_colors(self.adata.uns[Key.uns.colors(key)])
        else:
            colors_reference = _create_categorical_colors(n_categories)
        if len(colors_reference) < n_categories:
            colors_reference = _create_categorical_colors(n_categories)

        return _map_names_and_colors_from_proportions(
            proportions=proportions,
            series_query=series_query,
            weights=weights,
            colors_reference=colors_reference,
        )

    @log_writer
    @shadow
    def _write_states(
        self,
        which: Literal["initial", "terminal"],
        states: pd.Series | None,
        colors: np.ndarray | None,
        probs: pd.Series | None = None,
        params: dict[str, Any] = types.MappingProxyType({}),
        allow_overlap: bool = False,
    ) -> str:
        # fmt: off
        backward = which == "initial"
        if not allow_overlap:
            fwd_states, bwd_states = (self.terminal_states, states) if backward else (states, self.initial_states)
            if fwd_states is not None and bwd_states is not None:
                overlap = np.sum(~pd.isnull(fwd_states) & ~pd.isnull(bwd_states))
                if overlap:
                    raise ValueError(
                        f"Found `{overlap}` overlapping cells between initial and terminal states. "
                        f"If this is intended, please use `allow_overlap=True`."
                    )

        key = Key.obs.term_states(self.backward, bwd=backward)
        self._set(obj=self.adata.obs, key=key, value=states)
        self._set(obj=self.adata.obs, key=Key.obs.probs(key), value=probs)
        self._set(obj=self.adata.uns, key=Key.uns.colors(key), value=colors)
        if backward:
            self._init_states = self._init_states.set(assignment=states, probs=probs, colors=colors)
        else:
            self._term_states = self._term_states.set(assignment=states, probs=probs, colors=colors)
        self.params[key] = dict(params)
        # fmt: on

        return (
            f"Adding `adata.obs[{key!r}]`\n"
            f"       `adata.obs[{Key.obs.probs(key)!r}]`\n"
            f"       `.{which}_states`\n"
            f"       `.{which}_states_probabilities`\n"
            "    Finish"
        )

    def _read_from_adata(self, adata: AnnData, **kwargs: Any) -> bool:
        ok = super()._read_from_adata(adata, **kwargs)
        if not ok:
            return False

        # fmt: off
        for backward in [True, False]:
            key = Key.obs.term_states(self.backward, bwd=backward)
            with SafeGetter(self, allowed=KeyError) as sg:
                assignment = self._get(obj=adata.obs, key=key, shadow_attr="obs", dtype=pd.Series)
                probs = self._get(obj=adata.obs, key=Key.obs.probs(key), shadow_attr="obs", dtype=pd.Series)
                colors = self._get(obj=adata.uns, key=Key.uns.colors(key), shadow_attr="uns",
                                   dtype=(list, tuple, np.ndarray))
                colors = np.asarray([to_hex(c) for c in colors])
                if backward:
                    self._init_states = StatesHolder(assignment=assignment, probs=probs, colors=colors)
                else:
                    self._term_states = StatesHolder(assignment=assignment, probs=probs, colors=colors)
                self.params[key] = self._read_params(key)
        # fmt: on

        # status is based on `backward=False` by design
        return sg.ok

    def _format_params(self) -> str:
        fmt = super()._format_params()
        # fmt: off
        init_states = None if self.initial_states is None else sorted(self.initial_states.cat.categories)
        term_states = None if self.terminal_states is None else sorted(self.terminal_states.cat.categories)
        return fmt + f", initial_states={init_states}, terminal_states={term_states}"
        # fmt: on

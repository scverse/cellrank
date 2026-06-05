import abc
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData

from cellrank._utils._utils import _densify_squeeze

__all__ = ["Signal", "Gene", "Obs", "Obsm"]

#: Sentinel marking a deprecated argument as "not passed" (distinct from a meaningful ``None``).
_UNSET = object()


class Signal(abc.ABC):
    """Continuous, observation-aligned signal to fit along a trajectory.

    A signal identifies a single 1D quantity defined for every cell -- gene expression
    (:class:`Gene`), a covariate in :attr:`~anndata.AnnData.obs` (:class:`Obs`), or a column of an
    :attr:`~anndata.AnnData.obsm` array (:class:`Obsm`). It is the unit consumed by
    :meth:`~cellrank.models.BaseModel.prepare` and the trajectory plotting functions, replacing the
    older ``gene`` + ``data_key`` + ``use_raw`` triple.
    """

    #: Default y-axis label used when the signal name is shown as a title instead.
    ylabel: str = "value"
    #: Layer the values are fetched from, if any. Only meaningful for :class:`Gene`.
    layer: str | None = None
    #: Whether the values are fetched from :attr:`~anndata.AnnData.raw`. Only meaningful for :class:`Gene`.
    use_raw: bool = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name, used for titles, labels and result keys."""

    @abc.abstractmethod
    def validate(self, adata: AnnData) -> None:
        """Raise a descriptive error if this signal cannot be fetched from ``adata``."""

    @abc.abstractmethod
    def _fetch(self, adata: AnnData) -> np.ndarray:
        """Return the raw, observation-aligned values (possibly sparse or 2D with one column)."""

    def fetch(self, adata: AnnData, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
        """Return the signal as a dense 1D :class:`~numpy.ndarray` aligned to :attr:`~anndata.AnnData.obs_names`."""
        self.validate(adata)
        return _densify_squeeze(self._fetch(adata), dtype)


@dataclass(frozen=True)
class Gene(Signal):
    """Gene expression signal, fetched from :attr:`~anndata.AnnData.X`, a layer, or :attr:`~anndata.AnnData.raw`.

    Parameters
    ----------
    gene
        Gene in :attr:`~anndata.AnnData.var_names` (or :attr:`~anndata.AnnData.raw` ``.var_names`` when
        ``use_raw = True``).
    layer
        Key in :attr:`~anndata.AnnData.layers` to fetch from. If :obj:`None`, use :attr:`~anndata.AnnData.X`.
    use_raw
        Whether to fetch from :attr:`~anndata.AnnData.raw`. Cannot be combined with ``layer``.
    """

    gene: str
    layer: str | None = None
    use_raw: bool = False
    ylabel = "expression"

    @property
    def name(self) -> str:  # noqa: D102
        return self.gene

    def validate(self, adata: AnnData) -> None:  # noqa: D102
        if self.use_raw and self.layer is not None:
            raise ValueError(f"Gene `{self.gene!r}`: `use_raw=True` cannot be combined with a layer.")
        if self.use_raw:
            if adata.raw is None:
                raise AttributeError("AnnData object has no attribute `.raw`.")
            if self.gene not in adata.raw.var_names:
                raise KeyError(f"Gene `{self.gene!r}` not found in `adata.raw.var_names`.")
            return
        if self.gene not in adata.var_names:
            raise KeyError(f"Gene `{self.gene!r}` not found in `adata.var_names`.")
        if self.layer is not None and self.layer not in adata.layers:
            raise KeyError(f"Layer `{self.layer!r}` not found in `adata.layers`.")

    def _fetch(self, adata: AnnData) -> np.ndarray:  # noqa: D102
        if self.use_raw:
            raw = adata.raw
            ix = np.where(raw.var_names == self.gene)[0]
            return raw.X[:, ix]
        return adata.obs_vector(self.gene, layer=self.layer)


@dataclass(frozen=True)
class Obs(Signal):
    """A numeric per-cell covariate stored in :attr:`~anndata.AnnData.obs`, e.g. a gene module score.

    Parameters
    ----------
    key
        Column in :attr:`~anndata.AnnData.obs`.
    """

    key: str

    @property
    def name(self) -> str:  # noqa: D102
        return self.key

    def validate(self, adata: AnnData) -> None:  # noqa: D102
        if self.key not in adata.obs:
            raise KeyError(f"Key `{self.key!r}` not found in `adata.obs`.")
        if not pd.api.types.is_numeric_dtype(adata.obs[self.key]):
            raise TypeError(
                f"`adata.obs[{self.key!r}]` must be numeric to be fitted, found dtype `{adata.obs[self.key].dtype}`."
            )

    def _fetch(self, adata: AnnData) -> np.ndarray:  # noqa: D102
        return adata.obs[self.key].to_numpy()


@dataclass(frozen=True)
class Obsm(Signal):
    """A single column of an :attr:`~anndata.AnnData.obsm` array or :class:`~pandas.DataFrame`.

    Parameters
    ----------
    key
        Key in :attr:`~anndata.AnnData.obsm`.
    column
        Column to select: an integer position for an array, or a label for a :class:`~pandas.DataFrame`.
    """

    key: str
    column: int | str

    @property
    def name(self) -> str:  # noqa: D102
        return f"{self.key}[{self.column}]"

    def validate(self, adata: AnnData) -> None:  # noqa: D102
        if self.key not in adata.obsm:
            raise KeyError(f"Key `{self.key!r}` not found in `adata.obsm`.")
        arr = adata.obsm[self.key]
        if isinstance(arr, pd.DataFrame):
            if self.column not in arr.columns:
                raise KeyError(f"Column `{self.column!r}` not found in `adata.obsm[{self.key!r}]` columns.")
            return
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected `adata.obsm[{self.key!r}]` to be 2D, found `{arr.ndim}` dimension(s).")
        if not isinstance(self.column, (int, np.integer)):
            raise TypeError(
                f"`adata.obsm[{self.key!r}]` is an array; `column` must be an integer index, "
                f"found `{type(self.column).__name__}`."
            )
        if not -arr.shape[1] <= self.column < arr.shape[1]:
            raise IndexError(
                f"Column index `{self.column}` is out of bounds for `adata.obsm[{self.key!r}]` "
                f"with `{arr.shape[1]}` column(s)."
            )

    def _fetch(self, adata: AnnData) -> np.ndarray:  # noqa: D102
        arr = adata.obsm[self.key]
        if isinstance(arr, pd.DataFrame):
            return arr[self.column].to_numpy()
        return np.asarray(arr)[:, self.column]


def _coerce_signal(
    signal: "str | Signal" = _UNSET,
    *,
    gene: "str | Signal" = _UNSET,
    data_key: str | None = _UNSET,
    use_raw: bool = _UNSET,
    warn: bool = True,
    stacklevel: int = 4,
) -> Signal:
    """Resolve a :class:`Signal` from the new ``signal`` argument or the deprecated ``gene``/``data_key``/``use_raw``.

    A :class:`Signal` passed via either ``signal`` or ``gene`` is returned as-is. A bare string ``signal``
    is the supported shorthand for :class:`Gene` and does not warn. The legacy ``gene``/``data_key``/``use_raw``
    arguments still work but emit a :class:`DeprecationWarning` and are folded into the equivalent signal.
    """
    for candidate in (signal, gene):
        if isinstance(candidate, Signal):
            if data_key is not _UNSET or use_raw is not _UNSET:
                raise TypeError("Cannot combine a `Signal` with the deprecated `data_key`/`use_raw` arguments.")
            return candidate

    key = signal if signal is not _UNSET else gene
    if key is _UNSET:
        raise TypeError("Missing the required `signal` argument (a gene name or a `Gene`/`Obs`/`Obsm`).")
    if not isinstance(key, str):
        raise TypeError(f"Expected a `str` gene name or a `Signal`, found `{type(key).__name__}`.")

    if warn and (gene is not _UNSET or data_key is not _UNSET or use_raw is not _UNSET):
        warnings.warn(
            "Passing `gene`/`data_key`/`use_raw` will be deprecated in CellRank 2.4 and removed in a later release. "
            "Pass a `cellrank.models.Gene`, `Obs` or `Obsm` instead, e.g. `Gene('Gata1', layer='Ms')`, "
            "`Obs('module_score')` or `Obsm('X_pca', 0)`.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    dk = None if data_key is _UNSET else data_key
    if use_raw is not _UNSET and use_raw:
        return Gene(key, use_raw=True)
    if dk == "obs":
        return Obs(key)
    if dk in (None, "X"):
        return Gene(key)
    return Gene(key, layer=dk)

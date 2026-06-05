```{eval-rst}
.. module:: cellrank.models
```

# Models

Models fit gene expression trends in pseudotime; they assume some parametric form for the gene trend and estimate
parameters using an objective function. Note that some models require you to have [R](https://cran.r-project.org/)
and [rpy2](https://rpy2.github.io) installed.

```{eval-rst}
.. currentmodule:: cellrank
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummary/models

    models.GAM
    models.GAMR
    models.SKLearnModel
```

## Signals

Signals identify the observation-aligned quantity a model is fit on. Pass them to
{func}`cellrank.pl.gene_trends` or {meth}`cellrank.models.BaseModel.prepare` to plot gene expression,
an {attr}`~anndata.AnnData.obs` covariate (e.g. a gene module score), or a column of an
{attr}`~anndata.AnnData.obsm` array.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary/models

    models.Signal
    models.Gene
    models.Obs
    models.Obsm
```

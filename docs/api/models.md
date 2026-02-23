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

# Developer API

## Kernels

```{eval-rst}
.. autoclass:: cellrank.kernels.Kernel
    :members:
    :inherited-members:
```

```{eval-rst}
.. autoclass:: cellrank.kernels.UnidirectionalKernel
    :members:
    :inherited-members:
```

```{eval-rst}
.. autoclass:: cellrank.kernels.BidirectionalKernel
    :members:
    :inherited-members:
```

### Similarity

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.SimilarityABC
    :members:
    :special-members: __call__
    :inherited-members:
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.Cosine
    :members: __call__, hessian
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.Correlation
    :members: __call__, hessian
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.DotProduct
    :members: __call__, hessian
```
<!-- panache-ignore-format-end -->

### Threshold Scheme

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.ThresholdSchemeABC
    :members:
    :special-members: __call__
    :inherited-members:
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.HardThresholdScheme
    :members:
    :special-members: __call__
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.SoftThresholdScheme
    :members:
    :special-members: __call__
```
<!-- panache-ignore-format-end -->

<!-- panache-ignore-format-start -->
```{eval-rst}
.. autoclass:: cellrank.kernels.utils.CustomThresholdScheme
    :members:
    :special-members: __call__
```
<!-- panache-ignore-format-end -->

## Estimators

```{eval-rst}
.. autoclass:: cellrank.estimators.BaseEstimator
    :members:
```

```{eval-rst}
.. autoclass:: cellrank.estimators.TermStatesEstimator
    :members:
```

## Models

```{eval-rst}
.. autoclass:: cellrank.models.BaseModel
    :members:
```

## Lineage

```{eval-rst}
.. autoclass:: cellrank.Lineage
    :members: priming_degree, reduce, plot_pie, from_adata, X, T, view, names, colors
```

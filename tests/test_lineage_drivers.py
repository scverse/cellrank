import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellrank._utils._key import Key
from cellrank.estimators import GPCCA


def _to_dense(X) -> np.ndarray:
    """Return a fresh, owned dense ``float64`` copy (never an in-place view of ``X``)."""
    X = X.toarray() if sp.issparse(X) else X
    return np.array(X, dtype=np.float64, copy=True)


class TestLineageDrivers:
    @pytest.mark.parametrize("use_raw", [False, True])
    def test_normal_run(self, g: GPCCA, use_raw: bool):
        key = Key.varm.lineage_drivers(False)
        names = g.fate_probabilities.names
        if use_raw:
            g.adata.raw = g.adata.copy()

        g.compute_lineage_drivers(use_raw=use_raw)

        adata = g.adata.raw if use_raw else g.adata

        assert isinstance(adata.varm[key], pd.DataFrame)
        for name in names:
            assert np.all(adata.varm[key][f"{name}_corr"] >= -1.0)
            assert np.all(adata.varm[key][f"{name}_corr"] <= 1.0)

            assert np.all(adata.varm[key][f"{name}_qval"] >= 0)
            assert np.all(adata.varm[key][f"{name}_qval"] <= 1.0)

    def test_invalid_method(self, g: GPCCA):
        with pytest.raises(ValueError, match=r".*foobar.*"):
            g.compute_lineage_drivers(method="foobar")

    def test_invalid_n_perms_value(self, g: GPCCA):
        with pytest.raises(ValueError, match=r".*n_perms.*"):
            g.compute_lineage_drivers(n_perms=0, method="perm_test")

    def test_seed_reproducible(self, g: GPCCA):
        res_a = g.compute_lineage_drivers(
            n_perms=10,
            n_jobs=1,
            seed=0,
            method="perm_test",
        )
        res_b = g.compute_lineage_drivers(
            n_perms=10,
            n_jobs=1,
            seed=0,
            method="perm_test",
        )
        res_diff_seed = g.compute_lineage_drivers(
            n_perms=10,
            n_jobs=1,
            seed=1,
            method="perm_test",
        )

        assert res_a is not res_b
        np.testing.assert_array_equal(res_a.index, res_b.index)
        np.testing.assert_array_equal(res_a.columns, res_b.columns)
        np.testing.assert_allclose(res_a.values, res_b.values)

        assert not np.allclose(res_a.values, res_diff_seed.values)

    def test_seed_reproducible_parallel(self, g: GPCCA):
        res_a = g.compute_lineage_drivers(
            n_perms=10,
            n_jobs=2,
            backend="threading",
            seed=42,
            method="perm_test",
        )
        res_b = g.compute_lineage_drivers(
            n_perms=10,
            n_jobs=2,
            backend="threading",
            seed=42,
            method="perm_test",
        )

        assert res_a is not res_b
        np.testing.assert_array_equal(res_a.index, res_b.index)
        np.testing.assert_array_equal(res_a.columns, res_b.columns)
        np.testing.assert_allclose(res_a.values, res_b.values)

    def test_confidence_level(self, g: GPCCA):
        res_narrow = g.compute_lineage_drivers(confidence_level=0.95)
        res_wide = g.compute_lineage_drivers(confidence_level=0.99)

        for name in ["0", "1"]:
            assert np.all(res_narrow[f"{name}_ci_low"] >= res_wide[f"{name}_ci_low"])
            assert np.all(res_narrow[f"{name}_ci_high"] <= res_wide[f"{name}_ci_high"])

    def test_invalid_nan_policy(self, g: GPCCA):
        with pytest.raises(ValueError, match=r".*nan_policy.*"):
            g.compute_lineage_drivers(nan_policy="foobar")

    def test_nan_policy_omit_matches_propagate_without_nans(self, g: GPCCA):
        # `deep=True` isolates `adata` so mutating `.X` doesn't leak into the shared fixture
        g = g.copy(deep=True)
        g.adata.X = _to_dense(g.adata.X)

        res_propagate = g.compute_lineage_drivers(nan_policy="propagate")
        res_omit = g.compute_lineage_drivers(nan_policy="omit")

        np.testing.assert_array_equal(sorted(res_propagate.index), sorted(res_omit.index))
        np.testing.assert_array_equal(res_propagate.columns, res_omit.columns)
        # tiny FP differences in the masked matmuls can reorder near-tied genes, so align on the index
        np.testing.assert_allclose(res_propagate.values, res_omit.loc[res_propagate.index].values, atol=1e-8)

    def test_nan_policy_omit_handles_missing_values(self, g: GPCCA):
        g = g.copy(deep=True)
        names = g.fate_probabilities.names
        X = _to_dense(g.adata.X)
        rng = np.random.default_rng(0)
        # introduce missing values into the expression matrix
        X[rng.random(X.shape) < 0.1] = np.nan
        g.adata.X = X

        # `propagate` lets the missing values contaminate every gene -> all correlations are NaN
        res_propagate = g.compute_lineage_drivers(nan_policy="propagate")
        for name in names:
            assert res_propagate[f"{name}_corr"].isna().all()

        # `omit` correlates over the jointly observed cells -> finite correlations within `[-1, 1]`
        res_omit = g.compute_lineage_drivers(nan_policy="omit")
        for name in names:
            corr = res_omit[f"{name}_corr"]
            assert corr.notna().any()
            valid = corr.dropna()
            assert np.all(valid >= -1.0)
            assert np.all(valid <= 1.0)

    def test_nan_policy_omit_sparse_raises(self, g: GPCCA):
        g = g.copy(deep=True)
        g.adata.X = sp.csr_matrix(g.adata.X)
        with pytest.raises(NotImplementedError, match=r".*dense.*"):
            g.compute_lineage_drivers(nan_policy="omit")

    def test_nan_policy_omit_perm_test_raises(self, g: GPCCA):
        g = g.copy(deep=True)
        g.adata.X = _to_dense(g.adata.X)
        with pytest.raises(NotImplementedError, match=r".*fisher.*"):
            g.compute_lineage_drivers(nan_policy="omit", method="perm_test", n_perms=10)

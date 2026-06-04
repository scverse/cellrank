import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import is_color_like

from cellrank._utils._colors import (
    _create_categorical_colors,
    _map_names_and_colors,
    _map_names_and_colors_from_proportions,
)


class TestColors:
    def test_create_categorical_colors_too_many_colors(self):
        with pytest.raises(ValueError, match=r".* exceeded the maximum"):
            _create_categorical_colors(1000)

    def test_create_categorical_colors_no_categories(self):
        c = _create_categorical_colors(0)

        assert c == []

    def test_create_categorical_colors_neg_categories(self):
        with pytest.raises(RuntimeError, match="Unable to create"):
            _create_categorical_colors(-1)

    def test_create_categorical_colors_normal_run(self):
        colors = _create_categorical_colors(62)

        assert len(colors) == 62
        assert all(isinstance(c, str) for c in colors), colors
        assert all(is_color_like(c) for c in colors), colors


class TestMappingColors:
    def test_mapping_colors_not_categorical(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="str")
        reference = pd.Series(["foo", np.nan, "bar", "baz"], dtype="category")

        with pytest.raises(TypeError, match=r"Query series must be"):
            _map_names_and_colors(reference, query)

    def test_mapping_colors_invalid_size(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", np.nan, "bar", "baz"], dtype="category")

        with pytest.raises(ValueError, match=r".*to have the same length"):
            _map_names_and_colors(reference, query)

    def test_mapping_colors_different_index(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category", index=[2, 3, 4])
        reference = pd.Series(["foo", "bar", "baz"], dtype="category", index=[1, 2, 3])

        with pytest.raises(ValueError, match=r"Series indices do not match"):
            _map_names_and_colors(reference, query)

    def test_mapping_colors_invalid_colors(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        with pytest.raises(ValueError, match=r"Not all values are valid colors"):
            _map_names_and_colors(reference, query, colors_reference=["red", "green", "foo"])

    def test_mapping_colors_too_few_colors(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        with pytest.raises(ValueError, match=r"Length of reference colors"):
            _map_names_and_colors(reference, query, colors_reference=["red", "green"])

    def test_mapping_colors_simple_1(self):
        x = pd.Series(["a", "b", np.nan, "b", np.nan]).astype("category")
        y = pd.Series(["b", np.nan, np.nan, "d", "a"]).astype("category")
        expected = pd.Series(["a_1", "a_2", "b"])
        expected_index = pd.Index(["a", "b", "d"])

        res = _map_names_and_colors(x, y)

        assert isinstance(res, pd.Series)
        np.testing.assert_array_equal(res.values, expected.values)
        np.testing.assert_array_equal(res.index.values, expected_index.values)

    def test_mapping_colors_simple_2(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        res = _map_names_and_colors(reference, query)

        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert isinstance(res.dtype, pd.CategoricalDtype)

    def test_mapping_colors_simple_colors(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        res, c = _map_names_and_colors(reference, query, colors_reference=["red", "green", "blue"])

        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert isinstance(res.dtype, pd.CategoricalDtype)

        assert isinstance(c, list)
        assert c == ["#ff0000", "#008000", "#0000ff"]

    def test_mapping_colors_too_many_colors(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        res, c = _map_names_and_colors(reference, query, colors_reference=["red", "green", "blue", "black"])

        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert isinstance(res.dtype, pd.CategoricalDtype)

        assert isinstance(c, list)
        assert c == ["#ff0000", "#008000", "#0000ff"]

    def test_mapping_colors_different_color_representation(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        res, c = _map_names_and_colors(reference, query, colors_reference=[(1, 0, 0), "green", (0, 0, 1, 0)])

        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert isinstance(res.dtype, pd.CategoricalDtype)

        assert isinstance(c, list)
        assert c == ["#ff0000", "#008000", "#0000ff"]

    def test_mapping_colors_non_unique_colors(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        res, c = _map_names_and_colors(reference, query, colors_reference=["red", "red", "red"])

        assert isinstance(res, pd.Series)
        assert len(res) == 3
        assert isinstance(res.dtype, pd.CategoricalDtype)

        assert isinstance(c, list)
        assert c == ["#ff0000", "#ff0000", "#ff0000"]

    def test_mapping_colors_same_reference(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "foo", "foo"], dtype="category")

        r, c = _map_names_and_colors(reference, query, colors_reference=["red", "red", "red"])

        assert list(r.index) == ["bar", "baz", "foo"]
        assert list(r.values) == ["foo_1", "foo_2", "foo_3"]
        assert c == ["#b20000", "#d13200", "#f07300"]

    def test_mapping_colors_diff_query_reference(self):
        query = pd.Series(["bar", "bar", "bar"], dtype="category")
        reference = pd.Series(["foo", "foo", "foo"], dtype="category")

        r, c = _map_names_and_colors(reference, query, colors_reference=["red", "red", "red"])

        assert list(r.index) == ["bar"]
        assert list(r.values) == ["foo"]
        assert c == ["#ff0000"]

    def test_mapping_colors_empty(self):
        query = pd.Series([], dtype="category")
        reference = pd.Series([], dtype="category")

        r = _map_names_and_colors(reference, query)

        assert isinstance(r, pd.Series)
        assert isinstance(r.dtype, pd.CategoricalDtype)

    def test_mapping_colors_empty_with_color(self):
        query = pd.Series([], dtype="category")
        reference = pd.Series([], dtype="category")

        r, c = _map_names_and_colors(reference, query, colors_reference=[])

        assert isinstance(r, pd.Series)
        assert isinstance(r.dtype, pd.CategoricalDtype)
        assert isinstance(c, list)
        assert len(c) == 0

    def test_mapping_colors_negative_en_cutoff(self):
        query = pd.Series(["foo", "bar", "baz"], dtype="category")
        reference = pd.Series(["foo", "bar", "baz"], dtype="category")

        with pytest.raises(ValueError, match=".* entropy cutoff to be non-negative"):
            _map_names_and_colors(reference, query, en_cutoff=-1)

    def test_mapping_colors_0_en_cutoff(self):
        query = pd.Series(["bar", "bar", "bar"], dtype="category")
        reference = pd.Series(["bar", "bar", "bar"], dtype="category")

        # TODO: somehow extract the custom logger and check for logs
        r = _map_names_and_colors(reference, query, en_cutoff=0)

        assert isinstance(r, pd.Series)
        assert isinstance(r.dtype, pd.CategoricalDtype)
        assert list(r.index) == ["bar"]
        assert list(r.values) == ["bar"]

    def test_mapping_colors_merging(self):
        x = pd.Series(["a", "b", np.nan, "b", np.nan]).astype("category")
        y = pd.Series(["b", np.nan, np.nan, "d", "a"]).astype("category")

        res, colors = _map_names_and_colors(x, y, colors_reference=["red", "green"])

        assert isinstance(res, pd.Series)
        assert isinstance(colors, list)
        np.testing.assert_array_equal(colors, ["#b20000", "#e65c00", "#008000"])

    def test_mapping_colors_merging_more(self):
        x = pd.Series(["a", "b", np.nan, "b", np.nan]).astype("category")
        y = pd.Series(["b", np.nan, np.nan, "d", "a"]).astype("category")

        res, colors = _map_names_and_colors(x, y, colors_reference=["red", "green", "blue", "yellow"])

        assert isinstance(res, pd.Series)
        assert isinstance(colors, list)
        np.testing.assert_array_equal(colors, ["#b20000", "#e65c00", "#008000"])

    def test_mapping_colors_name_order_same_as_cat_order(self):
        x = pd.Series(["b", "a", np.nan, "a", np.nan]).astype("category")
        y = pd.Series(["a", np.nan, np.nan, "d", "b"]).astype("category")
        expected = pd.Series(["b", "a_1", "a_2"])
        expected_index = pd.Index(["a", "b", "d"])

        res = _map_names_and_colors(x, y)

        assert isinstance(res, pd.Series)
        assert isinstance(res.dtype, pd.CategoricalDtype)
        np.testing.assert_array_equal(res.values, expected.values)
        np.testing.assert_array_equal(res.index.values, expected_index.values)
        np.testing.assert_array_equal(res.cat.categories.values, res.values)


class TestMappingColorsFromProportions:
    def test_proportions_not_categorical_query(self):
        query = pd.Series(["x", "y", "z"], dtype="str")
        proportions = pd.DataFrame({"a": [1.0, 0.0, 0.5], "b": [0.0, 1.0, 0.5]})

        with pytest.raises(TypeError, match=r"Query series must be"):
            _map_names_and_colors_from_proportions(proportions, query)

    def test_proportions_length_mismatch(self):
        query = pd.Series(["x", "y", "z"], dtype="category")
        proportions = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})

        with pytest.raises(ValueError, match=r"same length"):
            _map_names_and_colors_from_proportions(proportions, query)

    def test_proportions_negative_en_cutoff(self):
        query = pd.Series(["x", "y"], dtype="category")
        proportions = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})

        with pytest.raises(ValueError, match=r"entropy cutoff to be non-negative"):
            _map_names_and_colors_from_proportions(proportions, query, en_cutoff=-1)

    def test_proportions_too_few_colors(self):
        query = pd.Series(["x", "y"], dtype="category")
        proportions = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})

        with pytest.raises(ValueError, match=r"smaller than"):
            _map_names_and_colors_from_proportions(proportions, query, colors_reference=["red"])

    def test_proportions_empty(self):
        query = pd.Series([], dtype="category")
        proportions = pd.DataFrame({"a": [], "b": []})

        r = _map_names_and_colors_from_proportions(proportions, query)

        assert isinstance(r, pd.Series)
        assert isinstance(r.dtype, pd.CategoricalDtype)

    def test_proportions_onehot_matches_counts(self):
        # one-hot proportions + unit weights reduce exactly to the hard-count mapping
        reference = pd.Series(["a", "b", np.nan, "b", "a"], dtype="category")
        query = pd.Series(["x", "x", np.nan, "y", "y"], dtype="category")
        proportions = pd.get_dummies(reference).astype(float)
        colors = ["red", "green"]

        names_ref, colors_ref = _map_names_and_colors(reference, query, colors_reference=colors)
        names_prop, colors_prop = _map_names_and_colors_from_proportions(proportions, query, colors_reference=colors)

        np.testing.assert_array_equal(names_prop.values, names_ref.values)
        np.testing.assert_array_equal(names_prop.index.values, names_ref.index.values)
        assert colors_prop == colors_ref

    def test_proportions_dominant_category(self):
        query = pd.Series(["x", "x", "y", "y"], dtype="category")
        proportions = pd.DataFrame({"a": [0.9, 0.8, 0.1, 0.2], "b": [0.1, 0.2, 0.9, 0.8]})

        names = _map_names_and_colors_from_proportions(proportions, query)

        assert list(names.index) == ["x", "y"]
        assert list(names.values) == ["a", "b"]

    def test_proportions_ignores_nan_query(self):
        # NaN query observations contribute to no category, like the hard-count path
        query = pd.Series(["x", np.nan, "x"], dtype="category")
        proportions = pd.DataFrame({"a": [0.9, 0.0, 0.8], "b": [0.1, 1.0, 0.2]})

        names = _map_names_and_colors_from_proportions(proportions, query)

        assert list(names.index) == ["x"]
        assert list(names.values) == ["a"]

    def test_proportions_weighting_changes_name(self):
        query = pd.Series(["x", "x"], dtype="category")
        proportions = pd.DataFrame({"a": [0.6, 0.2], "b": [0.4, 0.8]})

        # unweighted: a=0.8, b=1.2 -> dominant is b
        names_unw = _map_names_and_colors_from_proportions(proportions, query)
        assert list(names_unw.values) == ["b"]

        # weighting the first row heavily: a=6.2, b=4.8 -> dominant is a
        names_w = _map_names_and_colors_from_proportions(proportions, query, weights=np.array([10.0, 1.0]))
        assert list(names_w.values) == ["a"]

    def test_proportions_duplicate_dominant_deduped(self):
        query = pd.Series(["x", "y"], dtype="category")
        proportions = pd.DataFrame({"a": [0.9, 0.8], "b": [0.1, 0.2]})

        names, colors = _map_names_and_colors_from_proportions(proportions, query, colors_reference=["red", "green"])

        assert list(names.values) == ["a_1", "a_2"]
        assert colors[0] != colors[1]

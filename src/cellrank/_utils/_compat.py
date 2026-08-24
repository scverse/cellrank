"""Compatibility shims for private APIs of dependencies that moved between releases."""

__all__ = ["add_colors_for_categorical_sample_annotation", "vega_20_scanpy"]

try:  # scanpy >= 1.13 moved the legacy plotting API under `scanpy.plotting.legacy`
    from scanpy.plotting.legacy._utils import add_colors_for_categorical_sample_annotation
    from scanpy.plotting.legacy.palettes import vega_20_scanpy
except ImportError:  # scanpy < 1.13
    from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation
    from scanpy.plotting.palettes import vega_20_scanpy

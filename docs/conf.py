# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.config import MYST_ENABLE_EXTENSIONS
from scverse_doc.registry import intersphinx

# -- Project information -----------------------------------------------------

_metadata = metadata("cellrank")
project = _metadata["Name"]
author = _metadata["Author"]
version = _metadata["Version"]
copyright = f"{datetime.now():%Y}, Theislab"
source_repository = "https://github.com/scverse/cellrank"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "scverse_doc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tippy",
]
intersphinx_mapping = intersphinx() | {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "pygam": ("https://pygam.readthedocs.io/en/latest/", None),
    "pygpcca": ("https://pygpcca.readthedocs.io/en/latest/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "scvelo": ("https://scvelo.readthedocs.io/en/latest/", None),
    "moscot": ("https://moscot.readthedocs.io/en/latest/", None),
    "ot": ("https://pythonot.github.io/", None),
}
master_doc = "index"

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# myst
myst_enable_extensions = [*MYST_ENABLE_EXTENSIONS, "substitution"]
myst_heading_anchors = 2

# hover
tippy_anchor_parent_selector = "div.content"
tippy_enable_mathjax = True
# no need because of sphinxcontrib-bibtex
tippy_enable_doitips = False

# autodoc + napoleon
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
autodoc_mock_imports = ["moscot"]

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_exclude_patterns = ["references.md"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
]

linkcheck_ignore = [
    # 403 Client Error: Forbidden for url
    r"https://doi.org/10.1063/1.5064530",
    r"https://doi.org/10.1021/acs.jctc.8b00079",
    r"https://doi.org/10.1093/bioinformatics/btv325",
    r"https://doi.org/10.1242/dev.173849",
    r"https://doi.org/10.1242/dev.126011",
    r"https://doi.org/10.2337/db20-0599",
    r"https://doi.org/10.1210/en.2013-1663",
    r"https://doi.org/10.4161/isl.21984",
    r"https://doi.org/10.15252/msb.202110282",
    r"https://doi.org/10.1073/pnas.0500334102",
    r"https://doi.org/10.1126/science.aar3131",
    r"https://doi.org/10.1126/science.aax0249",
    r"https://doi.org/10.1126/science.aax3072",
    r"https://www.science.org/doi/full/10.1126/science.1247651",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/README.rst",
    "notebooks/notebooks/MK_2020-10-19_generate_docs_figure.ipynb",
    "release/changelog/*",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_static_path = ["_static"]
html_css_files = ["css/override.css"]

html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "image_light": "_static/img/light_mode_logo.png",
        "image_dark": "_static/img/dark_mode_logo.png",
    },
    "secondary_accent": True,
}

import quantem.widget

project = "quantem.widget"
copyright = "2026, quantem contributors"
version = quantem.widget.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "widgets"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/bobleesj/quantem.widget",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "show_nav_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/quantem-widget/",
            "icon": "fa-solid fa-box",
        },
    ],
}
html_sidebars = {
    "examples/**": [],  # no sidebars on example pages — full width for widgets
}

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "exclude-members": "__init__",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

nbsphinx_execute = "never"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autosummary_generate = True

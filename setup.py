"""Setup script for dss-pollution-extraction package."""

import functools
import operator
from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
try:
    with open("requirements.txt") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    # Fallback requirements if file doesn't exist
    requirements = [
        "xarray>=2022.3.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "geopandas>=0.10.0",
        "rioxarray>=0.11.0",
        "rasterio>=1.2.0",
        "shapely>=1.8.0",
        "cartopy>=0.20.0",
        "scipy>=1.8.0",
        "dask>=2022.1.0",
        "netcdf4>=1.5.0",
        "pyproj>=3.3.0",
    ]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "isort>=5.10.0",
        "mypy>=0.950",
        "pre-commit>=2.17.0",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        "nbsphinx>=0.8.0",
        "sphinx-autoapi>=1.8.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipywidgets>=7.7.0",
        "folium>=0.12.0",
        "plotly>=5.7.0",
    ],
    "performance": ["numba>=0.56.0", "bottleneck>=1.3.0", "cython>=0.29.0"],
}

# All extras combined
extras_require["all"] = list(
    set(functools.reduce(operator.iadd, extras_require.values(), []))
)

setup(
    name="dss-pollution-extraction",
    version="1.0.1",
    author="Muhammad Shafeeque",
    author_email="muhammad.shafeeque@awi.de",
    maintainer="Muhammad Shafeeque",
    maintainer_email="shafeequ@uni-bremen.de",
    description=(
        "A comprehensive package for analyzing pollution data from NetCDF files"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MuhammadShafeeque/dss-pollution-extraction",
    project_urls={
        "Bug Tracker": "https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues",
        "Documentation": "https://dss-pollution-extraction.readthedocs.io/",
        "Source Code": "https://github.com/MuhammadShafeeque/dss-pollution-extraction",
        "Download": "https://pypi.org/project/dss-pollution-extraction/",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "pollution_extraction": [
            "config/*.json",
            "config/*.yaml",
            "examples/*.py",
            "tests/*.py",
        ]
    },
    entry_points={
        "console_scripts": [
            "dss-pollution-analyze=pollution_extraction.cli:main",
            "pollution-analyze=pollution_extraction.cli:main",  # Alias
        ],
    },
    keywords=[
        "pollution",
        "air quality",
        "atmospheric science",
        "netcdf",
        "xarray",
        "geospatial",
        "environmental data",
        "PM2.5",
        "NO2",
        "black carbon",
        "data analysis",
        "visualization",
        "GIS",
        "decision support system",
        "DSS",
        "environmental monitoring",
        "temporal analysis",
        "spatial analysis",
        "NUTS3",
        "Europe",
        "LAEA",
        "downscaling",
        "AWI",
        "climate data",
    ],
    zip_safe=False,
    test_suite="tests",
    tests_require=["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    # Additional metadata
    platforms=["any"],
    license="MIT",
    # Package discovery
    package_dir={"": "."},
    # Data files
    data_files=[
        ("config", ["config/sample_config.yaml"]),
    ],
    # Options for different build systems
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python",
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
    # Setuptools specific options
    setup_requires=["setuptools>=45", "wheel", "setuptools_scm"],
    # Version management
    use_scm_version={
        "write_to": "pollution_extraction/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    # Download URL template
    download_url="https://github.com/MuhammadShafeeque/dss-pollution-extraction/archive/v1.0.0.tar.gz",
)

# Post-installation message
print(
    """
========================================
DSS Pollution Extraction Package
========================================

Thank you for installing dss-pollution-extraction!

Quick Start:
1. Import the package:
   from pollution_extraction import PollutionAnalyzer

2. Analyze your data:
   analyzer = PollutionAnalyzer('your_data.nc', pollution_type='pm25')
   analyzer.print_summary()

3. Use the command-line tool:
   dss-pollution-analyze data.nc --type pm25 --info

Documentation: https://dss-pollution-extraction.readthedocs.io/
Examples: https://github.com/MuhammadShafeeque/dss-pollution-extraction/tree/main/examples
Issues: https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues

Developed by Muhammad Shafeeque
Alfred Wegener Institute (AWI) & University of Bremen
========================================
"""
)

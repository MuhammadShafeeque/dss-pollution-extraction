"""
Pollution Data Extraction and Analysis Package

A comprehensive package for analyzing pollution data from NetCDF files,
including temporal aggregations, spatial extractions, and visualizations.
"""

from .core.data_reader import PollutionDataReader
from .core.temporal_aggregator import TemporalAggregator
from .core.spatial_extractor import SpatialExtractor
from .core.data_visualizer import DataVisualizer
from .core.data_exporter import DataExporter
from .analyzer import PollutionAnalyzer

__version__ = "1.0.0"
__author__ = "Muhammad Shafeeque"
__email__ = "muhammad.shafeeque@awi.de"
__institution__ = "Alfred Wegener Institute (AWI)"
__license__ = "MIT"

__all__ = [
    "PollutionDataReader",
    "TemporalAggregator",
    "SpatialExtractor",
    "DataVisualizer",
    "DataExporter",
    "PollutionAnalyzer",
]

# Package metadata
__title__ = "DSS Pollution Extraction"
__description__ = "Comprehensive pollution data analysis from NetCDF files"
__url__ = "https://github.com/MuhammadShafeeque/dss-pollution-extraction"
__download_url__ = "https://pypi.org/project/dss-pollution-extraction/"
__docs_url__ = "https://dss-pollution-extraction.readthedocs.io/"

# Version info
__version_info__ = tuple(map(int, __version__.split(".")))

# Supported pollution types
SUPPORTED_POLLUTANTS = {
    "bc": "Black Carbon",
    "no2": "Nitrogen Dioxide",
    "pm25": "Particulate Matter 2.5",
    "pm10": "Particulate Matter 10",
}


# Quick access functions
def get_version():
    """Get package version."""
    return __version__


def get_supported_pollutants():
    """Get list of supported pollution types."""
    return SUPPORTED_POLLUTANTS


def quick_analysis(file_path, pollution_type=None, output_dir="./output"):
    """
    Quick analysis function for basic pollution data analysis.

    Parameters
    ----------
    file_path : str
        Path to NetCDF pollution data file
    pollution_type : str, optional
        Type of pollution ('bc', 'no2', 'pm25', 'pm10')
    output_dir : str
        Output directory for results

    Returns
    -------
    PollutionAnalyzer
        Configured analyzer instance
    """
    analyzer = PollutionAnalyzer(file_path, pollution_type=pollution_type)

    # Print basic info
    print(f"Quick Analysis: {analyzer.pollution_type.upper()}")
    analyzer.print_summary()

    return analyzer


# Import error handling
try:
    import xarray as xr
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"Required dependency missing: {e}\n"
        "Please install with: pip install dss-pollution-extraction"
    )

# Optional dependency warnings
try:
    import cartopy
except ImportError:
    import warnings

    warnings.warn(
        "Cartopy not available. Geographic projections will be limited.", ImportWarning
    )

try:
    import geopandas
except ImportError:
    import warnings

    warnings.warn(
        "GeoPandas not available. Spatial operations will be limited.", ImportWarning
    )

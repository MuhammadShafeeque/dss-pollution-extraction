"""
Pollution Extraction Package

A comprehensive Python package for extracting, analyzing, and visualizing pollution data
from various data sources including NetCDF files, with special focus on air quality
monitoring and health impact assessment.
"""

__version__ = "0.1.0"
__author__ = "MuhammadShafeeque"
__email__ = "120993860+MuhammadShafeeque@users.noreply.github.com"

from .analyzer import PollutionAnalyzer
from .config import Config

__all__ = [
    "PollutionAnalyzer",
    "Config",
]

"""
Test package for pollution_extraction library.

This package contains comprehensive tests for all modules in the pollution_extraction library.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEMP_DIR = Path(tempfile.gettempdir()) / "pollution_extraction_tests"


def setup_test_environment():
    """Setup test environment and directories."""
    TEMP_DIR.mkdir(exist_ok=True)
    TEST_DATA_DIR.mkdir(exist_ok=True)


def cleanup_test_environment():
    """Clean up test environment."""
    import shutil

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


def create_sample_netcdf():
    """Create a sample NetCDF file for testing."""
    # Create sample data
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    lats = np.linspace(40, 60, 20)  # Europe-like coordinates
    lons = np.linspace(-10, 30, 30)

    # Create data arrays
    pm25_data = np.random.uniform(5, 50, (len(times), len(lats), len(lons)))
    no2_data = np.random.uniform(10, 80, (len(times), len(lats), len(lons)))

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "pm25": (
                ["time", "latitude", "longitude"],
                pm25_data,
                {"units": "μg/m³", "long_name": "PM2.5 concentration"},
            ),
            "no2": (
                ["time", "latitude", "longitude"],
                no2_data,
                {"units": "μg/m³", "long_name": "NO2 concentration"},
            ),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    return ds


def create_sample_csv():
    """Create sample CSV data for testing."""
    data = {
        "latitude": [52.5, 48.8, 41.9, 55.7],
        "longitude": [13.4, 2.3, 12.5, 37.6],
        "city": ["Berlin", "Paris", "Rome", "Moscow"],
        "pm25": [15.2, 18.7, 23.1, 19.8],
        "no2": [25.4, 31.2, 28.9, 22.1],
    }
    return pd.DataFrame(data)


def create_sample_geojson():
    """Create sample GeoJSON data for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Berlin", "id": "DE1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [13.0, 52.3],
                            [13.8, 52.3],
                            [13.8, 52.7],
                            [13.0, 52.7],
                            [13.0, 52.3],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Paris", "id": "FR1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2.0, 48.6],
                            [2.6, 48.6],
                            [2.6, 49.0],
                            [2.0, 49.0],
                            [2.0, 48.6],
                        ]
                    ],
                },
            },
        ],
    }


# Common test fixtures
SAMPLE_COORDINATES = [
    (52.5200, 13.4050),  # Berlin
    (48.8566, 2.3522),  # Paris
    (41.9028, 12.4964),  # Rome
    (55.7558, 37.6176),  # Moscow
]

SAMPLE_REGIONS = [
    {"name": "Central Europe", "bounds": [10, 45, 20, 55]},
    {"name": "Western Europe", "bounds": [0, 45, 10, 55]},
]

SAMPLE_TIME_RANGES = [
    ("2024-01-01", "2024-01-31"),
    ("2024-06-01", "2024-06-30"),
    ("2024-12-01", "2024-12-31"),
]

# Health thresholds for testing
WHO_THRESHOLDS = {
    "pm25": 15,  # μg/m³ annual mean
    "pm10": 45,  # μg/m³ annual mean
    "no2": 25,  # μg/m³ annual mean
    "o3": 100,  # μg/m³ daily max 8-hour mean
}

EU_THRESHOLDS = {
    "pm25": 25,  # μg/m³ annual mean
    "pm10": 50,  # μg/m³ annual mean
    "no2": 40,  # μg/m³ annual mean
    "o3": 120,  # μg/m³ daily max 8-hour mean
}

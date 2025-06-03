"""
Integration tests for pollution_extraction library.

Tests verify that all components work together correctly in realistic workflows.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pollution_extraction import PollutionAnalyzer
from tests import (
    TEMP_DIR,
    cleanup_test_environment,
    setup_test_environment,
)

logger = logging.getLogger(__name__)


class TestPollutionAnalyzerIntegration:
    """Test complete workflows using PollutionAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup test environment and synthetic test data."""
        setup_test_environment()

        # Create test dataset coordinates
        times = pd.date_range("2024-01-01", periods=365, freq="D")
        lats = np.linspace(35, 70, 36)  # Europe coverage
        lons = np.linspace(-10, 40, 51)

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # Create base PM2.5 field with urban hotspots
        urban_centers = [
            (52.5, 13.4, 35),  # Berlin
            (48.8, 2.3, 40),  # Paris
            (51.5, -0.1, 45),  # London
            (41.9, 12.5, 38),  # Rome
        ]

        # Initialize base field
        pm25_base = np.full_like(lat_grid, 12.0)  # Rural background

        # Add urban pollution hotspots
        for city_lat, city_lon, intensity in urban_centers:
            dist_sq = (lat_grid - city_lat) ** 2 + (lon_grid - city_lon) ** 2
            pm25_base += intensity * np.exp(-dist_sq / 4)

        # Create temporal patterns with realistic variations
        pm25_data = np.zeros((len(times), len(lats), len(lons)))
        for i, dtime in enumerate(times):
            # Seasonal cycle
            day = dtime.dayofyear
            seasonal = 1 + 0.4 * np.sin((day - 80) * 2 * np.pi / 365)

            # Winter heating
            winter = 1 + 0.3 * np.exp(-((day - 15) ** 2 + (day - 350) ** 2) / 1000)

            # Weekend effect
            weekend = 1 - 0.2 * (dtime.weekday() >= 5)

            # Weather variations
            weather = 1 + 0.2 * np.sin(i * 2 * np.pi / 10)

            # Combine factors
            pm25_data[i] = pm25_base * seasonal * winter * weekend * weather
            pm25_data[i] += np.random.normal(0, 2, pm25_data[i].shape)

        # Ensure non-negative values
        pm25_data = np.maximum(pm25_data, 0)

        # Create xarray Dataset
        self.data = xr.Dataset(
            {
                "PM2p5_downscaled": (
                    ["time", "latitude", "longitude"],
                    pm25_data,
                    {
                        "units": "μg/m³",
                        "long_name": "PM2.5 mass concentration",
                        "standard_name": (
                            "mass_concentration_of_pm2p5_"
                            "ambient_aerosol_particles_in_air"
                        ),
                    },
                ),
            },
            coords={
                "time": ("time", times, {"long_name": "time", "standard_name": "time"}),
                "latitude": (
                    "latitude",
                    lats,
                    {
                        "units": "degrees_north",
                        "long_name": "latitude",
                        "standard_name": "latitude",
                    },
                ),
                "longitude": (
                    "longitude",
                    lons,
                    {
                        "units": "degrees_east",
                        "long_name": "longitude",
                        "standard_name": "longitude",
                    },
                ),
            },
            attrs={
                "title": "Test Air Quality Dataset - Europe",
                "source": "Synthetic data for testing",
                "description": "Synthetic PM2.5 data for Europe",
            },
        )

        # Save test dataset
        self.data_file = TEMP_DIR / "test_pm25.nc"
        self.data.to_netcdf(self.data_file)

        # Test points
        self.test_points = [
            (52.5, 13.4),  # Berlin
            (48.8, 2.3),  # Paris
            (51.5, -0.1),  # London
        ]

        yield
        cleanup_test_environment()

    def test_basic_workflow(self) -> None:
        """Test basic analysis workflow."""
        # Initialize analyzer
        analyzer = PollutionAnalyzer(str(self.data_file), pollution_type="pm25")

        # Verify initialization
        assert analyzer.dataset is not None
        assert analyzer.pollution_type == "pm25"
        assert analyzer.pollution_variable == "PM2p5_downscaled"

        # Test info retrieval
        info = analyzer.get_info()
        assert info["basic_info"]["total_time_steps"] == 365
        assert info["data_summary"]["mean"] > 0

        # Test temporal analysis
        monthly = analyzer.get_monthly_averages()
        assert isinstance(monthly, xr.Dataset)
        assert len(monthly.time) == 12

        # Test spatial extraction
        points = analyzer.extract_at_points(self.test_points)
        assert isinstance(points, xr.Dataset)
        assert points.sizes["location"] == len(self.test_points)

        # Test exports
        out_dir = TEMP_DIR / "test_out"
        out_dir.mkdir(exist_ok=True)

        analyzer.export_to_csv(out_dir / "data.csv")
        assert (out_dir / "data.csv").exists()

        analyzer.export_to_netcdf(out_dir / "data.nc")
        assert (out_dir / "data.nc").exists()

        # Test comprehensive analysis
        results = analyzer.comprehensive_analysis(out_dir)
        assert isinstance(results, dict)
        assert "dataset_info" in results
        assert "temporal_analysis" in results

    def test_temporal_analysis(self) -> None:
        """Test temporal analysis features."""
        analyzer = PollutionAnalyzer(str(self.data_file), pollution_type="pm25")

        monthly = analyzer.get_monthly_averages()
        annual = analyzer.get_annual_averages()
        seasonal = analyzer.get_seasonal_averages()

        assert isinstance(monthly, xr.Dataset)
        assert isinstance(annual, xr.Dataset)
        assert isinstance(seasonal, xr.Dataset)

        # Verify seasonal patterns
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]

        winter = monthly.sel(time=monthly.time.dt.month.isin(winter_months))
        summer = monthly.sel(time=monthly.time.dt.month.isin(summer_months))

        # Winter PM2.5 should be higher due to heating/meteorology
        assert (
            winter.PM2p5_downscaled.mean().values
            > summer.PM2p5_downscaled.mean().values
        )

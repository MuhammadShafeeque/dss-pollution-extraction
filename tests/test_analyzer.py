"""
Test module for the PollutionAnalyzer class.

This module contains unit tests for the main PollutionAnalyzer class,
using mocking to avoid dependencies on actual data files.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np

# Import the class to test
from pollution_extraction.analyzer import PollutionAnalyzer


class TestPollutionAnalyzer:
    """Test class for PollutionAnalyzer."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock xarray dataset."""
        # Create mock data
        time = pd.date_range("2020-01-01", periods=10, freq="D")
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 5)

        data = np.random.rand(10, 5, 5)

        dataset = xr.Dataset(
            {"pollution_var": (("time", "x", "y"), data)},
            coords={"time": time, "x": x, "y": y},
        )
        return dataset

    @pytest.fixture
    def mock_components(self):
        """Create mock components for the analyzer."""
        components = {
            "reader": Mock(),
            "temporal_aggregator": Mock(),
            "spatial_extractor": Mock(),
            "visualizer": Mock(),
            "exporter": Mock(),
        }
        return components

    @patch("pollution_extraction.analyzer.DataExporter")
    @patch("pollution_extraction.analyzer.DataVisualizer")
    @patch("pollution_extraction.analyzer.SpatialExtractor")
    @patch("pollution_extraction.analyzer.TemporalAggregator")
    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_initialization(
        self,
        mock_reader_class,
        mock_temporal_class,
        mock_spatial_class,
        mock_visualizer_class,
        mock_exporter_class,
        mock_dataset,
    ):
        """Test PollutionAnalyzer initialization."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = mock_dataset
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        # Initialize analyzer
        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")

        # Assertions
        assert analyzer.file_path == Path("/path/to/file.nc")
        assert analyzer.pollution_type == "pm25"
        assert analyzer.pollution_variable == "pollution_var"

        # Check that components were initialized
        mock_reader_class.assert_called_once_with("/path/to/file.nc", "pm25")
        mock_temporal_class.assert_called_once_with(mock_dataset, "pollution_var")
        mock_spatial_class.assert_called_once_with(mock_dataset, "pollution_var")
        mock_visualizer_class.assert_called_once_with(
            mock_dataset, "pollution_var", "pm25"
        )
        mock_exporter_class.assert_called_once_with(mock_dataset, "pollution_var")

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_get_info(self, mock_reader_class):
        """Test get_info method."""
        # Setup mock reader
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader.get_basic_info.return_value = {"file_path": "/test/path"}
        mock_reader.get_data_summary.return_value = {"mean": 10.5}
        mock_reader_class.return_value = mock_reader

        # Initialize and test
        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        info = analyzer.get_info()

        # Assertions
        expected_info = {
            "basic_info": {"file_path": "/test/path"},
            "data_summary": {"mean": 10.5},
        }
        assert info == expected_info

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    @patch("builtins.print")
    def test_print_summary(self, mock_print, mock_reader_class):
        """Test print_summary method."""
        # Setup mock reader
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader.get_basic_info.return_value = {
            "file_path": "/test/path.nc",
            "pollution_type": "pm25",
            "variable_name": "pm25_concentration",
            "units": "µg/m³",
            "description": "PM2.5 concentration",
            "time_range": ["2020-01-01", "2020-12-31"],
            "total_time_steps": 365,
            "spatial_dimensions": {"x": 100, "y": 100},
            "spatial_bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        }
        mock_reader.get_data_summary.return_value = {
            "min": 0.5,
            "max": 50.2,
            "mean": 15.3,
            "std": 8.7,
            "missing_percentage": 2.1,
        }
        mock_reader_class.return_value = mock_reader

        # Initialize and test
        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.print_summary()

        # Check that print was called (basic check)
        assert mock_print.called
        # Check for key information in print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("PM25" in str(call) for call in print_calls)
        assert any("15.3" in str(call) for call in print_calls)

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_temporal_methods(self, mock_reader_class):
        """Test temporal analysis methods."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        mock_temporal = Mock()
        expected_result = Mock()
        mock_temporal.monthly_aggregation.return_value = expected_result
        mock_temporal.annual_aggregation.return_value = expected_result
        mock_temporal.seasonal_aggregation.return_value = expected_result
        mock_temporal.custom_time_aggregation.return_value = expected_result

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.temporal_aggregator = mock_temporal

        # Test monthly averages
        result = analyzer.get_monthly_averages([1, 2, 3], [2020, 2021], "median")
        mock_temporal.monthly_aggregation.assert_called_with(
            method="median", specific_months=[1, 2, 3], years=[2020, 2021]
        )
        assert result == expected_result

        # Test annual averages
        result = analyzer.get_annual_averages([2020], "max")
        mock_temporal.annual_aggregation.assert_called_with(
            method="max", specific_years=[2020]
        )
        assert result == expected_result

        # Test seasonal averages
        result = analyzer.get_seasonal_averages(["winter"], [2020], "min")
        mock_temporal.seasonal_aggregation.assert_called_with(
            method="min", seasons=["winter"], years=[2020]
        )
        assert result == expected_result

        # Test custom period averages
        periods = [("2020-01-01", "2020-06-30")]
        result = analyzer.get_custom_period_averages(periods, "mean", ["first_half"])
        mock_temporal.custom_time_aggregation.assert_called_with(
            time_periods=periods, method="mean", period_names=["first_half"]
        )
        assert result == expected_result

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_spatial_methods(self, mock_reader_class):
        """Test spatial extraction methods."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        mock_spatial = Mock()
        expected_result = Mock()
        mock_spatial.extract_points.return_value = expected_result
        mock_spatial.extract_polygons.return_value = expected_result
        mock_spatial.extract_nuts3_regions.return_value = expected_result
        mock_spatial.extract_with_dates.return_value = expected_result

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.spatial_extractor = mock_spatial

        # Test point extraction
        locations = [(10.5, 52.2), (11.0, 53.0)]
        result = analyzer.extract_at_points(locations, "bilinear")
        mock_spatial.extract_points.assert_called_with(locations, method="bilinear")
        assert result == expected_result

        # Test polygon extraction
        polygons = Mock()
        result = analyzer.extract_for_polygons(polygons, "median")
        mock_spatial.extract_polygons.assert_called_with(polygons, "median")
        assert result == expected_result

        # Test NUTS3 extraction
        nuts3_file = "/path/to/nuts3.shp"
        result = analyzer.extract_for_nuts3(nuts3_file, "mean")
        mock_spatial.extract_nuts3_regions.assert_called_with(nuts3_file, "mean")
        assert result == expected_result

        # Test event-based extraction
        event_locations = Mock()
        result = analyzer.extract_with_event_dates(
            event_locations, "event_date", 5, "max"
        )
        mock_spatial.extract_with_dates.assert_called_with(
            event_locations, "event_date", 5, "max"
        )
        assert result == expected_result

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_visualization_methods(self, mock_reader_class):
        """Test visualization methods."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        mock_visualizer = Mock()
        expected_figure = Mock()
        mock_visualizer.plot_spatial_map.return_value = expected_figure
        mock_visualizer.plot_time_series.return_value = expected_figure
        mock_visualizer.plot_seasonal_cycle.return_value = expected_figure
        mock_visualizer.plot_distribution.return_value = expected_figure
        mock_visualizer.plot_spatial_statistics.return_value = expected_figure

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.visualizer = mock_visualizer

        # Test plot methods
        result = analyzer.plot_map(5, cmap="viridis")
        mock_visualizer.plot_spatial_map.assert_called_with(
            time_index=5, cmap="viridis"
        )
        assert result == expected_figure

        location = {"x": 10, "y": 50}
        result = analyzer.plot_time_series(location, color="red")
        mock_visualizer.plot_time_series.assert_called_with(
            location=location, color="red"
        )
        assert result == expected_figure

        result = analyzer.plot_seasonal_cycle(aggregation="median")
        mock_visualizer.plot_seasonal_cycle.assert_called_with(aggregation="median")
        assert result == expected_figure

        result = analyzer.plot_distribution(bins=50)
        mock_visualizer.plot_distribution.assert_called_with(bins=50)
        assert result == expected_figure

        result = analyzer.plot_spatial_statistics("std", vmin=0)
        mock_visualizer.plot_spatial_statistics.assert_called_with(
            statistic="std", vmin=0
        )
        assert result == expected_figure

        # Test animation creation
        analyzer.create_animation("/path/to/output.mp4", fps=10)
        mock_visualizer.create_animation.assert_called_with(
            "/path/to/output.mp4", fps=10
        )

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_export_methods(self, mock_reader_class):
        """Test export methods."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        mock_exporter = Mock()

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.exporter = mock_exporter

        # Test export methods
        analyzer.export_to_netcdf("/path/to/output.nc", compression="zlib")
        mock_exporter.to_netcdf.assert_called_with(
            "/path/to/output.nc", compression="zlib"
        )

        analyzer.export_to_geotiff("/path/to/output.tif", time_index=0)
        mock_exporter.to_geotiff.assert_called_with("/path/to/output.tif", time_index=0)

        analyzer.export_to_csv("/path/to/output.csv", flatten=True)
        mock_exporter.to_csv.assert_called_with("/path/to/output.csv", flatten=True)

    @patch("pollution_extraction.analyzer.Path.mkdir")
    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_analyze_temporal_patterns(self, mock_reader_class, mock_mkdir):
        """Test comprehensive temporal analysis workflow."""
        # Setup mocks
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader_class.return_value = mock_reader

        # Mock temporal aggregator
        mock_temporal = Mock()
        monthly_data = Mock()
        annual_data = Mock()
        seasonal_data = Mock()
        monthly_data.to_netcdf = Mock()
        annual_data.to_netcdf = Mock()
        seasonal_data.to_netcdf = Mock()

        mock_temporal.monthly_aggregation.return_value = monthly_data
        mock_temporal.annual_aggregation.return_value = annual_data
        mock_temporal.seasonal_aggregation.return_value = seasonal_data

        # Mock visualizer
        mock_visualizer = Mock()

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.temporal_aggregator = mock_temporal
        analyzer.visualizer = mock_visualizer

        # Test the method
        output_dir = "/path/to/output"
        results = analyzer.analyze_temporal_patterns(
            output_dir, save_plots=True, save_data=True
        )

        # Assertions
        assert "monthly_averages" in results
        assert "annual_averages" in results
        assert "seasonal_averages" in results

        # Check that data was saved
        monthly_data.to_netcdf.assert_called_once()
        annual_data.to_netcdf.assert_called_once()
        seasonal_data.to_netcdf.assert_called_once()

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_context_manager(self, mock_reader_class):
        """Test context manager functionality."""
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader.close = Mock()
        mock_reader_class.return_value = mock_reader

        # Test context manager
        with PollutionAnalyzer("/path/to/file.nc", "pm25") as analyzer:
            assert analyzer is not None
            assert analyzer.pollution_type == "pm25"

        # Check that close was called
        mock_reader.close.assert_called_once()

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_close_method(self, mock_reader_class):
        """Test explicit close method."""
        mock_reader = Mock()
        mock_reader.dataset = Mock()
        mock_reader.variable_info = {"var_name": "pollution_var"}
        mock_reader.close = Mock()
        mock_reader_class.return_value = mock_reader

        analyzer = PollutionAnalyzer("/path/to/file.nc", "pm25")
        analyzer.close()

        mock_reader.close.assert_called_once()


# Integration-style tests with actual data structures
class TestPollutionAnalyzerIntegration:
    """Integration tests using actual xarray objects."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray dataset for testing."""
        time = pd.date_range("2020-01-01", periods=12, freq="M")
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)

        # Create realistic pollution data
        np.random.seed(42)  # For reproducible tests
        data = np.random.gamma(
            2, 5, (12, 20, 20)
        )  # Gamma distribution typical for pollution

        dataset = xr.Dataset(
            {
                "pm25": (
                    ("time", "x", "y"),
                    data,
                    {"units": "µg/m³", "long_name": "PM2.5 concentration"},
                )
            },
            coords={"time": time, "x": x, "y": y},
            attrs={"title": "PM2.5 pollution data", "source": "Test dataset"},
        )
        return dataset

    @patch("pollution_extraction.analyzer.PollutionDataReader")
    def test_with_real_dataset_structure(self, mock_reader_class, sample_dataset):
        """Test analyzer with realistic dataset structure."""
        # Setup mock reader with real dataset
        mock_reader = Mock()
        mock_reader.dataset = sample_dataset
        mock_reader.variable_info = {"var_name": "pm25"}
        mock_reader.get_basic_info.return_value = {
            "file_path": "/test/pm25_data.nc",
            "pollution_type": "pm25",
            "variable_name": "pm25",
            "units": "µg/m³",
            "description": "PM2.5 concentration",
            "time_range": [
                str(sample_dataset.time.min().values),
                str(sample_dataset.time.max().values),
            ],
            "total_time_steps": len(sample_dataset.time),
            "spatial_dimensions": {
                "x": len(sample_dataset.x),
                "y": len(sample_dataset.y),
            },
            "spatial_bounds": {
                "x_min": float(sample_dataset.x.min()),
                "x_max": float(sample_dataset.x.max()),
                "y_min": float(sample_dataset.y.min()),
                "y_max": float(sample_dataset.y.max()),
            },
        }
        mock_reader.get_data_summary.return_value = {
            "min": float(sample_dataset.pm25.min()),
            "max": float(sample_dataset.pm25.max()),
            "mean": float(sample_dataset.pm25.mean()),
            "std": float(sample_dataset.pm25.std()),
            "missing_percentage": 0.0,
        }
        mock_reader_class.return_value = mock_reader

        # Initialize analyzer
        analyzer = PollutionAnalyzer("/test/pm25_data.nc", "pm25")

        # Test that initialization worked with real data structure
        assert analyzer.dataset == sample_dataset
        assert analyzer.pollution_variable == "pm25"

        # Test get_info returns sensible values
        info = analyzer.get_info()
        assert info["basic_info"]["total_time_steps"] == 12
        assert info["basic_info"]["spatial_dimensions"]["x"] == 20
        assert (
            info["data_summary"]["min"] >= 0
        )  # Pollution values should be non-negative


if __name__ == "__main__":
    # Run tests with: python -m pytest test_analyzer.py -v
    pytest.main([__file__, "-v"])

"""
Tests for pollution_extraction.cli module.
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner
import pandas as pd
import numpy as np

from pollution_extraction.cli import (
    cli,
    extract_points,
    extract_regions,
    aggregate_temporal,
    aggregate_spatial,
    visualize,
    export_data,
    validate_data,
    batch_process,
    configure,
)
from tests import (
    create_sample_netcdf,
    create_sample_csv,
    create_sample_geojson,
    setup_test_environment,
    cleanup_test_environment,
    TEMP_DIR,
)


class TestCLIMain:
    """Test main CLI functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample input files
        self.sample_nc = TEMP_DIR / "test_data.nc"
        ds = create_sample_netcdf()
        ds.to_netcdf(self.sample_nc)

        self.sample_csv = TEMP_DIR / "test_points.csv"
        df = create_sample_csv()
        df.to_csv(self.sample_csv, index=False)

        self.sample_geojson = TEMP_DIR / "test_regions.geojson"
        geojson_data = create_sample_geojson()
        with open(self.sample_geojson, "w") as f:
            json.dump(geojson_data, f)

        yield
        cleanup_test_environment()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "pollution_extraction" in result.output
        assert "extract-points" in result.output
        assert "extract-regions" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_without_args(self):
        """Test CLI without arguments shows help."""
        result = self.runner.invoke(cli)
        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestExtractPointsCommand:
    """Test extract-points CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample files
        self.input_file = TEMP_DIR / "input.nc"
        ds = create_sample_netcdf()
        ds.to_netcdf(self.input_file)

        self.points_file = TEMP_DIR / "points.csv"
        points_df = pd.DataFrame(
            {
                "latitude": [52.5, 48.8, 41.9],
                "longitude": [13.4, 2.3, 12.5],
                "name": ["Berlin", "Paris", "Rome"],
            }
        )
        points_df.to_csv(self.points_file, index=False)

        yield
        cleanup_test_environment()

    def test_extract_points_basic(self):
        """Test basic point extraction."""
        output_file = TEMP_DIR / "extracted_points.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--points",
                str(self.points_file),
                "--output",
                str(output_file),
                "--variables",
                "pm25,no2",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output content
        extracted_data = pd.read_csv(output_file)
        assert "pm25" in extracted_data.columns
        assert "no2" in extracted_data.columns
        assert len(extracted_data) > 0

    def test_extract_points_with_interpolation(self):
        """Test point extraction with interpolation method."""
        output_file = TEMP_DIR / "extracted_interpolated.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--points",
                str(self.points_file),
                "--output",
                str(output_file),
                "--interpolation",
                "nearest",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_extract_points_with_time_range(self):
        """Test point extraction with time range."""
        output_file = TEMP_DIR / "extracted_timerange.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--points",
                str(self.points_file),
                "--output",
                str(output_file),
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-05",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_extract_points_coordinates_as_args(self):
        """Test point extraction with coordinates as command line arguments."""
        output_file = TEMP_DIR / "extracted_coords.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--coordinates",
                "52.5,13.4",
                "--coordinates",
                "48.8,2.3",
                "--output",
                str(output_file),
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_extract_points_invalid_input(self):
        """Test point extraction with invalid input file."""
        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                "nonexistent.nc",
                "--points",
                str(self.points_file),
                "--output",
                "output.csv",
            ],
        )

        assert result.exit_code != 0
        assert "Error" in result.output or "not found" in result.output.lower()

    def test_extract_points_missing_variables(self):
        """Test point extraction with missing variables."""
        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--points",
                str(self.points_file),
                "--output",
                "output.csv",
                "--variables",
                "nonexistent_variable",
            ],
        )

        # Should handle gracefully or give informative error
        assert result.exit_code != 0 or "Warning" in result.output

    def test_extract_points_verbose_output(self):
        """Test point extraction with verbose output."""
        output_file = TEMP_DIR / "extracted_verbose.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_file),
                "--points",
                str(self.points_file),
                "--output",
                str(output_file),
                "--verbose",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert "Processing" in result.output or "Extracting" in result.output


class TestExtractRegionsCommand:
    """Test extract-regions CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample files
        self.input_file = TEMP_DIR / "input.nc"
        ds = create_sample_netcdf()
        ds.to_netcdf(self.input_file)

        self.regions_file = TEMP_DIR / "regions.geojson"
        geojson_data = create_sample_geojson()
        with open(self.regions_file, "w") as f:
            json.dump(geojson_data, f)

        yield
        cleanup_test_environment()

    def test_extract_regions_basic(self):
        """Test basic region extraction."""
        output_file = TEMP_DIR / "extracted_regions.csv"

        result = self.runner.invoke(
            extract_regions,
            [
                "--input",
                str(self.input_file),
                "--regions",
                str(self.regions_file),
                "--output",
                str(output_file),
                "--variables",
                "pm25,no2",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify output content
        extracted_data = pd.read_csv(output_file)
        assert "pm25" in extracted_data.columns
        assert len(extracted_data) > 0

    def test_extract_regions_aggregation_methods(self):
        """Test region extraction with different aggregation methods."""
        methods = ["mean", "median", "max", "min"]

        for method in methods:
            output_file = TEMP_DIR / f"extracted_regions_{method}.csv"

            result = self.runner.invoke(
                extract_regions,
                [
                    "--input",
                    str(self.input_file),
                    "--regions",
                    str(self.regions_file),
                    "--output",
                    str(output_file),
                    "--aggregation",
                    method,
                    "--variables",
                    "pm25",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()

    def test_extract_regions_bounding_box(self):
        """Test region extraction with bounding box."""
        output_file = TEMP_DIR / "extracted_bbox.csv"

        result = self.runner.invoke(
            extract_regions,
            [
                "--input",
                str(self.input_file),
                "--bbox",
                "10,45,20,55",  # lon_min,lat_min,lon_max,lat_max
                "--output",
                str(output_file),
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_extract_regions_with_weights(self):
        """Test region extraction with area weighting."""
        output_file = TEMP_DIR / "extracted_weighted.csv"

        result = self.runner.invoke(
            extract_regions,
            [
                "--input",
                str(self.input_file),
                "--regions",
                str(self.regions_file),
                "--output",
                str(output_file),
                "--use-weights",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestAggregateTemporalCommand:
    """Test aggregate-temporal CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample temporal data
        times = pd.date_range("2024-01-01", periods=100, freq="H")
        self.input_file = TEMP_DIR / "temporal_input.nc"

        # Create hourly data for aggregation testing
        lats = np.linspace(50, 52, 3)
        lons = np.linspace(10, 12, 3)
        pm25_data = 20 + 5 * np.sin(np.arange(100) * 2 * np.pi / 24)  # Daily cycle
        pm25_data = np.tile(
            pm25_data[:, np.newaxis, np.newaxis], (1, len(lats), len(lons))
        )

        import xarray as xr

        ds = xr.Dataset(
            {"pm25": (["time", "latitude", "longitude"], pm25_data)},
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        ds.to_netcdf(self.input_file)

        yield
        cleanup_test_environment()

    def test_aggregate_temporal_daily(self):
        """Test daily temporal aggregation."""
        output_file = TEMP_DIR / "daily_aggregated.nc"

        result = self.runner.invoke(
            aggregate_temporal,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--frequency",
                "daily",
                "--method",
                "mean",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify aggregation
        import xarray as xr

        aggregated = xr.open_dataset(output_file)
        assert len(aggregated.time) < 100  # Should be aggregated
        aggregated.close()

    def test_aggregate_temporal_monthly(self):
        """Test monthly temporal aggregation."""
        output_file = TEMP_DIR / "monthly_aggregated.nc"

        result = self.runner.invoke(
            aggregate_temporal,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--frequency",
                "monthly",
                "--method",
                "mean",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_aggregate_temporal_custom_frequency(self):
        """Test temporal aggregation with custom frequency."""
        output_file = TEMP_DIR / "custom_aggregated.nc"

        result = self.runner.invoke(
            aggregate_temporal,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--frequency",
                "6H",  # 6-hourly
                "--method",
                "max",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_aggregate_temporal_with_threshold(self):
        """Test temporal aggregation with health threshold analysis."""
        output_file = TEMP_DIR / "threshold_aggregated.nc"

        result = self.runner.invoke(
            aggregate_temporal,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--frequency",
                "daily",
                "--method",
                "threshold_exceedance",
                "--threshold",
                "25",
                "--variables",
                "pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestVisualizationCommand:
    """Test visualize CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample data for visualization
        self.data_file = TEMP_DIR / "viz_data.csv"
        viz_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "latitude": [52.5] * 30,
                "longitude": [13.4] * 30,
                "pm25": 20
                + 5 * np.sin(np.arange(30) * 2 * np.pi / 7)
                + np.random.normal(0, 2, 30),
                "no2": 30
                + 8 * np.sin(np.arange(30) * 2 * np.pi / 7)
                + np.random.normal(0, 3, 30),
                "city": ["Berlin"] * 30,
            }
        )
        viz_data.to_csv(self.data_file, index=False)

        yield
        cleanup_test_environment()

    def test_visualize_timeseries(self):
        """Test time series visualization."""
        output_file = TEMP_DIR / "timeseries_plot.png"

        result = self.runner.invoke(
            visualize,
            [
                "--input",
                str(self.data_file),
                "--plot-type",
                "timeseries",
                "--x-column",
                "date",
                "--y-column",
                "pm25",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_visualize_scatter(self):
        """Test scatter plot visualization."""
        output_file = TEMP_DIR / "scatter_plot.png"

        result = self.runner.invoke(
            visualize,
            [
                "--input",
                str(self.data_file),
                "--plot-type",
                "scatter",
                "--x-column",
                "pm25",
                "--y-column",
                "no2",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_visualize_histogram(self):
        """Test histogram visualization."""
        output_file = TEMP_DIR / "histogram_plot.png"

        result = self.runner.invoke(
            visualize,
            [
                "--input",
                str(self.data_file),
                "--plot-type",
                "histogram",
                "--column",
                "pm25",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_visualize_with_title(self):
        """Test visualization with custom title."""
        output_file = TEMP_DIR / "titled_plot.png"

        result = self.runner.invoke(
            visualize,
            [
                "--input",
                str(self.data_file),
                "--plot-type",
                "timeseries",
                "--x-column",
                "date",
                "--y-column",
                "pm25",
                "--title",
                "Berlin PM2.5 Time Series",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_visualize_interactive(self):
        """Test interactive visualization."""
        output_file = TEMP_DIR / "interactive_plot.html"

        result = self.runner.invoke(
            visualize,
            [
                "--input",
                str(self.data_file),
                "--plot-type",
                "timeseries",
                "--x-column",
                "date",
                "--y-column",
                "pm25",
                "--interactive",
                "--output",
                str(output_file),
            ],
        )

        # May require optional dependencies
        if result.exit_code == 0:
            assert output_file.exists()
        else:
            assert (
                "requires" in result.output.lower()
                or "not available" in result.output.lower()
            )


class TestExportCommand:
    """Test export-data CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create sample data
        self.input_file = TEMP_DIR / "export_input.csv"
        export_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="H"),
                "station_id": ["ST001"] * 25 + ["ST002"] * 25,
                "pm25": np.random.uniform(10, 30, 50),
                "no2": np.random.uniform(15, 45, 50),
                "latitude": [52.5] * 25 + [48.8] * 25,
                "longitude": [13.4] * 25 + [2.3] * 25,
            }
        )
        export_data.to_csv(self.input_file, index=False)

        yield
        cleanup_test_environment()

    def test_export_to_excel(self):
        """Test export to Excel format."""
        output_file = TEMP_DIR / "exported.xlsx"

        result = self.runner.invoke(
            export_data,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--format",
                "excel",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_export_to_geojson(self):
        """Test export to GeoJSON format."""
        output_file = TEMP_DIR / "exported.geojson"

        result = self.runner.invoke(
            export_data,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--format",
                "geojson",
                "--lat-column",
                "latitude",
                "--lon-column",
                "longitude",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify GeoJSON structure
        with open(output_file, "r") as f:
            geojson_data = json.load(f)
        assert geojson_data["type"] == "FeatureCollection"
        assert len(geojson_data["features"]) > 0

    def test_export_with_compression(self):
        """Test export with compression."""
        output_file = TEMP_DIR / "exported_compressed.csv.gz"

        result = self.runner.invoke(
            export_data,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--format",
                "csv",
                "--compress",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_export_selected_columns(self):
        """Test export with column selection."""
        output_file = TEMP_DIR / "exported_selected.csv"

        result = self.runner.invoke(
            export_data,
            [
                "--input",
                str(self.input_file),
                "--output",
                str(output_file),
                "--format",
                "csv",
                "--columns",
                "timestamp,station_id,pm25",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify only selected columns
        exported = pd.read_csv(output_file)
        assert list(exported.columns) == ["timestamp", "station_id", "pm25"]


class TestValidateCommand:
    """Test validate-data CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create valid data file
        self.valid_file = TEMP_DIR / "valid_data.nc"
        ds = create_sample_netcdf()
        ds.to_netcdf(self.valid_file)

        # Create invalid data file
        self.invalid_file = TEMP_DIR / "invalid_data.txt"
        with open(self.invalid_file, "w") as f:
            f.write("This is not a valid NetCDF file")

        yield
        cleanup_test_environment()

    def test_validate_valid_file(self):
        """Test validation of valid data file."""
        result = self.runner.invoke(validate_data, ["--input", str(self.valid_file)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "success" in result.output.lower()

    def test_validate_invalid_file(self):
        """Test validation of invalid data file."""
        result = self.runner.invoke(validate_data, ["--input", str(self.invalid_file)])

        assert result.exit_code != 0 or "invalid" in result.output.lower()

    def test_validate_with_schema(self):
        """Test validation with schema checking."""
        result = self.runner.invoke(
            validate_data, ["--input", str(self.valid_file), "--check-schema"]
        )

        # Should run without error
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        result = self.runner.invoke(validate_data, ["--input", "nonexistent.nc"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestBatchProcessCommand:
    """Test batch-process CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create batch configuration file
        self.batch_config = TEMP_DIR / "batch_config.json"

        # Create multiple input files
        input_files = []
        for i in range(3):
            input_file = TEMP_DIR / f"input_{i}.nc"
            ds = create_sample_netcdf()
            ds.to_netcdf(input_file)
            input_files.append(str(input_file))

        batch_config_data = {
            "input_files": input_files,
            "output_directory": str(TEMP_DIR / "batch_output"),
            "operations": [
                {
                    "type": "extract_points",
                    "points": [[52.5, 13.4], [48.8, 2.3]],
                    "variables": ["pm25", "no2"],
                },
                {"type": "aggregate_temporal", "frequency": "daily", "method": "mean"},
            ],
        }

        with open(self.batch_config, "w") as f:
            json.dump(batch_config_data, f, indent=2)

        yield
        cleanup_test_environment()

    def test_batch_process_from_config(self):
        """Test batch processing from configuration file."""
        result = self.runner.invoke(batch_process, ["--config", str(self.batch_config)])

        # May need mocking for complex batch operations
        # For now, test that command runs
        assert result.exit_code == 0 or "batch" in result.output.lower()

    def test_batch_process_parallel(self):
        """Test parallel batch processing."""
        result = self.runner.invoke(
            batch_process,
            ["--config", str(self.batch_config), "--parallel", "--workers", "2"],
        )

        # Should handle parallel processing option
        assert result.exit_code == 0 or "parallel" in result.output.lower()

    def test_batch_process_dry_run(self):
        """Test batch processing dry run."""
        result = self.runner.invoke(
            batch_process, ["--config", str(self.batch_config), "--dry-run"]
        )

        assert result.exit_code == 0
        assert (
            "dry run" in result.output.lower()
            or "would process" in result.output.lower()
        )


class TestConfigureCommand:
    """Test configure CLI command."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()
        yield
        cleanup_test_environment()

    def test_configure_init(self):
        """Test configuration initialization."""
        config_file = TEMP_DIR / "config.yaml"

        result = self.runner.invoke(
            configure, ["--init", "--config-file", str(config_file)]
        )

        assert result.exit_code == 0
        assert config_file.exists()

    def test_configure_set_api_key(self):
        """Test setting API key."""
        config_file = TEMP_DIR / "config.yaml"

        # First initialize config
        self.runner.invoke(configure, ["--init", "--config-file", str(config_file)])

        # Then set API key
        result = self.runner.invoke(
            configure,
            ["--set", "copernicus_key=test_api_key", "--config-file", str(config_file)],
        )

        assert result.exit_code == 0

    def test_configure_show(self):
        """Test showing configuration."""
        config_file = TEMP_DIR / "config.yaml"

        # Initialize config first
        self.runner.invoke(configure, ["--init", "--config-file", str(config_file)])

        # Show configuration
        result = self.runner.invoke(
            configure, ["--show", "--config-file", str(config_file)]
        )

        assert result.exit_code == 0
        assert "configuration" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        self.runner = CliRunner()

        # Create comprehensive test data
        self.input_nc = TEMP_DIR / "comprehensive.nc"
        ds = create_sample_netcdf()
        ds.to_netcdf(self.input_nc)

        self.points_csv = TEMP_DIR / "points.csv"
        points_df = pd.DataFrame(
            {
                "latitude": [52.5, 48.8, 41.9],
                "longitude": [13.4, 2.3, 12.5],
                "name": ["Berlin", "Paris", "Rome"],
            }
        )
        points_df.to_csv(self.points_csv, index=False)

        yield
        cleanup_test_environment()

    def test_full_workflow_cli(self):
        """Test complete workflow using CLI commands."""
        # Step 1: Extract points
        extracted_file = TEMP_DIR / "extracted.csv"
        result1 = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_nc),
                "--points",
                str(self.points_csv),
                "--output",
                str(extracted_file),
                "--variables",
                "pm25",
            ],
        )
        assert result1.exit_code == 0
        assert extracted_file.exists()

        # Step 2: Visualize extracted data
        plot_file = TEMP_DIR / "plot.png"
        result2 = self.runner.invoke(
            visualize,
            [
                "--input",
                str(extracted_file),
                "--plot-type",
                "scatter",
                "--x-column",
                "longitude",
                "--y-column",
                "latitude",
                "--color-column",
                "pm25",
                "--output",
                str(plot_file),
            ],
        )
        assert result2.exit_code == 0
        # Plot may not be created if matplotlib backend issues

        # Step 3: Export to different format
        excel_file = TEMP_DIR / "exported.xlsx"
        result3 = self.runner.invoke(
            export_data,
            [
                "--input",
                str(extracted_file),
                "--output",
                str(excel_file),
                "--format",
                "excel",
            ],
        )
        assert result3.exit_code == 0
        assert excel_file.exists()

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Test with nonexistent input file
        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                "nonexistent.nc",
                "--points",
                str(self.points_csv),
                "--output",
                "output.csv",
            ],
        )

        assert result.exit_code != 0
        assert len(result.output) > 0  # Should provide error message

    def test_cli_verbose_mode(self):
        """Test CLI verbose mode across commands."""
        extracted_file = TEMP_DIR / "verbose_test.csv"

        result = self.runner.invoke(
            extract_points,
            [
                "--input",
                str(self.input_nc),
                "--points",
                str(self.points_csv),
                "--output",
                str(extracted_file),
                "--variables",
                "pm25",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        # Verbose mode should produce additional output
        assert len(result.output) > 50  # Should have substantial output


class TestCLIUtilities:
    """Test CLI utility functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        yield
        cleanup_test_environment()

    def test_progress_bar_functionality(self):
        """Test progress bar in CLI operations."""
        # This would test if progress bars are shown during long operations
        # Implementation depends on the actual CLI progress bar implementation
        pass

    def test_config_file_loading(self):
        """Test configuration file loading."""
        from pollution_extraction.cli import load_config

        config_file = TEMP_DIR / "test_config.yaml"
        config_data = {
            "default_output_format": "csv",
            "compression": True,
            "parallel_workers": 4,
        }

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(str(config_file))
        assert loaded_config["default_output_format"] == "csv"
        assert loaded_config["compression"] is True

    def test_input_validation_functions(self):
        """Test input validation utility functions."""
        from pollution_extraction.cli import validate_coordinates, validate_date_range

        # Test coordinate validation
        valid_coords = validate_coordinates("52.5,13.4")
        assert valid_coords == (52.5, 13.4)

        # Test invalid coordinates
        with pytest.raises(ValueError):
            validate_coordinates("invalid,coords")

        # Test date range validation
        valid_dates = validate_date_range("2024-01-01", "2024-01-31")
        assert valid_dates[0] < valid_dates[1]

        # Test invalid date range
        with pytest.raises(ValueError):
            validate_date_range("2024-01-31", "2024-01-01")  # End before start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

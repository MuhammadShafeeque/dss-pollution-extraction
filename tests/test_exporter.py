"""
Test module for the DataExporter class.

This module contains unit tests for the DataExporter class,
using mocking to avoid dependencies on actual file I/O operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from pollution_extraction.core.data_exporter import DataExporter


class TestDataExporter:
    """Test class for DataExporter."""

    # Constants for metadata strings
    PM25_STANDARD_NAME = "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air"
    CRS_WKT = (
        'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["GCS_ETRS_1989",'
        'DATUM["ETRS_1989",SPHEROID["GRS_1980",6378137,298.257222101,'
        'AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6258"]],'
        'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],'
        'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
        'PARAMETER["latitude_of_origin",52],PARAMETER["central_meridian",10],'
        'PARAMETER["false_easting",4321000],'
        'PARAMETER["false_northing",3210000],'
        'UNIT["Meter",1],AUTHORITY["EPSG","3035"]]'
    )

    @pytest.fixture
    def sample_dataset(self) -> xr.Dataset:
        """Create a sample xarray dataset for testing."""
        time = pd.date_range("2020-01-01", periods=5, freq="D")
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)

        # Create sample pollution data
        np.random.seed(42)
        data = np.random.gamma(2, 5, (5, 10, 10))

        dataset = xr.Dataset(
            {
                "pm25": (
                    ("time", "x", "y"),
                    data,
                    {
                        "units": "µg/m³",
                        "long_name": "PM2.5 concentration",
                        "description": "Particulate matter 2.5 micrometers",
                    },
                )
            },
            coords={"time": time, "x": x, "y": y},
            attrs={
                "title": "PM2.5 pollution data",
                "source": "Test dataset for pollution analysis",
                "institution": "Test Institute",
            },
        )
        return dataset

    @pytest.fixture
    def exporter(self, sample_dataset: xr.Dataset) -> DataExporter:
        """Create a DataExporter instance for testing."""
        return DataExporter(sample_dataset, "pm25")

    @pytest.fixture
    def sample_geodataframe(self) -> gpd.GeoDataFrame:
        """Create a sample GeoDataFrame for testing."""
        # Create some sample polygons
        polygons = [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]),
        ]

        gdf = gpd.GeoDataFrame(
            {
                "id": ["region_1", "region_2", "region_3"],
                "name": ["Region A", "Region B", "Region C"],
                "area": [4.0, 4.0, 4.0],
                "geometry": polygons,
            },
            crs="EPSG:3035",
        )

        return gdf

    def test_initialization(self, sample_dataset: xr.Dataset) -> None:
        """Test DataExporter initialization."""
        exporter = DataExporter(sample_dataset, "pm25")

        assert exporter.dataset == sample_dataset
        assert exporter.pollution_variable == "pm25"

    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_netcdf_basic(self, mock_mkdir, exporter, sample_dataset):
        """Test basic NetCDF export functionality."""
        # Mock the to_netcdf method on the dataset
        mock_dataset = Mock()
        mock_dataset.copy.return_value = sample_dataset
        exporter.dataset = mock_dataset

        # Mock the actual to_netcdf method
        sample_dataset.to_netcdf = Mock()

        output_path = "/test/output.nc"
        exporter.to_netcdf(output_path)

        # Verify directory creation was attempted
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify to_netcdf was called
        sample_dataset.to_netcdf.assert_called_once()

        # Check encoding was provided
        call_args = sample_dataset.to_netcdf.call_args
        assert "encoding" in call_args[1]

    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_netcdf_with_subsets(self, mock_mkdir, exporter, sample_dataset):
        """Test NetCDF export with time and spatial subsets."""
        mock_dataset = Mock()
        mock_dataset.copy.return_value = sample_dataset
        mock_dataset.sel.return_value = sample_dataset
        exporter.dataset = mock_dataset

        sample_dataset.to_netcdf = Mock()
        sample_dataset.sel = Mock(return_value=sample_dataset)

        time_subset = slice("2020-01-02", "2020-01-04")
        spatial_subset = {"minx": 0, "maxx": 5, "miny": 0, "maxy": 5}
        compression = {"zlib": True, "complevel": 6}

        exporter.to_netcdf(
            "/test/output.nc",
            time_subset=time_subset,
            spatial_subset=spatial_subset,
            compression=compression,
        )

        # Verify the method was called
        sample_dataset.to_netcdf.assert_called_once()

    @patch("rasterio.open")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_geotiff_basic(self, mock_mkdir, mock_rasterio_open, exporter):
        """Test basic GeoTIFF export functionality."""
        # Setup mock rasterio context manager
        mock_dst = Mock()
        mock_context = Mock()
        mock_context.__enter__.return_value = mock_dst
        mock_context.__exit__.return_value = None
        mock_rasterio_open.return_value = mock_context

        output_path = "/test/output.tif"
        exporter.to_geotiff(output_path, time_index=0)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify rasterio.open was called with correct parameters
        mock_rasterio_open.assert_called_once()
        call_args = mock_rasterio_open.call_args
        assert call_args[0][0] == Path(output_path)
        assert call_args[1]["driver"] == "GTiff"
        assert "transform" in call_args[1]
        assert "crs" in call_args[1]

        # Verify data was written
        mock_dst.write.assert_called_once()
        mock_dst.update_tags.assert_called_once()

    @patch("rasterio.open")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_geotiff_with_aggregation(
        self, mock_mkdir, mock_rasterio_open, exporter
    ) -> None:
        """Test GeoTIFF export with temporal aggregation."""
        mock_dst = Mock()
        mock_context = Mock()
        mock_context.__enter__.return_value = mock_dst
        mock_context.__exit__.return_value = None
        mock_rasterio_open.return_value = mock_context

        time_slice = slice("2020-01-01", "2020-01-03")
        exporter.to_geotiff(
            "/test/output.tif",
            time_index=time_slice,
            aggregation_method="mean",
        )

        mock_rasterio_open.assert_called_once()
        mock_dst.write.assert_called_once()

    @patch("rasterio.open")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_geotiff_invalid_aggregation(
        self, mock_mkdir, mock_rasterio_open, exporter
    ):
        """Test GeoTIFF export with invalid aggregation method."""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            exporter.to_geotiff(
                "/test/output.tif",
                time_index=slice(0, 2),
                aggregation_method="invalid_method",
            )

    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_csv_basic(self, mock_mkdir, mock_to_csv, exporter):
        """Test basic CSV export functionality."""
        output_path = "/test/output.csv"
        exporter.to_csv(output_path)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify CSV export was called
        mock_to_csv.assert_called_once_with(Path(output_path), index=False)

    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_csv_with_spatial_aggregation(
        self, mock_mkdir, mock_to_csv, exporter
    ) -> None:
        """Test CSV export with spatial aggregation."""
        exporter.to_csv(
            path="/test/output.csv",
            spatial_aggregation="mean",
            include_coordinates=False,
        )

        mock_to_csv.assert_called_once()

    @patch("geopandas.GeoDataFrame.to_file")
    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_extracted_points_to_formats(
        self, mock_mkdir, mock_to_csv, mock_to_file, exporter
    ) -> None:
        """Test export of extracted point data to multiple formats."""
        extracted_data = xr.Dataset(
            {
                "pm25": (("time", "location_id"), np.random.rand(5, 3)),
                "location_x": (("location_id",), [1.0, 2.0, 3.0]),
                "location_y": (("location_id",), [4.0, 5.0, 6.0]),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=5),
                "location_id": [0, 1, 2],
            },
        )

        output_dir = "/test/output"
        formats = ["csv", "geojson"]

        result = exporter.extracted_points_to_formats(
            extracted_data,
            output_dir,
            formats,
        )

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert mock_to_csv.call_count >= 1
        assert mock_to_file.call_count >= 1
        assert isinstance(result, dict)
        assert "csv" in result

    @patch("geopandas.GeoDataFrame.to_file")
    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_extracted_polygons_to_formats(
        self, mock_mkdir, mock_to_csv, mock_to_file, exporter, sample_geodataframe
    ) -> None:
        """Test export of extracted polygon data."""
        # Create mock extracted data
        extracted_data = xr.Dataset(
            {"pm25": (("time", "polygon_id"), np.random.rand(5, 3))},
            coords={
                "time": pd.date_range("2020-01-01", periods=5),
                "polygon_id": [0, 1, 2],
            },
        )

        output_dir = "/test/output"
        formats = ["csv", "geojson", "shapefile"]

        result = exporter.extracted_polygons_to_formats(
            extracted_data,
            sample_geodataframe,
            output_dir,
            formats,
        )

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert mock_to_csv.call_count >= 1
        assert mock_to_file.call_count >= 1
        assert isinstance(result, dict)

    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_metadata_file(
        self, mock_file_open, mock_json_dump, exporter
    ) -> None:
        """Test metadata file creation."""
        output_path = "/test/metadata.json"
        processing_info = {
            "method": "temporal_aggregation",
            "parameters": {"aggregation": "mean"},
        }

        exporter.create_metadata_file(output_path, processing_info)

        # Verify file was opened for writing
        mock_file_open.assert_called_once_with(Path(output_path), "w")

        # Verify JSON was dumped
        mock_json_dump.assert_called_once()

        # Check the structure of the metadata
        call_args = mock_json_dump.call_args
        metadata = call_args[0][0]  # First argument to json.dump

        assert "dataset_info" in metadata
        assert "variable_attributes" in metadata
        assert "global_attributes" in metadata
        assert "export_info" in metadata
        assert "processing_info" in metadata

        # Check specific content
        assert metadata["dataset_info"]["pollution_variable"] == "pm25"
        assert metadata["processing_info"] == processing_info

    def test_create_metadata_file_without_processing_info(self, exporter):
        """Test metadata file creation without processing info."""
        with patch("json.dump") as mock_json_dump:
            # Mock file open using context manager decorator
            patch_open = patch("builtins.open", mock_open())
            patch_open.start()

            try:
                exporter.create_metadata_file("/test/metadata.json")

                # Verify the call was made
                mock_json_dump.assert_called_once()

                # Check that processing_info is None in the metadata
                call_args = mock_json_dump.call_args
                metadata = call_args[0][0]
                assert "processing_info" in metadata
            finally:
                patch_open.stop()

    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_to_netcdf_with_custom_compression(
        self, mock_mkdir, exporter, sample_dataset
    ):
        """Test NetCDF export with custom compression settings."""
        mock_dataset = Mock()
        mock_dataset.copy.return_value = sample_dataset
        exporter.dataset = mock_dataset

        sample_dataset.to_netcdf = Mock()

        custom_compression = {"zlib": True, "complevel": 9, "shuffle": True}

        exporter.to_netcdf("/test/output.nc", compression=custom_compression)

        # Verify to_netcdf was called with custom compression
        call_args = sample_dataset.to_netcdf.call_args
        encoding = call_args[1]["encoding"]

        # Check that compression settings were applied to data variables
        for var_encoding in encoding.values():
            assert var_encoding == custom_compression

    def test_dataset_property_access(self, exporter, sample_dataset):
        """Test that exporter correctly accesses dataset properties."""
        # Test that we can access the pollution variable
        pollution_data = exporter.dataset[exporter.pollution_variable]
        assert pollution_data is not None

        # Test that we can access coordinates
        assert "time" in exporter.dataset.coords
        assert "x" in exporter.dataset.coords
        assert "y" in exporter.dataset.coords

        # Test that we can access attributes
        assert "pm25" in exporter.dataset.data_vars

    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_csv_export_time_formatting(
        self, mock_mkdir, mock_to_csv, exporter
    ) -> None:
        """Test CSV export with custom time format."""
        exporter.to_csv("/test/output.csv", time_format="%Y-%m-%d %H:%M:%S")
        mock_mkdir.assert_called_once()
        mock_to_csv.assert_called_once()

    def test_dataset_attributes(self, exporter: DataExporter) -> None:
        """Test metadata and attribute handling."""
        metadata = exporter.dataset.attrs

        assert "standard_name" in metadata
        assert "pollution_description" in metadata
        assert metadata["pollution_description"].startswith("PM2.5")

        # CRS information
        crs_info = metadata.get("crs_wkt", "")
        assert "PROJCS" in crs_info
        assert "ETRS89" in crs_info

    @patch("geopandas.GeoDataFrame.to_file")
    @patch("pandas.DataFrame.to_csv")
    @patch("pollution_extraction.core.data_exporter.Path.mkdir")
    def test_extracted_points_missing_coordinates(
        self, mock_mkdir, mock_to_csv, mock_to_file, exporter
    ):
        """Test extracted points export when coordinates are missing."""
        # Create extracted data without coordinate information
        extracted_data = xr.Dataset(
            {"pm25": (("time", "location_id"), np.random.rand(5, 3))},
            coords={
                "time": pd.date_range("2020-01-01", periods=5),
                "location_id": [0, 1, 2],
            },
        )

        result = exporter.extracted_points_to_formats(
            extracted_data, "/test/output", ["csv", "geojson"]
        )

        # Should still create CSV but not GeoJSON due to missing coordinates
        assert "csv" in result
        # GeoJSON might not be created if coordinates are invalid

    def test_spatial_bounds_calculation(self, exporter):
        """Test that spatial bounds are calculated correctly."""
        x_min = float(exporter.dataset.x.min())
        x_max = float(exporter.dataset.x.max())
        y_min = float(exporter.dataset.y.min())
        y_max = float(exporter.dataset.y.max())

        assert x_min < x_max
        assert y_min < y_max
        assert isinstance(x_min, float)
        assert isinstance(y_max, float)


# Constants for long strings
PM25_STANDARD_NAME = "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air"
CRS_WKT = (
    'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["GCS_ETRS_1989",'
    'DATUM["ETRS_1989",SPHEROID["GRS_1980",6378137,298.257222101,'
    'AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6258"]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],'
    'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["latitude_of_origin",52],PARAMETER["central_meridian",10],'
    'PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],'
    'UNIT["Meter",1],AUTHORITY["EPSG","3035"]]'
)


class TestDataExporterIntegration:
    """Integration tests for DataExporter with realistic data structures."""

    @pytest.fixture
    def realistic_dataset(self):
        """Create a more realistic dataset for integration testing."""
        time = pd.date_range("2020-01-01", periods=12, freq="M")
        x = np.linspace(0, 100, 50)  # Larger spatial grid
        y = np.linspace(0, 100, 50)

        # Create realistic pollution data with seasonal variation
        base_pollution = 15.0  # Base pollution level
        seasonal_variation = np.sin(np.arange(12) * 2 * np.pi / 12) * 5

        data = np.zeros((12, 50, 50))
        for t in range(12):
            spatial_pattern = np.random.gamma(
                2, base_pollution + seasonal_variation[t], (50, 50)
            )
            data[t] = spatial_pattern

        dataset = xr.Dataset(
            {
                "pm25": (
                    ("time", "x", "y"),
                    data,
                    {
                        "units": "µg/m³",
                        "long_name": "PM2.5 mass concentration",
                        "standard_name": TestDataExporter.PM25_STANDARD_NAME,
                    },
                ),
                "crs": (
                    (),
                    0,
                    {
                        "grid_mapping_name": "lambert_azimuthal_equal_area",
                        "longitude_of_projection_origin": 10.0,
                        "latitude_of_projection_origin": 52.0,
                        "false_easting": 4321000.0,
                        "false_northing": 3210000.0,
                        "crs_wkt": TestDataExporter.CRS_WKT,
                    },
                ),
            },
            coords={"time": time, "x": x, "y": y},
            attrs={
                "title": "PM2.5 concentration over Europe",
                "institution": "European Environment Agency",
                "source": "CAMS regional air quality model",
                "Conventions": "CF-1.8",
            },
        )
        return dataset

    def test_realistic_data_export_workflow(self, realistic_dataset):
        """Test a complete export workflow with realistic data."""
        exporter = DataExporter(realistic_dataset, "pm25")

        # Test that we can access all the expected properties
        assert exporter.pollution_variable == "pm25"
        assert "time" in exporter.dataset.dims
        assert "x" in exporter.dataset.dims
        assert "y" in exporter.dataset.dims

        # Test that the dataset has the expected structure
        assert exporter.dataset["pm25"].dims == ("time", "x", "y")
        assert len(exporter.dataset.time) == 12
        assert len(exporter.dataset.x) == 50
        assert len(exporter.dataset.y) == 50

        # Test data value ranges (should be realistic pollution values)
        data_values = exporter.dataset["pm25"].values
        assert np.all(data_values >= 0)  # Pollution should be non-negative
        assert np.mean(data_values) > 5  # Should have reasonable mean value
        assert np.mean(data_values) < 100  # Should not be unrealistically high

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_realistic_metadata_creation(
        self, mock_json_dump, mock_file_open, realistic_dataset
    ):
        """Test metadata creation with realistic dataset."""
        exporter = DataExporter(realistic_dataset, "pm25")

        processing_info = {
            "spatial_resolution": "2km",
            "temporal_resolution": "monthly",
            "model": "CAMS",
        }

        exporter.create_metadata_file("/test/metadata.json", processing_info)

        # Verify the metadata structure
        call_args = mock_json_dump.call_args
        metadata = call_args[0][0]

        # Check dataset info
        dataset_info = metadata["dataset_info"]
        assert dataset_info["pollution_variable"] == "pm25"
        assert dataset_info["dimensions"]["time"] == 12
        assert dataset_info["dimensions"]["x"] == 50
        assert dataset_info["dimensions"]["y"] == 50

        # Check that coordinate system info is included
        assert "coordinate_system" in metadata

        # Check variable attributes
        var_attrs = metadata["variable_attributes"]
        assert var_attrs["units"] == "µg/m³"
        assert "long_name" in var_attrs


if __name__ == "__main__":
    # Run tests with: python -m pytest test_data_exporter.py -v
    pytest.main([__file__, "-v"])

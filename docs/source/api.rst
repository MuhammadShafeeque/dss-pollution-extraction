API Reference
=============

This section provides detailed documentation for all classes and functions in the DSS Pollution Extraction library.

Main Module
-----------

.. automodule:: pollution_extraction
   :members:
   :undoc-members:
   :show-inheritance:

Core Modules
------------

Analyzer
~~~~~~~~

.. automodule:: pollution_extraction.analyzer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. automodule:: pollution_extraction.config
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.cli
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
~~~~~~~~~

.. automodule:: pollution_extraction.utils
   :members:
   :undoc-members:
   :show-inheritance:

Core Processing Modules
-----------------------

Data Reader
~~~~~~~~~~~

.. automodule:: pollution_extraction.core.data_reader
   :members:
   :undoc-members:
   :show-inheritance:

Spatial Extractor
~~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.spatial_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Temporal Aggregator
~~~~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.temporal_aggregator
   :members:
   :undoc-members:
   :show-inheritance:

Data Exporter
~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.data_exporter
   :members:
   :undoc-members:
   :show-inheritance:

Data Visualizer
~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.data_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Logging Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

Exporter Modules
----------------

Base Exporter
~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.exporters.base
   :members:
   :undoc-members:
   :show-inheritance:

Main Exporter
~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.exporters.main
   :members:
   :undoc-members:
   :show-inheritance:

Spatial Exporter
~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.exporters.spatial
   :members:
   :undoc-members:
   :show-inheritance:

Tabular Exporter
~~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.exporters.tabular
   :members:
   :undoc-members:
   :show-inheritance:

Raster Exporter
~~~~~~~~~~~~~~~

.. automodule:: pollution_extraction.core.exporters.raster
   :members:
   :undoc-members:
   :show-inheritance:

Types
~~~~~

.. automodule:: pollution_extraction.core.exporters.types
   :members:
   :undoc-members:
   :show-inheritance:

Class Index
-----------

Main Classes
~~~~~~~~~~~~

* :class:`pollution_extraction.analyzer.PollutionAnalyzer` - Main analysis class
* :class:`pollution_extraction.core.data_reader.DataReader` - Data loading functionality
* :class:`pollution_extraction.core.spatial_extractor.SpatialExtractor` - Spatial data extraction
* :class:`pollution_extraction.core.temporal_aggregator.TemporalAggregator` - Temporal aggregation
* :class:`pollution_extraction.core.data_exporter.DataExporter` - Data export functionality
* :class:`pollution_extraction.core.data_visualizer.DataVisualizer` - Visualization tools

Export Classes
~~~~~~~~~~~~~~

* :class:`pollution_extraction.core.exporters.base.BaseExporter` - Base exporter class
* :class:`pollution_extraction.core.exporters.main.MainExporter` - Main export coordinator
* :class:`pollution_extraction.core.exporters.spatial.SpatialExporter` - Spatial format exports
* :class:`pollution_extraction.core.exporters.tabular.TabularExporter` - Tabular format exports
* :class:`pollution_extraction.core.exporters.raster.RasterExporter` - Raster format exports

Function Index
--------------

Data Loading Functions
~~~~~~~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.core.data_reader.DataReader.read_netcdf` - Load NetCDF files
* :func:`pollution_extraction.core.data_reader.DataReader.read_csv` - Load CSV files
* :func:`pollution_extraction.core.data_reader.DataReader.read_geotiff` - Load GeoTIFF files

Spatial Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.core.spatial_extractor.SpatialExtractor.extract_bbox` - Extract by bounding box
* :func:`pollution_extraction.core.spatial_extractor.SpatialExtractor.extract_points` - Extract by points
* :func:`pollution_extraction.core.spatial_extractor.SpatialExtractor.extract_polygon` - Extract by polygon

Temporal Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.core.temporal_aggregator.TemporalAggregator.aggregate_monthly` - Monthly aggregation
* :func:`pollution_extraction.core.temporal_aggregator.TemporalAggregator.aggregate_annual` - Annual aggregation
* :func:`pollution_extraction.core.temporal_aggregator.TemporalAggregator.aggregate_seasonal` - Seasonal aggregation

Export Functions
~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.core.data_exporter.DataExporter.to_csv` - Export to CSV
* :func:`pollution_extraction.core.data_exporter.DataExporter.to_geotiff` - Export to GeoTIFF
* :func:`pollution_extraction.core.data_exporter.DataExporter.to_netcdf` - Export to NetCDF
* :func:`pollution_extraction.core.data_exporter.DataExporter.to_shapefile` - Export to Shapefile

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.core.data_visualizer.DataVisualizer.plot_spatial` - Create spatial maps
* :func:`pollution_extraction.core.data_visualizer.DataVisualizer.plot_timeseries` - Create time series plots
* :func:`pollution_extraction.core.data_visualizer.DataVisualizer.plot_distribution` - Create distribution plots

Utility Functions
~~~~~~~~~~~~~~~~~

* :func:`pollution_extraction.utils.validate_config` - Validate configuration
* :func:`pollution_extraction.utils.setup_directories` - Setup output directories
* :func:`pollution_extraction.core.logging_utils.setup_logging` - Configure logging

Error Classes
-------------

Custom Exceptions
~~~~~~~~~~~~~~~~~

The library defines several custom exception classes for better error handling:

.. autoexception:: pollution_extraction.core.exceptions.DataReadError
.. autoexception:: pollution_extraction.core.exceptions.SpatialExtractionError
.. autoexception:: pollution_extraction.core.exceptions.TemporalAggregationError
.. autoexception:: pollution_extraction.core.exceptions.ExportError
.. autoexception:: pollution_extraction.core.exceptions.ConfigurationError

Constants and Types
-------------------

Type Definitions
~~~~~~~~~~~~~~~~

.. autodata:: pollution_extraction.core.exporters.types.ExportFormat
.. autodata:: pollution_extraction.core.exporters.types.AggregationMethod
.. autodata:: pollution_extraction.core.exporters.types.InterpolationMethod

Configuration Constants
~~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: pollution_extraction.config.DEFAULT_CONFIG
.. autodata:: pollution_extraction.config.SUPPORTED_FORMATS
.. autodata:: pollution_extraction.config.TEMPORAL_AGGREGATIONS

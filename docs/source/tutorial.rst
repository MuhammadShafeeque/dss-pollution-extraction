Tutorial
========

This tutorial will guide you through the main features of the DSS Pollution Extraction library.

Getting Started
---------------

First, make sure you have installed the library following the :doc:`installation` guide.

Basic Usage
-----------

Import the main components:

.. code-block:: python

    from pollution_extraction import PollutionAnalyzer
    from pollution_extraction.core.data_reader import DataReader
    from pollution_extraction.core.spatial_extractor import SpatialExtractor

Loading Data
------------

The library supports various data formats including NetCDF files:

.. code-block:: python

    # Initialize the data reader
    reader = DataReader()
    
    # Load NetCDF data
    data = reader.read_netcdf("path/to/your/data.nc")
    
    # Display basic information about the dataset
    print(data.info())

Spatial Extraction
------------------

Extract pollution data for specific geographic regions:

.. code-block:: python

    from pollution_extraction.core.spatial_extractor import SpatialExtractor
    
    # Initialize the spatial extractor
    extractor = SpatialExtractor()
    
    # Extract data for a bounding box
    bbox_data = extractor.extract_bbox(
        data=data,
        bbox=[lon_min, lat_min, lon_max, lat_max]
    )
    
    # Extract data for specific points
    points_data = extractor.extract_points(
        data=data,
        points=[(lon1, lat1), (lon2, lat2)]
    )

Temporal Aggregation
--------------------

Aggregate data over different time periods:

.. code-block:: python

    from pollution_extraction.core.temporal_aggregator import TemporalAggregator
    
    # Initialize the temporal aggregator
    aggregator = TemporalAggregator()
    
    # Calculate monthly means
    monthly_data = aggregator.aggregate_monthly(data)
    
    # Calculate annual statistics
    annual_stats = aggregator.aggregate_annual(data, stats=['mean', 'max', 'min'])

Data Export
-----------

Export your processed data in various formats:

.. code-block:: python

    from pollution_extraction.core.data_exporter import DataExporter
    
    # Initialize the exporter
    exporter = DataExporter()
    
    # Export to CSV
    exporter.to_csv(data, "output/pollution_data.csv")
    
    # Export to GeoTIFF
    exporter.to_geotiff(data, "output/pollution_map.tif")
    
    # Export to NetCDF
    exporter.to_netcdf(data, "output/processed_data.nc")

Advanced Analysis
-----------------

Use the main analyzer for comprehensive pollution analysis:

.. code-block:: python

    # Initialize the analyzer
    analyzer = PollutionAnalyzer(config_path="config/user_config.yaml")
    
    # Run complete analysis workflow
    results = analyzer.analyze_pollution(
        input_file="data/pm25_data.nc",
        regions_file="data/study_regions.geojson",
        output_dir="output/"
    )
    
    # Generate visualizations
    analyzer.create_visualizations(results, output_dir="output/plots/")

Working with Configuration
--------------------------

Customize analysis parameters using configuration files:

.. code-block:: yaml

    # config/analysis_config.yaml
    data:
      variable_name: "pm25"
      time_range: ["2020-01-01", "2020-12-31"]
    
    processing:
      temporal_aggregation: "monthly"
      spatial_resolution: 0.1
    
    output:
      formats: ["csv", "geotiff", "netcdf"]
      create_plots: true

Error Handling
--------------

The library provides comprehensive error handling and logging:

.. code-block:: python

    import logging
    from pollution_extraction.core.logging_utils import setup_logging
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    try:
        # Your analysis code here
        results = analyzer.analyze_pollution(...)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
    except ValueError as e:
        logging.error(f"Invalid parameter: {e}")

Next Steps
----------

- Check out the :doc:`examples` for more detailed use cases
- Refer to the :doc:`api` documentation for complete function references
- Visit the project's GitHub repository for the latest updates and examples

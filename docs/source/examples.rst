Examples
========

This section provides practical examples of using the DSS Pollution Extraction library for various pollution data analysis tasks.

Basic Data Loading and Exploration
-----------------------------------

Load and explore PM2.5 pollution data:

.. code-block:: python

    from pollution_extraction import PollutionAnalyzer
    from pollution_extraction.core.data_reader import DataReader
    import matplotlib.pyplot as plt
    
    # Load sample data
    reader = DataReader()
    data = reader.read_netcdf("examples/data/sample_pm25.nc")
    
    # Display basic information
    print(f"Data shape: {data.shape}")
    print(f"Time range: {data.time.min().values} to {data.time.max().values}")
    print(f"Spatial extent: {data.lon.min().values:.2f} to {data.lon.max().values:.2f} longitude")
    print(f"                {data.lat.min().values:.2f} to {data.lat.max().values:.2f} latitude")

Point-Based Extraction
----------------------

Extract pollution data for specific monitoring stations:

.. code-block:: python

    from pollution_extraction.core.spatial_extractor import SpatialExtractor
    import pandas as pd
    
    # Define monitoring station locations
    stations = [
        {"name": "Station_A", "lon": 7.0, "lat": 51.5},
        {"name": "Station_B", "lon": 8.5, "lat": 52.2},
        {"name": "Station_C", "lon": 9.2, "lat": 53.1}
    ]
    
    # Extract data for each station
    extractor = SpatialExtractor()
    results = []
    
    for station in stations:
        station_data = extractor.extract_points(
            data=data,
            points=[(station["lon"], station["lat"])]
        )
        
        # Convert to time series
        ts = station_data.to_dataframe().reset_index()
        ts["station"] = station["name"]
        results.append(ts)
    
    # Combine all station data
    all_stations = pd.concat(results, ignore_index=True)
    print(all_stations.head())

Regional Analysis
-----------------

Analyze pollution levels across administrative regions:

.. code-block:: python

    import geopandas as gpd
    from pollution_extraction.core.spatial_extractor import SpatialExtractor
    from pollution_extraction.core.temporal_aggregator import TemporalAggregator
    
    # Load regional boundaries
    regions = gpd.read_file("examples/data/sample_regions.geojson")
    
    # Initialize extractors
    spatial_extractor = SpatialExtractor()
    temporal_aggregator = TemporalAggregator()
    
    regional_stats = []
    
    for idx, region in regions.iterrows():
        # Extract data for region
        region_data = spatial_extractor.extract_polygon(
            data=data,
            polygon=region.geometry
        )
        
        # Calculate temporal statistics
        stats = temporal_aggregator.aggregate_annual(
            region_data,
            stats=['mean', 'max', 'min', 'std']
        )
        
        # Store results
        regional_stats.append({
            'region_id': region['id'],
            'region_name': region['name'],
            'annual_mean': float(stats.mean()),
            'annual_max': float(stats.max()),
            'annual_min': float(stats.min()),
            'annual_std': float(stats.std())
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(regional_stats)
    print(results_df)

Time Series Analysis
--------------------

Analyze temporal patterns in pollution data:

.. code-block:: python

    from pollution_extraction.core.temporal_aggregator import TemporalAggregator
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract data for a specific region
    bbox = [6.0, 50.0, 10.0, 54.0]  # Germany bounding box
    region_data = extractor.extract_bbox(data, bbox)
    
    # Calculate temporal aggregations
    aggregator = TemporalAggregator()
    
    # Monthly means
    monthly_data = aggregator.aggregate_monthly(region_data)
    
    # Daily data for seasonal analysis
    daily_data = region_data.resample(time='D').mean()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Monthly time series
    monthly_data.plot(ax=axes[0, 0])
    axes[0, 0].set_title('Monthly Average PM2.5 Concentrations')
    axes[0, 0].set_ylabel('PM2.5 (μg/m³)')
    
    # Seasonal patterns
    daily_df = daily_data.to_dataframe().reset_index()
    daily_df['month'] = daily_df['time'].dt.month
    sns.boxplot(data=daily_df, x='month', y='pm25', ax=axes[0, 1])
    axes[0, 1].set_title('Seasonal Variation')
    
    # Annual trends
    yearly_data = aggregator.aggregate_annual(region_data)
    yearly_data.plot(ax=axes[1, 0])
    axes[1, 0].set_title('Annual Trends')
    
    # Distribution
    daily_df['pm25'].hist(bins=50, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Daily Values')
    axes[1, 1].set_xlabel('PM2.5 (μg/m³)')
    
    plt.tight_layout()
    plt.show()

Batch Processing
----------------

Process multiple files efficiently:

.. code-block:: python

    import os
    from pathlib import Path
    from pollution_extraction import PollutionAnalyzer
    
    # Setup batch processing
    input_dir = Path("data/raw/")
    output_dir = Path("data/processed/")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = PollutionAnalyzer(config_path="config/batch_config.yaml")
    
    # Process all NetCDF files in input directory
    netcdf_files = list(input_dir.glob("*.nc"))
    
    for file_path in netcdf_files:
        print(f"Processing {file_path.name}...")
        
        try:
            # Process file
            results = analyzer.analyze_pollution(
                input_file=str(file_path),
                output_dir=str(output_dir / file_path.stem)
            )
            
            print(f"Successfully processed {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

Advanced Visualization
----------------------

Create publication-ready maps and plots:

.. code-block:: python

    from pollution_extraction.core.data_visualizer import DataVisualizer
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Initialize visualizer
    visualizer = DataVisualizer()
    
    # Create spatial maps
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    
    # Plot annual mean
    annual_mean = data.mean(dim='time')
    
    im = visualizer.plot_spatial(
        data=annual_mean,
        ax=ax,
        title='Annual Mean PM2.5 Concentrations',
        cmap='YlOrRd',
        levels=20
    )
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.RIVERS)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='PM2.5 (μg/m³)')
    
    plt.show()

Export and Integration
----------------------

Export results for use in other applications:

.. code-block:: python

    from pollution_extraction.core.data_exporter import DataExporter
    
    # Initialize exporter
    exporter = DataExporter()
    
    # Export processed data in multiple formats
    base_filename = "output/processed_pollution"
    
    # CSV for spreadsheet applications
    exporter.to_csv(results_df, f"{base_filename}.csv")
    
    # GeoTIFF for GIS applications
    exporter.to_geotiff(annual_mean, f"{base_filename}.tif")
    
    # NetCDF for scientific applications
    exporter.to_netcdf(monthly_data, f"{base_filename}_monthly.nc")
    
    # JSON for web applications
    summary_stats = {
        'mean_concentration': float(annual_mean.mean()),
        'max_concentration': float(annual_mean.max()),
        'min_concentration': float(annual_mean.min()),
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open(f"{base_filename}_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

Jupyter Notebook Examples
-------------------------

The ``examples/notebooks/`` directory contains interactive Jupyter notebooks demonstrating:

* **data_extraction_analysis.ipynb**: Complete workflow from data loading to export
* **temporal_pattern_analysis.ipynb**: Detailed time series analysis techniques
* **advanced_spatial_analysis.ipynb**: Spatial statistics and kriging interpolation

To run the notebooks:

.. code-block:: bash

    cd examples/notebooks/
    jupyter notebook

Script Examples
---------------

Ready-to-use scripts are available in ``examples/scripts/``:

* **basic_workflow.py**: Simple extraction and export workflow
* **batch_processing.py**: Process multiple files in parallel
* **nuts3_analysis.py**: Analysis for European NUTS3 regions

Run scripts from the project root:

.. code-block:: bash

    python examples/scripts/basic_workflow.py --input data/sample.nc --output results/

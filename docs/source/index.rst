DSS Pollution Extraction Documentation
=====================================

Welcome to the DSS Pollution Extraction package documentation. This comprehensive Python package provides tools for analyzing atmospheric pollution data from NetCDF files, developed at the Alfred Wegener Institute (AWI).

Overview
--------

The DSS Pollution Extraction package offers:

- **Multi-pollutant analysis**: Support for Black Carbon (BC), NO₂, PM₂.₅, and PM₁₀
- **Temporal analysis**: Monthly, seasonal, annual, and custom period aggregations
- **Spatial analysis**: Point-based, polygon-based, and NUTS3 region extractions
- **Visualization**: Publication-ready maps, time series, and statistical plots
- **Data export**: Multiple formats including NetCDF, GeoTIFF, CSV, and GeoJSON
- **Health analysis**: WHO and EU air quality guideline assessments

Quick Start
-----------

.. code-block:: python

    from pollution_extraction import PollutionAnalyzer
    
    # Initialize analyzer
    analyzer = PollutionAnalyzer("data.nc", pollution_type="pm25")
    
    # Basic analysis
    analyzer.print_summary()
    analyzer.plot_map(time_index=0, save_path="map.png")
    
    # Export results
    analyzer.export_to_geotiff("output.tif")

Key Features
------------

Data Analysis
~~~~~~~~~~~~~

- Comprehensive temporal aggregations (monthly, seasonal, annual)
- Statistical analysis with quality control
- Multi-file batch processing capabilities
- Memory-efficient handling of large datasets

Visualization
~~~~~~~~~~~~~

- Spatial maps with Cartopy integration
- Time series plots and seasonal cycle analysis
- Statistical distribution plots
- Customizable colormaps and styling

Spatial Operations
~~~~~~~~~~~~~~~~~~

- Point-based data extraction
- Polygon-based regional analysis
- NUTS3 administrative region support
- Custom shapefile processing

Health Assessment
~~~~~~~~~~~~~~~~~

- WHO air quality guideline analysis
- EU regulatory compliance checking
- Exceedance mapping and statistics
- Public health impact assessment tools

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   tutorial
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Support
-------

- **Documentation**: https://dss-pollution-extraction.readthedocs.io/
- **Source Code**: https://github.com/MuhammadShafeeque/dss-pollution-extraction
- **Issues**: https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues
- **Contact**: muhammad.shafeeque@awi.de

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

    @software{shafeeque2024dss,
      title={DSS Pollution Extraction: A Python Package for Atmospheric Pollution Data Analysis},
      author={Shafeeque, Muhammad},
      year={2024},
      institution={Alfred Wegener Institute},
      url={https://github.com/MuhammadShafeeque/dss-pollution-extraction}
    }

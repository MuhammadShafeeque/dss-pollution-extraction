Installation Guide
==================

This guide provides detailed installation instructions for the DSS Pollution Extraction package.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 1GB free space

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Core dependencies that will be automatically installed:

- xarray >= 2022.3.0 (NetCDF data handling)
- pandas >= 1.4.0 (Data manipulation)
- numpy >= 1.21.0 (Numerical operations)
- geopandas >= 0.10.0 (Spatial data)
- rioxarray >= 0.11.0 (Raster I/O)
- matplotlib >= 3.5.0 (Plotting)
- seaborn >= 0.11.0 (Statistical visualization)
- cartopy >= 0.20.0 (Geographic projections)

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install dss-pollution-extraction

From Source
~~~~~~~~~~~

For development or the latest features:

.. code-block:: bash

    git clone https://github.com/MuhammadShafeeque/dss-pollution-extraction.git
    cd dss-pollution-extraction
    pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

    git clone https://github.com/MuhammadShafeeque/dss-pollution-extraction.git
    cd dss-pollution-extraction
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install with development dependencies
    pip install -e ".[dev,docs,jupyter]"
    
    # Install pre-commit hooks
    pre-commit install

Verification
------------

To verify your installation:

.. code-block:: python

    import pollution_extraction
    print(pollution_extraction.__version__)
    
    # Test basic functionality
    from pollution_extraction import PollutionAnalyzer
    print("Installation successful!")

Optional Dependencies
---------------------

For enhanced functionality, install optional packages:

.. code-block:: bash

    # For interactive notebooks
    pip install "dss-pollution-extraction[jupyter]"
    
    # For documentation building
    pip install "dss-pollution-extraction[docs]"
    
    # For performance optimization
    pip install "dss-pollution-extraction[performance]"
    
    # All optional dependencies
    pip install "dss-pollution-extraction[all]"

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError with GeoPandas**

If you encounter issues with GeoPandas installation:

.. code-block:: bash

    conda install -c conda-forge geopandas

**Cartopy Installation Issues**

For Cartopy dependencies on Ubuntu/Debian:

.. code-block:: bash

    sudo apt-get install libproj-dev proj-data proj-bin libgeos-dev

On macOS with Homebrew:

.. code-block:: bash

    brew install proj geos

**Memory Issues with Large Files**

For processing large NetCDF files, consider:

- Increasing system memory
- Using chunked processing (built into the package)
- Processing files in smaller temporal/spatial subsets

Getting Help
~~~~~~~~~~~~

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues>`_
2. Create a new issue with your system details and error messages
3. Contact the development team at muhammad.shafeeque@awi.de

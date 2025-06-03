from pollution_extraction.core.data_reader import PollutionDataReader
from pollution_extraction.core.temporal_aggregator import TemporalAggregator
from pollution_extraction.core.spatial_extractor import SpatialExtractor
from pollution_extraction.core.data_visualizer import DataVisualizer
from pollution_extraction.core.data_exporter import DataExporter
import matplotlib.pyplot as plt

# Path to your NetCDF file
file_path = "Z:/PM2p5_daily_1km_EU/PM2p5_downscaled_daily_lr_2006_01.nc"

# Initialize the reader (auto-detects pollution type if not specified)
reader = PollutionDataReader(
    file_path, pollution_type="pm25")

# Access the main data variable (xarray DataArray)
data = reader.data_variable
print("\nData variable shape:", data.shape)

# Print basic dataset info
info = reader.get_basic_info()
print("\nBasic Info:")
for k, v in info.items():
    print(f"  {k}: {v}")

# Print time range and spatial bounds
print("\nTime range:", reader.time_range)
print("Spatial bounds:", reader.spatial_bounds)

# Select a time range (first 7 days)
subset = data.isel(time=slice(0, 7))
print("\nSubset shape (first 7 days):", subset.shape)

# Plot and show stats for the first time slice (clip to min=0, use xarray plotting, control vmin/vmax)
first_slice = data.isel(time=0).clip(min=0)
print("\nFirst slice stats:")
print("  min:", float(first_slice.min().values))
print("  max:", float(first_slice.max().values))
print("  mean:", float(first_slice.mean().values))

try:
    first_slice.plot.imshow(vmin=0, vmax=40, cmap="Reds", origin='upper')
    plt.title("First Time Slice (time=0)")
    plt.show()
except Exception:
    plt.imshow(first_slice.values, origin='upper', vmin=0, vmax=40, cmap="Reds")
    plt.title("First Time Slice (time=0) [imshow fallback]")
    plt.colorbar()
    plt.show()

dataset = reader.dataset  # Ensure dataset is not None
if dataset is None:
    raise RuntimeError("Dataset could not be loaded by PollutionDataReader.")
var_name = reader.variable_info["var_name"]

# --- Temporal Aggregation: Compute monthly average (for this one-month file, it's the mean over time) ---
time_avg = dataset[var_name].mean(dim="time").clip(min=0)
print("\nTime-averaged (monthly mean) shape:", time_avg.shape)

# Plot the time-averaged map using xarray (control vmin/vmax, correct orientation)
time_avg.plot.imshow(vmin=0, vmax=40, cmap="RdYlBu_r", origin='upper')
plt.title("Monthly Mean (Time-Averaged) PM2.5")
plt.show()

# --- Spatial Extraction: Extract value at a specific point (center of domain, use nearest method and correct order) ---
# Note: xarray expects (y, x) order for imshow, and coordinates must match the dataset's coordinate system and range
spatial_ext = SpatialExtractor(dataset, var_name)
x_center = float(reader.spatial_bounds["x_min"] + (reader.spatial_bounds["x_max"] - reader.spatial_bounds["x_min"]) / 2)
y_center = float(reader.spatial_bounds["y_min"] + (reader.spatial_bounds["y_max"] - reader.spatial_bounds["y_min"]) / 2)
try:
    point_result = spatial_ext.extract_points([(x_center, y_center)], method="nearest")
    print(f"\nExtracted value at domain center (x={x_center:.1f}, y={y_center:.1f}):\n", point_result)
except KeyError as e:
    print(f"\n[SpatialExtractor] Extraction failed: {e}\nCheck if the coordinates are within the valid range and match the dataset's CRS.")

# --- Data Export: Export the time-averaged map to GeoTIFF (in-memory example, not saved) ---
exporter = DataExporter(dataset, var_name)
# NOTE: Commenting out the actual export to avoid dask computation hanging on large datasets
# exporter.to_geotiff("Z:/monthly_mean_pm25.tif", time_index=slice(None), aggregation_method="mean")
print("\n[DataExporter] Example: exporter.to_geotiff() can export data (commented out to avoid dask computation hang).")

# --- Visualization: Use DataVisualizer for a custom plot ---
visualizer = DataVisualizer(dataset, var_name, reader.pollution_type)
fig = visualizer.plot_spatial_map(time_index=0, vmin=0, vmax=40, title="PM2.5 Day 1 (Visualizer)")
plt.show()

# Don't forget to close the dataset when done
reader.close()

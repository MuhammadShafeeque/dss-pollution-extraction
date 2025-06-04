import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("All imports successful!")

# Test numpy
print("\nNumPy version:", np.__version__)
arr = np.array([1, 2, 3])
print("NumPy array:", arr)

# Test pandas
print("\nPandas version:", pd.__version__)
df = pd.DataFrame({"A": [1, 2, 3]})
print("Pandas DataFrame:\n", df)

# Test matplotlib
print("\nMatplotlib version:", plt.__version__)
plt.plot([1, 2, 3], [1, 2, 3])
plt.close()

print("\nAll basic functionality tests passed!")

import numpy as np

# Create array
arr = np.arange(1, 13)

print("Original Array:")
print(arr)
print("Shape:", arr.shape)

# Reshape to 3x4
reshaped = arr.reshape(3,4)

print("\nReshaped Array:")
print(reshaped)
print("Shape:", reshaped.shape)
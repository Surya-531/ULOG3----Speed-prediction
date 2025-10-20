import pandas as pd
import numpy as np

# Generate sample CubeSat speed dataset
np.random.seed(42)
n_samples = 1000

# Random initial and final positions (x, y) in a 2D plane
initial_x = np.random.randint(0, 1000, n_samples)
initial_y = np.random.randint(0, 1000, n_samples)
final_x = initial_x + np.random.randint(1, 100, n_samples)
final_y = initial_y + np.random.randint(1, 100, n_samples)

time = np.random.randint(1, 50, n_samples)  # Time taken in seconds

# Calculate distance using Euclidean formula
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

distance = [calculate_distance(initial_x[i], initial_y[i], final_x[i], final_y[i]) for i in range(n_samples)]

# Generate speed values based on distance and time
speed = [distance[i] / time[i] for i in range(n_samples)]

# Create DataFrame
df = pd.DataFrame({
    "initial_x": initial_x,
    "initial_y": initial_y,
    "final_x": final_x,
    "final_y": final_y,
    "time": time,
    "distance": distance,
    "speed": speed
})

# Save to CSV
df.to_csv("cubesat_speed_data.csv", index=False)
print("Sample dataset saved as cubesat_speed_data.csv")

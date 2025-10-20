import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("cubesat_speed_data.csv")  # Assuming dataset contains initial_x, initial_y, final_x, final_y, time

# Calculating distance using Euclidean formula
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

df["distance"] = df.apply(lambda row: calculate_distance(row["initial_x"], row["initial_y"], row["final_x"], row["final_y"]), axis=1)

# Selecting relevant features
features = ["distance", "time"]
X = df[features]
y = df["speed"]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

# Evaluating model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Example Prediction
def predict_speed(x1, y1, x2, y2, time):
    distance = calculate_distance(x1, y1, x2, y2)
    input_data = np.array([[distance, time]])
    input_scaled = scaler.transform(input_data)
    return model.predict(input_scaled)[0]

# Example Usage:
predicted_speed = predict_speed(871,794,961,824,8)
print(f"Predicted Speed: {predicted_speed} m/s")

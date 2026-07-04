import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("crop_data.csv")

# Convert Soil names to numbers
soil_map = {
    "Loamy": 0,
    "Black": 1,
    "Sandy": 2,
    "Clay": 3
}

data["Soil"] = data["Soil"].map(soil_map)

# Features and Target
X = data[["Temperature", "Rainfall", "Humidity", "Soil", "pH"]]
y = data["Crop"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Save Model
pickle.dump(model, open("crop_model.pkl", "wb"))

print("Model trained successfully!")
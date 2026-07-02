import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

st.title("🌾 Smart Agricultural Production Optimization Engine")

temperature = st.number_input("Temperature (°C)", value=25)
rainfall = st.number_input("Rainfall (mm)", value=200)
humidity = st.number_input("Humidity (%)", value=80)
soil = st.selectbox("Soil Type", ["Loamy", "Black", "Sandy", "Clay"])
ph = st.number_input("Soil pH", value=6.5)

soil_map = {
    "Loamy": 0,
    "Black": 1,
    "Sandy": 2,
    "Clay": 3
}

if st.button("Predict Crop"):
    data = pd.DataFrame([[
        temperature,
        rainfall,
        humidity,
        soil_map[soil],
        ph
    ]], columns=["Temperature","Rainfall","Humidity","Soil","pH"])

    prediction = model.predict(data)
    st.success(f"Recommended Crop: {prediction[0]}")

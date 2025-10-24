import streamlit as st
from src.predict import predict_crop

st.set_page_config(page_title="Crop Recommendation System 🌾", layout="centered")
st.title("Crop Recommendation System 🌾")
st.write("Enter soil and weather parameters to get the best crop recommendation.")

# Input sliders / number fields
n = st.slider("Nitrogen (N)", 0, 140, 50)
p = st.slider("Phosphorus (P)", 5, 145, 54)
k = st.slider("Potassium (K)", 5, 205, 50)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=70.0)
ph = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)

if st.button("Predict Crop"):
    features = {"N": n, "P": p, "K": k, "temperature": temperature,
                "humidity": humidity, "ph": ph, "rainfall": rainfall}
    crop, confidence = predict_crop(features)
    st.success(f"Recommended Crop: **{crop}** (Confidence: {confidence:.2f})")

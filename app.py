import streamlit as st
import numpy as np
import joblib

# Load trained model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Wine Type Prediction", layout="centered")

st.title("🍷 Wine Type Prediction App")
st.markdown("Predict whether a wine is **Red or White** using its chemical properties.")

st.divider()

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
free_sulfur = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 50.0)
density = st.number_input("Density", 0.9, 1.1, 0.995)
ph = st.number_input("pH", 2.0, 4.5, 3.2)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.5)
alcohol = st.number_input("Alcohol (%)", 5.0, 15.0, 10.0)

if st.button("🔍 Predict Wine Type"):
    input_data = np.array([[ 
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur,
        total_sulfur, density, ph, sulphates, alcohol
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    if prediction == 1:
        st.success(f"🍷 **Red Wine** (Confidence: {confidence:.2%})")
    else:
        st.success(f"🥂 **White Wine** (Confidence: {confidence:.2%})")

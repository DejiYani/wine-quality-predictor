import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("wine_quality_model.joblib")
scaler = joblib.load("scaler.joblib")

# --- Page Config ---
st.set_page_config(page_title="Wine Quality Predictor 🍷", page_icon="🍇", layout="centered")

# --- Title & Intro ---
st.markdown("<h1 style='text-align: center; color: #8B0000;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the chemical properties of the wine sample in the sidebar to predict whether it meets **premium quality standards**.")

# --- Sidebar for Inputs ---
st.sidebar.header("Wine Chemical Properties")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.7)
citric_acid = st.sidebar.number_input("Citric Acid", value=0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", value=1.9)
chlorides = st.sidebar.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=34.0)
density = st.sidebar.number_input("Density", value=0.9978)
pH = st.sidebar.number_input("pH", value=3.51)
sulphates = st.sidebar.number_input("Sulphates", value=0.56)
alcohol = st.sidebar.number_input("Alcohol", value=9.4)

# Collect features
features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                      density, pH, sulphates, alcohol]])

# Scale features
features_scaled = scaler.transform(features)

# --- Prediction ---
if st.sidebar.button("🍇 Predict Quality"):
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    result = "Good Quality 🍷" if prediction == 1 else "Not Good ❌"
    #I get this part^, but not this v#
    confidence = f"{max(proba)*100:.2f}%"

    # Display results in a styled box
    st.markdown("---")
    #if good
    if prediction == 1:
        st.markdown(
            f"<div style='padding:20px; background-color:#1f0c0c; border-radius:10px; text-align:center;'>"
            f"<h2 style='color:#006400;'>✅ {result}</h2>"
            f"<p style='font-size:18px;'>Confidence Score: <b>{confidence}</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )
    else: #if not good 
        st.markdown(
            f"<div style='padding:20px; background-color:#1f0c0c; border-radius:10px; text-align:center;'>"
            f"<h2 style='color:#8B0000;'>⚠️ {result}</h2>"
            f"<p style='font-size:18px;'>Confidence Score: <b>{confidence}</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )

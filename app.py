import streamlit as st
import joblib
import numpy as np

# Load the trained model
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    model = joblib.load("diamond_price_predictor_model.pkl")

# Mapping for color and clarity
color_mapping = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
clarity_mapping = {"IF": 1, "VVS1": 2, "VVS2": 3, "VS1": 4, "VS2": 5, "SI1": 6, "SI2": 7, "I1": 8}

# Streamlit application
st.title("Diamond Price Prediction")

# Input fields
carat = st.number_input("Carat", min_value=0.0, step=0.01)
color = st.selectbox("Color", options=list(color_mapping.keys()))
clarity = st.selectbox("Clarity", options=list(clarity_mapping.keys()))

if st.button("Predict"):
    color_num = color_mapping.get(color, 0)
    clarity_num = clarity_mapping.get(clarity, 0)

    try:
        prediction = model.predict([[carat, color_num, clarity_num]])
        st.write(f"Predicted Price: ${float(prediction[0]):,.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

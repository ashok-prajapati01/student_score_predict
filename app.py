import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="ML Model Prediction",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ----------------------------
# Header
# ----------------------------
st.title("🤖 Machine Learning Prediction App")
st.markdown("---")
st.write("Enter the feature values below and click **Predict**.")

# ----------------------------
# Input Fields
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict", use_container_width=True):
    features = np.array([[feature1, feature2, feature3, feature4]])

    prediction = model.predict(features)

    st.success("Prediction Completed ✅")

    st.metric(
        label="Predicted Value",
        value=f"{prediction[0]:.4f}"
    )

st.markdown("---")
st.caption("Developed using Streamlit & Scikit-learn")

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page & Layout Configuration
st.set_page_config(
    page_title="Academic Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# Custom Color CSS Injection (Deep Slate, Mint Accent, Crimson & Soft Gray)
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        div[data-testid="stMetricValue"] {
            font-size: 40px;
            color: #1e3a8a;
            font-weight: 700;
        }
        .stButton>button {
            background-color: #10b981 !important;
            color: white !important;
            border-radius: 8px;
            border: none;
            height: 3em;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #059669 !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            transform: translateY(-1px);
        }
    </style>
""", unsafe_allowed_html=True)

# 2. Safe Model Loader
@st.cache_resource
def load_performance_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_performance_model()
except Exception as e:
    st.error(f"Failed to load the prediction engine (model.pkl): {e}")
    st.stop()

# 3. Header Segment
st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("Assess final outcomes dynamically by adjusting lifestyle metrics and historic academic standings below.")
st.write("---")

# 4. Multi-Column Layout Architecture
col1, col2 = st.columns([4, 3], gap="large")

with col1:
    st.subheader("📊 Input Behavior & Academic Metrics")
    
    # Grid positioning for standardizing inputs
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        hours_studied = st.number_input(
            "Daily Study Hours", 
            min_value=0.0, max_value=24.0, value=6.0, step=0.5,
            help="Average total hours focused purely on learning per day."
        )
        attendance_percent = st.slider(
            "Class Attendance (%)", 
            min_value=0.0, max_value=100.0, value=85.0, step=1.0,
            help="Overall attendance track record."
        )

    with sub_col2:
        sleep_hours = st.number_input(
            "Nightly Sleep Hours", 
            min_value=0.0, max_value=24.0, value=7.5, step=0.5,
            help="Average daily rest cycle duration."
        )
        previous_scores = st.number_input(
            "Previous Exam Score (out of 100)", 
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            help="Grade baseline achieved in the last primary examination."
        )

with col2:
    st.subheader("🔮 Predictive Analytics Outcome")
    st.write("Click below to run the K-Nearest Neighbors evaluation pipeline.")
    
    # Construct precise DataFrame structure matching model expectancies
    features_df = pd.DataFrame([{
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'previous_scores': previous_scores
    }])
    
    predict_clicked = st.button("Evaluate Performance Profile", use_container_width=True)
    
    if predict_clicked:
        with st.spinner("Calculating metric weights..."):
            try:
                # Compute predictions
                predicted_score = model.predict(features_df)[0]
                score_delta = predicted_score - previous_scores
                
                # Render results nicely
                st.write("") 
                st.metric(
                    label="Estimated Final Score Target", 
                    value=f"{predicted_score:.2f} / 100",
                    delta=f"{score_delta:+.2f} shift vs historic baseline"
                )
                
                # Contextual evaluation callouts
                if predicted_score >= 85:
                    st.success("🌟 **High Performance Profile:** Parameters show excellent habits and top-tier predictability.")
                elif predicted_score >= 60:
                    st.info("👍 **Stable Profile:** Student is securely on track. Small gains in attendance or focus hours could push this profile higher.")
                else:
                    st.warning("⚠️ **At-Risk Target Alert:** Strategic adjustments to sleep patterns or study regimens are highly recommended.")
                    
            except Exception as error:
                st.error(f"Pipeline Execution Error: {error}")
    else:
        st.info("Awaiting structural inputs. Adjust variables on the left pane and run the assessment.")

# 5. Dashboard Footer
st.write("---")
st.caption("Engineered Framework: scikit-learn 1.6.1 | Web Interface Core: Streamlit Wide Layout")

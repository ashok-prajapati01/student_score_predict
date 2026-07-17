import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:white;
}

.subtitle{
    font-size:18px;
    text-align:center;
    color:white;
}

.header{
    background:linear-gradient(90deg,#4F46E5,#06B6D4);
    padding:30px;
    border-radius:15px;
    margin-bottom:25px;
}

.result{
    background:#ffffff;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
}

.stButton>button{
    width:100%;
    background:#4F46E5;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#4338CA;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="header">
<div class="title">🎓 Student Performance Predictor</div>
<div class="subtitle">
Predict Student Score using Machine Learning
</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📥 Enter Student Details")

    hours = st.slider(
        "Hours Studied",
        0.0,
        12.0,
        5.0,
        0.5
    )

    sleep = st.slider(
        "Sleep Hours",
        0.0,
        12.0,
        7.0,
        0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        80
    )

    previous = st.slider(
        "Previous Score",
        0,
        100,
        70
    )

    st.markdown("### 📊 Input Summary")

    st.progress(hours / 12)
    st.write(f"Hours Studied: **{hours} hrs**")

    st.progress(sleep / 12)
    st.write(f"Sleep Hours: **{sleep} hrs**")

    st.progress(attendance / 100)
    st.write(f"Attendance: **{attendance}%**")

    st.progress(previous / 100)
    st.write(f"Previous Score: **{previous}**")

with col2:

    st.subheader("🤖 Prediction")

    if st.button("Predict Performance"):

        features = np.array([[hours, sleep, attendance, previous]])

        prediction = model.predict(features)[0]

        st.markdown(
            f"""
            <div class="result">
            <h2 style="color:#4F46E5;">Predicted Score</h2>
            <h1>{prediction:.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction >= 85:
            st.success("🌟 Excellent Performance!")

        elif prediction >= 70:
            st.info("👍 Good Performance!")

        elif prediction >= 50:
            st.warning("📚 Average Performance. More practice recommended.")

        else:
            st.error("⚠ Needs Improvement.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-learn")

import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.main-title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#dddddd;
    font-size:18px;
    margin-bottom:30px;
}

.card{
    background:rgba(255,255,255,0.12);
    padding:25px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.4);
    margin-bottom:20px;
}

.result{
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    box-shadow:0px 5px 20px rgba(0,114,255,0.5);
}

.stButton>button{
    width:100%;
    background:#0072ff;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0056d6;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<h1 class="main-title">🎓 Student Score Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict student scores using Machine Learning</p>', unsafe_allow_html=True)

left, right = st.columns([2,1])

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Input Features")

    col1,col2=st.columns(2)

    with col1:
        f1=st.number_input("Feature 1",value=0.0)
        f2=st.number_input("Feature 2",value=0.0)

    with col2:
        f3=st.number_input("Feature 3",value=0.0)
        f4=st.number_input("Feature 4",value=0.0)

    predict=st.button("Predict Score")

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Instructions")

    st.write("""
    • Enter all four feature values.

    • Click **Predict Score**.

    • The predicted score will appear below.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if predict:

    data=np.array([[f1,f2,f3,f4]])

    prediction=model.predict(data)

    st.markdown(
        f"""
        <div class="result">
        Predicted Score<br><br>
        {prediction[0]:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Made with ❤️ using Streamlit")

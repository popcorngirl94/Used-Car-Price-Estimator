"""
app.py  -  Used Car Price Predictor
=====================================
Run:  streamlit run app.py
Requires best_xgboost.pkl in the same folder OR at the MODEL_PATH below.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoVal · Used Car Price Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0c0e14; color: #e8e6df; }
.block-container { padding-top: 2rem !important; max-width: 1100px !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero h1 { font-size: 3.2rem; font-weight: 800; color: #c2fe6a; margin-bottom: 0.3rem; }
.hero p  { color: #888; font-size: 1rem; }

.result-box {
    background: #1a1d24;
    border: 2px solid #c2fe6a;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-box .label { color: #888; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
.result-box .price { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: #c2fe6a; }

.section-card {
    background: #13161e;
    border: 1px solid #222631;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.section-title { font-family: 'Syne', sans-serif; font-size: 0.8rem; letter-spacing: 0.12em;
                 text-transform: uppercase; color: #c2fe6a; margin-bottom: 1rem; }

div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div > input {
    background: #1a1d24 !important;
    border: 1px solid #2e3340 !important;
    color: #e8e6df !important;
    border-radius: 8px !important;
}
label { color: #aaa !important; font-size: 0.82rem !important; }

div.stButton > button {
    background: #c2fe6a;
    color: #0c0e14;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.75rem 2.5rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-card {
    flex: 1;
    background: #1a1d24;
    border: 1px solid #2e3340;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-card .m-val { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #c2fe6a; }
.metric-card .m-lbl { color: #666; font-size: 0.75rem; margin-top: 0.2rem; }

.error-box {
    background: #2a1a1a;
    border: 1px solid #ff4b4b;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #ff4b4b;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FREQUENCY ENCODER  (must match train_and_save.py exactly)
# ─────────────────────────────────────────────────────────────────
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.maps = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.maps[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.maps[col]).fillna(0)
        return X

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    os.path.join(
        os.path.dirname(__file__),
        "ML_Models",
        "best_xgboostregressor.pkl"
    ),
    os.path.join(
        os.path.dirname(__file__),
        "best_xgboostregressor.pkl"
    )
]

@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

model, model_path = load_model()

# ─────────────────────────────────────────────────────────────────
# DROPDOWN OPTIONS  (from dataset)
# ─────────────────────────────────────────────────────────────────
BRANDS = sorted([
    "Acura","Alfa Romeo","Aston Martin","Audi","Bentley","BMW","Buick","Cadillac",
    "Chevrolet","Chrysler","Dodge","Ferrari","Fiat","Ford","Genesis","GMC",
    "Honda","Hyundai","INFINITI","Jaguar","Jeep","Kia","Lamborghini","Land",
    "Lexus","Lincoln","Lotus","Lucid","Maserati","Mazda","McLaren","Mercedes-Benz",
    "MINI","Mitsubishi","Nissan","Porsche","RAM","Rivian","Rolls-Royce","Subaru",
    "Tesla","Toyota","Volkswagen","Volvo","Other"
])

FUEL_TYPES = ["Gasoline", "Hybrid", "Electric", "Diesel", "Plug-In Hybrid",
              "E85 Flex Fuel", "Other"]

CLEAN_TITLE = ["Yes", "No", "unknown"]

ACCIDENT = ["NO", "YES"]

# ─────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚗 AutoVal</h1>
    <p>Used Car Price Estimator · XGBoost · R² 0.87</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MODEL STATUS
# ─────────────────────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="error-box">
        <b>Model not found.</b><br>
        Run <code>python train_and_save.py</code> first, then place
        <code>best_xgboost.pkl</code> in the same folder as this app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:

    # ── Car Identity ──────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Car Identity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Ford"))
    with c2:
        model_name = st.text_input("Model", value="F-150 XLT",
                                   help="Type the exact model name e.g. Camry LE, X5 xDrive40i")
    model_year = st.slider("Model Year", min_value=1990, max_value=2024, value=2018, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Specs ─────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Specs</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        milage = st.number_input("Mileage (miles)", min_value=0, max_value=500000,
                                  value=45000, step=1000)
        engine_capacity = st.number_input("Engine Capacity (L)", min_value=0.0, max_value=9.0,
                                           value=2.5, step=0.1, format="%.1f")
    with c2:
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        horse_power = st.number_input("Horsepower (HP)", min_value=0, max_value=1500,
                                       value=200, step=10)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Transmission & Condition ──────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Transmission & Condition</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        transmission = st.text_input("Transmission", value="Automatic",
                                      help="e.g. Automatic, 8-Speed Automatic, CVT, 6-Speed Manual")
        clean_title = st.selectbox("Clean Title", CLEAN_TITLE)
    with c2:
        accident = st.selectbox("Accident Occurred", ACCIDENT)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:

    # ── Colors ────────────────────────────────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">Colors</div>', unsafe_allow_html=True)
    ext_col = st.text_input("Exterior Color", value="White",
                             help="e.g. White, Black, Silver, Blue, Red")
    int_col = st.text_input("Interior Color", value="Black",
                             help="e.g. Black, Beige, Gray, Brown")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button + Result ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Estimate Price →")

    if predict_btn:
        # Build input DataFrame — column order must match training data
        input_df = pd.DataFrame([{
            "brand"          : brand,
            "model"          : model_name,
            "model_year"     : int(model_year),
            "milage"         : float(milage),
            "fuel_type"      : fuel_type,
            "transmission"   : transmission,
            "ext_col"        : ext_col,
            "int_col"        : int_col,
            "clean_title"    : clean_title,
            "Engine_Capacity": float(engine_capacity),
            "Horse_Power"    : float(horse_power),
            "Accident_Occured": accident,
        }])

        try:
            predicted_price = model.predict(input_df)[0]
            formatted_price = f"${predicted_price:,.0f}"

            st.markdown(f"""
            <div class="result-box">
                <div class="label">Estimated Market Value</div>
                <div class="price">{formatted_price}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence range ± 15%
            low_est  = predicted_price * 0.85
            high_est = predicted_price * 1.15
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="m-val">${low_est:,.0f}</div>
                    <div class="m-lbl">Low estimate (−15%)</div>
                </div>
                <div class="metric-card">
                    <div class="m-val">${high_est:,.0f}</div>
                    <div class="m-lbl">High estimate (+15%)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <b>Prediction error:</b> {e}
            </div>
            """, unsafe_allow_html=True)

    # ── Model Info ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Model Info</div>
        <div style="color:#666; font-size:0.8rem; line-height:1.8;">
            Algorithm &nbsp;&nbsp;&nbsp; XGBoost + log target<br>
            CV Score &nbsp;&nbsp;&nbsp;&nbsp; R² ≈ 0.87<br>
            MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~$7,252<br>
            RMSE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ~$12,817<br>
            Training rows &nbsp; ~3,960<br>
            Model file &nbsp;&nbsp;&nbsp; best_xgboost.pkl
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e2130; margin-top:2rem;">
<p style="text-align:center; color:#444; font-size:0.75rem;">
    AutoVal · XGBoost Used Car Price Estimator · For informational purposes only
</p>
""", unsafe_allow_html=True)
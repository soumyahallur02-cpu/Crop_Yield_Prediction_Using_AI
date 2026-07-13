"""
app_streamlit.py
-----------------
Interactive web dashboard for the Crop Yield Prediction & Optimization project.

Run with:
    streamlit run app_streamlit.py

Requires: streamlit (pip install streamlit)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import streamlit as st

from optimize import optimize_inputs, marginal_analysis

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_yield_model.pkl")

st.set_page_config(page_title="AI Crop Yield Predictor", page_icon="🌾", layout="wide")
st.title("🌾 AI-Powered Crop Yield Prediction & Optimization")
st.caption("Predict yield from soil/weather/management data, then find the input mix that maximizes it.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.sidebar:
    st.header("Field Conditions")
    crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Soybean", "Cotton"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    soil_type = st.selectbox("Soil type", ["Loamy", "Sandy", "Clayey", "Silty"])

    st.subheader("Weather (seasonal)")
    rainfall_mm = st.slider("Rainfall (mm)", 50, 2000, 500)
    avg_temp_c = st.slider("Avg temperature (°C)", 5, 45, 24)
    humidity_pct = st.slider("Humidity (%)", 10, 100, 60)
    sunlight_hours = st.slider("Sunlight (hrs/day)", 2.0, 12.0, 7.0)

    st.subheader("Soil")
    soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    soil_nitrogen_kg_ha = st.slider("Nitrogen (kg/ha)", 5, 120, 50)
    soil_phosphorus_kg_ha = st.slider("Phosphorus (kg/ha)", 5, 100, 40)
    soil_potassium_kg_ha = st.slider("Potassium (kg/ha)", 5, 110, 45)
    soil_organic_matter_pct = st.slider("Organic matter (%)", 0.2, 8.0, 3.0)

    st.subheader("Remote Sensing")
    ndvi = st.slider("NDVI (vegetation index)", 0.1, 0.95, 0.6)
    days_to_harvest = st.slider("Days to harvest", 60, 220, 120)

    st.subheader("Management (controllable)")
    fertilizer_kg_ha = st.slider("Fertilizer (kg/ha)", 0, 400, 150)
    irrigation_mm = st.slider("Irrigation (mm)", 0, 900, 300)
    pesticide_l_ha = st.slider("Pesticide (L/ha)", 0.0, 8.0, 2.0)
    planting_density_k_per_ha = st.slider("Planting density (k plants/ha)", 20, 200, 85)

    budget_usd = st.number_input("Optimization budget ($/ha, optional)", min_value=0, value=0,
                                   help="Set to 0 for no budget limit")

field = dict(
    crop=crop, region=region, soil_type=soil_type,
    rainfall_mm=rainfall_mm, avg_temp_c=avg_temp_c, humidity_pct=humidity_pct,
    sunlight_hours=sunlight_hours, soil_ph=soil_ph,
    soil_nitrogen_kg_ha=soil_nitrogen_kg_ha, soil_phosphorus_kg_ha=soil_phosphorus_kg_ha,
    soil_potassium_kg_ha=soil_potassium_kg_ha, soil_organic_matter_pct=soil_organic_matter_pct,
    fertilizer_kg_ha=fertilizer_kg_ha, irrigation_mm=irrigation_mm,
    pesticide_l_ha=pesticide_l_ha, ndvi=ndvi,
    planting_density_k_per_ha=planting_density_k_per_ha, days_to_harvest=days_to_harvest,
)

pred = model.predict(pd.DataFrame([field]))[0]

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Predicted Yield", f"{pred:.2f} t/ha")

with col2:
    st.subheader("🎯 Optimization: best input combination")
    fixed = {k: v for k, v in field.items()
             if k not in ("fertilizer_kg_ha", "irrigation_mm", "pesticide_l_ha", "planting_density_k_per_ha")}
    budget = budget_usd if budget_usd > 0 else None
    best = optimize_inputs(fixed, model=model, budget_usd=budget, top_k=5)
    st.dataframe(best, use_container_width=True)

    if not best.empty:
        top = best.iloc[0]
        uplift = top["predicted_yield_tons_ha"] - pred
        pct = (uplift / pred) * 100 if pred > 0 else 0
        st.success(
            f"Best combo could raise yield by **{uplift:+.2f} t/ha ({pct:+.1f}%)** "
            f"at an input cost of **${top['input_cost_usd_ha']:.0f}/ha**."
        )

st.divider()
st.subheader("📈 Sensitivity: how yield responds to each controllable input")
curves = marginal_analysis(fixed, model=model)
tabs = st.tabs(list(curves.keys()))
for tab, (var, df_curve) in zip(tabs, curves.items()):
    with tab:
        st.line_chart(df_curve.set_index(var))

st.caption("Model: Gradient Boosting Regressor trained on historical soil/weather/management data. "
           "Replace data/crop_yield_data.csv with real farm records to deploy on your own data.")

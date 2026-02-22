import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("used_car_price_model.pkl")
    return bundle["model"], bundle["feature_cols"]

model, feature_cols = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

df = load_data()

# -----------------------------
# Master data
# -----------------------------


# -----------------------------
# UI
# -----------------------------
st.title("🚗 Used Car Price Prediction (ML Model)")

# brand
brand_list = (
    df["brand"]
    .dropna()
    .str.lower()
    .sort_values()
    .unique()
)

brand = st.selectbox(
    "Select car brand",
    brand_list
)

# model
model_list = (
    df[df["brand"].str.lower() == brand]["model"]
    .dropna()
    .str.lower()
    .sort_values()
    .unique()
)

model_name = st.selectbox(
    "Select car model",
    model_list
)

# sub model
sub_model_list = (
    df[
        (df["brand"].str.lower() == brand) &
        (df["model"].str.lower() == model_name)
    ]["sub_model"]
    .dropna()
    .str.lower()
    .sort_values()
    .unique()
)

sub_model = st.selectbox(
    "Select sub-model (optional)",
    [""] + list(sub_model_list)
)

mfg_year = st.slider("Manufacturing Year", 2010, 2025)
mileage = st.number_input("Mileage (km)", min_value=0, step=1000)

fuel_type = st.radio(
    "Fuel type",
    ["petrol", "diesel", "hybrid", "ev"]
)

transmission = st.selectbox(
    "Transmission",
    ["manual", "automatic"]
)

# -----------------------------
# Build model input
# -----------------------------
def build_input_df():
    car_age = 2025 - mfg_year

    data = {
        "mileage": mileage,
        "car_age": car_age,
        "car_age_squ": car_age ** 2,
        "fuel_ev": 1 if fuel_type == "ev" else 0,
        "fuel_hybrid": 1 if fuel_type == "hybrid" else 0,
        "transmission_automatic": 1 if transmission == "automatic" else 0,
        "brand_toyota": 1 if brand == "toyota" else 0,
        "model_camry": 1 if model == "camry" else 0,
    }

    df_input = pd.DataFrame([data])

    # Ensure all columns exist
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_cols]

    return sm.add_constant(df_input, has_constant="add")

# -----------------------------
# Prediction
# -----------------------------
def estimate_price():
    X_input = build_input_df()

    pred = model.get_prediction(X_input).summary_frame(alpha=0.05)

    price = pred["mean"].iloc[0]
    low = pred["obs_ci_lower"].iloc[0]
    high = pred["obs_ci_upper"].iloc[0]

    st.success(f"💰 Estimated price: {price:,.0f} THB")
    st.info(f"📊 95% prediction range: {low:,.0f} – {high:,.0f} THB")

# -----------------------------
# Button
# -----------------------------
st.button("Estimate Price", on_click=estimate_price)

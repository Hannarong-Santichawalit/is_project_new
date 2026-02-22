import streamlit as st
import pandas as pd
from pathlib import Path

# =============================
# Page setup
# =============================
st.set_page_config(page_title="🚗 Freshness-Priority Used Car Recommender", layout="wide")
st.title("Used Car Recommendation (Freshness + Trust)")

st.markdown(
    """
This app recommends cars using:
1) **Feasibility filtering** (budget + requirements) based on **predicted price**  
2) **Ranking** that prioritizes **freshness (newer cars)** + **trust (BAR score)**  
3) **Diversity**: returns **unique models** to avoid repetitive results
"""
)

st.divider()

# =============================
# Load data
# =============================
DEFAULT_PATH = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\MS STAT\IS_Project\matrial\car_master_data_predicted.xlsx")  # change if needed

@st.cache_data
def load_excel(file_obj_or_path):
    df = pd.read_excel(file_obj_or_path)
    # Ensure predicted price is used as "price"
    if "price_predict" in df.columns:
        df["price"] = df["price_predict"]
    return df

st.title("🚗 Used Car Recommender")

uploaded = st.file_uploader(
    "Upload Excel (optional). If you don’t upload, we use the default dataset.",
    type=["xlsx"]
)

if uploaded is not None:
    df = load_excel(uploaded)
    st.success("Using uploaded file ✅")
else:
    if not DEFAULT_PATH.exists():
        st.error(f"Default file not found: {DEFAULT_PATH.resolve()}")
        st.stop()
    df = load_excel(DEFAULT_PATH)
    st.info(f"Using default file: {DEFAULT_PATH} ✅")

df["price_predict"] = df["price_predict"].round(-3)
st.write("Rows:", len(df))
# st.dataframe(df.head())

# =============================
# Validate required columns
# =============================
required_cols = ["brand", "model", "price_predict", "car_age", "car_type", "bar_score", "fuel_type", "transmission"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Your file is missing required columns: {missing}")
    st.stop()

# Convert numeric columns safely
for num_col in ["price_predict", "car_age", "bar_score"]:
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

# Drop rows with no predicted price or no age (core for scoring)
df = df.dropna(subset=["price_predict", "car_age"]).copy()

st.caption(f"Dataset loaded: **{len(df):,}** rows")

st.divider()

# =============================
# Inputs (Budget + Requirements)
# =============================
st.subheader("1) Budget & Requirements (Feasibility Filter)")

col1, col2, col3 = st.columns(3)

min_price = float(df["price_predict"].min())
max_price = float(df["price_predict"].max())

with col1:
    budget = st.slider(
        "Predicted price range (THB)",
        min_value=0,
        max_value=int(max_price),
        value=(int(max(min_price, 500_000)), int(min(max_price, 800_000))),
        step=10000
    )
    min_p, max_p = budget

with col2:
    fuel_options = sorted([x for x in df["fuel_type"].dropna().unique().tolist() if x.strip() != ""])
    fuel = st.selectbox("Fuel type", options=["(Any)"] + fuel_options, index=0)

with col3:
    trans_options = sorted([x for x in df["transmission"].dropna().unique().tolist() if x.strip() != ""])
    trans = st.selectbox("Transmission", options=["(Any)"] + trans_options, index=0)

col4, col5, col6 = st.columns(3)

with col4:
    car_type_options = sorted([x for x in df["car_type"].dropna().unique().tolist() if x.strip() != ""])
    car_type = st.selectbox("Car type", options=["(Any)"] + car_type_options, index=0)

with col5:
    top_n = st.number_input("Top N (unique models)", min_value=1, max_value=10, value=5, step=1)

with col6:
    dedupe_by = st.selectbox("Diversity key", options=["model", "brand+model"], index=0)

st.divider()

# =============================
# Scoring Logic (Freshness + Trust)
# =============================
st.subheader("2) Ranking Logic (Freshness + Trust)")

w_col1, w_col2, w_col3 = st.columns(3)

with w_col1:
    w_age = st.slider("Weight: Freshness (car age)", 0.0, 1.0, 0.5, 0.05)

with w_col2:
    w_trust = st.slider("Weight: Trust (BAR score)", 0.0, 1.0, 0.5, 0.05)

with w_col3:
    bar_default = st.slider("Default BAR score when missing", 0.0, 1.0, 0.5, 0.05)

# Normalize weights to sum to 1 (academic-friendly)
w_sum = w_age + w_trust
if w_sum == 0:
    st.warning("Both weights are 0. Please set at least one weight > 0.")
    st.stop()

w_age /= w_sum
w_trust /= w_sum

st.caption(f"Normalized weights → Freshness: **{w_age:.2f}**, Trust: **{w_trust:.2f}**")

def get_freshness_priority_recommendations(
    df: pd.DataFrame,
    min_p=None,
    max_p=None,
    fuel=None,
    trans=None,
    car_type=None,
    top_n=10,
    w_age=0.5,
    w_trust=0.5,
    bar_default=3.5,
    dedupe_by="model",
) -> pd.DataFrame:

    mask = pd.Series(True, index=df.index)

    if min_p is not None:
        mask &= df["price_predict"].ge(min_p)
    if max_p is not None:
        mask &= df["price_predict"].le(max_p)

    if fuel is not None:
        fuel = fuel.strip().lower()
        mask &= df["fuel_type"].astype(str).str.strip().str.lower().eq(fuel)

    if trans is not None:
        trans = trans.strip().lower()
        mask &= df["transmission"].astype(str).str.strip().str.lower().eq(trans)

    if car_type is not None:
        car_type = car_type.strip().lower()
        mask &= df["car_type"].astype(str).str.strip().str.lower().eq(car_type)

    filtered = df.loc[mask].copy()
    if filtered.empty:
        return filtered

    # Freshness score: newer (lower car_age) -> higher
    max_age = filtered["car_age"].max()
    min_age = filtered["car_age"].min()

    if pd.isna(max_age) or pd.isna(min_age):
        filtered["age_score"] = 0.0
    elif max_age != min_age:
        filtered["age_score"] = (max_age - filtered["car_age"]) / (max_age - min_age)
    else:
        filtered["age_score"] = 1.0

    # Normalize BAR Score (Highest quality = 1.0, Lowest quality in set = 0.0)
    max_bar = filtered['bar_score'].max()
    min_bar = filtered['bar_score'].min()
    
    if max_bar != min_bar:
        filtered['bar_norm'] = (filtered['bar_score'] - min_bar) / (max_bar - min_bar)
    else:
        filtered['bar_norm'] = 1.0

    # Trust score from BAR (0..1)
    bar = filtered["bar_norm"].fillna(bar_default).clip(0, 5)
    filtered["trust_score"] = (bar)

    # Final score
    filtered["final_score"] = (filtered["age_score"] * w_age) + (filtered["trust_score"] * w_trust)

    # Diversity (unique models)
    if dedupe_by == "brand+model":
        filtered["_dedupe_key"] = (
            filtered["brand"].astype(str).str.strip().str.lower()
            + "||"
            + filtered["model"].astype(str).str.strip().str.lower()
        )
        dedupe_key = "_dedupe_key"
    else:
        dedupe_key = "model"

    ranked = (
        filtered.sort_values("final_score", ascending=False)
                .drop_duplicates(subset=[dedupe_key])
                .head(int(top_n))
                .drop(columns=["_dedupe_key"], errors="ignore")
    )

    return ranked

# =============================
# Recommend
# =============================
st.subheader("3) Recommend")

run = st.button("🎯 Recommend cars", use_container_width=True)

if run:
    # Convert "(Any)" to None
    fuel_in = None if fuel == "(Any)" else fuel
    trans_in = None if trans == "(Any)" else trans
    type_in = None if car_type == "(Any)" else car_type

    results = get_freshness_priority_recommendations(
        df=df,
        min_p=min_p,
        max_p=max_p,
        fuel=fuel_in,
        trans=trans_in,
        car_type=type_in,
        top_n=top_n,
        w_age=w_age,
        w_trust=w_trust,
        bar_default=bar_default,
        dedupe_by=dedupe_by,
    )

    st.divider()
    st.subheader("✅ Results")

    st.write(f"Cars matching constraints: **{len(df[(df['price_predict']>=min_p) & (df['price_predict']<=max_p)]):,}** (before fuel/trans/type filters)")
    st.write(f"Recommended (unique {dedupe_by}): **{len(results):,}**")

    if results.empty:
        st.warning("No cars match your filters. Try widening budget or setting some fields to (Any).")
        st.stop()

    # Nice display table
    show_cols = [
        "brand", "model", "price_predict", "car_age", "car_type",
        "fuel_type", "transmission", "bar_score", "age_score", "trust_score", "final_score"
    ]
    show_cols = [c for c in show_cols if c in results.columns]

    # Format a copy for display
    view = results[show_cols].copy()
    view = view.rename(columns={"price_predict": "estimated_price"})

    st.dataframe(
        view.sort_values("final_score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Top cards
    st.subheader("🏁 Top Picks")
    for _, r in results.sort_values("final_score", ascending=False).iterrows():
        st.markdown(
            f"""
**{str(r['brand']).title()} {str(r['model']).title()}**  
Predicted Price: **{r['price_predict']:,.0f} THB**  
Car Age: **{r['car_age']:.1f} years**  
Type: **{str(r['car_type']).title()}** | ⛽ {str(r['fuel_type']).title()} | ⚙️ {str(r['transmission']).title()}  
BAR score: **{(r['bar_score'] if pd.notna(r['bar_score']) else bar_default):.2f} / 5**  
Final score: **{r['final_score']:.3f}**  (Freshness {w_age:.2f} + Trust {w_trust:.2f})
"""
        )
        st.divider()

# =============================
# Academic note
# =============================
with st.expander("📘 Academic Explanation (thesis-ready)"):
    st.markdown(
        """
### Recommendation Framework
This module implements a **constraint-based recommender** followed by a **freshness-prioritized ranking**.

**1) Feasibility filtering (constraints)**  
Cars are first filtered using user-defined constraints:  
- Budget range based on **predicted price**  
- Vehicle requirements (fuel type, transmission, car type)

**2) Freshness scoring**  
Freshness is operationalized using normalized vehicle age:
- Newer cars (lower `car_age`) receive higher utility.

**3) Trust scoring (social feedback)**  
Perceived reliability is represented by `bar_score` (scaled to 0–1). Missing values are handled using a conservative default.

**4) Final score (linear utility)**  
The final ranking score is computed as:
- `final_score = w_age * age_score + w_trust * trust_score`
where the weights are normalized to sum to 1 for interpretability.

**5) Diversity constraint**  
To avoid repetitive recommendations, the system enforces **unique models** (or brand+model) in the final list.

This design supports **interpretability**, aligns with **business feasibility**, and avoids user-tracking assumptions typical of collaborative filtering.
"""
    )
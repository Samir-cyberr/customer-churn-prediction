from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.predict import load_model, predict_churn


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 E-Commerce Customer Churn Predictor")
st.caption("Machine Learning portfolio project by Samir")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Run these commands first:\n\n"
        "python -m src.generate_data\n\n"
        "python -m src.train"
    )
    st.stop()

st.subheader("Customer information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    tenure_months = st.slider("Tenure (months)", 1, 60, 18)
    monthly_spend = st.number_input("Monthly spend ($)", 15.0, 250.0, 75.0)
    orders = st.number_input("Number of orders", 0, 60, 8)
    avg_order_value = st.number_input("Average order value ($)", 10.0, 250.0, 50.0)
    days_since_last_order = st.slider("Days since last order", 1, 180, 20)

with col2:
    sessions = st.slider("Monthly website sessions", 1, 60, 12)
    support_tickets = st.slider("Support tickets", 0, 10, 1)
    discount_usage = st.slider("Discount usage ratio", 0.0, 1.0, 0.3)
    satisfaction = st.slider("Satisfaction", 1.0, 5.0, 3.5, 0.1)
    mobile_app = st.selectbox("Uses mobile app?", ["Yes", "No"])

customer = {
    "age": age,
    "tenure_months": tenure_months,
    "monthly_spend": monthly_spend,
    "orders": orders,
    "avg_order_value": avg_order_value,
    "days_since_last_order": days_since_last_order,
    "sessions": sessions,
    "support_tickets": support_tickets,
    "discount_usage": discount_usage,
    "satisfaction": satisfaction,
    "mobile_app": 1 if mobile_app == "Yes" else 0,
}

if st.button("Predict churn risk", type="primary"):
    prediction, probability = predict_churn(model, customer)

    st.metric("Churn probability", f"{probability:.1%}")

    if prediction:
        st.warning("⚠️ High churn risk")
    else:
        st.success("✅ Low churn risk")

    st.progress(probability)

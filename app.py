import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# Load saved files
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# 🔥 Get feature names directly from scaler
feature_names = scaler.feature_names_in_

st.title("💳 AI Powered Credit Card Fraud Detection")
st.markdown("Real-time Transaction Risk Monitoring System")
st.markdown("---")

st.sidebar.header("Enter Transaction Details")

# All Inputs
transaction_amount = st.sidebar.number_input("Transaction Amount", 0.0)
transaction_time = st.sidebar.number_input("Transaction Time", 0.0)
merchant_category = st.sidebar.selectbox("Merchant Category", encoders["merchant_category"].classes_)
customer_age = st.sidebar.number_input("Customer Age", 18, 100)
customer_location = st.sidebar.selectbox("Customer Location", encoders["customer_location"].classes_)
device_type = st.sidebar.selectbox("Device Type", encoders["device_type"].classes_)
card_type = st.sidebar.selectbox("Card Type", encoders["card_type"].classes_)
transaction_type = st.sidebar.selectbox("Transaction Type", encoders["transaction_type"].classes_)
previous_fraud_count = st.sidebar.number_input("Previous Fraud Count", 0)
avg_transaction_amount = st.sidebar.number_input("Average Transaction Amount", 0.0)
account_age_days = st.sidebar.number_input("Account Age (Days)", 0)
num_transactions_24h = st.sidebar.number_input("Transactions (Last 24H)", 0)
num_transactions_7d = st.sidebar.number_input("Transactions (Last 7D)", 0)
is_international = st.sidebar.selectbox("International Transaction", [0, 1])
is_weekend = st.sidebar.selectbox("Weekend Transaction", [0, 1])
risk_score = st.sidebar.slider("Risk Score", 0.0, 1.0)

if st.sidebar.button("Analyze Transaction"):

    input_data = {
        "transaction_amount": transaction_amount,
        "transaction_time": transaction_time,
        "merchant_category": merchant_category,
        "customer_age": customer_age,
        "customer_location": customer_location,
        "device_type": device_type,
        "card_type": card_type,
        "transaction_type": transaction_type,
        "previous_fraud_count": previous_fraud_count,
        "avg_transaction_amount": avg_transaction_amount,
        "account_age_days": account_age_days,
        "num_transactions_24h": num_transactions_24h,
        "num_transactions_7d": num_transactions_7d,
        "is_international": is_international,
        "is_weekend": is_weekend,
        "risk_score": risk_score
    }

    df = pd.DataFrame([input_data])

    # Encode categorical
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])

    # 🔥 Ensure correct column order automatically
    df = df[feature_names]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    st.subheader("📊 Fraud Risk Analysis")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    if probability > 0.7:
        st.error("🚨 HIGH RISK TRANSACTION DETECTED")
    elif probability > 0.4:
        st.warning("⚠ MEDIUM RISK TRANSACTION")
    else:
        st.success("✅ LOW RISK TRANSACTION")

    st.write(f"Fraud Probability Score: **{probability:.4f}**")

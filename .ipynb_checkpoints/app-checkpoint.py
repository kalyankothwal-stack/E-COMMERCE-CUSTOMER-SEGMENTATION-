# ============================================================
# E-COMMERCE CUSTOMER SEGMENTATION APP
# Final Streamlit Deployment App
# Project by Kalyan
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# LOAD MODELS & ENCODERS
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    predictor = joblib.load("cluster_classifier.pkl")   # Random Forest
    scaler = joblib.load("robust_scaler.pkl")
    encoder = joblib.load("label_encoder_top_category.pkl")
    return predictor, scaler, encoder

predictor, scaler, encoder = load_models()

# ------------------------------------------------------------
# CLUSTER NAMES (MATCHES IPYNB – K = 4)
# ------------------------------------------------------------
CLUSTER_NAMES = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk Customers",
    3: "Occasional Buyers"
}

CLUSTER_RECOMMENDATIONS = {
    0: "Reward with VIP offers, exclusive deals, and early access.",
    1: "Upsell premium products and strengthen loyalty programs.",
    2: "Run win-back campaigns with discounts and re-engagement emails.",
    3: "Encourage more frequent purchases using targeted promotions."
}

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("📦 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "🔮 Predict Customer Segment", "📊 Cluster Analysis", "💡 Business Insights"]
)

# ------------------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🏠 Customer Segmentation Dashboard")

    st.markdown("""
    This dashboard presents an overview of customer segmentation using **RFM analysis** 
    and **K-Means clustering (K=4)**.  
    Predictions are powered by a **Random Forest classifier** trained on engineered customer features.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Clusters", "4")
    col2.metric("Model Used", "Random Forest")
    col3.metric("Segmentation Method", "RFM + K-Means")

   
# ------------------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------------------
elif page == "🔮 Predict Customer Segment":
    st.title("🔮 Predict Customer Segment")

    st.markdown("Enter customer details to predict their **segment and business action**.")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input("Recency (days)", min_value=0, value=30)
        frequency = st.number_input("Frequency", min_value=1, value=5)
        monetary_total = st.number_input("Total Monetary Value (£)", min_value=0.0, value=500.0)
        monetary_avg = st.number_input("Average Order Value (£)", min_value=0.0, value=100.0)
        purchase_freq_month = st.number_input("Purchase Frequency per Month", min_value=0.0, value=1.0)

    with col2:
        customer_age = st.number_input("Customer Age (days)", min_value=1, value=365)
        return_rate = st.slider("Return Rate (%)", 0.0, 100.0, 5.0)
        is_loyal = st.selectbox("Is Loyal Customer?", [0, 1])
        top_category = st.selectbox(
            "Top Category",
            encoder.classes_.tolist()
        )
        total_quantity = st.number_input("Total Quantity Purchased", min_value=1, value=10)

    if st.button("Predict Segment"):
        top_cat_encoded = encoder.transform([top_category])[0]

        # IMPORTANT:
        # ReturnRate was converted from 'inf' to 0 during IPYNB preprocessing.
        # Here we ensure it is in the same scale (0–1).
        features = np.array([[
            recency,
            frequency,
            monetary_total,
            monetary_avg,
            purchase_freq_month,
            customer_age,
            return_rate / 100,   # Scale match with training data
            is_loyal,
            top_cat_encoded,
            total_quantity
        ]])

        features_scaled = scaler.transform(features)

        prediction = predictor.predict(features_scaled)[0]
        probabilities = predictor.predict_proba(features_scaled)[0]

        st.success(f"🎯 Predicted Segment: **{CLUSTER_NAMES[prediction]}**")
        st.write(f"💡 Recommendation: {CLUSTER_RECOMMENDATIONS[prediction]}")

        prob_df = pd.DataFrame({
            "Cluster": [CLUSTER_NAMES[i] for i in range(len(probabilities))],
            "Confidence": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Cluster",
            y="Confidence",
            title="Prediction Confidence Scores",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# CLUSTER ANALYSIS PAGE
# ------------------------------------------------------------
elif page == "📊 Cluster Analysis":
    st.title("📊 Cluster Analysis")

    st.markdown("""
    This section provides a **high-level interpretation** of each customer segment 
    derived from clustering analysis.
    """)

    for cid, name in CLUSTER_NAMES.items():
        st.subheader(f"Cluster {cid}: {name}")
        st.write(CLUSTER_RECOMMENDATIONS[cid])

# ------------------------------------------------------------
# BUSINESS INSIGHTS PAGE
# ------------------------------------------------------------
elif page == "💡 Business Insights":
    st.title("💡 Business Insights & Strategy")

    st.markdown("""
    Actionable insights derived from customer segmentation:
    """)

    st.markdown("""
    **🏆 Champions**
    - Focus on retention and exclusivity
    - Early access to products
    - Premium loyalty programs

    **❤️ Loyal Customers**
    - Upsell high-margin products
    - Personalized offers
    - Membership incentives

    **⚠️ At-Risk Customers**
    - Re-engagement campaigns
    - Time-bound discounts
    - Reminder emails

    **🛒 Occasional Buyers**
    - Targeted promotions
    - Cross-selling strategies
    - Product recommendations
    """)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("📌 E-Commerce Customer Segmentation | Streamlit Deployment | By Kalyan")

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import openai

# Load model and scaler
bundle = joblib.load("marketing_classifier_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# --- Custom Style ---
pink = "#f08ebc"

st.set_page_config(page_title="Customer Adopter Prediction", layout="wide")
st.markdown(f"""
    <style>
    .main {{
        background-color: #fff0f5;
        overflow: hidden;
    }}
    .stButton>button, .stDownloadButton>button {{
        background-color: {pink};
        color: white;
        font-weight: bold;
    }}
    .block-container{{
        padding-top: 1rem;
        padding-bottom: 0rem;
    }}
    .element-container:has(.stDataFrame) {{
        max-height: 300px;
        overflow-y: auto;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Premium Adopter Prediction Dashboard")
st.markdown("Use this dashboard to identify customers most likely to adopt your premium offering.")

# --- Sidebar: Inputs ---
st.sidebar.header("🎯 Campaign ROI Simulator")
cost_per_customer = st.sidebar.number_input("Marketing Cost per Customer ($)", min_value=0.0, value=1.0, step=0.1)
revenue_per_conversion = st.sidebar.number_input("Revenue per Adopter ($)", min_value=0.0, value=10.0, step=0.5)
top_k_percent = st.sidebar.slider("Top % Customers to Target", min_value=5, max_value=50, value=20, step=5)

# --- Step 1: Upload File ---
st.subheader("1️⃣ Upload Customer Data")
uploaded_file = st.file_uploader("Upload a CSV file with customer features (excluding 'adopter')", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = (prob > 0.3).astype(int)

        df_result = df.copy()
        df_result["Predicted_Probability"] = prob
        df_result["Predicted_Adopter"] = pred

        st.subheader("2️⃣ Prediction Results")
        st.dataframe(df_result.sort_values("Predicted_Probability", ascending=False).head(20), height=300)

        csv_download = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("🔧 Download Predictions", data=csv_download, file_name="predicted_customers.csv", mime='text/csv')

        # --- ROI Calculator ---
        st.subheader("3️⃣ Campaign ROI Estimate")
        cutoff = int(len(df_result) * top_k_percent / 100)
        top_customers = df_result.sort_values("Predicted_Probability", ascending=False).head(cutoff)
        n_targeted = len(top_customers)
        n_predicted_adopters = top_customers["Predicted_Adopter"].sum()

        total_cost = n_targeted * cost_per_customer
        total_revenue = n_predicted_adopters * revenue_per_conversion
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Targeted Customers", n_targeted)
        col2.metric("Expected Adopters", int(n_predicted_adopters))
        col3.metric("Estimated ROI", f"{roi:.2f}")

        # --- SHAP Explainability ---
        st.subheader("4️⃣ What Drives Adoption?")
        with st.spinner("Explaining model predictions with SHAP..."):
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)

            st.markdown("**Top Features Influencing Adoption:**")
            fig_beeswarm = shap.plots.beeswarm(shap_values, max_display=10, show=False)
            plt.gcf().set_size_inches(6, 4)
            st.pyplot(bbox_inches="tight", dpi=250, clear_figure=True)

            fig_bar = shap.plots.bar(shap_values, max_display=10, show=False)
            plt.gcf().set_size_inches(6, 3)
            st.pyplot(bbox_inches="tight", dpi=250, clear_figure=True)

        # --- Lift Curve ---
        st.subheader("5️⃣ Lift Curve")
        lift_df = pd.DataFrame({"prob": prob})
        lift_df["actual"] = pred
        lift_df = lift_df.sort_values(by="prob", ascending=False).reset_index(drop=True)
        lift_df["x"] = (lift_df.index + 1) / len(lift_df)
        lift_df["y"] = (lift_df["actual"].cumsum() / lift_df["actual"].sum()) / lift_df["x"]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(lift_df["x"], lift_df["y"], color=pink, lw=2, label="Model")
        ax.axhline(1, color="grey", linestyle="--", label="Baseline")
        ax.set_xlabel("Fraction of Population (sorted)")
        ax.set_ylabel("Lift Ratio")
        ax.set_title("Lift Curve")
        ax.legend()
        st.pyplot(fig)

        # --- LLM Suggestion Layer ---
        st.subheader("6️⃣ Campaign Suggestions 🧪")
        top_features = shap_values.abs.mean(0).values
        feature_names = X_scaled.columns
        top_feature_df = pd.DataFrame({"Feature": feature_names, "Impact": top_features})
        top_feature_df = top_feature_df.sort_values(by="Impact", ascending=False).head(5)

        prompt = f"""
        Based on the following most important features influencing customer adoption:
        {top_feature_df.to_string(index=False)}

        Suggest 3 targeted marketing actions a business team can take to improve conversion.
        """

        # Simulated LLM Output (replace with actual LLM API call if available)
        st.markdown("**Suggested Campaign Ideas:**")
        st.markdown("1. Offer a personalized email campaign to customers with a spike in engagement (e.g., increased song listening).")
        st.markdown("2. Provide a time-limited discount to customers from key regions or countries likely to convert.")
        st.markdown("3. Highlight the value of loved/favorite tracks by promoting premium playlists and offline access features.")

    except Exception as e:
        st.error(f"There was a problem processing your file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

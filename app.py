# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import requests
import tempfile
from fpdf import FPDF

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
    }}
    .stButton>button, .stDownloadButton>button {{
        background-color: {pink};
        color: white;
        font-weight: bold;
    }}
    .block-container{{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Premium Adoption Prediction Dashboard")
st.markdown("Upload new customer leads and simulate campaign ROI.")

# Sidebar Inputs
st.sidebar.header("🎯 Campaign ROI Simulator")
cost_per_customer = st.sidebar.number_input("Marketing Cost per Customer ($)", min_value=0.0, value=1.0, step=0.1)
revenue_per_conversion = st.sidebar.number_input("Revenue per Adoption ($)", min_value=0.0, value=10.0, step=0.5)
top_k_percent = st.sidebar.slider("Top % Customers to Target", min_value=5, max_value=100, value=20, step=10)

# File Upload
uploaded_file = st.file_uploader("Upload CSV file with new customer leads", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        X_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = (prob > 0.3).astype(int)

        df_result = df.copy()
        df_result["Predicted_Probability"] = prob
        df_result["Predicted_Adopter"] = pred

        tabs = st.tabs(["📊 Results", "💡 Insights & Campaign Suggestions"])

        with tabs[0]:
            st.subheader("Prediction Results (Showing Top 20 Customers)")
            st.dataframe(df_result.sort_values("Predicted_Probability", ascending=False).head(20), use_container_width=True, height=250)
            csv_download = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv_download, file_name="predicted_customers.csv", mime='text/csv')

        with tabs[1]:
            st.subheader("🧠 Campaign Insights & Suggestions")

            # === ROI Metrics ===
            cutoff = int(len(df_result) * top_k_percent / 100)
            top_customers = df_result.sort_values("Predicted_Probability", ascending=False).head(cutoff)
            n_targeted = len(top_customers)
            n_predicted_adopters = top_customers["Predicted_Adopter"].sum()

            total_cost = n_targeted * cost_per_customer
            total_revenue = n_predicted_adopters * revenue_per_conversion
            roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0

            st.markdown("### 📊 Campaign ROI Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Targeted Customers", n_targeted, help="Number of customers you plan to reach in the campaign")
            col2.metric("📈 Expected Adopters", int(n_predicted_adopters), help="Predicted number of conversions")
            col3.metric("💰 Estimated ROI", f"{roi:.2f}", help="Return on investment for this campaign")

            # === SHAP Plot ===
            st.markdown("### 🔍 Top Features Influencing Adoption")
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)
            fig1 = plt.figure(figsize=(6, 4))
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig1)

            # === Top SHAP Features ===
            top_features = shap_values.abs.mean(0).values
            feature_names = X_scaled.columns
            top_feature_df = pd.DataFrame({"Feature": feature_names, "Impact": top_features})
            top_feature_df = top_feature_df.sort_values(by="Impact", ascending=False).head(10)

            # === Hugging Face API (replace with your secret) ===
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

            def query_hf(prompt):
                response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
                return response.json()[0]["generated_text"]

            # === LLM Explanation Prompt ===
            explain_prompt = f"""
            You are a marketing data strategist.

            Given the following top features ranked by their impact on customer adoption:
            {top_feature_df.to_string(index=False)}

            For each feature, explain what user behavior it captures and how it might relate to adoption of a premium subscription.
            Focus on providing business-relevant reasoning without technical jargon.
            """

            if st.button("🔄 Generate LLM Explanations"):
                st.markdown("### 📊 What Influences Adoption?")
                with st.spinner("Analyzing SHAP results with LLM..."):
                    try:
                        llm_explainer = query_hf(explain_prompt)
                        st.markdown(llm_explainer)
                    except:
                        st.warning("⚠️ Could not load explanation from LLM. Check API key or rate limit.")

                # === LLM Campaign Prompt ===
                campaign_prompt = f"""
                Based on these top features influencing adoption:
                {', '.join(top_feature_df['Feature'].tolist()[:5])}

                Suggest 3 targeted marketing campaigns that a business team can run to increase premium subscriptions.
                Tie each campaign back to specific user behaviors in the features.
                """

                st.markdown("### 💡 Campaign Recommendations")
                with st.spinner("Brainstorming campaign strategies..."):
                    try:
                        llm_campaigns = query_hf(campaign_prompt)
                        st.markdown(llm_campaigns)

                        # Downloadable report
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, llm_explainer + "\n\n" + llm_campaigns)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                            pdf.output(tmpfile.name)
                            with open(tmpfile.name, "rb") as f:
                                st.download_button("📄 Download Insights Report (PDF)", f, file_name="campaign_recommendations.pdf")

                    except:
                        st.warning("⚠️ Could not load suggestions from LLM.")

    except Exception as e:
        st.error(f"There was a problem processing your file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin analysis.")

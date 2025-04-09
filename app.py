# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from transformers import pipeline

# Load model and scaler
bundle = joblib.load("marketing_classifier_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# Load Hugging Face text-generation pipeline
llm_pipeline = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=256)

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
    .box {{
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
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
            st.subheader("Prediction Results (Top 20 Customers)")
            st.dataframe(df_result.sort_values("Predicted_Probability", ascending=False).head(20), use_container_width=True, height=250)
            csv_download = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv_download, file_name="predicted_customers.csv", mime='text/csv')

        with tabs[1]:
            st.markdown("<h3>📊 Campaign ROI Summary</h3>", unsafe_allow_html=True)
            cutoff = int(len(df_result) * top_k_percent / 100)
            top_customers = df_result.sort_values("Predicted_Probability", ascending=False).head(cutoff)
            n_targeted = len(top_customers)
            n_predicted_adopters = top_customers["Predicted_Adopter"].sum()
            total_cost = n_targeted * cost_per_customer
            total_revenue = n_predicted_adopters * revenue_per_conversion
            roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0

            col1, col2 = st.columns([1, 2])
            with col1:
                with st.container():
                    st.markdown("<div class='box'>", unsafe_allow_html=True)
                    st.metric("🎯 Targeted Customers", n_targeted)
                    st.metric("📈 Expected Adopters", int(n_predicted_adopters))
                    st.metric("💰 Estimated ROI", f"{roi:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                with st.container():
                    st.markdown("<div class='box'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Top Features Influencing Adoption")
                    explainer = shap.Explainer(model, X_scaled)
                    shap_values = explainer(X_scaled)
                    fig = plt.figure(figsize=(5.5, 3))
                    shap.plots.beeswarm(shap_values, max_display=10, show=False)
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

            # === LLM Explanations ===
            st.markdown("<h3>📊 What Influences Adoption?</h3>", unsafe_allow_html=True)
            top_features = shap_values.abs.mean(0).values
            feature_names = X_scaled.columns
            top_feature_df = pd.DataFrame({"Feature": feature_names, "Impact": top_features})
            top_feature_df = top_feature_df.sort_values(by="Impact", ascending=False).head(5)

            # Run LLM to explain features
            explanation_prompt = f"""
            You are a marketing data strategist.
            Given the following top features ranked by their impact on customer adoption:

            {top_feature_df.to_string(index=False)}

            For each feature, explain what user behavior it captures and how it might relate to adoption of a premium subscription. Use a clear, business-relevant explanation.
            """

            llm_response = llm_pipeline(explanation_prompt)[0]['generated_text']

            explanations = [line for line in llm_response.split("\n") if ":" in line]
            parsed_rows = [(line.split(":")[0].strip(), line.split(":")[1].strip()) for line in explanations[:5]]

            st.markdown("### 🧠 Feature Interpretations")
            feature_exp_df = pd.DataFrame(parsed_rows, columns=["Feature", "How does it impact?"])
            st.table(feature_exp_df)

            # Run LLM to generate campaign ideas
            campaign_prompt = f"""
            Based on these top features influencing adoption:
            {', '.join(top_feature_df['Feature'].tolist())}
            Suggest 3 targeted marketing campaigns a business team can run to increase premium subscriptions. Tie each campaign back to specific user behavior.
            """

            campaign_response = llm_pipeline(campaign_prompt)[0]['generated_text']
            ideas = [line for line in campaign_response.split("\n") if line.strip() and line[0].isdigit()]

            st.markdown("### 💡 Campaign Recommendations")
            for idea in ideas:
                st.markdown(f"{idea}")

    except Exception as e:
        st.error(f"There was a problem processing your file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin analysis.")

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import openai

# Load model and scaler
bundle = joblib.load("marketing_classifier_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# --- Custom Style ---
pink = "#f08ebc"
light_gray = "#f5f5f5"

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
    .roi-box, .shap-box {{
        background-color: {light_gray};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
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
            st.subheader("Prediction Results (Top 20)")
            st.dataframe(df_result.sort_values("Predicted_Probability", ascending=False).head(20), use_container_width=True, height=250)
            csv_download = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv_download, file_name="predicted_customers.csv", mime='text/csv')

        with tabs[1]:
            st.markdown("""
            <h3 style='color:#f08ebc;'>📊 Campaign ROI Summary & 🔍 Top Feature Insights</h3>
            <div style='display:flex; gap:2rem;'>
            """, unsafe_allow_html=True)

            # ROI Summary Box
            st.markdown("<div class='roi-box'>", unsafe_allow_html=True)
            st.markdown("### 📊 Campaign ROI Summary")
            cutoff = int(len(df_result) * top_k_percent / 100)
            top_customers = df_result.sort_values("Predicted_Probability", ascending=False).head(cutoff)
            n_targeted = len(top_customers)
            n_predicted_adopters = top_customers["Predicted_Adopter"].sum()
            total_cost = n_targeted * cost_per_customer
            total_revenue = n_predicted_adopters * revenue_per_conversion
            roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Targeted Customers", n_targeted)
            col2.metric("📈 Expected Adopters", int(n_predicted_adopters))
            col3.metric("💰 Estimated ROI", f"{roi:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # SHAP Summary Box
            st.markdown("<div class='shap-box'>", unsafe_allow_html=True)
            st.markdown("### 🔍 Top Features Influencing Adoption")
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)
            fig1 = plt.figure(figsize=(5, 3))
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # LLM Insight Generation
            top_features = shap_values.abs.mean(0).values
            feature_names = X_scaled.columns
            top_feature_df = pd.DataFrame({"Feature": feature_names, "Impact": top_features})
            top_feature_df = top_feature_df.sort_values(by="Impact", ascending=False).head(5)

            feature_text = "\n".join([f"{row.Feature} {row.Impact:.4f}" for _, row in top_feature_df.iterrows()])
            prompt_explain = f"""
            Given the following top features ranked by their impact on customer adoption:
            Feature       Impact
            {feature_text}

            For each feature, explain what user behavior it captures and how it might relate to adoption of a premium subscription.
            Focus on providing business-relevant reasoning without technical jargon.
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing data strategist."},
                    {"role": "user", "content": prompt_explain}
                ]
            )
            explanation_text = response["choices"][0]["message"]["content"]

            st.subheader("🧩 What Influences Adoption?")
            explanation_lines = explanation_text.split("\n")
            rows = [(line.split(":")[0].strip(), line.split(":")[1].strip()) for line in explanation_lines if ":" in line]
            st.table(pd.DataFrame(rows, columns=["Feature", "How does it impact?"]))

            # Campaign Suggestion
            prompt_campaign = f"""
            Based on these top features influencing adoption: {', '.join(top_feature_df['Feature'].tolist())},
            suggest 3 targeted marketing campaigns that a business team can run to increase premium subscriptions.
            Tie each campaign back to specific user behaviors in the features.
            """

            response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing strategist."},
                    {"role": "user", "content": prompt_campaign}
                ]
            )
            campaigns = response2["choices"][0]["message"]["content"]
            st.subheader("💡 Campaign Recommendations")
            st.markdown(campaigns)

    except Exception as e:
        st.error(f"There was a problem processing your file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin analysis.")

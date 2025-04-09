""# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import requests

# --- Page Config (must be first Streamlit command) ---
st.set_page_config(page_title="Customer Adopter Prediction", layout="wide")

# Load model and scaler
bundle = joblib.load("marketing_classifier_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# --- Hugging Face Inference API Setup ---
HF_API_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

# Define HF text generation call
HF_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

def query_hf_mistral(prompt, max_tokens=512):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens}
    }
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    try:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        else:
            return f"LLM error: unexpected response: {data}"
    except Exception as e:
        return f"LLM error: {str(e)}"


# --- Feature Name Mapping ---
feature_name_map = {
    "avg_friend_age": "Avg Age of Friends",
    "songsListened": "Songs Listened",
    "lovedTracks": "Loved Tracks",
    "age": "User Age",
    "subscriber_friend_cnt": "Friends Who Subscribed",
    "delta_songsListened": "Recent Listening Spike",
    "posts": "User Posts",
    "shouts": "Shouts Made",
    "male": "Is Male User"
}

# --- Custom Style ---
pink = "#f08ebc"

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
    .metric-box {{
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f8f8f8;
        margin-bottom: 1rem;
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
            st.subheader("✅ Campaign Summary & Insights")
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                st.markdown("### 📊 Campaign ROI Summary")
                cutoff = int(len(df_result) * top_k_percent / 100)
                top_customers = df_result.sort_values("Predicted_Probability", ascending=False).head(cutoff)
                n_targeted = len(top_customers)
                n_predicted_adopters = top_customers["Predicted_Adopter"].sum()

                total_cost = n_targeted * cost_per_customer
                total_revenue = n_predicted_adopters * revenue_per_conversion
                roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0

                st.metric("🎯 Targeted Customers", n_targeted)
                st.metric("📈 Expected Adopters", int(n_predicted_adopters))
                st.metric("💰 Estimated ROI", f"{roi:.2f}")

            with col2:
                st.markdown("### 🔍 Top Features Influencing Adoption")
                explainer = shap.Explainer(model, X_scaled)
                shap_values = explainer(X_scaled)

                # Compute SHAP impact
                shap_importance = shap_values.abs.mean(0).values
                feature_names_original = X_scaled.columns

                top_feature_df = pd.DataFrame({
                    "Feature": feature_names_original,
                    "Impact": shap_importance
                }).sort_values(by="Impact", ascending=False).head(5)

                # Apply readable names
                top_feature_df["Readable_Feature"] = top_feature_df["Feature"].apply(lambda x: feature_name_map.get(x, x))
                top5_llm_df = top_feature_df[["Readable_Feature", "Impact"]].rename(columns={"Readable_Feature": "Feature"})

                fig = plt.figure(figsize=(6, 3.5))
                shap_values.feature_names = [feature_name_map.get(name, name) for name in X_scaled.columns]  # Rename features in plot
                shap.plots.beeswarm(shap_values, max_display=10, show=False)
                st.pyplot(fig)

            st.divider()
            st.markdown("### 🧠 What Influences Adoption?")
            feature_text = top5_llm_df.to_string(index=False)
            llm_prompt = f"""
            Given the following top features ranked by their impact on customer adoption:
            {feature_text}

            For each feature, explain what user behavior it captures and how it might relate to adoption of a premium subscription.
            Provide only business-relevant insights.
            """
            llm_response = query_hf_mistral(llm_prompt)
            st.code(llm_response)  # Debug display

            if llm_response and "LLM error" not in llm_response:
                feature_explanations = llm_response.split("\n")
                clean_rows = [(line.split(":")[0].strip(), line.split(":")[1].strip()) for line in feature_explanations if ":" in line]
                feature_df = pd.DataFrame(clean_rows[:5], columns=["Feature", "How does it impact?"])
                st.table(feature_df)
            else:
                st.warning("LLM explanation could not be generated. Please try again later.")

            st.markdown("### 🎯 Campaign Recommendations")
            rec_prompt = f"""
            Based on these top features: {', '.join(top5_llm_df['Feature'].tolist())},
            suggest 3 campaign ideas that marketing professionals can run to increase premium subscriptions. 
            Make each suggestion concise, relevant, and tied to customer behavior.
            """

            campaign_response = query_hf_mistral(rec_prompt)
            if campaign_response and "LLM error" not in campaign_response:
                lines = [line.strip() for line in campaign_response.split("\n") if line.strip()]
                for line in lines:
                    st.markdown(f"- {line}")
            else:
                st.warning("LLM recommendation could not be generated. Please try again later.")

    except Exception as e:
        st.error(f"There was a problem processing your file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin analysis.")

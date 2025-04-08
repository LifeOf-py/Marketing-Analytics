# 🧠 Predicting Freemium to Premium Adoption 🎵

## 📌 Project Overview
Website XYZ, a music-listening social networking platform, operates on a freemium model — offering basic services for free, with additional premium features via a subscription. This project aims to predict which users are most likely to convert from free to premium subscribers within 6 months if targeted by a marketing campaign.

## 🎯 Business Objective
To identify high-potential customers likely to adopt premium features, enabling targeted promotional campaigns that increase ROI, reduce wasted outreach, and maximize customer value.

## 🔍 Problem Statement
Given user behavioral and social features, build a classifier to predict the probability of adoption if the user is included in the next marketing campaign.

## 🧪 Techniques Used
- Addressed class imbalance (few adopters)
- Nested Cross-Validation for robust model selection
- Hyperparameter tuning for XGBoost
- Threshold tuning for recall optimization
- SHAP explainability to uncover key drivers of adoption
- ROI simulation and LLM campaign suggestions

## 🧠 Final Model
- **Model**: XGBoost Classifier
- **Target Metric**: Maximized Recall (important for reaching all potential adopters)
- **Threshold**: Tuned to optimize lift and reduce false negatives
- **Top Model Features**:
  - Increase in songs listened over time
  - Count of loved tracks
  - Social network activity
  - Engagement spikes

## 📊 Interactive Dashboard (Streamlit)
Use the deployed dashboard to:
- Upload customer data
- Get live adoption predictions
- Simulate ROI with custom cost/revenue assumptions
- Visualize SHAP explanations and lift curves
- Receive LLM-based campaign suggestions

## 👉 Live App Link 
([Marketing Campaign - Customer Targeting Tool](https://marketing-decisions-mayank.streamlit.app))

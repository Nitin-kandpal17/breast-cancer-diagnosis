# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset and train model
@st.cache_resource
def load_model():
    df = pd.read_csv("data/breast_cancer.csv")
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    )
    model.fit(X, y)
    return model, X.columns, X.mean(), df

model, all_features, feature_means, df = load_model()

# Top SHAP-based features and tooltips
top_features = {
    "radius_mean": "Average distance from center to points on perimeter",
    "texture_mean": "Standard deviation of gray-scale values",
    "perimeter_mean": "Perimeter of the tumor",
    "area_mean": "Area of the tumor",
    "smoothness_mean": "Local variation in radius lengths",
    "concave_points_mean": "Number of concave portions of the contour",
    "compactness_mean": "Perimeter² / Area – 1.0"
}

# Sample input presets
samples = {
    "Sample: Benign": {
        "radius_mean": 11.42,
        "texture_mean": 17.7,
        "perimeter_mean": 73.34,
        "area_mean": 403.5,
        "smoothness_mean": 0.1028,
        "concave_points_mean": 0.0195,
        "compactness_mean": 0.0433
    },
    "Sample: Malignant": {
        "radius_mean": 20.5,
        "texture_mean": 22.0,
        "perimeter_mean": 130.0,
        "area_mean": 1200.0,
        "smoothness_mean": 0.12,
        "concave_points_mean": 0.15,
        "compactness_mean": 0.25
    }
}

# UI Header
st.title("🧪 Breast Cancer Diagnostic Tool")
st.markdown("Enter tumor feature values or use a sample input to predict whether it's **Malignant** or **Benign**.")

# Sidebar Input Section
st.sidebar.header("🔬 Tumor Feature Inputs")
sample_option = st.sidebar.selectbox(
    "Choose a sample input (optional):",
    ("Custom Input", "Sample: Benign", "Sample: Malignant")
)

user_input = {}

# Use actual dataset to set min/max/mean
for feature, tooltip in top_features.items():
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())

    if sample_option != "Custom Input":
        val = samples[sample_option][feature]
    else:
        val = float(df[feature].mean())

    val = st.sidebar.number_input(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=val,
        step=0.1,
        help=tooltip
    )
    user_input[feature] = val

# Prepare prediction input
input_df = pd.DataFrame([user_input])
for feature in all_features:
    if feature not in input_df.columns:
        input_df[feature] = feature_means[feature]
input_df = input_df[list(all_features)]  # Ensure correct order

# Predict
if st.button("🔍 Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    diagnosis = "🧨 Malignant" if prediction == 0 else "✅ Benign"
    confidence = probabilities[prediction] * 100

    st.subheader(f"Prediction: {diagnosis}")
    st.write(f"**Model Confidence:** {confidence:.2f}%")

    st.markdown("### 🔬 Prediction Breakdown:")
    st.write(f"Class 0 (Malignant): `{probabilities[0]:.2f}`")
    st.write(f"Class 1 (Benign): `{probabilities[1]:.2f}`")

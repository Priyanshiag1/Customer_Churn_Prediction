import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and features
model = pickle.load(open("customer.pkl.txt", "rb"))
feature_names = pickle.load(open("features.pkl.txt", "rb"))

# Helper: tenure group binning
def tenure_group(tenure):
    if tenure <= 12:
        return "1 - 12"
    elif tenure <= 24:
        return "13 - 24"
    elif tenure <= 36:
        return "25 - 36"
    elif tenure <= 48:
        return "37 - 48"
    elif tenure <= 60:
        return "49 - 60"
    else:
        return "61 - 72"

st.title("Customer Churn Prediction")

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=30.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=30.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input data
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "tenure_group": tenure_group(tenure)
    }
    df = pd.DataFrame([input_dict])

    # One-hot encoding for categorical variables
    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"
    ]
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    # Add missing columns (those present in training but not in this input)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure column order
    df_encoded = df_encoded[feature_names]

    # Predict
    pred = model.predict(df_encoded)[0]
    proba = model.predict_proba(df_encoded)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to churn! (Confidence: {proba*100:.2f}%)")
    else:
        st.success(f"✅ This customer is likely to stay. (Confidence: {(1-proba)*100:.2f}%)")

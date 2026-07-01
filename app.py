"""Streamlit demo for customer churn prediction."""

import streamlit as st

from src.save_model import load_artifacts

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer is likely to churn based on "
    "contract, billing, and service features."
)

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python main.py` first to train and save the model.")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

    with col2:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    import pandas as pd

    input_df = pd.DataFrame(
        [
            {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract,
                "PaymentMethod": payment_method,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "TechSupport": tech_support,
            }
        ]
    )

    features = preprocessor.transform(input_df)
    probability = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High churn risk — probability: {probability:.1%}")
    else:
        st.success(f"Likely to stay — churn probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "This demo uses the best model saved by the training pipeline. "
        "Run `streamlit run app.py` after training."
    )

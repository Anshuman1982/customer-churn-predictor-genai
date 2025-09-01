import streamlit as st 
import joblib 
import numpy as np
import pandas as pd 
import base64

# Setting Page Config 
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp{{
            background-image:url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment:fixed;
            background-position: center;
            background-repeat: no-repeat:
        }}

        .main{{
            background-color:rgba(255, 255, 255, 0.85)
            padding: 2rem;
            border-radius:10px
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local("bg.jpg")

#loding the model
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')


# title
st.markdown("<h1 style= 'text-align: center;'> Customer Churn Predictor</h1>", unsafe_allow_html=True)

#Input layout
col1 , col2 ,col3 =st.columns(3)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, step=10.0)
    
with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])


st.markdown("---")



#now preprocessing the raw data from user which can be understood by our modal
def preprocess_input(gender, senior_citizen, partner, tenure, monthly_charges,
                     total_charges, contract, paperless_billing, payment_method, model_columns):

    # Build raw input as a DataFrame
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    input_df = pd.DataFrame([input_dict])

    # Apply one-hot encoding to match training
    input_df = pd.get_dummies(input_df)

    # Reindex to match the model's expected input
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    return input_df


if st.button("Predict Churn"):
    input_data = preprocess_input(
        gender, senior_citizen, partner, tenure, monthly_charges,
        total_charges, contract, paperless_billing, payment_method, model_columns
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")



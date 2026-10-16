# app.py  -  Streamlit front end for the Wellness Tourism Package predictor

import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")


@st.cache_resource
def load_model():
    # config.json is written at deployment time and holds the model repo id
    model_repo = os.getenv("MODEL_REPO")
    if model_repo is None and os.path.exists("config.json"):
        with open("config.json") as f:
            model_repo = json.load(f)["model_repo"]

    model_file = hf_hub_download(repo_id=model_repo, filename="best_tourism_model_v1.joblib")
    meta_file = hf_hub_download(repo_id=model_repo, filename="model_meta.json")
    with open(meta_file) as f:
        meta = json.load(f)
    return joblib.load(model_file), meta


model, meta = load_model()
threshold = meta.get("threshold", 0.45)

st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Fill in the customer details below to find out how likely the customer is to "
    "buy the **Wellness Tourism Package** *before* the sales team makes the call."
)

# ---------------------------------------------------------------- inputs
st.subheader("Customer details")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"]
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    income = st.number_input("Monthly income", min_value=1000, max_value=200000,
                             value=22000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])

with c2:
    persons = st.number_input("Number of people visiting", 1, 10, 3)
    children = st.number_input("Children (below 5) visiting", 0, 5, 1)
    star = st.selectbox("Preferred property star rating", [3.0, 4.0, 5.0])
    trips = st.number_input("Average trips per year", 0, 30, 3)

st.subheader("Interaction details")
c3, c4 = st.columns(2)
with c3:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox(
        "Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    duration = st.number_input("Duration of pitch (minutes)", 1, 150, 15)
with c4:
    followups = st.number_input("Number of follow-ups", 0, 10, 4)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# the column names and order must be the same as the training data
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    st.metric("Probability of buying the package", f"{prob:.1%}")
    if prob >= threshold:
        st.success("✅ Likely to purchase - worth prioritising for the Wellness package campaign.")
    else:
        st.warning("❌ Unlikely to purchase - a lower priority lead for this campaign.")
    st.caption(f"Decision threshold used by the model: {threshold}")

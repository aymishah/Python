import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.write("Enter the car details to get the predicted selling price.")

model = load("./models/car_price_pipeline.pkl")

with st.form("pred_form"):
    name = st.text_input("Name", "Maruti Swift VXI")
    fuel = st.selectbox("Fuel", ["", "Petrol","Diesel","CNG","LPG","Electric"])
    seller_type = st.selectbox("Seller Type", ["", "Individual","Dealer","Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["", "Manual","Automatic"])
    owner = st.text_input("Owner", "First Owner")

    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015, step=1)
    km_driven = st.number_input("KM Driven", min_value=0, value=60000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=18.2, step=0.1)
    engine = st.number_input("Engine (cc)", min_value=0, value=1197, step=1)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=82.0, step=0.1)
    seats = st.number_input("Seats", min_value=1, max_value=10, value=5, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{
        "name": name, "fuel": fuel, "seller_type": seller_type,
        "transmission": transmission, "owner": owner, "year": year,
        "km_driven": km_driven, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats
    }])
    try:
        pred = model.predict(df)[0]
        st.success(f"Predicted Price: {pred:,.0f}")
    except Exception as e:
        st.error(str(e))
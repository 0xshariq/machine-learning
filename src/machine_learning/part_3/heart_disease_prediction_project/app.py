import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "logistic_heart_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler_heart.pkl")
expected_columns = joblib.load(BASE_DIR / "columns_heart.pkl")

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's information below to get a model prediction.")

# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=0,
            max_value=250,
            value=120
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=700,
            value=200
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:
        max_hr = st.slider("Maximum Heart Rate", 40, 220, 150)
        oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=-2.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"]
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["LVH", "Normal", "ST"]
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"]
        )

    st_slope = st.selectbox(
        "ST Slope",
        ["Down", "Flat", "Up"]
    )

    predict_button = st.form_submit_button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_button:
    # Start with all model features set to 0.
    input_data = {column: 0 for column in expected_columns}

    # Numeric features
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = fasting_bs
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # One-hot encoded features.
    # Reference categories:
    # Sex=F, ChestPainType=ASY, RestingECG=LVH,
    # ExerciseAngina=N, ST_Slope=Down.
    if sex == "M":
        input_data["Sex_M"] = 1

    if chest_pain != "ASY":
        input_data[f"ChestPainType_{chest_pain}"] = 1

    if resting_ecg != "LVH":
        input_data[f"RestingECG_{resting_ecg}"] = 1

    if exercise_angina == "Y":
        input_data["ExerciseAngina_Y"] = 1

    if st_slope != "Down":
        input_data[f"ST_Slope_{st_slope}"] = 1

    # Feature engineering used during training.
    input_data["MaxHR_ratio"] = max_hr / (220 - age)
    input_data["Has_ST_Depression"] = int(oldpeak > 0)

    # Exact feature order expected by the trained model.
    input_df = pd.DataFrame([input_data], columns=expected_columns)

    # Use the scaler fitted on the training data.
    scaled_input = scaler.transform(input_df)

    # Model prediction
    prediction = int(model.predict(scaled_input)[0])

    st.divider()

    if prediction == 1:
        st.error("⚠️ Model prediction: Heart Disease detected")
    else:
        st.success("✅ Model prediction: No Heart Disease detected")

    # Probability of class 1, if supported by the model.
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_input)[0][1])
        st.metric(
            "Predicted probability of Heart Disease",
            f"{probability * 100:.2f}%"
        )

    st.caption(
        "This prediction is produced by a machine-learning model and is "
        "not a medical diagnosis. Consult a qualified healthcare professional "
        "for medical advice."
    )

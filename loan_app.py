import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Loan Credit Risk App",
    layout="wide"
)

st.title("🏦 Credit Risk Prediction App")

# =========================
# Load Model & Tools
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_dict = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, le_dict, feature_names


model, scaler, le_dict, feature_names = load_model()

# =========================
# Sidebar Menu
# =========================
menu = st.sidebar.selectbox(
    "Menu",
    ["Beranda", "Prediksi Kredit"]
)

# =========================
# BERANDA
# =========================
if menu == "Beranda":
    st.markdown("""
    ### 📌 Aplikasi Prediksi Risiko Kredit
    Aplikasi ini digunakan untuk memprediksi **risiko kredit nasabah**
    menggunakan **Machine Learning (Random Forest)**.

    **Output:**
    - GOOD CREDIT → Risiko rendah
    - BAD CREDIT → Risiko tinggi / potensi default
    """)

# =========================
# PREDIKSI KREDIT
# =========================
elif menu == "Prediksi Kredit":

    st.subheader("🧾 Input Data Nasabah")

    col1, col2 = st.columns(2)

    with col1:
        status_checking = st.selectbox("Status Rekening", ["A11", "A12", "A13", "A14"])
        duration = st.number_input("Durasi Kredit (bulan)", 1, 120, 12)
        credit_history = st.selectbox("Riwayat Kredit", ["A30", "A31", "A32", "A33", "A34"])
        purpose = st.selectbox("Tujuan Kredit", ["A40", "A41", "A42", "A43", "A44", "A45"])
        credit_amount = st.number_input("Plafon Kredit", 100, 1000000, 5000)
        savings = st.selectbox("Tabungan", ["A61", "A62", "A63", "A64", "A65"])
        employment = st.selectbox("Lama Bekerja", ["A71", "A72", "A73", "A74", "A75"])
        installment_rate = st.slider("Cicilan (% pendapatan)", 1, 4, 2)
        personal_status = st.selectbox("Status Personal", ["A91", "A92", "A93", "A94", "A95"])
        other_debtors = st.selectbox("Penjamin", ["A101", "A102", "A103"])

    with col2:
        residence_since = st.slider("Lama Tinggal (tahun)", 1, 4, 2)
        property = st.selectbox("Properti", ["A121", "A122", "A123", "A124"])
        age = st.number_input("Umur", 18, 100, 30)
        other_installments = st.selectbox("Cicilan Lain", ["A141", "A142", "A143"])
        housing = st.selectbox("Status Rumah", ["A151", "A152", "A153"])
        existing_credits = st.slider("Jumlah Kredit Aktif", 1, 4, 1)
        job = st.selectbox("Pekerjaan", ["A171", "A172", "A173", "A174"])
        num_dependents = st.slider("Jumlah Tanggungan", 1, 2, 1)
        telephone = st.selectbox("Telepon", ["A191", "A192"])
        foreign_worker = st.selectbox("Foreign Worker", ["A201", "A202"])

    predict_button = st.button("🔍 Analisis Risiko Kredit")

    # =========================
    # PREDICTION (FIX FINAL)
    # =========================
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Analisis Risiko")

        input_data = {
            'status_checking': status_checking,
            'duration': duration,
            'credit_history': credit_history,
            'purpose': purpose,
            'credit_amount': credit_amount,
            'savings': savings,
            'employment': employment,
            'installment_rate': installment_rate,
            'personal_status': personal_status,
            'other_debtors': other_debtors,
            'residence_since': residence_since,
            'property': property,
            'age': age,
            'other_installments': other_installments,
            'housing': housing,
            'existing_credits': existing_credits,
            'job': job,
            'num_dependents': num_dependents,
            'telephone': telephone,
            'foreign_worker': foreign_worker
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical
        for col, le in le_dict.items():
            input_df[col] = le.transform(input_df[col].astype(str))

        # Scaling
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Output
        if prediction == 0:
            st.success("🟢 **GOOD CREDIT (Risiko Rendah)**")
            st.metric("Probabilitas Lancar", f"{prediction_proba[0]*100:.1f}%")
        else:
            st.error("🔴 **BAD CREDIT (Risiko Tinggi / Default)**")
            st.metric("Probabilitas Default", f"{prediction_proba[1]*100:.1f}%")

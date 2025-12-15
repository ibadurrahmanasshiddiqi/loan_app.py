import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Prediksi Kemampuan Bayar Angsuran",
    page_icon="💰",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Load German Credit Data from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    columns = [
        'status_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installments', 'housing',
        'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
    ]
    
    df = pd.read_csv(url, sep=' ', header=None, names=columns)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # Encode categorical features
    le_dict = {}
    categorical_cols = ['status_checking', 'credit_history', 'purpose', 'savings', 
                       'employment', 'personal_status', 'other_debtors', 'property',
                       'other_installments', 'housing', 'job', 'telephone', 'foreign_worker']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    return model, scaler, df, X, y, X_test, y_test, y_pred, accuracy_test, accuracy_train, le_dict, columns[:-1]

try:
    model, scaler, df, X, y, X_test, y_test, y_pred, accuracy_test, accuracy_train, le_dict, feature_names = load_and_prepare_data()
    st.sidebar.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Sidebar Menu
st.sidebar.title("💼 Menu Navigasi")
menu = st.sidebar.selectbox("Pilih Menu:", 
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚠️ Faktor Risiko", "🔬 Prediksi Kredit"])

# Feature Importance (calculate once)
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Menu: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown(f"""
    ## Sistem Prediksi Risiko Kredit
    
    ### Versi 1.0
    
    **Teknologi:** Python, Streamlit, Scikit-learn, Plotly
    
    ### Dataset
    **German Credit Data** dari UCI Machine Learning Repository
    - **Jumlah Data:** {len(df)} nasabah
    - **Jumlah Fitur:** {len(feature_names)} parameter
    - **Good Credit:** {(df['target'] == 0).sum()} ({(df['target'] == 0).sum()/len(df)*100:.1f}%)
    - **Bad Credit:** {(df['target'] == 1).sum()} ({(df['target'] == 1).sum()/len(df)*100:.1f}%)
    
    ### Model Machine Learning
    - **Algoritma:** Random Forest (200 trees)
    - **Akurasi Training:** {accuracy_train:.2%}
    - **Akurasi Testing:** {accuracy_test:.2%}
    - **Class Balancing:** Balanced weights
    
    ### Disclaimer
    ⚠️ Aplikasi ini untuk **analisis dan referensi**. Keputusan kredit harus melibatkan verifikasi dokumen, wawancara, dan analisis mendalam.
    """)

# Menu: Pengenalan
elif menu == "📚 Pengenalan Aplikasi":
    st.title("📚 Pengenalan Aplikasi")
    st.markdown("""
    ## Sistem Prediksi Risiko Kredit
    
    Aplikasi Machine Learning untuk menganalisis risiko kredit menggunakan **German Credit Data** (UCI).
    
    ### 🎯 Tujuan
    1. **Manajemen Risiko** - Identifikasi nasabah berisiko tinggi
    2. **Efisiensi** - Percepat analisis kredit
    3. **Data-Driven** - Keputusan berbasis data objektif
    4. **Early Warning** - Deteksi dini potensi default
    
    ### 🔬 Cara Kerja
    1. Input 20 parameter nasabah
    2. Data di-encode dan dinormalisasi
    3. Random Forest menganalisis pola
    4. Prediksi Good/Bad credit
    5. Probabilitas dan rekomendasi
    
    ### 📊 20 Parameter Analisis
    - Status rekening checking, durasi, jumlah kredit
    - Riwayat kredit, tujuan, tabungan
    - Status pekerjaan, usia, tanggungan
    - Property, housing, dan 10 parameter lainnya
    
    ### 🚀 Cara Menggunakan
    1. Menu **"🔬 Prediksi Kredit"**
    2. Input data nasabah
    3. Klik **"Analisis Risiko"**
    4. Review hasil dan rekomendasi
    """)

# Menu: Faktor Risiko
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Faktor Risiko Kredit")
    
    st.subheader("📊 Top 10 Fitur Paling Penting")
    fig_imp = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                     orientation='h', color='Importance', color_continuous_scale='Viridis')
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 🏦 Status Checking
        - **< 0 DM:** Overdraft - Risiko TINGGI 🔴
        - **0-200 DM:** Minimal - Risiko Sedang 🟠
        - **>= 200 DM:** Baik - Risiko Rendah 🟢
        
        ### 2. ⏱️ Durasi Kredit
        - **< 12 bulan:** Low risk
        - **12-24 bulan:** Medium risk
        - **> 24 bulan:** High risk
        
        ### 3. 💵 Jumlah Kredit
        - **< 2,500 DM:** Low
        - **2,500-5,000 DM:** Medium
        - **> 5,000 DM:** High
        
        ### 4. 💰 Tabungan
        - **< 100 DM:** Risiko Tinggi 🔴
        - **100-500 DM:** Sedang 🟠
        - **>= 1000 DM:** Baik ✅
        
        ### 5. 💼 Pekerjaan
        - **Unemployed:** Sangat Tinggi 🔴
        - **< 1 year:** Tinggi 🟠
        - **>= 7 years:** Stabil ✅
        """)
    
    with col2:
        st.markdown("""
        ### 6. 🎂 Usia
        - **< 25:** Belum stabil
        - **25-45:** Peak earning (optimal)
        - **> 60:** Mendekati pensiun
        
        ### 7. 📊 Tingkat Cicilan
        - **1-2%:** Ringan
        - **3%:** Sedang
        - **4%+:** Berat
        
        ### 8. 🏠 Property
        - **Real Estate:** Rendah ✅
        - **Car/Other:** Sedang
        - **No Property:** Tinggi 🔴
        
        ### 9. 👨‍👩‍👧 Tanggungan
        - **0-1:** Rendah
        - **2:** Sedang
        - **3+:** Tinggi
        
        ### 10. 📜 Riwayat Kredit
        - **Critical:** Sangat Tinggi 🔴
        - **Delay:** Tinggi 🟠
        - **Paid Duly:** Rendah ✅
        """)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Durasi", f"{df['duration'].mean():.1f} bulan")
    col2.metric("Rata-rata Jumlah", f"{df['credit_amount'].mean():.0f} DM")
    col3.metric("Rata-rata Usia", f"{df['age'].mean():.1f} tahun")

# Menu: Prediksi
elif menu == "🔬 Prediksi Kredit":
    st.title("🔬 Sistem Prediksi Risiko Kredit")
    
    st.sidebar.header("📋 Data Nasabah")
    
    # Inputs
    status_checking = st.sidebar.selectbox("Status Checking", ['A11', 'A12', 'A13', 'A14'],
        format_func=lambda x: {'A11': '< 0 DM', 'A12': '0-200 DM', 'A13': '>= 200 DM', 'A14': 'Tidak Ada'}[x])
    
    duration = st.sidebar.slider("Durasi (bulan)", 6, 72, 24)
    credit_amount = st.sidebar.number_input("Jumlah Kredit (DM)", 250, 20000, 3000, 250)
    
    credit_history = st.sidebar.selectbox("Riwayat Kredit", ['A30', 'A31', 'A32', 'A33', 'A34'],
        format_func=lambda x: {'A30': 'No credits', 'A31': 'All paid', 'A32': 'Existing paid',
                               'A33': 'Delay', 'A34': 'Critical'}[x])
    
    purpose = st.sidebar.selectbox("Tujuan", ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
        format_func=lambda x: {'A40': 'Mobil Baru', 'A41': 'Mobil Bekas', 'A42': 'Furniture',
                               'A43': 'Radio/TV', 'A44': 'Alat RT', 'A45': 'Perbaikan',
                               'A46': 'Pendidikan', 'A48': 'Retraining', 'A49': 'Bisnis', 'A410': 'Lainnya'}[x])
    
    savings = st.sidebar.selectbox("Tabungan", ['A61', 'A62', 'A63', 'A64', 'A65'],
        format_func=lambda x: {'A61': '< 100 DM', 'A62': '100-500 DM', 'A63': '500-1000 DM',
                               'A64': '>= 1000 DM', 'A65': 'None'}[x])
    
    employment = st.sidebar.selectbox("Pekerjaan", ['A71', 'A72', 'A73', 'A74', 'A75'],
        format_func=lambda x: {'A71': 'Unemployed', 'A72': '< 1 year', 'A73': '1-4 years',
                               'A74': '4-7 years', 'A75': '>= 7 years'}[x])
    
    installment_rate = st.sidebar.slider("Tingkat Cicilan (%)", 1, 4, 2)
    age = st.sidebar.slider("Usia", 19, 75, 35)
    
    personal_status = st.sidebar.selectbox("Status Personal", ['A91', 'A92', 'A93', 'A94', 'A95'],
        format_func=lambda x: {'A91': 'Male divorced', 'A92': 'Female', 'A93': 'Male single',
                               'A94': 'Male married', 'A95': 'Female single'}[x])
    
    num_dependents = st.sidebar.selectbox("Tanggungan", [1, 2])
    residence_since = st.sidebar.slider("Lama Tinggal", 1, 4, 2)
    
    property_val = st.sidebar.selectbox("Property", ['A121', 'A122', 'A123', 'A124'],
        format_func=lambda x: {'A121': 'Real estate', 'A122': 'Savings/Insurance',
                               'A123': 'Car', 'A124': 'No property'}[x])
    
    other_debtors = st.sidebar.selectbox("Other Debtors", ['A101', 'A102', 'A103'],
        format_func=lambda x: {'A101': 'None', 'A102': 'Co-applicant', 'A103': 'Guarantor'}[x])
    
    other_installments = st.sidebar.selectbox("Other Installments", ['A141', 'A142', 'A143'],
        format_func=lambda x: {'A141': 'Bank', 'A142': 'Stores', 'A143': 'None'}[x])
    
    housing = st.sidebar.selectbox("Housing", ['A151', 'A152', 'A153'],
        format_func=lambda x: {'A151': 'Rent', 'A152': 'Own', 'A153': 'Free'}[x])
    
    existing_credits = st.sidebar.selectbox("Kredit Existing", [1, 2, 3, 4])
    
    job = st.sidebar.selectbox("Tipe Pekerjaan", ['A171', 'A172', 'A173', 'A174'],
        format_func=lambda x: {'A171': 'Unemployed', 'A172': 'Unskilled', 
                               'A173': 'Skilled', 'A174': 'Management'}[x])
    
    telephone = st.sidebar.selectbox("Telepon", ['A191', 'A192'],
        format_func=lambda x: {'A191': 'None', 'A192': 'Yes'}[x])
    
    foreign_worker = st.sidebar.selectbox("Pekerja Asing", ['A201', 'A202'],
        format_func=lambda x: {'A201': 'Yes', 'A202': 'No'}[x])
    
    predict_button = st.sidebar.button("🔍 Analisis Risiko", use_container_width=True)
    
    # Main display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Akurasi Test", f"{accuracy_test:.1%}")
        col_m2.metric("Akurasi Train", f"{accuracy_train:.1%}")
        col_m3.metric("Total Data", len(df))
        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, labels=dict(x="Prediksi", y="Aktual"),
                           x=['Good', 'Bad'], y=['Good', 'Bad'], text_auto=True,
                           color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribusi")
        counts = df['target'].value_counts()
        fig_dist = go.Figure(data=[go.Pie(labels=['Good', 'Bad'], values=[counts[0], counts[1]],
                                           hole=0.4, marker=dict(colors=['#2ecc71', '#e74c3c']))])
        st.plotly_chart(fig_dist, use_container_width=True)
        st.metric("Good Rate", f"{counts[0]/len(df)*100:.1f}%")
        st.metric("Bad Rate", f"{counts[1]/len(df)*100:.1f}%")
    
    # Prediction
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Analisis")
        
        input_data = {
            'status_checking': status_checking, 'duration': duration, 'credit_history': credit_history,
            'purpose': purpose, 'credit_amount': credit_amount, 'savings': savings,
            'employment': employment, 'installment_rate': installment_rate,
            'personal_status': personal_status, 'other_debtors': other_debtors,
            'residence_since': residence_since, 'property': property_val, 'age': age,
            'other_installments': other_installments, 'housing': housing,
            'existing_credits': existing_credits, 'job': job, 'num_dependents': num_dependents,
            'telephone': telephone, 'foreign_worker': foreign_worker
        }
        
        input_encoded = []
        categorical_cols = ['status_checking', 'credit_history', 'purpose', 'savings', 
                           'employment', 'personal_status', 'other_debtors', 'property',
                           'other_installments', 'housing', 'job', 'telephone', 'foreign_worker']
        
        for col in feature_names:
            if col in categorical_cols:
                input_encoded.append(le_dict[col].transform([input_data[col]])[0])
            else:
                input_encoded.append(input_data[col])
        
        input_scaled = scaler.transform([input_encoded])
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 0:
                st.success("✅ GOOD CREDIT - Risiko Rendah")
                st.markdown(f"**Probabilitas Good:** {proba[0]:.1%}")
                st.markdown(f"**Probabilitas Bad:** {proba[1]:.1%}")
                st.markdown("""
                ### ✅ Rekomendasi: DISETUJUI
                - Kredit dapat disetujui
                - Rate normal
                - Monitoring standar
                - Profil kredit baik
                """)
            else:
                st.error("⚠️ BAD CREDIT - Risiko Tinggi")
                st.markdown(f"**Probabilitas Good:** {proba[0]:.1%}")
                st.markdown(f"**Probabilitas Bad:** {proba[1]:.1%}")
                st.markdown("""
                ### ⛔ Rekomendasi: REVIEW MENDALAM
                - Perlu analisis tambahan
                - Pertimbangkan penolakan
                - Jika disetujui: jaminan wajib
                - Interest rate lebih tinggi
                - Monitoring ketat
                """)
        
        st.markdown("---")
        st.subheader("📊 Probabilitas")
        fig_prob = go.Figure(data=[go.Bar(x=['Good', 'Bad'], y=proba,
                                          text=[f'{p:.1%}' for p in proba],
                                          marker=dict(color=['#2ecc71', '#e74c3c']))])
        fig_prob.update_layout(yaxis=dict(range=[0, 1]), showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Analisis Faktor")
            if status_checking == 'A11':
                st.error("❌ Checking: < 0 DM (BAHAYA)")
            elif status_checking == 'A13':
                st.success("✅ Checking: >= 200 DM (BAIK)")
            else:
                st.warning("⚠️ Checking: Perlu perhatian")
            
            if duration <= 12:
                st.success(f"✅ Durasi: {duration} bulan (Pendek)")
            elif duration > 36:
                st.error(f"❌ Durasi: {duration} bulan (Terlalu panjang)")
            else:
                st.info(f"ℹ️ Durasi: {duration} bulan")
        
        with col2:
            st.markdown("### 📋 Ringkasan")
            st.markdown(f"""
            - Jumlah: {credit_amount} DM
            - Durasi: {duration} bulan
            - Cicilan: {installment_rate}%
            - Usia: {age} tahun
            - Tanggungan: {num_dependents}
            - Kredit Existing: {existing_credits}
            """)
        
        st.markdown("---")
        st.subheader("📈 Top Fitur Penting")
        fig_top = px.bar(feature_importance.head(8), x='Importance', y='Feature',
                         orientation='h', color='Importance', color_continuous_scale='Blues')
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)

# Menu: Beranda
else:
    st.title("💰 Sistem Prediksi Risiko Kredit")
    st.markdown("Machine Learning dengan **German Credit Data** (UCI)")
    
    col1, col2, col3 = st.columns(3)
    col1.info(f"### 🎯 Akurat\nTesting: **{accuracy_test:.1%}**\nTraining: **{accuracy_train:.1%}**")
    col2.success("### 📊 Dataset Real\nGerman Credit (UCI)\n1000 nasabah, 20 fitur")
    col3.warning("### 🚀 Random Forest\n200 trees\nBalanced weights")
    
    st.markdown("---")
    st.subheader("🚀 Panduan Cepat")
    
    col1, col2 = st.columns(2)
    col1.markdown("""
    ### 📖 Pengguna Baru
    1. Baca **Pengenalan**
    2. Pelajari **Faktor Risiko**
    3. Lakukan **Prediksi**
    """)
    col2.markdown("""
    ### ⚡ Analis
    1. Menu **Prediksi Kredit**
    2. Input 20 parameter
    3. Klik **Analisis Risiko**
    """)
    
    st.markdown("---")
    st.subheader("📊 Statistik")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy_test:.1%}")
    col2.metric("Total Data", len(df))
    col3.metric("Good Rate", f"{(df['target']==0).sum()/len(df)*100:.1f}%")
    col4.metric("Bad Rate", f"{(df['target']==1).sum()/len(df)*100:.1f}%")
    
    st.markdown("---")
    st.subheader("📈 Top 5 Fitur Penting")
    fig_top5 = px.bar(feature_importance.head(5), x='Importance', y='Feature',
                      orientation='h', color='Importance', color_continuous_scale='Viridis')
    fig_top5.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_top5, use_container_width=True)
    
    st.error("""
    ### ⚠️ DISCLAIMER
    Model menggunakan **German Credit Data (UCI)** untuk analisis.
    Keputusan kredit final harus melibatkan verifikasi, wawancara, dan analisis mendalam.
    Prediksi ML adalah **alat bantu**, bukan pengganti judgment profesional.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center'><p>🏦 German Credit Analysis | © 2024</p></div>", 
            unsafe_allow_html=True)
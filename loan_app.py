import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    
    # Column names based on UCI documentation
    columns = [
        'status_checking', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installments', 'housing',
        'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
    ]
    
    df = pd.read_csv(url, sep=' ', header=None, names=columns)
    
    # Target: 1 = Good, 2 = Bad -> Convert to 0 = Good, 1 = Bad (default)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # Create interpretable features for display
    df_display = df.copy()
    
    # Convert codes to readable format
    status_map = {'A11': '< 0 DM', 'A12': '0-200 DM', 'A13': '>= 200 DM', 'A14': 'Tidak Ada'}
    purpose_map = {'A40': 'Mobil Baru', 'A41': 'Mobil Bekas', 'A42': 'Furniture', 'A43': 'TV/Radio', 
                   'A44': 'Alat Rumah Tangga', 'A45': 'Perbaikan', 'A46': 'Pendidikan', 
                   'A48': 'Pelatihan', 'A49': 'Bisnis', 'A410': 'Lainnya'}
    
    # Encode categorical features untuk modeling
    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    categorical_cols = ['status_checking', 'credit_history', 'purpose', 'savings', 
                       'employment', 'personal_status', 'other_debtors', 'property',
                       'other_installments', 'housing', 'job', 'telephone', 'foreign_worker']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Select features
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with better parameters
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    return model, scaler, df, df_display, X, y, X_test, y_test, y_pred, accuracy_test, accuracy_train, le_dict, columns[:-1]

# Load data and model
try:
    model, scaler, df, df_display, X, y, X_test, y_test, y_pred, accuracy_test, accuracy_train, le_dict, feature_names = load_and_prepare_data()
    st.sidebar.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Menu
st.sidebar.title("💼 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚠️ Faktor Risiko", "🔬 Prediksi Kredit"]
)

# Menu: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    
    st.markdown(f"""
    ## Sistem Prediksi Risiko Kredit
    
    ### Versi 1.0
    
    **Dikembangkan oleh:** Tim Risk Management & Data Science
    
    **Teknologi yang Digunakan:**
    - Python 3.x
    - Streamlit (Framework Web)
    - Scikit-learn (Machine Learning)
    - Random Forest Classifier
    - Plotly (Visualisasi Data)
    - Pandas & NumPy (Pengolahan Data)
    
    ### Tentang Dataset
    Dataset yang digunakan adalah **German Credit Data** dari UCI Machine Learning Repository, dataset standar industri untuk credit risk modeling.
    
    **Sumber:** UCI Machine Learning Repository
    
    **Jumlah Data:** {len(df)} nasabah
    
    **Jumlah Fitur:** {len(feature_names)} parameter
    
    **Target:** Prediksi risiko kredit (Good/Bad)
    
    **Distribusi:**
    - Good Credit (Lancar): {(df['target'] == 0).sum()} nasabah ({(df['target'] == 0).sum()/len(df)*100:.1f}%)
    - Bad Credit (Default): {(df['target'] == 1).sum()} nasabah ({(df['target'] == 1).sum()/len(df)*100:.1f}%)
    
    ### Model Machine Learning
    - **Algoritma:** Random Forest Classifier (200 trees)
    - **Akurasi Training:** {accuracy_train:.2%}
    - **Akurasi Testing:** {accuracy_test:.2%}
    - **Preprocessing:** Label Encoding + Standardisasi
    - **Class Balancing:** Balanced class weights
    
    ### Fitur Dataset
    Dataset mencakup informasi:
    - Status rekening checking
    - Durasi kredit (bulan)
    - Riwayat kredit
    - Tujuan kredit
    - Jumlah kredit
    - Status tabungan
    - Status pekerjaan
    - Tingkat cicilan
    - Status personal & tanggungan
    - Dan 11 fitur lainnya
    
    ### Disclaimer
    ⚠️ **Penting:** Aplikasi ini untuk tujuan **analisis risiko dan referensi internal**. Keputusan kredit harus mempertimbangkan:
    - Verifikasi dokumen lengkap
    - Wawancara langsung
    - Site visit
    - Analisis mendalam
    - Pertimbangan aspek karakter
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Gunakan menu sidebar untuk navigasi ke berbagai fitur aplikasi.")

# Menu: Pengenalan Aplikasi
elif menu == "📚 Pengenalan Aplikasi":
    st.title("📚 Pengenalan Aplikasi")
    
    st.markdown("""
    ## Apa itu Sistem Prediksi Risiko Kredit?
    
    Sistem Prediksi Risiko Kredit adalah aplikasi berbasis **Machine Learning** yang dirancang untuk membantu lembaga keuangan menganalisis risiko kredit menggunakan German Credit Data - dataset standar industri dari UCI.
    
    ### 🎯 Tujuan Aplikasi
    
    1. **Manajemen Risiko:** Mengidentifikasi nasabah berisiko tinggi
    2. **Efisiensi Proses:** Mempercepat analisis kredit
    3. **Data-Driven Decision:** Keputusan berbasis data objektif
    4. **Early Warning System:** Deteksi dini potensi default
    5. **Benchmark Industry:** Menggunakan dataset standar UCI
    
    ### 🔬 Cara Kerja Aplikasi
    
    1. **Input Data:** Masukkan 20 parameter nasabah
    2. **Preprocessing:** Data di-encode dan dinormalisasi
    3. **Analisis ML:** Random Forest menganalisis pola risiko
    4. **Prediksi:** Klasifikasi Good/Bad credit risk
    5. **Probabilitas:** Tingkat kepercayaan prediksi
    6. **Rekomendasi:** Saran tindakan berdasarkan hasil
    
    ### 📊 Parameter yang Dianalisis
    
    **20 Parameter** meliputi:
    
    **Data Finansial:**
    - Status rekening checking
    - Durasi kredit (bulan)
    - Jumlah kredit (DM)
    - Tingkat cicilan
    - Status tabungan
    - Kredit yang sudah ada
    
    **Data Demografis & Employment:**
    - Usia
    - Status pekerjaan
    - Lama bekerja
    - Jumlah tanggungan
    - Status kepemilikan rumah
    - Telephone (landline)
    
    **Data Kredit History:**
    - Riwayat kredit
    - Tujuan kredit
    - Other debtors/guarantors
    - Other installment plans
    - Property ownership
    
    ### 📈 Kategori Risiko
    
    | Kategori | Prediksi | Tindakan |
    |----------|----------|----------|
    | **🟢 Good Credit** | Tidak akan default | Disetujui dengan rate normal |
    | **🔴 Bad Credit** | Berpotensi default | Ditolak atau persyaratan ketat |
    
    ### ✨ Keunggulan Aplikasi
    
    - ✅ **Dataset Real:** German Credit Data (standar industri)
    - ✅ **Akurasi Tinggi:** Model dengan performa optimal
    - ✅ **Random Forest:** Ensemble learning untuk akurasi lebih baik
    - ✅ **Balanced:** Handle imbalanced data
    - ✅ **Interpretable:** Feature importance analysis
    - ✅ **User-Friendly:** Interface intuitif
    
    ### 🚀 Cara Menggunakan
    
    1. Pilih menu **"🔬 Prediksi Kredit"**
    2. Input semua parameter nasabah di sidebar
    3. Klik **"🔍 Analisis Risiko"**
    4. Review hasil prediksi dan probabilitas
    5. Ikuti rekomendasi yang diberikan
    
    ### 💡 Best Practices
    
    - ✅ Pastikan data akurat dan terverifikasi
    - ✅ Gunakan data terbaru (< 30 hari)
    - ✅ Cross-check dengan dokumen
    - ✅ Kombinasikan dengan analisis kualitatif
    - ✅ Update berkala untuk monitoring
    """)
    
    st.success("✅ Siap menggunakan aplikasi? Pilih menu **'🔬 Prediksi Kredit'** untuk mulai!")

# Menu: Faktor Risiko
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Faktor Risiko Kredit")
    
    st.markdown("""
    ## Memahami Faktor Risiko dalam German Credit Data
    
    Dataset ini menganalisis 20 faktor yang mempengaruhi risiko kredit berdasarkan data historis dari Jerman.
    """)
    
    # Feature Importance
    st.subheader("📊 Tingkat Kepentingan Fitur")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Fitur Paling Penting',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Faktor Risiko Detail
    st.subheader("💰 Faktor Risiko Utama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 🏦 Status Rekening Checking
        **Indikator paling penting**
        
        - **< 0 DM:** Overdraft - Risiko SANGAT TINGGI 🔴
        - **0-200 DM:** Saldo minimal - Risiko Tinggi 🟠
        - **>= 200 DM:** Saldo baik - Risiko Rendah 🟢
        - **Tidak Ada:** No checking account - Perhatian 🟡
        
        **Mengapa Penting:**
        - Menunjukkan cash flow management
        - Indikator financial discipline
        - Prediktor kuat kemampuan bayar
        
        ### 2. ⏱️ Durasi Kredit
        **Tenor kredit dalam bulan**
        
        - **< 12 bulan:** Risiko rendah (short-term)
        - **12-24 bulan:** Risiko sedang
        - **24-36 bulan:** Risiko tinggi
        - **> 36 bulan:** Risiko sangat tinggi
        
        **Pertimbangan:**
        - Semakin lama, semakin banyak uncertainty
        - Life events bisa terjadi
        - Economic cycles berubah
        
        ### 3. 📜 Riwayat Kredit
        **Track record pembayaran**
        
        - **Critical Account:** Risiko SANGAT TINGGI 🔴
        - **Delay in Past:** Risiko Tinggi 🟠
        - **Paid Duly:** Risiko Rendah 🟢
        - **No Credits:** New customer - perlu hati-hati
        
        ### 4. 💵 Jumlah Kredit
        **Credit amount dalam Deutsche Mark**
        
        - **< 2,500 DM:** Low risk
        - **2,500-5,000 DM:** Medium risk
        - **5,000-10,000 DM:** High risk
        - **> 10,000 DM:** Very high risk
        
        **Risiko:**
        - Exposure lebih besar
        - Recovery lebih sulit
        - Impact lebih besar jika default
        """)
    
    with col2:
        st.markdown("""
        ### 5. 💰 Status Tabungan
        **Savings account/bonds**
        
        - **< 100 DM:** Risiko Tinggi 🔴
        - **100-500 DM:** Risiko Sedang 🟠
        - **500-1000 DM:** Risiko Rendah 🟢
        - **>= 1000 DM:** Sangat Baik ✅
        
        **Emergency Buffer:**
        - Tabungan = safety net
        - Bisa cover unexpected expenses
        - Mengurangi risiko default
        
        ### 6. 💼 Status Pekerjaan
        **Employment duration**
        
        - **Unemployed:** Risiko SANGAT TINGGI 🔴
        - **< 1 year:** Risiko Tinggi 🟠
        - **1-4 years:** Risiko Sedang 🟡
        - **4-7 years:** Risiko Rendah 🟢
        - **>= 7 years:** Sangat Stabil ✅
        
        ### 7. 📊 Tingkat Cicilan
        **Installment rate (% of disposable income)**
        
        - **1%:** Sangat ringan
        - **2%:** Ringan
        - **3%:** Sedang
        - **4%+:** Berat - Risiko tinggi
        
        ### 8. 🎂 Usia
        **Age in years**
        
        - **< 25:** Risiko Tinggi (belum stabil)
        - **25-45:** Risiko Rendah (peak earning)
        - **45-60:** Risiko Sedang (stabil tapi...)
        - **> 60:** Risiko Tinggi (mendekati pensiun)
        
        ### 9. 🏠 Property Ownership
        **Jenis aset yang dimiliki**
        
        - **Real Estate:** Risiko RENDAH ✅
        - **Building Society/Life Insurance:** Sedang
        - **Car/Other:** Tinggi
        - **No Property:** Sangat Tinggi 🔴
        
        ### 10. 👨‍👩‍👧 Jumlah Tanggungan
        **Number of dependents**
        
        - **0-1:** Risiko rendah
        - **2:** Risiko sedang
        - **3+:** Risiko tinggi (beban besar)
        """)
    
    st.markdown("---")
    
    # Statistics from actual data
    st.subheader("📈 Statistik Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rata-rata Durasi", f"{df['duration'].mean():.1f} bulan")
        st.metric("Rata-rata Jumlah Kredit", f"{df['credit_amount'].mean():.0f} DM")
    
    with col2:
        st.metric("Rata-rata Usia", f"{df['age'].mean():.1f} tahun")
        st.metric("Median Installment Rate", f"{df['installment_rate'].median():.0f}%")
    
    with col3:
        good_rate = (df['target'] == 0).sum() / len(df) * 100
        st.metric("Good Credit Rate", f"{good_rate:.1f}%")
        st.metric("Bad Credit Rate", f"{100-good_rate:.1f}%")
    
    st.markdown("---")
    
    st.info("""
    ### 💡 Tips Pencegahan Default
    
    **Untuk Lembaga Keuangan:**
    1. ✅ Fokus pada checking account status - prediktor #1
    2. ✅ Limit durasi kredit maksimal 24 bulan untuk high-risk
    3. ✅ Require higher down payment untuk no savings
    4. ✅ Cross-check employment stability
    5. ✅ Prioritize applicants dengan property ownership
    
    **Untuk Nasabah:**
    1. ✅ Maintain positive checking account balance
    2. ✅ Build savings (minimal 6 bulan expenses)
    3. ✅ Establish good credit history
    4. ✅ Keep employment stable (minimum 1 tahun)
    5. ✅ Apply for reasonable loan amount
    6. ✅ Avoid high installment rates (< 3% ideal)
    """)

# Menu: Prediksi Kredit
elif menu == "🔬 Prediksi Kredit":
    st.title("🔬 Sistem Prediksi Risiko Kredit")
    st.markdown("""
    Masukkan data nasabah berdasarkan German Credit Data format untuk analisis risiko kredit.
    """)
    
    # Sidebar inputs
    st.sidebar.header("📋 Data Nasabah")
    st.sidebar.markdown("**Informasi Finansial**")
    
    status_checking = st.sidebar.selectbox(
        "Status Rekening Checking",
        options=['A11', 'A12', 'A13', 'A14'],
        format_func=lambda x: {'A11': '< 0 DM (Overdraft)', 'A12': '0-200 DM', 
                               'A13': '>= 200 DM', 'A14': 'Tidak Ada'}[x]
    )
    
    duration = st.sidebar.slider("Durasi Kredit (bulan)", 6, 72, 24)
    
    credit_amount = st.sidebar.number_input(
        "Jumlah Kredit (DM)",
        min_value=250,
        max_value=20000,
        value=3000,
        step=250
    )
    
    credit_history = st.sidebar.selectbox(
        "Riwayat Kredit",
        options=['A30', 'A31', 'A32', 'A33', 'A34'],
        format_func=lambda x: {
            'A30': 'No credits / All paid',
            'A31': 'All paid at this bank',
            'A32': 'Existing paid duly',
            'A33': 'Delay in past',
            'A34': 'Critical account'
        }[x]
    )
    
    purpose = st.sidebar.selectbox(
        "Tujuan Kredit",
        options=['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
        format_func=lambda x: {
            'A40': 'Mobil Baru', 'A41': 'Mobil Bekas', 'A42': 'Furniture',
            'A43': 'Radio/TV', 'A44': 'Alat Rumah Tangga', 'A45': 'Perbaikan',
            'A46': 'Pendidikan', 'A48': 'Retraining', 'A49': 'Bisnis', 'A410': 'Lainnya'
        }[x]
    )
    
    savings = st.sidebar.selectbox(
        "Status Tabungan",
        options=['A61', 'A62', 'A63', 'A64', 'A65'],
        format_func=lambda x: {
            'A61': '< 100 DM', 'A62': '100-500 DM',
            'A63': '500-1000 DM', 'A64': '>= 1000 DM', 'A65': 'Unknown/None'
        }[x]
    )
    
    employment = st.sidebar.selectbox(
        "Status Pekerjaan",
        options=['A71', 'A72', 'A73', 'A74', 'A75'],
        format_func=lambda x: {
            'A71': 'Unemployed', 'A72': '< 1 year',
            'A73': '1-4 years', 'A74': '4-7 years', 'A75': '>= 7 years'
        }[x]
    )
    
    installment_rate = st.sidebar.slider("Tingkat Cicilan (% pendapatan)", 1, 4, 2)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Informasi Personal**")
    
    age = st.sidebar.slider("Usia (tahun)", 19, 75, 35)
    
    personal_status = st.sidebar.selectbox(
        "Status Personal",
        options=['A91', 'A92', 'A93', 'A94', 'A95'],
        format_func=lambda x: {
            'A91': 'Male : divorced/separated',
            'A92': 'Female : divorced/separated/married',
            'A93': 'Male : single',
            'A94': 'Male : married/widowed',
            'A95': 'Female : single'
        }[x]
    )
    
    num_dependents = st.sidebar.selectbox("Jumlah Tanggungan", [1, 2])
    
    residence_since = st.sidebar.slider("Lama Tinggal (tahun)", 1, 4, 2)
    
    property = st.sidebar.selectbox(
        "Property",
        options=['A121', 'A122', 'A123', 'A124'],
        format_func=lambda x: {
            'A121': 'Real estate',
            'A122': 'Building society savings/Life insurance',
            'A123': 'Car or other',
            'A124': 'Unknown / No property'
        }[x]
    )
    
    other_debtors = st.sidebar.selectbox(
        "Other Debtors/Guarantors",
        options=['A101', 'A102', 'A103'],
        format_func=lambda x: {
            'A101': 'None',
            'A102': 'Co-applicant',
            'A103': 'Guarantor'
        }[x]
    )
    
    other_installments = st.sidebar.selectbox(
        "Other Installment Plans",
        options=['A141', 'A142', 'A143'],
        format_func=lambda x: {
            'A141': 'Bank',
            'A142': 'Stores',
            'A143': 'None'
        }[x]
    )
    
    housing = st.sidebar.selectbox(
        "Housing",
        options=['A151', 'A152', 'A153'],
        format_func=lambda x: {
            'A151': 'Rent',
            'A152': 'Own',
            'A153': 'For free'
        }[x]
    )
    
    existing_credits = st.sidebar.selectbox("Jumlah Kredit di Bank Ini", [1, 2, 3, 4])
    
    job = st.sidebar.selectbox(
        "Tipe Pekerjaan",
        options=['A171', 'A172', 'A173', 'A174'],
        format_func=lambda x: {
            'A171': 'Unemployed/unskilled - non-resident',
            'A172': 'Unskilled - resident',
            'A173': 'Skilled employee / official',
            'A174': 'Management / self-employed'
        }[x]
    )
    
    telephone = st.sidebar.selectbox(
        "Telepon",
        options=['A191', 'A192'],
        format_func=lambda x: {'A191': 'None', 'A192': 'Yes'}[x]
    )
    
    foreign_worker = st.sidebar.selectbox(
        "Pekerja Asing",
        options=['A201', 'A202'],
        format_func=lambda x: {'A201': 'Yes', 'A202': 'No'}[x]
    )
    
    predict_button = st.sidebar.button("🔍 Analisis Risiko", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Akurasi Testing", f"{accuracy_test:.1%}")
        col_m2.metric("Akurasi Training", f"{accuracy_train:.1%}")
        col_m3.metric("Total Data", len(df))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm,
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=['Good Credit', 'Bad Credit'],
                           y=['Good Credit', 'Bad Credit'],
                           text_auto=True,
                           color_continuous_scale='RdYlGn_r')
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribusi Risiko")
        
        risk_counts = df['target'].value_counts()
        fig_dist = go.Figure(data=[go.Pie(
            labels=['Good Credit', 'Bad Credit'],
            values=[risk_counts[0], risk_counts[1]],
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c'])
        )])
        fig_dist.update_layout(title="Target Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.metric("Good Rate", f"{risk_counts[0]/len(df)*100:.1f}%")
        st.metric("Bad Rate", f"{risk_counts[1]/len(df)*100:.1f}%")
    
    # Prediction
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Analisis Risiko")
        
        # Prepare input
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
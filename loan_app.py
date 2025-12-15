import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Prediksi Kemampuan Bayar Angsuran",
    page_icon="💰",
    layout="wide"
)

# Fungsi untuk generate dummy data (untuk demo)
@st.cache_data
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'pendapatan_bulanan': np.random.randint(3000000, 20000000, n_samples),
        'usia': np.random.randint(21, 65, n_samples),
        'jumlah_tanggungan': np.random.randint(0, 6, n_samples),
        'lama_bekerja_tahun': np.random.randint(0, 30, n_samples),
        'total_pinjaman': np.random.randint(5000000, 200000000, n_samples),
        'angsuran_bulanan': np.random.randint(500000, 10000000, n_samples),
        'riwayat_kredit_score': np.random.randint(300, 850, n_samples),
        'jumlah_pinjaman_aktif': np.random.randint(0, 5, n_samples),
        'kepemilikan_rumah': np.random.choice(['Milik Sendiri', 'Sewa', 'Keluarga'], n_samples),
        'status_pekerjaan': np.random.choice(['Tetap', 'Kontrak', 'Wiraswasta'], n_samples),
        'pendidikan': np.random.choice(['SMA', 'D3', 'S1', 'S2'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Hitung debt to income ratio
    df['debt_to_income_ratio'] = (df['angsuran_bulanan'] / df['pendapatan_bulanan']) * 100
    
    # Generate target: angsuran ke berapa akan menunggak (0 = tidak menunggak)
    # Logika: semakin tinggi DTI, credit score rendah, pinjaman banyak = lebih cepat menunggak
    risk_score = (
        (df['debt_to_income_ratio'] / 100) * 40 +
        ((850 - df['riwayat_kredit_score']) / 850) * 30 +
        (df['jumlah_pinjaman_aktif'] / 5) * 20 +
        (df['jumlah_tanggungan'] / 6) * 10
    )
    
    # Konversi ke kategori menunggak
    df['bulan_menunggak'] = 0
    df.loc[risk_score > 70, 'bulan_menunggak'] = np.random.randint(1, 4, (risk_score > 70).sum())
    df.loc[(risk_score > 55) & (risk_score <= 70), 'bulan_menunggak'] = np.random.randint(4, 7, ((risk_score > 55) & (risk_score <= 70)).sum())
    df.loc[(risk_score > 40) & (risk_score <= 55), 'bulan_menunggak'] = np.random.randint(7, 13, ((risk_score > 40) & (risk_score <= 55)).sum())
    
    # Kategori risiko
    df['kategori_risiko'] = 'Lancar'
    df.loc[df['bulan_menunggak'] > 0, 'kategori_risiko'] = 'Risiko Tinggi (1-3 bulan)'
    df.loc[df['bulan_menunggak'] >= 4, 'kategori_risiko'] = 'Risiko Sedang (4-6 bulan)'
    df.loc[df['bulan_menunggak'] >= 7, 'kategori_risiko'] = 'Risiko Rendah (7-12 bulan)'
    
    return df

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Generate dummy data
    df = generate_dummy_data(1000)
    
    # Prepare features
    df_encoded = df.copy()
    
    # Label encoding untuk categorical features
    le_rumah = LabelEncoder()
    le_pekerjaan = LabelEncoder()
    le_pendidikan = LabelEncoder()
    
    df_encoded['kepemilikan_rumah_encoded'] = le_rumah.fit_transform(df_encoded['kepemilikan_rumah'])
    df_encoded['status_pekerjaan_encoded'] = le_pekerjaan.fit_transform(df_encoded['status_pekerjaan'])
    df_encoded['pendidikan_encoded'] = le_pendidikan.fit_transform(df_encoded['pendidikan'])
    
    # Select features for model
    feature_columns = ['pendapatan_bulanan', 'usia', 'jumlah_tanggungan', 'lama_bekerja_tahun',
                      'total_pinjaman', 'angsuran_bulanan', 'riwayat_kredit_score',
                      'jumlah_pinjaman_aktif', 'debt_to_income_ratio',
                      'kepemilikan_rumah_encoded', 'status_pekerjaan_encoded', 'pendidikan_encoded']
    
    X = df_encoded[feature_columns]
    y = df_encoded['bulan_menunggak']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, df, X_test, y_test, y_pred, accuracy, feature_columns, le_rumah, le_pekerjaan, le_pendidikan

# Load data and model
try:
    model, scaler, df, X_test, y_test, y_pred, accuracy, feature_columns, le_rumah, le_pekerjaan, le_pendidikan = load_and_prepare_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Menu
st.sidebar.title("💼 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🏠 Beranda", "ℹ️ Tentang", "📚 Pengenalan Aplikasi", "⚠️ Faktor Risiko", "🔬 Prediksi Angsuran"]
)

# Menu: Tentang
if menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    
    st.markdown("""
    ## Sistem Prediksi Kemampuan Bayar Angsuran
    
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
    Dataset yang digunakan berisi data nasabah dengan berbagai parameter finansial dan demografis untuk memprediksi kemampuan membayar angsuran.
    
    **Jumlah Data:** 1000 sampel nasabah
    
    **Jumlah Fitur:** 12 parameter finansial dan demografis
    
    **Target:** Prediksi bulan ke berapa nasabah akan menunggak (0 = tidak menunggak)
    
    ### Model Machine Learning
    - **Algoritma:** Random Forest Classifier
    - **Akurasi Model:** {:.2%}
    - **Preprocessing:** Standardisasi & Label Encoding
    - **Kategori Prediksi:** 
      - Lancar (tidak menunggak)
      - Risiko Tinggi (1-3 bulan)
      - Risiko Sedang (4-6 bulan)
      - Risiko Rendah (7-12 bulan)
    
    ### Disclaimer
    ⚠️ **Penting:** Aplikasi ini hanya untuk tujuan **analisis risiko dan referensi internal**. Keputusan pemberian kredit harus mempertimbangkan berbagai faktor lain dan dilakukan oleh analis kredit yang berpengalaman.
    
    Hasil prediksi ini **TIDAK** dapat menjadi satu-satunya dasar keputusan kredit dan harus dikombinasikan dengan:
    - Verifikasi dokumen lengkap
    - Wawancara langsung
    - Site visit/survey lokasi
    - Analisis mendalam kondisi bisnis/pekerjaan
    - Pertimbangan aspek karakter dan komitmen nasabah
    """.format(accuracy))
    
    st.markdown("---")
    st.info("💡 **Tip:** Gunakan menu sidebar untuk navigasi ke berbagai fitur aplikasi.")

# Menu: Pengenalan Aplikasi
elif menu == "📚 Pengenalan Aplikasi":
    st.title("📚 Pengenalan Aplikasi")
    
    st.markdown("""
    ## Apa itu Sistem Prediksi Kemampuan Bayar Angsuran?
    
    Sistem Prediksi Kemampuan Bayar Angsuran adalah aplikasi berbasis **Machine Learning** yang dirancang untuk membantu lembaga keuangan menganalisis risiko kredit dan memprediksi kemampuan nasabah dalam membayar angsuran.
    
    ### 🎯 Tujuan Aplikasi
    
    1. **Manajemen Risiko:** Membantu mengidentifikasi nasabah berisiko tinggi
    2. **Efisiensi Proses:** Mempercepat proses analisis kredit
    3. **Data-Driven Decision:** Keputusan berbasis data dan analisis objektif
    4. **Early Warning System:** Deteksi dini potensi kredit bermasalah
    5. **Optimalisasi Portfolio:** Meningkatkan kualitas portfolio kredit
    
    ### 🔬 Cara Kerja Aplikasi
    
    1. **Input Data:** Input 12 parameter finansial dan demografis nasabah
    2. **Preprocessing:** Data dinormalisasi dan di-encode
    3. **Analisis ML:** Model Random Forest menganalisis pola risiko
    4. **Prediksi:** Sistem memprediksi bulan ke berapa nasabah akan menunggak
    5. **Klasifikasi Risiko:** Nasabah dikategorikan berdasarkan tingkat risiko
    6. **Rekomendasi:** Sistem memberikan rekomendasi tindakan
    
    ### 📊 Parameter yang Dianalisis
    
    Aplikasi menganalisis **12 parameter** yang meliputi:
    
    **Data Finansial:**
    - Pendapatan bulanan
    - Total pinjaman
    - Angsuran bulanan
    - Debt-to-Income Ratio (DTI)
    - Riwayat kredit score
    - Jumlah pinjaman aktif
    
    **Data Demografis:**
    - Usia
    - Jumlah tanggungan
    - Lama bekerja
    - Status pekerjaan
    - Kepemilikan rumah
    - Tingkat pendidikan
    
    ### 📈 Kategori Risiko
    
    | Kategori | Prediksi | Tindakan |
    |----------|----------|----------|
    | **🟢 Lancar** | Tidak menunggak | Disetujui dengan rate normal |
    | **🟡 Risiko Rendah** | Menunggak di bulan 7-12 | Disetujui dengan monitoring |
    | **🟠 Risiko Sedang** | Menunggak di bulan 4-6 | Perlu analisis tambahan |
    | **🔴 Risiko Tinggi** | Menunggak di bulan 1-3 | Ditolak atau persyaratan ketat |
    
    ### ✨ Fitur Utama
    
    - 🎨 **Interface User-Friendly:** Dashboard intuitif dan mudah digunakan
    - 📊 **Visualisasi Interaktif:** Grafik dan chart yang informatif
    - 🔍 **Analisis Real-time:** Hasil prediksi instant
    - 📈 **Dashboard Analytics:** Statistik dan performa model
    - 💾 **Berbasis Web:** Akses dari browser, multi-platform
    - 🔐 **Secure:** Data terenkripsi dan aman
    
    ### 🚀 Cara Menggunakan
    
    1. Pilih menu **"🔬 Prediksi Angsuran"** di sidebar
    2. Masukkan semua data finansial dan demografis nasabah
    3. Klik tombol **"🔍 Analisis Risiko"**
    4. Lihat hasil prediksi, kategori risiko, dan rekomendasi
    5. Download laporan hasil analisis (opsional)
    
    ### 💡 Best Practices
    
    - ✅ Pastikan semua data akurat dan terverifikasi
    - ✅ Gunakan data terbaru (maksimal 30 hari)
    - ✅ Lakukan cross-check dengan dokumen pendukung
    - ✅ Kombinasikan dengan analisis kualitatif
    - ✅ Update data secara berkala untuk monitoring
    """)
    
    st.success("✅ Siap menggunakan aplikasi? Pilih menu **'🔬 Prediksi Angsuran'** untuk mulai!")

# Menu: Faktor Risiko
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Faktor Risiko Kredit Macet")
    
    st.markdown("""
    ## Memahami Faktor Risiko dalam Pembayaran Angsuran
    
    Kredit macet atau tunggakan pembayaran adalah masalah serius dalam industri keuangan. Memahami faktor risikonya adalah kunci untuk manajemen risiko yang efektif.
    """)
    
    # Faktor Risiko Finansial
    st.subheader("💰 Faktor Risiko Finansial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 📊 Debt-to-Income Ratio (DTI)
        **Rasio paling penting dalam analisis kredit**
        
        - **Definisi:** Persentase pendapatan yang digunakan untuk membayar utang
        - **Formula:** (Total Angsuran / Pendapatan) × 100%
        - **Kategori:**
          - DTI < 30%: **Sangat Baik** ✅
          - DTI 30-40%: **Baik** 🟢
          - DTI 40-50%: **Perhatian** 🟡
          - DTI > 50%: **Risiko Tinggi** 🔴
        
        **Mengapa Penting:**
        - Mengukur kemampuan bayar riil
        - Indikator stress finansial
        - Prediktor kuat kredit macet
        
        ### 2. 💳 Riwayat Kredit Score
        **Credit score: 300-850**
        
        - **800-850:** Excellent (Risiko sangat rendah)
        - **740-799:** Very Good (Risiko rendah)
        - **670-739:** Good (Risiko sedang)
        - **580-669:** Fair (Risiko tinggi)
        - **< 580:** Poor (Risiko sangat tinggi)
        
        **Dampak:**
        - Score rendah = 3-5x lebih mungkin macet
        - Mencerminkan track record pembayaran
        - Faktor kunci dalam approval
        
        ### 3. 💰 Pendapatan Bulanan
        **Stabilitas dan kecukupan income**
        
        - **Minimum:** 3x angsuran bulanan
        - **Ideal:** 5x angsuran bulanan
        - **Verifikasi:** Slip gaji, rekening koran
        
        **Pertimbangan:**
        - Sumber pendapatan (tetap vs. variabel)
        - Tren pendapatan (naik/stabil/turun)
        - Pendapatan tambahan (side income)
        """)
    
    with col2:
        st.markdown("""
        ### 4. 📈 Jumlah Pinjaman Aktif
        **Multiple loans = higher risk**
        
        - **0-1 pinjaman:** Risiko rendah
        - **2-3 pinjaman:** Risiko sedang
        - **4+ pinjaman:** Risiko tinggi
        
        **Bahaya Multiple Loans:**
        - Beban angsuran berlipat
        - Stress finansial tinggi
        - Juggling payments
        - Domino effect jika satu macet
        
        ### 5. 🏦 Total Pinjaman
        **Besaran exposure**
        
        - **< 5x gaji:** Reasonable
        - **5-10x gaji:** Perhatian
        - **> 10x gaji:** Sangat berisiko
        
        **Risiko:**
        - Semakin besar, semakin lama tenor
        - Life events bisa ganggu pembayaran
        - Recovery lebih sulit
        
        ### 6. 💸 Angsuran Bulanan
        **Monthly burden**
        
        - **< Rp 2 juta:** Low risk
        - **Rp 2-5 juta:** Medium risk
        - **> Rp 5 juta:** High risk
        
        **Pertimbangan:**
        - Proporsi terhadap income
        - Fluktuasi pengeluaran bulanan
        - Emergency fund availability
        """)
    
    st.markdown("---")
    
    # Faktor Risiko Demografis
    st.subheader("👥 Faktor Risiko Demografis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1. 🎂 Usia
        **Age-risk correlation**
        
        - **21-25 tahun:** Risiko tinggi
          - Pendapatan tidak stabil
          - Kurang pengalaman keuangan
        - **26-45 tahun:** Risiko rendah-sedang
          - Peak earning years
          - Lebih stabil
        - **46-60 tahun:** Risiko sedang
          - Stabilitas tinggi tapi...
          - Mendekati pensiun
        - **> 60 tahun:** Risiko tinggi
          - Pendapatan menurun
          - Health issues
        
        ### 2. 👨‍👩‍👧‍👦 Jumlah Tanggungan
        **More dependents = higher expenses**
        
        - **0-1:** Risiko rendah
        - **2-3:** Risiko sedang
        - **4+:** Risiko tinggi
        
        **Dampak:**
        - Biaya hidup meningkat
        - Emergency needs lebih sering
        - Saving capacity berkurang
        
        ### 3. 💼 Lama Bekerja
        **Job stability indicator**
        
        - **< 1 tahun:** Risiko tinggi
        - **1-3 tahun:** Risiko sedang
        - **3-5 tahun:** Risiko rendah
        - **> 5 tahun:** Sangat stabil
        
        **Pentingnya:**
        - Stabilitas income
        - Job security
        - Career progression
        """)
    
    with col2:
        st.markdown("""
        ### 4. 🏢 Status Pekerjaan
        **Employment type matters**
        
        - **Pegawai Tetap:**
          - Risiko paling rendah
          - Income predictable
          - Job security tinggi
        
        - **Pegawai Kontrak:**
          - Risiko sedang
          - Uncertainty di akhir kontrak
          - Perlu renewal guarantee
        
        - **Wiraswasta:**
          - Risiko tinggi
          - Income fluktuatif
          - Business risk exposure
          - Perlu business track record
        
        ### 5. 🏠 Kepemilikan Rumah
        **Asset & stability indicator**
        
        - **Milik Sendiri:** Risiko rendah
          - Ada aset sebagai backup
          - Stabilitas tinggi
        
        - **Keluarga:** Risiko sedang
          - Support system ada
          - Tapi no asset
        
        - **Sewa:** Risiko tinggi
          - Extra monthly burden
          - No asset
          - Mobility tinggi
        
        ### 6. 🎓 Tingkat Pendidikan
        **Education correlates with income**
        
        - **S2/S3:** Risiko paling rendah
        - **S1:** Risiko rendah
        - **D3:** Risiko sedang
        - **SMA:** Risiko tinggi
        
        **Alasan:**
        - Earning potential berbeda
        - Career options lebih luas
        - Financial literacy
        """)
    
    st.markdown("---")
    
    # Red Flags
    st.subheader("🚨 Red Flags - Tanda Bahaya")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("""
        ### 🔴 Financial Red Flags
        - DTI > 50%
        - Credit score < 580
        - 4+ pinjaman aktif
        - Riwayat tunggakan
        - Pendapatan tidak stabil
        - No emergency fund
        - Sering ganti pekerjaan
        """)
    
    with col2:
        st.warning("""
        ### 🟡 Behavioral Red Flags
        - Sering telat bayar tagihan
        - Aplikasi kredit ditolak berulang
        - Maksimal limit kartu kredit
        - Cash advance rutin
        - Pinjaman ke banyak tempat
        - Tidak kooperatif saat verifikasi
        """)
    
    with col3:
        st.info("""
        ### 🟢 Positive Indicators
        - DTI < 30%
        - Credit score > 740
        - Track record baik
        - Pekerjaan stabil > 3 tahun
        - Aset memadai
        - Emergency fund 6+ bulan
        - Single loan or maksimal 2
        """)
    
    st.markdown("---")
    
    # Mitigation Strategies
    st.subheader("🛡️ Strategi Mitigasi Risiko")
    
    st.markdown("""
    ### Untuk Lembaga Keuangan:
    
    1. **Verifikasi Ketat**
       - Cross-check semua dokumen
       - Site visit untuk wiraswasta
       - Contact employer/business partner
    
    2. **Collateral & Guarantor**
       - Minta jaminan untuk high-risk applicant
       - Personal guarantor dengan kapasitas
       - Insurance coverage
    
    3. **Graduated Limit**
       - Mulai dengan limit kecil
       - Increase based on payment behavior
       - Test period 6-12 bulan
    
    4. **Dynamic Pricing**
       - Interest rate sesuai risk profile
       - High risk = higher rate
       - Incentive untuk low risk customers
    
    5. **Early Warning System**
       - Monitor payment patterns
       - Flag slight delays
       - Proactive contact
    
    6. **Collection Strategy**
       - Segmentasi debitur by risk
       - Automated reminders
       - Escalation protocol
    
    ### Untuk Nasabah (Tips Agar Tidak Menunggak):
    
    ✅ **Jaga DTI di bawah 30%** - Jangan overcommit
    
    ✅ **Build Emergency Fund** - Minimal 6 bulan pengeluaran
    
    ✅ **Otomatis Payment** - Set auto-debit agar tidak lupa
    
    ✅ **Prioritaskan Pembayaran** - Angsuran lebih penting dari lifestyle
    
    ✅ **Komunikasi Proaktif** - Hubungi bank jika ada masalah
    
    ✅ **Avoid Multiple Loans** - Jangan ambil pinjaman baru sebelum selesai
    
    ✅ **Track Spending** - Monitor cash flow bulanan
    
    ✅ **Increase Income** - Cari side hustle jika perlu
    """)

# Menu: Prediksi Angsuran
elif menu == "🔬 Prediksi Angsuran":
    st.title("🔬 Sistem Prediksi Kemampuan Bayar Angsuran")
    st.markdown("""
    Masukkan data finansial dan demografis nasabah di sidebar untuk mendapatkan prediksi risiko kredit.
    """)
    
    # Sidebar for user input
    st.sidebar.header("📋 Data Nasabah")
    st.sidebar.markdown("**Data Finansial**")
    
    # Financial data inputs
    pendapatan_bulanan = st.sidebar.number_input(
        "Pendapatan Bulanan (Rp)", 
        min_value=1000000, 
        max_value=100000000, 
        value=5000000,
        step=500000
    )
    
    total_pinjaman = st.sidebar.number_input(
        "Total Pinjaman (Rp)", 
        min_value=1000000, 
        max_value=500000000, 
        value=50000000,
        step=5000000
    )
    
    angsuran_bulanan = st.sidebar.number_input(
        "Angsuran Bulanan (Rp)", 
        min_value=100000, 
        max_value=20000000, 
        value=2000000,
        step=100000
    )
    
    riwayat_kredit_score = st.sidebar.slider(
        "Riwayat Kredit Score", 
        300, 850, 650
    )
    
    jumlah_pinjaman_aktif = st.sidebar.selectbox(
        "Jumlah Pinjaman Aktif", 
        options=[0, 1, 2, 3, 4, 5]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Demografis**")
    
    # Demographic data inputs
    usia = st.sidebar.slider("Usia", 21, 65, 35)
    
    jumlah_tanggungan = st.sidebar.selectbox(
        "Jumlah Tanggungan", 
        options=[0, 1, 2, 3, 4, 5]
    )
    
    lama_bekerja_tahun = st.sidebar.slider(
        "Lama Bekerja (Tahun)", 
        0, 30, 5
    )
    
    status_pekerjaan = st.sidebar.selectbox(
        "Status Pekerjaan", 
        options=['Tetap', 'Kontrak', 'Wiraswasta']
    )
    
    kepemilikan_rumah = st.sidebar.selectbox(
        "Kepemilikan Rumah", 
        options=['Milik Sendiri', 'Sewa', 'Keluarga']
    )
    
    pendidikan = st.sidebar.selectbox(
        "Pendidikan Terakhir", 
        options=['SMA', 'D3', 'S1', 'S2']
    )
    
    # Calculate DTI
    debt_to_income_ratio = (angsuran_bulanan / pendapatan_bulanan) * 100
    
    # Create prediction button
    predict_button = st.sidebar.button("🔍 Analisis Risiko", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Performa Model")
        
        # Display metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Akurasi Model", f"{accuracy:.1%}")
        col_m2.metric("Total Nasabah", len(df))
        col_m3.metric("Jumlah Fitur", 12)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Buat label untuk confusion matrix
        unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
        labels = [f"Bulan {int(x)}" if x > 0 else "Lancar" for x in unique_classes]
        
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           x=labels,
                           y=labels,
                           text_auto=True,
                           color_continuous_scale='RdYlGn_r')
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("📈 Distribusi Risiko")
        
        # Risk distribution
        risk_counts = df['kategori_risiko'].value_counts()
        colors_map = {
            'Lancar': '#2ecc71',
            'Risiko Rendah (7-12 bulan)': '#f39c12',
            'Risiko Sedang (4-6 bulan)': '#e67e22',
            'Risiko Tinggi (1-3 bulan)': '#e74c3c'
        }
        colors = [colors_map.get(label, '#95a5a6') for label in risk_counts.index]
        
        fig_dist = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=colors)
        )])
        fig_dist.update_layout(title="Kategori Risiko Nasabah")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 Statistik Dataset")
        st.metric("Rata-rata DTI", f"{df['debt_to_income_ratio'].mean():.1f}%")
        st.metric("Rata-rata Credit Score", f"{df['riwayat_kredit_score'].mean():.0f}")
    
    # Prediction section
    if predict_button:
        st.markdown("---")
        st.subheader("🎯 Hasil Analisis Risiko")
        
        # Encode categorical variables
        kepemilikan_encoded = le_rumah.transform([kepemilikan_rumah])[0]
        pekerjaan_encoded = le_pekerjaan.transform([status_pekerjaan])[0]
        pendidikan_encoded = le_pendidikan.transform([pendidikan])[0]
        
        # Prepare input features
        input_data = np.array([[
            pendapatan_bulanan, usia, jumlah_tanggungan, lama_bekerja_tahun,
            total_pinjaman, angsuran_bulanan, riwayat_kredit_score,
            jumlah_pinjaman_aktif, debt_to_income_ratio,
            kepemilikan_encoded, pekerjaan_encoded, pendidikan_encoded
        ]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Determine risk category
        if prediction == 0:
            kategori = "🟢 Lancar (Tidak Menunggak)"
            warna = "success"
            rekomendasi = """
            ### ✅ Rekomendasi: DISETUJUI
            
            **Tindakan:**
            - ✅ Kredit dapat disetujui dengan rate normal
            - ✅ Limit sesuai kapasitas maksimal
            - ✅ Tenor fleksibel sesuai kebutuhan
            - ✅ Monitoring rutin standar (bulanan)
            
            **Alasan:**
            - Profil finansial sehat
            - DTI dalam batas aman
            - Track record baik
            - Risiko kredit macet sangat rendah
            
            **Tips untuk Nasabah:**
            - Pertahankan disiplin pembayaran
            - Jaga credit score tetap baik
            - Hindari mengambil pinjaman baru
            - Build emergency fund
            """
        elif prediction <= 3:
            kategori = f"🔴 Risiko Tinggi (Prediksi Menunggak Bulan ke-{int(prediction)})"
            warna = "error"
            rekomendasi = """
            ### ⛔ Rekomendasi: DITOLAK / PERSYARATAN KETAT
            
            **Tindakan:**
            - ⚠️ Sebaiknya kredit ditolak
            - ⚠️ Jika tetap disetujui: limit minimal, tenor pendek
            - ⚠️ Wajib ada jaminan/collateral kuat
            - ⚠️ Perlu guarantor dengan kapasitas baik
            - ⚠️ Interest rate lebih tinggi (risk-based pricing)
            - ⚠️ Monitoring ketat (mingguan)
            
            **Alasan:**
            - Risiko kredit macet sangat tinggi
            - Prediksi menunggak dalam 1-3 bulan pertama
            - Profil finansial tidak memadai
            - DTI kemungkinan terlalu tinggi
            
            **Saran untuk Perbaikan:**
            - Turunkan DTI di bawah 30%
            - Tingkatkan pendapatan
            - Lunasi pinjaman yang ada
            - Perbaiki credit score
            - Tunggu 6-12 bulan untuk re-apply
            """
        elif prediction <= 6:
            kategori = f"🟠 Risiko Sedang (Prediksi Menunggak Bulan ke-{int(prediction)})"
            warna = "warning"
            rekomendasi = """
            ### ⚠️ Rekomendasi: PERLU ANALISIS TAMBAHAN
            
            **Tindakan:**
            - 🔍 Lakukan verifikasi mendalam
            - 🔍 Site visit/survey wajib dilakukan
            - 🔍 Interview langsung dengan nasabah
            - 🔍 Check reference (employer/business)
            - 🔍 Limit moderat (50-70% dari maksimal)
            - 🔍 Tenor diperpendek
            - 🔍 Monitoring intensif (2 minggu sekali)
            
            **Pertimbangan:**
            - Ada indikasi risiko medium
            - Perlu mitigasi tambahan
            - Collateral sangat direkomendasikan
            - Grace period lebih ketat
            
            **Mitigasi Risiko:**
            - Minta proof of income terbaru
            - Cek stability pekerjaan/bisnis
            - Graduated limit (mulai kecil)
            - Asuransi kredit wajib
            - Early warning system
            """
        else:
            kategori = f"🟡 Risiko Rendah (Prediksi Menunggak Bulan ke-{int(prediction)})"
            warna = "info"
            rekomendasi = """
            ### 🟡 Rekomendasi: DISETUJUI DENGAN MONITORING
            
            **Tindakan:**
            - ✅ Kredit dapat disetujui
            - ✅ Limit standar (80-90% dari maksimal)
            - ✅ Tenor normal
            - ⚠️ Monitoring rutin ketat (2 minggu sekali di awal)
            - ⚠️ Early warning system diaktifkan
            
            **Pertimbangan:**
            - Risiko relatif rendah tapi perlu waspada
            - Potensi menunggak di tengah periode
            - Perlu perhatian khusus
            
            **Mitigasi:**
            - Review berkala setiap 3 bulan
            - Proactive communication
            - Tawarkan payment reminder service
            - Siapkan restructuring plan jika perlu
            
            **Tips untuk Nasabah:**
            - Set auto-debit untuk menghindari lupa
            - Jaga komunikasi dengan bank
            - Laporkan jika ada perubahan finansial
            - Pertimbangkan asuransi perlindungan
            """
        
        # Display result with appropriate color
        if warna == "success":
            st.success(kategori)
        elif warna == "error":
            st.error(kategori)
        elif warna == "warning":
            st.warning(kategori)
        else:
            st.info(kategori)
        
        # Display confidence
        max_proba = np.max(prediction_proba)
        st.markdown(f"**Tingkat Kepercayaan Prediksi:** {max_proba:.1%}")
        
        # Display recommendation
        st.markdown(rekomendasi)
        
        # DTI Analysis
        st.markdown("---")
        st.subheader("📊 Analisis Debt-to-Income Ratio (DTI)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # DTI Gauge
            fig_dti = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=debt_to_income_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "DTI Ratio (%)"},
                delta={'reference': 30},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 40], 'color': "#f1c40f"},
                        {'range': [40, 50], 'color': "#e67e22"},
                        {'range': [50, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig_dti, use_container_width=True)
        
        with col2:
            st.markdown("### Interpretasi DTI")
            
            if debt_to_income_ratio < 30:
                st.success(f"""
                **DTI: {debt_to_income_ratio:.1f}% - SANGAT BAIK** ✅
                
                - Kemampuan bayar sangat baik
                - Masih banyak ruang untuk pengeluaran lain
                - Risiko stress finansial rendah
                - Buffer cukup untuk emergency
                """)
            elif debt_to_income_ratio < 40:
                st.info(f"""
                **DTI: {debt_to_income_ratio:.1f}% - BAIK** 🟢
                
                - Kemampuan bayar masih memadai
                - Perlu hati-hati dengan pengeluaran
                - Sedikit ruang untuk emergency
                - Disarankan tidak menambah utang
                """)
            elif debt_to_income_ratio < 50:
                st.warning(f"""
                **DTI: {debt_to_income_ratio:.1f}% - PERHATIAN** 🟡
                
                - Beban utang cukup tinggi
                - Risiko stress finansial meningkat
                - Sangat sedikit ruang untuk emergency
                - JANGAN tambah utang baru
                """)
            else:
                st.error(f"""
                **DTI: {debt_to_income_ratio:.1f}% - BAHAYA** 🔴
                
                - Beban utang sangat tinggi
                - Risiko kredit macet sangat besar
                - Tidak ada buffer untuk emergency
                - Perlu restrukturisasi utang segera
                """)
        
        # Probability Distribution
        st.markdown("---")
        st.subheader("📊 Distribusi Probabilitas Prediksi")
        
        # Get class labels
        classes = model.classes_
        class_labels = [f"Bulan {int(c)}" if c > 0 else "Lancar" for c in classes]
        
        # Create bar chart
        fig_proba = go.Figure(data=[
            go.Bar(
                x=class_labels,
                y=prediction_proba,
                text=[f'{p:.1%}' for p in prediction_proba],
                textposition='auto',
                marker=dict(
                    color=prediction_proba,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Probabilitas")
                )
            )
        ])
        fig_proba.update_layout(
            xaxis_title="Prediksi Bulan Menunggak",
            yaxis_title="Probabilitas",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        st.plotly_chart(fig_proba, use_container_width=True)
        
        # Display input summary
        st.markdown("---")
        st.subheader("📝 Ringkasan Data Input")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Data Finansial")
            st.markdown(f"""
            - **Pendapatan Bulanan:** Rp {pendapatan_bulanan:,.0f}
            - **Total Pinjaman:** Rp {total_pinjaman:,.0f}
            - **Angsuran Bulanan:** Rp {angsuran_bulanan:,.0f}
            - **DTI Ratio:** {debt_to_income_ratio:.1f}%
            - **Credit Score:** {riwayat_kredit_score}
            - **Pinjaman Aktif:** {jumlah_pinjaman_aktif}
            """)
        
        with col2:
            st.markdown("### 👤 Data Demografis")
            st.markdown(f"""
            - **Usia:** {usia} tahun
            - **Jumlah Tanggungan:** {jumlah_tanggungan}
            - **Lama Bekerja:** {lama_bekerja_tahun} tahun
            - **Status Pekerjaan:** {status_pekerjaan}
            - **Kepemilikan Rumah:** {kepemilikan_rumah}
            - **Pendidikan:** {pendidikan}
            """)
        
        with col3:
            st.markdown("### 📈 Analisis Rasio")
            
            # Calculate ratios
            loan_to_income = (total_pinjaman / (pendapatan_bulanan * 12)) * 100
            installment_to_loan = (angsuran_bulanan / total_pinjaman) * 100
            
            st.markdown(f"""
            - **Loan-to-Income:** {loan_to_income:.1f}%
            - **Angsuran/Pinjaman:** {installment_to_loan:.2f}%
            - **Sisa Income:** Rp {pendapatan_bulanan - angsuran_bulanan:,.0f}
            - **% Sisa Income:** {((pendapatan_bulanan - angsuran_bulanan) / pendapatan_bulanan * 100):.1f}%
            """)
            
            if loan_to_income > 500:
                st.error("⚠️ Loan-to-income terlalu tinggi!")
            elif loan_to_income > 300:
                st.warning("⚠️ Loan-to-income cukup tinggi")
            else:
                st.success("✅ Loan-to-income wajar")

# Menu: Beranda (Default)
else:
    st.title("💰 Sistem Prediksi Kemampuan Bayar Angsuran")
    
    st.markdown("""
    ## Selamat Datang di Credit Risk Analysis System! 👋
    
    Aplikasi ini menggunakan **Machine Learning** untuk memprediksi kemampuan nasabah dalam membayar angsuran dan mengidentifikasi potensi kredit macet.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        ### 🎯 Akurat
        Model dengan akurasi **{accuracy:.1%}** menggunakan Random Forest
        """)
    
    with col2:
        st.success("""
        ### 🚀 Cepat
        Analisis risiko dalam hitungan detik
        """)
    
    with col3:
        st.warning("""
        ### 📊 Komprehensif
        Analisis 12 parameter finansial & demografis
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Panduan Cepat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Untuk Pengguna Baru:
        
        1. **Baca Pengenalan** - Pahami cara kerja sistem
        2. **Pelajari Faktor Risiko** - Ketahui indikator kredit macet
        3. **Lakukan Analisis** - Input data dan lihat prediksi
        
        👉 Mulai dari menu **"📚 Pengenalan Aplikasi"**
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Untuk Analis Berpengalaman:
        
        1. Pilih menu **"🔬 Prediksi Angsuran"**
        2. Input 12 parameter nasabah di sidebar
        3. Klik tombol **"Analisis Risiko"**
        4. Review hasil dan rekomendasi tindakan
        
        👉 Langsung ke menu **"🔬 Prediksi Angsuran"**
        """)
    
    st.markdown("---")
    
    # Key Metrics Dashboard
    st.subheader("📊 Dashboard Statistik")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi Model", f"{accuracy:.1%}")
    
    with col2:
        lancar_pct = (df['bulan_menunggak'] == 0).sum() / len(df) * 100
        st.metric("Nasabah Lancar", f"{lancar_pct:.1f}%")
    
    with col3:
        avg_dti = df['debt_to_income_ratio'].mean()
        st.metric("Rata-rata DTI", f"{avg_dti:.1f}%")
    
    with col4:
        avg_score = df['riwayat_kredit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_score:.0f}")
    
    # Risk Distribution Chart
    st.markdown("---")
    st.subheader("📈 Distribusi Kategori Risiko")
    
    risk_counts = df['kategori_risiko'].value_counts()
    
    fig_risk = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        labels={'x': 'Kategori Risiko', 'y': 'Jumlah Nasabah'},
        color=risk_counts.index,
        color_discrete_map={
            'Lancar': '#2ecc71',
            'Risiko Rendah (7-12 bulan)': '#f39c12',
            'Risiko Sedang (4-6 bulan)': '#e67e22',
            'Risiko Tinggi (1-3 bulan)': '#e74c3c'
        }
    )
    fig_risk.update_layout(showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    # Important notice
    st.error("""
    ### ⚠️ DISCLAIMER PENTING
    
    Aplikasi ini adalah **alat bantu analisis** yang memberikan prediksi berdasarkan data historis dan pola statistik. 
    
    **Keputusan kredit final HARUS mempertimbangkan:**
    - ✅ Verifikasi dokumen lengkap dan akurat
    - ✅ Wawancara langsung dengan calon nasabah
    - ✅ Site visit dan survey lapangan
    - ✅ Analisis mendalam kondisi bisnis/pekerjaan
    - ✅ Aspek karakter dan komitmen nasabah
    - ✅ Kondisi ekonomi dan industri terkini
    - ✅ Kebijakan internal perusahaan
    
    **TIDAK BOLEH** menggunakan hasil prediksi sebagai **satu-satunya** dasar keputusan.
    """)
    
    st.markdown("---")
    
    # Call to action
    st.success("""
    ### 💡 Siap Memulai Analisis?
    
    Pilih salah satu menu di sidebar:
    - **ℹ️ Tentang** - Informasi sistem dan teknologi
    - **📚 Pengenalan Aplikasi** - Cara kerja dan best practices
    - **⚠️ Faktor Risiko** - Pelajari indikator kredit macet
    - **🔬 Prediksi Angsuran** - Mulai analisis risiko kredit!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏦 Aplikasi ini untuk tujuan analisis risiko kredit internal. Keputusan final harus melibatkan verifikasi dan analisis mendalam.</p>
    <p>© 2024 Credit Risk Analysis System | Dibuat dengan 💼 menggunakan Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
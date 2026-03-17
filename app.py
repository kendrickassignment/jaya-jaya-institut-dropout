"""
Jaya Jaya Institut — Sistem Deteksi Dini Dropout Mahasiswa
Aplikasi Streamlit dengan UI premium & fitur batch prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import io
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(
    page_title="Jaya Jaya Institut — Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Premium
st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 13px !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #f8fafc !important; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2744 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        border: 1px solid rgba(99,179,237,0.2);
    }
    .main-header h1 { color: #f0f9ff; margin: 0; font-size: 28px; font-weight: 700; }
    .main-header p  { color: #94c6ed; margin: 8px 0 0; font-size: 15px; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .val { font-size: 36px; font-weight: 700; line-height: 1.1; }
    .metric-card .lbl { font-size: 13px; color: #64748b; margin-top: 4px; }

    /* Risk gauge card */
    .risk-card {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
        border: 2px solid;
    }
    .risk-low    { background: #f0fdf4; border-color: #22c55e; }
    .risk-medium { background: #fffbeb; border-color: #f59e0b; }
    .risk-high   { background: #fef2f2; border-color: #ef4444; }
    .risk-card .risk-pct { font-size: 52px; font-weight: 700; line-height: 1; }
    .risk-card .risk-label { font-size: 18px; font-weight: 600; margin-top: 8px; }
    .risk-card .risk-desc { font-size: 14px; color: #64748b; margin-top: 6px; }
    .risk-low    .risk-pct   { color: #16a34a; }
    .risk-medium .risk-pct   { color: #d97706; }
    .risk-high   .risk-pct   { color: #dc2626; }
    .risk-low    .risk-label { color: #15803d; }
    .risk-medium .risk-label { color: #b45309; }
    .risk-high   .risk-label { color: #b91c1c; }

    /* Section title */
    .section-title {
        font-size: 16px; font-weight: 600; color: #1e293b;
        margin: 20px 0 12px; padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Action items */
    .action-item {
        background: #f8fafc;
        border-left: 4px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #334155;
    }
    .action-urgent   { border-left-color: #ef4444; }
    .action-warning  { border-left-color: #f59e0b; }
    .action-normal   { border-left-color: #3b82f6; }

    /* Batch table */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        padding: 10px 20px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        font-size: 15px !important;
        transition: all 0.2s !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():
    model_path = Path("model_lgb.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_feature_list():
    feat_path = Path("model_features.json")
    if feat_path.exists():
        with open(feat_path) as f:
            return json.load(f)
    # Fallback default features
    return [
        'Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
        'Age_at_enrollment', 'Admission_grade',
    ]


model = load_model()
FEATURES = load_feature_list()

if model is None:
    st.error(
        "⚠️ **File model tidak ditemukan.**\n\n"
        "Pastikan `model_lgb.pkl` berada di direktori yang sama dengan `app.py`. "
        "Jalankan `notebook.ipynb` terlebih dahulu untuk membuat model."
    )
    st.stop()


# Helper Functions
def get_risk_level(prob: float) -> tuple[str, str, str]:
    """Return (level, label, css_class) based on dropout probability."""
    if prob < 0.35:
        return "AMAN", "✅ Risiko Rendah — Aman", "risk-low"
    elif prob < 0.65:
        return "WASPADA", "⚠️ Risiko Sedang — Perlu Pemantauan", "risk-medium"
    else:
        return "BAHAYA", "🚨 Risiko Tinggi — Butuh Intervensi Segera", "risk-high"


def create_gauge(prob: float) -> go.Figure:
    """Create a gauge chart for dropout probability."""
    color = "#22c55e" if prob < 0.35 else "#f59e0b" if prob < 0.65 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b",
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#dcfce7"},
                {"range": [35, 65], "color": "#fef9c3"},
                {"range": [65, 100],"color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": prob * 100,
            },
        },
        title={"text": "Probabilitas Dropout", "font": {"size": 14, "color": "#64748b"}},
    ))
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


def create_feature_radar(input_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing input values to average dropout/graduate profiles."""
    norm_features = [
        'Tuition_fees_up_to_date', 'Scholarship_holder',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    ]
    labels_nice = ['Lunas SPP', 'Beasiswa', 'SKS S1', 'Nilai S1', 'SKS S2', 'Nilai S2']

    # Profil rata-rata (dari domain knowledge / data pelatihan)
    profile_grad   = [1.0, 0.28, 5.9, 13.2, 5.5, 12.8]
    profile_drop   = [0.2, 0.12, 2.4,  9.1, 1.8,  8.5]
    max_vals       = [1.0, 1.0,  26.0, 20.0, 26.0, 20.0]

    # Normalisasi input ke 0-1
    input_vals = []
    for feat, mx in zip(norm_features, max_vals):
        if feat in input_df.columns:
            val = float(input_df[feat].iloc[0]) / mx
        else:
            val = 0.0
        input_vals.append(min(val, 1.0))

    cats = labels_nice + [labels_nice[0]]  # close the radar

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[v/mx for v, mx in zip(profile_grad, max_vals)] + [[v/mx for v, mx in zip(profile_grad, max_vals)][0]],
        theta=cats, fill='toself', name='Profil Graduate',
        line_color='#22c55e', fillcolor='rgba(34,197,94,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[v/mx for v, mx in zip(profile_drop, max_vals)] + [[v/mx for v, mx in zip(profile_drop, max_vals)][0]],
        theta=cats, fill='toself', name='Profil Dropout',
        line_color='#ef4444', fillcolor='rgba(239,68,68,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=input_vals + [input_vals[0]],
        theta=cats, fill='toself', name='Mahasiswa Ini',
        line_color='#3b82f6', fillcolor='rgba(59,130,246,0.2)', line_width=2.5
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "size": 12},
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def predict_single(inputs: dict) -> tuple[int, float]:
    """Run prediction for a single student."""
    input_df = pd.DataFrame([inputs])
    # Pastikan kolom sesuai model
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURES]
    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])
    return pred, prob



# SIDEBAR — Input Data Mahasiswa

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:36px;'>🎓</div>
        <div style='font-size:17px; font-weight:700; color:#f0f9ff;'>Jaya Jaya Institut</div>
        <div style='font-size:12px; color:#94a3b8; margin-top:4px;'>Sistem Deteksi Dini Dropout</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📋 Data Mahasiswa")

    # ── Finansial ──
    st.markdown("**💰 Status Finansial**")
    tuition = st.selectbox(
        "Pembayaran SPP",
        options=[1, 0],
        format_func=lambda x: "✅ Lunas" if x == 1 else "❌ Menunggak",
        help="Apakah mahasiswa sudah membayar SPP semester ini?"
    )
    scholarship = st.selectbox(
        "Status Beasiswa",
        options=[0, 1],
        format_func=lambda x: "✅ Penerima Beasiswa" if x == 1 else "❌ Bukan Penerima",
    )
    debtor = st.selectbox(
        "Status Debitur / Utang",
        options=[0, 1],
        format_func=lambda x: "✅ Ada Utang" if x == 1 else "❌ Tidak Berutang",
    )

    st.divider()

    # ── Akademik Semester 1 ──
    st.markdown("**📚 Semester 1**")
    sem1_approved = st.slider("SKS Lulus", 0, 26, 5, key="s1a",
                               help="Jumlah mata kuliah/SKS yang berhasil dilulus di semester 1")
    sem1_grade = st.number_input("Nilai Rata-rata (0–20)", 0.0, 20.0, 12.0, 0.1, key="s1g",
                                  help="Nilai rata-rata semua mata kuliah semester 1, skala 0–20")

    st.divider()

    # ── Akademik Semester 2 ──
    st.markdown("**📚 Semester 2**")
    sem2_approved = st.slider("SKS Lulus", 0, 26, 5, key="s2a")
    sem2_grade = st.number_input("Nilai Rata-rata (0–20)", 0.0, 20.0, 12.0, 0.1, key="s2g")

    st.divider()

    # ── Demografis ──
    st.markdown("**👤 Profil Mahasiswa**")
    age = st.slider("Usia saat Enrollment", 17, 60, 20,
                    help="Usia mahasiswa pada saat pertama kali mendaftar")
    admission_grade = st.number_input("Nilai Masuk (0–200)", 0.0, 200.0, 130.0, 0.5,
                                       help="Nilai ujian masuk mahasiswa")

    st.divider()
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)


# MAIN CONTENT

# Header
st.markdown("""
<div class='main-header'>
    <h1>🎓 Sistem Deteksi Dini Dropout Mahasiswa</h1>
    <p>Dashboard prediksi berbasis AI · Model LightGBM · Jaya Jaya Institut</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediksi Individual", "📊 Prediksi Batch (CSV)"])



# TAB 1 — Individual Prediction

with tab1:

    # Inisiasi state kosong
    if not predict_btn:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info("**Langkah 1**\n\nIsi data mahasiswa di panel kiri (sidebar)")
        with col_info2:
            st.info("**Langkah 2**\n\nKlik tombol **Prediksi Sekarang** di bawah form")
        with col_info3:
            st.info("**Langkah 3**\n\nLihat hasil prediksi, analisis risiko, dan rekomendasi tindakan")

        st.markdown("""
        <div style='background:#f8fafc;border-radius:12px;padding:20px;border:1px solid #e2e8f0;margin-top:16px;'>
            <h4 style='margin:0 0 8px;color:#1e293b;'>ℹ️ Tentang Model Prediksi</h4>
            <p style='margin:0;color:#475569;font-size:14px;line-height:1.7;'>
            Model LightGBM ini dilatih menggunakan data historis 4.424 mahasiswa Jaya Jaya Institut
            dengan akurasi tinggi dan <strong>Recall > 85%</strong> untuk kelas Dropout.
            Model menganalisis 9 faktor kunci (akademik, finansial, demografis) untuk menghasilkan
            prediksi probabilistik beserta tingkat risiko dropout setiap mahasiswa.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        # Kumpulkan input
        inputs = {
            'Tuition_fees_up_to_date'           : tuition,
            'Scholarship_holder'                 : scholarship,
            'Debtor'                             : debtor,
            'Curricular_units_1st_sem_approved'  : sem1_approved,
            'Curricular_units_1st_sem_grade'     : sem1_grade,
            'Curricular_units_2nd_sem_approved'  : sem2_approved,
            'Curricular_units_2nd_sem_grade'     : sem2_grade,
            'Age_at_enrollment'                  : age,
            'Admission_grade'                    : admission_grade,
        }
        input_df = pd.DataFrame([inputs])

        pred, prob = predict_single(inputs)
        level, label, css_class = get_risk_level(prob)

        # ── Baris atas: Gauge + Risk Card + Radar ────────────────────────────
        col_gauge, col_radar = st.columns([1, 1])

        with col_gauge:
            st.markdown("<div class='section-title'>📊 Hasil Analisis Risiko</div>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge(prob), use_container_width=True)

            st.markdown(f"""
            <div class='risk-card {css_class}'>
                <div class='risk-pct'>{prob*100:.1f}%</div>
                <div class='risk-label'>{label}</div>
                <div class='risk-desc'>Probabilitas mahasiswa ini mengalami <em>dropout</em></div>
            </div>
            """, unsafe_allow_html=True)

        with col_radar:
            st.markdown("<div class='section-title'>📐 Perbandingan dengan Profil Rata-rata</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(create_feature_radar(input_df), use_container_width=True)
            st.caption("Garis biru = mahasiswa ini · Hijau = profil rata-rata Graduate · Merah = profil rata-rata Dropout")

        st.divider()

        # Detail faktor risiko 
        st.markdown("<div class='section-title'>🔎 Analisis Faktor Risiko Individual</div>",
                    unsafe_allow_html=True)

        risk_factors = []
        if tuition == 0:
            risk_factors.append(("🔴 SPP Menunggak", "Faktor risiko terkuat. Mahasiswa yang tidak melunasi SPP memiliki dropout rate >70%.", "urgent"))
        if debtor == 1:
            risk_factors.append(("🔴 Memiliki Utang", "Status debitur meningkatkan tekanan finansial dan risiko dropout.", "urgent"))
        if sem1_approved <= 2:
            risk_factors.append(("🔴 SKS Semester 1 Sangat Rendah", f"Hanya {sem1_approved} SKS lulus. Di bawah rata-rata graduate (5.9 SKS).", "urgent"))
        elif sem1_approved <= 4:
            risk_factors.append(("🟡 SKS Semester 1 Rendah", f"{sem1_approved} SKS lulus. Perlu ditingkatkan untuk mencapai target.", "warning"))
        if sem2_approved <= 2:
            risk_factors.append(("🔴 SKS Semester 2 Sangat Rendah", f"Hanya {sem2_approved} SKS lulus. Tren akademik menurun.", "urgent"))
        if sem1_grade < 10:
            risk_factors.append(("🔴 Nilai Semester 1 Rendah", f"Nilai {sem1_grade:.1f} jauh di bawah rata-rata graduate (13.2).", "urgent"))
        elif sem1_grade < 12:
            risk_factors.append(("🟡 Nilai Semester 1 Di Bawah Rata-rata", f"Nilai {sem1_grade:.1f}. Perlu peningkatan performa.", "warning"))
        if age > 30:
            risk_factors.append(("🟡 Mahasiswa Usia Lebih Tua", f"Usia {age} tahun. Mahasiswa dewasa lebih rentan dropout karena tanggung jawab lain.", "warning"))
        if scholarship == 0 and tuition == 0:
            risk_factors.append(("🟡 Tidak Ada Dukungan Finansial", "Bukan penerima beasiswa dan menunggak SPP — risiko finansial kumulatif.", "warning"))

        if not risk_factors and pred == 0:
            st.success("✅ **Tidak ditemukan faktor risiko signifikan.** Mahasiswa ini memiliki profil yang kuat untuk menyelesaikan studi.")
        else:
            col_r1, col_r2 = st.columns(2)
            for i, (title, desc, severity) in enumerate(risk_factors):
                col = col_r1 if i % 2 == 0 else col_r2
                css_action = "action-urgent" if severity == "urgent" else "action-warning" if severity == "warning" else "action-normal"
                col.markdown(f"""
                <div class='action-item {css_action}'>
                    <strong>{title}</strong><br>
                    <span style='color:#64748b;font-size:13px;'>{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Rekomendasi Tindakan
        st.markdown("<div class='section-title'>📋 Rekomendasi Tindakan</div>", unsafe_allow_html=True)

        if pred == 1 or prob >= 0.65:
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.error("### 🚨 Intervensi Segera Diperlukan")
                actions_urgent = []
                if tuition == 0 or debtor == 1:
                    actions_urgent.append("💳 **Konsultasi Finansial**: Tawarkan skema cicilan SPP atau beasiswa darurat.")
                if sem1_approved <= 3 or sem2_approved <= 3:
                    actions_urgent.append("📚 **Bimbingan Akademik Intensif**: Jadwalkan sesi tutoring 2x seminggu.")
                if sem1_grade < 10 or sem2_grade < 10:
                    actions_urgent.append("✏️ **Program Remediasi**: Enroll dalam kelas remedial untuk mata kuliah yang gagal.")
                actions_urgent.append("🤝 **Sesi Konseling**: Pertemukan dengan konselor akademik dalam 1 minggu ke depan.")
                for a in actions_urgent:
                    st.markdown(f"- {a}")
            with col_rec2:
                st.warning("### ⏰ Monitoring Jangka Pendek")
                st.markdown("""
                - 📅 **Review Mingguan**: Pantau presensi dan pengumpulan tugas setiap minggu.
                - 👥 **Program Mentoring**: Pasangkan dengan mahasiswa senior berprestasi.
                - 📊 **Re-evaluasi**: Lakukan prediksi ulang setelah tengah semester.
                - 📱 **Notifikasi Orang Tua**: Informasikan situasi akademik kepada wali mahasiswa.
                """)
        elif prob >= 0.35:
            st.warning("### ⚠️ Pemantauan Aktif Disarankan")
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.markdown("""
                **Tindakan Preventif:**
                - 📅 **Check-in Bulanan**: Jadwalkan pertemuan konselor setiap bulan.
                - 📚 **Dukungan Akademik Ringan**: Tawarkan akses ke kelompok belajar.
                - 💰 **Informasi Beasiswa**: Kirimkan informasi program beasiswa yang tersedia.
                """)
            with col_rec2:
                st.markdown("""
                **Monitoring Berkelanjutan:**
                - 📊 **Pantau Nilai Mid-Semester**: Flag jika nilai turun signifikan.
                - 🏫 **Tingkatkan Keterlibatan**: Ajak ikut kegiatan akademik kampus.
                """)
        else:
            st.success("### ✅ Tidak Diperlukan Intervensi Khusus")
            st.markdown("""
            Mahasiswa ini berada dalam kondisi aman. Beberapa hal yang tetap disarankan:
            - 🌟 **Pertahankan Performa**: Dorong untuk terus meningkatkan nilai dan kehadiran.
            - 🎯 **Rencanakan Karier**: Tawarkan mentoring karier dan magang.
            - 👏 **Apresiasi**: Pertimbangkan nominasi untuk program penghargaan mahasiswa berprestasi.
            """)


#
# TAB 2 — Batch Prediction
#
with tab2:
    st.markdown("### 📊 Prediksi Massal dari File CSV")
    st.markdown("""
    Upload file CSV berisi data banyak mahasiswa sekaligus.
    Sistem akan memprediksi risiko dropout untuk setiap mahasiswa secara otomatis.
    """)

    # Template download
    template_data = pd.DataFrame({
        'Tuition_fees_up_to_date'           : [1, 0, 1, 0, 1],
        'Scholarship_holder'                 : [0, 0, 1, 0, 0],
        'Debtor'                             : [0, 1, 0, 1, 0],
        'Curricular_units_1st_sem_approved'  : [6, 1, 5, 2, 4],
        'Curricular_units_1st_sem_grade'     : [13.5, 8.2, 14.1, 7.5, 12.0],
        'Curricular_units_2nd_sem_approved'  : [6, 0, 5, 1, 4],
        'Curricular_units_2nd_sem_grade'     : [13.2, 0.0, 13.8, 6.1, 11.5],
        'Age_at_enrollment'                  : [19, 25, 20, 30, 22],
        'Admission_grade'                    : [135.0, 110.0, 145.0, 108.0, 128.0],
    })

    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            "⬇️ Download Template CSV",
            data=csv_template,
            file_name="template_batch_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Gunakan template ini sebagai panduan format kolom yang diperlukan.")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV Data Mahasiswa",
            type=['csv'],
            help="File harus mengandung kolom sesuai template."
        )

    if uploaded_file:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload: **{len(df_batch):,} mahasiswa** ditemukan.")

            # Validasi kolom
            missing_cols = [c for c in FEATURES if c not in df_batch.columns]
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan: `{', '.join(missing_cols)}`")
                st.stop()

            # Prediksi batch
            with st.spinner("🔄 Memproses prediksi..."):
                X_batch = df_batch[FEATURES].fillna(0)
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]

            # Tambahkan hasil ke dataframe
            df_result = df_batch.copy()
            df_result['Dropout_Probability'] = (probs * 100).round(1)
            df_result['Prediction']          = ['Dropout' if p == 1 else 'Graduate' for p in preds]
            df_result['Risk_Level']          = ['TINGGI' if p >= 0.65 else 'SEDANG' if p >= 0.35 else 'RENDAH'
                                                  for p in probs]

            # Urutkan berdasarkan risiko tertinggi
            df_result = df_result.sort_values('Dropout_Probability', ascending=False).reset_index(drop=True)

            # Summary stats
            n_high   = (probs >= 0.65).sum()
            n_medium = ((probs >= 0.35) & (probs < 0.65)).sum()
            n_low    = (probs < 0.35).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Mahasiswa", f"{len(df_batch):,}", delta=None)
            c2.metric("🔴 Risiko Tinggi",  f"{n_high:,}",   delta=f"{n_high/len(df_batch)*100:.0f}%", delta_color="inverse")
            c3.metric("🟡 Risiko Sedang",  f"{n_medium:,}", delta=f"{n_medium/len(df_batch)*100:.0f}%", delta_color="off")
            c4.metric("🟢 Risiko Rendah",  f"{n_low:,}",   delta=f"{n_low/len(df_batch)*100:.0f}%", delta_color="normal")

            # Distribusi visual
            fig_dist = px.histogram(
                df_result, x='Dropout_Probability', nbins=20,
                color='Risk_Level',
                color_discrete_map={'TINGGI': '#ef4444', 'SEDANG': '#f59e0b', 'RENDAH': '#22c55e'},
                title='Distribusi Probabilitas Dropout — Seluruh Mahasiswa',
                labels={'Dropout_Probability': 'Probabilitas Dropout (%)', 'count': 'Jumlah Mahasiswa'},
                category_orders={'Risk_Level': ['TINGGI', 'SEDANG', 'RENDAH']},
            )
            fig_dist.update_layout(
                height=340, plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Plus Jakarta Sans'},
                legend_title_text='Level Risiko',
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Tabel hasil
            st.markdown("#### 📋 Tabel Hasil Prediksi (Diurutkan: Risiko Tertinggi → Terendah)")

            # Color-code tabel
            def highlight_risk(row):
                if row['Risk_Level'] == 'TINGGI':
                    return ['background-color: #fef2f2'] * len(row)
                elif row['Risk_Level'] == 'SEDANG':
                    return ['background-color: #fffbeb'] * len(row)
                else:
                    return ['background-color: #f0fdf4'] * len(row)

            display_cols = FEATURES + ['Dropout_Probability', 'Prediction', 'Risk_Level']
            st.dataframe(
                df_result[display_cols].style.apply(highlight_risk, axis=1),
                use_container_width=True,
                height=400,
            )

            # Download hasil
            csv_result = df_result.to_csv(index=False)
            st.download_button(
                "⬇️ Download Hasil Prediksi (CSV)",
                data=csv_result,
                file_name="hasil_prediksi_dropout.csv",
                mime="text/csv",
                type="primary",
            )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")

    else:
        st.info("⬆️ Upload file CSV di atas untuk memulai prediksi batch. Gunakan tombol **Download Template** untuk panduan format.")

# Footer
st.markdown("""
<div style='text-align:center; padding:24px 0 8px; color:#94a3b8; font-size:12px;'>
    Jaya Jaya Institut · Sistem Deteksi Dini Dropout · Dikembangkan oleh Kendrick Filbert ·
    Model: LightGBM (Hyperparameter Tuned)
</div>
""", unsafe_allow_html=True)

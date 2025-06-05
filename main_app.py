import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import matplotlib.pyplot as plt
import gdown
import os

file_id = "1yAFMcFBAQo_Q--CUxrdSOHq15nuYVGen"
output = "data_bersih2.csv"

if not os.path.exists(output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)


st.set_page_config(page_title="Prediksi Risiko Korupsi", page_icon="🧾", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #000000;'>🚨 Deteksi Risiko Korupsi Tender Pengadaan</h1>
     <p style='text-align: center; font-size: 18px; margin: auto;'>
        Sistem ini menggunakan model pembelajaran mesin untuk memprediksi potensi risiko korupsi 
        dalam proses tender pengadaan publik berdasarkan berbagai fitur dan data historis. 
        Dengan otomatisasi analisis ini, diharapkan dapat membantu pihak terkait dalam pengawasan 
        dan pengambilan keputusan yang lebih cepat dan tepat.
    </p>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data_bersih2.csv")

data = load_data()

feature_names = [
    'jenis_prosedur', 'jenis_pengadaan', 'jumlah_lot',
    'jumlah_penawaran_terekam', 'harga_estimasi', 'harga_penawaran',
    'status_lot', 'jumlah_penawar', 'negara_instansi', 'tipe_penyedia',
    'penyedia_menang', 'sumber_data', 'tahun_tender', 'harga_digiwhist',
    'filter_instansi_valid', 'filter_penyedia_valid', 'filter_dibatalkan',
    'filter_terbuka', 'filter_tahun_valid', 'filter_penawar_kalah',
    'data_valid', 'durasi_penawaran', 'durasi_keputusan',
    'ada_harga_penawaran'
]

default_features = {
    "sumber_data": "http://inaproc.id/lpse/_tender",
    "status_lot": "AWARDED",
    "negara_instansi": "ID"
}

# Mapping label yang lebih jelas
feature_labels = {
    "jenis_prosedur": "Jenis Prosedur Tender",
    "jenis_pengadaan": "Jenis Barang/Jasa yang Diadakan",
    "jumlah_lot": "Jumlah Lot dalam Tender",
    "jumlah_penawaran_terekam": "Jumlah Penawaran Terekam",
    "harga_estimasi": "Harga Estimasi (Rp)",
    "harga_penawaran": "Harga Penawaran (Rp)",
    "jumlah_penawar": "Jumlah Penawar yang Mengikuti Tender",
    "tipe_penyedia": "Tipe Penyedia",
    "penyedia_menang": "Apakah Ada Penyedia yang Menang?",
    "tahun_tender": "Tahun Tender",
    "harga_digiwhist": "Harga Referensi Digiwhist",
    "filter_instansi_valid": "Apakah Instansi Valid?",
    "filter_penyedia_valid": "Apakah Penyedia Valid?",
    "filter_dibatalkan": "Apakah Tender Dibatalkan?",
    "filter_terbuka": "Apakah Tender Terbuka untuk Umum?",
    "filter_tahun_valid": "Apakah Tahun Valid?",
    "filter_penawar_kalah": "Apakah Ada Penawar yang Kalah?",
    "data_valid": "Apakah Data Valid?",
    "durasi_penawaran": "Durasi Penawaran (hari)",
    "durasi_keputusan": "Durasi Keputusan (hari)",
    "ada_harga_penawaran": "Apakah Ada Harga Penawaran?"
}

# Buat label encoder
def build_label_encoders(df, feature_names):
    le_dict = {}
    for col in feature_names:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    return le_dict

le_dict = build_label_encoders(data.copy(), feature_names)

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

rf_model = load_model("rf_manual_model.pkl")
svm_model = load_model("svm_manual_model.pkl")

st.markdown("## 📥 Input Data Tender")

input_data = {}

for fitur in feature_names:
    if fitur in default_features:
        val = le_dict[fitur].transform([default_features[fitur]])[0]
        input_data[fitur] = val
        continue

    label = feature_labels.get(fitur, fitur.replace("_", " ").capitalize())

    if fitur in [
        "filter_instansi_valid", "filter_penyedia_valid", "filter_dibatalkan",
        "filter_terbuka", "filter_tahun_valid", "filter_penawar_kalah",
        "data_valid", "ada_harga_penawaran"
    ]:
        pilihan = st.selectbox(label, ["Tidak", "Ya"])
        input_data[fitur] = 1 if pilihan == "Ya" else 0

    elif fitur == "penyedia_menang":
        opsi_asli = list(le_dict[fitur].classes_)
        opsi_nama = ["Ada" if x == "t" else "Tidak" if x == "f" else "UNKNOWN" for x in opsi_asli]
        pilihan_user = st.selectbox(feature_labels[fitur], opsi_nama)
        indeks = opsi_nama.index(pilihan_user)
        input_data[fitur] = le_dict[fitur].transform([opsi_asli[indeks]])[0]

    elif fitur in le_dict:
        options = list(le_dict[fitur].classes_)
        pilihan = st.selectbox(label, options)
        input_data[fitur] = le_dict[fitur].transform([pilihan])[0]

    else:
        val = st.number_input(label, value=0.0)
        input_data[fitur] = val

if st.button("🔍 Prediksi Risiko"):
    with st.spinner("⏳ Memproses prediksi..."):
        input_df = pd.DataFrame([input_data])
        pred_rf = rf_model.predict(input_df)[0]
        pred_svm_raw = svm_model.predict(input_df)[0]
        pred_svm = 0 if pred_svm_raw == -1 else 1

    st.markdown("## 📊 Hasil Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest Manual", "Korupsi" if pred_rf == 1 else "Tidak Korupsi", delta="Terdeteksi" if pred_rf == 1 else "Aman")
    with col2:
        st.metric("SVM Manual", "Korupsi" if pred_svm == 1 else "Tidak Korupsi", delta="Terdeteksi" if pred_svm == 1 else "Aman")

    st.markdown("## 🧾 Status Akhir Evaluasi:")
    if pred_rf == 1 or pred_svm == 1:
        st.markdown("### ❌ **Risiko Korupsi TERDETEKSI!**")
        st.error("Data ini menunjukkan kemungkinan adanya penyimpangan dalam tender. Segera lakukan investigasi lanjut.")
    else:
        st.markdown("### ✅ **Tidak Terdeteksi Risiko Korupsi**")
        st.success("Tender ini tampak aman berdasarkan input dan hasil model. Monitoring tetap disarankan.")

    st.markdown("## 🔍 Data yang Anda Masukkan:")
    st.dataframe(input_df)

    input_df['Prediksi_RF'] = "Korupsi" if pred_rf == 1 else "Tidak Korupsi"
    input_df['Prediksi_SVM'] = "Korupsi" if pred_svm == 1 else "Tidak Korupsi"

    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        input_df.to_excel(writer, index=False, sheet_name="Hasil_Prediksi")
    towrite.seek(0)

    with st.expander("🔎 Mengapa tender ini berisiko?"):
        alasan = []

        if input_data.get("jumlah_penawar", 0) == 1:
            alasan.append("💡 Hanya ada satu penawar, potensi kompetisi rendah.")
        if input_data.get("jumlah_lot", 0) == 1:
            alasan.append("📦 Tender hanya memiliki satu lot, berpotensi memusatkan nilai kontrak.")
        if input_data.get("harga_penawaran", 0) == 0 or input_data.get("harga_estimasi", 0) == 0:
            alasan.append("💰 Harga penawaran atau estimasi tidak tercatat, data tidak lengkap.")
        if input_data.get("durasi_penawaran", 0) < 3:
            alasan.append("⏱️ Durasi penawaran terlalu singkat, mengurangi waktu persiapan penawar.")
        if input_data.get("durasi_keputusan", 0) < 1:
            alasan.append("⚖️ Keputusan pemenang diambil terlalu cepat, berisiko tidak objektif.")
        if input_data.get("filter_dibatalkan", 0) == 1:
            alasan.append("🚫 Tender dibatalkan, bisa menjadi indikator inkonsistensi proses.")
        if input_data.get("penyedia_menang", 0) == 0:
            alasan.append("🏗️ Tidak ada penyedia tercatat menang, proses tender tidak selesai.")
        if input_data.get("ada_harga_penawaran", 0) == 0:
            alasan.append("🔍 Tidak ada harga penawaran yang tercatat dalam proses.")
        if input_data.get("filter_penawar_kalah", 0) == 0:
            alasan.append("🏁 Tidak ada data penawar yang kalah, bisa mengindikasikan tender non-kompetitif.")
        if input_data.get("filter_terbuka", 0) == 0:
            alasan.append("🔒 Tender tidak bersifat terbuka, menurunkan transparansi.")
        if input_data.get("data_valid", 0) == 0:
            alasan.append("🛑 Validitas data rendah, sulit mengevaluasi secara objektif.")

        if not alasan:
            alasan.append("✅️ Tender ini aman. Tapi tetap lakukan pengecekan berkala ya!!")

        for a in alasan:
            st.markdown(f"- {a}")

    # fig, ax = plt.subplots()
    # labels = ['Korupsi', 'Tidak Korupsi']
    # sizes = [pred_rf, 1 - pred_rf]
    # colors = ['#d32f2f', '#4caf50']

    # ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    # ax.axis('equal')  
    # st.pyplot(fig)

    st.download_button(
        label="📥 Download Hasil sebagai Excel",
        data=towrite,
        file_name="hasil_prediksi_korupsi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

"""
eda.py

Halaman "EDA" pada aplikasi Streamlit Investor Risk Profile Classification. Menampilkan gambaran umum dataset dan beberapa insight eksplorasi data yang relevan dengan objective project (klasifikasi profil risiko nasabah).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from pathlib import Path

# Mendefinisikan lokasi file dataset dan image berdasarkan lokasi file ini
# (bukan cwd), agar app tetap berjalan benar tidak peduli dari direktori
# mana Streamlit/Docker dijalankan.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "profile_investor.csv"
IMAGE_PATH = BASE_DIR / "docs" / "profile_resiko.png"

@st.cache_data(show_spinner="Memuat dataset...")
def load_data(path: Path) -> pd.DataFrame:
    """
    Membaca dataset dan melakukan caching agar file CSV hanya dibaca sekali per session.
    """
    if not path.exists():
        st.error(
            f"File dataset tidak ditemukan di `{path}`. "
            "Pastikan `profile_investor.csv` ada di folder `data/` "
            "agar halaman EDA dapat berjalan."
        )
        st.stop()
    return pd.read_csv(path)

def run():
  st.title("Application for Investor Risk Profile Prediction")
  # Menampilkan cover gambar
  img = Image.open(IMAGE_PATH)
  st.image(img, caption="Risk Profile Prediction")
  # Menampilkan summary singkat
  st.markdown(
        "Dataset ini berisi profil dan riwayat transaksi investasi nasabah "
        "pada sebuah institusi keuangan. Target klasifikasi (`riskLevel`) "
        "memiliki 4 kelas: **Conservative**, **Income**, **Balanced**, dan "
        "**Aggressive**, dengan distribusi yang tidak seimbang."
    )
  # Proses Load Dataset 
  data = load_data(DATA_PATH)
  # Menampilkan table dataset
  st.subheader("Overview Dataset")
  col1, col2 = st.columns(2)
  col1.metric("Jumlah Baris", f"{data.shape[0]:,}")
  col2.metric("Jumlah Kolom", data.shape[1])
  st.dataframe(data.head(20), use_container_width=True)
  st.markdown("---")
  # Menampilkan visualisasi histogram dari Distribusi target
  st.subheader("Distribusi Target (riskLevel)")
  st.caption(
      "Distribusi kelas tidak seimbang: Income dan Balanced mendominasi, "
      "sementara Aggressive dan Conservative adalah kelas minoritas. "
      "Ketidakseimbangan ini menjadi salah satu alasan penggunaan metrik "
      "F1-Macro selain Accuracy pada tahap evaluasi model."
  )
  fig = plt.figure(figsize=(8, 4))
  sns.countplot(
      data=data,
      x="riskLevel",
      order=data["riskLevel"].value_counts().index,
  )
  plt.xlabel("Profil Risiko")
  plt.ylabel("Jumlah Nasabah")
  st.pyplot(fig)
  st.markdown("---")
  # Menampilkan pilihan untuk melihat distribusi feature numerik
  st.subheader("Distribusi Feature Numerik Berdasarkan Pilihan User")
  numeric_cols = [
      "buy_count",
      "num_unique_assets",
      "total_buy_value",
      "buy_sell_ratio",
      "customer_tenure_days",
  ]
  option = st.selectbox("Pilih kolom numerik:", numeric_cols)
  fig2 = plt.figure(figsize=(8, 4))
  sns.histplot(data[option], bins=30, kde=True)
  plt.xlabel(option)
  st.pyplot(fig2)
  st.markdown("---")
  # Tampilan Kapasitas Investasi vs Profile
  st.subheader("Kapasitas Investasi vs Profil Risiko")
  st.caption(
      "investmentCapacity adalah feature paling berpengaruh berdasarkan "
      "Feature Importance model. Grafik ini menunjukkan bagaimana "
      "kapasitas investasi berhubungan dengan profil risiko aktual nasabah."
  )
  fig3 = px.histogram(
      data,
      x="investmentCapacity",
      color="riskLevel",
      barmode="group",
  )
  st.plotly_chart(fig3, use_container_width=True)
  st.markdown("---")
  # Tampilan Total Nilai Pembelian vs Jml Trx Beli
  st.subheader("Total Nilai Pembelian vs Jumlah Transaksi Beli")
  fig4 = px.scatter(
      data,
      x="buy_count",
      y="total_buy_value",
      color="riskLevel",
      hover_data=["customerType", "customer_tenure_days"],
      log_y=True,
  )
  st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    run()
"""
streamlit_app.py

Entry point aplikasi Streamlit "Investor Risk Profile Classification".
File ini hanya bertugas mengatur navigasi antar halaman (routing),
seluruh logic EDA dan prediksi didelegasikan ke module masing-masing
(eda.py, prediction.py) supaya setiap file punya satu tanggung jawab
yang jelas dan mudah dites/dikembangkan secara independen.
"""

import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title="Investor Risk Profile Classification",
    page_icon="📑",
    layout="wide",
)

# Membuat button choice untuk pindah menu
PAGES = {
    "EDA": eda.run,
    "Prediction": prediction.run,
}

# Tampilan menu
st.sidebar.title("Investor Risk Profile")
st.sidebar.markdown(
    "Decision support tool untuk membantu tim Relationship Manager dan "
    "Compliance/Risk Management memprioritaskan review profil risiko nasabah."
)
page = st.sidebar.radio("Pilih Halaman", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: XGBoost (tuned) | Accuracy 47.80% | F1-Macro 38.08% "
    "pada test set. Bukan pengganti kuesioner resmi MiFID II."
)

PAGES[page]()

"""
prediction.py

Halaman "Prediction" pada aplikasi Streamlit Investor Risk Profile Classification. Mengumpulkan input nasabah baru melalui form, mereplikasi preprocessing (handling cardinality + feature selection) yang identik dengan notebook inference, lalu menjalankan prediksi menggunakan pipeline XGBoost yang sudah dilatih.
"""
# Import Library
import sys
from pathlib import Path
import pandas as pd
import pickle
import streamlit as st
from custom_transformers import IQRCapper

# Definisi lokasi penyimpanan file model, berdasarkan lokasi file ini
# (bukan cwd) agar app tetap berjalan benar tidak peduli dari direktori
# mana Streamlit/Docker dijalankan.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_xgb_tuned.pkl"

# Threshold confidence untuk flag review manual didefinisikan sebesar 50%
CONFIDENCE_THRESHOLD = 0.50
# Mendefinisikan nilai dari feature kategorikal
CUSTOMER_TYPE_OPTIONS = ["Mass", "Premium", "Professional", "Legal Entity", "Inactive"]
INVESTMENT_CAPACITY_OPTIONS = ["CAP_LT30K", "CAP_30K_80K", "CAP_80K_300K", "CAP_GT300K"]
CHANNEL_OPTIONS = ["Branch", "Internet Banking", "Phone Banking"]
ASSET_CATEGORY_OPTIONS = ["Stock", "Bond", "MTF"]

# Loader model (di-cache agar file .pkl hanya dibaca sekali per session)
@st.cache_resource(show_spinner="Memuat model...")
def load_model_artifact(path: Path):
    """
    Memuat model artifact (pipeline, label encoder, daftar feature) dari
    file pickle hasil training.

    Sebelum unpickle dijalankan, class IQRCapper didaftarkan secara
    manual ke sys.modules['__main__']. Ini diperlukan karena pipeline
    di-pickle dari dalam notebook, sehingga class IQRCapper tercatat
    berasal dari module '__main__' pada saat itu (lihat penjelasan
    lengkap di custom_transformers.py).
    """
    sys.modules["__main__"].IQRCapper = IQRCapper
    # Cek apakah file path ditemukan
    if not path.exists():
        st.error(
            f"File model tidak ditemukan di `{path}`. "
            "Pastikan file `model_xgb_tuned.pkl` sudah ada di folder `models/` "
            "sebelum menjalankan aplikasi ini."
        )
        st.stop()

    with open(path, "rb") as f:
        artifact = pickle.load(f)
    # Simpan semua model artifact ke dalam variabel
    required_keys = {"pipeline", "label_encoder", "feature_columns"}
    missing = required_keys - set(artifact.keys())
    if missing:
        st.error(f"Model artifact tidak lengkap, key yang hilang: {missing}")
        st.stop()

    return artifact["pipeline"], artifact["label_encoder"], artifact["feature_columns"]

# Replikasi preprocessing manual(harus identik dengan notebook inference)
def clean_investment_capacity(value: str) -> str:
    """
    Replikasi 'Handling Cardinality' pada investmentCapacity:
    menghapus prefix 'Predicted_' agar kategori tergabung dengan kategori
    non-prediksi yang sama, identik dengan proses pada notebook utama
    dan notebook inference.
    """
    return value.replace("Predicted_", "")

# Membuat function untuk label feature
def build_feature_row(
    customer_type: str,
    investment_capacity: str,
    channel: str,
    asset_category: str,
    buy_count: int,
    sell_count: int,
    num_unique_assets: int,
    num_unique_asset_categories: int,
    total_buy_value: float,
    customer_tenure_days: int,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Menyusun satu baris data siap-prediksi dengan urutan kolom yang
    identik dengan `feature_columns` hasil training (10 kolom akhir
    setelah Feature Selection).

    - Catatan desain penting - buy_sell_ratio:
    Kolom `buy_sell_ratio` bukan diminta langsung ke user (user awam
    tidak akan tahu artinya "rasio"), melainkan dihitung dari dua input
    yaitu: jumlah transaksi beli (buy_count) dan jual
    (sell_count). Berdasarkan pola pada data training/inference asli
    (contoh: buy_count=11, sell_count=7 maka buy_sell_ratio=1.5714),
    formula yang direplikasi adalah buy_count / sell_count.

    Untuk nasabah yang belum pernah menjual (sell_count = 0), data asli
    menunjukkan buy_sell_ratio bernilai 1.0 (bukan infinity/error), maka
    aturan yang sama diterapkan di sini sebagai fallback.
    """
    if sell_count > 0:
        buy_sell_ratio = buy_count / sell_count
    else:
        buy_sell_ratio = 1.0

    row = {
        "customerType": customer_type,
        "investmentCapacity": clean_investment_capacity(investment_capacity),
        "dominant_channel": channel,
        "dominant_asset_category": asset_category,
        "buy_count": buy_count,
        "num_unique_assets": num_unique_assets,
        "num_unique_asset_categories": num_unique_asset_categories,
        "total_buy_value": total_buy_value,
        "buy_sell_ratio": round(buy_sell_ratio, 4),
        "customer_tenure_days": customer_tenure_days,
    }
    # Buat dataframe row
    df_row = pd.DataFrame([row])

    # Pesan keamanan: pastikan urutan & nama kolom identik dengan saat training.
    # Jika ada mismatch (misalnya feature_columns berubah karena retraining),
    # aplikasi berhenti dengan pesan error yang jelas, bukan silent failure yang menghasilkan prediksi salah tanpa disadari.
    assert list(df_row.columns) == list(feature_columns), (
        "Struktur kolom input tidak sesuai dengan feature_columns model. "
        f"Input: {list(df_row.columns)} | Model: {list(feature_columns)}"
    )
    # mengembalika nilai df_row
    return df_row

# Halaman utama
def run():
    st.title("Prediksi Profil Risiko Investasi Nasabah")
    st.caption(
        "Model ini adalah decision support tool, bukan pengganti kuesioner "
        "resmi MiFID II. Gunakan hasil prediksi sebagai alat bantu "
        "prioritisasi review, bukan keputusan akhir compliance."
    )
    # Simpan semua model artifact ke dalam variabel
    pipeline, label_encoder, feature_columns = load_model_artifact(MODEL_PATH)
    # Membuat form input untuk user
    with st.form("form_risk_profile"):
        st.subheader("Profil Nasabah")
        col1, col2 = st.columns(2)
        with col1:
            customer_type = st.selectbox("Tipe Nasabah (customerType)", CUSTOMER_TYPE_OPTIONS)
            investment_capacity = st.selectbox(
                "Kapasitas Investasi (investmentCapacity)", INVESTMENT_CAPACITY_OPTIONS
            )
        with col2:
            channel = st.selectbox("Channel Transaksi Dominan", CHANNEL_OPTIONS)
            asset_category = st.selectbox("Kategori Aset Dominan", ASSET_CATEGORY_OPTIONS)
        st.markdown("---")
        st.subheader("Aktivitas Transaksi")
        col3, col4 = st.columns(2)
        with col3:
            buy_count = st.number_input("Jumlah Transaksi Beli (buy_count)", min_value=0, value=1, step=1)
            num_unique_assets = st.number_input("Jumlah Aset Unik yang Dimiliki", min_value=0, value=1, step=1)
            total_buy_value = st.number_input(
                "Total Nilai Pembelian (total_buy_value)", min_value=0, value=1_000_000, step=100_000
            )
        with col4:
            sell_count = st.number_input("Jumlah Transaksi Jual (sell_count)", min_value=0, value=0, step=1)
            num_unique_asset_categories = st.number_input(
                "Jumlah Kategori Aset Unik (maks. 3: Stock/Bond/MTF)", min_value=1, max_value=3, value=1, step=1
            )
            customer_tenure_days = st.number_input(
                "Lama Menjadi Nasabah Aktif (hari)", min_value=0, value=365, step=1
            )
        # Buat tombol submit untuk mengirim semua data inputan ke Model
        submit = st.form_submit_button("Prediksi Profil Risiko")
    # Cek kondisi apakah submit ditekan
    if not submit:
        return
    # Inisialisasi data requierment
    data_final = build_feature_row(
        customer_type=customer_type,
        investment_capacity=investment_capacity,
        channel=channel,
        asset_category=asset_category,
        buy_count=int(buy_count),
        sell_count=int(sell_count),
        num_unique_assets=int(num_unique_assets),
        num_unique_asset_categories=int(num_unique_asset_categories),
        total_buy_value=float(total_buy_value),
        customer_tenure_days=int(customer_tenure_days),
        feature_columns=feature_columns,
    )
    # Menjalankan proses prediksi
    pred_encoded = pipeline.predict(data_final)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    pred_proba = pipeline.predict_proba(data_final)[0]
    proba_map = dict(zip(label_encoder.classes_, pred_proba))
    confidence = max(pred_proba)
    perlu_review = confidence < CONFIDENCE_THRESHOLD

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col_result, col_flag = st.columns(2)
    with col_result:
        st.metric("Prediksi Profil Risiko", pred_label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    with col_flag:
        if perlu_review:
            st.warning(
                f"Perlu Review Manual — confidence di bawah threshold "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}%."
            )
        else:
            st.success("Confidence di atas threshold, tidak otomatis ditandai untuk review.")

    st.write("Probabilitas per kelas:")
    df_proba = pd.DataFrame(
        {"Profil Risiko": list(proba_map.keys()), "Probabilitas": list(proba_map.values())}
    ).sort_values("Probabilitas", ascending=False).reset_index(drop=True)
    df_proba["Probabilitas"] = df_proba["Probabilitas"].apply(lambda x: f"{x * 100:.2f}%")
    st.dataframe(df_proba, use_container_width=True, hide_index=True)

    with st.expander("Lihat data final yang dikirim ke model"):
        st.dataframe(data_final, use_container_width=True)

    st.caption(
        "Catatan: berdasarkan hasil evaluasi model, recall pada kelas "
        "Aggressive dan Conservative masih rendah. Prediksi dengan confidence "
        "rendah, atau nasabah dengan aktivitas transaksi sangat minim dan "
        "tenure panjang, sebaiknya tetap direview manual meski tidak "
        "otomatis ditandai oleh threshold di atas."
    )

# Inisiasi main program
if __name__ == "__main__":
    run()

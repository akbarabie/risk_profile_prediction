"""
custom_transformers.py

Berisi custom transformer sklearn yang digunakan di dalam Pipeline model
(model_xgb_tuned.pkl). Class ini WAJIB tersedia sebelum pickle.load()
dipanggil, karena pipeline yang di-pickle menyimpan referensi ke class
ini, bukan menyalin isinya.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer untuk melakukan capping (winsorizing) terhadap
    nilai ekstrem pada kolom numerik menggunakan metode IQR (Interquartile
    Range).

    Batas bawah dan batas atas dihitung hanya dari data training saat
    fit(), lalu batas yang sama digunakan untuk memotong (clip) nilai
    ekstrem baik pada data training maupun data baru saat transform() untuk mencegah data leakage.
    """

    def __init__(self, factor: float = 1.5):
        # factor menentukan seberapa jauh batas capping dari Q1/Q3.
        # Nilai default 1.5 mengikuti konvensi umum metode IQR (Tukey's rule).
        self.factor = factor

    def fit(self, X):
        X = pd.DataFrame(X)

        # Hitung Q1, Q3, dan batas bawah/atas HANYA dari data training
        self.Q1_ = X.quantile(0.25)
        self.Q3_ = X.quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        self.lower_bound_ = self.Q1_ - self.factor * self.IQR_
        self.upper_bound_ = self.Q3_ + self.factor * self.IQR_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # Lakukan capping menggunakan batas yang sudah dihitung saat fit
        for kolom in X.columns:
            X[kolom] = X[kolom].clip(
                lower=self.lower_bound_[kolom], upper=self.upper_bound_[kolom]
            )
        return X.values

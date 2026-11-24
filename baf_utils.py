"""Fungsi pembantu untuk Kasus A (BAF). Diimpor oleh notebook dan dibutuhkan saat memuat model."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

LABEL = "fraud_bool"

# Kolom kategorikal (string) pada BAF
CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# Kolom numerik yang dipakai sebagai fitur
NUM_COLS = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "bank_months_count", "has_other_cards", "proposed_credit_limit",
    "foreign_request", "session_length_in_minutes", "keep_alive_session",
    "device_distinct_emails_8w",
]

# Kolom yang tidak dipakai: label, bulan (bukan fitur yang tersedia saat skor
# pengajuan baru), dan device_fraud_count (konstan = 0 di varian Base)
DROP_COLS = [LABEL, "month", "device_fraud_count"]

# Pada BAF, nilai hilang ditulis sebagai -1 (kolom di bawah) atau angka negatif
MINUS1_MISSING = [
    "prev_address_months_count", "current_address_months_count",
    "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w",
]
NEGATIVE_MISSING = ["intended_balcon_amount"]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buang kolom non-fitur dan ubah kode 'hilang' (-1 / negatif) menjadi NaN."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    for c in MINUS1_MISSING:
        if c in df.columns:
            df[c] = df[c].where(df[c] != -1, np.nan)
    for c in NEGATIVE_MISSING:
        if c in df.columns:
            df[c] = df[c].where(df[c] >= 0, np.nan)
    return df


def tpr_at_fpr(y_true, scores, fpr_target: float = 0.05):
    """TPR tertinggi dengan FPR <= fpr_target, beserta ambang skor (flag jika skor >= ambang)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[idx]), float(thr[idx])

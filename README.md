# Kasus A — Deteksi Penipuan Pembukaan Rekening (BAF Base, bulan 0–3)

**Tujuan:** memberi skor kemungkinan penipuan tiap pengajuan baru; tangkap penipu sebanyak mungkin dengan nasabah jujur tertandai ≤ 5%. **Metrik:** TPR pada FPR 5%.

## Metode
- **Data:** BAF varian Base, hanya baris `month` 0–3 (di-*filter* per chunk).
- **Pra-pemrosesan:** kode nilai hilang (`-1`, dan nilai negatif pada `intended_balcon_amount`) diubah menjadi `NaN`; kolom `month`, `device_fraud_count` (konstan), dan label dibuang dari fitur.
- **Model:** `HistGradientBoostingClassifier` (scikit-learn) dalam satu `Pipeline` bersama pembersihan fitur, sehingga menerima data mentah. Kategorikal ditangani native oleh model.
- **Alasan memilih:** data tabular campuran (numerik + kategorikal) dengan relasi non-linear dan banyak nilai hilang; GBM menangkap interaksi antar-fitur, menangani `NaN` secara native, dan cepat dilatih. Regresi logistik dipakai sebagai baseline.
- **Ketidakseimbangan kelas (~1,1% penipu):** tidak di-*resampling*. Metrik berbasis peringkat (TPR@FPR) tidak butuh kelas seimbang; opsi `class_weight` ikut diuji saat tuning.
- **Ambang:** skor ≥ ambang (disimpan di `models/baf_threshold.json`) ditandai untuk verifikasi manual; ambang dipilih agar FPR validasi ≤ 5%.

## Pembagian data
Berdasarkan **waktu**: latih = bulan 0–2, validasi = bulan 3. Alasan: di produksi model menilai pengajuan masa depan, pola penipuan berubah dari waktu ke waktu, dan pembagian acak membuat skor terlalu optimis. Tuning hyperparameter memakai latih bulan 0–1 → uji bulan 2, sehingga bulan 3 hanya dipakai sekali untuk pelaporan.

## Skor di data validasi (bulan 3)
| Model | TPR @ FPR 5% | ROC-AUC |
|---|---|---|
| Regresi logistik (baseline) | [ISI] | [ISI] |
| **GBM (final)** | **[ISI]** (95% CI [ISI]–[ISI]) | [ISI] |

## Cara menjalankan
```bash
pip install -r requirements.txt
# letakkan Base.csv (Kaggle) di folder data/
jupyter notebook baf_fraud_detection.ipynb   # Run All
```
Muat ulang model: `joblib.load("models/baf_gbm_pipeline.joblib")` (folder harus berisi `baf_utils.py`). Contoh pemakaian ada di bagian 8 notebook.

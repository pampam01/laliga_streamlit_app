# Streamlit - Testing Model Prediksi LaLiga

Folder ini berisi aplikasi Streamlit untuk **menguji** 2 model (.joblib):

- `model_points_laliga.joblib`
- `model_goal_diff_laliga.joblib`

Aplikasi sudah ada **mode demo 1-klik** (dosen penguji tinggal klik) dan juga opsi **upload dataset**.

## Struktur folder yang disarankan

Letakkan model di folder `models/` supaya otomatis terdeteksi.

```
.
├─ app.py
├─ requirements.txt
├─ preprocessing_laliga.joblib    (opsional tapi recommended)
└─ models/
   ├─ model_points_laliga.joblib
   └─ model_goal_diff_laliga.joblib
```

## Cara menjalankan

1) Install dependency

```bash
pip install -r requirements.txt
```

2) Jalankan app

```bash
streamlit run app.py
```

## Catatan penting soal preprocessing

Model Anda di notebook memakai:
- `StandardScaler()` untuk scaling fitur
- `LabelEncoder()` untuk encoding nama team

Supaya hasil **paling konsisten** dengan training, sebaiknya Anda juga menyimpan preprocessing tersebut ke file:
- `preprocessing_laliga.joblib`

Di app ini Anda bisa membuat file tersebut lewat tab **Export Preprocessing Bundle** (upload dataset historis lalu klik generate).

Kalau bundle tidak ada dan Anda juga tidak upload historical, app tetap bisa jalan menggunakan mode fallback/demo, tapi hasilnya bisa berbeda dari skripsi.

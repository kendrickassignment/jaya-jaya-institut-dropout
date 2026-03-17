"""
pd.py — Konversi Dataset untuk Upload ke Looker Studio

Script ini mengubah format separator CSV dari titik koma (;) ke koma (,)
yang merupakan format standar yang dibutuhkan oleh Looker Studio.

Cara pakai:
    python pd.py

Output:
    data_looker_studio.csv (boleh direname menjadi data.csv) — siap diupload ke Looker Studio
"""

import pandas as pd
import os

# Konfigurasi
INPUT_FILE  = 'data.csv'
OUTPUT_FILE = 'data_looker_studio.csv'

# Load & Validasi
if not os.path.exists(INPUT_FILE):
    print(f"❌ File '{INPUT_FILE}' tidak ditemukan.")
    print("   Pastikan Anda menjalankan script ini di direktori yang sama dengan data.csv")
    exit(1)

print(f"📂 Membaca '{INPUT_FILE}'...")
df = pd.read_csv(INPUT_FILE, sep=';')
print(f"   ✅ {len(df):,} baris, {len(df.columns)} kolom ditemukan.")

# Simpan dengan format koma
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\n✅ File berhasil dikonversi dan disimpan sebagai '{OUTPUT_FILE}'")
print(f"   → Siap diupload ke Looker Studio!")

# Info ringkas
print(f"\n📊 Ringkasan Data:")
print(f"   Total mahasiswa : {len(df):,}")
print(f"   Kolom           : {len(df.columns)}")
for status, count in df['Status'].value_counts().items():
    print(f"   {status:<12}: {count:,} ({count/len(df)*100:.1f}%)")

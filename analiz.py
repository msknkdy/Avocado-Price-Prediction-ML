import sys
print("Boss, sistem hazır!")
print("Kullanılan Sürüm:", sys.version)
import pandas as pd

# 1. Veriyi yükle
df = pd.read_csv('avocado.csv')

# 2. Verinin genel yapısını kontrol et
print("--- Verinin İlk 5 Satırı ---")
print(df.head())

# 3. Hangi sütunlar var ve veri tipleri neler?
print("\n--- Veri Bilgisi ---")
print(df.info())

# 4. Eksik veri var mı?
print("\n--- Eksik Veri Sayısı ---")
print(df.isnull().sum())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. VERİ YÜKLEME VE ÖN İŞLEME [cite: 7-21, 84-85]
path = '/Users/ekineryigit/Desktop/avocado project/avocado.csv'
df = pd.read_csv(path)

# Gereksiz sütunları temizleme [cite: 19]
df = df.drop(['Unnamed: 0', 'Date'], axis=1) 

# Özellikler ve Hedef Değişken [cite: 10]
X = df.drop('AveragePrice', axis=1)
y = df['AveragePrice']

# Veriyi Bölme (80% Train, 20% Test) [cite: 35, 38]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kategorik Değişkenlerin Dönüştürülmesi (Encoding) [cite: 18]
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test).reindex(columns=X_train_encoded.columns, fill_value=0)

# 2. MODEL KARŞILAŞTIRMA (Yönerge Madde 9 & 10) [cite: 26, 27]
print("\n--- Model Karşılaştırma Sonuçları ---")
models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    preds = model.predict(X_test_encoded)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"> {name} -> MAE: {mae:.4f}, R-Squared: {r2:.4f}")

# 3. FINAL MODEL VE HIPERPARAMETRE AYARI [cite: 28, 33]
# Not: GridSearchCV ile daha önce en iyi parametreler tespit edilmiştir.
# Tespit edilen parametreler: max_depth=None, min_samples_split=10
print("\n--- En İyi Parametrelerle Final Model Eğitiliyor ---")

final_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=10, 
    random_state=42
)
final_model.fit(X_train_encoded, y_train)

# Final Metrikler [cite: 51, 56]
y_pred_final = final_model.predict(X_test_encoded)
print(f"Final Random Forest MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")
print(f"Final Random Forest R2: {r2_score(y_test, y_pred_final):.4f}")

# 4. GÖRSELLEŞTİRME (Yönerge Madde 23 & 24) [cite: 65, 67]
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_final, color='#BE6EE0', alpha=0.4, label='Predictions') 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line')
plt.title('Actual vs Predicted Avocado Prices (Random Forest)', fontsize=14)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Hata Dağılım Grafiği
residuals = y_test - y_pred_final
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='#BE6EE0', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', lw=2)
plt.title('Distribution of Residuals (Errors)', fontsize=14)
plt.xlabel('Error Amount')
plt.ylabel('Frequency')
plt.show()
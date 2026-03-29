import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. VERİ YÜKLEME VE ÖN İŞLEME
# Not: path kısmını kendi dosya yoluna göre kontrol etmeyi unutma Boss
path = 'avocado.csv' 
df = pd.read_csv(path)

# Gereksiz sütunları temizleme
df = df.drop(['Unnamed: 0', 'Date'], axis=1) 

# Özellikler ve Hedef Değişken
X = df.drop('AveragePrice', axis=1)
y = df['AveragePrice']

# Kategorik Değişkenlerin Dönüştürülmesi (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Veriyi Bölme (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjusted R2 Fonksiyonu
def get_adj_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# 2. MODEL KARŞILAŞTIRMA (Burası raporundaki Tablo 7'yi dolduracak)
print("\n--- Model Karşılaştırma Sonuçları ---")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

n_test = len(y_test)
k_features = X_test.shape[1]

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    adj_r2 = get_adj_r2(r2, n_test, k_features)
    
    print(f"\n> {name}:")
    print(f"  MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
    print(f"  R-Squared: {r2:.4f} | Adjusted R2: {adj_r2:.4f}")

# 3. HİPERPARAMETRE AYARI (GridSearchCV)
print("\n--- Hyperparameter Tuning (GridSearch) Başlatılıyor... ---")
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Final modelimizi en iyi parametrelerle alıyoruz
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test)
residuals = y_test - y_pred_final

print(f"\nEn İyi Parametreler: {grid_search.best_params_}")
print(f"Final Adjusted R2: {get_adj_r2(r2_score(y_test, y_pred_final), n_test, k_features):.4f}")

# 4. FİNAL GÖRSELLEŞTİRME (Yan yana iki grafik: İkisini de tek seferde alacağız)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Sol Grafik: Actual vs Predicted
ax1.scatter(y_test, y_pred_final, color='#BE6EE0', alpha=0.4, label='Predictions')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line')
ax1.set_title('Final Model: Actual vs Predicted Prices', fontsize=14)
ax1.set_xlabel('Actual Price ($)', fontsize=12)
ax1.set_ylabel('Predicted Price ($)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Sağ Grafik: Distribution of Residuals (Çan Eğrisi)
sns.histplot(residuals, bins=30, kde=True, color='#BE6EE0', ax=ax2)
ax2.axvline(0, color='red', linestyle='--', lw=2)
ax2.set_title('Distribution of Residuals (Error Analysis)', fontsize=14)
ax2.set_xlabel('Error Amount', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show() # Bu satırda iki grafik birden açılacak 

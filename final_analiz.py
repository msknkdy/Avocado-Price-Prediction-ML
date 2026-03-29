import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = 'avocado.csv' 
df = pd.read_csv(path)
df = df.drop(['Unnamed: 0', 'Date'], axis=1) 

X = df.drop('AveragePrice', axis=1)
y = df['AveragePrice']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def get_adj_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

print("\n--- Model Karşılaştırma Sonuçları ---")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
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

print("\n--- Hyperparameter Tuning (GridSearch) Başlatılıyor... ---")
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 20],
    'min_samples_split': [2, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv_strategy, scoring='r2')
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test)
residuals = y_test - y_pred_final

print(f"\nEn İyi Parametreler: {grid_search.best_params_}")
print(f"Final Adjusted R2: {get_adj_r2(r2_score(y_test, y_pred_final), n_test, k_features):.4f}")

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6), num=1)
plt.scatter(y_test, y_pred_final, color='#BE6EE0', alpha=0.3, s=15, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line')
plt.title('Figure 1: Final Model - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('figure1_actual_vs_predicted.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6), num=2)
sns.histplot(residuals, bins=50, kde=True, color='#BE6EE0', edgecolor='white')
plt.axvline(0, color='red', linestyle='--', lw=2)
plt.title('Figure 2: Distribution of Residuals (Error Analysis)', fontsize=14, fontweight='bold')
plt.xlabel('Error Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('figure2_residuals.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6), num=3)
importances = final_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Figure 3: Top 10 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure3_feature_importance.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 8), num=4)
numeric_df = df.select_dtypes(include=[np.number]).drop(['Unnamed: 0'], axis=1, errors='ignore')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Figure 4: Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure4_correlation_heatmap.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6), num=5)
sns.boxplot(x='type', y='AveragePrice', data=df, palette='magma')
plt.title('Figure 5: Average Price Distribution by Avocado Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure5_price_boxplot.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6), num=6)
sns.regplot(x='Total Volume', y='AveragePrice', data=df.sample(2000), 
            scatter_kws={'alpha':0.3, 'color':'#BE6EE0'}, line_kws={'color':'red'})
plt.title('Figure 6: Total Volume vs Average Price Trend', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure6_volume_price_trend.png', dpi=300)
plt.show()

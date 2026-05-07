import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

# =====================
# 1. LOAD DATA
# =====================
file_path = r"models-6-5\three_months_model\three_month_final.csv"
df = pd.read_csv(file_path)

# =====================
# 2. CLEAN COLUMN NAMES
# =====================
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

# =====================
# 3. DEFINE TARGET
# =====================
target_col = "players_month_3_after_release"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

# Drop rows where target is missing
df = df.dropna(subset=[target_col])

# =====================
# 4. DROP IRRELEVANT COLUMNS
# =====================
drop_cols = [
    col for col in df.columns
    if "name" in col.lower() or "app_id" in col.lower()
]

# =====================
# 5. ENCODE CATEGORICALS
# =====================
categorical_cols = ["genre", "categories"]

for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# =====================
# 6. SPLIT FEATURES & TARGET
# =====================
X = df.drop(columns=drop_cols + [target_col], errors="ignore")
y = df[target_col]

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# =====================
# 7. HANDLE MISSING VALUES
# =====================
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =====================
# 8. FEATURE TRANSFORMATIONS
# =====================
# Log transform skewed features
for col in X.columns:
    if "review" in col.lower() or "player" in col.lower():
        X[col] = np.log1p(X[col])

# Log transform target (IMPORTANT)
y = np.log1p(y)

# =====================
# 9. TRAIN-TEST SPLIT (85/15)
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# =====================
# 10. MODEL
# =====================
model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

model_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# =====================
# 11. TRAIN
# =====================
model.fit(X_train, y_train)

model_rf.fit(X_train, y_train)


# =====================
# 12. PREDICT
# =====================
y_pred = model.predict(X_test)

y_pred_rf = model_rf.predict(X_test)

# =====================
# 13. EVALUATE
# =====================

print("Gradient Boosting Regressor Performance:")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE (log scale): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nRandom Forest Regressor Performance:")

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE (log scale): {mae_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")

print(y.describe())


# # =====================
# # 14. FEATURE IMPORTANCE
# # =====================
# importances = pd.Series(model.feature_importances_, index=X.columns)
# print("\nTop 10 Important Features:")
# print(importances.sort_values(ascending=False).head(10))
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

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
# 3. DEFINE REQUIRED COLUMNS
# =====================
month3_col = "players_month_3_after_release"
week1_col = "players_7days_after_release"

required_cols = [month3_col, week1_col]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found.")

# Drop rows where required values are missing
df = df.dropna(subset=required_cols)

# =====================
# 4. FILTER LOW EARLY PLAYER COUNT
# =====================
print(f"Rows before filtering: {len(df)}")

df = df[df[week1_col] >= 100]

print(f"Rows after filtering: {len(df)}")

# =====================
# 5. CREATE RETENTION TARGET
# =====================
df["retention_ratio"] = df[month3_col] / df[week1_col]

target_col = "retention_ratio"

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target_col])

# Optional: remove extreme retention outliers
# This helps avoid tiny or unusual games dominating the model
lower_bound = df[target_col].quantile(0.01)
upper_bound = df[target_col].quantile(0.99)

df = df[
    (df[target_col] >= lower_bound) &
    (df[target_col] <= upper_bound)
]

print(f"Rows after removing extreme retention outliers: {len(df)}")

# =====================
# 6. DROP IRRELEVANT / LEAKAGE COLUMNS
# =====================
drop_cols = [
    col for col in df.columns
    if "name" in col.lower() or "app_id" in col.lower()
]

# IMPORTANT:
# month3_col must be dropped because it is future information.
leakage_cols = [
    month3_col
]

# =====================
# 7. ENCODE CATEGORICALS
# =====================
categorical_cols = ["genre", "categories"]

for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# =====================
# 8. SPLIT FEATURES & TARGET
# =====================
X = df.drop(columns=drop_cols + leakage_cols + [target_col], errors="ignore")
y = df[target_col]

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# =====================
# 9. HANDLE MISSING VALUES
# =====================
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =====================
# 10. FEATURE TRANSFORMATIONS
# =====================
# Log transform skewed input features
for col in X.columns:
    if "review" in col.lower() or "player" in col.lower():
        X[col] = np.log1p(X[col])

# Log transform target
# This makes retention ratios easier to model, especially when skewed.
y = np.log1p(y)

# =====================
# 11. TRAIN-TEST SPLIT (85/15)
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# =====================
# 12. MODEL
# =====================
model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

# =====================
# 13. TRAIN
# =====================
model.fit(X_train, y_train)

# =====================
# 14. PREDICT
# =====================
y_pred = model.predict(X_test)

# =====================
# 15. EVALUATE ON LOG SCALE
# =====================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE (log retention ratio): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# =====================
# 16. CONVERT BACK TO ACTUAL RETENTION RATIO
# =====================
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"MAE (actual retention ratio): {mae_actual:.4f}")

# Avoid division by zero in MAPE
mape = np.mean(
    np.abs((y_test_actual - y_pred_actual) / y_test_actual)
) * 100

print(f"MAPE: {mape:.2f}%")

# =====================
# 17. FEATURE IMPORTANCE
# =====================
importances = pd.Series(model.feature_importances_, index=X.columns)

print("\nTop 10 Important Features:")
print(importances.sort_values(ascending=False).head(10))

# =====================
# 18. OPTIONAL BUSINESS-FRIENDLY OUTPUT
# =====================
comparison = pd.DataFrame({
    "actual_retention_ratio": y_test_actual,
    "predicted_retention_ratio": y_pred_actual,
    "absolute_error": np.abs(y_test_actual - y_pred_actual)
})

print("\nSample Predictions:")
print(comparison.head(10))
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

for col in [month3_col, week1_col]:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found.")

df = df.dropna(subset=[month3_col, week1_col])

# =====================
# 4. FILTER LOW EARLY PLAYER COUNT
# =====================
print(f"Rows before filtering: {len(df)}")

df = df[df[week1_col] >= 100]

print(f"Rows after filtering: {len(df)}")

# =====================
# 5. CREATE TARGET: PLAYER INCREASE %
# =====================
df["player_increase_pct"] = (
    (df[month3_col] - df[week1_col]) / df[week1_col]
) * 100

target_col = "player_increase_pct"

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target_col])

# Optional: remove extreme outliers
lower_bound = df[target_col].quantile(0.01)
upper_bound = df[target_col].quantile(0.99)

df = df[
    (df[target_col] >= lower_bound) &
    (df[target_col] <= upper_bound)
]

print(f"Rows after removing extreme growth outliers: {len(df)}")

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

X = X.select_dtypes(include=[np.number])

# =====================
# 9. HANDLE MISSING VALUES
# =====================
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =====================
# 10. FEATURE TRANSFORMATIONS
# =====================
for col in X.columns:
    if "review" in col.lower() or "player" in col.lower():
        X[col] = np.log1p(X[col])

# NOTE:
# Do NOT use np.log1p(y) here because percentage increase can be negative.

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
# 15. EVALUATE
# =====================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE (percentage points): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# =====================
# 16. BUSINESS-FRIENDLY INTERPRETATION
# =====================
print(f"\nInterpretation:")
print(f"On average, the model is off by {mae:.2f} percentage points in predicted player growth.")

# =====================
# 17. FEATURE IMPORTANCE
# =====================
importances = pd.Series(model.feature_importances_, index=X.columns)

print("\nTop 10 Important Features:")
print(importances.sort_values(ascending=False).head(10))

# =====================
# 18. SAMPLE PREDICTIONS
# =====================
comparison = pd.DataFrame({
    "actual_player_increase_pct": y_test,
    "predicted_player_increase_pct": y_pred,
    "absolute_error_percentage_points": np.abs(y_test - y_pred)
})

print("\nSample Predictions:")
print(comparison.head(100))
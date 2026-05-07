import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# =====================
# 1. LOAD DATA
# =====================
file_path = r"models-6-5\twelve_months_model\twelve_month_final.csv"
df = pd.read_csv(file_path)

# =====================
# 2. CLEAN COLUMN NAMES
# =====================
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

# =====================
# 3. DEFINE TARGET
# =====================
target_col = "players_month_12_after_release"

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
# 9.5 SCALE FEATURES (ONLY FOR LINEAR REGRESSION)
# =====================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # fit ONLY on train
X_test = scaler.transform(X_test)         # apply same scaling to test

# =====================
# 10. MODEL
# =====================
model_lr = Lasso(alpha=0.01)

# =====================
# 11. TRAIN
# =====================
model_lr.fit(X_train, y_train)


# =====================
# 12. PREDICT
# =====================
y_pred_lr = model_lr.predict(X_test)

# =====================
# 13. EVALUATE
# =====================

print("\nLinear Regression Performance:")

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MAE (log scale): {mae_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")

print(y.describe())


# =====================
# 14. FEATURE IMPORTANCE
# =====================
importances = pd.Series(model_lr.coef_, index=X.columns)
print("\nTop 10 Important Features:")
print(importances.sort_values(ascending=False).head(10))
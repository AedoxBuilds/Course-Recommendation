"""
Course recommendation - single file pipeline.

Steps (in this order, nothing else):
  1. Preprocess course_recommendation_dataset.csv
  2. Save the preprocessed data as CSV
  3. Load that preprocessed CSV
  4. Split into train / test
  5. Create the model
  6. Train
  7. Evaluate on test
  8. Save the model
"""

import joblib
import pandas as pd

from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

set_config(transform_output="pandas")

# ==========================================
# CONFIG
# ==========================================
FILE_PATH = "course_recommendation_dataset.csv"
TARGET = "recommended_course"
DROP_COLUMNS = ["student_id"]            # ID column, not a feature
PROCESSED_CSV = "preprocessed_dataset.csv"
PREPROCESSOR_FILE = "preprocessor.joblib"
MODEL_FILE = "course_model.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==========================================
# 1. PREPROCESSING
# ==========================================
df = pd.read_csv(FILE_PATH)
df = df.drop(columns=DROP_COLUMNS, errors="ignore")
# NOTE: no drop_duplicates() here. Once student_id is removed, different
# students with the same answers look identical, and they are valid rows.
df = df.dropna(subset=[TARGET])

# In this dataset an empty previous_skill_area means "no previous skill"
# (it is empty exactly when has_previous_digital_skill == "No"),
# so it is a real category, not missing data.
df["previous_skill_area"] = df["previous_skill_area"].fillna("None")

X = df.drop(columns=[TARGET])
y = df[TARGET]
X = X.loc[:, X.nunique(dropna=False) > 1]            # drop constant columns

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

X_processed = preprocessor.fit_transform(X)

# ==========================================
# 2. SAVE PREPROCESSED CSV
# ==========================================
processed = X_processed.copy()
processed[TARGET] = y.values
processed.to_csv(PROCESSED_CSV, index=False)
joblib.dump(preprocessor, PREPROCESSOR_FILE)         # reuse on new data later
print(f"Saved {PROCESSED_CSV}: {processed.shape}")

# ==========================================
# 3. LOAD PREPROCESSED CSV
# ==========================================
data = pd.read_csv(PROCESSED_CSV)
X = data.drop(columns=[TARGET])
y = data[TARGET]

# ==========================================
# 4. SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ==========================================
# 5. CREATE MODEL
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=50,          # keeps the forest from memorizing noisy labels
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ==========================================
# 6. TRAIN
# ==========================================
model.fit(X_train, y_train)

# ==========================================
# 7. EVALUATE
# ==========================================
train_acc = accuracy_score(y_train, model.predict(X_train))
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}\n")
print(classification_report(y_test, y_pred))

# ==========================================
# 8. SAVE MODEL
# ==========================================
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
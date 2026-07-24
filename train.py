"""Trains the grid stability model from data/sg_raw_data.csv and writes
the four artifacts app.py loads: grid_stability_model.pkl, scaler.pkl,
label_encoder.pkl, feature_columns.pkl.
"""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4",
]

df = pd.read_csv("data/sg_raw_data.csv")

X = df[FEATURE_COLUMNS]
y_raw = df["stabf"]  # "stab" is dropped: it's the continuous value stabf is derived from (leakage)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(model, "grid_stability_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")
print("Artifacts written.")

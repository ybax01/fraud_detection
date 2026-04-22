import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import os

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")

# ─────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────
# Scale Amount and Time (V1-V28 are already scaled by the bank)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

# Drop original unscaled columns
df = df.drop(columns=["Amount", "Time"])

# Features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
# stratify=y ensures both splits keep the 0.17% fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Fraud in test    : {y_test.sum()} transactions")

# ─────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────
print("\nTraining Random Forest... (this may take 1-2 minutes)")

model = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees
    class_weight="balanced", # handles the 0.17% imbalance automatically
    random_state=42,
    n_jobs=-1                # use all CPU cores to speed up training
)

model.fit(X_train, y_train)
print("Training complete!")

# ─────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of fraud

auc = roc_auc_score(y_test, y_pred_prob)

print("\n─── Results ───────────────────────────")
print(f"AUC-ROC Score : {auc:.4f}  (target: > 0.95)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# ─────────────────────────────────────────
# 6. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (AUC-ROC: {auc:.4f})")
plt.tight_layout()
plt.savefig("app/models/confusion_matrix.png")
plt.show()
print("Confusion matrix saved.")

# ─────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────
os.makedirs("app/models", exist_ok=True)
joblib.dump(model, "app/models/saved_model.pkl")
joblib.dump(scaler, "app/models/scaler.pkl")  # save scaler too (needed in Flask)

print("\nModel saved to app/models/saved_model.pkl")
print("Scaler saved to app/models/scaler.pkl")
print("\nAll done! You can now build the Flask app.")
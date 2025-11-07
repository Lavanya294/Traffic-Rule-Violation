# =========================================================
# 🚦 Traffic Violation Prediction (EDA + Modeling + Tuning)
# Dataset: merged_traffic_dataset.csv
# Author: Your Name
# =========================================================

# --- 1️⃣ Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# --- 2️⃣ Load Dataset ---
DATA_PATH = "Indian_Traffic_Violations.csv"
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded:", df.shape)
df.head()

# --- 3️⃣ Quick Data Overview ---
print("\n--- Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\n--- Sample Rows ---")
print(df.sample(5))


# --- 4️⃣ Map Violation_Type into broader target groups ---
def map_violation_type(val):
    if pd.isna(val): return "Others"
    s = str(val).lower()
    if "speed" in s or "over-speed" in s or "overspeed" in s:
        return "Over-Speed"
    if "drunk" in s or "alcohol" in s:
        return "Drunken Driving"
    if "wrong" in s and "side" in s:
        return "Driving on Wrong Side"
    if "signal" in s or "red" in s or "jump" in s:
        return "Jumping Red Light / Signal"
    if "mobile" in s or "phone" in s or "distract" in s:
        return "Use of Mobile / Distracted Driving"
    return "Others"

df['target_group'] = df['Violation_Type'].apply(map_violation_type)

print("\n--- Target Distribution ---")
print(df['target_group'].value_counts())

# --- 5️⃣ Feature Engineering ---
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['month'] = df['Date'].dt.month.fillna(0).astype(int)
    df['dayofweek'] = df['Date'].dt.dayofweek.fillna(0).astype(int)
else:
    df['month'] = 0
    df['dayofweek'] = 0

# --- 6️⃣ Exploratory Data Analysis (EDA) ---
plt.figure(figsize=(7,5))
sns.countplot(y='target_group', data=df, order=df['target_group'].value_counts().index)
plt.title("Distribution of Violation Categories")
plt.xlabel("Count")
plt.ylabel("Violation Type")
plt.show()

if 'Vehicle_Type' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(y='Vehicle_Type', data=df, order=df['Vehicle_Type'].value_counts().index[:10])
    plt.title("Top Vehicle Types Involved")
    plt.show()

if 'Driver_Age' in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df['Driver_Age'].dropna(), kde=True, bins=20)
    plt.title("Driver Age Distribution")
    plt.show()

if 'Fine_Amount' in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df['Fine_Amount'], bins=30, kde=True)
    plt.title("Distribution of Fine Amounts")
    plt.show()

# Correlation heatmap (numerical only)
plt.figure(figsize=(8,5))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# --- 7️⃣ Feature Selection ---
features = []
if 'Registration_State' in df.columns: features.append('Registration_State')
if 'Vehicle_Type' in df.columns: features.append('Vehicle_Type')
if 'Fine_Amount' in df.columns: features.append('Fine_Amount')
if 'Driver_Age' in df.columns: features.append('Driver_Age')
features += ['month','dayofweek']

print("\n✅ Features selected:", features)

# --- 8️⃣ Preprocessing ---
X = df[features].copy()
y = df['target_group'].copy()

le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

X_enc = X.copy()
label_encoders = {}
for col in X_enc.select_dtypes(include=['object','category']).columns:
    le = LabelEncoder()
    X_enc[col] = X_enc[col].fillna('NA').astype(str)
    X_enc[col] = le.fit_transform(X_enc[col])
    label_encoders[col] = le

imputer = SimpleImputer(strategy='median')
X_enc = pd.DataFrame(imputer.fit_transform(X_enc), columns=X_enc.columns)

# --- 9️⃣ Split data ---
X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)

# --- 🔟 Baseline Models ---
rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

def evaluate(model, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1:", round(f1, 4))
    print(classification_report(y_test, preds, target_names=le_target.classes_))
    return acc, f1, preds

rf_acc, rf_f1, rf_preds = evaluate(rf, "Random Forest (Baseline)")
gb_acc, gb_f1, gb_preds = evaluate(gb, "Gradient Boosting (Baseline)")

# --- 11️⃣ Hyperparameter Tuning (GridSearch) ---
print("\n⏳ Running GridSearchCV for RandomForest...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                       param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("✅ Best RandomForest Params:", grid_rf.best_params_)

print("\n⏳ Running GridSearchCV for GradientBoosting...")
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
}
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42),
                       param_grid_gb, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_gb.fit(X_train, y_train)
print("✅ Best GradientBoosting Params:", grid_gb.best_params_)

# --- 12️⃣ Evaluate Tuned Models ---
rf_best = grid_rf.best_estimator_
gb_best = grid_gb.best_estimator_

rf_best.fit(X_train, y_train)
gb_best.fit(X_train, y_train)

rf_acc2, rf_f12, rf_preds2 = evaluate(rf_best, "RandomForest (Tuned)")
gb_acc2, gb_f12, gb_preds2 = evaluate(gb_best, "GradientBoosting (Tuned)")

# --- 13️⃣ Confusion Matrix for Tuned RF ---
plt.figure(figsize=(8,5))
cm = confusion_matrix(y_test, rf_preds2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title("Random Forest (Tuned) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- 14️⃣ Feature Importances ---
fi = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=fi.head(10))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

# --- 15️⃣ Save Models ---
joblib.dump(rf_best, "model_random_forest_tuned.joblib")
joblib.dump(gb_best, "model_gradient_boosting_tuned.joblib")
joblib.dump(le_target, "label_encoder_target.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")
joblib.dump(imputer, "imputer.joblib")
print("\n💾 Tuned Models saved successfully!")

# --- 16️⃣ Summary ---
print("\n🎯 Final Model Performance Summary:")
print(f"Baseline RandomForest: Accuracy={rf_acc:.3f}, F1={rf_f1:.3f}")
print(f"Tuned RandomForest: Accuracy={rf_acc2:.3f}, F1={rf_f12:.3f}")
print(f"Baseline GradientBoosting: Accuracy={gb_acc:.3f}, F1={gb_f1:.3f}")
print(f"Tuned GradientBoosting: Accuracy={gb_acc2:.3f}, F1={gb_f12:.3f}")

# Diabetes Readmission & Hypertension Prediction - Full Pipeline

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv("archive/diabetic_data.csv")

# 2. Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# 3. Drop irrelevant columns
df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# 4. Create binary target for readmission (<30 days = 1, else = 0)
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# 5. Create hypertension column (ICD-9: 401–405)
def has_hypertension(row):
    for col in ['diag_1', 'diag_2', 'diag_3']:
        try:
            code = float(row[col])
            if 401 <= code < 406:
                return 1
        except:
            continue
    return 0

df['has_hypertension'] = df.apply(has_hypertension, axis=1)

# 6. Drop original readmitted + diagnosis columns
df.drop(['readmitted', 'diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# 7. Separate features and target
X = df.drop('readmitted_binary', axis=1)
y = df['readmitted_binary']

# 8. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 9. ColumnTransformer pipeline (one-hot encoding + scaling)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

# 10. Apply SMOTE after preprocessing
X_processed = preprocessor.fit_transform(X)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_processed, y)

# 11. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 12. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# 13. Train and evaluate
for name, model in models.items():
    print(f"\n📌 Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("-" * 60)

# 14. Feature importance (Random Forest)
# Extra: pegar nomes das colunas transformadas
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
ohe_cols = ohe.get_feature_names_out(categorical_cols)
feature_names = np.concatenate([numeric_cols, ohe_cols])

importances = models["Random Forest"].feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Features (Random Forest)")
plt.barh(range(len(indices)), importances[indices], color="skyblue")
plt.yticks(range(len(indices)), feature_names[indices])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ----- App Title -----
st.title("📊 Hospital Readmission Prediction (Diabetes & Hypertension)")

# ----- Load and Preprocess Dataset -----
@st.cache_data
def load_data():
    df = pd.read_csv("archive/diabetic_data.csv")
    df.replace("?", np.nan, inplace=True)
    df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

    # Create binary target variable: 1 if readmitted <30 days, 0 otherwise
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Identify hypertension based on ICD-9 codes 401–405
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

    # Drop original target and diagnosis columns
    df.drop(['readmitted', 'diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.str.replace(r'\[|\]|<|>| ', '', regex=True)
    return df

df = load_data()
st.success("✅ Dataset successfully loaded")

# ----- Split features and target -----
X = df.drop('readmitted_binary', axis=1)
y = df['readmitted_binary']

# ----- Balance dataset using SMOTE -----
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# ----- Train/test split -----
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ----- Feature scaling -----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----- Model definitions -----
models = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

# ----- Train selected model -----
selected_model = models["Random Forest"]
selected_model.fit(X_train, y_train)

# ----- User Input Section -----
st.header("🧪 Predict Readmission for a New Patient")

# Input fields for patient data
user_input = {
    "age": st.selectbox("Age Range", options=[
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
        '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ]),
    "gender": st.selectbox("Gender", options=["Male", "Female"]),
    "race": st.selectbox("Race", options=["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]),
    "has_hypertension": st.selectbox("Has Hypertension?", options=[0, 1]),
    "A1Cresult": st.selectbox("A1C Result", options=["None", "Norm", ">7", ">8"]),
    "insulin": st.selectbox("Insulin Use", options=["No", "Steady", "Up", "Down"]),
    "change": st.selectbox("Medication Change?", options=["No", "Ch"]),
    "diabetesMed": st.selectbox("Takes Diabetes Medication?", options=["Yes", "No"])
}

# Create DataFrame from input
input_df = pd.DataFrame([user_input])

# Encode and align input with training columns
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Readmission"):
    prediction = selected_model.predict(input_scaled)[0]
    proba = selected_model.predict_proba(input_scaled)[0][1]

    # Show result
    st.subheader("🩺 Prediction Result:")
    st.write("🔁 **High chance of readmission.**" if prediction == 1 else "✅ **Low chance of readmission.**")
    st.write(f"📊 Probability of readmission: **{proba:.2%}**")
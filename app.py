import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ----- App Title -----
st.title("📊 Hospital Readmission Prediction (Diabetes & Hypertension)")

# ----- Load and Preprocess Dataset -----
@st.cache_data
def load_data():
    df = pd.read_csv("archive/diabetic_data.csv")
    df.replace("?", np.nan, inplace=True)
    df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

    # Target: readmission <30 days
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Hypertension detection
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

    df.drop(['readmitted', 'diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.str.replace(r'\[|\]|<|>| ', '', regex=True)
    return df

df = load_data()
st.success("✅ Dataset successfully loaded")

# ----- Split features/target -----
X = df.drop('readmitted_binary', axis=1)
y = df['readmitted_binary']

# ----- Initialize session state (model, scaler, columns) -----
if 'model' not in st.session_state:
    with st.spinner("🔧 Training model..."):

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier()
        model.fit(X_train_scaled, y_train)

        # Armazenar no estado da sessão
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.X_columns = X.columns

    st.success("✅ Model trained successfully!")

# ----- Prediction Form -----
st.header("🧪 Predict Readmission for a New Patient")

with st.form("predict_form"):
    age = st.selectbox("Age Range", options=[
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
        '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    race = st.selectbox("Race", options=["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])

    # Mapeando as opções "Yes" e "No" para 1 e 0
    hypertension_map = {"Yes": 1, "No": 0}
    has_hypertension_label = st.selectbox("Has Hypertension?", options=list(hypertension_map.keys()))

    A1Cresult = st.selectbox("A1C Result", options=["None", "Norm", ">7", ">8"])
    insulin = st.selectbox("Insulin Use", options=["No", "Steady", "Up", "Down"])
    change = st.selectbox("Medication Change?", options=["No", "Ch"])
    diabetesMed = st.selectbox("Takes Diabetes Medication?", options=["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Readmission")

    if submitted:
        # Montando o dicionário com os valores escolhidos
        user_input = {
            "age": age,
            "gender": gender,
            "race": race,
            "has_hypertension": hypertension_map[has_hypertension_label],  # Usando o mapeamento
            "A1Cresult": A1Cresult,
            "insulin": insulin,
            "change": change,
            "diabetesMed": diabetesMed
        }

        # Criando o DataFrame com o input do usuário
        input_df = pd.DataFrame([user_input])

        # Realizando o one-hot encoding
        input_df = pd.get_dummies(input_df)

        # Ajustando as colunas para que se alinhem com o modelo treinado
        input_df = input_df.reindex(columns=st.session_state.X_columns, fill_value=0)

        # Aplicando a transformação do scaler
        input_scaled = st.session_state.scaler.transform(input_df)

        # Realizando a predição
        prediction = st.session_state.model.predict(input_scaled)[0]
        proba = st.session_state.model.predict_proba(input_scaled)[0][1]

        # Exibindo o resultado
        st.subheader("🩺 Prediction Result:")
        st.write("🔁 **High chance of readmission.**" if prediction == 1 else "✅ **Low chance of readmission.**")
        st.write(f"📊 Probability of readmission: **{proba:.2%}**")
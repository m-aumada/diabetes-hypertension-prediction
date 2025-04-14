# Diabetes Readmission & Hypertension Prediction

## 📊 **Project Overview**

This project aims to predict patient readmission to hospitals within 30 days and identify patients with hypertension using a dataset from 130 U.S. hospitals. The main goal is to apply machine learning techniques for healthcare predictions that can help improve patient outcomes and resource allocation.

## 🧠 **Objective**

- **Predict if a diabetic or hypertensive patient will be readmitted to a hospital within 30 days**.
- Use real-world clinical data to develop a predictive model that can be used in hospital settings to help healthcare professionals identify patients at risk of readmission.

## 🏥 **Dataset**

The dataset used for this project is the **Diabetes 130-US hospitals dataset** from Kaggle.

- **Link to dataset**: [Diabetes 130-US hospitals dataset](https://www.kaggle.com/datasets/uciml/diabetes-130-us-hospitals)
- This dataset includes information about patient demographics, diagnoses, and previous medical records.

## 🧬 **Key Features**

- **readmitted_binary**: Target variable indicating if the patient was readmitted within 30 days (1: yes, 0: no).
- **has_hypertension**: A new feature indicating whether the patient has hypertension based on ICD-9 codes.
- Various clinical and demographic features like age, gender, admission type, and medical conditions.

## 🔧 **Tools and Technologies**

- **Python**: The main programming language used.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning algorithms and preprocessing tools.
- **XGBoost & LightGBM**: Gradient boosting algorithms for predictive modeling.
- **Imbalanced-learn (SMOTE)**: Synthetic minority over-sampling technique to handle class imbalance.
- **Streamlit**: Web framework for creating interactive dashboards.

## 📄 **Project Structure**

```
Diabetes-Hypertension-Prediction/
│
├── model_exploration.ipynb      # Jupyter notebook for data exploration and model training
├── model_training.py            # Python script for training and evaluating models
├── app.py                       # Streamlit app for model predictions and dashboard
├── archive/                     # Folder containing the dataset
│   └── diabetic_data.csv        # Diabetes dataset (raw data)
├── requirements.txt             # List of Python dependencies
└── README.md                    # Project documentation
```

## 🚀 **How to Run the Project**

### 1. **Clone the Repository**
First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Diabetes-Hypertension-Prediction.git
```

### 2. **Install Dependencies**
Make sure you have **Python 3.x** installed on your system. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. **Run Model Training**
You can run the **model_exploration.ipynb** notebook to explore the data and train the models. This will help you understand the steps involved in training the models, evaluating them, and performing feature importance analysis.

Alternatively, you can run the model training script with:

```bash
python model_training.py
```

This script will handle data preprocessing, model training, and evaluation for multiple machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM).

### 4. **Run the Streamlit App**
To interact with the model predictions in a web dashboard, run the Streamlit app:

```bash
streamlit run app.py
```

This will open an interactive web app where you can visualize model evaluation metrics, confusion matrices, and the most important features of the models.

## 📈 **Project Results**

After training and evaluating the models, here are the key results and insights:

### **Model Performance**

| Model             | AUC   | Accuracy | Precision | Recall | F1-Score |
|-------------------|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.85  | 0.78     | 0.80      | 0.75   | 0.77     |
| Random Forest       | 0.87  | 0.82     | 0.81      | 0.79   | 0.80     |
| XGBoost            | 0.89  | 0.83     | 0.85      | 0.81   | 0.83     |
| LightGBM           | 0.88  | 0.84     | 0.83      | 0.80   | 0.81     |

The **XGBoost** model performed the best with an **AUC of 0.89**, indicating its high ability to distinguish between patients who will and will not be readmitted within 30 days.

### **Evaluation Metrics** 

- **AUC**: The Area Under the Curve (AUC) for the models ranged from 0.85 to 0.89. A higher AUC indicates a better-performing model. A model with an AUC close to 1.0 is generally considered excellent at distinguishing between classes.
- **Accuracy**: Accuracy ranged from 0.78 to 0.84 across models, with **LightGBM** achieving the highest accuracy. However, accuracy can be misleading for imbalanced datasets, which is why AUC and F1-Score are more relevant here.
- **Precision, Recall, and F1-Score**: Precision and recall values were important for understanding how well the model identified patients at risk of readmission. The **F1-Score** provides a balance between precision and recall.

### **Feature Importance**

- The **Random Forest** model was used to identify the top 10 most important features in predicting patient readmission. The most important features included:
  - **Age**: Older patients had a higher likelihood of being readmitted.
  - **Number of previous admissions**: Patients with multiple previous admissions were more likely to be readmitted.
  - **Medical conditions**: Certain conditions, like hypertension, significantly contributed to the likelihood of readmission.

### **Confusion Matrix & ROC Curve**

- The confusion matrices for each model helped in visualizing the number of false positives and false negatives, highlighting how well the models are classifying patients.
- The ROC curves and AUC scores further reinforced the performance of each model.

## 🔍 **Why This Project Matters**

### **Importance of Accurate Predictions**

- **Improved Patient Care**: Predicting hospital readmissions helps healthcare providers identify high-risk patients and intervene early. This can reduce unnecessary hospital readmissions, improve patient outcomes, and optimize resource allocation.
- **Cost Reduction**: Reducing unnecessary readmissions can lower healthcare costs significantly, which is a critical issue for hospitals and healthcare systems.
- **Personalized Healthcare**: Understanding which factors contribute to readmissions allows healthcare providers to offer more tailored care to patients, addressing their specific needs.

### **Handling Class Imbalance**

Since readmission events are relatively rare, the models were trained using **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the class imbalance. This technique improves model performance by generating synthetic samples for the minority class, ensuring the model does not ignore patients at risk of readmission.

## 🔄 **Future Improvements**
- **Deployment**: Deploy the Streamlit app on a cloud platform like **AWS** for real-time prediction and monitoring.

## 💡 **Contributions**
Any contributions or suggestions are highly welcome

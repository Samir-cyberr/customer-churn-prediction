# 🛒 E-Commerce Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether an e-commerce customer is likely to churn.

The project demonstrates a realistic ML workflow:

**Data generation → EDA → preprocessing → model comparison → evaluation → model persistence → Streamlit app**

## 🎯 Problem

Customer churn is an important business problem for e-commerce companies. The goal of this project is to identify customers who are likely to stop using the service so that retention teams can prioritize them.

> This is an educational portfolio project using a reproducible synthetic dataset. It should not be used for real business decisions without validation on production data.

## 🧠 Features

The model uses behavioral and customer-level features such as:

- Customer age
- Tenure
- Monthly spending
- Number of orders
- Average order value
- Days since last order
- Website sessions
- Support tickets
- Discount usage
- Satisfaction score
- Mobile-app usage

## 🤖 Models

The training pipeline compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best model is selected using ROC-AUC on the validation set.

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 📁 Project Structure

```text
ecommerce-customer-churn-ml/
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Run locally

### 1. Clone

```bash
git clone https://github.com/Samir-cyberr/ecommerce-customer-churn-ml.git
cd ecommerce-customer-churn-ml
```

### 2. Install dependencies

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python -m src.generate_data
```

### 4. Train the models

```bash
python -m src.train
```

This creates the trained model and evaluation report in `models/`.

### 5. Launch the application

```bash
streamlit run app/streamlit_app.py
```

## 📊 Evaluation

The training script reports:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The model selection is based primarily on ROC-AUC because it evaluates ranking quality across classification thresholds.

## 🔍 Example workflow

```text
Customer data
      ↓
Data validation
      ↓
Train / validation split
      ↓
Preprocessing pipeline
      ↓
Model comparison
      ↓
Best model
      ↓
Evaluation
      ↓
Joblib model
      ↓
Streamlit prediction app
```

## 🔮 Future improvements

- Add SHAP explainability
- Add MLflow experiment tracking
- Deploy the Streamlit application
- Add automated CI tests
- Replace synthetic data with an approved real-world dataset
- Add model monitoring and drift detection

## 👨‍💻 Author

**Samir**

GitHub: https://github.com/Samir-cyberr

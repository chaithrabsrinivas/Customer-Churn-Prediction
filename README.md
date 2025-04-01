# Customer Churn Prediction

## Overview
This project aims to build a machine learning model to predict whether a customer will discontinue a subscription-based service. The goal is to identify customers at risk of churning and understand key factors influencing churn.

## Dataset
- The dataset includes historical customer data with attributes such as:
  - **Customer ID**
  - **Demographic details (Age, Gender, Location, etc.)**
  - **Subscription Duration**
  - **Usage Patterns (Frequency, Activity, etc.)**
  - **Payment History**
  - **Customer Support Interactions**
  - **Churn Label (0 = Active, 1 = Churned)**
- Ensure the dataset (`customer_data.csv`) is placed in the `dataset/` folder.

## Project Structure
```
customer-churn-prediction/
│── dataset/              # Dataset files
│── models/               # Trained models
│── notebooks/            # Jupyter notebooks (if any)
│── src/                  # Python scripts
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── .gitignore            # Ignore unnecessary files
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocess and Train Models**
   Run the `churn_prediction.py` script to train models:
   ```bash
   python src/churn_prediction.py
   ```
   This will save trained models in the `models/` directory.

2. **Make Predictions**
   Load a trained model and predict whether a customer is likely to churn:
   ```python
   import joblib
   import pandas as pd
   
   model = joblib.load('models/churn_model.pkl')
   new_customer = pd.DataFrame([{ "age": 35, "subscription_duration": 12, "usage": 5, "support_calls": 1 }])
   prediction = model.predict(new_customer)
   print("Churn Risk:" if prediction[0] == 1 else "Retained Customer")
   ```

## Models Used
- **Random Forest Classifier**
- **Logistic Regression**
- **Gradient Boosting (XGBoost, LightGBM)**

## Evaluation Metrics
- **Accuracy**
- **Precision, Recall & F1 Score**
- **ROC-AUC Score**

## Contributions
Feel free to contribute by submitting pull requests or opening issues.

## License
This project is licensed under the MIT License.


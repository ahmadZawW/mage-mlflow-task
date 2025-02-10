# Telco Customer Churn Prediction Pipeline

## Project Overview
This project implements a complete machine learning pipeline for customer churn prediction using Mage as the orchestration tool and FastAPI for model serving. The pipeline includes data ingestion, feature engineering, model training with MLflow tracking, and a REST API endpoint for predictions.

## Project Structure
```
.
├── api/
│   └── main.py                    # FastAPI implementation
├── mage_pipeline/
│   ├── data_loaders/
│   │   └── data_loader.py         # Data ingestion
│   ├── transformers/
│   │   ├── data_validation.py     # Data validation
│   │   ├── feature_engineering.py # Feature processing
│   │   ├── model_training.py      # Model training
│   │   └── model_prediction.py    # Prediction pipeline
│   ├── encoders/                  # Saved label encoders
│   ├── scalers/                   # Saved scalers
│   ├── models/                    # Saved models
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── mlruns/                        # MLflow tracking
└── requirements.txt
```

## Components

### 1. Mage Pipeline
The pipeline is orchestrated using Mage and consists of the following blocks:

- **Data Loading**: Loads and preprocesses the Telco Customer Churn dataset
- **Data Validation**: Validates data structure and quality
- **Feature Engineering**: Handles categorical and numerical features
- **Model Training**: Trains multiple models with MLflow tracking
- **Prediction**: Makes predictions using the best model

### 2. Model Tracking
MLflow is used for experiment tracking, storing:
- metrics tracking:
    - Accuracy, Precision, Recall, F1-Score, ROC AUC
- Feature importance visualization
- Model signatures and input examples
- parameter logging
- Dataset information tracking

### 3. API Service
FastAPI implementation for model serving:

#### Endpoint
- **URL**: `/predict`
- **Method**: POST
- **Input Schema**: JSON payload with customer features

Example curl request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "gender": "Female",
          "SeniorCitizen": 0,
          "Partner": "Yes",
          "Dependents": "No",
          "tenure": 1,
          "PhoneService": "No",
          "MultipleLines": "No phone service",
          "InternetService": "DSL",
          "OnlineSecurity": "No",
          "OnlineBackup": "Yes",
          "DeviceProtection": "No",
          "TechSupport": "No",
          "StreamingTV": "No",
          "StreamingMovies": "No",
          "Contract": "Month-to-month",
          "PaperlessBilling": "Yes",
          "PaymentMethod": "Electronic check",
          "MonthlyCharges": 29.85,
          "TotalCharges": 29.85
        }'
```

## Setup Instructions

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Start Mage Pipeline**
```bash
mage start
```

3. **Run FastAPI Server**
```bash
cd api
python main.py
```

## Usage

1. **Training Pipeline**
   - Access Mage UI at `http://localhost:6789`
   - Run the pipeline blocks in sequence
   - Monitor training in MLflow UI

2. **Making Predictions**
   - Use the FastAPI endpoint at `http://localhost:8000/predict`
   - Send POST requests with customer data
   - Receive churn predictions

## Model Artifacts
The pipeline saves several artifacts:
- Label encoders for categorical features
- Standard scaler for numerical features
- Trained models (Random Forest, Gradient Boosting, Logistic Regression)
- Best performing model
- Feature names for prediction alignment

## Monitoring
- MLflow tracking server for model performance monitoring
- FastAPI built-in documentation at `/docs`
- Request/response logging in FastAPI




## Notes
- The API expects all features to be present in the input
- Categorical features must match training data categories
- Numerical features are automatically scaled
- The model returns binary predictions (0: No Churn, 1: Churn)
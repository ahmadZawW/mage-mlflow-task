from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Define paths
ENCODERS_PATH = "mage_pipeline/encoders"
SCALERS_PATH = "mage_pipeline/scalers"
MODELS_PATH = "mage_pipeline/models"

# Define Input Schema
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Load feature names
        feature_names = joblib.load(f"{MODELS_PATH}/feature_names.joblib")
        
        # Load encoders and transform categorical features
        categorical_cols = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        
        for col in categorical_cols:
            encoder = joblib.load(f"{ENCODERS_PATH}/{col}_encoder.joblib")
            input_df[col] = encoder.transform(input_df[col])

        # Load scaler and transform numerical features
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        scaler = joblib.load(f"{SCALERS_PATH}/numerical_scaler.joblib")
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Ensure column order matches training data
        input_df = input_df[feature_names]

        # Load the trained model
        model = joblib.load(f"{MODELS_PATH}/best_model.joblib")

        # Make prediction
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




#  curl -X POST "http://127.0.0.1:8000/predict" \                           ✔    env311   US  
#      -H "Content-Type: application/json" \
#      -d '{
#           "gender": "Female",
#           "SeniorCitizen": 0,
#           "Partner": "Yes",
#           "Dependents": "No",
#           "tenure": 1,
#           "PhoneService": "No",
#           "MultipleLines": "No phone service",
#           "InternetService": "DSL",
#           "OnlineSecurity": "No",
#           "OnlineBackup": "Yes",
#           "DeviceProtection": "No",
#           "TechSupport": "No",
#           "StreamingTV": "No",
#           "StreamingMovies": "No",
#           "Contract": "Month-to-month",
#           "PaperlessBilling": "Yes",
#           "PaymentMethod": "Electronic check",
#           "MonthlyCharges": 29.85,
#           "TotalCharges": 29.85
#         }'

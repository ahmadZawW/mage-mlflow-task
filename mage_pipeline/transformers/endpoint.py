@transformer
def generate_api_endpoint(model_data, *args, **kwargs):
    """
    Generate FastAPI endpoint code for model deployment
    """
    import os
    
    api_code = """
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

@app.post("/predict")
async def predict(input_data):
    try:
        # Load the model
        model = joblib.load('models/best_model.joblib')
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    """
    
    # Save the API code
    os.makedirs('api', exist_ok=True)
    with open('api/main.py', 'w') as f:
        f.write(api_code)
    
    return {"api_code": api_code}
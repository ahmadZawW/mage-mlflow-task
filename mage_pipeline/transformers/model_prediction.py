@transformer
def predict(data_dict, *args, **kwargs):

    import os
    import joblib
    import pandas as pd

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
    
    # Load the beest model
    model_path = os.path.join(MODELS_PATH, 'best_model.joblib')
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names = joblib.load(os.path.join(MODELS_PATH, 'feature_names.joblib'))
    
    df = data_dict['test_data']

    # this commented code for the upcomming prediction test but 
    # its already done with the data comming from previous block
    
    # ENCODERS_PATH = "mage_pipeline/encoders"
    # SCALERS_PATH = "mage_pipeline/scalers"

    # for col in categorical_cols:
    #     encoder = joblib.load(f"{ENCODERS_PATH}/{col}_encoder.joblib")
    #     df[col] = encoder.transform(df[col])

    # scaler = joblib.load(f"{SCALERS_PATH}/numerical_scaler.joblib")
    # numerical_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    # df[numerical_cols] = scaler.transform(df[numerical_cols])


    # Ensure DataFrame has the correct feature names
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Test data must be a pandas DataFrame")
    
    # Verify and align features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    # Make predictions
    predictions = model.predict(df)
    
    # Return predictions in a structured format
    results = pd.DataFrame({
        'Predicted_Churn': predictions
    })
    
    return results
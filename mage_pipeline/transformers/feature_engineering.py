@transformer
def engineer_features(df, *args, **kwargs):

    import os
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
    
    # Define paths relative to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENCODERS_PATH = os.path.join(PROJECT_ROOT, 'encoders')
    SCALERS_PATH = os.path.join(PROJECT_ROOT, 'scalers')
    
    os.makedirs(ENCODERS_PATH, exist_ok=True)
    os.makedirs(SCALERS_PATH, exist_ok=True)
    
    # Label encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Save encoder with proper path
        encoder_path = os.path.join(ENCODERS_PATH, f'{col}_encoder.joblib')
        joblib.dump(le, encoder_path)
    
    # Scale numerical features
    numerical_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save scaler with proper path
    scaler_path = os.path.join(SCALERS_PATH, 'numerical_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    return df
@transformer
def validate_data(df, *args, **kwargs):

    import pandas as pd
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Check required columns
    required_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Warning: Found null values in columns:", 
              null_counts[null_counts > 0].to_dict())
    
    return df